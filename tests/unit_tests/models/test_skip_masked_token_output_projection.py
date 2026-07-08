# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.multi_token_prediction import process_mtp_loss


class _SparseLMHeadHarness:
    compute_language_model_loss_from_hidden_states = (
        LanguageModule.compute_language_model_loss_from_hidden_states
    )
    _compute_sparse_language_model_loss_from_hidden_states = (
        LanguageModule._compute_sparse_language_model_loss_from_hidden_states
    )
    _compute_active_language_model_logits = LanguageModule._compute_active_language_model_logits
    _compute_cp_balanced_sparse_language_model_loss = (
        LanguageModule._compute_cp_balanced_sparse_language_model_loss
    )
    _get_cp_balanced_active_token_splits = staticmethod(
        LanguageModule._get_cp_balanced_active_token_splits
    )

    def __init__(
        self,
        skip_masked_token_output_projection: bool,
        balance_masked_token_output_projection_across_cp: bool = False,
    ):
        self.config = SimpleNamespace(
            skip_masked_token_output_projection=skip_masked_token_output_projection,
            balance_masked_token_output_projection_across_cp=(
                balance_masked_token_output_projection_across_cp
            ),
            defer_embedding_wgrad_compute=False,
            _cpu_offloading_context=None,
        )
        self.tp_group = None

    def compute_language_model_loss(self, labels, logits):
        sequence_length, batch_size, vocab_size = logits.shape
        labels_by_sequence = labels.transpose(0, 1).contiguous()
        loss = F.cross_entropy(
            logits.reshape(sequence_length * batch_size, vocab_size),
            labels_by_sequence.reshape(-1),
            reduction="none",
        )
        return loss.view(sequence_length, batch_size).transpose(0, 1).contiguous()


class _FakeOutputLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = None
        self.gather_output = False
        self.sequence_parallel = False
        self.allreduce_dgrad = False
        self.explicit_expert_comm = False
        self.disable_grad_reduce = True
        self.gradient_accumulation_fusion = False
        self.skip_bias_add = False
        self.tp_group = None
        self.projected_rows = []

    def _forward_impl(
        self,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    ):
        del gradient_accumulation_fusion
        del allreduce_dgrad
        del sequence_parallel
        del grad_output_buffer
        del wgrad_deferral_limit
        del tp_group
        self.projected_rows.append(input.numel() // input.shape[-1])
        output = input.matmul(weight.t())
        if bias is not None:
            output = output + bias
        return output

    def forward(self, input_, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        if weight is None:
            weight = self.weight
        return (
            self._forward_impl(
                input=input_,
                weight=weight,
                bias=self.bias,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=False,
                grad_output_buffer=None,
                wgrad_deferral_limit=None,
                tp_group=None,
            ),
            None,
        )


def test_sparse_output_projection_matches_dense_masked_loss_and_gradients():
    torch.manual_seed(1234)
    sequence_length, batch_size, hidden_size, vocab_size = 6, 2, 4, 9
    hidden = torch.randn(sequence_length, batch_size, hidden_size)
    labels = torch.randint(0, vocab_size, (batch_size, sequence_length))
    loss_mask = torch.tensor(
        [[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0]], dtype=torch.float32
    )
    weight = torch.randn(vocab_size, hidden_size)

    dense_hidden = hidden.clone().requires_grad_(True)
    sparse_hidden = hidden.clone().requires_grad_(True)
    dense_output_layer = _FakeOutputLayer(weight)
    sparse_output_layer = _FakeOutputLayer(weight)

    dense_harness = _SparseLMHeadHarness(skip_masked_token_output_projection=False)
    sparse_harness = _SparseLMHeadHarness(skip_masked_token_output_projection=True)

    dense_loss = dense_harness.compute_language_model_loss_from_hidden_states(
        hidden_states=dense_hidden,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=dense_output_layer,
    )
    sparse_loss = sparse_harness.compute_language_model_loss_from_hidden_states(
        hidden_states=sparse_hidden,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=sparse_output_layer,
    )

    dense_total = (dense_loss * loss_mask).sum()
    sparse_total = (sparse_loss * loss_mask).sum()
    dense_total.backward()
    sparse_total.backward()

    assert sparse_output_layer.projected_rows == [int(loss_mask.count_nonzero().item())]
    assert dense_output_layer.projected_rows == [sequence_length * batch_size]
    torch.testing.assert_close(sparse_loss * loss_mask, dense_loss * loss_mask)
    torch.testing.assert_close(sparse_total, dense_total)
    torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(sparse_output_layer.weight.grad, dense_output_layer.weight.grad)


def test_sparse_output_projection_returns_active_logits_for_mtp_logging():
    torch.manual_seed(4321)
    sequence_length, batch_size, hidden_size, vocab_size = 5, 2, 3, 7
    hidden = torch.randn(sequence_length, batch_size, hidden_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, sequence_length))
    loss_mask = torch.tensor(
        [[0, 1, 0, 0, 1], [1, 0, 0, 1, 0]], dtype=torch.float32
    )
    output_layer = _FakeOutputLayer(torch.randn(vocab_size, hidden_size))
    harness = _SparseLMHeadHarness(skip_masked_token_output_projection=True)

    loss, logits, active_labels, active_loss_mask = (
        harness.compute_language_model_loss_from_hidden_states(
            hidden_states=hidden,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            return_logits=True,
        )
    )

    active_count = int(loss_mask.count_nonzero().item())
    assert logits.shape == (active_count, 1, vocab_size)
    assert active_labels.shape == (1, active_count)
    assert active_loss_mask.shape == (1, active_count)
    assert output_layer.projected_rows == [active_count]
    assert torch.count_nonzero(loss.detach()) == active_count


def test_process_mtp_loss_uses_sparse_projection_for_rolled_masks():
    torch.manual_seed(5678)
    sequence_length, batch_size, hidden_size, vocab_size = 5, 2, 4, 11
    mtp_num_layers = 2
    hidden = torch.randn(
        sequence_length * (1 + mtp_num_layers), batch_size, hidden_size
    )
    labels = torch.randint(0, vocab_size, (batch_size, sequence_length))
    loss_mask = torch.tensor(
        [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.float32
    )
    weight = torch.randn(vocab_size, hidden_size)
    config = SimpleNamespace(
        mtp_num_layers=mtp_num_layers,
        mtp_detach_heads=False,
        mtp_loss_scaling_factor=1.0,
        calculate_per_token_loss=False,
    )

    dense_hidden = hidden.clone().requires_grad_(True)
    sparse_hidden = hidden.clone().requires_grad_(True)
    dense_output_layer = _FakeOutputLayer(weight)
    sparse_output_layer = _FakeOutputLayer(weight)
    dense_harness = _SparseLMHeadHarness(skip_masked_token_output_projection=False)
    sparse_harness = _SparseLMHeadHarness(skip_masked_token_output_projection=True)

    dense_output = process_mtp_loss(
        hidden_states=dense_hidden,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=dense_output_layer,
        output_weight=None,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=dense_harness.compute_language_model_loss,
        config=config,
    )
    sparse_output = process_mtp_loss(
        hidden_states=sparse_hidden,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=sparse_output_layer,
        output_weight=None,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=sparse_harness.compute_language_model_loss,
        compute_language_model_loss_from_hidden_states=(
            sparse_harness.compute_language_model_loss_from_hidden_states
        ),
        config=config,
    )

    dense_output.sum().backward()
    sparse_output.sum().backward()

    first_rolled_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    first_rolled_mask[:, -1] = 0
    second_rolled_mask = torch.roll(first_rolled_mask, shifts=-1, dims=-1)
    second_rolled_mask[:, -1] = 0
    expected_sparse_rows = [
        int(first_rolled_mask.count_nonzero().item()),
        int(second_rolled_mask.count_nonzero().item()),
    ]

    assert dense_output_layer.projected_rows == [
        sequence_length * batch_size,
        sequence_length * batch_size,
    ]
    assert sparse_output_layer.projected_rows == expected_sparse_rows
    torch.testing.assert_close(sparse_output, dense_output)
    torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(sparse_output_layer.weight.grad, dense_output_layer.weight.grad)


def test_cp_balanced_active_token_splits_balance_hot_rank():
    active_counts = [8, 0]
    assert LanguageModule._get_cp_balanced_active_token_splits(active_counts, 0) == (
        [4, 4],
        [4, 0],
    )
    assert LanguageModule._get_cp_balanced_active_token_splits(active_counts, 1) == (
        [0, 0],
        [4, 0],
    )


def _run_distributed_cp_balanced_projection_check():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "distributed CP balance check expects two ranks"

    def debug(message):
        if os.environ.get("DEBUG_CP_BALANCE_TEST"):
            print(f"[rank{rank}] {message}", flush=True)

    torch.manual_seed(2468)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    sequence_length, batch_size, hidden_size, vocab_size = 8, 1, 4, 13
    hidden = torch.randn(sequence_length, batch_size, hidden_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    loss_mask = torch.ones(batch_size, sequence_length, device=device)
    if rank == 1:
        loss_mask.zero_()
    weight = torch.randn(vocab_size, hidden_size, device=device)
    dist.broadcast(weight, src=0)

    dense_hidden = hidden.clone().requires_grad_(True)
    sparse_hidden = hidden.clone().requires_grad_(True)
    dense_output_layer = _FakeOutputLayer(weight)
    sparse_output_layer = _FakeOutputLayer(weight)
    dense_harness = _SparseLMHeadHarness(skip_masked_token_output_projection=False)
    sparse_harness = _SparseLMHeadHarness(
        skip_masked_token_output_projection=True,
        balance_masked_token_output_projection_across_cp=True,
    )

    dense_loss = dense_harness.compute_language_model_loss_from_hidden_states(
        hidden_states=dense_hidden,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=dense_output_layer,
    )
    debug("dense forward done")
    sparse_loss = sparse_harness.compute_language_model_loss_from_hidden_states(
        hidden_states=sparse_hidden,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=sparse_output_layer,
        cp_group=dist.group.WORLD,
    )
    debug("sparse forward done")

    dense_total = (dense_loss * loss_mask).sum()
    sparse_total = (sparse_loss * loss_mask).sum()
    dense_total.backward()
    debug("dense backward done")
    sparse_total.backward()
    debug("sparse backward done")

    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        (sparse_loss * loss_mask).detach().cpu(),
        (dense_loss * loss_mask).detach().cpu(),
    )
    torch.testing.assert_close(
        sparse_hidden.grad.detach().cpu(),
        dense_hidden.grad.detach().cpu(),
    )
    debug("local loss and hidden grad checks done")
    dist.barrier()

    dense_weight_grad = dense_output_layer.weight.grad.detach().clone()
    sparse_weight_grad = sparse_output_layer.weight.grad.detach().clone()
    dist.all_reduce(dense_weight_grad)
    debug("dense weight grad all_reduce done")
    dist.all_reduce(sparse_weight_grad)
    debug("sparse weight grad all_reduce done")
    torch.testing.assert_close(sparse_weight_grad, dense_weight_grad)

    projected_rows = torch.tensor(
        [dense_output_layer.projected_rows[0], sparse_output_layer.projected_rows[0]],
        device=device,
    )
    gathered_projected_rows = [torch.empty_like(projected_rows) for _ in range(world_size)]
    dist.all_gather(gathered_projected_rows, projected_rows)
    debug("projected rows all_gather done")
    if rank == 0:
        dense_rows = [int(rows[0].item()) for rows in gathered_projected_rows]
        sparse_rows = [int(rows[1].item()) for rows in gathered_projected_rows]
        assert dense_rows == [sequence_length * batch_size, sequence_length * batch_size]
        assert sparse_rows == [sequence_length * batch_size // 2, sequence_length * batch_size // 2]


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        _run_distributed_cp_balanced_projection_check()
        dist.destroy_process_group()
        if int(os.environ.get("RANK", "0")) == 0:
            print("distributed CP-balanced sparse LM-head test passed")
    else:
        test_sparse_output_projection_matches_dense_masked_loss_and_gradients()
        test_sparse_output_projection_returns_active_logits_for_mtp_logging()
        test_process_mtp_loss_uses_sparse_projection_for_rolled_masks()
        test_cp_balanced_active_token_splits_balance_hot_rank()
        print("synthetic sparse LM-head tests passed")
