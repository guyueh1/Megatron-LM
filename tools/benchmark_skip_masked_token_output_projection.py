# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark sparse LM-head projection for masked-token training.

This script uses synthetic hidden states, labels, and loss masks to compare the dense
output-projection path against ``skip_masked_token_output_projection``. It benchmarks the
LM-head/loss portion only, including backward through hidden states and output weights.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.models.common.language_module.language_module import LanguageModule


class SparseLMHeadHarness:
    """Small harness exposing the LanguageModule sparse-loss helper."""

    compute_language_model_loss_from_hidden_states = (
        LanguageModule.compute_language_model_loss_from_hidden_states
    )
    _compute_sparse_language_model_loss_from_hidden_states = (
        LanguageModule._compute_sparse_language_model_loss_from_hidden_states
    )
    _compute_active_language_model_logits = LanguageModule._compute_active_language_model_logits

    def __init__(self, skip_masked_token_output_projection: bool):
        self.config = SimpleNamespace(
            skip_masked_token_output_projection=skip_masked_token_output_projection,
            defer_embedding_wgrad_compute=False,
            _cpu_offloading_context=None,
        )
        self.tp_group = None

    def compute_language_model_loss(self, labels, logits):
        sequence_length, batch_size, vocab_size = logits.shape
        labels_by_sequence = labels.transpose(0, 1).contiguous()
        loss = F.cross_entropy(
            logits.float().reshape(sequence_length * batch_size, vocab_size),
            labels_by_sequence.reshape(-1),
            reduction="none",
        )
        return loss.view(sequence_length, batch_size).transpose(0, 1).contiguous()


class SyntheticOutputLayer(torch.nn.Module):
    """Minimal output layer with the ColumnParallelLinear hooks used by the sparse helper."""

    def __init__(self, weight: torch.Tensor):
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
        self.projected_rows = 0

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
        self.projected_rows += input.numel() // input.shape[-1]
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


def roll_left_zero(tensor: torch.Tensor) -> torch.Tensor:
    rolled = torch.roll(tensor, shifts=-1, dims=-1)
    rolled[..., -1] = 0
    return rolled


def make_loss_mask(batch_size: int, sequence_length: int, active_fraction: float, device):
    total = batch_size * sequence_length
    active = int(round(total * active_fraction))
    mask = torch.zeros(total, device=device, dtype=torch.float32)
    if active > 0:
        mask[torch.randperm(total, device=device)[:active]] = 1.0
    return mask.view(batch_size, sequence_length)


def make_head_inputs(base_labels: torch.Tensor, base_loss_mask: torch.Tensor, heads: int):
    labels = [base_labels]
    masks = [base_loss_mask]
    for _ in range(1, heads):
        labels.append(roll_left_zero(labels[-1]))
        masks.append(roll_left_zero(masks[-1]))
    return labels, masks


def lm_head_step(harness, output_layer, hidden_bases, labels, loss_masks):
    total_loss = None
    hidden_states = []
    for hidden_base, head_labels, head_loss_mask in zip(hidden_bases, labels, loss_masks):
        hidden = hidden_base.detach().clone().requires_grad_(True)
        hidden_states.append(hidden)
        per_token_loss = harness.compute_language_model_loss_from_hidden_states(
            hidden_states=hidden,
            labels=head_labels,
            loss_mask=head_loss_mask,
            output_layer=output_layer,
        )
        head_loss = (per_token_loss * head_loss_mask).sum()
        total_loss = head_loss if total_loss is None else total_loss + head_loss
    total_loss.backward()
    hidden_grad_norm = sum(hidden.grad.float().norm().item() for hidden in hidden_states)
    weight_grad_norm = output_layer.weight.grad.float().norm().item()
    return float(total_loss.detach().float().item()), hidden_grad_norm, weight_grad_norm


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_step(device, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        synchronize(device)
        return start.elapsed_time(end) / iters

    start_time = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start_time) * 1000.0 / iters


def run_correctness_check(device):
    torch.manual_seed(2026)
    sequence_length, batch_size, hidden_size, vocab_size, heads = 32, 2, 64, 257, 3
    hidden_bases = [
        torch.randn(sequence_length, batch_size, hidden_size, device=device) for _ in range(heads)
    ]
    labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    loss_mask = make_loss_mask(batch_size, sequence_length, 0.4, device)
    head_labels, head_masks = make_head_inputs(labels, loss_mask, heads)
    weight = torch.randn(vocab_size, hidden_size, device=device)

    dense_layer = SyntheticOutputLayer(weight)
    sparse_layer = SyntheticOutputLayer(weight)
    dense = SparseLMHeadHarness(skip_masked_token_output_projection=False)
    sparse = SparseLMHeadHarness(skip_masked_token_output_projection=True)

    dense_loss, dense_hidden_grad, dense_weight_grad = lm_head_step(
        dense, dense_layer, hidden_bases, head_labels, head_masks
    )
    sparse_loss, sparse_hidden_grad, sparse_weight_grad = lm_head_step(
        sparse, sparse_layer, hidden_bases, head_labels, head_masks
    )
    torch.testing.assert_close(
        torch.tensor(sparse_loss, device=device), torch.tensor(dense_loss, device=device)
    )
    torch.testing.assert_close(
        torch.tensor(sparse_hidden_grad, device=device),
        torch.tensor(dense_hidden_grad, device=device),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        torch.tensor(sparse_weight_grad, device=device),
        torch.tensor(dense_weight_grad, device=device),
        rtol=1e-5,
        atol=1e-5,
    )

    return {
        "dense_projected_rows": dense_layer.projected_rows,
        "sparse_projected_rows": sparse_layer.projected_rows,
        "loss": dense_loss,
    }


def run_benchmark(args, device, heads: int, active_fraction: float):
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    hidden_bases = [
        torch.randn(
            args.tokens,
            args.batch_size,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        for _ in range(heads)
    ]
    labels = torch.randint(0, args.vocab_size, (args.batch_size, args.tokens), device=device)
    loss_mask = make_loss_mask(args.batch_size, args.tokens, active_fraction, device)
    head_labels, head_masks = make_head_inputs(labels, loss_mask, heads)
    weight = torch.randn(args.vocab_size, args.hidden_size, device=device, dtype=dtype)

    dense_harness = SparseLMHeadHarness(skip_masked_token_output_projection=False)
    sparse_harness = SparseLMHeadHarness(skip_masked_token_output_projection=True)
    dense_layer = SyntheticOutputLayer(weight)
    sparse_layer = SyntheticOutputLayer(weight)

    def dense_fn():
        dense_layer.zero_grad(set_to_none=True)
        lm_head_step(dense_harness, dense_layer, hidden_bases, head_labels, head_masks)

    def sparse_fn():
        sparse_layer.zero_grad(set_to_none=True)
        lm_head_step(sparse_harness, sparse_layer, hidden_bases, head_labels, head_masks)

    dense_ms = time_step(device, dense_fn, args.warmup, args.iters)
    sparse_ms = time_step(device, sparse_fn, args.warmup, args.iters)

    return {
        "heads": heads,
        "active_fraction": active_fraction,
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "speedup": dense_ms / sparse_ms,
        "dense_projected_rows_per_step": dense_layer.projected_rows // (args.warmup + args.iters),
        "sparse_projected_rows_per_step": sparse_layer.projected_rows // (args.warmup + args.iters),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--active-fractions", type=float, nargs="+", default=[0.75, 0.5, 0.4, 0.25])
    parser.add_argument("--head-counts", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True

    correctness = run_correctness_check(device)
    results = []
    for heads in args.head_counts:
        for active_fraction in args.active_fractions:
            results.append(run_benchmark(args, device, heads, active_fraction))

    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "tokens": args.tokens,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "correctness": correctness,
        "results": results,
    }

    print(json.dumps(report, indent=2))
    if args.json_output is not None:
        Path(args.json_output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
