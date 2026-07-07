# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import logging
import os
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.cuda_graphs import CudaGraphManager

try:
    from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy
except:
    te_parallel_cross_entropy = None
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_token_prediction import tie_word_embeddings_state_dict
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    is_te_min_version,
    make_tp_sharded_tensor_for_checkpoint,
)


class LanguageModule(MegatronModule):
    """Base language module that has common helper functions used across GPT, BERT etc.

    Args:
        config (TransformerConfig): Input transformer config for the model
        pg_collection (ProcessGroupCollection): Model communication process groups
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        super().__init__(config=config)
        self._set_attention_backend()
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = get_tensor_model_parallel_group_if_none(pg_collection.tp)
        self.pp_group = pg_collection.pp
        assert hasattr(self.pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            "If you are using the default process group, please use"
            "`parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        self.embd_group = pg_collection.embd
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

    def _setup_mtp_cuda_graphs(self):
        """Wrap `compute_mtp_single_step` with a CudaGraphManager.

        Must be called by subclasses after `self.mtp` is created.
        """
        if self.config.cuda_graph_impl == "local":
            self._mtp_cudagraph_manager = CudaGraphManager(
                self.config,
                base_module=self,
                function_name="compute_mtp_single_step",
                need_backward=False,
                inline_capture=True,
            )

    def _is_in_embd_group(self):
        if self.embd_group is None:
            return False
        if torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(
            self.embd_group
        ):
            if getattr(self, 'mtp_process', False):
                return True
            if (
                torch.distributed.get_rank()
                == torch.distributed.get_process_group_ranks(self.embd_group)[0]
            ):
                return is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(
                    self.pp_group
                )
            elif (
                torch.distributed.get_rank()
                == torch.distributed.get_process_group_ranks(self.embd_group)[-1]
            ):
                return is_vp_last_stage(self.vp_stage, self.vp_size) and is_pp_last_stage(
                    self.pp_group
                )
            else:
                return True
        return False

    # pylint: disable=line-too-long
    def _set_attention_backend(self):
        """Set attention backend

        Transformer engine works based on optout. By default all three attention backend flags are set to 1. So if the user choses a particular attention backend we set the other two to 0. If the user choses local, we set all 3 TE env variables to 0.
        """

        def check_and_set_env_variable(
            env_variable_name: str, expected_value: int, attn_type: AttnBackend
        ) -> None:
            current_value = os.getenv(env_variable_name)
            assert current_value is None or current_value == str(
                expected_value
            ), f'{env_variable_name} set to {current_value}, but expected {expected_value} for attention backend type {attn_type.name}. unset NVTE_FLASH_ATTN, NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN. Use the --attention-backend argument if you want to choose between (flash/fused/unfused/auto/local). Default is auto.'
            os.environ[env_variable_name] = str(expected_value)

        if self.config.attention_backend == AttnBackend.local:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 0, AttnBackend.flash)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 0, AttnBackend.flash)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 0, AttnBackend.flash)
        elif self.config.attention_backend == AttnBackend.flash:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 1, AttnBackend.flash)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 0, AttnBackend.flash)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 0, AttnBackend.flash)
        elif self.config.attention_backend == AttnBackend.fused:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 0, AttnBackend.fused)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 1, AttnBackend.fused)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 0, AttnBackend.fused)
        elif self.config.attention_backend == AttnBackend.unfused:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 0, AttnBackend.unfused)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 0, AttnBackend.unfused)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 1, AttnBackend.unfused)
        elif self.config.attention_backend == AttnBackend.auto:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 1, AttnBackend.auto)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 1, AttnBackend.auto)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 1, AttnBackend.auto)

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if self.config.cross_entropy_loss_fusion:
            if self.config.cross_entropy_fusion_impl == 'te':
                if te_parallel_cross_entropy is not None:
                    labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
                    # Use is_cg_capturable=True for full iteration CUDA graphs to avoid torch.equal checks
                    is_cg_capturable = (
                        hasattr(self.config, 'cuda_graph_impl')
                        and self.config.cuda_graph_impl == "full_iteration"
                    )
                    if is_cg_capturable and not is_te_min_version("2.7.0"):
                        from megatron.core.utils import get_te_version

                        current_version = get_te_version()
                        raise AssertionError(
                            f"CUDA graph compatible cross entropy requires TransformerEngine >= 2.7.0, "
                            f"but found version {current_version}. Please upgrade TransformerEngine "
                            f"or set cuda_graph_impl to a value other than 'full_iteration'."
                        )

                    loss = te_parallel_cross_entropy(
                        logits, labels, self.pg_collection.tp, is_cg_capturable
                    )
                else:
                    raise RuntimeError("Trying to use a TE block when it's not present.")
            elif self.config.cross_entropy_fusion_impl == 'native':
                loss = fused_vocab_parallel_cross_entropy(logits, labels, self.pg_collection.tp)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(
                logits, labels, tp_group=self.tp_group
            )

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def compute_language_model_loss_from_hidden_states(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor],
        output_layer: Callable,
        output_weight: Optional[Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
        return_logits: bool = False,
    ):
        """Compute LM loss from hidden states, optionally skipping masked output positions.

        When ``skip_masked_token_output_projection`` is enabled and supported for the current
        output-head mode, the vocab projection and cross entropy run only on positions whose
        loss mask is non-zero. The returned loss keeps the standard dense ``[batch, sequence]``
        shape with zeros at skipped positions, so existing downstream loss reduction can still
        apply ``loss_mask`` normally.
        """
        sparse_result = self._compute_sparse_language_model_loss_from_hidden_states(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=output_weight,
            runtime_gather_output=runtime_gather_output,
            scale_logits_fn=scale_logits_fn,
            return_logits=return_logits,
        )
        if sparse_result is not None:
            return sparse_result

        logits, _ = output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        if scale_logits_fn is not None:
            logits = scale_logits_fn(logits)
        loss = self.compute_language_model_loss(labels, logits)
        if return_logits:
            return loss, logits, labels, loss_mask
        return loss

    def _compute_sparse_language_model_loss_from_hidden_states(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor],
        output_layer: Callable,
        output_weight: Optional[Tensor],
        runtime_gather_output: Optional[bool],
        scale_logits_fn: Optional[Callable[[Tensor], Tensor]],
        return_logits: bool,
    ):
        if not getattr(self.config, "skip_masked_token_output_projection", False):
            return None
        if labels is None or loss_mask is None:
            return None
        if getattr(self.config, "defer_embedding_wgrad_compute", False):
            return None
        if getattr(self.config, "_cpu_offloading_context", None) is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                return None

        gather_output = (
            runtime_gather_output
            if runtime_gather_output is not None
            else getattr(output_layer, "gather_output", False)
        )
        if gather_output:
            return None

        if not hasattr(output_layer, "_forward_impl"):
            return None

        weight = output_weight if output_weight is not None else getattr(output_layer, "weight", None)
        if weight is None:
            return None

        tp_group = getattr(output_layer, "tp_group", self.tp_group)
        uses_sequence_parallel = getattr(output_layer, "sequence_parallel", False)
        hidden_states_for_projection = hidden_states
        if uses_sequence_parallel:
            hidden_states_for_projection = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True, group=tp_group
            )

        sequence_length, batch_size, hidden_size = hidden_states_for_projection.shape
        if labels.shape != (batch_size, sequence_length):
            return None
        if loss_mask.shape != (batch_size, sequence_length):
            return None

        loss_mask_by_sequence = loss_mask.transpose(0, 1).contiguous()
        active_mask = loss_mask_by_sequence.reshape(-1) != 0
        active_indices = active_mask.nonzero(as_tuple=False).view(-1)
        num_positions = sequence_length * batch_size
        active_count = active_indices.numel()

        if active_count == num_positions:
            return None

        labels_by_sequence = labels.transpose(0, 1).contiguous()
        flat_labels = labels_by_sequence.reshape(-1)
        flat_loss_mask = loss_mask_by_sequence.reshape(-1)

        if active_count == 0:
            zero = hidden_states.sum() * 0.0
            loss = (hidden_states.new_zeros((batch_size, sequence_length)) + zero).contiguous()
            if return_logits:
                logits = hidden_states.new_empty((0, 1, weight.size(0)))
                active_labels = flat_labels.new_empty((1, 0))
                active_loss_mask = flat_loss_mask.new_empty((1, 0))
                return loss, logits, active_labels, active_loss_mask
            return loss

        flat_hidden_states = hidden_states_for_projection.contiguous().view(
            num_positions, hidden_size
        )
        active_hidden_states = flat_hidden_states.index_select(0, active_indices).view(
            active_count, 1, hidden_size
        )
        active_labels = flat_labels.index_select(0, active_indices).view(1, active_count)
        active_loss_mask = flat_loss_mask.index_select(0, active_indices).view(1, active_count)

        logits = self._compute_active_language_model_logits(
            active_hidden_states=active_hidden_states,
            output_layer=output_layer,
            weight=weight,
            sequence_parallel_input_gathered=uses_sequence_parallel,
        )
        if scale_logits_fn is not None:
            logits = scale_logits_fn(logits)

        active_loss = self.compute_language_model_loss(active_labels, logits).reshape(-1)
        flat_loss = active_loss.new_zeros(num_positions)
        flat_loss = flat_loss.scatter(0, active_indices, active_loss)
        loss = flat_loss.view(sequence_length, batch_size).transpose(0, 1).contiguous()

        if return_logits:
            return loss, logits, active_labels, active_loss_mask
        return loss

    def _compute_active_language_model_logits(
        self,
        active_hidden_states: Tensor,
        output_layer: Callable,
        weight: Tensor,
        sequence_parallel_input_gathered: bool,
    ) -> Tensor:
        """Project active hidden states with the LM head while preserving TP autograd behavior."""
        bias = getattr(output_layer, "bias", None)
        if getattr(output_layer, "skip_bias_add", False):
            bias = None

        if sequence_parallel_input_gathered:
            input_parallel = active_hidden_states
            allreduce_dgrad = False
            sequence_parallel = False
        else:
            if (
                getattr(output_layer, "allreduce_dgrad", False)
                or getattr(output_layer, "sequence_parallel", False)
                or getattr(output_layer, "explicit_expert_comm", False)
                or getattr(output_layer, "disable_grad_reduce", False)
            ):
                input_parallel = active_hidden_states
            else:
                input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(
                    active_hidden_states, group=getattr(output_layer, "tp_group", self.tp_group)
                )
            allreduce_dgrad = (
                False
                if getattr(output_layer, "explicit_expert_comm", False)
                else getattr(output_layer, "allreduce_dgrad", False)
            )
            sequence_parallel = False

        return output_layer._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=getattr(
                output_layer, "gradient_accumulation_fusion", False
            ),
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            grad_output_buffer=None,
            wgrad_deferral_limit=None,
            tp_group=getattr(output_layer, "tp_group", self.tp_group),
        )

    def setup_embeddings_and_output_layer(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initalizes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.

        Parameter attributes set:
        - `is_embedding_or_output_parameter`: True for embedding + output layer weights.
          Used by decoupled_lr, Muon optimizer, and other Megatron features.
        - `is_embedding_parameter`: True for MuP "embedding-class" parameters.
          Used by MuP for table-8 style optimizer grouping (base LR/eps for vector-like params).
        """

        # Mark embedding and output layer for decoupled_lr and other features.
        # This is the original Megatron attribute used by decoupled_lr, Muon, FSDP, etc.
        # Include MTP-stage embedding too: it is a duplicated copy of the pre_process
        # embedding (kept in sync via cross-stage all-reduce). Without this tag, the
        # LayerWise distributed optimizer routes it to its Muon-managed buffer and
        # `_emit_bucket(shared_embedding=True)` replicates the (vocab x hidden) tensor
        # across all dp_size shards, blowing up the chunk's buffer by ~8x.
        if (self.pre_process or getattr(self, 'mtp_process', False)) and hasattr(self, 'embedding'):
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if (
            self.post_process
            and hasattr(self, 'output_layer')
            and self.output_layer.weight is not None
        ):
            self.output_layer.weight.is_embedding_or_output_parameter = True

        # Mark embedding-class parameters for MuP optimizer grouping.
        # Under MuP table-8-style grouping, embeddings/output use base LR/eps while
        # hidden matrix-like params use width-scaled LR/eps.
        mtp_process = getattr(self, 'mtp_process', False)
        if self.config.use_mup and (self.pre_process or mtp_process) and hasattr(self, 'embedding'):
            for param in self.embedding.parameters():
                param.is_embedding_parameter = True
        if (
            self.config.use_mup
            and self.post_process
            and hasattr(self, 'output_layer')
            and self.output_layer.weight is not None
        ):
            self.output_layer.weight.is_embedding_parameter = True

        # If share_embeddings_and_output_weights is True, we need to maintain duplicated
        # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
        # we also need to maintain duplicated embedding weights in mtp process stage.
        # So we need to copy embedding weights from pre processing stage as initial parameters
        # in these cases.
        if not self.share_embeddings_and_output_weights and not getattr(
            self.config, 'mtp_num_layers', 0
        ):
            return

        if self.config.pipeline_model_parallel_size == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            return

        if (
            is_vp_first_stage(self.vp_stage, self.vp_size)
            and is_pp_first_stage(self.pp_group)
            and self.pre_process
            and not self.post_process
        ):
            self.shared_embedding_or_output_weight().shared_embedding = True

        if (
            (self.post_process and self.share_embeddings_and_output_weights)
            or getattr(self, 'mtp_process', False)
        ) and not self.pre_process:
            assert not (
                is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(self.pp_group)
            )
            # set weights of the duplicated embedding to 0 here,
            # then copy weights from pre processing stage using all_reduce below.
            weight = self.shared_embedding_or_output_weight()
            weight.data.fill_(0)
            weight.shared = True
            weight.shared_embedding = True
            # Keep optimizer grouping consistent for tied embedding/output copies.
            if self.config.use_mup:
                weight.is_embedding_parameter = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if self._is_in_embd_group() and not self.config.init_model_with_meta_device:
                weight = self.shared_embedding_or_output_weight()
                weight.data = weight.data.cuda()
                torch.distributed.all_reduce(weight.data, group=self.embd_group)

        elif not getattr(LanguageModule, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageModule.embedding_warning_printed = True

    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Apply MuP output scaling to logits.

        When MuP is enabled, scales logits by mup_output_mult (auto-set to 1/width_mult
        if left at default) to keep output variance stable across widths.

        Args:
            logits (Tensor): Raw logits from the output layer.

        Returns:
            Tensor: Scaled logits if MuP is enabled and mup_output_mult != 1.0,
                    otherwise unchanged logits.
        """
        if not self.config.use_mup:
            return logits
        if self.config.mup_output_mult != 1.0:
            return logits * self.config.mup_output_mult
        return logits

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share embedding and output weights set to True
          or when use Multi-Token Prediction (MTP).

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight while during post processing it returns the final output layers weight
        """
        if self.pre_process or getattr(self, 'mtp_process', False):
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # When share_embeddings_and_output_weights is True, the embedding weight is the
            # canonical shared weight and is passed to the output layer during forward.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    @torch.inference_mode()
    def compute_mtp_single_step(
        self,
        hidden_states: Tensor,
        next_token_ids: Tensor,
        position_ids: Tensor,
        depth: Optional[int] = None,
        eager: bool = False,
        cache_key=None,
    ) -> tuple:
        """Compute a single MTP depth for speculative decoding.

        This is called after speculative token verification to compute MTP
        predictions conditioned on verified tokens only.

        Args:
            hidden_states (Tensor): Hidden states at last accepted positions.
            next_token_ids (Tensor): Correct next token IDs [1, N].
            position_ids (Tensor): Position IDs for the next tokens [1, N].
            depth (int, optional): MTP depth index. Only needed when `mtp_use_repeated_layer` is
                False (each depth uses a distinct layer). Omit for repeated-layer models so that a
                single CUDA graph can serve all depths.
            eager, cache_key: The `CudaGraphManager` works by monkey-patching this argument onto the
                function signature. Explictly including them removes the need for a monkey-patch,
                and makes it straightforward to call the same method with and without eager mode.
                These arguments are consumed by `CudaGraphManager`, if it exists.

        Returns:
            tuple: (new_hidden_states, logits [N, 1, vocab_size]).
        """
        # CudaGraphManager consumes these args, if it exists
        del eager, cache_key
        layer_idx = 0 if depth is None else depth
        mtp_hidden = self.mtp.layers[layer_idx].forward_single_position(
            hidden_states=hidden_states,
            next_token_ids=next_token_ids,
            position_ids=position_ids,
            embedding=self.embedding,
        )

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(mtp_hidden, weight=output_weight, runtime_gather_output=True)
        logits = self._scale_logits(logits)

        return mtp_hidden, logits

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict implementation that handles the output layer weights tying.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the LanguageModel
        """
        assert not sharded_offsets, "Unexpected sharded offsets"

        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight'
        output_layer_weight_key = f'{prefix}output_layer.weight'
        output_layer_bias_key = f'{prefix}output_layer.bias'

        # Multi-Token Prediction (MTP) needs embedding layer in mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Note: MTP loss is computed at post_process stage, so the output_layer on mtp_process
        # rank doesn't need special tying - it's not used for loss computation.
        if getattr(self, 'mtp_process', False) and not self.pre_process:
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(
                sharded_state_dict,
                emb_weight,
                first_stage_word_emb_key,
                tp_group=self.tp_group,
                dp_cp_group=metadata['dp_cp_group'],
            )
        if self.share_embeddings_and_output_weights:
            self.tie_embeddings_and_output_weights_state_dict(
                sharded_state_dict, output_layer_weight_key, first_stage_word_emb_key, metadata
            )
        elif self.post_process:
            # Make sure the output layer follows the embeddings padding logic
            sharded_state_dict[output_layer_weight_key].allow_shape_mismatch = True

        # Regardless of sharing the output weights with embeddings, we must handle the bias padding
        if self.post_process and output_layer_bias_key in sharded_state_dict:
            sharded_state_dict[output_layer_bias_key].allow_shape_mismatch = True

        return sharded_state_dict

    def tie_embeddings_and_output_weights_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
        metadata: dict = {},
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if not self.post_process:
            # No output layer
            assert output_layer_weight_key not in sharded_state_dict, sharded_state_dict.keys()
            return

        if self.pre_process:
            # Output layer is equivalent to the embedding already
            return

        # If use Multi-Token Prediction (MTP), we need maintain both embedding layer and output
        # layer in mtp process stage. In this case, if share_embeddings_and_output_weights is True,
        # the shared weights will be stored in embedding layer, and output layer will not have
        # any weight.
        if getattr(self, 'mtp_process', False):
            # No output layer
            assert output_layer_weight_key not in sharded_state_dict, sharded_state_dict.keys()
            return

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        tensor = self.shared_embedding_or_output_weight()
        last_stage_word_emb_replica_id = (
            1,  # copy of first stage embedding
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=first_stage_word_emb_key,
            replica_id=last_stage_word_emb_replica_id,
            allow_shape_mismatch=True,
            tp_group=self.tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )
