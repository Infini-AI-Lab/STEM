# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
IIR-STEM: Language-model wrappers that augment each STEM embedding with
an IIR / EMA memory trace of past embeddings.

The inner transformer blocks, FFN classes, and attention modules are
**reused unchanged** from the base STEM implementation.  Only the
embedding callback (``stem_embeddings_fn``) is overridden so that each
STEM layer receives:

    e_tilde_t = e_t + alpha * ctx_t

where ``ctx_t`` is the causal EMA memory (see :mod:`lingua.stem_iir`).

Model registry
--------------
``IIR_STEM_MODEL_REGISTRY`` maps model-type strings to the usual 5-tuple
expected by ``stem_train.py``:

    ``"llama_iir"``  – LLaMA attention, pre-norm
    ``"qwen3_iir"``  – Qwen3 per-head QK-norm, pre-norm
    ``"olmo3_iir"``  – OLMo3 full-dim QK-norm, post-norm
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from lingua.stem_iir import IIRStemTransformerArgs, IIRMemory
from lingua.stem_dist_utils import ParallelEmbedding

from apps.main.stem import (
    LMTransformer,
    Qwen3LMTransformer,
    OLMo3LMTransformer,
    build_stem_lm_fsdp_grouping_plan,
    build_qwen3_stem_lm_fsdp_grouping_plan,
    build_olmo3_stem_lm_fsdp_grouping_plan,
)
from apps.main.transformer import (
    LMTransformerArgs,
    get_no_recompute_ops as llama_get_no_recompute_ops,
    get_num_flop_per_token as llama_get_num_flop_per_token,
)
from apps.main.qwen3 import (
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
    get_num_flop_per_token as qwen3_get_num_flop_per_token,
)
from apps.main.olmo3 import (
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
    get_num_flop_per_token as olmo3_get_num_flop_per_token,
)


# =========================================================================
# Args
# =========================================================================

@dataclass
class IIRStemLMTransformerArgs(IIRStemTransformerArgs, LMTransformerArgs):
    """Combined args for IIR-STEM language-model transformers."""
    pass


# =========================================================================
# Base IIR-STEM LM Transformer (LLaMA attention, pre-norm)
# =========================================================================

class IIRStemLMTransformer(nn.Module):
    """Higher-level wrapper analogous to :class:`StemLMTransformer` but with
    per-layer :class:`IIRMemory` modules that inject causal EMA context into
    the STEM embeddings before they enter the FFN.

    The inner ``lm_transformer`` is the same FSDP-wrappable backbone used by
    the base STEM variant; only the embedding callback is different.

    Extra modules compared to ``StemLMTransformer``:

    * ``self.iir_memories``  – ``nn.ModuleList`` of :class:`IIRMemory`, one
      per stem layer.  These hold the learnable ``mu``, ``alpha`` (and
      optionally ``m0``) parameters.

    Optimiser integration
    ---------------------
    ``iir_memories`` and ``stem_embeddings`` are **not** FSDP-wrapped (they
    are regular ``nn.Module`` / ``nn.Parameter``).  Use
    :meth:`stem_parameters` to collect all non-FSDP parameters for a
    dedicated optimiser group.
    """

    # Subclasses override to use Qwen3/OLMo3 backbones.
    _lm_transformer_cls = LMTransformer

    def __init__(self, args: IIRStemLMTransformerArgs):
        super().__init__()
        self.args = args

        # 1) FSDP-wrappable language-model backbone
        self.lm_transformer = self._lm_transformer_cls(args)

        # 2) STEM embedding tables (same as base StemLMTransformer)
        assert args.stem_embedding_dim is not None, (
            "stem_embedding_dim must be provided"
        )
        device = next(iter(self.lm_transformer.parameters())).device
        self.stem_embeddings = nn.ModuleList([
            ParallelEmbedding(
                args.vocab_size, args.stem_embedding_dim, device=device,
            )
            for _ in range(len(self.lm_transformer.stem_layers))
        ])

        # 3) Per-layer IIR memory modules
        self.iir_memories = nn.ModuleList([
            IIRMemory(
                d_ff=args.stem_embedding_dim,
                mu_init=args.iir_mu_init,
                alpha_init=args.iir_alpha_init,
                learnable_m0=args.iir_learnable_m0,
                per_dim_alpha=args.iir_per_dim_alpha,
            )
            for _ in range(len(self.lm_transformer.stem_layers))
        ])

        # 4) layer_idx -> stem list index mapping
        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(self.lm_transformer.stem_layers)
        }

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        """Forward pass that injects IIR-contextual STEM embeddings."""

        def stem_embeddings_fn(
            layer_idx: int, token_values: torch.Tensor,
        ) -> torch.Tensor:
            stem_idx = self._layer_to_stem_idx[layer_idx]
            e = self.stem_embeddings[stem_idx](token_values)  # [B, L, d_ff]
            e_tilde = self.iir_memories[stem_idx](e)          # [B, L, d_ff]
            return e_tilde

        return self.lm_transformer(
            token_values=token_values,
            target=target,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
            stem_embeddings_fn=stem_embeddings_fn,
        )

    # -----------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def reset_stem_embeddings(self):
        """Reset parameters of ``stem_embeddings`` (IIR params are kept)."""
        import logging
        logger = logging.getLogger()
        for i, embedding in enumerate(self.stem_embeddings):
            weight_device = embedding.weight.device
            if weight_device.type == "meta":
                logger.warning(
                    f"stem_embeddings[{i}].weight on meta device, skipping init"
                )
                continue
            embedding.reset_parameters()
            if embedding.weight.numel() > 0:
                weight_norm = embedding.weight.norm().item()
                is_zero = (embedding.weight.abs().max() == 0).item()
                if is_zero:
                    logger.error(
                        f"stem_embeddings[{i}].weight still zeros after reset!"
                    )
                else:
                    logger.debug(
                        f"stem_embeddings[{i}].weight initialised: "
                        f"norm={weight_norm:.6f}, device={weight_device}"
                    )

    def init_weights(self):
        """Initialise all weights (backbone + stem embeddings + IIR params).

        IIR parameters (``mu``, ``alpha``, ``m0``) are re-set to their
        configured initial values because ``to_empty()`` leaves the
        meta-materialised parameter memory uninitialised.
        """
        self.lm_transformer.init_weights()
        self.reset_stem_embeddings()
        for iir_mem in self.iir_memories:
            iir_mem.reset_parameters()

    # -----------------------------------------------------------------
    # Convenience: all non-FSDP ("stem") parameters
    # -----------------------------------------------------------------

    def stem_parameters(self):
        """Yield all parameters that live *outside* ``lm_transformer``.

        This includes ``stem_embeddings`` **and** ``iir_memories`` (mu, alpha,
        m0).  Useful for building a dedicated optimiser group that is separate
        from the FSDP-managed backbone parameters.
        """
        yield from self.stem_embeddings.parameters()
        yield from self.iir_memories.parameters()

    # -----------------------------------------------------------------
    # Delegated properties (compatibility with training harness)
    # -----------------------------------------------------------------

    @property
    def layers(self):
        return self.lm_transformer.layers

    @property
    def stem_layers(self):
        return self.lm_transformer.stem_layers

    @property
    def rope_embeddings(self):
        return self.lm_transformer.rope_embeddings

    @property
    def max_seqlen(self):
        return self.lm_transformer.max_seqlen

    @property
    def dim(self):
        return self.lm_transformer.dim

    @property
    def weight_tying(self):
        return self.lm_transformer.weight_tying

    @property
    def sliding_window(self):
        return self.lm_transformer.sliding_window

    def set_requires_gradient_sync(self, requires_sync: bool):
        """Delegate FSDP gradient sync to inner backbone."""
        if hasattr(self.lm_transformer, "set_requires_gradient_sync"):
            self.lm_transformer.set_requires_gradient_sync(requires_sync)


# =========================================================================
# Qwen3 & OLMo3 IIR-STEM variants
# =========================================================================

class Qwen3IIRStemLMTransformer(IIRStemLMTransformer):
    """Qwen3 backbone (per-head QK-norm, pre-norm) + IIR-STEM."""
    _lm_transformer_cls = Qwen3LMTransformer


class OLMo3IIRStemLMTransformer(IIRStemLMTransformer):
    """OLMo3 backbone (full-dim QK-norm, post-norm) + IIR-STEM."""
    _lm_transformer_cls = OLMo3LMTransformer


# =========================================================================
# FSDP grouping plans (prefixed with 'lm_transformer.')
# =========================================================================
# The plans are identical to the base STEM ones because the inner
# ``lm_transformer`` has the same structure.

def build_iir_stem_lm_fsdp_grouping_plan(model_args: IIRStemLMTransformerArgs):
    """FSDP plan for LLaMA-based IIR-StemLMTransformer."""
    return build_stem_lm_fsdp_grouping_plan(model_args)


def build_qwen3_iir_stem_lm_fsdp_grouping_plan(model_args: IIRStemLMTransformerArgs):
    """FSDP plan for Qwen3-based IIR-StemLMTransformer."""
    return build_qwen3_stem_lm_fsdp_grouping_plan(model_args)


def build_olmo3_iir_stem_lm_fsdp_grouping_plan(model_args: IIRStemLMTransformerArgs):
    """FSDP plan for OLMo3-based IIR-StemLMTransformer."""
    return build_olmo3_stem_lm_fsdp_grouping_plan(model_args)


# =========================================================================
# IIR-STEM model registry
#   model_type -> (stem_lm_cls, args_cls, build_fsdp_plan,
#                  get_no_recompute_ops, get_num_flop_per_token)
# =========================================================================

IIR_STEM_MODEL_REGISTRY = {
    "llama_iir": (
        IIRStemLMTransformer,
        IIRStemLMTransformerArgs,
        build_iir_stem_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3_iir": (
        Qwen3IIRStemLMTransformer,
        IIRStemLMTransformerArgs,
        build_qwen3_iir_stem_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3_iir": (
        OLMo3IIRStemLMTransformer,
        IIRStemLMTransformerArgs,
        build_olmo3_iir_stem_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}

