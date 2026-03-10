# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Selective-State-Space-STEM: Language-model wrappers with Mamba2-style
input-conditioned scalar gates for the STEM embedding memory.

Extends IIR-STEM by replacing the fixed-parameter ``IIRMemory`` with
:class:`~lingua.selective_iir.SelectiveIIRMemory`, whose forget gate
``μ_t`` and readout scale ``α_t`` are **conditioned on the transformer
hidden state** ``x_t`` at each position:

    μ_t       = σ(w_μᵀ x_t + b_μ)
    α_t       = w_αᵀ x_t + b_α
    m_t       = μ_t · m_{t-1} + (1 − μ_t) · e_t
    ẽ_t       = e_t + α_t · m_{t-1}

To thread the hidden state to the memory module, this file provides
thin ``LMTransformer`` subclasses whose ``forward()`` passes the
current residual-stream tensor ``h`` (the layer input) as a third
argument to ``stem_embeddings_fn``.  **No existing files are modified.**

Model registry
--------------
``SELECTIVE_IIR_STEM_MODEL_REGISTRY`` maps model-type strings to the
usual 5-tuple expected by the training harness:

    ``"llama_selective_iir"``  – LLaMA attention, pre-norm
    ``"qwen3_selective_iir"``  – Qwen3 per-head QK-norm, pre-norm
    ``"olmo3_selective_iir"``  – OLMo3 full-dim QK-norm, post-norm
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from lingua.transformer import RMSNorm, TiedLinear, cross_entropy
from lingua.stem_sss import (
    SSSStemTransformerArgs,
    SSSMemory,
)
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
class SelectiveIIRStemLMTransformerArgs(
    SSSStemTransformerArgs, LMTransformerArgs
):
    """Combined args for Selective-IIR-STEM language-model transformers.

    Configs use selective_iir_* names; these map to SSSMemory init params.
    """

    # Selective-IIR hyper-parameters (config key: selective_iir_mu_init, etc.)
    selective_iir_mu_init: float = 0.9
    selective_iir_alpha_init: float = 0.3
    selective_iir_learnable_m0: bool = False


# Alias for training script compatibility
SSSStemLMTransformerArgs = SelectiveIIRStemLMTransformerArgs


# =========================================================================
# LMTransformer subclasses that thread hidden states to the callback
# =========================================================================
# The only difference from the base ``LMTransformer.forward()`` is that
# ``stem_embeddings_fn`` is called with the current hidden state ``h``
# as a third positional argument:
#
#     y = stem_embeddings_fn(layer_idx, token_values, h)
#
# This is a one-line change, but we make it via subclassing to avoid
# modifying any existing file.

def _selective_stem_forward(
    self,
    token_values: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    tok_idx: Optional[torch.Tensor] = None,
    mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
    attn_impl: str = "sdpa",
    stem_embeddings_fn: Optional[
        Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
):
    """Forward pass that threads ``h`` to ``stem_embeddings_fn``.

    Identical to :meth:`LMTransformer.forward` except that the
    ``stem_embeddings_fn`` callback receives the current residual
    stream ``h`` as its third argument.
    """
    bsz, seqlen = token_values.shape

    h = self.tok_embeddings(token_values)

    mask = (
        mask
        if mask is not None
        else self._create_causal_mask(seqlen, attn_impl, self.sliding_window)
    )
    freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

    for i, layer in enumerate(self.layers):
        if i in self.stem_layers:
            if stem_embeddings_fn is not None:
                # ---- THE KEY CHANGE: pass h as third arg ----
                y = stem_embeddings_fn(i, token_values, h)
                h = layer(
                    h, freq_cis, y=y,
                    tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
                )
            else:
                h = layer(
                    h, freq_cis, y=None,
                    tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
                )
        else:
            h = layer(
                h, freq_cis,
                tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
            )

    logits = self.output(self.norm(h))
    if target is not None:
        return cross_entropy(logits, target)
    return logits


class SelectiveLMTransformer(LMTransformer):
    """LLaMA LMTransformer that threads hidden states to the callback."""
    forward = _selective_stem_forward


class SelectiveQwen3LMTransformer(Qwen3LMTransformer):
    """Qwen3 LMTransformer that threads hidden states to the callback."""
    forward = _selective_stem_forward


class SelectiveOLMo3LMTransformer(OLMo3LMTransformer):
    """OLMo3 LMTransformer that threads hidden states to the callback."""
    forward = _selective_stem_forward


# =========================================================================
# Selective-IIR-STEM LM Transformer (higher-level API)
# =========================================================================

class SelectiveIIRStemLMTransformer(nn.Module):
    """Higher-level wrapper analogous to :class:`IIRStemLMTransformer` but
    with :class:`SelectiveIIRMemory` modules that condition the forget gate
    and readout scale on the transformer hidden state.

    Architecture
    ------------
    * ``self.lm_transformer`` – FSDP-wrappable backbone (uses
      :class:`SelectiveLMTransformer` which threads ``h`` through the
      ``stem_embeddings_fn`` callback).
    * ``self.stem_embeddings`` – per-STEM-layer token embedding tables
      (``ParallelEmbedding``, managed outside FSDP).
    * ``self.selective_iir_memories`` – per-STEM-layer
      :class:`SelectiveIIRMemory` modules holding ``w_mu``, ``w_alpha``
      projections and optional ``m0``.

    Optimiser integration
    ---------------------
    Both ``stem_embeddings`` and ``selective_iir_memories`` live outside
    FSDP.  Use :meth:`stem_parameters` to collect them for a dedicated
    optimiser group.
    """

    # Subclasses override to use Qwen3/OLMo3 backbones.
    _lm_transformer_cls = SelectiveLMTransformer

    def __init__(self, args: SelectiveIIRStemLMTransformerArgs):
        super().__init__()
        self.args = args

        # 1) FSDP-wrappable language-model backbone
        self.lm_transformer = self._lm_transformer_cls(args)

        # 2) STEM embedding tables
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

        # 3) Per-layer Selective-IIR memory modules
        self.selective_iir_memories = nn.ModuleList([
            SSSMemory(
                dim=args.dim,
                d_ff=args.stem_embedding_dim,
                mu_init=args.selective_iir_mu_init,
                alpha_init=args.selective_iir_alpha_init,
                learnable_m0=args.selective_iir_learnable_m0,
            )
            for _ in range(len(self.lm_transformer.stem_layers))
        ])

        # 4) layer_idx → stem list index mapping
        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(
                self.lm_transformer.stem_layers
            )
        }

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[
            Union[BlockMask, AttentionBias, torch.Tensor, str]
        ] = None,
        attn_impl: str = "sdpa",
    ):
        """Forward pass with selective-IIR-contextual STEM embeddings."""

        def stem_embeddings_fn(
            layer_idx: int,
            token_values: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            stem_idx = self._layer_to_stem_idx[layer_idx]
            e = self.stem_embeddings[stem_idx](token_values)   # [B, L, d_ff]
            e_tilde = self.selective_iir_memories[stem_idx](
                e, hidden_states,
            )                                                   # [B, L, d_ff]
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
        """Reset parameters of ``stem_embeddings``."""
        import logging
        logger = logging.getLogger()
        for i, embedding in enumerate(self.stem_embeddings):
            weight_device = embedding.weight.device
            if weight_device.type == "meta":
                logger.warning(
                    f"stem_embeddings[{i}].weight on meta device, "
                    f"skipping init"
                )
                continue
            embedding.reset_parameters()
            if embedding.weight.numel() > 0:
                weight_norm = embedding.weight.norm().item()
                is_zero = (embedding.weight.abs().max() == 0).item()
                if is_zero:
                    logger.error(
                        f"stem_embeddings[{i}].weight still zeros "
                        f"after reset!"
                    )
                else:
                    logger.debug(
                        f"stem_embeddings[{i}].weight initialised: "
                        f"norm={weight_norm:.6f}, device={weight_device}"
                    )

    def init_weights(self):
        """Initialise all weights (backbone + stem embeddings + IIR gates).

        Selective-IIR parameters (``w_mu``, ``w_alpha``, ``m0``) are
        re-set to their configured initial values because ``to_empty()``
        leaves meta-materialised parameter memory uninitialised.
        """
        self.lm_transformer.init_weights()
        self.reset_stem_embeddings()
        for sel_iir_mem in self.selective_iir_memories:
            sel_iir_mem.reset_parameters()

    # -----------------------------------------------------------------
    # Convenience: all non-FSDP ("stem") parameters
    # -----------------------------------------------------------------

    def stem_parameters(self):
        """Yield all parameters that live *outside* ``lm_transformer``.

        This includes ``stem_embeddings`` **and**
        ``selective_iir_memories`` (w_mu, w_alpha, m0).
        """
        yield from self.stem_embeddings.parameters()
        yield from self.selective_iir_memories.parameters()

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
# Qwen3 & OLMo3 Selective-IIR-STEM variants
# =========================================================================

class Qwen3SelectiveIIRStemLMTransformer(SelectiveIIRStemLMTransformer):
    """Qwen3 backbone (per-head QK-norm, pre-norm) + Selective-IIR-STEM."""
    _lm_transformer_cls = SelectiveQwen3LMTransformer


class OLMo3SelectiveIIRStemLMTransformer(SelectiveIIRStemLMTransformer):
    """OLMo3 backbone (full-dim QK-norm, post-norm) + Selective-IIR-STEM."""
    _lm_transformer_cls = SelectiveOLMo3LMTransformer


# =========================================================================
# FSDP grouping plans
# =========================================================================
# Inner lm_transformer has the same structure as base STEM, so the
# FSDP plans are identical (prefixed with 'lm_transformer.').

def build_selective_iir_fsdp_plan(model_args):
    return build_stem_lm_fsdp_grouping_plan(model_args)


def build_qwen3_selective_iir_fsdp_plan(model_args):
    return build_qwen3_stem_lm_fsdp_grouping_plan(model_args)


def build_olmo3_selective_iir_fsdp_plan(model_args):
    return build_olmo3_stem_lm_fsdp_grouping_plan(model_args)


# =========================================================================
# Selective-IIR-STEM model registry
# =========================================================================

SELECTIVE_IIR_STEM_MODEL_REGISTRY = {
    "llama_selective_iir": (
        SelectiveIIRStemLMTransformer,
        SelectiveIIRStemLMTransformerArgs,
        build_selective_iir_fsdp_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "llama_sss": (
        SelectiveIIRStemLMTransformer,
        SelectiveIIRStemLMTransformerArgs,
        build_selective_iir_fsdp_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3_selective_iir": (
        Qwen3SelectiveIIRStemLMTransformer,
        SelectiveIIRStemLMTransformerArgs,
        build_qwen3_selective_iir_fsdp_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "qwen3_sss": (
        Qwen3SelectiveIIRStemLMTransformer,
        SelectiveIIRStemLMTransformerArgs,
        build_qwen3_selective_iir_fsdp_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3_selective_iir": (
        OLMo3SelectiveIIRStemLMTransformer,
        SelectiveIIRStemLMTransformerArgs,
        build_olmo3_selective_iir_fsdp_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
    "olmo3_sss": (
        OLMo3SelectiveIIRStemLMTransformer,
        SelectiveIIRStemLMTransformerArgs,
        build_olmo3_selective_iir_fsdp_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}