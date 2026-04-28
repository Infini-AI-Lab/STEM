# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
DAG-STEM: Language-model wrappers with DAG (alpha-gated) FFN for STEM layers.

The DAG variant replaces the StemFeedForward with STEMDagFeedForward in stem
layers, adding a learnable alpha gate that blends between w3(x) and the stem
embedding y:

    up = sigmoid(alpha) * w3(x) + (1 - sigmoid(alpha)) * y

The inner transformer blocks keep w3 as a learnable projection (unlike base
STEM which removes w3 entirely), allowing the model to smoothly interpolate
between using the hidden state context and the stem embedding signal.

Model registry
--------------
``DAG_STEM_MODEL_REGISTRY`` maps model-type strings to the usual 5-tuple
expected by ``stem_train.py``:

    ``"llama_dag"``  -- LLaMA attention, pre-norm
    ``"qwen3_dag"``  -- Qwen3 per-head QK-norm, pre-norm
    ``"olmo3_dag"``  -- OLMo3 full-dim QK-norm, post-norm

``DAG_STEM_DISTILL_MODEL_REGISTRY`` adds layerwise down-projection distill
DAG variants (``*_dag_distill``): CE on the base FFN path; stem embeddings
trained via per-layer MSE (see ``lingua.stem_dag`` docstring).

With ``distributed.compile: true``, the distill training loss tail runs
outside ``torch.compile`` (``torch.compiler.disable``) so Inductor does not
trace ``StemTrainLossOut`` / per-layer ``last_distill_loss`` assembly; the
main path up to ``logits`` still compiles.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import nn

from lingua.transformer import RMSNorm, TiedLinear, cross_entropy
from lingua.stem import StemTransformer
from lingua.stem_dag import (
    STEMDagTransformerArgs,
    STEMDagFeedForward,
    STEMDagTransformerBlock,
    STEMDagTransformer,
    STEMDagDistillTransformerArgs,
    STEMDagDistillFeedForward,
    STEMDagDistillTransformer,
)

from apps.main.stem import (
    LMTransformer,
    Qwen3StemTransformerBlock,
    OLMo3StemTransformerBlock,
    StemLMTransformer,
    build_stem_lm_fsdp_grouping_plan,
    build_qwen3_stem_lm_fsdp_grouping_plan,
    build_olmo3_stem_lm_fsdp_grouping_plan,
)
from apps.main.stem_train import StemTrainLossOut
from apps.main.transformer import (
    LMTransformerArgs,
    create_causal_mask as llama_create_causal_mask,
    get_no_recompute_ops as llama_get_no_recompute_ops,
    get_num_flop_per_token as llama_get_num_flop_per_token,
)
from apps.main.qwen3 import (
    create_causal_mask as qwen3_create_causal_mask,
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
    get_num_flop_per_token as qwen3_get_num_flop_per_token,
)
from apps.main.olmo3 import (
    create_causal_mask as olmo3_create_causal_mask,
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
    get_num_flop_per_token as olmo3_get_num_flop_per_token,
)


# =========================================================================
# Args
# =========================================================================

@dataclass
class STEMDagLMTransformerArgs(STEMDagTransformerArgs, LMTransformerArgs):
    """Combined args for DAG-STEM language-model transformers."""
    pass


@dataclass
class STEMDagDistillLMTransformerArgs(STEMDagDistillTransformerArgs, LMTransformerArgs):
    """Combined args for DAG-distill STEM language-model transformers."""
    pass


# =========================================================================
# Qwen3 / OLMo3 DAG transformer blocks
# =========================================================================

class Qwen3STEMDagTransformerBlock(Qwen3StemTransformerBlock):
    """Qwen3-style block (per-head QK-norm, pre-norm) with DAG FFN for stem layers."""
    def __init__(self, layer_idx: int, args: STEMDagTransformerArgs):
        super().__init__(layer_idx, args)
        if layer_idx in args.stem_layers:
            self.feed_forward = STEMDagFeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                alpha_init=args.alpha_init,
            )


class OLMo3STEMDagTransformerBlock(OLMo3StemTransformerBlock):
    """OLMo3-style block (full-dim QK-norm, post-norm) with DAG FFN for stem layers."""
    def __init__(self, layer_idx: int, args: STEMDagTransformerArgs):
        super().__init__(layer_idx, args)
        if layer_idx in args.stem_layers:
            self.feed_forward = STEMDagFeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                alpha_init=args.alpha_init,
            )


# =========================================================================
# StemTransformer subclasses for Qwen3 / OLMo3 DAG
# =========================================================================

class Qwen3STEMDagTransformer(StemTransformer):
    _block_cls = Qwen3STEMDagTransformerBlock


class OLMo3STEMDagTransformer(StemTransformer):
    _block_cls = OLMo3STEMDagTransformerBlock


# =========================================================================
# Qwen3 / OLMo3 DAG-distill transformer blocks
# =========================================================================


class Qwen3STEMDagDistillTransformerBlock(Qwen3StemTransformerBlock):
    """Qwen3-style block with layerwise down-proj distill FFN on stem layers."""

    def __init__(self, layer_idx: int, args: STEMDagDistillTransformerArgs):
        super().__init__(layer_idx, args)
        if layer_idx in args.stem_layers:
            self.feed_forward = STEMDagDistillFeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                alpha_init=args.alpha_init,
            )


class OLMo3STEMDagDistillTransformerBlock(OLMo3StemTransformerBlock):
    """OLMo3-style block with layerwise down-proj distill FFN on stem layers."""

    def __init__(self, layer_idx: int, args: STEMDagDistillTransformerArgs):
        super().__init__(layer_idx, args)
        if layer_idx in args.stem_layers:
            self.feed_forward = STEMDagDistillFeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                alpha_init=args.alpha_init,
            )


class Qwen3STEMDagDistillTransformer(StemTransformer):
    _block_cls = Qwen3STEMDagDistillTransformerBlock


class OLMo3STEMDagDistillTransformer(StemTransformer):
    _block_cls = OLMo3STEMDagDistillTransformerBlock


# =========================================================================
# Base DAG LMTransformer (LLaMA attention, pre-norm)
# =========================================================================

class DagLMTransformer(STEMDagTransformer):
    """DAG LM transformer with LLaMA attention (pre-norm)."""
    _create_causal_mask = staticmethod(llama_create_causal_mask)

    def __init__(self, args: STEMDagLMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = LMTransformer.forward

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
            )


# =========================================================================
# Qwen3 DAG LMTransformer
# =========================================================================

class Qwen3DagLMTransformer(Qwen3STEMDagTransformer):
    """Qwen3 DAG LM transformer (per-head QK-norm, pre-norm)."""
    _create_causal_mask = staticmethod(qwen3_create_causal_mask)

    def __init__(self, args: STEMDagLMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = LMTransformer.forward

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
            )


# =========================================================================
# OLMo3 DAG LMTransformer
# =========================================================================

class OLMo3DagLMTransformer(OLMo3STEMDagTransformer):
    """OLMo3 DAG LM transformer (full-dim QK-norm, post-norm)."""
    _create_causal_mask = staticmethod(olmo3_create_causal_mask)

    def __init__(self, args: STEMDagLMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = LMTransformer.forward

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
            )


def dag_distill_lm_forward(
    self,
    token_values: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    tok_idx: Optional[torch.Tensor] = None,
    mask: Optional[Any] = None,
    attn_impl: str = "sdpa",
    stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
):
    _, seqlen = token_values.shape

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
                y = stem_embeddings_fn(i, token_values)
                h = layer(h, freq_cis, y=y, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
            else:
                h = layer(h, freq_cis, y=None, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        else:
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

    logits = self.output(self.norm(h))
    if target is None:
        return logits

    ce = cross_entropy(logits, target)
    distill = 0.0
    for i in self.stem_layers:
        distill += self.layers[i].feed_forward.last_distill_loss
    
    total = ce + self.stem_distill_loss_weight * (distill / len(self.stem_layers))
    loss_for_backward = total + (ce - total).detach()
    return StemTrainLossOut(loss=loss_for_backward, distill_loss=distill.detach())


# =========================================================================
# DAG-distill LMTransformers
# =========================================================================


class DagDistillLMTransformer(STEMDagDistillTransformer):
    """LLaMA attention DAG-distill LM (pre-norm)."""

    _create_causal_mask = staticmethod(llama_create_causal_mask)

    def __init__(self, args: STEMDagDistillLMTransformerArgs):
        super().__init__(args)
        self.stem_distill_loss_weight = args.stem_distill_loss_weight
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = dag_distill_lm_forward

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
            )


class Qwen3DagDistillLMTransformer(Qwen3STEMDagDistillTransformer):
    """Qwen3 DAG-distill LM transformer (per-head QK-norm, pre-norm)."""

    _create_causal_mask = staticmethod(qwen3_create_causal_mask)

    def __init__(self, args: STEMDagDistillLMTransformerArgs):
        super().__init__(args)
        self.stem_distill_loss_weight = args.stem_distill_loss_weight
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = dag_distill_lm_forward

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
            )


class OLMo3DagDistillLMTransformer(OLMo3STEMDagDistillTransformer):
    """OLMo3 DAG-distill LM transformer (full-dim QK-norm, post-norm)."""

    _create_causal_mask = staticmethod(olmo3_create_causal_mask)

    def __init__(self, args: STEMDagDistillLMTransformerArgs):
        super().__init__(args)
        self.stem_distill_loss_weight = args.stem_distill_loss_weight
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = dag_distill_lm_forward

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std,
            )


# =========================================================================
# DAG StemLMTransformer (higher-level API with separate stem_embeddings)
# =========================================================================

class STEMDagLMTransformer(StemLMTransformer):
    """LLaMA-based DAG StemLMTransformer."""
    _lm_transformer_cls = DagLMTransformer


class Qwen3STEMDagLMTransformer(StemLMTransformer):
    """Qwen3-based DAG StemLMTransformer."""
    _lm_transformer_cls = Qwen3DagLMTransformer


class OLMo3STEMDagLMTransformer(StemLMTransformer):
    """OLMo3-based DAG StemLMTransformer."""
    _lm_transformer_cls = OLMo3DagLMTransformer


class STEMDagDistillLMTransformer(StemLMTransformer):
    """LLaMA-based DAG-distill StemLMTransformer."""
    _lm_transformer_cls = DagDistillLMTransformer


class Qwen3STEMDagDistillLMTransformer(StemLMTransformer):
    """Qwen3-based DAG-distill StemLMTransformer."""
    _lm_transformer_cls = Qwen3DagDistillLMTransformer


class OLMo3STEMDagDistillLMTransformer(StemLMTransformer):
    """OLMo3-based DAG-distill StemLMTransformer."""
    _lm_transformer_cls = OLMo3DagDistillLMTransformer


# =========================================================================
# FSDP grouping plans (prefixed with 'lm_transformer.')
# =========================================================================
# The plans are identical to the base STEM ones because the inner
# ``lm_transformer`` has the same layer structure.

def build_dag_stem_lm_fsdp_grouping_plan(model_args: STEMDagLMTransformerArgs):
    """FSDP plan for LLaMA-based DAG-StemLMTransformer."""
    return build_stem_lm_fsdp_grouping_plan(model_args)


def build_qwen3_dag_stem_lm_fsdp_grouping_plan(model_args: STEMDagLMTransformerArgs):
    """FSDP plan for Qwen3-based DAG-StemLMTransformer."""
    return build_qwen3_stem_lm_fsdp_grouping_plan(model_args)


def build_olmo3_dag_stem_lm_fsdp_grouping_plan(model_args: STEMDagLMTransformerArgs):
    """FSDP plan for OLMo3-based DAG-StemLMTransformer."""
    return build_olmo3_stem_lm_fsdp_grouping_plan(model_args)


def build_dag_stem_distill_lm_fsdp_grouping_plan(model_args: STEMDagDistillLMTransformerArgs):
    """FSDP plan for LLaMA-based DAG-distill StemLMTransformer."""
    return build_stem_lm_fsdp_grouping_plan(model_args)


def build_qwen3_dag_stem_distill_lm_fsdp_grouping_plan(model_args: STEMDagDistillLMTransformerArgs):
    """FSDP plan for Qwen3-based DAG-distill StemLMTransformer."""
    return build_qwen3_stem_lm_fsdp_grouping_plan(model_args)


def build_olmo3_dag_stem_distill_lm_fsdp_grouping_plan(model_args: STEMDagDistillLMTransformerArgs):
    """FSDP plan for OLMo3-based DAG-distill StemLMTransformer."""
    return build_olmo3_stem_lm_fsdp_grouping_plan(model_args)


# =========================================================================
# DAG-STEM model registry
#   model_type -> (stem_lm_cls, args_cls, build_fsdp_plan,
#                  get_no_recompute_ops, get_num_flop_per_token)
# =========================================================================

DAG_STEM_MODEL_REGISTRY = {
    "llama_dag": (
        STEMDagLMTransformer,
        STEMDagLMTransformerArgs,
        build_dag_stem_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3_dag": (
        Qwen3STEMDagLMTransformer,
        STEMDagLMTransformerArgs,
        build_qwen3_dag_stem_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3_dag": (
        OLMo3STEMDagLMTransformer,
        STEMDagLMTransformerArgs,
        build_olmo3_dag_stem_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}


DAG_STEM_DISTILL_MODEL_REGISTRY = {
    "llama_dag_distill": (
        STEMDagDistillLMTransformer,
        STEMDagDistillLMTransformerArgs,
        build_dag_stem_distill_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3_dag_distill": (
        Qwen3STEMDagDistillLMTransformer,
        STEMDagDistillLMTransformerArgs,
        build_qwen3_dag_stem_distill_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3_dag_distill": (
        OLMo3STEMDagDistillLMTransformer,
        STEMDagDistillLMTransformerArgs,
        build_olmo3_dag_stem_distill_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}
