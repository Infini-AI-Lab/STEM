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
"""

from dataclasses import dataclass

import torch
from torch import nn

from lingua.transformer import RMSNorm, TiedLinear
from lingua.stem import StemTransformer
from lingua.stem_dag import (
    STEMDagTransformerArgs,
    STEMDagFeedForward,
    STEMDagTransformerBlock,
    STEMDagTransformer,
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
