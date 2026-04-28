# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core DAG (Directed Acyclic Graph) FFN and transformer block components.

In the DAG variant, stem layers use a learnable alpha gate that blends
between a local projection (w3(x)) and the stem embedding (y):

    up = sigmoid(alpha) * w3(x) + (1 - sigmoid(alpha)) * y

This allows the model to smoothly interpolate between using the
hidden state context (via w3) and the stem embedding signal.

This module lives at the library level and does **not** import from
``apps.main``.  Application-level wrappers (LM head, FSDP plans,
registry) are in ``apps.main.stem_dag``.

**DAG distill variant** (``STEMDagDistill*``): stem layers use the standard
base FFN residual ``w2(SiLU(w1(x)) * w3(x))`` only (no addition of stem ``y``
in the main path). When ``y`` is provided and gradients are enabled, a
layerwise MSE distills ``w2(gate * y)`` toward ``w2(gate * w3(x))`` with
``gate`` and ``w2`` detached in that branch so only stem embeddings receive
gradients from the distill term; CE uses the main path without ``y``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from lingua.stem import (
    StemTransformerArgs,
    StemFeedForward,
    StemTransformerBlock,
    StemTransformer,
)


# =========================================================================
# Args
# =========================================================================

@dataclass
class STEMDagTransformerArgs(StemTransformerArgs):
    alpha_init: float = -5.0  # sigmoid(-5.0) ≈ 0.0067 (almost pure y to start)


# =========================================================================
# DAG FeedForward
# =========================================================================

class STEMDagFeedForward(StemFeedForward):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
        alpha_init: float = -5.0,
    ):
        super().__init__(dim, hidden_dim, multiple_of, ffn_dim_multiplier, mp_size)
        self.w3 = nn.Linear(dim, self.hidden_dim, bias=False)
        # self.alpha_init = alpha_init
        # self.alpha = nn.Parameter(torch.tensor([alpha_init]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        # sigmoid_alpha = torch.sigmoid(self.alpha)
        # up = (1.0 - sigmoid_alpha) * x3 + sigmoid_alpha * y
        up = x3 + y # ablation: remove gating and just sum the projection and stem signal
        output = self.w2(F.silu(x1) * up)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        super().reset_parameters(init_std, factor)
        in_init_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        # self.alpha.data.fill_(self.alpha_init)


# =========================================================================
# DAG Transformer Block (LLaMA attention, pre-norm)
# =========================================================================

class STEMDagTransformerBlock(StemTransformerBlock):
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
# DAG StemTransformer
# =========================================================================

class STEMDagTransformer(StemTransformer):
    _block_cls = STEMDagTransformerBlock


# =========================================================================
# DAG distill: args, FFN, block, transformer
# =========================================================================


@dataclass
class STEMDagDistillTransformerArgs(STEMDagTransformerArgs):
    """Extra hyper-parameters for layerwise down-projection distill DAG-STEM."""

    stem_distill_loss_weight: float = 1.0


class STEMDagDistillFeedForward(StemFeedForward):
    """Stem FFN: main path uses ``w3`` only; optional layerwise down-proj MSE vs ``y``."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
        alpha_init: float = -5.0,
    ):
        super().__init__(dim, hidden_dim, multiple_of, ffn_dim_multiplier, mp_size)
        self.w3 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.last_distill_loss = None

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # y is Optional because LMTransformer passes y=None when stem_embeddings_fn is absent.
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        gate = F.silu(x1)
        h = gate * x3

        # MSE in FFN *hidden* space ([B,S,H] with H~11k) blows up memory; distill on
        # *down-projected* activations ([B,S,dim]) matches stem_layerwise_finetune down_proj.
        self.last_distill_loss = None
        w2_snapshot = None
        if y is not None and torch.is_grad_enabled():
            with torch.no_grad():
                w2_snapshot = (
                    self.w2.weight.detach().clone().to(dtype=gate.dtype).contiguous()
                )

        ffn_out = self.w2(h)

        if w2_snapshot is not None:
            gf = gate.detach()
            pred = F.linear(gf * y.to(dtype=gf.dtype), w2_snapshot)
            self.last_distill_loss = F.mse_loss(pred, ffn_out.detach())

        return ffn_out
        

    def reset_parameters(self, init_std=None, factor=1.0):
        super().reset_parameters(init_std, factor)
        in_init_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )


class STEMDagDistillTransformerBlock(StemTransformerBlock):
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


class STEMDagDistillTransformer(StemTransformer):
    _block_cls = STEMDagDistillTransformerBlock
