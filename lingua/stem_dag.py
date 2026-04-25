# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core DAG (Directed Acyclic Graph) FFN and transformer block components.

In the DAG variant, stem layers combine a local projection ``w3(x)`` and the
stem embedding ``y`` in the gate position of the SwiGLU FFN.  Two combination
modes are supported, selected by ``alpha_mode`` on
:class:`STEMDagTransformerArgs`:

* ``"sigmoid_gated"`` (default) — the original formulation with a learnable
  per-layer scalar ``alpha``::

      up = (1 - sigmoid(alpha)) * w3(x) + sigmoid(alpha) * y

  This smoothly interpolates between the SwiGLU gate ``w3(x)`` and the stem
  signal ``y``.

* ``"sum"`` — no gate, no learnable scalar; the two paths are simply added::

      up = w3(x) + y

  This matches the behaviour of DAG checkpoints that were trained to add the
  two paths directly and therefore never saved an ``alpha`` parameter.  When
  this mode is selected the ``alpha`` parameter is *not* constructed, so the
  backbone checkpoint load path does not look for a missing key and no stale
  ``alpha_init`` scalar silently suppresses the stem contribution at eval
  time.

This module lives at the library level and does **not** import from
``apps.main``.  Application-level wrappers (LM head, FSDP plans,
registry) are in ``apps.main.stem_dag``.
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

_DAG_ALPHA_MODES = ("sigmoid_gated", "sum")


@dataclass
class STEMDagTransformerArgs(StemTransformerArgs):
    alpha_init: float = -5.0  # sigmoid(-5.0) ≈ 0.0067 (almost pure y to start)
    # How to combine w3(x) and y in the SwiGLU gate slot for stem layers.
    # "sigmoid_gated": up = (1 - sigmoid(alpha)) * w3(x) + sigmoid(alpha) * y
    #                  with a learnable per-layer scalar alpha initialised to alpha_init.
    # "sum":           up = w3(x) + y, no learnable gate (no alpha parameter).
    #                  Use this to evaluate checkpoints trained with the
    #                  simple-sum variant (no saved alpha).
    alpha_mode: str = "sigmoid_gated"


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
        alpha_mode: str = "sigmoid_gated",
    ):
        super().__init__(dim, hidden_dim, multiple_of, ffn_dim_multiplier, mp_size)
        if alpha_mode not in _DAG_ALPHA_MODES:
            raise ValueError(
                f"Unknown alpha_mode={alpha_mode!r}; expected one of {_DAG_ALPHA_MODES}"
            )
        self.w3 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.alpha_mode = alpha_mode
        self.alpha_init = alpha_init
        if alpha_mode == "sigmoid_gated":
            # Learnable blending scalar only exists in the gated variant.  In
            # "sum" mode we deliberately do not register an ``alpha`` parameter
            # so that (a) checkpoint loading does not flag it as missing and
            # (b) no stale ``alpha_init`` can silently suppress the stem
            # contribution through sigmoid(alpha_init).
            self.alpha = nn.Parameter(torch.tensor([alpha_init]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        if self.alpha_mode == "sum":
            up = x3 + y
        else:
            sigmoid_alpha = torch.sigmoid(self.alpha)
            up = (1.0 - sigmoid_alpha) * x3 + sigmoid_alpha * y
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
        if self.alpha_mode == "sigmoid_gated":
            self.alpha.data.fill_(self.alpha_init)


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
                alpha_mode=args.alpha_mode,
            )


# =========================================================================
# DAG StemTransformer
# =========================================================================

class STEMDagTransformer(StemTransformer):
    _block_cls = STEMDagTransformerBlock
