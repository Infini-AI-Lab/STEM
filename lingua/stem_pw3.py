# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
PW3 (Pseudo W3), a low-rank sparse matrix for element-wise gating of STEM embeddings.

PW3 is a low-rank sparse matrix that is used to gate the STEM embeddings element-wise.
It is defined as:
    PW3 = U @ V^T
where U and V are low-rank matrices.

The low-rank structure of PW3 allows for efficient computation of the gate matrix.

This module lives at the library level and does **not** import from
``apps.main``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from lingua.stem import StemTransformerArgs
from lingua.sss import (
    selective_iir_scan_triton,
    selective_iir_scan_generate_triton,
    _inverse_sigmoid,
)

# ---------------------------------------------------------------------------
# PW3 class
# ---------------------------------------------------------------------------

class PW3(nn.Module):
    """PW3 (Pseudo W3), a low-rank sparse matrix for element-wise gating of STEM embeddings."""

    def __init__(self, d_model: int, d_ff: int, r: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.r = r
        self.dtype = dtype
        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(r, d_ff, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.U.weight, mean=0.0, std=0.02, a=-2, b=2)
        nn.init.trunc_normal_(self.V.weight, mean=0.0, std=0.02, a=-2, b=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.V(F.silu(self.U(x)))
        return self.V(self.U(x))

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class PW3StemTransformerArgs(StemTransformerArgs):
    """Extra hyper-parameters for the PW3 (Pseudo W3) stem transformer."""

    r: int = 64

