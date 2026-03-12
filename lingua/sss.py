# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Selective-State-Space: Mamba2-style input-conditioned scalar gates for STEM
embedding memory, with fused Triton kernels.

Extends the fixed-parameter Selective-IIR (:mod:`lingua.sss`) with
**input-conditioned** scalar forget and readout gates, analogous to the
selective mechanism in Mamba2's Structured State Space Duality (SSD):

    mu_t       = sigma(w_mu^T x_t + b_mu)         scalar forget gate in (0,1)
    alpha_t    = w_alpha^T x_t + b_alpha           scalar readout scale in R
    m_t        = mu_t * m_{t-1} + (1 - mu_t) * e_t   selective memory update
    ctx_t      = m_{t-1}                              causal context
    e_tilde_t  = e_t + alpha_t * ctx_t                contextual STEM embedding

where ``x_t`` is the transformer hidden state (residual stream input to
the current layer, shape ``[B, L, dim]``) and ``e_t`` is the raw STEM
embedding (``stem_emb[token_id]``, shape ``[B, L, d_ff]``).

Triton kernels
--------------
Three fused Triton kernels are provided:

* ``selective_iir_fwd_kernel`` -- training forward: given pre-computed
  ``mu_seq [B,L,1]`` and ``alpha_seq [B,L,1]``, runs the causal scan
  over D-dimension blocks, outputting e_tilde and saving ctx (the
  per-timestep m_{t-1}) for the backward pass.

* ``selective_iir_bwd_kernel`` -- training backward: reverse-time scan
  that computes dL/de, dL/d_mu_seq, dL/d_alpha_seq, and dL/d_m0.
  Scalar gradients for mu_seq and alpha_seq are reduced across dimension
  blocks via ``tl.atomic_add``.

* ``selective_iir_fwd_generate_kernel`` -- inference-only: per-batch
  initial memory m0 [B,D], outputs final memory state for caching.

The w_mu/w_alpha linear projections and sigmoid are computed in PyTorch
*before* the kernel call.  Autograd back-propagates through them
automatically since the kernel provides grad_mu_seq and grad_alpha_seq.

Key design choice
-----------------
Both mu_t and alpha_t are **scalars** (not per-dimension vectors).
This is deliberate:

1. Keeps parameter count negligible: 2*dim + 2 per STEM layer.
2. Preserves hardware-efficient scan structure (scalar decay =>
   the affine recurrence monoid has scalar-times-vector composition).
3. Per-dimension selectivity is already provided by the downstream
   SiLU gate w1(x) in the FFN.

Initialization
--------------
Gate projections are initialized to **zero weights** with biases set so
that the initial behavior matches a fixed IIR with the configured
mu_init and alpha_init.  This means the model starts in the IIR-STEM
regime and learns to deviate as training progresses.

This module lives at the library level and does **not** import from
``apps.main``.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import triton
import triton.language as tl

from lingua.stem import StemTransformerArgs


# =========================================================================
# Helpers
# =========================================================================

_TORCH_TO_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _triton_dtype(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL_DTYPE:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return _TORCH_TO_TL_DTYPE[dtype]


def _inverse_sigmoid(x: float) -> float:
    """Compute logit(x) = log(x / (1 - x)) for x in (0, 1)."""
    assert 0.0 < x < 1.0, f"mu_init must be in (0, 1), got {x}"
    return math.log(x / (1.0 - x))


# =========================================================================
# Triton forward kernel (training)
# =========================================================================

@triton.jit
def selective_iir_fwd_kernel(
    # Pointers
    e_ptr, out_ptr, ctx_ptr,
    mu_seq_ptr, alpha_seq_ptr, m0_ptr,
    # Dimensions
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # Strides for e, out, ctx: [B, L, D]
    stride_eb: tl.constexpr, stride_el: tl.constexpr, stride_ed: tl.constexpr,
    stride_ob: tl.constexpr, stride_ol: tl.constexpr, stride_od: tl.constexpr,
    stride_cb: tl.constexpr, stride_cl: tl.constexpr, stride_cd: tl.constexpr,
    # Strides for mu_seq, alpha_seq: [B, L, 1]
    stride_mub: tl.constexpr, stride_mul: tl.constexpr,
    stride_ab: tl.constexpr, stride_al: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Forward scan: one program per (batch, dim-block).

    Sequential over time, parallel over dimension blocks.
    Each program processes BLOCK_D dimensions for one batch element.
    """
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d < D

    # Initial memory m_{-1} = m0, shared across batch (broadcast)
    m_prev = tl.load(m0_ptr + d, mask=mask_d, other=0.0).to(tl.float32)

    # Base pointers for this batch element
    e_b = e_ptr + pid_b * stride_eb
    o_b = out_ptr + pid_b * stride_ob
    c_b = ctx_ptr + pid_b * stride_cb
    mu_b = mu_seq_ptr + pid_b * stride_mub
    a_b = alpha_seq_ptr + pid_b * stride_ab

    for t in range(0, L):
        # Load per-timestep scalar gates (broadcast to [BLOCK_D])
        mu_t = tl.load(mu_b + t * stride_mul).to(tl.float32)
        alpha_t = tl.load(a_b + t * stride_al).to(tl.float32)

        # Load embedding
        e_t = tl.load(
            e_b + t * stride_el + d * stride_ed,
            mask=mask_d, other=0.0,
        ).to(tl.float32)

        # Save ctx[t] = m_{t-1} for backward
        tl.store(
            c_b + t * stride_cl + d * stride_cd,
            m_prev.to(OUT_DTYPE), mask=mask_d,
        )

        # Output: e_tilde_t = e_t + alpha_t * m_{t-1}
        out_t = e_t + alpha_t * m_prev
        tl.store(
            o_b + t * stride_ol + d * stride_od,
            out_t.to(OUT_DTYPE), mask=mask_d,
        )

        # Memory update: m_t = mu_t * m_{t-1} + (1 - mu_t) * e_t
        m_prev = mu_t * m_prev + (1.0 - mu_t) * e_t


# =========================================================================
# Triton backward kernel (training)
# =========================================================================

@triton.jit
def selective_iir_bwd_kernel(
    # Saved from forward
    e_ptr, ctx_ptr,
    # Upstream gradient
    gy_ptr,
    # Output gradients
    ge_ptr, gmu_seq_ptr, galpha_seq_ptr, gm0_ptr,
    # Gate sequences (needed for the recurrence coefficients)
    mu_seq_ptr, alpha_seq_ptr,
    # Dimensions
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # Strides for e, ctx, gy, ge: [B, L, D]
    stride_eb: tl.constexpr, stride_el: tl.constexpr, stride_ed: tl.constexpr,
    stride_cb: tl.constexpr, stride_cl: tl.constexpr, stride_cd: tl.constexpr,
    stride_gyb: tl.constexpr, stride_gyl: tl.constexpr, stride_gyd: tl.constexpr,
    stride_geb: tl.constexpr, stride_gel: tl.constexpr, stride_ged: tl.constexpr,
    # Strides for mu_seq, alpha_seq: [B, L, 1]
    stride_mub: tl.constexpr, stride_mul: tl.constexpr,
    stride_ab: tl.constexpr, stride_al: tl.constexpr,
    # Strides for gmu_seq, galpha_seq: [B, L, 1]
    stride_gmub: tl.constexpr, stride_gmul: tl.constexpr,
    stride_gab: tl.constexpr, stride_gal: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
    GE_DTYPE: tl.constexpr,
):
    """Backward scan: reverse-time adjoint for the selective IIR recurrence.

    Produces gradients for:
    - ge:         [B, L, D]  gradient w.r.t. STEM embeddings e
    - gmu_seq:    [B, L, 1]  gradient w.r.t. post-sigmoid forget gates
    - galpha_seq: [B, L, 1]  gradient w.r.t. readout scales
    - gm0:        [D]        gradient w.r.t. initial memory

    Since gmu_seq and galpha_seq are per-(batch, timestep) scalars
    obtained by reducing across D, and each program handles a BLOCK_D
    chunk, partial sums are accumulated via tl.atomic_add.
    """
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d < D

    # Base pointers for this batch element
    e_b = e_ptr + pid_b * stride_eb
    c_b = ctx_ptr + pid_b * stride_cb
    gy_b = gy_ptr + pid_b * stride_gyb
    ge_b = ge_ptr + pid_b * stride_geb
    mu_b = mu_seq_ptr + pid_b * stride_mub
    a_b = alpha_seq_ptr + pid_b * stride_ab
    gmu_b = gmu_seq_ptr + pid_b * stride_gmub
    ga_b = galpha_seq_ptr + pid_b * stride_gab

    # Adjoint for memory state: starts at 0 (no loss depends on m_L)
    gm_next = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Reverse scan over time
    for t in range(L - 1, -1, -1):
        # Load saved values
        e_t = tl.load(
            e_b + t * stride_el + d * stride_ed,
            mask=mask_d, other=0.0,
        ).to(tl.float32)
        m_prev = tl.load(
            c_b + t * stride_cl + d * stride_cd,
            mask=mask_d, other=0.0,
        ).to(tl.float32)
        gy_t = tl.load(
            gy_b + t * stride_gyl + d * stride_gyd,
            mask=mask_d, other=0.0,
        ).to(tl.float32)

        # Load gates for this timestep
        mu_t = tl.load(mu_b + t * stride_mul).to(tl.float32)
        alpha_t = tl.load(a_b + t * stride_al).to(tl.float32)

        # ---- Gradient of out_t = e_t + alpha_t * m_{t-1} ----
        ge_t = gy_t                                     # dL/de_t from output
        galpha_t_partial = tl.sum(gy_t * m_prev, axis=0)  # scalar partial
        gm_from_out = alpha_t * gy_t                    # dL/dm_{t-1} from output

        # ---- Gradient of m_t = mu_t * m_{t-1} + (1 - mu_t) * e_t ----
        ge_t += (1.0 - mu_t) * gm_next                 # dL/de_t from recurrence
        gmu_t_partial = tl.sum(gm_next * (m_prev - e_t), axis=0)

        # Propagate memory adjoint backward
        gm_next = gm_from_out + mu_t * gm_next

        # Store dL/de_t
        tl.store(
            ge_b + t * stride_gel + d * stride_ged,
            ge_t.to(GE_DTYPE), mask=mask_d,
        )

        # Atomically accumulate scalar gradients (reduced across D blocks)
        tl.atomic_add(gmu_b + t * stride_gmul, gmu_t_partial)
        tl.atomic_add(ga_b + t * stride_gal, galpha_t_partial)

    # After loop: gm_next is dL/dm0 for this (batch, dim-block).
    # m0 is broadcast across batch => accumulate over B via atomic_add.
    tl.atomic_add(gm0_ptr + d, gm_next, mask=mask_d)


# =========================================================================
# Autograd wrapper
# =========================================================================

class SelectiveIIRScanFn(torch.autograd.Function):
    """Custom autograd function wrapping the Triton selective-IIR scan.

    Forward inputs:
        e:         [B, L, D]  raw STEM embeddings (cuda, contiguous, bf16/fp16)
        mu_seq:    [B, L, 1]  per-timestep forget gates in (0,1), after sigmoid
        alpha_seq: [B, L, 1]  per-timestep readout scales
        m0:        [D]        initial memory vector (float32)

    Returns:
        out:       [B, L, D]  contextual STEM embeddings (same dtype as e)

    Note: mu_seq here is the *post-sigmoid* value.  The w_mu linear
    projection and sigmoid happen in PyTorch, so autograd will chain the
    kernel's grad_mu_seq back through sigmoid and nn.Linear automatically.
    """

    @staticmethod
    def forward(ctx, e, mu_seq, alpha_seq, m0):
        assert e.is_cuda and e.is_contiguous()
        assert e.ndim == 3
        B, L, D = e.shape
        assert mu_seq.shape == (B, L, 1), f"Expected ({B},{L},1), got {mu_seq.shape}"
        assert alpha_seq.shape == (B, L, 1), f"Expected ({B},{L},1), got {alpha_seq.shape}"
        assert m0.is_cuda and m0.ndim == 1 and m0.numel() == D

        mu_seq = mu_seq.contiguous()
        alpha_seq = alpha_seq.contiguous()

        out = torch.empty_like(e)
        ctx_mem = torch.empty_like(e)   # stores m_{t-1} per timestep

        OUT_DTYPE = _triton_dtype(e.dtype)
        BLOCK_D = 256 if D >= 256 else 128
        grid = (B, triton.cdiv(D, BLOCK_D))

        selective_iir_fwd_kernel[grid](
            e, out, ctx_mem,
            mu_seq, alpha_seq, m0,
            B=B, L=L, D=D,
            stride_eb=e.stride(0), stride_el=e.stride(1), stride_ed=e.stride(2),
            stride_ob=out.stride(0), stride_ol=out.stride(1), stride_od=out.stride(2),
            stride_cb=ctx_mem.stride(0), stride_cl=ctx_mem.stride(1), stride_cd=ctx_mem.stride(2),
            stride_mub=mu_seq.stride(0), stride_mul=mu_seq.stride(1),
            stride_ab=alpha_seq.stride(0), stride_al=alpha_seq.stride(1),
            BLOCK_D=BLOCK_D,
            OUT_DTYPE=OUT_DTYPE,
            num_warps=4,
        )

        ctx.save_for_backward(e, ctx_mem, mu_seq, alpha_seq, m0)
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, grad_out):
        e, ctx_mem, mu_seq, alpha_seq, m0 = ctx.saved_tensors
        assert grad_out.is_cuda and grad_out.is_contiguous()

        B, L, D = e.shape
        BLOCK_D = ctx.BLOCK_D

        ge = torch.empty_like(e)
        gmu_seq = torch.zeros(B, L, 1, dtype=torch.float32, device=e.device)
        galpha_seq = torch.zeros(B, L, 1, dtype=torch.float32, device=e.device)
        gm0 = torch.zeros(D, dtype=torch.float32, device=e.device)

        GE_DTYPE = _triton_dtype(ge.dtype)
        grid = (B, triton.cdiv(D, BLOCK_D))

        selective_iir_bwd_kernel[grid](
            e, ctx_mem,
            grad_out,
            ge, gmu_seq, galpha_seq, gm0,
            mu_seq, alpha_seq,
            B=B, L=L, D=D,
            stride_eb=e.stride(0), stride_el=e.stride(1), stride_ed=e.stride(2),
            stride_cb=ctx_mem.stride(0), stride_cl=ctx_mem.stride(1), stride_cd=ctx_mem.stride(2),
            stride_gyb=grad_out.stride(0), stride_gyl=grad_out.stride(1), stride_gyd=grad_out.stride(2),
            stride_geb=ge.stride(0), stride_gel=ge.stride(1), stride_ged=ge.stride(2),
            stride_mub=mu_seq.stride(0), stride_mul=mu_seq.stride(1),
            stride_ab=alpha_seq.stride(0), stride_al=alpha_seq.stride(1),
            stride_gmub=gmu_seq.stride(0), stride_gmul=gmu_seq.stride(1),
            stride_gab=galpha_seq.stride(0), stride_gal=galpha_seq.stride(1),
            BLOCK_D=BLOCK_D,
            GE_DTYPE=GE_DTYPE,
            num_warps=4,
        )

        return ge, gmu_seq, galpha_seq, gm0


def selective_iir_scan_triton(
    e: torch.Tensor,
    mu_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    m0: torch.Tensor,
) -> torch.Tensor:
    """Fused Triton selective-IIR scan (training, with autograd).

    Args:
        e:         [B, L, D]  STEM embeddings (cuda, contiguous, bf16/fp16).
        mu_seq:    [B, L, 1]  per-timestep forget gates in (0, 1).
        alpha_seq: [B, L, 1]  per-timestep readout scales.
        m0:        [D]        initial memory vector (float32).

    Returns:
        [B, L, D] contextual STEM embeddings (same dtype as e).
    """
    return SelectiveIIRScanFn.apply(e, mu_seq, alpha_seq, m0)


# =========================================================================
# Triton generate kernel (inference only)
# =========================================================================

@triton.jit
def selective_iir_fwd_generate_kernel(
    e_ptr, out_ptr,
    mu_seq_ptr, alpha_seq_ptr,
    m0_ptr, m_final_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    stride_eb: tl.constexpr, stride_el: tl.constexpr, stride_ed: tl.constexpr,
    stride_ob: tl.constexpr, stride_ol: tl.constexpr, stride_od: tl.constexpr,
    stride_mub: tl.constexpr, stride_mul: tl.constexpr,
    stride_ab: tl.constexpr, stride_al: tl.constexpr,
    stride_m0b: tl.constexpr, stride_m0d: tl.constexpr,
    stride_mfb: tl.constexpr, stride_mfd: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Inference-only forward scan with per-batch initial memory."""
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d < D

    m_prev = tl.load(
        m0_ptr + pid_b * stride_m0b + d * stride_m0d,
        mask=mask_d, other=0.0,
    ).to(tl.float32)

    e_b = e_ptr + pid_b * stride_eb
    o_b = out_ptr + pid_b * stride_ob
    mu_b = mu_seq_ptr + pid_b * stride_mub
    a_b = alpha_seq_ptr + pid_b * stride_ab

    for t in range(0, L):
        mu_t = tl.load(mu_b + t * stride_mul).to(tl.float32)
        alpha_t = tl.load(a_b + t * stride_al).to(tl.float32)

        e_t = tl.load(
            e_b + t * stride_el + d * stride_ed,
            mask=mask_d, other=0.0,
        ).to(tl.float32)

        out_t = e_t + alpha_t * m_prev
        tl.store(
            o_b + t * stride_ol + d * stride_od,
            out_t.to(OUT_DTYPE), mask=mask_d,
        )

        m_prev = mu_t * m_prev + (1.0 - mu_t) * e_t

    tl.store(
        m_final_ptr + pid_b * stride_mfb + d * stride_mfd,
        m_prev, mask=mask_d,
    )


def selective_iir_scan_generate_triton(
    e: torch.Tensor,
    mu_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    m0_batch: torch.Tensor,
) -> tuple:
    """Inference-only selective-IIR scan with per-batch initial memory.

    Args:
        e:         [B, L, D]  input (cuda, contiguous, bf16/fp16).
        mu_seq:    [B, L, 1]  per-timestep forget gates in (0, 1).
        alpha_seq: [B, L, 1]  per-timestep readout scales.
        m0_batch:  [B, D]     per-batch initial memory (float32).

    Returns:
        (out, m_final):
        - out:     [B, L, D] (same dtype as e)
        - m_final: [B, D] (float32) -- memory after last timestep.
    """
    assert e.is_cuda and e.is_contiguous() and e.ndim == 3
    B, L, D = e.shape
    assert m0_batch.shape == (B, D)

    mu_seq = mu_seq.contiguous()
    alpha_seq = alpha_seq.contiguous()

    out = torch.empty_like(e)
    m_final = torch.empty(B, D, dtype=torch.float32, device=e.device)

    OUT_DTYPE = _triton_dtype(e.dtype)
    BLOCK_D = 256 if D >= 256 else 128
    grid = (B, triton.cdiv(D, BLOCK_D))

    selective_iir_fwd_generate_kernel[grid](
        e, out,
        mu_seq, alpha_seq,
        m0_batch, m_final,
        B=B, L=L, D=D,
        stride_eb=e.stride(0), stride_el=e.stride(1), stride_ed=e.stride(2),
        stride_ob=out.stride(0), stride_ol=out.stride(1), stride_od=out.stride(2),
        stride_mub=mu_seq.stride(0), stride_mul=mu_seq.stride(1),
        stride_ab=alpha_seq.stride(0), stride_al=alpha_seq.stride(1),
        stride_m0b=m0_batch.stride(0), stride_m0d=m0_batch.stride(1),
        stride_mfb=m_final.stride(0), stride_mfd=m_final.stride(1),
        BLOCK_D=BLOCK_D,
        OUT_DTYPE=OUT_DTYPE,
        num_warps=4,
    )

    return out, m_final


# =========================================================================
# Pure-PyTorch reference scan (fallback / testing)
# =========================================================================

def selective_iir_scan_ref(
    e: torch.Tensor,
    mu_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    m0: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch sequential scan (autograd-compatible reference).

    Used as fallback when Triton is unavailable, and as the correctness
    reference in unit tests.
    """
    B, L, D = e.shape
    orig_dtype = e.dtype

    e_f = e.float()
    mu_f = mu_seq.float()
    alpha_f = alpha_seq.float()
    m = m0.float().unsqueeze(0).expand(B, -1).clone()

    outputs = torch.empty(B, L, D, dtype=torch.float32, device=e.device)

    for t in range(L):
        mu_t = mu_f[:, t]
        alpha_t = alpha_f[:, t]
        e_t = e_f[:, t]

        outputs[:, t] = e_t + alpha_t * m
        m = mu_t * m + (1.0 - mu_t) * e_t

    return outputs.to(orig_dtype)


# =========================================================================
# Unit tests
# =========================================================================

import unittest


class TestSelectiveIIRScanTriton(unittest.TestCase):
    """Compare Triton selective-IIR scan against the PyTorch reference."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    ATOL_FWD = 2e-2
    RTOL_FWD = 1e-2
    ATOL_BWD = 5e-2
    RTOL_BWD = 2e-2

    def _make_inputs(self, B, L, D, requires_grad=True):
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu_seq = torch.sigmoid(
            torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32)
        )
        alpha_seq = torch.randn(
            B, L, 1, device=self.DEVICE, dtype=torch.float32
        ) * 0.5
        m0 = torch.randn(D, device=self.DEVICE, dtype=torch.float32) * 0.1
        if requires_grad:
            for t in (e, mu_seq, alpha_seq, m0):
                t.requires_grad_(True)
        return e, mu_seq, alpha_seq, m0

    def _clone(self, *tensors):
        return tuple(
            t.detach().clone().requires_grad_(t.requires_grad) for t in tensors
        )

    def _run_forward_test(self, B, L, D):
        e, mu_seq, alpha_seq, m0 = self._make_inputs(B, L, D, requires_grad=False)
        out_triton = selective_iir_scan_triton(e, mu_seq, alpha_seq, m0)
        out_ref = selective_iir_scan_ref(e, mu_seq, alpha_seq, m0)
        torch.testing.assert_close(
            out_triton.float(), out_ref.float(),
            atol=self.ATOL_FWD, rtol=self.RTOL_FWD,
        )

    def test_forward_basic(self):
        self._run_forward_test(2, 64, 512)

    def test_forward_short_seq(self):
        self._run_forward_test(4, 1, 128)

    def test_forward_long_seq(self):
        self._run_forward_test(1, 2048, 256)

    def test_forward_large_dim(self):
        self._run_forward_test(2, 128, 2048)

    def _run_backward_test(self, B, L, D):
        e_t, mu_t, alpha_t, m0_t = self._make_inputs(B, L, D)
        e_r, mu_r, alpha_r, m0_r = self._clone(e_t, mu_t, alpha_t, m0_t)

        out_triton = selective_iir_scan_triton(e_t, mu_t, alpha_t, m0_t)
        out_ref = selective_iir_scan_ref(e_r, mu_r, alpha_r, m0_r)

        grad_out = torch.randn_like(out_triton)
        out_triton.backward(grad_out)
        out_ref.backward(grad_out.float())

        def _f32(t, like=None):
            if t is None:
                return torch.zeros_like(like, dtype=torch.float32)
            return t.float()

        for name, gt, gr, ref in [
            ("grad_e", e_t.grad, e_r.grad, e_t),
            ("grad_mu_seq", mu_t.grad, mu_r.grad, mu_t),
            ("grad_alpha_seq", alpha_t.grad, alpha_r.grad, alpha_t),
            ("grad_m0", m0_t.grad, m0_r.grad, m0_t),
        ]:
            torch.testing.assert_close(
                _f32(gt, like=ref), _f32(gr, like=ref),
                atol=self.ATOL_BWD, rtol=self.RTOL_BWD,
                msg=f"{name} mismatch",
            )

    def test_backward_basic(self):
        self._run_backward_test(2, 64, 512)

    def test_backward_short_seq(self):
        self._run_backward_test(4, 1, 128)

    def test_backward_long_seq(self):
        self._run_backward_test(1, 2048, 256)


class TestSelectiveIIRGenerateTriton(unittest.TestCase):
    """Test the inference-only generate kernel with per-batch m0."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    ATOL = 2e-2
    RTOL = 1e-2

    def _ref_scan(self, e, mu_seq, alpha_seq, m0_batch):
        B, L, D = e.shape
        m_prev = m0_batch.clone().float()
        out = torch.empty(B, L, D, dtype=torch.float32, device=e.device)
        for t in range(L):
            mu_t = mu_seq[:, t].float()
            alpha_t = alpha_seq[:, t].float()
            e_t = e[:, t].float()
            out[:, t] = e_t + alpha_t * m_prev
            m_prev = mu_t * m_prev + (1.0 - mu_t) * e_t
        return out, m_prev

    def test_single_batch(self):
        B, L, D = 1, 128, 256
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu_seq = torch.sigmoid(torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32))
        alpha_seq = torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32) * 0.5
        m0 = torch.randn(B, D, device=self.DEVICE, dtype=torch.float32) * 0.1

        out_t, mf_t = selective_iir_scan_generate_triton(e, mu_seq, alpha_seq, m0)
        out_r, mf_r = self._ref_scan(e, mu_seq, alpha_seq, m0)
        torch.testing.assert_close(out_t.float(), out_r, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(mf_t, mf_r, atol=self.ATOL, rtol=self.RTOL)

    def test_multi_batch(self):
        B, L, D = 4, 64, 512
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu_seq = torch.sigmoid(torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32))
        alpha_seq = torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32) * 0.5
        m0 = torch.randn(B, D, device=self.DEVICE, dtype=torch.float32) * 0.1

        out_t, mf_t = selective_iir_scan_generate_triton(e, mu_seq, alpha_seq, m0)
        out_r, mf_r = self._ref_scan(e, mu_seq, alpha_seq, m0)
        torch.testing.assert_close(out_t.float(), out_r, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(mf_t, mf_r, atol=self.ATOL, rtol=self.RTOL)

    def test_consistency_with_training_kernel(self):
        """Generate kernel should match training kernel when m0 is shared."""
        B, L, D = 2, 64, 256
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu_seq = torch.sigmoid(torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32))
        alpha_seq = torch.randn(B, L, 1, device=self.DEVICE, dtype=torch.float32) * 0.5
        m0_shared = torch.randn(D, device=self.DEVICE, dtype=torch.float32) * 0.1
        m0_batch = m0_shared.unsqueeze(0).expand(B, D).contiguous()

        out_train = selective_iir_scan_triton(e, mu_seq, alpha_seq, m0_shared)
        out_gen, _ = selective_iir_scan_generate_triton(e, mu_seq, alpha_seq, m0_batch)
        torch.testing.assert_close(out_gen, out_train, atol=1e-5, rtol=1e-5)


# # Import here to avoid circular import (lingua.stem_sss imports lingua.sss)
# from lingua.stem_sss import SSSMemory


class TestSSSMemoryModule(unittest.TestCase):
    """Integration test: SSSMemory module end-to-end."""

    DEVICE = "cuda"
    DIM = 128
    D_FF = 256
    SEQ_LEN = 32
    BATCH = 2

    def test_forward_backward(self):
        from lingua.stem_sss import SSSMemory
        mem = SSSMemory(
            dim=self.DIM, d_ff=self.D_FF, mu_init=0.9, alpha_init=0.5,
        ).to(self.DEVICE)

        e = torch.randn(
            self.BATCH, self.SEQ_LEN, self.D_FF,
            device=self.DEVICE, dtype=torch.bfloat16, requires_grad=True,
        )
        x = torch.randn(
            self.BATCH, self.SEQ_LEN, self.DIM,
            device=self.DEVICE, dtype=torch.bfloat16, requires_grad=True,
        )

        out = mem(e, x)
        self.assertEqual(out.shape, e.shape)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(e.grad)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(mem.w_mu.weight.grad)
        self.assertIsNotNone(mem.w_alpha.weight.grad)

    def test_init_matches_fixed_iir(self):
        """At init (zero weights), module should match fixed-IIR scan."""
        MU_INIT = 0.85
        ALPHA_INIT = 0.7

        mem = SSSMemory(
            dim=self.DIM, d_ff=self.D_FF,
            mu_init=MU_INIT, alpha_init=ALPHA_INIT,
        ).to(self.DEVICE)

        e = torch.randn(
            self.BATCH, self.SEQ_LEN, self.D_FF,
            device=self.DEVICE, dtype=torch.bfloat16,
        )
        x = torch.randn(
            self.BATCH, self.SEQ_LEN, self.DIM,
            device=self.DEVICE, dtype=torch.bfloat16,
        )

        with torch.no_grad():
            out_selective = mem(e, x)

        m0 = torch.zeros(self.D_FF, device=self.DEVICE, dtype=torch.float32)
        mu_fixed = torch.full(
            (self.BATCH, self.SEQ_LEN, 1), MU_INIT,
            device=self.DEVICE, dtype=torch.float32,
        )
        alpha_fixed = torch.full(
            (self.BATCH, self.SEQ_LEN, 1), ALPHA_INIT,
            device=self.DEVICE, dtype=torch.float32,
        )
        out_fixed = selective_iir_scan_ref(e, mu_fixed, alpha_fixed, m0)

        torch.testing.assert_close(
            out_selective.float(), out_fixed.float(), atol=2e-2, rtol=1e-2,
        )


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        sys.argv.remove("--bench")
        # Inline benchmark
        import time

        configs = [
            ("Small",  4,  128,  512),
            ("Medium", 4,  512, 1024),
            ("Large",  4, 2048, 2048),
            ("XL",     8, 4096, 2048),
        ]
        WARMUP, REPEATS = 10, 50
        print(f"{'Config':<10} {'Shape':>20} {'PyTorch fwd':>14} {'Triton fwd':>14} {'Speedup':>8}"
              f" {'PyTorch bwd':>14} {'Triton bwd':>14} {'Speedup':>8}")
        print("-" * 120)
        for name, B, L, D in configs:
            e = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
            mu_seq = torch.sigmoid(torch.randn(B, L, 1, device="cuda", dtype=torch.float32))
            alpha_seq = torch.randn(B, L, 1, device="cuda", dtype=torch.float32) * 0.5
            m0 = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1
            grad_out = torch.randn_like(e)

            for _ in range(WARMUP):
                selective_iir_scan_ref(e, mu_seq, alpha_seq, m0)
                selective_iir_scan_triton(e, mu_seq, alpha_seq, m0)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                selective_iir_scan_ref(e, mu_seq, alpha_seq, m0)
                torch.cuda.synchronize()
            t_ref_fwd = (time.perf_counter() - t0) / REPEATS * 1000

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                selective_iir_scan_triton(e, mu_seq, alpha_seq, m0)
                torch.cuda.synchronize()
            t_tri_fwd = (time.perf_counter() - t0) / REPEATS * 1000

            for _ in range(WARMUP):
                e_r = e.float().requires_grad_(True)
                out_r = selective_iir_scan_ref(e_r, mu_seq.clone().requires_grad_(True), alpha_seq.clone().requires_grad_(True), m0.clone().requires_grad_(True))
                out_r.backward(grad_out.float())
                e_t = e.clone().requires_grad_(True)
                out_t = selective_iir_scan_triton(e_t, mu_seq.clone().requires_grad_(True), alpha_seq.clone().requires_grad_(True), m0.clone().requires_grad_(True))
                out_t.backward(grad_out)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                e_r = e.float().requires_grad_(True)
                out_r = selective_iir_scan_ref(e_r, mu_seq.clone().requires_grad_(True), alpha_seq.clone().requires_grad_(True), m0.clone().requires_grad_(True))
                out_r.backward(grad_out.float())
                torch.cuda.synchronize()
            t_ref_bwd = (time.perf_counter() - t0) / REPEATS * 1000

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                e_t = e.clone().requires_grad_(True)
                out_t = selective_iir_scan_triton(e_t, mu_seq.clone().requires_grad_(True), alpha_seq.clone().requires_grad_(True), m0.clone().requires_grad_(True))
                out_t.backward(grad_out)
                torch.cuda.synchronize()
            t_tri_bwd = (time.perf_counter() - t0) / REPEATS * 1000

            shape_str = f"({B},{L},{D})"
            print(f"{name:<10} {shape_str:>20} {t_ref_fwd:>11.3f} ms {t_tri_fwd:>11.3f} ms {t_ref_fwd/t_tri_fwd:>7.2f}x"
                  f" {t_ref_bwd:>11.3f} ms {t_tri_bwd:>11.3f} ms {t_ref_bwd/t_tri_bwd:>7.2f}x")
    else:
        unittest.main()