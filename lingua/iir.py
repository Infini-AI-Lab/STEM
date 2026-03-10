import torch
import triton
import triton.language as tl


_TORCH_TO_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _triton_dtype(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL_DTYPE:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return _TORCH_TO_TL_DTYPE[dtype]


# -------------------------
# Forward kernel
# -------------------------
@triton.jit
def iir_fwd_kernel(
    e_ptr, out_ptr, ctx_ptr,
    mu_ptr, alpha_ptr, m0_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    stride_eb: tl.constexpr, stride_el: tl.constexpr, stride_ed: tl.constexpr,
    stride_ob: tl.constexpr, stride_ol: tl.constexpr, stride_od: tl.constexpr,
    stride_cb: tl.constexpr, stride_cl: tl.constexpr, stride_cd: tl.constexpr,
    ALPHA_IS_VECTOR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,   # tl.float16 or tl.bfloat16
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d < D

    mu = tl.load(mu_ptr).to(tl.float32)
    one_minus_mu = 1.0 - mu

    if ALPHA_IS_VECTOR:
        alpha = tl.load(alpha_ptr + d, mask=mask_d, other=0.0).to(tl.float32)  # [BLOCK_D]
    else:
        alpha = tl.load(alpha_ptr).to(tl.float32)  # scalar

    # initial memory m_{-1} = m0, broadcast across batch
    m_prev = tl.load(m0_ptr + d, mask=mask_d, other=0.0).to(tl.float32)

    e_b = e_ptr + pid_b * stride_eb
    o_b = out_ptr + pid_b * stride_ob
    c_b = ctx_ptr + pid_b * stride_cb

    for t in range(0, L):
        e_t = tl.load(e_b + t * stride_el + d * stride_ed, mask=mask_d, other=0.0).to(tl.float32)

        # ctx[t] = m_{t-1}
        tl.store(c_b + t * stride_cl + d * stride_cd, m_prev.to(OUT_DTYPE), mask=mask_d)

        # out[t] = e_t + alpha * ctx[t]
        if ALPHA_IS_VECTOR:
            out_t = e_t + alpha * m_prev
        else:
            out_t = e_t + alpha * m_prev
        tl.store(o_b + t * stride_ol + d * stride_od, out_t.to(OUT_DTYPE), mask=mask_d)

        # update memory: m_t = mu*m_{t-1} + (1-mu)*e_t
        m_prev = mu * m_prev + one_minus_mu * e_t


# -------------------------
# Backward kernel
# -------------------------
@triton.jit
def iir_bwd_kernel(
    e_ptr, ctx_ptr, gy_ptr,
    ge_ptr,
    mu_ptr, alpha_ptr,
    gmu_ptr, galpha_ptr, gm0_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    stride_eb: tl.constexpr, stride_el: tl.constexpr, stride_ed: tl.constexpr,
    stride_cb: tl.constexpr, stride_cl: tl.constexpr, stride_cd: tl.constexpr,
    stride_gyb: tl.constexpr, stride_gyl: tl.constexpr, stride_gyd: tl.constexpr,
    stride_geb: tl.constexpr, stride_gel: tl.constexpr, stride_ged: tl.constexpr,
    ALPHA_IS_VECTOR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GE_DTYPE: tl.constexpr,  # tl.float16 or tl.bfloat16
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d < D

    mu = tl.load(mu_ptr).to(tl.float32)
    one_minus_mu = 1.0 - mu

    if ALPHA_IS_VECTOR:
        alpha = tl.load(alpha_ptr + d, mask=mask_d, other=0.0).to(tl.float32)  # [BLOCK_D]
    else:
        alpha = tl.load(alpha_ptr).to(tl.float32)  # scalar

    e_b   = e_ptr   + pid_b * stride_eb
    c_b   = ctx_ptr + pid_b * stride_cb
    gy_b  = gy_ptr  + pid_b * stride_gyb
    ge_b  = ge_ptr  + pid_b * stride_geb

    # adjoint for memory state: g^m_t. Start at g^m_{L-1} = 0
    gm_next = tl.zeros([BLOCK_D], dtype=tl.float32)

    # partial reductions (per program)
    gmu_acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # we'll reduce across D lanes
    if ALPHA_IS_VECTOR:
        galpha_acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # per-d component
    else:
        galpha_acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # we'll reduce across D lanes

    # reverse scan over time
    for t in range(L - 1, -1, -1):
        e_t  = tl.load(e_b  + t * stride_el  + d * stride_ed,  mask=mask_d, other=0.0).to(tl.float32)
        mprv = tl.load(c_b  + t * stride_cl  + d * stride_cd,  mask=mask_d, other=0.0).to(tl.float32)  # m_{t-1}
        gy_t = tl.load(gy_b + t * stride_gyl + d * stride_gyd, mask=mask_d, other=0.0).to(tl.float32)

        # ge_t starts with direct path from y_t = e_t + alpha*m_{t-1}
        ge_t = gy_t

        # contribution to g_alpha from y_t
        if ALPHA_IS_VECTOR:
            galpha_acc += gy_t * mprv
            gm_prev_from_y = gy_t * alpha
        else:
            galpha_acc += gy_t * mprv
            gm_prev_from_y = gy_t * alpha

        # from EMA update m_t = mu*m_{t-1} + (1-mu)*e_t
        # ge_t += (1-mu) * g^m_t
        ge_t += one_minus_mu * gm_next

        # g_mu += <g^m_t, m_{t-1} - e_t>
        gmu_acc += gm_next * (mprv - e_t)

        # g^m_{t-1} accumulates:
        #   from y_t: alpha*gy_t
        #   from recurrence: mu*gm_next
        gm_next = gm_prev_from_y + mu * gm_next

        # store grad_e
        tl.store(ge_b + t * stride_gel + d * stride_ged, ge_t.to(GE_DTYPE), mask=mask_d)

    # After loop, gm_next is g^m_{-1} for this (batch, d-block).
    # m0 was broadcast across batch, so gm0[d] accumulates sum over B.
    tl.atomic_add(gm0_ptr + d, gm_next, mask=mask_d)

    # Reduce across the BLOCK_D lanes and atomically accumulate scalars.
    # tl.sum yields a scalar; each program instance writes once.
    gmu_scalar = tl.sum(gmu_acc, axis=0)
    tl.atomic_add(gmu_ptr, gmu_scalar)

    if ALPHA_IS_VECTOR:
        tl.atomic_add(galpha_ptr + d, galpha_acc, mask=mask_d)
    else:
        galpha_scalar = tl.sum(galpha_acc, axis=0)
        tl.atomic_add(galpha_ptr, galpha_scalar)


# -------------------------
# Autograd wrapper
# -------------------------
class IIRScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, mu, alpha, m0):
        """
        e: [B,L,D] (cuda, contiguous)
        mu: scalar cuda tensor
        alpha: scalar cuda tensor OR [D] cuda tensor
        m0: [D] cuda tensor
        """
        assert e.is_cuda and e.is_contiguous()
        assert e.ndim == 3
        B, L, D = e.shape
        assert mu.is_cuda and mu.numel() == 1
        assert m0.is_cuda and m0.ndim == 1 and m0.numel() == D
        assert alpha.is_cuda and (alpha.numel() == 1 or alpha.numel() == D)

        alpha_is_vector = (alpha.numel() == D)

        out = torch.empty_like(e)
        ctx_mem = torch.empty_like(e)  # save ctx for backward: m_{t-1}

        OUT_DTYPE = _triton_dtype(e.dtype)

        BLOCK_D = 256 if D >= 256 else 128
        grid = (B, triton.cdiv(D, BLOCK_D))

        iir_fwd_kernel[grid](
            e, out, ctx_mem,
            mu, alpha, m0,
            B=B, L=L, D=D,
            stride_eb=e.stride(0), stride_el=e.stride(1), stride_ed=e.stride(2),
            stride_ob=out.stride(0), stride_ol=out.stride(1), stride_od=out.stride(2),
            stride_cb=ctx_mem.stride(0), stride_cl=ctx_mem.stride(1), stride_cd=ctx_mem.stride(2),
            ALPHA_IS_VECTOR=alpha_is_vector,
            BLOCK_D=BLOCK_D,
            OUT_DTYPE=OUT_DTYPE,
            num_warps=4,
        )

        # Save for backward
        ctx.save_for_backward(e, ctx_mem, mu, alpha, m0)
        ctx.alpha_is_vector = alpha_is_vector
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, grad_out):
        e, ctx_mem, mu, alpha, m0 = ctx.saved_tensors
        assert grad_out.is_cuda and grad_out.is_contiguous()

        B, L, D = e.shape
        alpha_is_vector = ctx.alpha_is_vector
        BLOCK_D = ctx.BLOCK_D

        ge = torch.empty_like(e)

        # grads for scalars / vectors
        gmu = torch.zeros_like(mu)  # scalar
        if alpha_is_vector:
            galpha = torch.zeros_like(alpha)  # [D]
        else:
            galpha = torch.zeros_like(alpha)  # scalar
        gm0 = torch.zeros_like(m0)  # [D]

        GE_DTYPE = _triton_dtype(ge.dtype)

        grid = (B, triton.cdiv(D, BLOCK_D))

        iir_bwd_kernel[grid](
            e, ctx_mem, grad_out,
            ge,
            mu, alpha,
            gmu, galpha, gm0,
            B=B, L=L, D=D,
            stride_eb=e.stride(0), stride_el=e.stride(1), stride_ed=e.stride(2),
            stride_cb=ctx_mem.stride(0), stride_cl=ctx_mem.stride(1), stride_cd=ctx_mem.stride(2),
            stride_gyb=grad_out.stride(0), stride_gyl=grad_out.stride(1), stride_gyd=grad_out.stride(2),
            stride_geb=ge.stride(0), stride_gel=ge.stride(1), stride_ged=ge.stride(2),
            ALPHA_IS_VECTOR=alpha_is_vector,
            BLOCK_D=BLOCK_D,
            GE_DTYPE=GE_DTYPE,
            num_warps=4,
        )

        # Return grads aligned to forward inputs: (e, mu, alpha, m0)
        return ge, gmu, galpha, gm0


def iir_scan_triton(e: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
    return IIRScanFn.apply(e, mu, alpha, m0)


# -------------------------
# Inference-only forward kernel (per-batch m0, outputs final m_prev)
# -------------------------
@triton.jit
def iir_fwd_generate_kernel(
    e_ptr, out_ptr,
    mu_ptr, alpha_ptr, m0_ptr, m_final_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    stride_eb: tl.constexpr, stride_el: tl.constexpr, stride_ed: tl.constexpr,
    stride_ob: tl.constexpr, stride_ol: tl.constexpr, stride_od: tl.constexpr,
    stride_m0b: tl.constexpr, stride_m0d: tl.constexpr,
    stride_mfb: tl.constexpr, stride_mfd: tl.constexpr,
    ALPHA_IS_VECTOR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d < D

    mu = tl.load(mu_ptr).to(tl.float32)
    one_minus_mu = 1.0 - mu

    if ALPHA_IS_VECTOR:
        alpha = tl.load(alpha_ptr + d, mask=mask_d, other=0.0).to(tl.float32)
    else:
        alpha = tl.load(alpha_ptr).to(tl.float32)

    # Per-batch initial memory
    m_prev = tl.load(m0_ptr + pid_b * stride_m0b + d * stride_m0d,
                     mask=mask_d, other=0.0).to(tl.float32)

    e_b = e_ptr + pid_b * stride_eb
    o_b = out_ptr + pid_b * stride_ob

    for t in range(0, L):
        e_t = tl.load(e_b + t * stride_el + d * stride_ed,
                      mask=mask_d, other=0.0).to(tl.float32)

        out_t = e_t + alpha * m_prev
        tl.store(o_b + t * stride_ol + d * stride_od,
                 out_t.to(OUT_DTYPE), mask=mask_d)

        m_prev = mu * m_prev + one_minus_mu * e_t

    # Store final memory state per batch element
    tl.store(m_final_ptr + pid_b * stride_mfb + d * stride_mfd,
             m_prev, mask=mask_d)


def iir_scan_generate_triton(
    e: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    m0_batch: torch.Tensor,
) -> tuple:
    """Inference-only IIR scan with per-batch initial memory.

    Args:
        e:        ``[B, L, D]`` input (cuda, contiguous, fp16/bf16).
        mu:       scalar tensor in (0, 1).
        alpha:    scalar or ``[D]`` tensor.
        m0_batch: ``[B, D]`` per-batch initial memory (float32).

    Returns:
        ``(out, m_final)`` where ``out`` is ``[B, L, D]`` (same dtype as e)
        and ``m_final`` is ``[B, D]`` (float32) — the memory state after the
        last time step of each batch element.
    """
    assert e.is_cuda and e.is_contiguous() and e.ndim == 3
    B, L, D = e.shape
    assert m0_batch.shape == (B, D)

    alpha_is_vector = (alpha.numel() == D)

    out = torch.empty_like(e)
    m_final = torch.empty(B, D, dtype=torch.float32, device=e.device)

    OUT_DTYPE = _triton_dtype(e.dtype)

    BLOCK_D = 256 if D >= 256 else 128
    grid = (B, triton.cdiv(D, BLOCK_D))

    iir_fwd_generate_kernel[grid](
        e, out,
        mu, alpha, m0_batch, m_final,
        B=B, L=L, D=D,
        stride_eb=e.stride(0), stride_el=e.stride(1), stride_ed=e.stride(2),
        stride_ob=out.stride(0), stride_ol=out.stride(1), stride_od=out.stride(2),
        stride_m0b=m0_batch.stride(0), stride_m0d=m0_batch.stride(1),
        stride_mfb=m_final.stride(0), stride_mfd=m_final.stride(1),
        ALPHA_IS_VECTOR=alpha_is_vector,
        BLOCK_D=BLOCK_D,
        OUT_DTYPE=OUT_DTYPE,
        num_warps=4,
    )

    return out, m_final


# -------------------------
# Unit tests
# -------------------------
import unittest


def _iir_scan_ref(e, mu, alpha, m0):
    """Pure-PyTorch reference (mirrors lingua.stem_iir.iir_scan)."""
    B, L, D = e.shape
    m_prev = m0.unsqueeze(0).expand(B, D).clone()
    ctx = torch.zeros_like(e)
    one_minus_mu = 1.0 - mu
    for t in range(L):
        ctx[:, t, :] = m_prev
        m_prev = mu * m_prev + one_minus_mu * e[:, t, :]
    return e + alpha * ctx


class TestIIRScanTriton(unittest.TestCase):
    """Compare Triton IIR scan against the pure-PyTorch reference."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    ATOL_FWD = 2e-2
    RTOL_FWD = 1e-2
    ATOL_BWD = 5e-2
    RTOL_BWD = 2e-2

    # ---------- helpers ----------

    def _make_inputs(self, B, L, D, per_dim_alpha=False, requires_grad=True):
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu = torch.tensor(0.85, device=self.DEVICE, dtype=torch.float32)
        if per_dim_alpha:
            alpha = torch.randn(D, device=self.DEVICE, dtype=torch.float32) * 0.5
        else:
            alpha = torch.tensor(1.0, device=self.DEVICE, dtype=torch.float32)
        m0 = torch.randn(D, device=self.DEVICE, dtype=torch.float32) * 0.1

        if requires_grad:
            for t in (e, mu, alpha, m0):
                t.requires_grad_(True)
        return e, mu, alpha, m0

    def _clone_detach(self, *tensors):
        return tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in tensors)

    # ---------- forward ----------

    def test_forward_scalar_alpha(self):
        e, mu, alpha, m0 = self._make_inputs(2, 64, 512, per_dim_alpha=False, requires_grad=False)
        out_triton = iir_scan_triton(e, mu, alpha, m0)
        out_ref = _iir_scan_ref(e.float(), mu, alpha, m0)
        torch.testing.assert_close(out_triton.float(), out_ref, atol=self.ATOL_FWD, rtol=self.RTOL_FWD)

    def test_forward_vector_alpha(self):
        e, mu, alpha, m0 = self._make_inputs(2, 64, 512, per_dim_alpha=True, requires_grad=False)
        out_triton = iir_scan_triton(e, mu, alpha, m0)
        out_ref = _iir_scan_ref(e.float(), mu, alpha, m0)
        torch.testing.assert_close(out_triton.float(), out_ref, atol=self.ATOL_FWD, rtol=self.RTOL_FWD)

    def test_forward_short_seq(self):
        e, mu, alpha, m0 = self._make_inputs(4, 1, 128, per_dim_alpha=False, requires_grad=False)
        out_triton = iir_scan_triton(e, mu, alpha, m0)
        out_ref = _iir_scan_ref(e.float(), mu, alpha, m0)
        torch.testing.assert_close(out_triton.float(), out_ref, atol=self.ATOL_FWD, rtol=self.RTOL_FWD)

    def test_forward_long_seq(self):
        e, mu, alpha, m0 = self._make_inputs(1, 2048, 256, per_dim_alpha=True, requires_grad=False)
        out_triton = iir_scan_triton(e, mu, alpha, m0)
        out_ref = _iir_scan_ref(e.float(), mu, alpha, m0)
        torch.testing.assert_close(out_triton.float(), out_ref, atol=self.ATOL_FWD, rtol=self.RTOL_FWD)

    # ---------- backward ----------

    def _run_backward_test(self, B, L, D, per_dim_alpha):
        e_t, mu_t, alpha_t, m0_t = self._make_inputs(B, L, D, per_dim_alpha=per_dim_alpha)
        e_r, mu_r, alpha_r, m0_r = self._clone_detach(e_t, mu_t, alpha_t, m0_t)

        out_triton = iir_scan_triton(e_t, mu_t, alpha_t, m0_t)
        out_ref = _iir_scan_ref(e_r.float(), mu_r, alpha_r, m0_r)

        grad_out = torch.randn_like(out_triton)
        out_triton.backward(grad_out)
        out_ref.backward(grad_out.float())

        def _f32(t, like=None):
            if t is None:
                return torch.zeros_like(like, dtype=torch.float32)
            return t.float()

        for name, gt, gr, ref_shape in [
            ("grad_e", e_t.grad, e_r.grad, e_t),
            ("grad_mu", mu_t.grad, mu_r.grad, mu_t),
            ("grad_alpha", alpha_t.grad, alpha_r.grad, alpha_t),
            ("grad_m0", m0_t.grad, m0_r.grad, m0_t),
        ]:
            torch.testing.assert_close(
                _f32(gt, like=ref_shape), _f32(gr, like=ref_shape),
                atol=self.ATOL_BWD, rtol=self.RTOL_BWD, msg=f"{name} mismatch",
            )

    def test_backward_scalar_alpha(self):
        self._run_backward_test(2, 64, 512, per_dim_alpha=False)

    def test_backward_vector_alpha(self):
        self._run_backward_test(2, 64, 512, per_dim_alpha=True)

    def test_backward_short_seq(self):
        self._run_backward_test(4, 1, 128, per_dim_alpha=False)

    def test_backward_long_seq(self):
        self._run_backward_test(1, 2048, 256, per_dim_alpha=True)


class TestIIRScanGenerateTriton(unittest.TestCase):
    """Test the inference-only generate kernel with per-batch m0."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    ATOL = 2e-2
    RTOL = 1e-2

    def _ref_scan(self, e, mu, alpha, m0_batch):
        """Pure-PyTorch reference with per-batch m0 (no grad needed)."""
        B, L, D = e.shape
        m_prev = m0_batch.clone().float()
        out = torch.empty(B, L, D, dtype=torch.float32, device=e.device)
        mu_f = mu.float().item()
        one_minus_mu = 1.0 - mu_f
        alpha_f = alpha.float()
        for t in range(L):
            e_t = e[:, t, :].float()
            out[:, t, :] = e_t + alpha_f * m_prev
            m_prev = mu_f * m_prev + one_minus_mu * e_t
        return out, m_prev

    def test_single_batch(self):
        B, L, D = 1, 128, 256
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu = torch.tensor([0.9], device=self.DEVICE, dtype=torch.float32)
        alpha = torch.tensor([1.0], device=self.DEVICE, dtype=torch.float32)
        m0 = torch.randn(B, D, device=self.DEVICE, dtype=torch.float32) * 0.1

        out_t, mf_t = iir_scan_generate_triton(e, mu, alpha, m0)
        out_r, mf_r = self._ref_scan(e, mu, alpha, m0)
        torch.testing.assert_close(out_t.float(), out_r, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(mf_t, mf_r, atol=self.ATOL, rtol=self.RTOL)

    def test_multi_batch_different_m0(self):
        B, L, D = 4, 64, 512
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu = torch.tensor([0.85], device=self.DEVICE, dtype=torch.float32)
        alpha = torch.randn(D, device=self.DEVICE, dtype=torch.float32) * 0.5
        m0 = torch.randn(B, D, device=self.DEVICE, dtype=torch.float32) * 0.1

        out_t, mf_t = iir_scan_generate_triton(e, mu, alpha, m0)
        out_r, mf_r = self._ref_scan(e, mu, alpha, m0)
        torch.testing.assert_close(out_t.float(), out_r, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(mf_t, mf_r, atol=self.ATOL, rtol=self.RTOL)

    def test_single_step(self):
        B, L, D = 8, 1, 256
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu = torch.tensor([0.95], device=self.DEVICE, dtype=torch.float32)
        alpha = torch.tensor([1.0], device=self.DEVICE, dtype=torch.float32)
        m0 = torch.randn(B, D, device=self.DEVICE, dtype=torch.float32) * 0.1

        out_t, mf_t = iir_scan_generate_triton(e, mu, alpha, m0)
        out_r, mf_r = self._ref_scan(e, mu, alpha, m0)
        torch.testing.assert_close(out_t.float(), out_r, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(mf_t, mf_r, atol=self.ATOL, rtol=self.RTOL)

    def test_consistency_with_training_kernel(self):
        """When m0 is the same for all batches, generate kernel should match training kernel."""
        B, L, D = 2, 64, 256
        e = torch.randn(B, L, D, device=self.DEVICE, dtype=self.DTYPE)
        mu = torch.tensor([0.9], device=self.DEVICE, dtype=torch.float32)
        alpha = torch.tensor([1.0], device=self.DEVICE, dtype=torch.float32)
        m0_shared = torch.randn(D, device=self.DEVICE, dtype=torch.float32) * 0.1
        m0_batch = m0_shared.unsqueeze(0).expand(B, D).contiguous()

        out_train = iir_scan_triton(e, mu, alpha, m0_shared)
        out_gen, _ = iir_scan_generate_triton(e, mu, alpha, m0_batch)
        torch.testing.assert_close(out_gen, out_train, atol=1e-5, rtol=1e-5)


def benchmark():
    """Benchmark Triton vs PyTorch reference for forward and backward."""
    import time

    configs = [
        ("Small",  4,  128,  512, False),
        ("Medium", 4,  512, 1024, False),
        ("Large",  4, 2048, 2048, True),
        ("XL",     8, 4096, 2048, True),
    ]

    WARMUP = 10
    REPEATS = 50

    print(f"{'Config':<10} {'Shape':>20} {'PyTorch fwd':>14} {'Triton fwd':>14} {'Speedup':>8}"
          f" {'PyTorch bwd':>14} {'Triton bwd':>14} {'Speedup':>8}")
    print("-" * 120)

    for name, B, L, D, per_dim_alpha in configs:
        e = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
        mu = torch.tensor(0.9, device="cuda", dtype=torch.float32)
        alpha = (torch.randn(D, device="cuda", dtype=torch.float32) * 0.5
                 if per_dim_alpha else
                 torch.tensor(1.0, device="cuda", dtype=torch.float32))
        m0 = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1
        grad_out = torch.randn_like(e)

        # --- Forward benchmark ---
        for _ in range(WARMUP):
            _iir_scan_ref(e.float(), mu, alpha, m0)
            iir_scan_triton(e, mu, alpha, m0)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            _iir_scan_ref(e.float(), mu, alpha, m0)
            torch.cuda.synchronize()
        t_ref_fwd = (time.perf_counter() - t0) / REPEATS * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            iir_scan_triton(e, mu, alpha, m0)
            torch.cuda.synchronize()
        t_tri_fwd = (time.perf_counter() - t0) / REPEATS * 1000

        # --- Backward benchmark ---
        for _ in range(WARMUP):
            e_r = e.float().requires_grad_(True)
            out_r = _iir_scan_ref(e_r, mu.clone().requires_grad_(True), alpha.clone().requires_grad_(True), m0.clone().requires_grad_(True))
            out_r.backward(grad_out.float())
            e_t = e.clone().requires_grad_(True)
            out_t = iir_scan_triton(e_t, mu.clone().requires_grad_(True), alpha.clone().requires_grad_(True), m0.clone().requires_grad_(True))
            out_t.backward(grad_out)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            e_r = e.float().requires_grad_(True)
            out_r = _iir_scan_ref(e_r, mu.clone().requires_grad_(True), alpha.clone().requires_grad_(True), m0.clone().requires_grad_(True))
            out_r.backward(grad_out.float())
            torch.cuda.synchronize()
        t_ref_bwd = (time.perf_counter() - t0) / REPEATS * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            e_t = e.clone().requires_grad_(True)
            out_t = iir_scan_triton(e_t, mu.clone().requires_grad_(True), alpha.clone().requires_grad_(True), m0.clone().requires_grad_(True))
            out_t.backward(grad_out)
            torch.cuda.synchronize()
        t_tri_bwd = (time.perf_counter() - t0) / REPEATS * 1000

        shape_str = f"({B},{L},{D})"
        fwd_speedup = t_ref_fwd / t_tri_fwd
        bwd_speedup = t_ref_bwd / t_tri_bwd
        print(f"{name:<10} {shape_str:>20} {t_ref_fwd:>11.3f} ms {t_tri_fwd:>11.3f} ms {fwd_speedup:>7.2f}x"
              f" {t_ref_bwd:>11.3f} ms {t_tri_bwd:>11.3f} ms {bwd_speedup:>7.2f}x")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        sys.argv.remove("--bench")
        benchmark()
    else:
        unittest.main()
