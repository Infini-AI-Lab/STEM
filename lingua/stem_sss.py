# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
SSS (Selective State Space) memory extension for STEM embeddings.

Replaces the fixed-parameter IIR/EMA (:mod:`lingua.stem_iir`) with
**input-conditioned scalar gates** inspired by Mamba2's Structured State
Space Duality (SSD).  Instead of a single learnable scalar ``mu`` and
``alpha`` shared across all timesteps, the forget gate ``mu_t`` and
readout scale ``alpha_t`` are computed from the transformer's hidden
state ``x_t`` at each position:

    mu_t       = sigma(w_mu^T x_t + b_mu)          (selective forget)
    alpha_t    = w_alpha^T x_t + b_alpha            (selective readout)
    m_t        = mu_t * m_{t-1} + (1 - mu_t) * e_t (memory update)
    ctx_t      = m_{t-1}                            (causal: previous memory)
    e_tilde_t  = e_t + alpha_t * ctx_t              (contextual output)

Both ``mu_t`` and ``alpha_t`` are **scalars** (not per-dimension vectors).
The gate projections ``w_mu, w_alpha`` are ``nn.Linear(dim, 1)`` modules
whose weights are initialized to zero and biases to ``logit(mu_init)``
and ``alpha_init`` respectively, so that the model starts behaving
identically to fixed-parameter IIR-STEM and learns to deviate during
training.

Relationship to :mod:`lingua.stem_iir`
---------------------------------------
This module is a strict generalisation.  Every component has a 1-to-1
correspondence:

    stem_iir                      stem_sss (this file)
    ─────────────────────         ──────────────────────────────
    IIRStemTransformerArgs   →    SSSStemTransformerArgs
    IIRCache                 →    SSSCache
    IIRMemory                →    SSSMemory
    iir_scan(e,mu,alpha,m0)  →    sss_scan(e,x,w_mu,w_alpha,m0)
    _iir_scan_generate(...)  →    _sss_scan_generate(...)

The critical API difference: every scan function and the ``SSSMemory``
forward method take the transformer hidden state ``x`` (shape
``[B, L, dim]``) as an additional argument, since the gates are
input-conditioned.

This module lives at the library level and does **not** import from
``apps.main``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from lingua.stem import StemTransformerArgs
from lingua.sss import (
    selective_iir_scan_triton,
    selective_iir_scan_generate_triton,
    _inverse_sigmoid,
)


# ---------------------------------------------------------------------------
# Generation-mode SSS cache
# ---------------------------------------------------------------------------

class SSSCache:
    """Holds per-document SSS memory state across autoregressive decode steps.

    Analogous to :class:`~lingua.stem_iir.IIRCache` for fixed-parameter IIR.
    Attached to a :class:`SSSMemory` module by the generator before inference
    begins.

    Attributes
    ----------
    m_prev : Tensor [n_seqs, d_ff]
        Running memory state per document (updated in-place during decode).
    doc_lengths : Optional[Tensor]
        Number of tokens per document in the packed sequence.  Set by the
        generator at each step.
    """

    def __init__(
        self,
        n_seqs: int,
        d_ff: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.m_prev = torch.zeros(n_seqs, d_ff, dtype=dtype, device=device)
        self.doc_lengths: Optional[torch.Tensor] = None

    def reset(self):
        self.m_prev.zero_()
        self.doc_lengths = None


def _sss_scan_generate(
    e: torch.Tensor,
    x: torch.Tensor,
    w_mu: nn.Linear,
    w_alpha: nn.Linear,
    cache: SSSCache,
) -> torch.Tensor:
    """Selective-IIR scan for generation with per-document cached memory.

    The packed sequence ``e`` of shape ``[1, total_len, D]`` contains
    ``n_seqs`` documents whose boundaries are given by
    ``cache.doc_lengths``.  Each document's scan starts from
    ``cache.m_prev[doc_idx]`` and the final memory state is written back.

    Decode (L=1 per document) uses a fully vectorized PyTorch fast path.
    Prefill dispatches each document to the Triton generate kernel.

    Args:
        e:       ``[1, total_len, d_ff]`` raw STEM embeddings.
        x:       ``[1, total_len, dim]``  transformer hidden states.
        w_mu:    ``nn.Linear(dim, 1)``    forget-gate projection.
        w_alpha: ``nn.Linear(dim, 1)``    readout-gate projection.
        cache:   :class:`SSSCache` with ``m_prev`` and ``doc_lengths``.

    Returns:
        ``[1, total_len, d_ff]`` contextual STEM embeddings.
    """
    _, total_len, D = e.shape
    lengths = cache.doc_lengths
    n_seqs = lengths.size(0)
    out = torch.empty_like(e)

    # Compute gates for the full packed sequence
    mu_all = torch.sigmoid(w_mu(x))    # [1, total_len, 1]
    alpha_all = w_alpha(x)              # [1, total_len, 1]

    # ── Fast decode path: one token per document ────────────────────
    if total_len == n_seqs and (lengths == 1).all():
        m_prev = cache.m_prev                         # [n_seqs, D]
        e_flat = e[0]                                  # [n_seqs, D]
        mu_flat = mu_all[0]                            # [n_seqs, 1]
        alpha_flat = alpha_all[0]                      # [n_seqs, 1]

        out[0] = e_flat + alpha_flat * m_prev
        cache.m_prev = mu_flat * m_prev + (1.0 - mu_flat) * e_flat
        return out

    # ── Prefill path: Triton generate kernel per document ───────────
    e_0 = e[0]                                         # [total_len, D]
    len_list = lengths.tolist()
    e_docs = e_0.split(len_list, dim=0)                # list of [L_i, D]
    mu_docs = mu_all[0].split(len_list, dim=0)         # list of [L_i, 1]
    alpha_docs = alpha_all[0].split(len_list, dim=0)   # list of [L_i, 1]

    out_parts = []
    for i, (e_doc, mu_doc, alpha_doc) in enumerate(
        zip(e_docs, mu_docs, alpha_docs)
    ):
        e_doc_3d = e_doc.unsqueeze(0).contiguous()      # [1, L_i, D]
        mu_doc_3d = mu_doc.unsqueeze(0).contiguous()    # [1, L_i, 1]
        a_doc_3d = alpha_doc.unsqueeze(0).contiguous()  # [1, L_i, 1]
        m0_i = cache.m_prev[i : i + 1].float()         # [1, D], float32

        out_i, m_final_i = selective_iir_scan_generate_triton(
            e_doc_3d, mu_doc_3d, a_doc_3d, m0_i,
        )
        out_parts.append(out_i[0])                      # [L_i, D]
        cache.m_prev[i] = m_final_i[0]

    out[0] = torch.cat(out_parts, dim=0)
    return out


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class SSSStemTransformerArgs(StemTransformerArgs):
    """Extra hyper-parameters for the SSS (Selective State Space) STEM variant.

    Direct counterpart of :class:`~lingua.stem_iir.IIRStemTransformerArgs`.
    The ``iir_per_dim_alpha`` flag is absent because the selective variant
    uses scalar gates by design (per the Mamba2 SSD restriction).
    """

    # Initial EMA decay factor.  The forget-gate bias is set to
    # logit(sss_mu_init) so that the initial mu_t ~ sss_mu_init.
    sss_mu_init: float = 0.9

    # Initial readout scale.  The readout-gate bias is set to this value.
    sss_alpha_init: float = 1.0

    # Whether ``m0`` (initial memory vector) is a learnable parameter or a
    # fixed zero buffer.
    sss_learnable_m0: bool = False


# ---------------------------------------------------------------------------
# Core SSS scan (Triton-accelerated, via lingua.selective_iir)
# ---------------------------------------------------------------------------

def sss_scan(
    e: torch.Tensor,
    mu_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    m0: torch.Tensor,
) -> torch.Tensor:
    """Causal selective-IIR scan over the sequence dimension.

    Counterpart of :func:`lingua.stem_iir.iir_scan`.

    Args:
        e:         ``[B, L, D]`` raw stem embeddings (CUDA, contiguous).
        mu_seq:    ``[B, L, 1]`` per-timestep forget gates in (0, 1).
        alpha_seq: ``[B, L, 1]`` per-timestep readout scales.
        m0:        ``[D]``       initial memory vector (float32).

    Returns:
        e_tilde: ``[B, L, D]``  contextual stem embeddings.
    """
    return selective_iir_scan_triton(e.contiguous(), mu_seq, alpha_seq, m0)


# ---------------------------------------------------------------------------
# Per-layer SSS module
# ---------------------------------------------------------------------------

class SSSMemory(nn.Module):
    """Learnable input-conditioned (selective) IIR memory for a single STEM
    layer.

    Counterpart of :class:`~lingua.stem_iir.IIRMemory`.

    Holds per-layer gate projections ``w_mu``, ``w_alpha`` (each
    ``nn.Linear(dim, 1)``) and optionally ``m0`` (initial memory vector).

    Key API difference from ``IIRMemory``:  ``forward(e, x)`` takes the
    transformer hidden state ``x`` as a second argument, since the gates
    are conditioned on it.

    Parameters
    ----------
    dim : int
        Transformer hidden dimension (dimension of ``x_t``).
    d_ff : int
        Dimensionality of the STEM (= FFN hidden dim, dimension of ``e_t``).
    mu_init : float
        Initial value for the EMA decay in (0, 1).  The forget-gate bias
        is set to ``logit(mu_init)`` so that at initialisation, ``mu_t``
        equals ``mu_init`` regardless of the input ``x_t``.
    alpha_init : float
        Initial scale for the memory contribution.  The readout-gate bias
        is set to this value.
    learnable_m0 : bool
        If *True*, ``m0`` is an ``nn.Parameter``; otherwise a zero buffer.
    """

    def __init__(
        self,
        dim: int,
        d_ff: int,
        mu_init: float = 0.9,
        alpha_init: float = 1.0,
        learnable_m0: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff
        self._mu_init = mu_init
        self._alpha_init = alpha_init

        # Gate projections: dim → 1 (scalar output per timestep).
        self.w_mu = nn.Linear(dim, 1, bias=False)
        self.w_alpha = nn.Linear(dim, 1, bias=False)

        # m0: initial memory vector.
        if learnable_m0:
            self.m0 = nn.Parameter(torch.zeros(d_ff))
        else:
            self.register_buffer("m0", torch.zeros(d_ff))

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(self):
        """Re-set SSS parameters to their configured initial values.

        Initialises gate weights to zero and biases to match the fixed-IIR
        regime (mu_t = mu_init, alpha_t = alpha_init for all inputs).

        Must be called after ``to_empty()`` since the meta→device
        materialisation leaves parameter memory uninitialised.
        """
        
        if self.w_mu.bias is not None:
            nn.init.zeros_(self.w_mu.weight)
            self.w_mu.bias.fill_(_inverse_sigmoid(self._mu_init))

        if self.w_alpha.bias is not None:
            nn.init.zeros_(self.w_alpha.weight)
            self.w_alpha.bias.fill_(self._alpha_init)

        self.m0.zero_()

    # ------------------------------------------------------------------
    # Properties (for logging)
    # ------------------------------------------------------------------

    @property
    def mu(self) -> torch.Tensor:
        """Effective EMA decay factor (input-independent component).

        Returns ``sigma(w_mu.bias)``, i.e. the forget gate value when
        the input contribution is zero (at initialisation, or if the
        model learns to ignore the input).

        For the actual per-timestep ``mu_t`` values during a forward pass,
        see the logged metrics in the training script.
        """
        return torch.sigmoid(self.w_mu.bias)

    @property
    def mu_bias_value(self) -> float:
        """Raw forget-gate bias (before sigmoid)."""
        return self.w_mu.bias.item()

    @property
    def alpha_bias_value(self) -> float:
        """Readout-gate bias."""
        return self.w_alpha.bias.item()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the selective-IIR memory scan to raw stem embeddings.

        In training mode (no ``sss_cache`` attribute), dispatches to the
        fused Triton kernel via :func:`sss_scan`.  In generation mode
        (``sss_cache`` is set by the generator), uses a stateful
        per-document path that persists memory across decode steps.

        Args:
            e: ``[B, L, D]`` raw (token-level) stem embeddings.
            x: ``[B, L, dim]`` transformer hidden states (pre-FFN, the
               residual stream input to the current layer).

        Returns:
            e_tilde: ``[B, L, D]`` contextual stem embeddings.
        """
        if hasattr(self, "sss_cache"):
            return _sss_scan_generate(
                e, x, self.w_mu, self.w_alpha, self.sss_cache,
            )
        # Training path: compute gates in PyTorch (autograd-tracked),
        # then dispatch recurrence to the fused Triton kernel.
        mu_seq = torch.sigmoid(self.w_mu(x))    # [B, L, 1]
        alpha_seq = self.w_alpha(x)               # [B, L, 1]
        return sss_scan(e, mu_seq, alpha_seq, self.m0)