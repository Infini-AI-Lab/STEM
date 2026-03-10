# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
IIR (Infinite Impulse Response) / EMA memory extension for STEM embeddings.

Instead of feeding the raw per-token STEM vector ``e_t = stem_embed(token_t)``
into each STEM-replaced FFN layer, we blend in a causal exponential moving
average (EMA) of past STEM embeddings:

    m_t       = mu * m_{t-1} + (1 - mu) * e_t          (memory update)
    ctx_t     = m_{t-1}                                  (causal: previous memory)
    e_tilde_t = e_t + alpha * ctx_t                      (contextual output)

The parameters ``mu``, ``alpha`` (and optionally the initial memory ``m0``)
are **per-stem-layer** and **learnable**.

This module lives at the embedding level and does **not** change the
transformer blocks or FFN classes.
"""

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import nn

from lingua.stem import StemTransformerArgs
from lingua.iir import iir_scan_triton, iir_scan_generate_triton


# ---------------------------------------------------------------------------
# Generation-mode IIR cache
# ---------------------------------------------------------------------------

class IIRCache:
    """Holds per-document IIR memory state across autoregressive decode steps.

    Analogous to :class:`KVCache` for attention layers.  Attached to an
    :class:`IIRMemory` module by the generator before inference begins.
    """

    def __init__(self, n_seqs: int, d_ff: int, dtype: torch.dtype, device: torch.device):
        self.m_prev = torch.zeros(n_seqs, d_ff, dtype=dtype, device=device)
        self.doc_lengths: Optional[torch.Tensor] = None

    def reset(self):
        self.m_prev.zero_()
        self.doc_lengths = None


def _iir_scan_generate(
    e: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    cache: IIRCache,
) -> torch.Tensor:
    """IIR scan for generation with per-document cached memory.

    The packed sequence ``e`` of shape ``[1, total_len, D]`` contains
    ``n_seqs`` documents whose boundaries are given by
    ``cache.doc_lengths``.  Each document's IIR scan starts from
    ``cache.m_prev[doc_idx]`` and the final memory state is written back.

    Decode (L=1) uses a fully vectorized PyTorch fast path.  Prefill
    dispatches each document to the Triton generate kernel.
    """
    _, total_len, D = e.shape
    lengths = cache.doc_lengths
    n_seqs = lengths.size(0)
    out = torch.empty_like(e)
    one_minus_mu = (1.0 - mu).squeeze()
    mu_s = mu.squeeze()

    if total_len == n_seqs and (lengths == 1).all():
        # Fast path: decode phase — one token per document, fully vectorized
        m_prev = cache.m_prev                        # [n_seqs, D]
        e_flat = e[0]                                 # [n_seqs, D]
        out[0] = e_flat + alpha * m_prev
        cache.m_prev = mu_s * m_prev + one_minus_mu * e_flat
        return out

    # Prefill path: call Triton kernel per document (eliminates the Python
    # time-loop entirely; n_seqs kernel launches ≪ max_len iterations).
    e_0 = e[0]                                        # [total_len, D]
    len_list = lengths.tolist()
    docs = e_0.split(len_list, dim=0)                 # list of [L_i, D]

    out_parts = []
    for i, doc in enumerate(docs):
        doc_3d = doc.unsqueeze(0).contiguous()        # [1, L_i, D]
        m0_i = cache.m_prev[i : i + 1].float()       # [1, D], float32
        out_i, m_final_i = iir_scan_generate_triton(doc_3d, mu, alpha, m0_i)
        out_parts.append(out_i[0])                    # [L_i, D]
        cache.m_prev[i] = m_final_i[0]

    out[0] = torch.cat(out_parts, dim=0)
    return out


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class IIRStemTransformerArgs(StemTransformerArgs):
    """Extra hyper-parameters for the IIR / EMA memory variant of STEM."""

    # Initial EMA decay factor.  Stored internally as ``sigmoid^{-1}(mu_init)``
    # so that the optimiser sees an unconstrained real while ``mu`` stays in (0, 1).
    iir_mu_init: float = 0.9

    # Initial scale applied to the memory contribution.
    iir_alpha_init: float = 1.0

    # Whether ``m0`` (initial memory vector) is a learnable parameter or a
    # fixed zero buffer.
    iir_learnable_m0: bool = False

    # If True, ``alpha`` is a per-dimension vector in R^{d_ff} instead of a
    # scalar.  Allows the model to weight dimensions differently.
    iir_per_dim_alpha: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: float) -> float:
    """Compute logit (inverse sigmoid) so that ``sigmoid(result) ≈ x``."""
    assert 0.0 < x < 1.0, f"mu_init must be in (0, 1), got {x}"
    return math.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# Core IIR scan (Triton-accelerated)
# ---------------------------------------------------------------------------

def iir_scan(
    e: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    m0: torch.Tensor,
) -> torch.Tensor:
    """Causal IIR / EMA scan over the sequence dimension.

    Args:
        e:     ``[B, L, D]`` raw stem embeddings (CUDA, contiguous).
        mu:    scalar tensor in (0, 1) – EMA decay factor.
        alpha: scalar **or** ``[D]`` tensor – scale for memory contribution.
        m0:    ``[D]`` initial memory vector.

    Returns:
        e_tilde: ``[B, L, D]``  =  ``e + alpha * ctx``
            where ``ctx[:, t, :] = m_{t-1}`` (causal context).
    """
    return iir_scan_triton(e.contiguous(), mu, alpha, m0)


# ---------------------------------------------------------------------------
# Per-layer IIR module
# ---------------------------------------------------------------------------

class IIRMemory(nn.Module):
    """Learnable IIR / EMA memory for a single STEM layer.

    Holds the per-layer parameters ``mu``, ``alpha``, and ``m0`` and
    wraps :func:`iir_scan`.

    Parameters
    ----------
    d_ff : int
        Dimensionality of the STEM (= FFN hidden dim).
    mu_init : float
        Initial value for the EMA decay in (0, 1).
    alpha_init : float
        Initial scale for the memory contribution.
    learnable_m0 : bool
        If *True*, ``m0`` is an ``nn.Parameter``; otherwise a zero buffer.
    per_dim_alpha : bool
        If *True*, ``alpha`` is a ``[d_ff]`` vector; otherwise a scalar.
    """

    def __init__(
        self,
        d_ff: int,
        mu_init: float = 0.9,
        alpha_init: float = 1.0,
        learnable_m0: bool = False,
        per_dim_alpha: bool = False,
    ):
        super().__init__()
        self.d_ff = d_ff
        self._mu_init = mu_init
        self._alpha_init = alpha_init

        # mu: stored as raw logit, mapped through sigmoid to stay in (0, 1).
        raw_mu = _inverse_sigmoid(mu_init)
        self.raw_mu = nn.Parameter(torch.tensor([raw_mu]))

        # alpha: scale factor for the memory contribution.
        if per_dim_alpha:
            self.alpha = nn.Parameter(torch.full((d_ff,), alpha_init))
        else:
            self.alpha = nn.Parameter(torch.tensor([alpha_init]))

        # m0: initial memory vector.
        if learnable_m0:
            self.m0 = nn.Parameter(torch.zeros(d_ff))
        else:
            self.register_buffer("m0", torch.zeros(d_ff))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(self):
        """Re-set IIR parameters to their configured initial values.

        Must be called after ``to_empty()`` since the meta->device
        materialisation leaves parameter memory uninitialised.
        """
        raw_mu = _inverse_sigmoid(self._mu_init)
        self.raw_mu.fill_(raw_mu)
        self.alpha.fill_(self._alpha_init)
        self.m0.zero_()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mu(self) -> torch.Tensor:
        """EMA decay factor, always in (0, 1)."""
        return torch.sigmoid(self.raw_mu)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Apply the IIR memory scan to raw stem embeddings.

        In training mode (no ``iir_cache`` attribute), dispatches to the
        fused Triton kernel.  In generation mode (``iir_cache`` is set by
        the generator), uses a pure-PyTorch path that respects per-document
        boundaries and persists memory across decode steps.

        Args:
            e: ``[B, L, D]`` raw (token-level) stem embeddings.

        Returns:
            e_tilde: ``[B, L, D]`` contextual stem embeddings.
        """
        if hasattr(self, "iir_cache"):
            return _iir_scan_generate(e, self.mu, self.alpha, self.iir_cache)
        return iir_scan(e, self.mu, self.alpha, self.m0)

