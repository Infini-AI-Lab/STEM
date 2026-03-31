# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Sliding-window STEM embedding extension.

This module replaces tokenwise STEM embeddings with a causal trailing-window
average over the sequence:

    e_avg[t] = mean(e[max(0, t - W + 1) : t + 1])

When local token indices ``tok_idx`` are available, windows are reset at
positions where ``tok_idx == 0`` (packed-document boundaries).
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from lingua.stem import StemTransformerArgs


@dataclass
class WindowStemTransformerArgs(StemTransformerArgs):
    """Extra hyper-parameters for the sliding-window STEM variant."""

    stem_window_size: int = 16


class SlidingWindowCache:
    """Per-document decode cache for causal sliding-window averages."""

    def __init__(
        self,
        n_seqs: int,
        d_ff: int,
        window_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.window_size = window_size
        self.buffer = torch.zeros(
            n_seqs, window_size, d_ff, dtype=dtype, device=device,
        )
        self.running_sum = torch.zeros(n_seqs, d_ff, dtype=dtype, device=device)
        self.write_pos = torch.zeros(n_seqs, dtype=torch.long, device=device)
        self.valid_count = torch.zeros(n_seqs, dtype=torch.long, device=device)
        self.doc_lengths: Optional[torch.Tensor] = None

    def reset(self):
        self.buffer.zero_()
        self.running_sum.zero_()
        self.write_pos.zero_()
        self.valid_count.zero_()
        self.doc_lengths = None


def _moving_avg_conv1d(e: torch.Tensor, window_size: int) -> torch.Tensor:
    """Causal trailing average with depthwise conv1d.

    Args:
        e: [B, L, D]
        window_size: positive integer W.
    """
    bsz, seqlen, dim = e.shape
    if seqlen == 0:
        return e

    x = e.transpose(1, 2)  # [B, D, L]
    x = F.pad(x, (window_size - 1, 0))  # causal left padding

    # Sum over trailing windows per embedding channel.
    k_sum = torch.ones(dim, 1, window_size, device=e.device, dtype=e.dtype)
    num = F.conv1d(x, k_sum, groups=dim)  # [B, D, L]

    # Dynamic denominator near starts (1, 2, ..., W, W, ...).
    ones = torch.ones(bsz, 1, seqlen, device=e.device, dtype=e.dtype)
    ones = F.pad(ones, (window_size - 1, 0))
    k_den = torch.ones(1, 1, window_size, device=e.device, dtype=e.dtype)
    den = F.conv1d(ones, k_den).clamp_min(1.0)  # [B, 1, L]

    return (num / den).transpose(1, 2)  # [B, L, D]


def _segment_starts(tok_idx: torch.Tensor, seqlen: int) -> list[int]:
    starts = (tok_idx == 0).nonzero(as_tuple=False).flatten().tolist()
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts = [s for s in starts if 0 <= s < seqlen]
    return starts


def _causal_trailing_avg(
    e: torch.Tensor,
    window_size: int,
    tok_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute causal trailing-window average over sequence dim.

    With ``tok_idx`` provided, windows reset at ``tok_idx == 0`` boundaries.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    bsz, seqlen, _ = e.shape
    if seqlen == 0:
        return e

    if tok_idx is None:
        return _moving_avg_conv1d(e, window_size)

    if tok_idx.dim() == 1:
        if tok_idx.shape[0] != seqlen:
            raise ValueError("tok_idx [L] must match e.shape[1].")
        out = torch.empty_like(e)
        starts = _segment_starts(tok_idx, seqlen)
        for s, t in zip(starts, starts[1:] + [seqlen]):
            out[:, s:t, :] = _moving_avg_conv1d(e[:, s:t, :], window_size)
        return out

    if tok_idx.dim() != 2 or tok_idx.shape != (bsz, seqlen):
        raise ValueError("tok_idx must have shape [L] or [B, L].")

    out = torch.empty_like(e)
    for b in range(bsz):
        starts = _segment_starts(tok_idx[b], seqlen)
        for s, t in zip(starts, starts[1:] + [seqlen]):
            out[b : b + 1, s:t, :] = _moving_avg_conv1d(
                e[b : b + 1, s:t, :], window_size,
            )
    return out


def _sliding_window_generate(
    e: torch.Tensor,
    cache: SlidingWindowCache,
) -> torch.Tensor:
    """Sliding-window averaging with persistent cache across decode steps."""
    _, total_len, _ = e.shape
    lengths = cache.doc_lengths
    if lengths is None:
        raise RuntimeError("SlidingWindowCache.doc_lengths must be set before forward.")

    n_seqs = lengths.size(0)
    if total_len == n_seqs and (lengths == 1).all():
        # Decode fast path: one token per document.
        x = e[0].to(cache.running_sum.dtype)  # [n_seqs, D]
        seq_idx = torch.arange(n_seqs, device=e.device)
        pos = cache.write_pos
        oldest = cache.buffer[seq_idx, pos]
        full = (cache.valid_count >= cache.window_size).to(x.dtype).unsqueeze(1)

        cache.running_sum = cache.running_sum - oldest * full + x
        cache.buffer[seq_idx, pos] = x
        cache.write_pos = (cache.write_pos + 1) % cache.window_size
        cache.valid_count = torch.clamp(cache.valid_count + 1, max=cache.window_size)

        denom = cache.valid_count.to(cache.running_sum.dtype).unsqueeze(1)
        out = cache.running_sum / denom
        return out.to(e.dtype).unsqueeze(0)

    # Prefill path for packed docs with known lengths.
    docs = e[0].split(lengths.tolist(), dim=0)
    out_parts = []

    for i, doc in enumerate(docs):
        if doc.numel() == 0:
            continue
        avg_doc = _moving_avg_conv1d(doc.unsqueeze(0), cache.window_size)
        out_parts.append(avg_doc[0])

        keep = min(cache.window_size, doc.size(0))
        tail = doc[-keep:].to(cache.running_sum.dtype)
        cache.buffer[i].zero_()
        cache.buffer[i, :keep] = tail
        cache.running_sum[i] = tail.sum(dim=0)
        cache.valid_count[i] = keep
        cache.write_pos[i] = keep % cache.window_size

    out = torch.cat(out_parts, dim=0).unsqueeze(0)
    return out


class SlidingWindowMemory(nn.Module):
    """Applies a causal sliding-window average to STEM embeddings."""

    def __init__(self, d_ff: int, window_size: int = 16):
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self.d_ff = d_ff
        self.window_size = window_size

    @torch.no_grad()
    def reset_parameters(self):
        """No learnable params. Kept for uniform init APIs."""
        return None

    def forward(
        self,
        e: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self, "window_cache"):
            return _sliding_window_generate(e, self.window_cache)
        return _causal_trailing_avg(e, self.window_size, tok_idx=tok_idx)
