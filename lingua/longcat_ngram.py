# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# N-gram enriched token embeddings (Longcat-style), pure PyTorch.
# Reference: Meituan Longcat N-gram embedding; HF adapter is deferred to Phase 2.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from lingua.stem_dist_utils import VocabParallelEmbedding


def segment_ids_from_packed_lengths(
    lengths: torch.Tensor, *, device: torch.device, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """
    Per-position document index for a single concatenation of run lengths, shape ``(sum(lengths),)``.
    Used to align n-gram segments with :func:`pack_prompts` / block attention in packed prefill.
    """
    if lengths.dim() != 1:
        raise ValueError("lengths must be 1-D")
    parts: List[torch.Tensor] = []
    for i, ell in enumerate(lengths.tolist()):
        parts.append(torch.full((int(ell),), int(i), device=device, dtype=dtype))
    if not parts:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.cat(parts, dim=0)


def _as_reset_token_ids(value: object, eos_id: int) -> Tuple[int, ...]:
    """If None, return (eos_id,). Empty means no id-based n-gram segment breaks (only ``ngram_segment_ids``).

    Accepts a scalar, ``list``/``tuple``/``range``, or a :class:`collections.abc.Sequence`
    (e.g. OmegaConf ``ListConfig``) of ints.
    """
    if value is None:
        return (int(eos_id),)
    if isinstance(value, (str, bytes)):
        raise TypeError("ngram_reset_token_ids must be a sequence of ints, not a string/bytes")
    if isinstance(value, Sequence):
        return tuple(int(x) for x in value)
    return (int(value),)  # type: ignore[arg-type]


@dataclass
class LongcatNgramConfig:
    """Hyperparameters for :class:`NgramEmbedding`."""

    vocab_size: int
    hidden_size: int
    pad_token_id: int
    eos_token_id: int
    emb_neighbor_num: int
    """Maximum n-gram order (n in the reference); uses orders 2..n."""
    emb_split_num: int
    """Splits per order (k in the reference)."""
    ngram_vocab_size_ratio: float
    """Base multiplier m = ratio * vocab_size for n-gram table sizes."""
    ngram_reset_token_ids: Optional[Union[Tuple[int, ...], List[int]]] = field(
        default=None
    )
    """
    N-gram context resets after these token ids: the next position starts a new
    n-gram segment. If None, it defaults to ``(eos_token_id,)`` (unchanged
    from the original EOS-only shift). If empty, only ``ngram_segment_ids`` in
    :meth:`NgramEmbedding.forward` and position 0 set boundaries.
    """

    def __post_init__(self) -> None:
        n = self.emb_neighbor_num
        k = self.emb_split_num
        if n < 2 or k < 1:
            raise ValueError("emb_neighbor_num must be >= 2 and emb_split_num >= 1")
        num = k * (n - 1)
        if self.hidden_size % num != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by k*(n-1) = {num}"
            )


class NgramEmbedding(nn.Module):
    """
    Hashed n-gram shard embeddings only (no token table, no per-shard up-projections).

    Forward returns the concatenation of all shard vectors along the last axis
    (shape ``(..., hidden_size)`` with ``hidden_size = emb_dim * k * (n-1)``).
    Base token embeddings and :class:`torch.nn.Linear` projections into model
    width are composed outside this module (see :class:`apps.main.longcat.NgramLMTransformer`).
    """

    def __init__(self, config: Union[LongcatNgramConfig, object], device: Optional[torch.device] = None):
        super().__init__()
        self.config = config

        self.m = float(config.ngram_vocab_size_ratio) * int(config.vocab_size)
        self.k = int(config.emb_split_num)
        self.n = int(config.emb_neighbor_num)
        self.num_embedders = self.k * (self.n - 1)
        self.emb_dim = int(self.config.hidden_size) // self.num_embedders

        self._init_ngram_embeddings(device=device)
        self._vocab_mods_cache: Optional[Dict[Tuple[int, int], List[int]]] = None
        self._ngram_reset_ids: Tuple[int, ...] = _as_reset_token_ids(
            getattr(config, "ngram_reset_token_ids", None), int(config.eos_token_id)
        )

    def _init_ngram_embeddings(self, device: Optional[torch.device] = None) -> None:
        pad_id = int(self.config.pad_token_id)

        embedders = []
        for i in range(self.num_embedders):
            vs = int(self.m + i * 2 + 1)
            pidx = pad_id if pad_id < vs else None
            emb = VocabParallelEmbedding(
                vs, self.emb_dim, padding_idx=pidx, device=device
            )
            embedders.append(emb)

        self.embedders = nn.ModuleList(embedders)

    def _build_segment_starts(
        self,
        context: torch.Tensor,
        ngram_segment_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        For each (batch, t), true means position t is the first token of an
        n-gram segment. Boundaries: position 0; a change in ``ngram_segment_ids``;
        or the position after a token in ``_ngram_reset_ids``.
        """
        batch_size, seq_len = context.shape
        device = context.device
        st = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        st[:, 0] = True
        if ngram_segment_ids is not None:
            if ngram_segment_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"ngram_segment_ids {ngram_segment_ids.shape} != context {context.shape}"
                )
            st[:, 1:] = ngram_segment_ids[:, 1:] != ngram_segment_ids[:, :-1]
        for rid in self._ngram_reset_ids:
            st[:, 1:] |= context[:, :-1] == int(rid)
        return st

    def _shift_right_at_starts(
        self, tensor: torch.Tensor, n: int, segment_starts: torch.Tensor
    ) -> torch.Tensor:
        """Right-shift by n within each segment delimited by ``segment_starts`` (zeros elsewhere)."""
        batch_size, seq_len = tensor.shape
        result = torch.zeros_like(tensor)
        for i in range(batch_size):
            starts = segment_starts[i].nonzero(as_tuple=True)[0]
            for j in range(len(starts)):
                start = int(starts[j].item())
                end = int(starts[j + 1].item()) if j + 1 < len(starts) else seq_len
                if end - start > n:
                    result[i, start + n : end] = tensor[i, start : end - n]
        return result

    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods: Dict[Tuple[int, int], List[int]] = {}
        vocab_size = int(self.config.vocab_size)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self,
        input_ids: torch.Tensor,
        shifted_ids: Dict[int, torch.Tensor],
        vocab_mods: List[int],
        ngram: int,
    ) -> torch.Tensor:
        ngram_ids = input_ids.clone()
        for kk in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[kk] * vocab_mods[kk - 2]
        return ngram_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor] = None,
        ngram_segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(-1)

        if ngram_context is not None:
            context = torch.cat(
                [ngram_context[..., -(self.n - 1) :], input_ids], dim=-1
            )
        else:
            context = input_ids

        seg_for_context: Optional[torch.Tensor] = None
        if ngram_segment_ids is not None:
            if ngram_context is not None:
                p = ngram_segment_ids[:, 0:1].expand(-1, self.n - 1)
                seg_for_context = torch.cat([p, ngram_segment_ids], dim=-1)
            else:
                seg_for_context = ngram_segment_ids
            if seg_for_context.shape != context.shape:
                raise ValueError(
                    f"ngram_segment_ids (extended) shape {seg_for_context.shape} != context {context.shape}"
                )
        # when ngram_segment_ids is None, use only _ngram_reset_ids (default eos) — training default

        seg_starts = self._build_segment_starts(context, seg_for_context)

        vocab_mods = self._precompute_vocab_mods()

        shifted_ids: Dict[int, torch.Tensor] = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_at_starts(context, i - 1, seg_starts)

        out = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]

                x_ngram = self.embedders[index](new_ids)
                out.append(x_ngram)

        return torch.cat(out, dim=-1)


__all__ = ["LongcatNgramConfig", "NgramEmbedding", "segment_ids_from_packed_lengths"]
