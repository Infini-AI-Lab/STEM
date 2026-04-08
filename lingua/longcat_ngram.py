# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# N-gram enriched token embeddings (Longcat-style), pure PyTorch.
# Reference: Meituan Longcat N-gram embedding; HF adapter is deferred to Phase 2.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn


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
    Stateless embedding: base token vectors plus a sum of hashed n-gram features.

    ``ngram_context`` (optional) is up to ``n-1`` prior token ids per row, used
    when decoding step-by-step; for standard teacher-forced LM training, pass
    ``ngram_context=None`` so context is taken from ``input_ids`` only.
    """

    def __init__(self, config: Union[LongcatNgramConfig, object], base_embeddings: nn.Embedding):
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings

        self.m = float(config.ngram_vocab_size_ratio) * int(config.vocab_size)
        self.k = int(config.emb_split_num)
        self.n = int(config.emb_neighbor_num)

        self._init_ngram_embeddings()
        self._vocab_mods_cache: Optional[Dict[Tuple[int, int], List[int]]] = None

    def _init_ngram_embeddings(self) -> None:
        num_embedders = self.k * (self.n - 1)
        emb_dim = int(self.config.hidden_size) // num_embedders

        offsets: List[int] = []
        o = 0
        for i in range(num_embedders):
            offsets.append(o)
            o += int(self.m + i * 2 + 1)
        # One shared padding row (per-branch padding_idx is not expressible in a single table).
        self.ngram_embedding = nn.Embedding(
            o + 1, emb_dim, padding_idx=o
        )
        self.register_buffer(
            "ngram_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )

        self.post_proj = nn.Linear(
            int(self.config.hidden_size), int(self.config.hidden_size), bias=False
        )

    def _shift_right_ignore_eos(
        self, tensor: torch.Tensor, n: int, eos_token_id: int
    ) -> torch.Tensor:
        """Shift tensor right by n positions, resetting at EOS tokens."""
        batch_size, seq_len = tensor.shape
        result = torch.zeros_like(tensor)
        eos_mask = tensor == eos_token_id

        for i in range(batch_size):
            eos_positions = eos_mask[i].nonzero(as_tuple=True)[0]
            prev_idx = 0

            for eos_idx in eos_positions:
                end_idx = eos_idx.item() + 1
                if end_idx - prev_idx > n:
                    result[i, prev_idx + n : end_idx] = tensor[i, prev_idx : end_idx - n]
                prev_idx = end_idx

            if prev_idx < seq_len and seq_len - prev_idx > n:
                result[i, prev_idx + n : seq_len] = tensor[i, prev_idx : seq_len - n]

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
    ) -> torch.Tensor:
        seq_len = input_ids.size(-1)

        if ngram_context is not None:
            context = torch.cat(
                [ngram_context[..., -(self.n - 1) :], input_ids], dim=-1
            )
        else:
            context = input_ids

        x = self.word_embeddings(input_ids).clone() 

        vocab_mods = self._precompute_vocab_mods()
        eos_id = int(self.config.eos_token_id)
        pad_id = int(self.config.pad_token_id)
        pad_row = self.ngram_embedding.padding_idx
        assert pad_row is not None

        shifted_ids: Dict[int, torch.Tensor] = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(context, i - 1, eos_id)

        flat_stack: List[torch.Tensor] = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:].long()
                flat = self.ngram_offsets[index] + new_ids
                if pad_id < emb_vocab_dim:
                    flat = flat.masked_fill(new_ids == pad_id, pad_row)
                flat_stack.append(flat)

        # (B, S, R, emb_dim) -> (B, S, hidden_size); single gather + one projection.
        ngram_flat = self.ngram_embedding(torch.stack(flat_stack, dim=-1)).flatten(-2, -1)
        x = x + self.post_proj(ngram_flat)

        x = x / (1 + self.k * (self.n - 1))
        return x


__all__ = ["LongcatNgramConfig", "NgramEmbedding"]
