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
from lingua.transformer import BaseTransformerArgs

# ``VocabParallelEmbedding`` needs ``num_embeddings`` divisible by the MP group size;
# pad tables to a fixed multiple so common widths (2, 4, 8, 16, 32) all divide evenly.
_NGRAM_VOCAB_ALIGNMENT = 32


def _align_ngram_num_embeddings(logical_num_embeddings: int) -> int:
    n = int(logical_num_embeddings)
    a = _NGRAM_VOCAB_ALIGNMENT
    return (n + a - 1) // a * a


def segment_ids_from_packed_lengths(
    lengths: torch.Tensor, *, device: torch.device, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    if lengths.dim() != 1:
        raise ValueError("lengths must be 1-D")

    lengths = lengths.to(device=device, dtype=torch.long)

    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device, dtype=dtype),
        lengths,
    )
    
    
def validate_longcat_ngram_layout(obj: object) -> None:
    """Require ``dim % (k * (n-1)) == 0`` for n-gram shard layout."""
    n = int(getattr(obj, "emb_neighbor_num"))
    k = int(getattr(obj, "emb_split_num"))
    dim = int(getattr(obj, "dim"))
    if n < 2 or k < 1:
        raise ValueError("emb_neighbor_num must be >= 2 and emb_split_num >= 1")
    num = k * (n - 1)
    if dim % num != 0:
        raise ValueError(
            f"dim ({dim}) must be divisible by k*(n-1) = {num} for Longcat n-gram layout"
        )

@dataclass
class LongcatNgramConfig(BaseTransformerArgs):
    pad_token_id: int = 0
    eos_token_id: int = 1
    emb_neighbor_num: int = 3
    """Maximum n-gram order (n in the reference); uses orders 2..n."""
    emb_split_num: int = 2
    """Splits per order (k in the reference)."""
    vocab_size: int = -1
    ngram_vocab_size_ratio: float = 16
    """Base multiplier m = ratio * vocab_size for n-gram table sizes."""
    
    def __post_init__(self) -> None:
        validate_longcat_ngram_layout(self)
        

class NgramEmbedding(nn.Module):
    def __init__(self, config: Union[LongcatNgramConfig, object], device: Optional[torch.device] = None):
        super().__init__()
        self.config = config

        self.m = float(config.ngram_vocab_size_ratio) * int(config.vocab_size)
        self.k = int(config.emb_split_num)
        self.n = int(config.emb_neighbor_num)
        self.num_embedders = self.k * (self.n - 1)
        self.emb_dim = int(self.config.dim) // self.num_embedders

        # Segment boundaries in ``_build_segment_starts``: after pad / eos (config-driven).
        pad_id = int(self.config.pad_token_id)
        eos_id = int(self.config.eos_token_id)
        self._ngram_reset_ids: Tuple[int, ...] = tuple(
            dict.fromkeys((pad_id, eos_id))
        )

        self._init_ngram_embeddings(device=device)
        self._vocab_mods_cache: Optional[Dict[Tuple[int, int], List[int]]] = None
        
        self.ngram_context = None
        self.max_context_len = self.n - 1
        
    def _init_ngram_embeddings(self, device: Optional[torch.device] = None) -> None:
        embedders = []
        pad_token_id = int(self.config.pad_token_id)
        for i in range(self.num_embedders):
            logical_vs = int(self.m + i * 2 + 1)
            padded_vs = _align_ngram_num_embeddings(logical_vs)
            pidx = pad_token_id if pad_token_id < logical_vs else None
            emb = VocabParallelEmbedding(
                padded_vs, self.emb_dim, padding_idx=pidx, device=device
            )
            embedders.append(emb)

        self.embedders = nn.ModuleList(embedders)
        
        
    def update_ngram_context(self, context: torch.Tensor) -> None:
        if self.ngram_context is None:
            self.ngram_context = context.clone()
        else:
            self.ngram_context = torch.cat([self.ngram_context, context], dim=-1)
            
        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len:]
            
    def reset_ngram_context(self) -> None:  
        self.ngram_context = None
    
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
        else:
            for rid in self._ngram_reset_ids:
                st[:, 1:] |= context[:, :-1] == int(rid)
        return st    
    
        
    def _shift_right_at_starts(self, x: torch.Tensor, n: int, segment_starts: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device
        
        # [1, seq_len]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)

        # segment id per position
        # starts should be 1/True at segment starts
        seg_ids = segment_starts.to(torch.long).cumsum(dim=1) - 1

        # start position of each segment, broadcast per token
        start_pos = torch.zeros_like(pos.expand(batch_size, -1))
        start_pos = start_pos.scatter(1, pos.expand(batch_size, -1), pos.expand(batch_size, -1) * segment_starts.to(torch.long))
        start_pos = torch.cummax(start_pos, dim=1).values

        # source position after shifting right by n
        src_pos = pos.expand(batch_size, -1) - n

        # valid only if source stays inside same segment
        valid = src_pos >= start_pos

        src_pos = src_pos.clamp_min(0)

        gathered = x.gather(dim=1, index=src_pos)
        return torch.where(valid, gathered, torch.zeros_like(x))
    
    
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
        ngram_segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        ngram_context = self.ngram_context
        is_decode = ngram_context is not None
        if is_decode:
            # useful for decoding
            input_ids = input_ids.transpose(0, 1).contiguous()
            context = torch.cat(
                [ngram_context[..., -(self.n - 1) :], input_ids], dim=-1
            )
            seg_starts = torch.zeros_like(context, dtype=torch.bool)
            seg_starts[:, 0] = True
        else:
            context = input_ids
            seg_starts = self._build_segment_starts(context, ngram_segment_ids)

        seq_len = input_ids.size(-1)
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

        out = torch.cat(out, dim=-1)
        
        if is_decode:
            return out.transpose(0, 1).contiguous()
        return out