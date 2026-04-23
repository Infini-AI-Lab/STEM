# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Minimal toy LM + helpers kept for ``stem_longcat_ngram_eval`` checkpoints only.

from __future__ import annotations

import torch
import torch.nn as nn

from lingua.longcat_ngram import LongcatNgramConfig, NgramEmbedding


class ToyLongcatNgramLM(nn.Module):
    def __init__(self, cfg: LongcatNgramConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.ngram = NgramEmbedding(cfg)
        n_shards = cfg.emb_split_num * (cfg.emb_neighbor_num - 1)
        self._ngram_emb_dim = cfg.hidden_size // n_shards
        self.post_projs = nn.ModuleList(
            nn.Linear(self._ngram_emb_dim, cfg.hidden_size, bias=False)
            for _ in range(n_shards)
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        ngram_cat = self.ngram(input_ids, ngram_context=None)
        for chunk, proj in zip(
            ngram_cat.split(self._ngram_emb_dim, dim=-1), self.post_projs
        ):
            h = h + proj(chunk)
        h = h / (1 + len(self.post_projs))
        return self.lm_head(h)


def _synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    g: torch.Generator,
) -> torch.Tensor:
    x = torch.randint(
        low=2,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        generator=g,
    )
    x[:, 0] = pad_id
    return x
