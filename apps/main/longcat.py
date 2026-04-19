# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from apps.main.transformer import (
    LMTransformer,
    LMTransformerArgs,
    build_fsdp_grouping_plan as llama_build_fsdp_grouping_plan,
    create_causal_mask,
)
from lingua.longcat_ngram import LongcatNgramConfig, NgramEmbedding
from lingua.transformer import BaseTransformer, cross_entropy


@dataclass
class LongcatLMTransformerArgs(LMTransformerArgs, LongcatNgramConfig):
    """If true, :meth:`LongcatLMTransformer.reset_ngram_embeddings` zero-fills tables."""

    ngram_embeddings_zero_reset: bool = False

    def __post_init__(self) -> None:
        LongcatNgramConfig.__post_init__(self)
        if int(self.dim) != int(self.hidden_size):
            raise ValueError(
                f"dim ({self.dim}) must equal hidden_size ({self.hidden_size}) "
                "for Longcat n-gram embedding layout."
            )


def _lm_prefix_plan(
    base_plan: List[Tuple[str, bool]],
) -> List[Tuple[str, bool]]:
    return [(f"lm_transformer.{path}", reshard) for path, reshard in base_plan]


def build_longcat_lm_fsdp_grouping_plan(
    model_args: LongcatLMTransformerArgs,
) -> List[Tuple[str, bool]]:
    """FSDP leaf groups under ``lm_transformer`` plus ``ngram_fused`` (Longcat path)."""
    plan: List[Tuple[str, bool]] = []
    for path, r in _lm_prefix_plan(llama_build_fsdp_grouping_plan(model_args)):
        plan.append((path, r))
        if path == "lm_transformer.tok_embeddings":
            plan.append(("lm_transformer.ngram_fused", False))
    return plan


class NgramLMTransformer(LMTransformer):
    """
    Causal LM whose input is base :class:`torch.nn.Embedding` plus Longcat-style
    n-gram shards: concatenated shard vectors are mapped with one
    :class:`torch.nn.Linear` (``dim`` → ``dim``) whose weight is the horizontal
    stack of per-shard maps, equivalent to summing separate projections into the
    token vector, then scaled as in the Longcat reference
    ``1 / (1 + k * (n - 1))``.

    Pass ``ngram_embeddings`` as the concatenated shard tensor from
    :class:`lingua.longcat_ngram.NgramEmbedding` (shape ``(..., dim)``), or
    ``None`` to use only token embeddings.
    """

    def __init__(self, args: LongcatLMTransformerArgs):
        super().__init__(args)
        n = int(args.emb_neighbor_num)
        k = int(args.emb_split_num)
        self._ngram_num_shards = k * (n - 1)
        if args.dim % self._ngram_num_shards != 0:
            raise ValueError(
                f"dim ({args.dim}) must be divisible by k*(n-1) = {self._ngram_num_shards}"
            )
        # One Linear(dim, dim): weight columns are the k*(n-1) shard matrices
        # concatenated, so concat(shard_vecs) @ W^T == sum_i proj_i(shard_i).
        self.ngram_fused = nn.Linear(args.dim, args.dim, bias=False)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
        ngram_embeddings: Optional[torch.Tensor] = None,
    ):
        _, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)
        if ngram_embeddings is not None:
            ngram_cat = ngram_embeddings
            if ngram_cat.shape[-1] != self.dim:
                raise RuntimeError(
                    f"ngram embedding last dim {ngram_cat.shape[-1]} != model dim {self.dim}"
                )
            h = h + self.ngram_fused(ngram_cat)
            h = h / (1 + self._ngram_num_shards)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = BaseTransformer.forward(
            self, h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl
        )

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        return logits

    def reset_parameters(self, init_std=None):
        super().reset_parameters(init_std)
        init_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.ngram_fused.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )


class LongcatLMTransformer(nn.Module):
    """
    Wraps :class:`NgramLMTransformer` with a separate :class:`NgramEmbedding` so
    FSDP or custom parallel plans can treat n-gram tables independently of the
    backbone if desired.
    """

    _lm_transformer_cls = NgramLMTransformer

    def __init__(self, args: LongcatLMTransformerArgs):
        super().__init__()
        self.args = args
        self.lm_transformer = self._lm_transformer_cls(args)
        self.ngram_embeddings = NgramEmbedding(args)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        ngram_context: Optional[torch.Tensor] = None
        if hasattr(self, "ngram_cache"):
            ngram_context = self.ngram_cache.update(token_values)

        ngram_emb = self.ngram_embeddings(token_values, ngram_context)

        return self.lm_transformer(
            token_values=token_values,
            target=target,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
            ngram_embeddings=ngram_emb,
        )

    def init_weights(self):
        self.lm_transformer.init_weights()
        for emb in self.ngram_embeddings.embedders:
            emb.reset_parameters()

    @torch.no_grad()
    def reset_ngram_embeddings(self):
        """Re-init or zero n-gram :class:`~lingua.stem_dist_utils.VocabParallelEmbedding` tables."""
        if getattr(self.args, "ngram_embeddings_zero_reset", False):
            for emb in self.ngram_embeddings.embedders:
                emb.weight.zero_()
        else:
            for emb in self.ngram_embeddings.embedders:
                emb.reset_parameters()

    @property
    def rope_embeddings(self):
        return self.lm_transformer.rope_embeddings

    @property
    def layers(self):
        return self.lm_transformer.layers

    @property
    def max_seqlen(self):
        return self.lm_transformer.max_seqlen

    @property
    def dim(self):
        return self.lm_transformer.dim

    @property
    def weight_tying(self):
        return self.lm_transformer.weight_tying

    @property
    def sliding_window(self):
        return self.lm_transformer.sliding_window

    def set_requires_gradient_sync(self, requires_sync: bool):
        if hasattr(self.lm_transformer, "set_requires_gradient_sync"):
            self.lm_transformer.set_requires_gradient_sync(requires_sync)

    def reset_parameters(self, init_std=None):
        self.lm_transformer.reset_parameters(init_std)
        for emb in self.ngram_embeddings.embedders:
            emb.reset_parameters()
