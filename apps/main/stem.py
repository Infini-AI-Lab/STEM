# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from xformers.ops import AttentionBias
from lingua.transformer import (
    RMSNorm,
    TiedLinear,
    cross_entropy,
)
from lingua.stem import (
    StemTransformer,
    StemTransformerArgs,
)
from lingua.stem_dist_utils import ParallelEmbedding

from apps.main.transformer import create_causal_mask, LMTransformerArgs, build_fsdp_grouping_plan


@dataclass
class StemLMTransformerArgs(StemTransformerArgs, LMTransformerArgs):
    pass
    
    
class StemLMTransformer(StemTransformer):
    def __init__(self, args: StemLMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )
            
        # dimension parallel impl of stem_embeddings
        self.stem_embeddings = nn.ModuleList([
            ParallelEmbedding(args.vocab_size, args.stem_embedding_dim) for _ in range(len(self.stem_layers))
        ])
        
    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)
        
        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        
        j = 0
        for i, layer in enumerate(self.layers):
            if i in self.stem_layers:
                y = self.stem_embeddings[j](token_values)
                h = layer(h, freq_cis, y=y, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                j += 1
            else:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                
        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits
                
    