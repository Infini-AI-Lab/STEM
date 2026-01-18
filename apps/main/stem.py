# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Callable
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
from lingua.stem_dist_utils import ParallelEmbedding, is_stem_process_group_initialized

from apps.main.transformer import create_causal_mask, LMTransformerArgs, build_fsdp_grouping_plan


@dataclass
class StemLMTransformerArgs(StemTransformerArgs, LMTransformerArgs):
    pass


class LMTransformer(StemTransformer):
    """
    Language model transformer without stem embeddings.
    This is the FSDP-wrappable part of the model.
    """
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
        
    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
        stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Forward pass of the language model transformer.
        
        Args:
            token_values: Input token IDs
            target: Optional target tokens for loss computation
            tok_idx: Optional token indices for RoPE
            mask: Optional attention mask
            attn_impl: Attention implementation to use
            stem_embeddings_fn: Optional callable that takes (layer_idx, token_values) and returns
                              stem embeddings for that layer. If None, stem layers will not receive
                              stem embeddings (y=None).
        """
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)
        
        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        
        for i, layer in enumerate(self.layers):
            if i in self.stem_layers:
                if stem_embeddings_fn is not None:
                    y = stem_embeddings_fn(i, token_values)
                    h = layer(h, freq_cis, y=y, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                else:
                    # If no stem_embeddings_fn provided, pass y=None (may cause errors in StemFeedForward)
                    h = layer(h, freq_cis, y=None, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
            else:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                
        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

class StemLMTransformer(nn.Module):
    """
    Higher-level API that decouples stem_embeddings from the rest of the model.
    This allows FSDP wrapping of lm_transformer without affecting stem_embeddings,
    which are managed manually for parallelization and gradient sync.
    """
    def __init__(self, args: StemLMTransformerArgs):
        super().__init__()
        self.args = args
        
        # Create the FSDP-wrappable language model transformer (without stem_embeddings)
        self.lm_transformer = LMTransformer(args)
        
        # Create stem_embeddings separately (not part of lm_transformer, manually managed)
        assert args.stem_embedding_dim is not None, "stem_embedding_dim must be provided when using StemLMTransformer"
        # Get device from existing parameters to ensure stem_embeddings are on the same device
        device = self.lm_transformer.output.weight.device
        
        # Check if stem process group is initialized
        if is_stem_process_group_initialized():
            self.stem_embeddings = nn.ModuleList([
                ParallelEmbedding(args.vocab_size, args.stem_embedding_dim, device=device) 
                for _ in range(len(self.lm_transformer.stem_layers))
            ])
        else:
            self.stem_embeddings = nn.ModuleList([
                nn.Embedding(args.vocab_size, args.stem_embedding_dim).to(device) 
                for _ in range(len(self.lm_transformer.stem_layers))
            ])
        
        # Create mapping from layer index to stem_embeddings index
        self._layer_to_stem_idx = {
            layer_idx: stem_idx 
            for stem_idx, layer_idx in enumerate(self.lm_transformer.stem_layers)
        }
        
    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        """
        Forward pass that coordinates between lm_transformer and stem_embeddings.
        """
        def stem_embeddings_fn(layer_idx: int, token_values: torch.Tensor) -> torch.Tensor:
            """Get stem embeddings for a given layer index."""
            stem_idx = self._layer_to_stem_idx[layer_idx]
            return self.stem_embeddings[stem_idx](token_values)
        
        return self.lm_transformer(
            token_values=token_values,
            target=target,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
            stem_embeddings_fn=stem_embeddings_fn,
        )
        
    @torch.no_grad()
    def reset_stem_embeddings(self):
        """Reset parameters of stem_embeddings."""
        import logging
        logger = logging.getLogger()
        for i, embedding in enumerate(self.stem_embeddings):
            # Verify device before resetting
            weight_device = embedding.weight.device
            if weight_device.type == "meta":
                logger.warning(
                    f"stem_embeddings[{i}].weight is still on meta device, skipping initialization"
                )
                continue
            embedding.reset_parameters()
            # Verify initialization succeeded
            if embedding.weight.numel() > 0:
                weight_norm = embedding.weight.norm().item()
                is_zero = (embedding.weight.abs().max() == 0).item()
                if is_zero:
                    logger.error(
                        f"stem_embeddings[{i}].weight is still all zeros after reset_parameters()!"
                    )
                else:
                    logger.debug(
                        f"stem_embeddings[{i}].weight initialized: norm={weight_norm:.6f}, device={weight_device}"
                    )
    
    def init_weights(self):
        """Initialize weights of the language model transformer and stem_embeddings."""
        self.lm_transformer.init_weights()
        # Initialize stem_embeddings after lm_transformer to ensure proper initialization
        self.reset_stem_embeddings()
    
    # Delegate other methods/properties to lm_transformer for compatibility
    @property
    def layers(self):
        return self.lm_transformer.layers
    
    @property
    def stem_layers(self):
        return self.lm_transformer.stem_layers
    
    @property
    def rope_embeddings(self):
        return self.lm_transformer.rope_embeddings
    
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
        """Delegate gradient sync requirement to lm_transformer (FSDP-wrapped)."""
        if hasattr(self.lm_transformer, "set_requires_gradient_sync"):
            self.lm_transformer.set_requires_gradient_sync(requires_sync)


def build_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """
    Build FSDP grouping plan for StemLMTransformer.
    This prefixes all paths with 'lm_transformer.' to wrap only the language model
    transformer, excluding stem_embeddings which are managed manually.
    """
    base_plan = build_fsdp_grouping_plan(model_args)
    # Prefix all paths with 'lm_transformer.' to target the submodule
    return [(f"lm_transformer.{path}", reshard_after_forward) for path, reshard_after_forward in base_plan]
    
    