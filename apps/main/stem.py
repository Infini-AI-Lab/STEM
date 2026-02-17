# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Callable
import torch
from torch import nn
from torch.nn import functional as F
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
    StemTransformerBlock,
    StemFeedForward,
    FeedForward,
)
from lingua.stem_dist_utils import ParallelEmbedding

from apps.main.transformer import (
    create_causal_mask as llama_create_causal_mask,
    LMTransformerArgs,
    build_fsdp_grouping_plan as llama_build_fsdp_grouping_plan,
    get_no_recompute_ops as llama_get_no_recompute_ops,
    get_num_flop_per_token as llama_get_num_flop_per_token,
)
from apps.main.qwen3 import (
    Qwen3Attention,
    Qwen3LMTransformerArgs,
    create_causal_mask as qwen3_create_causal_mask,
    build_fsdp_grouping_plan as qwen3_build_fsdp_grouping_plan,
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
    get_num_flop_per_token as qwen3_get_num_flop_per_token,
)
from apps.main.olmo3 import (
    OLMo3Attention,
    OLMo3LMTransformerArgs,
    create_causal_mask as olmo3_create_causal_mask,
    build_fsdp_grouping_plan as olmo3_build_fsdp_grouping_plan,
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
    get_num_flop_per_token as olmo3_get_num_flop_per_token,
)


@dataclass
class StemLMTransformerArgs(StemTransformerArgs, LMTransformerArgs):
    pass


# =============================================================================
# Qwen3 STEM transformer block: pre-norm with Qwen3Attention (per-head QK-Norm)
# =============================================================================
class Qwen3StemTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: StemTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert self.n_heads % self.n_kv_heads == 0
        assert args.dim % self.n_heads == 0

        self.attention = Qwen3Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
        )

        ffn_cls = StemFeedForward if layer_idx in args.stem_layers else FeedForward
        self.feed_forward = ffn_cls(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        if layer_idx in args.stem_layers:
            assert args.stem_embedding_dim is not None
            assert args.stem_embedding_dim == self.feed_forward.hidden_dim
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h), y)
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


# =============================================================================
# OLMo3 STEM transformer block: post-norm with OLMo3Attention (full-dim QK-Norm)
# =============================================================================
class OLMo3StemTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: StemTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert self.n_heads % self.n_kv_heads == 0
        assert args.dim % self.n_heads == 0

        self.attention = OLMo3Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
        )

        ffn_cls = StemFeedForward if layer_idx in args.stem_layers else FeedForward
        self.feed_forward = ffn_cls(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        if layer_idx in args.stem_layers:
            assert args.stem_embedding_dim is not None
            assert args.stem_embedding_dim == self.feed_forward.hidden_dim
        # OLMo3: post-norm (applied after sublayers, before residual addition)
        self.post_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_feedforward_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # OLMo3 post-norm: attention(x) -> norm -> residual add
        h = x + self.post_attention_norm(
            self.attention(
                x,
                freq_cis,
                tok_idx=tok_idx,
                mask=mask,
                attn_impl=attn_impl,
            )
        )
        # OLMo3 post-norm: ffn(h, y) -> norm -> residual add
        out = h + self.post_feedforward_norm(self.feed_forward(h, y))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.post_attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.post_feedforward_norm.reset_parameters()


# =============================================================================
# StemTransformer subclasses for Qwen3 and OLMo3
# =============================================================================
class Qwen3StemTransformer(StemTransformer):
    _block_cls = Qwen3StemTransformerBlock


class OLMo3StemTransformer(StemTransformer):
    _block_cls = OLMo3StemTransformerBlock


# =============================================================================
# Base LMTransformer (stem version) — pre-norm, LLaMA attention
# =============================================================================
class LMTransformer(StemTransformer):
    """
    Language model transformer without stem embeddings.
    This is the FSDP-wrappable part of the model.
    """
    _create_causal_mask = staticmethod(llama_create_causal_mask)

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
            else self._create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        
        for i, layer in enumerate(self.layers):
            if i in self.stem_layers:
                if stem_embeddings_fn is not None:
                    y = stem_embeddings_fn(i, token_values)
                    h = layer(h, freq_cis, y=y, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                else:
                    h = layer(h, freq_cis, y=None, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
            else:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                
        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
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


# =============================================================================
# Qwen3 LMTransformer (stem version)
# =============================================================================
class Qwen3LMTransformer(Qwen3StemTransformer):
    """Qwen3 LM transformer with stem support (per-head QK-Norm, pre-norm)."""
    _create_causal_mask = staticmethod(qwen3_create_causal_mask)

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
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = LMTransformer.forward
    reset_parameters = LMTransformer.reset_parameters


# =============================================================================
# OLMo3 LMTransformer (stem version)
# =============================================================================
class OLMo3LMTransformer(OLMo3StemTransformer):
    """OLMo3 LM transformer with stem support (full-dim QK-Norm, post-norm)."""
    _create_causal_mask = staticmethod(olmo3_create_causal_mask)

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
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    forward = LMTransformer.forward
    reset_parameters = LMTransformer.reset_parameters

class StemLMTransformer(nn.Module):
    """
    Higher-level API that decouples stem_embeddings from the rest of the model.
    This allows FSDP wrapping of lm_transformer without affecting stem_embeddings,
    which are managed manually for parallelization and gradient sync.
    """
    # Subclasses (or the registry) override this to use Qwen3/OLMo3 LM transformer
    _lm_transformer_cls = LMTransformer

    def __init__(self, args: StemLMTransformerArgs):
        super().__init__()
        self.args = args
        
        # Create the FSDP-wrappable language model transformer (without stem_embeddings)
        self.lm_transformer = self._lm_transformer_cls(args)
        
        # Create stem_embeddings separately (not part of lm_transformer, manually managed)
        assert args.stem_embedding_dim is not None, "stem_embedding_dim must be provided when using StemLMTransformer"
        # Get device from existing parameters to ensure stem_embeddings are on the same device
        device = next(iter(self.lm_transformer.parameters())).device
        self.stem_embeddings = nn.ModuleList([
            ParallelEmbedding(args.vocab_size, args.stem_embedding_dim, device=device) 
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


# =============================================================================
# Qwen3 & OLMo3 StemLMTransformer variants
# =============================================================================
class Qwen3StemLMTransformer(StemLMTransformer):
    _lm_transformer_cls = Qwen3LMTransformer


class OLMo3StemLMTransformer(StemLMTransformer):
    _lm_transformer_cls = OLMo3LMTransformer


# =============================================================================
# FSDP grouping plans (prefixed with 'lm_transformer.')
# =============================================================================
def _prefix_plan(base_plan):
    return [(f"lm_transformer.{path}", reshard) for path, reshard in base_plan]


def build_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """FSDP plan for LLaMA-based StemLMTransformer."""
    return _prefix_plan(llama_build_fsdp_grouping_plan(model_args))


def build_qwen3_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """FSDP plan for Qwen3-based StemLMTransformer."""
    return _prefix_plan(qwen3_build_fsdp_grouping_plan(model_args))


def build_olmo3_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """FSDP plan for OLMo3-based StemLMTransformer."""
    return _prefix_plan(olmo3_build_fsdp_grouping_plan(model_args))


# =============================================================================
# STEM model registry:
#   model_type -> (stem_lm_cls, args_cls, build_fsdp_plan,
#                  get_no_recompute_ops, get_num_flop_per_token)
# =============================================================================
STEM_MODEL_REGISTRY = {
    "llama": (
        StemLMTransformer, StemLMTransformerArgs,
        build_stem_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops, llama_get_num_flop_per_token,
    ),
    "qwen3": (
        Qwen3StemLMTransformer, StemLMTransformerArgs,
        build_qwen3_stem_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops, qwen3_get_num_flop_per_token,
    ),
    "olmo3": (
        OLMo3StemLMTransformer, StemLMTransformerArgs,
        build_olmo3_stem_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops, olmo3_get_num_flop_per_token,
    ),
}
