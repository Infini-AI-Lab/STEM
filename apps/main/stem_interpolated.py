# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Interpolated STEM architecture for smooth finetuning.

Instead of completely replacing the up-projection (w3) with STEM embeddings,
this architecture interpolates between them:

    up = alpha * w3(x) + (1 - alpha) * stem_embedding
    output = w2(silu(w1(x)) * up)

Alpha is scheduled to decay exponentially from 1 (baseline) to 0 (STEM-only)
during finetuning, ensuring smooth transition and lower initial loss.

The up-projections (w3) remain frozen throughout training, since they are
dispensed with once finetuning completes and we fall back to STEM-only.
"""

import math
from typing import Optional, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask

from xformers.ops import AttentionBias

from lingua.transformer import (
    RMSNorm,
    TiedLinear,
    cross_entropy,
    Attention,
)
from lingua.stem import (
    StemTransformer,
    StemTransformerArgs,
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
from apps.main.stem import StemLMTransformerArgs


# =============================================================================
# Interpolation schedule
# =============================================================================

def compute_interp_alpha(
    step: int,
    half_life: int,
    interp_steps: Optional[int] = None,
) -> float:
    """
    Compute interpolation alpha for the given training step.

    alpha(t) = exp(-t * ln(2) / half_life)

    After ``half_life`` steps, alpha ≈ 0.5.
    After ``5 * half_life`` steps, alpha ≈ 0.03.

    If ``interp_steps`` is set, alpha is clamped to 0 after that many steps
    (i.e., we fall back to pure STEM embedding).

    Args:
        step: Current training step.
        half_life: Number of steps for alpha to halve.
        interp_steps: Total interpolation phase length; alpha = 0 after this.

    Returns:
        alpha in [0, 1].
    """
    if interp_steps is not None and step >= interp_steps:
        return 0.0
    if half_life <= 0:
        return 0.0
    return math.exp(-step * math.log(2) / half_life)


# =============================================================================
# InterpolatedStemFeedForward
# =============================================================================

class InterpolatedStemFeedForward(nn.Module):
    """
    FFN with interpolation between up-projection (w3) and STEM embedding.

    At stem layers the forward computes:
        x1 = w1(x)          -- gate projection
        x3 = w3(x)          -- up-projection  (frozen)
        up = alpha * x3 + (1 - alpha) * y   -- interpolation
        output = w2(SiLU(x1) * up)

    When alpha = 1 this is identical to the baseline FeedForward.
    When alpha = 0 this is identical to StemFeedForward.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)    # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)    # up projection (frozen)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        if y is not None:
            up = alpha * x3 + (1.0 - alpha) * y
        else:
            up = x3  # Behave like standard FeedForward when y is absent
        output = self.w2(F.silu(x1) * up)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


# =============================================================================
# LLaMA-style interpolated STEM transformer block (pre-norm)
# =============================================================================

class InterpolatedStemTransformerBlock(nn.Module):
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

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )

        self._is_stem_layer = layer_idx in args.stem_layers
        ffn_cls = InterpolatedStemFeedForward if self._is_stem_layer else FeedForward
        self.feed_forward = ffn_cls(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        if self._is_stem_layer:
            assert args.stem_embedding_dim is not None
            assert args.stem_embedding_dim == self.feed_forward.hidden_dim

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
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
        if self._is_stem_layer:
            out = h + self.feed_forward(self.ffn_norm(h), y, alpha)
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


# =============================================================================
# Qwen3 interpolated STEM transformer block (pre-norm, per-head QK-Norm)
# =============================================================================

class Qwen3InterpolatedStemTransformerBlock(nn.Module):
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

        self._is_stem_layer = layer_idx in args.stem_layers
        ffn_cls = InterpolatedStemFeedForward if self._is_stem_layer else FeedForward
        self.feed_forward = ffn_cls(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        if self._is_stem_layer:
            assert args.stem_embedding_dim is not None
            assert args.stem_embedding_dim == self.feed_forward.hidden_dim

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
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
        if self._is_stem_layer:
            out = h + self.feed_forward(self.ffn_norm(h), y, alpha)
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


# =============================================================================
# OLMo3 interpolated STEM transformer block (post-norm, full-dim QK-Norm)
# =============================================================================

class OLMo3InterpolatedStemTransformerBlock(nn.Module):
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

        self._is_stem_layer = layer_idx in args.stem_layers
        ffn_cls = InterpolatedStemFeedForward if self._is_stem_layer else FeedForward
        self.feed_forward = ffn_cls(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        if self._is_stem_layer:
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
        alpha: float = 1.0,
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
        if self._is_stem_layer:
            out = h + self.post_feedforward_norm(
                self.feed_forward(h, y, alpha)
            )
        else:
            out = h + self.post_feedforward_norm(
                self.feed_forward(h)
            )
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.post_attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.post_feedforward_norm.reset_parameters()


# =============================================================================
# StemTransformer subclasses using interpolated blocks
# =============================================================================

class InterpolatedStemTransformer(StemTransformer):
    _block_cls = InterpolatedStemTransformerBlock


class Qwen3InterpolatedStemTransformer(StemTransformer):
    _block_cls = Qwen3InterpolatedStemTransformerBlock


class OLMo3InterpolatedStemTransformer(StemTransformer):
    _block_cls = OLMo3InterpolatedStemTransformerBlock


# =============================================================================
# Base InterpolatedLMTransformer (LLaMA attention, pre-norm)
# =============================================================================

class InterpolatedLMTransformer(InterpolatedStemTransformer):
    """
    Language model transformer with interpolated STEM FFN.
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
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
        stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        alpha: float = 1.0,
    ):
        """
        Forward pass with interpolated STEM.

        Args:
            token_values: Input token IDs.
            target: Optional target tokens for loss computation.
            tok_idx: Optional token indices for RoPE.
            mask: Optional attention mask.
            attn_impl: Attention implementation to use.
            stem_embeddings_fn: Callable(layer_idx, token_values) -> stem embeddings.
            alpha: Interpolation weight (1 = full up-projection, 0 = full STEM).
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
                    h = layer(
                        h, freq_cis, y=y, alpha=alpha,
                        tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
                    )
                else:
                    h = layer(
                        h, freq_cis, y=None, alpha=alpha,
                        tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
                    )
            else:
                h = layer(
                    h, freq_cis,
                    tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
                )

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
# Qwen3 InterpolatedLMTransformer (pre-norm, per-head QK-Norm)
# =============================================================================

class Qwen3InterpolatedLMTransformer(Qwen3InterpolatedStemTransformer):
    """Qwen3 LM transformer with interpolated STEM support."""
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

    forward = InterpolatedLMTransformer.forward
    reset_parameters = InterpolatedLMTransformer.reset_parameters


# =============================================================================
# OLMo3 InterpolatedLMTransformer (post-norm, full-dim QK-Norm)
# =============================================================================

class OLMo3InterpolatedLMTransformer(OLMo3InterpolatedStemTransformer):
    """OLMo3 LM transformer with interpolated STEM support."""
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

    forward = InterpolatedLMTransformer.forward
    reset_parameters = InterpolatedLMTransformer.reset_parameters


# =============================================================================
# InterpolatedStemLMTransformer: top-level wrapper (stem_embeddings + LM)
# =============================================================================

class InterpolatedStemLMTransformer(nn.Module):
    """
    Higher-level API that decouples stem_embeddings from the rest of the model.
    This allows FSDP wrapping of lm_transformer without affecting stem_embeddings,
    which are managed manually for parallelization and gradient sync.

    Also manages the interpolation alpha that controls the transition from
    baseline up-projection to STEM embedding.
    """
    _lm_transformer_cls = InterpolatedLMTransformer

    def __init__(self, args: StemLMTransformerArgs):
        super().__init__()
        self.args = args

        # Create the FSDP-wrappable language model transformer
        self.lm_transformer = self._lm_transformer_cls(args)

        # Create stem_embeddings separately (managed manually, not FSDP-wrapped)
        assert args.stem_embedding_dim is not None, (
            "stem_embedding_dim must be provided when using InterpolatedStemLMTransformer"
        )
        device = next(iter(self.lm_transformer.parameters())).device
        self.stem_embeddings = nn.ModuleList([
            ParallelEmbedding(args.vocab_size, args.stem_embedding_dim, device=device)
            for _ in range(len(self.lm_transformer.stem_layers))
        ])

        # Mapping: layer index -> stem_embeddings index
        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(self.lm_transformer.stem_layers)
        }

        # Interpolation alpha (updated by the training loop each step)
        self._alpha = 1.0

    # ---- Interpolation alpha management ------------------------------------

    def set_alpha(self, alpha: float):
        """Set interpolation alpha (1 = full up-projection, 0 = full STEM)."""
        self._alpha = float(alpha)

    def get_alpha(self) -> float:
        return self._alpha

    # ---- Freeze up-projections (w3) ----------------------------------------

    def freeze_up_projections(self):
        """
        Freeze w3 (up-projection) parameters in all stem layers.
        Must be called after checkpoint loading and before optimizer creation.
        Returns the number of parameters frozen.
        """
        import logging
        logger = logging.getLogger()
        count = 0
        for layer_idx in self.lm_transformer.stem_layers:
            layer = self.lm_transformer.layers[layer_idx]
            ffn = layer.feed_forward
            if hasattr(ffn, 'w3'):
                for param in ffn.w3.parameters():
                    param.requires_grad = False
                    count += 1
                logger.info(
                    f"Frozen w3 at layer {layer_idx} "
                    f"({ffn.w3.weight.shape})"
                )
        logger.info(f"Total w3 parameters frozen: {count}")
        return count

    # ---- Forward -----------------------------------------------------------

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        """
        Forward pass that coordinates between lm_transformer and stem_embeddings,
        passing the current interpolation alpha.
        """
        alpha = self._alpha

        def stem_embeddings_fn(
            layer_idx: int, token_values: torch.Tensor
        ) -> torch.Tensor:
            stem_idx = self._layer_to_stem_idx[layer_idx]
            return self.stem_embeddings[stem_idx](token_values)

        return self.lm_transformer(
            token_values=token_values,
            target=target,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
            stem_embeddings_fn=stem_embeddings_fn,
            alpha=alpha,
        )

    # ---- Initialization ----------------------------------------------------

    @torch.no_grad()
    def reset_stem_embeddings(self):
        """Reset parameters of stem_embeddings."""
        import logging
        logger = logging.getLogger()
        zero_reset = getattr(self.args, "stem_embeddings_zero_reset", False)
        for i, embedding in enumerate(self.stem_embeddings):
            weight_device = embedding.weight.device
            if weight_device.type == "meta":
                logger.warning(
                    f"stem_embeddings[{i}].weight is still on meta device, "
                    f"skipping initialization"
                )
                continue
            if zero_reset:
                embedding.weight.zero_()
                if embedding.weight.numel() > 0:
                    logger.debug(
                        f"stem_embeddings[{i}].weight zero-initialized, device={weight_device}"
                    )
                continue
            embedding.reset_parameters()
            if embedding.weight.numel() > 0:
                weight_norm = embedding.weight.norm().item()
                is_zero = (embedding.weight.abs().max() == 0).item()
                if is_zero:
                    logger.error(
                        f"stem_embeddings[{i}].weight is still all zeros "
                        f"after reset_parameters()!"
                    )
                else:
                    logger.debug(
                        f"stem_embeddings[{i}].weight initialized: "
                        f"norm={weight_norm:.6f}, device={weight_device}"
                    )

    def init_weights(self):
        """Initialize weights of the language model transformer and stem_embeddings."""
        self.lm_transformer.init_weights()
        self.reset_stem_embeddings()

    # ---- Property delegation for compatibility -----------------------------

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
# Qwen3 & OLMo3 InterpolatedStemLMTransformer variants
# =============================================================================

class Qwen3InterpolatedStemLMTransformer(InterpolatedStemLMTransformer):
    _lm_transformer_cls = Qwen3InterpolatedLMTransformer


class OLMo3InterpolatedStemLMTransformer(InterpolatedStemLMTransformer):
    _lm_transformer_cls = OLMo3InterpolatedLMTransformer


# =============================================================================
# FSDP grouping plans (prefixed with 'lm_transformer.')
# =============================================================================

def _prefix_plan(base_plan):
    return [(f"lm_transformer.{path}", reshard) for path, reshard in base_plan]


def build_interp_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """FSDP plan for LLaMA-based InterpolatedStemLMTransformer."""
    return _prefix_plan(llama_build_fsdp_grouping_plan(model_args))


def build_qwen3_interp_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """FSDP plan for Qwen3-based InterpolatedStemLMTransformer."""
    return _prefix_plan(qwen3_build_fsdp_grouping_plan(model_args))


def build_olmo3_interp_stem_lm_fsdp_grouping_plan(model_args: StemLMTransformerArgs):
    """FSDP plan for OLMo3-based InterpolatedStemLMTransformer."""
    return _prefix_plan(olmo3_build_fsdp_grouping_plan(model_args))


# =============================================================================
# Interpolated STEM model registry:
#   model_type -> (interp_stem_lm_cls, args_cls, build_fsdp_plan,
#                  get_no_recompute_ops, get_num_flop_per_token)
# =============================================================================

INTERP_STEM_MODEL_REGISTRY = {
    "llama": (
        InterpolatedStemLMTransformer, StemLMTransformerArgs,
        build_interp_stem_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops, llama_get_num_flop_per_token,
    ),
    "qwen3": (
        Qwen3InterpolatedStemLMTransformer, StemLMTransformerArgs,
        build_qwen3_interp_stem_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops, qwen3_get_num_flop_per_token,
    ),
    "olmo3": (
        OLMo3InterpolatedStemLMTransformer, StemLMTransformerArgs,
        build_olmo3_interp_stem_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops, olmo3_get_num_flop_per_token,
    ),
}

