# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Callable, Dict
import math
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
from lingua.stem_dist_utils import (
    ParallelEmbedding,
    get_stem_model_parallel_world_size,
    get_stem_model_parallel_rank,
)

from apps.main.transformer import create_causal_mask, LMTransformerArgs, build_fsdp_grouping_plan


@dataclass
class StemLMTransformerArgs(StemTransformerArgs, LMTransformerArgs):
    init_type: str = "normal"


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
    def reset_stem_embeddings(
        self, 
        pretrained_w3_weights: Optional[Dict[int, torch.Tensor]] = None,
        pretrained_w3_stats: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ):
        """Reset parameters of stem_embeddings."""
        import logging
        logger = logging.getLogger()

        # Reverse mapping: stem_idx -> layer_idx
        stem_idx_to_layer = {v: k for k, v in self._layer_to_stem_idx.items()}

        for i, embedding in enumerate(self.stem_embeddings):
            # Verify device before resetting
            weight_device = embedding.weight.device
            if weight_device.type == "meta":
                logger.warning(
                    f"stem_embeddings[{i}].weight is still on meta device, skipping initialization"
                )
                continue

            if self.args.init_type == "normal":
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

            elif self.args.init_type == "zero":
                nn.init.zeros_(embedding.weight)

            elif self.args.init_type == "scaled_normal":
                std = 1.0 / math.sqrt(self.args.dim)
                nn.init.normal_(embedding.weight, mean=0.0, std=std)

            elif self.args.init_type == "uniform":
                nn.init.uniform_(embedding.weight, a=-0.02, b=0.02)

            elif self.args.init_type == "pretrained_stats":
                # ============================================================
                # Initialize stem embeddings from per-neuron statistics of
                # the pretrained product E_tok @ W3^T.
                #
                # Motivation:
                #   `from_lm_transformer` copies the exact matrix E_tok @ W3^T,
                #   which is a deterministic first-order approximation. This
                #   strategy instead draws each token's embedding from:
                #
                #       stem_emb[t, j] ~ N(μ_j, σ_j)
                #
                #   where μ_j and σ_j are the mean and std of the j-th neuron's
                #   response across the full vocabulary. This preserves the
                #   per-neuron activation scale (critical for stable gating and
                #   w2 input magnitudes) while breaking token-level correlations
                #   for better optimization.
                #
                # When to prefer over from_lm_transformer:
                #   - When the approximation h ≈ tok_emb is too coarse (deeper
                #     stem layers where attention has significantly mixed tokens)
                #   - When you want stochastic diversity in the init
                #   - When serializing/transferring the full (V, d_ffn) matrix
                #     per layer is too expensive (stats are only 2 × d_ffn)
                #
                # Stats shape:
                #   mean: (d_ffn,) — per-neuron mean across vocab
                #   std:  (d_ffn,) — per-neuron std across vocab
                # ============================================================

                if pretrained_w3_stats is None:
                    raise ValueError(
                        "init_type='pretrained_stats' requires `pretrained_w3_stats` — "
                        "a dict mapping layer_idx -> {'mean': Tensor(d_ffn,), "
                        "'std': Tensor(d_ffn,)}.\n"
                        "Use extract_pretrained_w3_stats() to compute from the pretrained "
                        "checkpoint:\n"
                        "  from apps.main.stem_lm_transformer import extract_pretrained_w3_stats\n"
                        "  stats = extract_pretrained_w3_stats(state_dict, stem_layer_indices)\n"
                        "Then pass to reset_stem_embeddings(pretrained_w3_stats=stats)."
                    )

                layer_idx = stem_idx_to_layer[i]

                if layer_idx not in pretrained_w3_stats:
                    raise KeyError(
                        f"pretrained_w3_stats missing entry for layer {layer_idx} "
                        f"(stem_embeddings[{i}]). Available keys: "
                        f"{sorted(pretrained_w3_stats.keys())}"
                    )

                layer_stats = pretrained_w3_stats[layer_idx]

                # Validate stats structure
                for required_key in ("mean", "std"):
                    if required_key not in layer_stats:
                        raise KeyError(
                            f"pretrained_w3_stats[{layer_idx}] missing '{required_key}'. "
                            f"Available keys: {list(layer_stats.keys())}. "
                            f"Expected: {{'mean': Tensor(d_ffn,), 'std': Tensor(d_ffn,)}}"
                        )

                full_mean = layer_stats["mean"]  # (d_ffn,)
                full_std = layer_stats["std"]    # (d_ffn,)

                # Validate shapes against stem_embedding_dim
                expected_dim = self.args.stem_embedding_dim
                if full_mean.shape[0] != expected_dim:
                    raise ValueError(
                        f"Stats mean has dim {full_mean.shape[0]} but "
                        f"stem_embedding_dim={expected_dim} for layer {layer_idx}."
                    )
                if full_std.shape[0] != expected_dim:
                    raise ValueError(
                        f"Stats std has dim {full_std.shape[0]} but "
                        f"stem_embedding_dim={expected_dim} for layer {layer_idx}."
                    )

                # Handle ParallelEmbedding dimension sharding.
                # ParallelEmbedding shards along the EMBEDDING dim (d_ffn),
                # so each rank holds (V, d_ffn / world_size).
                local_dim = embedding.weight.shape[1]
                world_size = get_stem_model_parallel_world_size()
                rank = get_stem_model_parallel_rank()

                if world_size > 1:
                    shard_start = rank * local_dim
                    shard_end = shard_start + local_dim
                    local_mean = full_mean[shard_start:shard_end].to(
                        device=weight_device, dtype=torch.float32
                    )
                    local_std = full_std[shard_start:shard_end].to(
                        device=weight_device, dtype=torch.float32
                    )
                else:
                    local_mean = full_mean.to(device=weight_device, dtype=torch.float32)
                    local_std = full_std.to(device=weight_device, dtype=torch.float32)

                # Draw from N(0,1) then scale/shift per neuron
                embedding.weight.data.normal_(mean=0.0, std=1.0)
                w_f32 = embedding.weight.data.float()
                w_f32.mul_(local_std.unsqueeze(0))    # (V, d_local) * (1, d_local)
                w_f32.add_(local_mean.unsqueeze(0))   # (V, d_local) + (1, d_local)
                embedding.weight.data.copy_(w_f32.to(embedding.weight.dtype))

                logger.info(
                    f"stem_embeddings[{i}] (layer {layer_idx}): pretrained_stats init, "
                    f"mean_range=[{local_mean.min().item():.4f}, {local_mean.max().item():.4f}], "
                    f"std_range=[{local_std.min().item():.4f}, {local_std.max().item():.4f}]"
                )

            elif self.args.init_type == "from_lm_transformer":
                # ============================================================
                # Initialize stem embeddings as: E_stem = E_tok @ W3^T
                #
                # Motivation:
                #   In SwiGLU FFN, the computation is:
                #     y = W2 @ (SiLU(W1 @ x) * (W3 @ x))
                #   STEM replaces the token-dependent up-projection W3 @ x
                #   with a per-token embedding lookup: stem_emb[token_id].
                #
                #   At early layers, the residual stream h ≈ tok_emb[token_id]
                #   (before significant transformation by attention), so:
                #     W3 @ h ≈ W3 @ tok_emb[token_id]
                #
                #   Pre-computing this product for all tokens gives a
                #   functionally-equivalent initialization that preserves
                #   the pretrained model's behavior at the point of
                #   architectural surgery.
                #
                # Shapes:
                #   E_tok  : (V, d_model)    — pretrained token embeddings
                #   W3     : (d_ffn, d_model) — pretrained up-projection
                #   E_stem : (V, d_ffn)      — stem embedding to initialize
                #
                # Note: This is a first-order approximation. It does not
                # account for attention, layer norms, or positional
                # encodings applied before the FFN, but provides a strong
                # initialization that the model can refine during training.
                # ============================================================
                if pretrained_w3_weights is None:
                    raise ValueError(
                        "init_type='from_lm_transformer' requires `pretrained_w3_weights` — "
                        "a dict mapping layer_idx -> w3 weight tensor (d_ffn, d_model). "
                        "Extract these from the pretrained checkpoint before STEM conversion:\n"
                        "  pretrained_w3 = {}\n"
                        "  for idx in stem_layer_indices:\n"
                        "      pretrained_w3[idx] = model.layers[idx].feed_forward.w3.weight.data.clone()\n"
                        "Then pass to reset_stem_embeddings(pretrained_w3_weights=pretrained_w3)."
                    )

                layer_idx = stem_idx_to_layer[i]

                if layer_idx not in pretrained_w3_weights:
                    raise KeyError(
                        f"pretrained_w3_weights missing entry for layer {layer_idx} "
                        f"(stem_embeddings[{i}]). Available keys: "
                        f"{sorted(pretrained_w3_weights.keys())}"
                    )

                w3_weight = pretrained_w3_weights[layer_idx]
                tok_emb_weight = self.lm_transformer.tok_embeddings.weight.data

                V, d_model = tok_emb_weight.shape
                d_ffn, d_model_w3 = w3_weight.shape

                if d_model != d_model_w3:
                    raise ValueError(
                        f"Dimension mismatch: tok_embeddings has d_model={d_model} but "
                        f"w3 for layer {layer_idx} has input dim={d_model_w3}."
                    )

                expected_stem_dim = self.args.stem_embedding_dim
                if d_ffn != expected_stem_dim:
                    raise ValueError(
                        f"w3 output dim ({d_ffn}) != stem_embedding_dim ({expected_stem_dim}) "
                        f"for layer {layer_idx}. These must match."
                    )

                # Handle ParallelEmbedding dimension sharding.
                # --- Compute E_tok @ W3^T in float32 for numerical precision ---
                compute_device = weight_device if weight_device.type != "meta" else torch.device("cpu")
                tok_f32 = tok_emb_weight.to(device=compute_device, dtype=torch.float32)
                w3_f32 = w3_weight.to(device=compute_device, dtype=torch.float32)
                full_init = tok_f32 @ w3_f32.t()  # (V, d_ffn)

                # --- Load via ParallelEmbedding.load_full_weight() ---
                # Correctly handles column-shard slicing for the current
                # STEM model-parallel rank.
                embedding.load_full_weight(full_init)

                # Free large intermediates immediately
                del tok_f32, w3_f32, full_init

            else: 
                raise ValueError(
                    f"Invalid init_type: {self.args.init_type}. Must be one of: "
                    "normal, zero, scaled_normal, uniform, pretrained_stats, from_lm_transformer"
                )

            # Verify initialization succeeded
            if embedding.weight.numel() > 0:
                weight_norm = embedding.weight.norm().item()
                is_zero = (embedding.weight.abs().max() == 0).item()
                if is_zero and self.args.init_type != "zero":
                    logger.error(
                        f"stem_embeddings[{i}].weight is all zeros after init "
                        f"(init_type={self.args.init_type})!"
                    )
                else:
                    logger.info(
                        f"stem_embeddings[{i}].weight initialized: "
                        f"init_type={self.args.init_type}, norm={weight_norm:.6f}, "
                        f"device={weight_device}"
                    )
    
    def init_weights(
        self,
        pretrained_w3_weights: Optional[Dict[int, torch.Tensor]] = None,
        pretrained_w3_stats: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ):
        """Initialize weights of the language model transformer and stem_embeddings.

        Args:
            pretrained_w3_weights: Required when init_type='from_lm_transformer'.
                Dict mapping layer_idx -> w3 weight tensor (d_ffn, d_model).
            pretrained_w3_stats: Required when init_type='pretrained_stats'.
                Dict mapping layer_idx -> {'mean': Tensor, 'std': Tensor}.
        """
        self.lm_transformer.init_weights()
        self.reset_stem_embeddings(
            pretrained_w3_weights=pretrained_w3_weights,
            pretrained_w3_stats=pretrained_w3_stats,
        )
    
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
    
# =============================================================================
# Utility: Extract pretrained w3 weights (exact) for from_lm_transformer init
# =============================================================================

def extract_pretrained_w3_weights(
    pretrained_state_dict: Dict[str, torch.Tensor],
    stem_layer_indices: List[int],
    w3_key_template: str = "layers.{layer_idx}.feed_forward.w3.weight",
) -> Dict[int, torch.Tensor]:
    """
    Extract pretrained w3 (up-projection) weights from a checkpoint state_dict.
    
    Call this BEFORE constructing StemLMTransformer, on the original pretrained 
    checkpoint, to capture the w3 weights that will be removed during STEM conversion.
    
    Args:
        pretrained_state_dict: State dict from the pretrained (non-STEM) checkpoint.
        stem_layer_indices: List of layer indices that will become STEM layers.
        w3_key_template: Format string for the w3 weight key in the state dict.
            
    Returns:
        Dict mapping layer_idx -> w3 weight tensor (d_ffn, d_model), cloned and detached.
    """
    w3_weights = {}
    missing_keys = []

    for layer_idx in stem_layer_indices:
        key = w3_key_template.format(layer_idx=layer_idx)
        if key in pretrained_state_dict:
            w3_weights[layer_idx] = pretrained_state_dict[key].clone().detach()
        else:
            missing_keys.append((layer_idx, key))

    if missing_keys:
        available_w3_keys = [k for k in pretrained_state_dict.keys() if "w3" in k]
        raise KeyError(
            f"Could not find w3 weights for layers: "
            f"{[idx for idx, _ in missing_keys]}. "
            f"Tried keys: {[k for _, k in missing_keys]}. "
            f"Available w3-related keys in state_dict: {available_w3_keys}"
        )

    return w3_weights


# =============================================================================
# Utility: Extract pretrained w3 statistics for pretrained_stats init
# =============================================================================

def extract_pretrained_w3_stats(
    pretrained_state_dict: Dict[str, torch.Tensor],
    stem_layer_indices: List[int],
    tok_emb_key: str = "tok_embeddings.weight",
    w3_key_template: str = "layers.{layer_idx}.feed_forward.w3.weight",
    compute_device: Optional[torch.device] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract per-neuron activation statistics of E_tok @ W3^T from a pretrained
    checkpoint, for use with init_type='pretrained_stats'.

    Computes the product E_tok @ W3^T for each stem layer and returns
    per-output-dimension (per-neuron) mean and std across the vocabulary.
    Peak memory is O(V × d_ffn) — one layer at a time.

    Args:
        pretrained_state_dict: State dict from the pretrained (non-STEM) checkpoint.
            Must contain the token embedding and w3 weights for each stem layer.
        stem_layer_indices: Layer indices that will become STEM layers.
        tok_emb_key: Key for the token embedding weight in the state dict.
        w3_key_template: Format string for the w3 weight key.
        compute_device: Device for the matmul. Defaults to CPU to avoid GPU OOM
            on large vocabularies. The resulting stats are tiny (2 × d_ffn per layer).

    Returns:
        Dict mapping layer_idx -> {"mean": Tensor(d_ffn,), "std": Tensor(d_ffn,)}.
        All tensors are float32 on CPU.

    Example:
        >>> state_dict = torch.load("pretrained_checkpoint.pt")
        >>> stats = extract_pretrained_w3_stats(state_dict, [1, 3, 5, 7])
        >>> model = StemLMTransformer(args)
        >>> model.reset_stem_embeddings(pretrained_w3_stats=stats)
    """
    import logging
    logger = logging.getLogger()

    if compute_device is None:
        compute_device = torch.device("cpu")

    # Validate tok_emb key
    if tok_emb_key not in pretrained_state_dict:
        raise KeyError(
            f"Token embedding key '{tok_emb_key}' not found in state dict. "
            f"Available keys containing 'emb': "
            f"{[k for k in pretrained_state_dict if 'emb' in k.lower()]}"
        )

    tok_emb = pretrained_state_dict[tok_emb_key]  # (V, d_model)
    tok_emb_f32 = tok_emb.to(device=compute_device, dtype=torch.float32)

    stats: Dict[int, Dict[str, torch.Tensor]] = {}
    missing_keys = []

    for layer_idx in stem_layer_indices:
        key = w3_key_template.format(layer_idx=layer_idx)
        if key not in pretrained_state_dict:
            missing_keys.append((layer_idx, key))
            continue

        w3_weight = pretrained_state_dict[key]  # (d_ffn, d_model)
        w3_f32 = w3_weight.to(device=compute_device, dtype=torch.float32)

        # E_tok @ W3^T → (V, d_ffn): the per-token up-projection activations
        products = tok_emb_f32 @ w3_f32.t()

        # Per-neuron statistics across the vocabulary dimension
        stats[layer_idx] = {
            "mean": products.mean(dim=0).cpu(),   # (d_ffn,)
            "std": products.std(dim=0).cpu(),      # (d_ffn,)
        }

        logger.info(
            f"  Layer {layer_idx} w3 stats: "
            f"global_mean={products.mean().item():.6f}, "
            f"global_std={products.std().item():.6f}, "
            f"neuron_std_range=[{stats[layer_idx]['std'].min().item():.6f}, "
            f"{stats[layer_idx]['std'].max().item():.6f}]"
        )

        del w3_f32, products

    if missing_keys:
        available_w3_keys = [k for k in pretrained_state_dict if "w3" in k]
        raise KeyError(
            f"Could not find w3 weights for layers: "
            f"{[idx for idx, _ in missing_keys]}. "
            f"Tried keys: {[k for _, k in missing_keys]}. "
            f"Available w3-related keys: {available_w3_keys}"
        )

    del tok_emb_f32
    return stats