# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Prepare STEM checkpoint by computing layerwise stem init from token embeddings.

For each stem layer *i*, computes:

    stem_embedding_table = stem_input_i(tok_embeddings.weight) @ w3_i.weight.T
    shape: (vocab_size, hidden_dim)

where ``stem_input_i`` depends on model style:
- LLaMA/Qwen style: ``RMSNorm_i`` using ``ffn_norm`` weights
- OLMo style: identity (no pre-FFN norm)

i.e., for every token *t*:

    stem_emb[t] = w3_i.weight @ stem_input_i(tok_embeddings.weight[t])

This initialises STEM embeddings so that each token's embedding matches the
original FFN up-projection output (after normalization), providing a
"functionally equivalent" starting point for STEM training.

The output directory contains:
  - A DCP backbone checkpoint with the w3 weights removed for stem layers
  - An updated ``params.json`` with ``stem_layers`` and ``stem_parallel_size``
  - A ``stem_shards/`` subdirectory with the pre-computed embeddings, sharded
    along the embedding dimension to match ``stem_parallel_size``.

The resulting directory can be used directly as ``checkpoint.init_ckpt_path``
in ``stem_train.py``.

Usage
-----
    python apps/main/prepare_stem_checkpoint.py \\
        --ckpt-path checkpoints/Llama-3.2-1B/distcp \\
        --output-dir checkpoints/Llama-3.2-1B-stem-init \\
        --stem-layers 1 3 5 7 9 11 13 15 \\
        --stem-parallel-size 8
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from lingua.tokenizer import CompressedTokenizer, build_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_consolidated_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load the consolidated ``.pth`` checkpoint.

    Expects ``<ckpt_path>/consolidated/consolidated.pth`` to exist.
    """
    consolidated_path = Path(ckpt_path) / "consolidated" / "consolidated.pth"
    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"Consolidated checkpoint not found at {consolidated_path}. "
            f"Please run checkpoint consolidation first or provide a checkpoint "
            f"directory that contains consolidated/consolidated.pth."
        )
    logger.info(f"Loading consolidated checkpoint from {consolidated_path}")
    state_dict = torch.load(consolidated_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded {len(state_dict)} keys from checkpoint")
    if "model" in state_dict:
        logger.info("Checkpoint has 'model' key; using state_dict['model']")
        state_dict = state_dict["model"]
        state_dict = {"model." + k: v for k, v in state_dict.items()}
    return state_dict


# ---------------------------------------------------------------------------
# Stem-embedding computation
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * weight


def compute_stem_embeddings(
    state_dict: Dict[str, torch.Tensor],
    stem_layers: List[int],
    compressed_lookup: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Compute stem embedding init ``stem_input(tok_emb) @ w3.weight.T``.

    For each stem layer:
    - if ``model.layers.<i>.ffn_norm.weight`` exists, apply RMSNorm first
      (LLaMA/Qwen-style pre-FFN norm),
    - else if ``model.layers.<i>.post_feedforward_norm.weight`` exists, use
      identity input (OLMo-style post-FFN norm),
    - else raise.

    If ``compressed_lookup`` is provided (shape: ``[vocab_size]``), each
    layer's full-vocab table is reduced to compressed-vocab rows by averaging
    all rows that map to the same compressed token id.

    Returns a dict mapping *stem_index* → ``(stem_vocab_size, hidden_dim)``
    tensor (float32), where ``stem_vocab_size`` is ``vocab_size`` in normal
    mode and compressed vocab size in compressed mode.
    """
    tok_emb_key = "model.tok_embeddings.weight"
    if tok_emb_key not in state_dict:
        raise KeyError(
            f"Key '{tok_emb_key}' not found in checkpoint. "
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )

    tok_emb = state_dict[tok_emb_key].float()  # (vocab_size, dim)
    vocab_size, dim = tok_emb.shape
    logger.info(f"Token embeddings: shape={tuple(tok_emb.shape)}, dtype=float32 (cast)")

    compressed_vocab_size = None
    if compressed_lookup is not None:
        if compressed_lookup.ndim != 1 or compressed_lookup.numel() != vocab_size:
            raise ValueError(
                "compressed_lookup must be 1-D with length equal to checkpoint vocab_size "
                f"({vocab_size}), got shape={tuple(compressed_lookup.shape)}"
            )
        if compressed_lookup.dtype != torch.long:
            compressed_lookup = compressed_lookup.to(torch.long)
        compressed_lookup = compressed_lookup.contiguous()
        compressed_vocab_size = int(compressed_lookup.max().item()) + 1
        logger.info(
            f"Using compressed STEM init: vocab_size={vocab_size} -> "
            f"stem_vocab_size={compressed_vocab_size}"
        )

    stem_weights: Dict[int, torch.Tensor] = {}
    for stem_idx, layer_idx in enumerate(stem_layers):
        w3_key = f"model.layers.{layer_idx}.feed_forward.w3.weight"
        ffn_norm_key = f"model.layers.{layer_idx}.ffn_norm.weight"
        post_ffn_norm_key = f"model.layers.{layer_idx}.post_feedforward_norm.weight"

        if w3_key not in state_dict:
            available_layers = sorted(
                {
                    int(k.split(".")[2])
                    for k in state_dict
                    if k.startswith("model.layers.") and "feed_forward.w3" in k
                }
            )
            raise KeyError(
                f"Key '{w3_key}' not found in checkpoint. "
                f"Layers with w3: {available_layers}"
            )
        w3_weight = state_dict[w3_key].float()  # (hidden_dim, dim)
        hidden_dim = w3_weight.shape[0]
        norm_source = "identity"
        if ffn_norm_key in state_dict:
            ffn_norm_weight = state_dict[ffn_norm_key].float()  # (dim,)
            normed_tok_emb = rms_norm(tok_emb, ffn_norm_weight)
            norm_source = "ffn_norm"
        elif post_ffn_norm_key in state_dict:
            # OLMo blocks use post-FFN RMSNorm; w3 input is not pre-normalized.
            normed_tok_emb = tok_emb
            norm_source = "post_feedforward_norm->identity_input"
        else:
            raise KeyError(
                f"Neither '{ffn_norm_key}' nor '{post_ffn_norm_key}' found in checkpoint. "
                f"Available keys (first 10): {list(state_dict.keys())[:10]}"
            )
        stem_weight = normed_tok_emb @ w3_weight.T  # (vocab_size, hidden_dim)

        if compressed_lookup is not None:
            assert compressed_vocab_size is not None
            compressed_weight = torch.zeros(
                compressed_vocab_size, hidden_dim, dtype=stem_weight.dtype
            )
            counts = torch.zeros(compressed_vocab_size, dtype=stem_weight.dtype)
            compressed_weight.index_add_(0, compressed_lookup, stem_weight)
            counts.index_add_(
                0,
                compressed_lookup,
                torch.ones(vocab_size, dtype=stem_weight.dtype),
            )
            stem_weight = compressed_weight / counts.clamp_min(1.0).unsqueeze(1)

        stem_weights[stem_idx] = stem_weight
        
        del state_dict[w3_key]

        norms = stem_weight.norm(dim=-1)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        logger.info(
            f"  Layer {layer_idx:>2} (stem idx {stem_idx}): "
            f"norm_source={norm_source}, "
            f"w3_weight {tuple(w3_weight.shape)}, "
            f" -> stem {tuple(stem_weight.shape)}, "
            f"norm={stem_weight.norm():.4f}, "
            f"mean={mean_norm:.6f}, std={std_norm:.6f}"
        )

    return stem_weights


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_stem_shards(
    stem_weights: Dict[int, torch.Tensor],
    output_dir: Path,
    stem_parallel_size: int,
):
    """Save stem embeddings as sharded checkpoint files.

    Creates ``<output_dir>/stem_shards/stem_model_mp{rank}.pt`` for each MP
    rank.  The embedding dimension is split evenly across shards.
    """
    stem_dir = output_dir / "stem_shards"
    stem_dir.mkdir(parents=True, exist_ok=True)

    first_weight = next(iter(stem_weights.values()))
    vocab_size, hidden_dim = first_weight.shape

    assert hidden_dim % stem_parallel_size == 0, (
        f"hidden_dim ({hidden_dim}) must be divisible by "
        f"stem_parallel_size ({stem_parallel_size})"
    )
    shard_size = hidden_dim // stem_parallel_size

    for mp_rank in range(stem_parallel_size):
        shard_dict: Dict[str, torch.Tensor] = {}
        start = mp_rank * shard_size
        end = start + shard_size

        for stem_idx, full_weight in stem_weights.items():
            key = f"stem_embeddings.{stem_idx}.weight"
            # .clone() is critical: column-slicing creates a view that shares
            # the full underlying storage; without clone torch.save would
            # serialise the entire tensor for every shard.
            shard_dict[key] = full_weight[:, start:end].clone().cpu()

        shard_path = stem_dir / f"stem_model_mp{mp_rank}.pt"
        torch.save(shard_dict, shard_path)
        logger.info(
            f"  Saved stem shard mp_rank={mp_rank}: {shard_path} "
            f"(shape per embedding: ({vocab_size}, {shard_size}))"
        )
    
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare STEM checkpoint by computing layerwise "
            "w3 × RMSNorm(tok_embeddings) and saving as stem_shards."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help=(
            "Path to pretrained DCP checkpoint directory "
            "(must contain consolidated/consolidated.pth)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the new checkpoint with stem embeddings",
    )
    parser.add_argument(
        "--stem-layers",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9, 11, 13, 15],
        help="Which transformer layers get stem embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--stem-parallel-size",
        type=int,
        default=8,
        help="Number of stem model-parallel shards (default: %(default)s)",
    )
    parser.add_argument(
        "--use-compressed-tokenizer",
        action="store_true",
        help=(
            "If set, initialize stem embedding rows in compressed-token space "
            "(one row per CompressedTokenizer id)."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help=(
            "Tokenizer type for build_tokenizer (required with "
            "--use-compressed-tokenizer)."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help=(
            "Tokenizer path for build_tokenizer (required with "
            "--use-compressed-tokenizer)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_dir)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint path does not exist: {ckpt_path}")
        sys.exit(1)

    # ---- 1. Load consolidated checkpoint ----
    state_dict = load_consolidated_checkpoint(str(ckpt_path))

    compressed_lookup = None
    compressed_vocab_size = None
    if args.use_compressed_tokenizer:
        if not args.tokenizer_name or not args.tokenizer_path:
            logger.error(
                "--tokenizer-name and --tokenizer-path are required when "
                "--use-compressed-tokenizer is set."
            )
            sys.exit(1)
        logger.info(
            "Building CompressedTokenizer lookup table "
            f"(name={args.tokenizer_name}, path={args.tokenizer_path})"
        )
        tokenizer = build_tokenizer(args.tokenizer_name, args.tokenizer_path)
        compressed_tokenizer = CompressedTokenizer(tokenizer)
        compressed_lookup = torch.from_numpy(compressed_tokenizer.lookup_table).to(torch.long)
        compressed_vocab_size = len(compressed_tokenizer)
        logger.info(
            f"Compressed tokenizer built: original_vocab={compressed_lookup.numel()}, "
            f"compressed_vocab={compressed_vocab_size}"
        )

    # ---- 2. Compute stem embeddings (also removes w3 keys from state_dict) ----
    logger.info(f"Computing stem embeddings for layers: {args.stem_layers}")
    stem_weights = compute_stem_embeddings(
        state_dict,
        args.stem_layers,
        compressed_lookup=compressed_lookup,
    )

    # ---- 3. Save modified backbone via DCP ----
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)

    logger.info(f"Saving modified backbone (w3 removed for stem layers) to {output_dir}")
    dcp.save(state_dict, checkpoint_id=str(output_dir))
    logger.info("Backbone DCP checkpoint saved")

    dist.destroy_process_group()

    # Free memory – we no longer need the full checkpoint
    del state_dict

    # ---- 4. Save updated params.json ----
    with open(ckpt_path / "params.json", "r") as f:
        params_dict = json.load(f)
    if "model" not in params_dict:
        params_dict = {"model": params_dict}
    params_dict["model"]["stem_layers"] = args.stem_layers
    if args.use_compressed_tokenizer:
        if compressed_vocab_size is None:
            raise RuntimeError("compressed_vocab_size is not set")
        params_dict["model"]["stem_vocab_size"] = compressed_vocab_size
    params_dict["distributed"] = params_dict.get("distributed", {})
    params_dict["distributed"]["stem_parallel_size"] = args.stem_parallel_size
    with open(output_dir / "params.json", "w") as f:
        json.dump(params_dict, f)
    logger.info(
        f"Saved params.json with stem_layers={args.stem_layers}, "
        f"stem_parallel_size={args.stem_parallel_size}, "
        f"stem_vocab_size={params_dict['model'].get('stem_vocab_size', 'full_vocab')}"
    )

    # ---- 5. Save stem shards ----
    if not (output_dir / "stem_shards").exists():
        logger.info(
            f"Saving stem shards with stem_parallel_size={args.stem_parallel_size}"
        )
        save_stem_shards(stem_weights, output_dir, args.stem_parallel_size)
    else:
        logger.info(f"Stem shards already exist at {output_dir / 'stem_shards'}")

    # ---- Done ----
    logger.info("")
    logger.info(f"STEM checkpoint saved to: {output_dir}")
    logger.info(
        f"To use in stem_train.py, set:  checkpoint.init_ckpt_path={output_dir}"
    )
    logger.info(
        f"Stem layers: {args.stem_layers}  |  "
        f"Parallel size: {args.stem_parallel_size}  |  "
        f"Shards: {args.stem_parallel_size}  |  "
        f"Compressed init: {args.use_compressed_tokenizer}"
    )


if __name__ == "__main__":
    main()

