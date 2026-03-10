# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Prepare DAG-STEM checkpoint by computing layerwise w3 x RMSNorm(tok_embeddings).

For each stem layer *i*, computes:

    stem_embedding_table = RMSNorm_i(tok_embeddings.weight) @ w3_i.weight.T
    shape: (vocab_size, hidden_dim)

where ``RMSNorm_i`` uses the ``ffn_norm`` weights from layer *i*.

**Key difference from base STEM** (``prepare_stem_checkpoint.py``):
    - Base STEM **removes** w3 from stem layers after computing stem embeddings
      (the stem embedding completely replaces w3).
    - DAG-STEM **keeps** w3 in stem layers and applies each layer's ``ffn_norm``
      (RMSNorm) to the token embeddings before the w3 projection.  The DAG FFN
      blends w3(x) with the stem embedding y via a learnable alpha gate:

          up = sigmoid(alpha) * w3(x) + (1 - sigmoid(alpha)) * y

      With the default ``alpha_init=-5.0`` (sigmoid ~ 0.007), the model starts
      by almost entirely using the stem embedding (like base STEM), and can
      gradually learn to blend in w3(x) context during training.

The output directory contains:
  - A DCP backbone checkpoint with w3 weights **preserved** for stem layers
  - An updated ``params.json`` with ``stem_layers``, ``stem_parallel_size``,
    and ``alpha_init``
  - A ``stem_shards/`` subdirectory with the pre-computed embeddings, sharded
    along the embedding dimension to match ``stem_parallel_size``

The resulting directory can be used directly as ``checkpoint.init_ckpt_path``
in ``stem_dag_train.py``.

Usage
-----
    python apps/main/prepare_dag_stem_checkpoint.py \\
        --ckpt-path checkpoints/Llama-3.2-1B/distcp \\
        --output-dir checkpoints/Llama-3.2-1B-dag-stem-init \\
        --stem-layers 2 6 10 14 \\
        --stem-parallel-size 4 \\
        --alpha-init -5.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_consolidated_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load the consolidated ``.pth`` checkpoint."""
    consolidated_path = Path(ckpt_path) / "consolidated" / "consolidated.pth"
    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"Consolidated checkpoint not found at {consolidated_path}. "
            f"Please run checkpoint consolidation first."
        )
    logger.info(f"Loading consolidated checkpoint from {consolidated_path}")
    state_dict = torch.load(consolidated_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded {len(state_dict)} keys from checkpoint")
    return state_dict


# ---------------------------------------------------------------------------
# Stem-embedding computation
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * weight


def compute_dag_stem_embeddings(
    state_dict: Dict[str, torch.Tensor],
    stem_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Compute ``RMSNorm(tok_embeddings.weight) @ w3.weight.T`` per stem layer.

    For each stem layer, the token embeddings are first passed through the
    layer's ``ffn_norm`` (RMSNorm) before being projected by w3.  This matches
    the runtime computation more closely since the FFN input is always
    RMSNorm'd.

    Unlike base STEM, w3 weights are **kept** in the state dict (not removed)
    because the DAG FFN uses both w3(x) and the stem embedding y.

    Returns a dict mapping *stem_index* -> ``(vocab_size, hidden_dim)`` tensor.
    """
    tok_emb_key = "model.tok_embeddings.weight"
    if tok_emb_key not in state_dict:
        raise KeyError(
            f"Key '{tok_emb_key}' not found in checkpoint. "
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )

    tok_emb = state_dict[tok_emb_key].float()  # (vocab_size, dim)
    logger.info(f"Token embeddings: shape={tuple(tok_emb.shape)}, dtype=float32 (cast)")

    stem_weights: Dict[int, torch.Tensor] = {}
    for stem_idx, layer_idx in enumerate(stem_layers):
        w3_key = f"model.layers.{layer_idx}.feed_forward.w3.weight"
        ffn_norm_key = f"model.layers.{layer_idx}.ffn_norm.weight"

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
        if ffn_norm_key not in state_dict:
            raise KeyError(
                f"Key '{ffn_norm_key}' not found in checkpoint. "
                f"Available keys (first 10): {list(state_dict.keys())[:10]}"
            )

        w3_weight = state_dict[w3_key].float()  # (hidden_dim, dim)
        ffn_norm_weight = state_dict[ffn_norm_key].float()  # (dim,)

        normed_tok_emb = rms_norm(tok_emb, ffn_norm_weight)
        stem_weight = normed_tok_emb @ w3_weight.T

        stem_weights[stem_idx] = stem_weight

        logger.info(
            f"  Layer {layer_idx:>2} (stem idx {stem_idx}): "
            f"ffn_norm {tuple(ffn_norm_weight.shape)}, "
            f"w3_weight {tuple(w3_weight.shape)}, "
            f" -> stem {tuple(stem_weight.shape)}, "
            f"norm={stem_weight.norm():.4f}, "
            f"mean={stem_weight.mean():.6f}, std={stem_weight.std():.6f}"
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
    """Save stem embeddings as sharded checkpoint files."""
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
            "Prepare DAG-STEM checkpoint by computing layerwise "
            "w3 x RMSNorm(tok_embeddings) while keeping w3 in the backbone."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to pretrained DCP checkpoint directory",
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
        default=[2, 6, 10, 14],
        help="Which transformer layers get stem embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--stem-parallel-size",
        type=int,
        default=4,
        help="Number of stem model-parallel shards (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=-5.0,
        help=(
            "Initial alpha value for DAG FFN gate. "
            "sigmoid(-5.0) ~ 0.007 means almost pure stem embedding to start. "
            "(default: %(default)s)"
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

    # ---- 2. Compute stem embeddings (w3 is KEPT in state_dict) ----
    logger.info(f"Computing DAG stem embeddings for layers: {args.stem_layers}")
    stem_weights = compute_dag_stem_embeddings(state_dict, args.stem_layers)

    # ---- 3. Inject alpha parameters for stem layers ----
    for layer_idx in args.stem_layers:
        alpha_key = f"model.layers.{layer_idx}.feed_forward.alpha"
        state_dict[alpha_key] = torch.tensor([args.alpha_init])
        logger.info(f"  Injected {alpha_key} = [{args.alpha_init}]")

    # ---- 4. Save backbone via DCP (w3 preserved for DAG FFN) ----
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)

    logger.info(f"Saving backbone (w3 preserved for stem layers) to {output_dir}")
    dcp.save(state_dict, checkpoint_id=str(output_dir))
    logger.info("Backbone DCP checkpoint saved")

    dist.destroy_process_group()
    del state_dict

    # ---- 4. Save updated params.json ----
    with open(ckpt_path / "params.json", "r") as f:
        params_dict = json.load(f)
    if "model" not in params_dict:
        params_dict = {"model": params_dict}

    params_dict["model"]["stem_layers"] = args.stem_layers
    params_dict["model"]["alpha_init"] = args.alpha_init
    params_dict["model_type"] = "llama_dag"
    params_dict["distributed"] = params_dict.get("distributed", {})
    params_dict["distributed"]["stem_parallel_size"] = args.stem_parallel_size

    with open(output_dir / "params.json", "w") as f:
        json.dump(params_dict, f)
    logger.info(
        f"Saved params.json with stem_layers={args.stem_layers}, "
        f"alpha_init={args.alpha_init}, "
        f"stem_parallel_size={args.stem_parallel_size}"
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
    logger.info(f"DAG-STEM checkpoint saved to: {output_dir}")
    logger.info(
        f"To use in stem_dag_train.py, set:  checkpoint.init_ckpt_path={output_dir}"
    )
    logger.info(
        f"Stem layers: {args.stem_layers}  |  "
        f"Alpha init: {args.alpha_init}  |  "
        f"Parallel size: {args.stem_parallel_size}"
    )


if __name__ == "__main__":
    main()
