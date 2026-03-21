#!/usr/bin/env python3
"""
Check how similar STEM embeddings are for tokens that collapse to the same
compressed token id.

Unlike the initial version, this script reads precomputed STEM embeddings from
`stem_shards/` in a prepared STEM checkpoint (no recomputation from
tok_embeddings + w3).
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lingua.tokenizer import CompressedTokenizer, build_tokenizer

SHARD_RE = re.compile(r"stem_model_mp(\d+)\.pt$")


def resolve_stem_dir(path_like: str) -> Path:
    p = Path(path_like)
    if p.name == "stem_shards":
        return p
    candidate = p / "stem_shards"
    return candidate if candidate.exists() else p


def discover_shards(stem_dir: Path) -> List[Path]:
    shard_paths = list(stem_dir.glob("stem_model_mp*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in: {stem_dir}")

    def shard_rank(path: Path) -> int:
        m = SHARD_RE.search(path.name)
        if m is None:
            raise ValueError(f"Unexpected shard filename: {path.name}")
        return int(m.group(1))

    shard_paths.sort(key=shard_rank)
    return shard_paths


def load_params_if_exists(ckpt_path: Path) -> Optional[Dict]:
    params_path = ckpt_path / "params.json"
    if not params_path.exists():
        return None
    with params_path.open("r") as f:
        return json.load(f)


def resolve_layer_to_stem_idx(
    raw_layers: Optional[Sequence[int]],
    ckpt_path: Path,
) -> List[Tuple[int, int]]:
    requested = list(raw_layers) if raw_layers is not None and len(raw_layers) > 0 else None
    params = load_params_if_exists(ckpt_path)
    params_layers: Optional[List[int]] = None
    if params is not None:
        model = params.get("model", params)
        stem_layers = model.get("stem_layers")
        if stem_layers is not None and len(stem_layers) > 0:
            params_layers = [int(x) for x in stem_layers]

    if requested is None:
        if params_layers is None:
            raise ValueError(
                "Could not infer stem layers from params.json. Pass --stem-layers explicitly."
            )
        return [(layer_idx, stem_idx) for stem_idx, layer_idx in enumerate(params_layers)]

    if params_layers is None:
        return [(layer_idx, stem_idx) for stem_idx, layer_idx in enumerate(requested)]

    mapping = []
    for layer_idx in requested:
        if layer_idx not in params_layers:
            raise ValueError(
                f"Requested layer {layer_idx} is not in params.json stem_layers={params_layers}"
            )
        mapping.append((layer_idx, params_layers.index(layer_idx)))
    return mapping


def build_groups(lookup: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    num_groups = int(lookup.max()) + 1 if lookup.size > 0 else 0
    buckets: List[List[int]] = [[] for _ in range(num_groups)]
    for token_id, compressed_id in enumerate(lookup.tolist()):
        buckets[int(compressed_id)].append(token_id)

    groups = [
        (compressed_id, np.asarray(ids, dtype=np.int64))
        for compressed_id, ids in enumerate(buckets)
        if len(ids) >= 2
    ]
    groups.sort(key=lambda x: x[1].size, reverse=True)
    return groups


def token_preview(tokenizer, token_ids: Sequence[int], max_items: int = 4) -> List[str]:
    out = []
    for tid in token_ids[:max_items]:
        try:
            txt = tokenizer.decode([int(tid)], skip_special_tokens=False)
        except TypeError:
            txt = tokenizer.decode([int(tid)])
        out.append(txt.replace("\n", "\\n"))
    return out


def pairwise_cosine_stats(
    normalized_vectors: torch.Tensor,
    sample_pairs_per_group: int,
    exact_group_size_limit: int,
    rng: torch.Generator,
) -> Tuple[int, float, float, float, float]:
    n = normalized_vectors.shape[0]
    if n < 2:
        return 0, 0.0, 0.0, 0.0, 0.0

    total_pairs = n * (n - 1) // 2
    if n <= exact_group_size_limit:
        sim = normalized_vectors @ normalized_vectors.T
        i, j = torch.triu_indices(n, n, offset=1)
        vals = sim[i, j]
    else:
        sample = min(sample_pairs_per_group, total_pairs)
        i = torch.randint(0, n, (sample,), generator=rng)
        j = torch.randint(0, n - 1, (sample,), generator=rng)
        j = j + (j >= i).to(torch.long)
        vals = (normalized_vectors[i] * normalized_vectors[j]).sum(dim=1)

    vals64 = vals.to(torch.float64)
    return (
        int(vals64.numel()),
        float(vals64.sum().item()),
        float((vals64 * vals64).sum().item()),
        float(vals64.min().item()),
        float(vals64.max().item()),
    )


def load_layer_shards(
    shard_paths: Sequence[Path],
    stem_idx: int,
) -> Tuple[List[torch.Tensor], int]:
    key = f"stem_embeddings.{stem_idx}.weight"
    chunks: List[torch.Tensor] = []
    vocab_size: Optional[int] = None
    for shard_path in shard_paths:
        sd = torch.load(shard_path, map_location="cpu")
        if key not in sd:
            raise KeyError(f"Missing key '{key}' in shard {shard_path}")
        chunk = sd[key]
        if chunk.ndim != 2:
            raise ValueError(f"Expected 2D tensor for '{key}' in {shard_path}, got {tuple(chunk.shape)}")
        if vocab_size is None:
            vocab_size = int(chunk.shape[0])
        elif int(chunk.shape[0]) != vocab_size:
            raise ValueError(
                f"Inconsistent vocab size for '{key}': expected {vocab_size}, got {chunk.shape[0]}"
            )
        chunks.append(chunk)
    assert vocab_size is not None
    return chunks, vocab_size


def analyze_layer(
    shard_chunks: Sequence[torch.Tensor],
    groups: List[Tuple[int, np.ndarray]],
    sample_pairs_per_group: int,
    exact_group_size_limit: int,
    top_k_worst_groups: int,
    rng: torch.Generator,
    tokenizer=None,
) -> Dict:
    global_pairs = 0
    global_sum = 0.0
    global_sum_sq = 0.0
    global_min = math.inf
    global_max = -math.inf

    centroid_cos_sum = 0.0
    centroid_cos_count = 0

    group_records = []
    for compressed_token_id, token_ids_np in groups:
        token_ids = torch.from_numpy(token_ids_np)
        parts = [chunk.index_select(0, token_ids) for chunk in shard_chunks]
        stem_rows = torch.cat(parts, dim=1).float()
        normed = F.normalize(stem_rows, p=2, dim=1, eps=1e-12)

        num_pairs, pair_sum, pair_sum_sq, min_cos, max_cos = pairwise_cosine_stats(
            normalized_vectors=normed,
            sample_pairs_per_group=sample_pairs_per_group,
            exact_group_size_limit=exact_group_size_limit,
            rng=rng,
        )
        if num_pairs == 0:
            continue

        global_pairs += num_pairs
        global_sum += pair_sum
        global_sum_sq += pair_sum_sq
        global_min = min(global_min, min_cos)
        global_max = max(global_max, max_cos)

        centroid = F.normalize(normed.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-12)
        centroid_cos = (normed @ centroid.T).squeeze(1)
        centroid_cos_sum += float(centroid_cos.sum().item())
        centroid_cos_count += int(centroid_cos.numel())

        mean_cos = pair_sum / float(num_pairs)
        record = {
            "group_size": int(token_ids_np.size),
            "num_pairs_used": int(num_pairs),
            "mean_pairwise_cosine": float(mean_cos),
            "min_pairwise_cosine": float(min_cos),
            "max_pairwise_cosine": float(max_cos),
            "mean_centroid_cosine": float(centroid_cos.mean().item()),
            "compressed_token_id": int(compressed_token_id),
            "token_ids_preview": [int(t) for t in token_ids_np[:8].tolist()],
        }
        if tokenizer is not None:
            record["token_text_preview"] = token_preview(tokenizer, token_ids_np.tolist(), max_items=4)
        group_records.append(record)

    if global_pairs == 0:
        raise RuntimeError("No valid token groups found for cosine analysis.")

    global_mean = global_sum / float(global_pairs)
    global_var = max(global_sum_sq / float(global_pairs) - global_mean * global_mean, 0.0)
    global_std = math.sqrt(global_var)
    centroid_mean = centroid_cos_sum / float(max(centroid_cos_count, 1))

    group_records.sort(key=lambda x: x["mean_pairwise_cosine"])
    worst_groups = group_records[:top_k_worst_groups]
    best_groups = list(reversed(group_records[-top_k_worst_groups:]))

    return {
        "num_groups_analyzed": len(group_records),
        "global_pairwise_cosine": {
            "num_pairs": int(global_pairs),
            "mean": float(global_mean),
            "std": float(global_std),
            "min": float(global_min),
            "max": float(global_max),
        },
        "global_mean_centroid_cosine": float(centroid_mean),
        "worst_groups_by_mean_pairwise_cosine": worst_groups,
        "best_groups_by_mean_pairwise_cosine": best_groups,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check similarity of precomputed STEM embeddings for compressed token groups"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Prepared STEM checkpoint root (containing stem_shards) or direct stem_shards path",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        required=True,
        help="Tokenizer type passed to build_tokenizer (e.g. tiktoken, sp, huggingface)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Tokenizer path passed to build_tokenizer",
    )
    parser.add_argument(
        "--stem-layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional stem layer indices. If omitted, inferred from ckpt params.json",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="If >0, only analyze the largest N compressed groups (for faster debugging)",
    )
    parser.add_argument(
        "--sample-pairs-per-group",
        type=int,
        default=10000,
        help="For large groups, number of random pairs used to estimate pairwise cosine",
    )
    parser.add_argument(
        "--exact-group-size-limit",
        type=int,
        default=64,
        help="Groups up to this size use exact all-pairs cosine",
    )
    parser.add_argument(
        "--top-k-worst-groups",
        type=int,
        default=10,
        help="How many low-similarity groups to include per layer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for large-group pair sampling",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full JSON report",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    stem_dir = resolve_stem_dir(args.ckpt_path)
    shard_paths = discover_shards(stem_dir)
    layer_to_stem_idx = resolve_layer_to_stem_idx(args.stem_layers, ckpt_path)

    tokenizer = build_tokenizer(args.tokenizer_name, args.tokenizer_path)
    compressed_tokenizer = CompressedTokenizer(tokenizer)
    lookup = compressed_tokenizer.lookup_table

    all_groups = build_groups(lookup)
    groups = all_groups
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    _, first_stem_idx = layer_to_stem_idx[0]
    first_chunks, vocab_size = load_layer_shards(shard_paths, first_stem_idx)
    hidden_dim = int(sum(chunk.shape[1] for chunk in first_chunks))

    if lookup.shape[0] != vocab_size:
        raise ValueError(
            f"Tokenizer vocab mismatch: lookup has {lookup.shape[0]} rows but checkpoint has {vocab_size}"
        )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    report = {
        "metadata": {
            "ckpt_path": str(ckpt_path),
            "stem_dir": str(stem_dir),
            "num_shards": len(shard_paths),
            "tokenizer_name": args.tokenizer_name,
            "tokenizer_path": args.tokenizer_path,
            "stem_layers": [layer for layer, _ in layer_to_stem_idx],
            "vocab_size": int(vocab_size),
            "stem_hidden_dim": hidden_dim,
            "compressed_vocab_size": int(len(compressed_tokenizer)),
            "num_groups_size_ge_2": int(len(all_groups)),
            "num_groups_analyzed": int(len(groups)),
            "sample_pairs_per_group": int(args.sample_pairs_per_group),
            "exact_group_size_limit": int(args.exact_group_size_limit),
        },
        "per_layer": {},
    }

    print(
        f"vocab_size={vocab_size}, stem_hidden_dim={hidden_dim}, "
        f"compressed_vocab_size={len(compressed_tokenizer)}"
    )
    print(f"stem_shards={stem_dir}, num_shards={len(shard_paths)}")
    print(f"groups(size>=2): total={report['metadata']['num_groups_size_ge_2']}, analyzed={len(groups)}")

    cached_first = {first_stem_idx: first_chunks}
    for layer_idx, stem_idx in layer_to_stem_idx:
        if stem_idx in cached_first:
            layer_chunks = cached_first[stem_idx]
        else:
            layer_chunks, layer_vocab = load_layer_shards(shard_paths, stem_idx)
            if layer_vocab != vocab_size:
                raise ValueError(
                    f"Layer {layer_idx} has vocab_size={layer_vocab}, expected {vocab_size}"
                )
        print(f"[layer {layer_idx}] analyzing {len(groups)} groups...")
        layer_report = analyze_layer(
            shard_chunks=layer_chunks,
            groups=groups,
            sample_pairs_per_group=args.sample_pairs_per_group,
            exact_group_size_limit=args.exact_group_size_limit,
            top_k_worst_groups=args.top_k_worst_groups,
            rng=rng,
            tokenizer=tokenizer,
        )
        report["per_layer"][str(layer_idx)] = layer_report

        gp = layer_report["global_pairwise_cosine"]
        print(
            f"  pairwise cosine: mean={gp['mean']:.6f}, std={gp['std']:.6f}, "
            f"min={gp['min']:.6f}, max={gp['max']:.6f}, num_pairs={gp['num_pairs']}"
        )
        print(f"  mean cosine-to-centroid: {layer_report['global_mean_centroid_cosine']:.6f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved JSON report to {out}")


if __name__ == "__main__":
    main()
