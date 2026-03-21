#!/usr/bin/env python3
"""
Compute row-wise STEM embedding distributions from stem shards.

This is intentionally minimal and focused on row statistics only.
Input is a checkpoint root or a direct `stem_shards/` path.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


SHARD_RE = re.compile(r"stem_model_mp(\d+)\.pt$")


def summarize_vector(
    x: torch.Tensor,
    bins: int,
    hist_range: Optional[Tuple[float, float]] = None,
) -> Dict:
    if x.numel() == 0:
        return {}

    q_levels = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float64)
    q = torch.quantile(x, q_levels)
    out = {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p01": float(q[0].item()),
        "p05": float(q[1].item()),
        "p25": float(q[2].item()),
        "p50": float(q[3].item()),
        "p75": float(q[4].item()),
        "p95": float(q[5].item()),
        "p99": float(q[6].item()),
        "max": float(x.max().item()),
    }

    if bins > 0:
        if hist_range is None:
            lo = float(x.min().item())
            hi = float(x.max().item())
            if hi <= lo:
                pad = 1.0 if lo == 0.0 else abs(lo) * 1e-6
                hist_range = (lo - pad, hi + pad)
            else:
                hist_range = (lo, hi)
        hist = torch.histc(x.float(), bins=bins, min=hist_range[0], max=hist_range[1])
        out["histogram"] = {
            "bins": bins,
            "range": [hist_range[0], hist_range[1]],
            "counts": [int(v) for v in hist.tolist()],
        }
    return out


@dataclass
class LayerAccumulator:
    layer_idx: int
    vocab_size: int
    embed_dim: int = 0

    def __post_init__(self):
        self.row_l1 = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.row_l2_sq = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.row_nnz = torch.zeros(self.vocab_size, dtype=torch.int64)

    def update(self, chunk: torch.Tensor, eps: float):
        if chunk.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding chunk for layer {self.layer_idx}, got {tuple(chunk.shape)}"
            )
        if chunk.shape[0] != self.vocab_size:
            raise ValueError(
                f"Inconsistent vocab size for layer {self.layer_idx}: expected {self.vocab_size}, got {chunk.shape[0]}"
            )

        x = chunk.to(dtype=torch.float32)
        abs_x = x.abs()
        self.embed_dim += x.shape[1]
        self.row_l1 += abs_x.sum(dim=1, dtype=torch.float64)
        self.row_l2_sq += (x * x).sum(dim=1, dtype=torch.float64)
        self.row_nnz += (abs_x > eps).sum(dim=1, dtype=torch.int64)

    def finalize(self, bins: int) -> Dict:
        row_l2 = torch.sqrt(self.row_l2_sq.clamp_min(0.0))
        row_density = self.row_nnz.to(torch.float64) / float(self.embed_dim)
        row_l1_over_l2 = self.row_l1 / row_l2.clamp_min(1e-12)

        sqrt_d = math.sqrt(self.embed_dim)
        if self.embed_dim > 1:
            row_hoyer = (sqrt_d - row_l1_over_l2) / (sqrt_d - 1.0)
            row_hoyer = row_hoyer.clamp(0.0, 1.0)
        else:
            row_hoyer = torch.zeros_like(row_density)

        return {
            "layer_idx": self.layer_idx,
            "shape": [self.vocab_size, self.embed_dim],
            "row_distributions": {
                "nnz_count": summarize_vector(self.row_nnz.to(torch.float64), bins=bins),
                "density": summarize_vector(row_density, bins=bins, hist_range=(0.0, 1.0)),
                "l1": summarize_vector(self.row_l1, bins=bins),
                "l2": summarize_vector(row_l2, bins=bins),
                "hoyer_sparsity": summarize_vector(row_hoyer, bins=bins, hist_range=(0.0, 1.0)),
            },
        }


def resolve_stem_dir(path_like: str) -> Path:
    p = Path(path_like)
    if p.name == "stem_shards":
        return p
    stem_dir = p / "stem_shards"
    return stem_dir if stem_dir.exists() else p


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


def analyze_shards(stem_dir: Path, eps: float, bins: int, layer_key_regex: str) -> Dict:
    layer_re = re.compile(layer_key_regex)
    shard_paths = discover_shards(stem_dir)

    first_sd = torch.load(shard_paths[0], map_location="cpu")
    layer_accs: Dict[str, LayerAccumulator] = {}
    for key, tensor in first_sd.items():
        m = layer_re.match(key)
        if m is None or tensor.ndim != 2:
            continue
        layer_accs[key] = LayerAccumulator(layer_idx=int(m.group(1)), vocab_size=tensor.shape[0])

    if not layer_accs:
        raise RuntimeError(f"No keys matched regex '{layer_key_regex}' in {shard_paths[0]}")

    print(f"Found {len(shard_paths)} shards in {stem_dir}")
    print(f"Tracking {len(layer_accs)} embedding layers")

    for i, shard_path in enumerate(shard_paths, start=1):
        print(f"[{i}/{len(shard_paths)}] Loading {shard_path.name}")
        sd = torch.load(shard_path, map_location="cpu")
        for key, acc in layer_accs.items():
            if key not in sd:
                raise KeyError(f"Missing key '{key}' in shard {shard_path}")
            acc.update(sd[key], eps=eps)

    per_layer = [acc.finalize(bins=bins) for _, acc in sorted(layer_accs.items(), key=lambda kv: kv[1].layer_idx)]
    return {
        "metadata": {
            "stem_dir": str(stem_dir),
            "num_shards": len(shard_paths),
            "eps": eps,
            "bins": bins,
            "layer_key_regex": layer_key_regex,
            "num_layers": len(per_layer),
        },
        "per_layer": per_layer,
    }


def print_row_summary(report: Dict):
    print("\n=== Row Distribution Medians ===")
    for layer in report["per_layer"]:
        d = layer["row_distributions"]
        print(
            "layer={:>2d} shape={} density_p50={:.6f} hoyer_p50={:.6f} l2_p50={:.6f}".format(
                int(layer["layer_idx"]),
                tuple(layer["shape"]),
                d["density"]["p50"],
                d["hoyer_sparsity"]["p50"],
                d["l2"]["p50"],
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute row-wise STEM embedding distribution stats")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Checkpoint root or direct stem_shards path")
    parser.add_argument("--eps", type=float, default=0.0, help="Count non-zero as |x| > eps")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins per row metric")
    parser.add_argument(
        "--layer-key-regex",
        type=str,
        default=r"^stem_embeddings\.(\d+)\.weight$",
        help="Regex selecting embedding keys; must capture layer index",
    )
    parser.add_argument("--output-json", type=str, required=True, help="Path to save JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    stem_dir = resolve_stem_dir(args.ckpt_path)
    report = analyze_shards(stem_dir=stem_dir, eps=args.eps, bins=args.bins, layer_key_regex=args.layer_key_regex)
    print_row_summary(report)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Analyze sparsity characteristics of STEM embedding checkpoints.

This script reads STEM embedding shard files (``stem_model_mp*.pt``) and
computes global and per-layer statistics such as:
  - L0 count / density (with configurable epsilon threshold)
  - L1 / L2 / Linf norms
  - mean, std, signed mean
  - per-token (row-wise) distributions of:
      * non-zero count
      * density
      * L1 norm
      * L2 norm
      * Hoyer sparsity

It is designed to be memory-safe for large checkpoints by aggregating shard
contributions instead of reconstructing full embedding matrices in memory.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


SHARD_RE = re.compile(r"stem_model_mp(\d+)\.pt$")


@dataclass
class LayerAccumulator:
    layer_idx: int
    vocab_size: int
    embed_dim: int = 0
    numel: int = 0
    l0: int = 0
    l1: float = 0.0
    l2_sq: float = 0.0
    linf: float = 0.0
    sum_val: float = 0.0
    sum_sq: float = 0.0
    pos_count: int = 0
    neg_count: int = 0

    def __post_init__(self):
        self.row_l1 = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.row_l2_sq = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.row_nnz = torch.zeros(self.vocab_size, dtype=torch.int64)

    def update(self, chunk: torch.Tensor, eps: float):
        # chunk shape: [vocab_size, shard_dim]
        if chunk.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding chunk for layer {self.layer_idx}, got {tuple(chunk.shape)}"
            )

        if chunk.shape[0] != self.vocab_size:
            raise ValueError(
                f"Inconsistent vocab size for layer {self.layer_idx}: expected {self.vocab_size}, "
                f"got {chunk.shape[0]}"
            )

        x = chunk.to(dtype=torch.float32)
        abs_x = x.abs()
        sq_x = x * x
        nz = abs_x > eps

        self.embed_dim += x.shape[1]
        self.numel += x.numel()
        self.l0 += int(nz.sum().item())
        self.l1 += float(abs_x.sum().item())
        self.l2_sq += float(sq_x.sum().item())
        self.linf = max(self.linf, float(abs_x.max().item()))
        self.sum_val += float(x.sum().item())
        self.sum_sq += float(sq_x.sum().item())
        self.pos_count += int((x > 0).sum().item())
        self.neg_count += int((x < 0).sum().item())

        self.row_l1 += abs_x.sum(dim=1, dtype=torch.float64)
        self.row_l2_sq += sq_x.sum(dim=1, dtype=torch.float64)
        self.row_nnz += nz.sum(dim=1, dtype=torch.int64)

    def finalize(self, bins: int) -> Dict:
        if self.numel == 0:
            return {}

        row_l2 = torch.sqrt(self.row_l2_sq.clamp_min(0.0))
        row_density = self.row_nnz.to(torch.float64) / float(self.embed_dim)
        row_l1_over_l2 = self.row_l1 / row_l2.clamp_min(1e-12)
        sqrt_d = math.sqrt(self.embed_dim)
        if self.embed_dim > 1:
            row_hoyer = (sqrt_d - row_l1_over_l2) / (sqrt_d - 1.0)
            row_hoyer = row_hoyer.clamp(0.0, 1.0)
        else:
            # With a 1D vector, Hoyer sparsity is not informative.
            row_hoyer = torch.zeros_like(row_density)

        l0_density = self.l0 / float(self.numel)
        zero_fraction = 1.0 - l0_density
        mean_val = self.sum_val / float(self.numel)
        var = max(self.sum_sq / float(self.numel) - mean_val * mean_val, 0.0)
        std_val = math.sqrt(var)

        return {
            "layer_idx": self.layer_idx,
            "shape": [self.vocab_size, self.embed_dim],
            "numel": self.numel,
            "l0_count": self.l0,
            "l0_density": l0_density,
            "zero_fraction": zero_fraction,
            "l1": self.l1,
            "l2": math.sqrt(self.l2_sq),
            "linf": self.linf,
            "mean": mean_val,
            "std": std_val,
            "positive_fraction": self.pos_count / float(self.numel),
            "negative_fraction": self.neg_count / float(self.numel),
            "row_distributions": {
                "nnz_count": summarize_vector(self.row_nnz.to(torch.float64), bins=bins),
                "density": summarize_vector(row_density, hist_range=(0.0, 1.0), bins=bins),
                "l1": summarize_vector(self.row_l1, bins=bins),
                "l2": summarize_vector(row_l2, bins=bins),
                "hoyer_sparsity": summarize_vector(row_hoyer, hist_range=(0.0, 1.0), bins=bins),
            },
        }


def summarize_vector(
    x: torch.Tensor,
    hist_range: Optional[Tuple[float, float]] = None,
    bins: Optional[int] = None,
) -> Dict:
    if x.numel() == 0:
        return {}
    q_levels = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float64)
    q = torch.quantile(x, q_levels)
    out = {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p01": float(q[0].item()),
        "p05": float(q[1].item()),
        "p25": float(q[2].item()),
        "p50": float(q[3].item()),
        "p75": float(q[4].item()),
        "p95": float(q[5].item()),
        "p99": float(q[6].item()),
        "max": float(x.max().item()),
    }
    if bins is not None:
        if hist_range is None:
            min_v = float(x.min().item())
            max_v = float(x.max().item())
            # Avoid degenerate histogram range when all values are equal.
            if max_v <= min_v:
                pad = 1.0 if min_v == 0.0 else abs(min_v) * 1e-6
                hist_range = (min_v - pad, max_v + pad)
            else:
                hist_range = (min_v, max_v)
        hist = torch.histc(x.float(), bins=bins, min=hist_range[0], max=hist_range[1])
        out["histogram"] = {
            "bins": bins,
            "range": [hist_range[0], hist_range[1]],
            "counts": [int(v) for v in hist.tolist()],
        }
    return out


def discover_shards(stem_dir: Path) -> List[Path]:
    shard_paths = list(stem_dir.glob("stem_model_mp*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in: {stem_dir}")

    def shard_rank(path: Path) -> int:
        m = SHARD_RE.search(path.name)
        if m is None:
            raise ValueError(f"Unexpected shard filename format: {path.name}")
        return int(m.group(1))

    shard_paths.sort(key=shard_rank)
    return shard_paths


def resolve_stem_dir(path_like: str) -> Path:
    p = Path(path_like)
    if p.name == "stem_shards":
        return p
    cand = p / "stem_shards"
    if cand.exists():
        return cand
    return p


def analyze_shards(
    stem_dir: Path,
    eps: float,
    bins: int,
    layer_key_regex: str,
) -> Dict:
    layer_re = re.compile(layer_key_regex)
    shard_paths = discover_shards(stem_dir)

    # Initialize layer accumulators from the first shard.
    first_sd = torch.load(shard_paths[0], map_location="cpu")
    layer_accs: Dict[str, LayerAccumulator] = {}
    for key, tensor in first_sd.items():
        m = layer_re.match(key)
        if m is None:
            continue
        if tensor.ndim != 2:
            continue
        layer_idx = int(m.group(1))
        layer_accs[key] = LayerAccumulator(layer_idx=layer_idx, vocab_size=tensor.shape[0])

    if not layer_accs:
        raise RuntimeError(
            f"No keys matched regex '{layer_key_regex}' in shard {shard_paths[0]}"
        )

    print(f"Found {len(shard_paths)} shards in {stem_dir}")
    print(f"Tracking {len(layer_accs)} embedding layers")

    for i, shard_path in enumerate(shard_paths, start=1):
        print(f"[{i}/{len(shard_paths)}] Loading {shard_path.name}")
        sd = torch.load(shard_path, map_location="cpu")
        for key, acc in layer_accs.items():
            if key not in sd:
                raise KeyError(f"Missing key '{key}' in shard {shard_path}")
            acc.update(sd[key], eps=eps)
        del sd

    # Finalize per-layer and aggregate global.
    per_layer = []
    for _, acc in sorted(layer_accs.items(), key=lambda kv: kv[1].layer_idx):
        per_layer.append(acc.finalize(bins=bins))

    global_numel = sum(item["numel"] for item in per_layer)
    global_l0 = sum(item["l0_count"] for item in per_layer)
    global_l1 = sum(item["l1"] for item in per_layer)
    global_l2_sq = sum(item["l2"] ** 2 for item in per_layer)
    global_linf = max(item["linf"] for item in per_layer)
    global_sum = sum(item["mean"] * item["numel"] for item in per_layer)
    global_sumsq = sum((item["std"] ** 2 + item["mean"] ** 2) * item["numel"] for item in per_layer)

    global_mean = global_sum / float(global_numel)
    global_var = max(global_sumsq / float(global_numel) - global_mean * global_mean, 0.0)
    global_std = math.sqrt(global_var)
    global_l0_density = global_l0 / float(global_numel)

    layer_densities = torch.tensor([x["l0_density"] for x in per_layer], dtype=torch.float64)
    layer_hoyer_means = torch.tensor(
        [x["row_distributions"]["hoyer_sparsity"]["mean"] for x in per_layer],
        dtype=torch.float64,
    )

    result = {
        "metadata": {
            "stem_dir": str(stem_dir),
            "num_shards": len(shard_paths),
            "eps": eps,
            "layer_key_regex": layer_key_regex,
            "num_layers": len(per_layer),
        },
        "global": {
            "numel": global_numel,
            "l0_count": global_l0,
            "l0_density": global_l0_density,
            "zero_fraction": 1.0 - global_l0_density,
            "l1": global_l1,
            "l2": math.sqrt(global_l2_sq),
            "linf": global_linf,
            "mean": global_mean,
            "std": global_std,
        },
        "across_layers": {
            "l0_density": summarize_vector(layer_densities),
            "mean_hoyer_sparsity": summarize_vector(layer_hoyer_means, hist_range=(0.0, 1.0), bins=bins),
        },
        "per_layer": per_layer,
    }
    return result


def print_human_summary(report: Dict, top_k: int):
    g = report["global"]
    print("\n=== Global STEM Embedding Stats ===")
    print(f"numel:         {g['numel']:,}")
    print(f"L0 density:    {g['l0_density']:.8f}  (zero_fraction={g['zero_fraction']:.8f})")
    print(f"L1:            {g['l1']:.6e}")
    print(f"L2:            {g['l2']:.6e}")
    print(f"Linf:          {g['linf']:.6e}")
    print(f"mean/std:      {g['mean']:.6e} / {g['std']:.6e}")

    per_layer = report["per_layer"]
    ranked_sparse = sorted(per_layer, key=lambda x: x["l0_density"])
    ranked_dense = sorted(per_layer, key=lambda x: x["l0_density"], reverse=True)

    print(f"\n=== Top {top_k} Sparsest Layers (by L0 density) ===")
    for item in ranked_sparse[:top_k]:
        layer = item["layer_idx"]
        dens = item["l0_density"]
        hoyer = item["row_distributions"]["hoyer_sparsity"]["mean"]
        print(
            f"layer={layer:>2d}  shape={tuple(item['shape'])}  "
            f"l0_density={dens:.8f}  mean_hoyer={hoyer:.6f}"
        )

    print(f"\n=== Top {top_k} Densest Layers (by L0 density) ===")
    for item in ranked_dense[:top_k]:
        layer = item["layer_idx"]
        dens = item["l0_density"]
        hoyer = item["row_distributions"]["hoyer_sparsity"]["mean"]
        print(
            f"layer={layer:>2d}  shape={tuple(item['shape'])}  "
            f"l0_density={dens:.8f}  mean_hoyer={hoyer:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute sparsity and norm statistics for STEM embedding shards",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to checkpoint root or directly to stem_shards",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Absolute threshold for counting non-zero entries (|x| > eps)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Histogram bins for row distributions in JSON output",
    )
    parser.add_argument(
        "--layer-key-regex",
        type=str,
        default=r"^stem_embeddings\.(\d+)\.weight$",
        help="Regex for selecting embedding keys in each shard (must capture layer index)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full report as JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many layers to print in sparsest/densest summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stem_dir = resolve_stem_dir(args.ckpt_path)
    report = analyze_shards(
        stem_dir=stem_dir,
        eps=args.eps,
        bins=args.bins,
        layer_key_regex=args.layer_key_regex,
    )
    print_human_summary(report, top_k=args.top_k)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
