#!/usr/bin/env python3
"""
Visualize row-wise STEM embedding distribution stats from JSON output.

Expected input JSON comes from:
  apps/main/stem_embedding_sparsity.py --output-json <path>

For each selected metric and layer, this script plots a histogram panel with
the median (p50) highlighted.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = ["nnz_count", "density", "l1", "l2", "hoyer_sparsity"]


def parse_layer_list(layer_arg: str, available_layers: List[int]) -> List[int]:
    if layer_arg.strip().lower() == "all":
        return list(available_layers)
    out: List[int] = []
    for token in layer_arg.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def get_layer_entry(report: Dict, layer_idx: int) -> Dict:
    for item in report["per_layer"]:
        if int(item["layer_idx"]) == int(layer_idx):
            return item
    raise KeyError(f"Layer {layer_idx} not found in report")


def build_bin_edges(hist_info: Dict) -> np.ndarray:
    n_bins = int(hist_info["bins"])
    lo, hi = hist_info["range"]
    return np.linspace(lo, hi, n_bins + 1)


def plot_metric_panels(
    report: Dict,
    metric: str,
    layer_ids: List[int],
    output_png: Path,
    normalize: bool,
):
    n = len(layer_ids)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle(f"STEM Row Distribution: {metric}", fontsize=14, y=0.995)

    for plot_i, layer_id in enumerate(layer_ids):
        r = plot_i // cols
        c = plot_i % cols
        ax = axes[r][c]

        layer_entry = get_layer_entry(report, layer_id)
        metric_stats = layer_entry["row_distributions"][metric]
        if "histogram" not in metric_stats:
            raise KeyError(
                f"Metric '{metric}' in layer {layer_id} has no histogram. "
                "Re-run analyzer with --bins > 0."
            )

        hist = metric_stats["histogram"]
        counts = np.array(hist["counts"], dtype=float)
        if normalize and counts.sum() > 0:
            counts = counts / counts.sum()

        edges = build_bin_edges(hist)
        widths = np.diff(edges)
        centers = edges[:-1] + 0.5 * widths
        ax.bar(centers, counts, width=widths, align="center", alpha=0.75, color="#4472c4")

        median = float(metric_stats["p50"])
        ax.axvline(median, color="crimson", linestyle="--", linewidth=2.0, label=f"median={median:.4g}")

        ax.set_title(f"Layer {layer_id}")
        ax.set_xlabel(metric)
        ax.set_ylabel("probability" if normalize else "count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    # Hide empty panels
    for plot_i in range(n, rows * cols):
        r = plot_i // cols
        c = plot_i % cols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot row distribution stats from stem_embedding_sparsity JSON output",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to JSON report produced by stem_embedding_sparsity.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to plot",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer IDs (e.g. 0,1,2) or 'all'",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Plot normalized histogram probability instead of raw counts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with in_path.open("r") as f:
        report = json.load(f)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    available_layers = [int(x["layer_idx"]) for x in report["per_layer"]]
    layer_ids = parse_layer_list(args.layers, available_layers)

    for layer_id in layer_ids:
        if layer_id not in available_layers:
            raise ValueError(f"Requested layer {layer_id} not in available layers: {available_layers}")

    for metric in metrics:
        png_path = out_dir / f"row_dist_{metric}.png"
        plot_metric_panels(
            report=report,
            metric=metric,
            layer_ids=layer_ids,
            output_png=png_path,
            normalize=args.normalize,
        )
        print(f"Saved {png_path}")

if __name__ == "__main__":
    main()
