#!/usr/bin/env python3
"""
Compute per-layer up-projection activation norm statistics for the base model.

This script:
1) Loads a base LM checkpoint from a consolidated directory.
2) Streams batches from the pretraining dataloader (e.g. DCLM).
3) Runs only the transformer backbone (no vocab projection) to avoid logits OOM.
4) Collects L2 norms of FFN up-projection activations (feed_forward.w3 output)
   for every token in each layer.
5) Reports mean/std/p25/p50/p75/p90 per layer.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from apps.main.generate import load_consolidated_model_and_tokenizer
from apps.main.transformer import create_causal_mask
from lingua.args import dataclass_from_dict
from lingua.data import DataArgs, build_dataloader_from_args, init_dataloader_state_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-layer up-projection norm statistics on pretraining data",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/Llama-3.2-1B/distcp/consolidated",
        help="Path to consolidated checkpoint directory (contains params.json + consolidated.pth)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="apps/main/configs/llama3_1B.yaml",
        help="YAML config used to build DataArgs",
    )
    parser.add_argument("--max-batches", type=int, default=32, help="Number of batches to process")
    parser.add_argument(
        "--override-seq-len",
        type=int,
        default=None,
        help="Optional data.seq_len override for faster analysis",
    )
    parser.add_argument(
        "--override-batch-size",
        type=int,
        default=None,
        help="Optional data.batch_size override for faster analysis",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "fmha"],
        help="Attention kernel used for analysis forward pass",
    )
    return parser.parse_args()


@torch.no_grad()
def run_backbone_and_collect(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attn_impl: str,
    layer_values: Dict[int, List[torch.Tensor]],
) -> None:
    # Mirror LMTransformer.forward without output projection to save memory.
    _, seqlen = input_ids.shape
    h = model.tok_embeddings(input_ids)
    mask = create_causal_mask(seqlen, attn_impl, model.sliding_window)
    freq_cis = model.rope_embeddings(seqlen=model.max_seqlen, tok_idx=None)

    for layer_idx, layer in enumerate(model.layers):
        h = h + layer.attention(
            layer.attention_norm(h),
            freq_cis,
            tok_idx=None,
            mask=mask,
            attn_impl=attn_impl,
        )

        ffn_in = layer.ffn_norm(h)
        x1 = layer.feed_forward.w1(ffn_in.view_as(ffn_in))
        x3 = layer.feed_forward.w3(ffn_in.view_as(ffn_in))

        # Per-token up-projection vector norm.
        norms = torch.linalg.vector_norm(x3.float(), ord=2, dim=-1).reshape(-1).cpu()
        layer_values[layer_idx].append(norms)

        h = h + layer.feed_forward.w2(F.silu(x1) * x3)


def summarize(x: torch.Tensor) -> Dict[str, float]:
    q = torch.quantile(x, torch.tensor([0.25, 0.50, 0.75, 0.90], dtype=torch.float32))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p25": float(q[0].item()),
        "p50": float(q[1].item()),
        "p75": float(q[2].item()),
        "p90": float(q[3].item()),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this script to run in reasonable time.")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    print(f"Loading base model from: {ckpt_path}")
    model, _, _ = load_consolidated_model_and_tokenizer(str(ckpt_path))
    model.eval()

    cfg = OmegaConf.load(args.data_config)
    data_cfg = dataclass_from_dict(DataArgs, OmegaConf.to_object(cfg.data), strict=False)
    if args.override_seq_len is not None:
        data_cfg.seq_len = args.override_seq_len
    if args.override_batch_size is not None:
        data_cfg.batch_size = args.override_batch_size

    print(
        f"Data settings: root_dir={data_cfg.root_dir}, sources={data_cfg.sources}, "
        f"batch_size={data_cfg.batch_size}, seq_len={data_cfg.seq_len}"
    )

    state = init_dataloader_state_from_args(data_cfg, rank=0, world_size=1)
    n_layers = len(model.layers)
    layer_values: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}

    processed_batches = 0
    processed_tokens = 0
    with build_dataloader_from_args(data_cfg, state=state) as data_loader:
        while processed_batches < args.max_batches:
            batch, state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)
            input_ids = batch[:, :, 0].to(device, non_blocking=True)

            run_backbone_and_collect(
                model=model,
                input_ids=input_ids,
                attn_impl=args.attn_impl,
                layer_values=layer_values,
            )

            processed_batches += 1
            processed_tokens += input_ids.numel()
            if processed_batches % 5 == 0:
                print(
                    f"Processed {processed_batches}/{args.max_batches} batches "
                    f"({processed_tokens:,} tokens)"
                )

    print("\n=== Up-projection (w3 output) L2 norm stats per layer ===")
    print(f"Total tokens processed: {processed_tokens:,}")
    print("layer\tmean\tstd\tp25\tp50\tp75\tp90")
    for layer_idx in range(n_layers):
        values = torch.cat(layer_values[layer_idx], dim=0)
        s = summarize(values)
        print(
            f"{layer_idx}\t{s['mean']:.6f}\t{s['std']:.6f}\t{s['p25']:.6f}\t"
            f"{s['p50']:.6f}\t{s['p75']:.6f}\t{s['p90']:.6f}"
        )


if __name__ == "__main__":
    main()
