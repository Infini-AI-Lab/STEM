# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Diagnostic toolkit for STEM-based language models.
# Computes three macro-level metrics for diagnosing STEM behavior:
#   1. Per-position / per-token-type loss decomposition (Delta-ell)
#   2. Residual stream contribution norm analysis
#   3. CKA between STEM FFN outputs and attention outputs (per-layer)
#
# Usage (single GPU, consolidated checkpoint):
#   python -m apps.main.stem_diagnostics config=apps/main/configs/stem_diagnostics.yaml
#
# The script loads a STEM checkpoint, runs forward passes with hooks on
# evaluation data, and writes JSON results + optional wandb logging.

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lingua.args import dataclass_from_dict, dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.data import init_choice_state, setup_sources
from lingua.tokenizer import build_tokenizer

from apps.main.stem_generate import (
    load_consolidated_model_and_tokenizer as load_stem_model,
)
from apps.main.generate import (
    load_consolidated_model_and_tokenizer as load_vanilla_model,
)
from apps.main.stem import (
    StemLMTransformer,
    StemLMTransformerArgs,
    STEM_MODEL_REGISTRY,
)
from apps.main.eval import MODEL_REGISTRY

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================
@dataclass
class DiagnosticsArgs:
    name: str = "stem_diagnostics"
    dump_dir: str = "logs/diagnostics"
    model_type: str = "llama"

    # Checkpoint paths (consolidated or pre-consolidation dirs)
    stem_ckpt_dir: str = ""
    vanilla_ckpt_dir: Optional[str] = None  # If None, STEM-zeroed ablation is used

    # Data for evaluation
    data_root_dir: str = ""
    data_sources: Optional[List[str]] = None  # e.g. ["fineweb_edu"]
    tokenizer_name: str = "tiktoken"
    tokenizer_path: str = ""
    seq_len: int = 2048
    max_batches: int = 50
    batch_size: int = 4

    # Which metrics to compute (all by default)
    compute_loss_decomposition: bool = True
    compute_residual_norms: bool = True
    compute_cka: bool = True

    # Misc
    seed: int = 42
    wandb: Optional[Any] = None


# =============================================================================
# Metric 1: Per-Position Loss Decomposition
# =============================================================================
class PerPositionLossDecomposition:
    """Computes per-position cross-entropy for STEM-enabled vs STEM-disabled forward passes.

    For each position t in a sequence, the loss delta is:
        delta_ell(t) = ell_base(t) - ell_stem(t)
    Positive delta means STEM helps at that position.

    Also stratifies by token frequency bin (rare / medium / frequent) as a proxy
    for token type, since we typically lack syntactic labels at scale.

    Interpretation guide:
    - If delta_ell(t) grows with position t: STEM is providing genuine long-range memory.
    - If delta_ell(t) is flat or concentrated at small t: STEM acts as a redundant
      local feature buffer (failure mode).
    - Per-frequency-bin: STEM should help rare/content tokens more than frequent/function
      tokens. If the reverse holds, STEM is overfitting surface statistics.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_freq_bins: int = 3):
        self.seq_len = seq_len
        self.n_freq_bins = n_freq_bins
        self.vocab_size = vocab_size

        # Accumulators: (seq_len,) tensors
        self.stem_loss_sum = torch.zeros(seq_len, dtype=torch.float64)
        self.base_loss_sum = torch.zeros(seq_len, dtype=torch.float64)
        self.count = torch.zeros(seq_len, dtype=torch.int64)

        # Per-frequency-bin accumulators
        self.freq_bin_stem = defaultdict(float)
        self.freq_bin_base = defaultdict(float)
        self.freq_bin_count = defaultdict(int)

        # Token frequency counter (built incrementally from evaluation data)
        self.token_counts = torch.zeros(vocab_size, dtype=torch.int64)

    def _compute_per_token_ce(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token cross-entropy loss.

        Args:
            logits: (B, S, V) raw logits
            targets: (B, S) target token IDs

        Returns:
            (B, S) per-token negative log-likelihood
        """
        log_probs = F.log_softmax(logits.float(), dim=-1)
        per_token_nll = -log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        return per_token_nll

    def _get_freq_bin(self, token_id: int) -> int:
        """Map a token to a frequency bin.

        Returns:
            0 = rare (freq < 1e-5), 1 = medium, 2 = frequent (freq >= 1e-3) -> Adjust freq thresholds accordingly
        """
        total = self.token_counts.sum().item()
        if total == 0:
            return 1
        freq = self.token_counts[token_id].item() / total
        if freq < 1e-5:
            return 0
        elif freq < 1e-3:
            return 1
        else:
            return 2

    @torch.no_grad()
    def update(
        self,
        stem_logits: torch.Tensor,
        base_logits: torch.Tensor,
        targets: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        """Accumulate per-position and per-frequency-bin loss statistics.

        Args:
            stem_logits: (B, S, V) from STEM-enabled forward pass
            base_logits: (B, S, V) from STEM-disabled (or vanilla) forward pass
            targets: (B, S) target token IDs
            input_ids: (B, S) input token IDs (for frequency binning)
        """
        stem_ce = self._compute_per_token_ce(stem_logits, targets)  # (B, S)
        base_ce = self._compute_per_token_ce(base_logits, targets)

        B, S = stem_ce.shape
        S_eff = min(S, self.seq_len)

        # Per-position accumulation
        self.stem_loss_sum[:S_eff] += stem_ce[:, :S_eff].sum(dim=0).cpu().double()
        self.base_loss_sum[:S_eff] += base_ce[:, :S_eff].sum(dim=0).cpu().double()
        self.count[:S_eff] += B

        # Token frequency tracking
        self.token_counts.scatter_add_(
            0,
            input_ids.flatten().cpu(),
            torch.ones(input_ids.numel(), dtype=torch.int64),
        )

        # Per-frequency-bin accumulation (vectorized over batch, loop over positions)
        targets_cpu = targets[:, :S_eff].cpu()
        stem_ce_cpu = stem_ce[:, :S_eff].cpu()
        base_ce_cpu = base_ce[:, :S_eff].cpu()
        for b in range(B):
            for t in range(S_eff):
                tok = targets_cpu[b, t].item()
                fbin = self._get_freq_bin(tok)
                self.freq_bin_stem[fbin] += stem_ce_cpu[b, t].item()
                self.freq_bin_base[fbin] += base_ce_cpu[b, t].item()
                self.freq_bin_count[fbin] += 1

    def compute(self) -> Dict[str, Any]:
        """Compute final per-position and per-frequency-bin metrics.

        Returns:
            Dictionary with keys:
            - per_position: binned position-wise delta_ell, stem_ce, base_ce
            - per_freq_bin: rare/medium/frequent token-type breakdown
            - global_delta_ell: scalar summary (positive = STEM helps overall)
            - global_stem_ppl / global_base_ppl: perplexities
        """
        mask = self.count > 0
        stem_mean = torch.zeros_like(self.stem_loss_sum)
        base_mean = torch.zeros_like(self.base_loss_sum)
        stem_mean[mask] = self.stem_loss_sum[mask] / self.count[mask].double()
        base_mean[mask] = self.base_loss_sum[mask] / self.count[mask].double()
        delta = base_mean - stem_mean  # positive = STEM helps

        # Aggregate into position bins for readability
        n_bins = 16
        bin_edges = torch.linspace(0, self.seq_len, n_bins + 1).long()
        position_bins = {}
        for i in range(n_bins):
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
            if hi <= lo:
                continue
            bin_mask = mask[lo:hi]
            if bin_mask.any():
                position_bins[f"pos_{lo}_{hi}"] = {
                    "delta_ell_mean": delta[lo:hi][bin_mask].mean().item(),
                    "stem_ce_mean": stem_mean[lo:hi][bin_mask].mean().item(),
                    "base_ce_mean": base_mean[lo:hi][bin_mask].mean().item(),
                }

        # Per-frequency-bin results
        freq_bins = {}
        bin_names = {0: "rare", 1: "medium", 2: "frequent"}
        for fbin in sorted(self.freq_bin_count.keys()):
            cnt = self.freq_bin_count[fbin]
            if cnt > 0:
                freq_bins[bin_names.get(fbin, str(fbin))] = {
                    "delta_ell_mean": (
                        self.freq_bin_base[fbin] - self.freq_bin_stem[fbin]
                    )
                    / cnt,
                    "stem_ce_mean": self.freq_bin_stem[fbin] / cnt,
                    "base_ce_mean": self.freq_bin_base[fbin] / cnt,
                    "count": cnt,
                }

        valid = mask.sum().item()
        return {
            "per_position": position_bins,
            "per_freq_bin": freq_bins,
            "global_delta_ell": delta[mask].mean().item() if valid > 0 else 0.0,
            "global_stem_ppl": (
                math.exp(stem_mean[mask].mean().item()) if valid > 0 else float("inf")
            ),
            "global_base_ppl": (
                math.exp(base_mean[mask].mean().item()) if valid > 0 else float("inf")
            ),
            "total_tokens": valid,
        }


# =============================================================================
# Metric 2: Residual Stream Contribution Norm Analysis
# =============================================================================
class ResidualStreamAnalyzer:
    """Hooks into the STEM model to decompose the residual stream.

    For each layer l, captures:
      - attn_contrib: output of the attention sublayer (before residual add)
      - ffn_contrib:  output of the FFN sublayer (before residual add)
      - h_post:       the full residual stream after layer l

    Then computes per-layer:
      - ||attn_contrib|| / ||h_post||    (attention contribution fraction)
      - ||ffn_contrib||  / ||h_post||    (FFN/STEM contribution fraction)
      - cosine(ffn_contrib, h_post)      (alignment: is the model "reading" STEM?)
      - cosine(attn_contrib, h_post)     (baseline comparison)

    Interpretation guide:
    - If ffn_contribution_ratio is near-zero at STEM layers: the model learned to
      gate out STEM (dead module). Check initialization and warmup schedule.
    - If ffn_residual_cosine is near-zero: STEM writes in a subspace the downstream
      layers don't read from ("language barrier"). Redesign injection mechanism.
    - Compare STEM vs non-STEM layers: STEM layers should have comparable or higher
      ffn_contribution_ratio if STEM is providing useful features.
    """

    def __init__(self, model: StemLMTransformer):
        self.model = model
        self.stem_layer_set = set(model.stem_layers)
        self.n_layers = len(model.layers)
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Running accumulators: layer_idx -> scalar sums
        self.attn_norm_sum: Dict[int, float] = defaultdict(float)
        self.ffn_norm_sum: Dict[int, float] = defaultdict(float)
        self.residual_norm_sum: Dict[int, float] = defaultdict(float)
        self.ffn_cos_sum: Dict[int, float] = defaultdict(float)
        self.attn_cos_sum: Dict[int, float] = defaultdict(float)
        self.count: Dict[int, int] = defaultdict(int)

    def install_hooks(self):
        """Install forward hooks on attention, FFN, and block modules.

        Hook architecture (per block):
          1. attention module hook  -> captures attn output (additive contribution)
          2. feed_forward module hook -> captures FFN output (additive contribution)
          3. block-level hook       -> reads both, computes norms/cosines, accumulates
        """
        self._hooks = []

        for layer_idx, block in enumerate(self.model.layers):
            attn_storage = {"output": None}
            ffn_storage = {"output": None}

            def _make_sublayer_hook(storage):
                def hook(module, args, output):
                    storage["output"] = output.detach()
                return hook

            h_attn = block.attention.register_forward_hook(
                _make_sublayer_hook(attn_storage)
            )
            h_ffn = block.feed_forward.register_forward_hook(
                _make_sublayer_hook(ffn_storage)
            )

            def _make_block_hook(l_idx, a_store, f_store):
                def hook(module, args, output):
                    if a_store["output"] is None or f_store["output"] is None:
                        return
                    with torch.no_grad():
                        attn_out = a_store["output"]  # (B, S, D)
                        ffn_out = f_store["output"]  # (B, S, D)
                        residual = output  # (B, S, D) — full residual stream post-block

                        # Flatten to (N, D) for per-token statistics
                        attn_flat = attn_out.reshape(-1, attn_out.shape[-1])
                        ffn_flat = ffn_out.reshape(-1, ffn_out.shape[-1])
                        res_flat = residual.reshape(-1, residual.shape[-1])

                        N = attn_flat.shape[0]

                        # Per-token norms
                        attn_norms = attn_flat.norm(dim=-1)
                        ffn_norms = ffn_flat.norm(dim=-1)
                        res_norms = res_flat.norm(dim=-1)

                        # Cosine similarity
                        ffn_cos = F.cosine_similarity(ffn_flat, res_flat, dim=-1)
                        attn_cos = F.cosine_similarity(attn_flat, res_flat, dim=-1)

                        self.attn_norm_sum[l_idx] += attn_norms.sum().item()
                        self.ffn_norm_sum[l_idx] += ffn_norms.sum().item()
                        self.residual_norm_sum[l_idx] += res_norms.sum().item()
                        self.ffn_cos_sum[l_idx] += ffn_cos.sum().item()
                        self.attn_cos_sum[l_idx] += attn_cos.sum().item()
                        self.count[l_idx] += N

                    # Prevent memory accumulation
                    a_store["output"] = None
                    f_store["output"] = None

                return hook

            h_block = block.register_forward_hook(
                _make_block_hook(layer_idx, attn_storage, ffn_storage)
            )
            self._hooks.extend([h_attn, h_ffn, h_block])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute(self) -> Dict[str, Any]:
        results = {}
        for l_idx in range(self.n_layers):
            cnt = self.count.get(l_idx, 0)
            if cnt == 0:
                continue
            is_stem = l_idx in self.stem_layer_set
            res_norm = self.residual_norm_sum[l_idx] / cnt
            attn_norm = self.attn_norm_sum[l_idx] / cnt
            ffn_norm = self.ffn_norm_sum[l_idx] / cnt

            results[f"layer_{l_idx}"] = {
                "is_stem_layer": is_stem,
                "attn_norm_mean": attn_norm,
                "ffn_norm_mean": ffn_norm,
                "residual_norm_mean": res_norm,
                "ffn_contribution_ratio": ffn_norm / res_norm if res_norm > 0 else 0.0,
                "attn_contribution_ratio": attn_norm / res_norm if res_norm > 0 else 0.0,
                "ffn_residual_cosine": self.ffn_cos_sum[l_idx] / cnt,
                "attn_residual_cosine": self.attn_cos_sum[l_idx] / cnt,
            }

        # Summary: STEM vs non-STEM layers
        stem_entries = [v for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer")]
        nonstem_entries = [v for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is False]

        def _mean(entries, key):
            vals = [e[key] for e in entries]
            return sum(vals) / len(vals) if vals else 0.0

        results["summary"] = {
            "stem_layers_mean_ffn_ratio": _mean(stem_entries, "ffn_contribution_ratio"),
            "nonstem_layers_mean_ffn_ratio": _mean(nonstem_entries, "ffn_contribution_ratio"),
            "stem_layers_mean_ffn_cos": _mean(stem_entries, "ffn_residual_cosine"),
            "nonstem_layers_mean_ffn_cos": _mean(nonstem_entries, "ffn_residual_cosine"),
            "stem_layers_mean_attn_ratio": _mean(stem_entries, "attn_contribution_ratio"),
            "nonstem_layers_mean_attn_ratio": _mean(nonstem_entries, "attn_contribution_ratio"),
        }
        return results


# =============================================================================
# Metric 3: Linear CKA Between STEM FFN Outputs and Attention Outputs
# =============================================================================
class CKAAnalyzer:
    """Computes linear CKA between FFN outputs and attention outputs per layer.

    Linear CKA (Kornblith et al., 2019):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    where X, Y are centered activation matrices.

    For memory efficiency, we accumulate the cross- and auto-Gram matrices
    incrementally across batches: X^T X, Y^T Y, Y^T X.

    Interpretation guide:
    - High CKA between STEM-layer FFN and attention at the same layer:
      STEM is redundant with attention — wasted capacity.
    - Low CKA: STEM learns complementary features — desired behavior.
    - CKA increasing over training: the backbone is absorbing STEM's function
      (the "absorption" phenomenon).
    """

    def __init__(self, model: StemLMTransformer):
        self.model = model
        self.stem_layer_set = set(model.stem_layers)
        self.n_layers = len(model.layers)
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Gram matrix accumulators: layer_idx -> {XtX, YtY, YtX}
        # X = attention output, Y = FFN output
        self._gram: Dict[int, Dict[str, torch.Tensor]] = {}
        self._n_tokens: Dict[int, int] = defaultdict(int)

    def install_hooks(self):
        self._hooks = []

        for layer_idx, block in enumerate(self.model.layers):
            attn_store = {"output": None}
            ffn_store = {"output": None}

            def _make_hook(store):
                def hook(module, args, output):
                    store["output"] = output.detach()
                return hook

            h_attn = block.attention.register_forward_hook(_make_hook(attn_store))
            h_ffn = block.feed_forward.register_forward_hook(_make_hook(ffn_store))

            def _make_block_hook(l_idx, a_store, f_store):
                def hook(module, args, output):
                    if a_store["output"] is None or f_store["output"] is None:
                        return
                    with torch.no_grad():
                        X = a_store["output"].reshape(
                            -1, a_store["output"].shape[-1]
                        ).float()
                        Y = f_store["output"].reshape(
                            -1, f_store["output"].shape[-1]
                        ).float()

                        # Center columns (required for linear CKA)
                        X = X - X.mean(dim=0, keepdim=True)
                        Y = Y - Y.mean(dim=0, keepdim=True)

                        if l_idx not in self._gram:
                            D = X.shape[1]
                            device = X.device
                            self._gram[l_idx] = {
                                "XtX": torch.zeros(D, D, device=device, dtype=torch.float64),
                                "YtY": torch.zeros(D, D, device=device, dtype=torch.float64),
                                "YtX": torch.zeros(D, D, device=device, dtype=torch.float64),
                            }

                        g = self._gram[l_idx]
                        X64 = X.double()
                        Y64 = Y.double()
                        g["XtX"].add_(X64.T @ X64)
                        g["YtY"].add_(Y64.T @ Y64)
                        g["YtX"].add_(Y64.T @ X64)
                        self._n_tokens[l_idx] += X.shape[0]

                    a_store["output"] = None
                    f_store["output"] = None

                return hook

            h_block = block.register_forward_hook(
                _make_block_hook(layer_idx, attn_store, ffn_store)
            )
            self._hooks.extend([h_attn, h_ffn, h_block])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute(self) -> Dict[str, Any]:
        results = {}
        for l_idx in range(self.n_layers):
            if l_idx not in self._gram:
                continue

            g = self._gram[l_idx]
            XtX = g["XtX"]
            YtY = g["YtY"]
            YtX = g["YtX"]

            # CKA = ||YtX||_F^2 / (||XtX||_F * ||YtY||_F)
            numerator = (YtX * YtX).sum().item()
            denom_X = (XtX * XtX).sum().sqrt().item()
            denom_Y = (YtY * YtY).sum().sqrt().item()
            denom = denom_X * denom_Y

            cka = numerator / denom if denom > 1e-12 else 0.0

            results[f"layer_{l_idx}"] = {
                "is_stem_layer": l_idx in self.stem_layer_set,
                "cka_attn_vs_ffn": cka,
                "n_tokens": self._n_tokens[l_idx],
            }

        # Summary statistics
        stem_cka = [
            v["cka_attn_vs_ffn"]
            for v in results.values()
            if isinstance(v, dict) and v.get("is_stem_layer")
        ]
        nonstem_cka = [
            v["cka_attn_vs_ffn"]
            for v in results.values()
            if isinstance(v, dict) and v.get("is_stem_layer") is False
        ]
        results["summary"] = {
            "stem_layers_mean_cka": (
                sum(stem_cka) / len(stem_cka) if stem_cka else 0.0
            ),
            "nonstem_layers_mean_cka": (
                sum(nonstem_cka) / len(nonstem_cka) if nonstem_cka else 0.0
            ),
        }
        return results


# =============================================================================
# Forward Pass Variants for Metric 1
# =============================================================================
@torch.no_grad()
def forward_stem_disabled(
    model: StemLMTransformer, token_values: torch.Tensor
) -> torch.Tensor:
    """Run forward pass with STEM embeddings zeroed out.

    In StemFeedForward: output = w2(silu(w1(x)) * y).
    Setting y=0 zeroes the entire FFN contribution at STEM layers,
    isolating the backbone's own capacity without STEM.

    This is the strongest ablation: it removes STEM's contribution entirely
    rather than replacing it with learned w3(x) (which would require the
    vanilla checkpoint). Use this when no separate vanilla checkpoint exists.
    """
    bsz, seqlen = token_values.shape
    lm = model.lm_transformer

    h = lm.tok_embeddings(token_values)
    mask = lm._create_causal_mask(seqlen, "sdpa", lm.sliding_window)
    freq_cis = lm.rope_embeddings(seqlen=lm.max_seqlen, tok_idx=None)

    for i, layer in enumerate(lm.layers):
        if i in lm.stem_layers:
            y_zeros = torch.zeros(
                bsz, seqlen, layer.feed_forward.hidden_dim,
                device=h.device, dtype=h.dtype,
            )
            h = layer(
                h, freq_cis, y=y_zeros, tok_idx=None,
                mask=mask, attn_impl="sdpa",
            )
        else:
            h = layer(h, freq_cis, tok_idx=None, mask=mask, attn_impl="sdpa")

    logits = lm.output(lm.norm(h))
    return logits


@torch.no_grad()
def forward_stem_enabled(
    model: StemLMTransformer, token_values: torch.Tensor
) -> torch.Tensor:
    """Run normal STEM forward pass, returning logits (not loss)."""
    return model(token_values=token_values, target=None)


@torch.no_grad()
def forward_vanilla(model, token_values: torch.Tensor) -> torch.Tensor:
    """Run vanilla (non-STEM) model forward pass, returning logits."""
    return model(token_values=token_values, target=None)


# =============================================================================
# Data Loading
# =============================================================================
def build_eval_batches(cfg: DiagnosticsArgs, tokenizer):
    """Yield (input_ids, targets) pairs from validation data.

    If no data sources are specified, generates random input for a smoke test.
    Otherwise, reads from the lingua data pipeline's *.val.jsonl files.
    """
    srcs = {}
    for src in (cfg.data_sources or []):
        path = os.path.join(cfg.data_root_dir, src)
        srcs[path] = 1.0

    if not srcs:
        logger.warning(
            "No data sources specified; generating random input for smoke test."
        )
        for _ in range(cfg.max_batches):
            ids = torch.randint(
                0, tokenizer.n_words, (cfg.batch_size, cfg.seq_len + 1)
            )
            yield ids[:, :-1].cuda(), ids[:, 1:].cuda()
        return

    multi_state = init_choice_state("", srcs, 0, 0, 1, "*.val.jsonl")
    path_to_iter = setup_sources(multi_state)

    batch_count = 0
    token_buf = []
    for src in path_to_iter:
        for step, (content, state) in enumerate(path_to_iter[src]):
            if state["current_iter"] > 0 or batch_count >= cfg.max_batches:
                break
            text = content.get("text", content.get("content", ""))
            tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
            token_buf.extend(tokens)

            while len(token_buf) >= (cfg.seq_len + 1) * cfg.batch_size:
                chunk = token_buf[: (cfg.seq_len + 1) * cfg.batch_size]
                token_buf = token_buf[(cfg.seq_len + 1) * cfg.batch_size :]
                t = torch.tensor(chunk, dtype=torch.long).reshape(
                    cfg.batch_size, cfg.seq_len + 1
                )
                yield t[:, :-1].cuda(), t[:, 1:].cuda()
                batch_count += 1
                if batch_count >= cfg.max_batches:
                    return


# =============================================================================
# Main Diagnostic Pipeline
# =============================================================================
@torch.no_grad()
def run_diagnostics(cfg: DiagnosticsArgs):
    """Execute the full three-metric diagnostic pipeline.

    Pipeline:
      Phase 1 — STEM-enabled forward pass with hooks for Metrics 2 (residual
                norms) and 3 (CKA). Single pass over the data.
      Phase 2 — Paired forward passes (STEM-enabled vs STEM-disabled) for
                Metric 1 (loss decomposition). Second pass over the data.

    If a separate vanilla_ckpt_dir is provided, Metric 1 uses that model
    instead of the STEM-zeroed ablation for a more faithful comparison.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Resolve and load STEM checkpoint ---
    stem_ckpt = Path(cfg.stem_ckpt_dir)
    if not (stem_ckpt / "params.json").exists():
        consolidate_path = stem_ckpt / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            consolidate_path = consolidate_checkpoints(cfg.stem_ckpt_dir)
        stem_ckpt = consolidate_path

    if cfg.model_type not in STEM_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}'. "
            f"Available: {list(STEM_MODEL_REGISTRY.keys())}"
        )
    stem_model_cls, stem_args_cls = STEM_MODEL_REGISTRY[cfg.model_type][:2]

    logger.info(f"Loading STEM model from {stem_ckpt}")
    stem_model, tokenizer, train_cfg = load_stem_model(
        str(stem_ckpt),
        model_cls=stem_model_cls,
        model_args_cls=stem_args_cls,
    )
    stem_model.eval()
    logger.info(
        f"STEM model loaded: {len(stem_model.layers)} layers, "
        f"stem_layers={list(stem_model.stem_layers)}"
    )

    # --- Optionally load vanilla model for Metric 1 ---
    vanilla_model = None
    if cfg.vanilla_ckpt_dir and cfg.compute_loss_decomposition:
        van_ckpt = Path(cfg.vanilla_ckpt_dir)
        if not (van_ckpt / "params.json").exists():
            van_consolidate = van_ckpt / CONSOLIDATE_FOLDER
            if not van_consolidate.exists():
                van_consolidate = consolidate_checkpoints(cfg.vanilla_ckpt_dir)
            van_ckpt = van_consolidate

        van_model_cls, van_args_cls = MODEL_REGISTRY.get(
            cfg.model_type, MODEL_REGISTRY["llama"]
        )
        logger.info(f"Loading vanilla model from {van_ckpt}")
        vanilla_model, _, _ = load_vanilla_model(
            str(van_ckpt),
            model_cls=van_model_cls,
            model_args_cls=van_args_cls,
        )
        vanilla_model.eval()
        logger.info("Vanilla model loaded")

    # --- Use tokenizer from train config if not explicitly set ---
    if not cfg.tokenizer_path and hasattr(train_cfg, "data"):
        cfg.tokenizer_path = train_cfg.data.tokenizer.path
        cfg.tokenizer_name = train_cfg.data.tokenizer.name
        tokenizer = build_tokenizer(cfg.tokenizer_name, cfg.tokenizer_path)

    # --- Initialize metric objects ---
    loss_decomp = None
    residual_analyzer = None
    cka_analyzer = None

    if cfg.compute_loss_decomposition:
        vocab_size = stem_model.lm_transformer.tok_embeddings.weight.shape[0]
        loss_decomp = PerPositionLossDecomposition(
            vocab_size=vocab_size, seq_len=cfg.seq_len
        )
        logger.info("Metric 1 (Loss Decomposition) initialized")

    if cfg.compute_residual_norms:
        residual_analyzer = ResidualStreamAnalyzer(stem_model)
        logger.info("Metric 2 (Residual Stream Norms) initialized")

    if cfg.compute_cka:
        cka_analyzer = CKAAnalyzer(stem_model)
        logger.info("Metric 3 (CKA) initialized")

    # =================================================================
    # Phase 1: STEM-enabled forward pass with hooks (Metrics 2 & 3)
    # =================================================================
    if cfg.compute_residual_norms or cfg.compute_cka:
        if residual_analyzer:
            residual_analyzer.install_hooks()
        if cka_analyzer:
            cka_analyzer.install_hooks()

        logger.info("Phase 1: STEM-enabled forward pass with hooks (Metrics 2 & 3)")
        for batch_idx, (input_ids, targets) in enumerate(
            build_eval_batches(cfg, tokenizer)
        ):
            _ = forward_stem_enabled(stem_model, input_ids)
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{cfg.max_batches}")

        if residual_analyzer:
            residual_analyzer.remove_hooks()
        if cka_analyzer:
            cka_analyzer.remove_hooks()

    # =================================================================
    # Phase 2: Loss decomposition (STEM-enabled vs baseline)
    # =================================================================
    if cfg.compute_loss_decomposition:
        logger.info("Phase 2: Loss decomposition (Metric 1)")
        for batch_idx, (input_ids, targets) in enumerate(
            build_eval_batches(cfg, tokenizer)
        ):
            stem_logits = forward_stem_enabled(stem_model, input_ids)
            if vanilla_model is not None:
                base_logits = forward_vanilla(vanilla_model, input_ids)
            else:
                base_logits = forward_stem_disabled(stem_model, input_ids)
            loss_decomp.update(stem_logits, base_logits, targets, input_ids)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{cfg.max_batches}")

    # =================================================================
    # Compute and save results
    # =================================================================
    results = {
        "config": {
            "stem_ckpt_dir": cfg.stem_ckpt_dir,
            "vanilla_ckpt_dir": cfg.vanilla_ckpt_dir,
            "model_type": cfg.model_type,
            "seq_len": cfg.seq_len,
            "max_batches": cfg.max_batches,
            "batch_size": cfg.batch_size,
            "ablation_mode": (
                "vanilla_checkpoint" if vanilla_model is not None else "stem_zeroed"
            ),
        }
    }

    if loss_decomp:
        results["loss_decomposition"] = loss_decomp.compute()
        ld = results["loss_decomposition"]
        logger.info(
            f"Metric 1 — Global delta_ell: {ld['global_delta_ell']:.4f}, "
            f"STEM PPL: {ld['global_stem_ppl']:.2f}, "
            f"Base PPL: {ld['global_base_ppl']:.2f}"
        )

    if residual_analyzer:
        results["residual_norms"] = residual_analyzer.compute()
        s = results["residual_norms"]["summary"]
        logger.info(
            f"Metric 2 — STEM FFN ratio: {s['stem_layers_mean_ffn_ratio']:.4f} "
            f"(non-STEM: {s['nonstem_layers_mean_ffn_ratio']:.4f}), "
            f"STEM FFN cos: {s['stem_layers_mean_ffn_cos']:.4f} "
            f"(non-STEM: {s['nonstem_layers_mean_ffn_cos']:.4f})"
        )

    if cka_analyzer:
        results["cka"] = cka_analyzer.compute()
        s = results["cka"]["summary"]
        logger.info(
            f"Metric 3 — STEM CKA(attn,ffn): {s['stem_layers_mean_cka']:.4f} "
            f"(non-STEM: {s['nonstem_layers_mean_cka']:.4f})"
        )

    # --- Save JSON ---
    out_dir = Path(cfg.dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostics.json"

    def _make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return str(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # --- Optional wandb logging ---
    if cfg.wandb is not None:
        try:
            import wandb as wb

            wandb_kwargs = cfg.wandb if isinstance(cfg.wandb, dict) else {}
            wb.init(**wandb_kwargs)

            flat = {}
            if loss_decomp:
                flat["diag/global_delta_ell"] = ld["global_delta_ell"]
                flat["diag/stem_ppl"] = ld["global_stem_ppl"]
                flat["diag/base_ppl"] = ld["global_base_ppl"]

            if residual_analyzer:
                rn = results["residual_norms"]
                for key, val in rn.items():
                    if key == "summary":
                        for k, v in val.items():
                            flat[f"diag/residual/{k}"] = v
                    elif isinstance(val, dict):
                        flat[f"diag/residual/{key}/ffn_ratio"] = val[
                            "ffn_contribution_ratio"
                        ]
                        flat[f"diag/residual/{key}/ffn_cos"] = val[
                            "ffn_residual_cosine"
                        ]

            if cka_analyzer:
                for key, val in results["cka"].items():
                    if key == "summary":
                        for k, v in val.items():
                            flat[f"diag/cka/{k}"] = v
                    elif isinstance(val, dict):
                        flat[f"diag/cka/{key}"] = val["cka_attn_vs_ffn"]

            wb.log(flat)
            wb.finish()
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return results


# =============================================================================
# CLI entry point
# =============================================================================
def launch_diagnostics(cfg):
    """Entry point from Python (dict or dataclass)."""
    if isinstance(cfg, dict):
        cfg = dataclass_from_dict(DiagnosticsArgs, cfg, strict=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    return run_diagnostics(cfg)


def main():
    """CLI entry point using OmegaConf (matches lingua convention).

    Usage:
        python -m apps.main.stem_diagnostics config=apps/main/configs/stem_diagnostics.yaml
        python -m apps.main.stem_diagnostics config=path/to/config.yaml stem_ckpt_dir=path/to/ckpt
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(DiagnosticsArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_diagnostics(cfg)


if __name__ == "__main__":
    main()