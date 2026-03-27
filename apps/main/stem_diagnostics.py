# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Diagnostic toolkit for STEM-based language models.
# Computes five metrics for diagnosing STEM behavior:
#   1. Per-position / per-token-type loss decomposition (Delta-ell)
#   2. Residual stream contribution norm analysis
#   3. CKA between STEM FFN outputs and attention outputs (per-layer)
#   A. Cross-model hidden state divergence (STEM vs vanilla, per-layer)
#   B. STEM embedding effective rank and spectral energy
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
from lingua.stem_checkpoint import CONSOLIDATE_STEM_NAME
from lingua.data import init_choice_state, setup_sources
from lingua.tokenizer import build_tokenizer
from lingua.stem_dist_utils import ParallelEmbedding

from apps.main.stem_generate import (
    load_consolidated_model_and_tokenizer as load_stem_model,
)
from apps.main.generate import (
    load_consolidated_model_and_tokenizer as load_vanilla_model,
)
from apps.main.stem import (
    StemLMTransformer,
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

    # Checkpoint paths
    stem_ckpt_dir: str = ""
    vanilla_ckpt_dir: Optional[str] = None

    # Data for evaluation
    data_root_dir: str = ""
    data_sources: Optional[List[str]] = None
    tokenizer_name: str = "tiktoken"
    tokenizer_path: str = ""
    seq_len: int = 2048
    max_batches: int = 50
    batch_size: int = 4

    # Which metrics to compute
    compute_loss_decomposition: bool = True
    compute_residual_norms: bool = True
    compute_cka: bool = True
    compute_hidden_divergence: bool = True   # Metric A: requires vanilla_ckpt_dir
    compute_embedding_rank: bool = True      # Metric B: no forward pass needed

    # Misc
    seed: int = 42
    wandb: Optional[Any] = None


# =============================================================================
# Metric 1: Per-Position Loss Decomposition
# =============================================================================
class PerPositionLossDecomposition:
    """Per-position cross-entropy for STEM-enabled vs STEM-disabled forward passes.

    delta_ell(t) = ell_base(t) - ell_stem(t). Positive = STEM helps.
    Also stratifies by token frequency bin (rare / medium / frequent).
    """

    def __init__(self, vocab_size: int, seq_len: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.stem_loss_sum = torch.zeros(seq_len, dtype=torch.float64)
        self.base_loss_sum = torch.zeros(seq_len, dtype=torch.float64)
        self.count = torch.zeros(seq_len, dtype=torch.int64)
        self.freq_bin_stem = torch.zeros(3, dtype=torch.float64)
        self.freq_bin_base = torch.zeros(3, dtype=torch.float64)
        self.freq_bin_count = torch.zeros(3, dtype=torch.int64)
        self.token_counts = torch.zeros(vocab_size, dtype=torch.int64)

    def _compute_per_token_ce(self, logits, targets):
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    def _compute_freq_bins_vectorized(self, token_ids_flat):
        total = self.token_counts.sum().item()
        if total == 0:
            return torch.ones(token_ids_flat.shape[0], dtype=torch.long)
        freqs = self.token_counts[token_ids_flat].float() / total
        bins = torch.ones(token_ids_flat.shape[0], dtype=torch.long)
        bins[freqs < 1e-5] = 0
        bins[freqs >= 1e-3] = 2
        return bins

    @torch.no_grad()
    def update(self, stem_logits, base_logits, targets, input_ids):
        stem_ce = self._compute_per_token_ce(stem_logits, targets)
        base_ce = self._compute_per_token_ce(base_logits, targets)
        B, S = stem_ce.shape
        S_eff = min(S, self.seq_len)
        stem_cpu = stem_ce[:, :S_eff].cpu().double()
        base_cpu = base_ce[:, :S_eff].cpu().double()
        targets_cpu = targets[:, :S_eff].cpu()
        self.stem_loss_sum[:S_eff] += stem_cpu.sum(dim=0)
        self.base_loss_sum[:S_eff] += base_cpu.sum(dim=0)
        self.count[:S_eff] += B
        self.token_counts.scatter_add_(
            0, input_ids.cpu().flatten(),
            torch.ones(input_ids.numel(), dtype=torch.int64),
        )
        targets_flat = targets_cpu.reshape(-1)
        bins = self._compute_freq_bins_vectorized(targets_flat)
        stem_flat = stem_cpu.reshape(-1)
        base_flat = base_cpu.reshape(-1)
        for b_id in range(3):
            mask_b = bins == b_id
            if mask_b.any():
                self.freq_bin_stem[b_id] += stem_flat[mask_b].sum().item()
                self.freq_bin_base[b_id] += base_flat[mask_b].sum().item()
                self.freq_bin_count[b_id] += mask_b.sum().item()

    def compute(self):
        mask = self.count > 0
        stem_mean = torch.zeros_like(self.stem_loss_sum)
        base_mean = torch.zeros_like(self.base_loss_sum)
        stem_mean[mask] = self.stem_loss_sum[mask] / self.count[mask].double()
        base_mean[mask] = self.base_loss_sum[mask] / self.count[mask].double()
        delta = base_mean - stem_mean
        n_bins = 16
        bin_edges = torch.linspace(0, self.seq_len, n_bins + 1).long()
        position_bins = {}
        for i in range(n_bins):
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
            if hi <= lo:
                continue
            bm = mask[lo:hi]
            if bm.any():
                position_bins[f"pos_{lo}_{hi}"] = {
                    "delta_ell_mean": delta[lo:hi][bm].mean().item(),
                    "stem_ce_mean": stem_mean[lo:hi][bm].mean().item(),
                    "base_ce_mean": base_mean[lo:hi][bm].mean().item(),
                }
        freq_bins = {}
        bn = {0: "rare", 1: "medium", 2: "frequent"}
        for fbin in range(3):
            cnt = self.freq_bin_count[fbin].item()
            if cnt > 0:
                freq_bins[bn[fbin]] = {
                    "delta_ell_mean": (self.freq_bin_base[fbin].item() - self.freq_bin_stem[fbin].item()) / cnt,
                    "stem_ce_mean": self.freq_bin_stem[fbin].item() / cnt,
                    "base_ce_mean": self.freq_bin_base[fbin].item() / cnt,
                    "count": int(cnt),
                }
        valid = mask.sum().item()
        return {
            "per_position": position_bins,
            "per_freq_bin": freq_bins,
            "global_delta_ell": delta[mask].mean().item() if valid > 0 else 0.0,
            "global_stem_ppl": math.exp(stem_mean[mask].mean().item()) if valid > 0 else float("inf"),
            "global_base_ppl": math.exp(base_mean[mask].mean().item()) if valid > 0 else float("inf"),
            "total_tokens": int(valid),
        }


# =============================================================================
# Helpers
# =============================================================================
def _is_postnorm_block(block: nn.Module) -> bool:
    """Detect OLMo3-style post-norm blocks."""
    return hasattr(block, "post_attention_norm") and hasattr(block, "post_feedforward_norm")


# =============================================================================
# Metric 2: Residual Stream Contribution Norm Analysis
# =============================================================================
class ResidualStreamAnalyzer:
    """Per-layer norms and cosine alignment of attention/FFN contributions
    relative to the full residual stream. Handles pre-norm and post-norm blocks."""

    def __init__(self, model: StemLMTransformer):
        self.model = model
        self.stem_layer_set = set(model.stem_layers)
        self.n_layers = len(model.layers)
        self._hooks = []
        self.attn_norm_sum = defaultdict(float)
        self.ffn_norm_sum = defaultdict(float)
        self.residual_norm_sum = defaultdict(float)
        self.ffn_cos_sum = defaultdict(float)
        self.attn_cos_sum = defaultdict(float)
        self.count = defaultdict(int)

    def install_hooks(self):
        self._hooks = []
        for layer_idx, block in enumerate(self.model.layers):
            attn_s = {"o": None}
            ffn_s = {"o": None}
            is_pn = _is_postnorm_block(block)

            def _sh(s):
                def hook(m, a, o):
                    s["o"] = o.detach()
                return hook

            h1 = block.attention.register_forward_hook(_sh(attn_s))
            h2 = block.feed_forward.register_forward_hook(_sh(ffn_s))

            def _bh(li, a_s, f_s, blk, pn):
                def hook(m, a, o):
                    if a_s["o"] is None or f_s["o"] is None:
                        return
                    with torch.no_grad():
                        ao = a_s["o"]
                        fo = f_s["o"]
                        if pn:
                            ao = blk.post_attention_norm(ao)
                            fo = blk.post_feedforward_norm(fo)
                        af = ao.reshape(-1, ao.shape[-1])
                        ff = fo.reshape(-1, fo.shape[-1])
                        rf = o.reshape(-1, o.shape[-1])
                        N = af.shape[0]
                        self.attn_norm_sum[li] += af.norm(dim=-1).sum().item()
                        self.ffn_norm_sum[li] += ff.norm(dim=-1).sum().item()
                        self.residual_norm_sum[li] += rf.norm(dim=-1).sum().item()
                        self.ffn_cos_sum[li] += F.cosine_similarity(ff, rf, dim=-1).sum().item()
                        self.attn_cos_sum[li] += F.cosine_similarity(af, rf, dim=-1).sum().item()
                        self.count[li] += N
                    a_s["o"] = None
                    f_s["o"] = None
                return hook

            h3 = block.register_forward_hook(_bh(layer_idx, attn_s, ffn_s, block, is_pn))
            self._hooks.extend([h1, h2, h3])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute(self):
        results = {}
        for li in range(self.n_layers):
            cnt = self.count.get(li, 0)
            if cnt == 0:
                continue
            rn = self.residual_norm_sum[li] / cnt
            results[f"layer_{li}"] = {
                "is_stem_layer": li in self.stem_layer_set,
                "attn_norm_mean": self.attn_norm_sum[li] / cnt,
                "ffn_norm_mean": self.ffn_norm_sum[li] / cnt,
                "residual_norm_mean": rn,
                "ffn_contribution_ratio": (self.ffn_norm_sum[li] / cnt) / rn if rn > 0 else 0.0,
                "attn_contribution_ratio": (self.attn_norm_sum[li] / cnt) / rn if rn > 0 else 0.0,
                "ffn_residual_cosine": self.ffn_cos_sum[li] / cnt,
                "attn_residual_cosine": self.attn_cos_sum[li] / cnt,
            }
        se = [v for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is True]
        ne = [v for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is False]
        _m = lambda entries, k: sum(e[k] for e in entries) / len(entries) if entries else 0.0
        results["summary"] = {
            "stem_layers_mean_ffn_ratio": _m(se, "ffn_contribution_ratio"),
            "nonstem_layers_mean_ffn_ratio": _m(ne, "ffn_contribution_ratio"),
            "stem_layers_mean_ffn_cos": _m(se, "ffn_residual_cosine"),
            "nonstem_layers_mean_ffn_cos": _m(ne, "ffn_residual_cosine"),
            "stem_layers_mean_attn_ratio": _m(se, "attn_contribution_ratio"),
            "nonstem_layers_mean_attn_ratio": _m(ne, "attn_contribution_ratio"),
        }
        return results


# =============================================================================
# Metric 3: Linear CKA (deferred global centering, Grams on CPU)
# =============================================================================
class CKAAnalyzer:
    """Linear CKA between FFN and attention outputs per layer.
    Accumulates uncentered Grams + column sums on CPU; applies centering at compute()."""

    def __init__(self, model: StemLMTransformer):
        self.model = model
        self.stem_layer_set = set(model.stem_layers)
        self.n_layers = len(model.layers)
        self._hooks = []
        self._accum: Dict[int, Dict[str, Any]] = {}

    def install_hooks(self):
        self._hooks = []
        for layer_idx, block in enumerate(self.model.layers):
            a_s = {"o": None}
            f_s = {"o": None}
            is_pn = _is_postnorm_block(block)

            def _sh(s):
                def hook(m, a, o):
                    s["o"] = o.detach()
                return hook

            h1 = block.attention.register_forward_hook(_sh(a_s))
            h2 = block.feed_forward.register_forward_hook(_sh(f_s))

            def _bh(li, a_st, f_st, blk, pn):
                def hook(m, a, o):
                    if a_st["o"] is None or f_st["o"] is None:
                        return
                    with torch.no_grad():
                        ao = a_st["o"]
                        fo = f_st["o"]
                        if pn:
                            ao = blk.post_attention_norm(ao)
                            fo = blk.post_feedforward_norm(fo)
                        X = ao.reshape(-1, ao.shape[-1]).float()
                        Y = fo.reshape(-1, fo.shape[-1]).float()
                        Nb = X.shape[0]
                        if li not in self._accum:
                            D = X.shape[1]
                            self._accum[li] = {
                                "XtX": torch.zeros(D, D, dtype=torch.float64),
                                "YtY": torch.zeros(D, D, dtype=torch.float64),
                                "YtX": torch.zeros(D, D, dtype=torch.float64),
                                "sum_X": torch.zeros(D, dtype=torch.float64),
                                "sum_Y": torch.zeros(D, dtype=torch.float64),
                                "N": 0,
                            }
                        X64 = X.double()
                        Y64 = Y.double()
                        acc = self._accum[li]
                        acc["XtX"].add_((X64.T @ X64).cpu())
                        acc["YtY"].add_((Y64.T @ Y64).cpu())
                        acc["YtX"].add_((Y64.T @ X64).cpu())
                        acc["sum_X"].add_(X64.sum(dim=0).cpu())
                        acc["sum_Y"].add_(Y64.sum(dim=0).cpu())
                        acc["N"] += Nb
                    a_st["o"] = None
                    f_st["o"] = None
                return hook

            h3 = block.register_forward_hook(_bh(layer_idx, a_s, f_s, block, is_pn))
            self._hooks.extend([h1, h2, h3])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute(self):
        results = {}
        for li in range(self.n_layers):
            if li not in self._accum:
                continue
            acc = self._accum[li]
            N = acc["N"]
            if N == 0:
                continue
            sX, sY = acc["sum_X"], acc["sum_Y"]
            XtX_c = acc["XtX"] - (1.0 / N) * sX.unsqueeze(1) * sX.unsqueeze(0)
            YtY_c = acc["YtY"] - (1.0 / N) * sY.unsqueeze(1) * sY.unsqueeze(0)
            YtX_c = acc["YtX"] - (1.0 / N) * sY.unsqueeze(1) * sX.unsqueeze(0)
            num = (YtX_c * YtX_c).sum().item()
            dX = (XtX_c * XtX_c).sum().sqrt().item()
            dY = (YtY_c * YtY_c).sum().sqrt().item()
            den = dX * dY
            results[f"layer_{li}"] = {
                "is_stem_layer": li in self.stem_layer_set,
                "cka_attn_vs_ffn": num / den if den > 1e-12 else 0.0,
                "n_tokens": N,
            }
        sc = [v["cka_attn_vs_ffn"] for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is True]
        nc = [v["cka_attn_vs_ffn"] for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is False]
        results["summary"] = {
            "stem_layers_mean_cka": sum(sc) / len(sc) if sc else 0.0,
            "nonstem_layers_mean_cka": sum(nc) / len(nc) if nc else 0.0,
        }
        return results


# =============================================================================
# Metric A: Cross-Model Hidden State Divergence
# =============================================================================
class HiddenStateDivergenceAnalyzer:
    """Computes per-layer normalized L2 divergence between STEM and vanilla models.

    For each layer l and each token position, computes:
        d_l = || h_l^stem - h_l^vanilla ||_2  /  || h_l^vanilla ||_2

    Averaged across all tokens and batches.

    Interpretation:
    - Divergence spikes at STEM layers that stay high / grow downstream
      -> STEM injects representation error the backbone cannot correct.
    - Divergence is modest at STEM layers but amplifies downstream
      -> STEM pushes hidden states into an OOD region for pretrained weights.
    """

    def __init__(self, stem_model: StemLMTransformer, vanilla_model: nn.Module):
        self.stem_model = stem_model
        self.vanilla_model = vanilla_model
        self.stem_layer_set = set(stem_model.stem_layers)
        self.n_layers = len(stem_model.layers)
        assert len(vanilla_model.layers) == self.n_layers, (
            f"Layer count mismatch: STEM has {self.n_layers}, "
            f"vanilla has {len(vanilla_model.layers)}"
        )
        self._hooks = []
        self._stem_states: Dict[int, torch.Tensor] = {}
        self._vanilla_states: Dict[int, torch.Tensor] = {}

        # Running accumulators
        self.divergence_sum: Dict[int, float] = defaultdict(float)
        self.count: Dict[int, int] = defaultdict(int)

    def install_hooks(self):
        """Install block-level hooks on both models to capture hidden states."""
        self._hooks = []

        # STEM model hooks
        for li, block in enumerate(self.stem_model.layers):
            def _make_hook(storage, layer_idx):
                def hook(module, args, output):
                    storage[layer_idx] = output.detach()
                return hook
            h = block.register_forward_hook(_make_hook(self._stem_states, li))
            self._hooks.append(h)

        # Vanilla model hooks
        for li, block in enumerate(self.vanilla_model.layers):
            def _make_hook(storage, layer_idx):
                def hook(module, args, output):
                    storage[layer_idx] = output.detach()
                return hook
            h = block.register_forward_hook(_make_hook(self._vanilla_states, li))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def update(self, input_ids: torch.Tensor):
        """Run same input through both models and compute per-layer divergence."""
        self._stem_states.clear()
        self._vanilla_states.clear()

        # Forward through both models
        _ = forward_stem_enabled(self.stem_model, input_ids)
        _ = forward_vanilla(self.vanilla_model, input_ids)

        # Compare hidden states at each layer
        for li in range(self.n_layers):
            h_stem = self._stem_states.get(li)
            h_van = self._vanilla_states.get(li)
            if h_stem is None or h_van is None:
                continue

            # Flatten to (N, D) for per-token computation
            hs = h_stem.reshape(-1, h_stem.shape[-1])
            hv = h_van.reshape(-1, h_van.shape[-1])
            N = hs.shape[0]

            # Normalized L2 divergence: ||h_stem - h_vanilla|| / ||h_vanilla||
            diff_norms = (hs - hv).norm(dim=-1)      # (N,)
            van_norms = hv.norm(dim=-1).clamp(min=1e-8)  # (N,)
            normed_div = diff_norms / van_norms       # (N,)

            self.divergence_sum[li] += normed_div.sum().item()
            self.count[li] += N

        # Free stored states
        self._stem_states.clear()
        self._vanilla_states.clear()

    def compute(self) -> Dict[str, Any]:
        results = {}
        for li in range(self.n_layers):
            cnt = self.count.get(li, 0)
            if cnt == 0:
                continue
            results[f"layer_{li}"] = {
                "is_stem_layer": li in self.stem_layer_set,
                "mean_normalized_divergence": self.divergence_sum[li] / cnt,
            }

        stem_divs = [v["mean_normalized_divergence"] for v in results.values()
                     if isinstance(v, dict) and v.get("is_stem_layer") is True]
        nonstem_divs = [v["mean_normalized_divergence"] for v in results.values()
                        if isinstance(v, dict) and v.get("is_stem_layer") is False]

        # Detect amplification: does divergence grow layer-over-layer?
        all_divs = []
        for li in range(self.n_layers):
            entry = results.get(f"layer_{li}")
            if entry:
                all_divs.append(entry["mean_normalized_divergence"])

        # Simple amplification metric: ratio of last-quarter mean to first-quarter mean
        if len(all_divs) >= 4:
            q = len(all_divs) // 4
            first_q = sum(all_divs[:q]) / q
            last_q = sum(all_divs[-q:]) / q
            amplification_ratio = last_q / first_q if first_q > 1e-12 else 0.0
        else:
            amplification_ratio = 0.0

        results["summary"] = {
            "stem_layers_mean_divergence": sum(stem_divs) / len(stem_divs) if stem_divs else 0.0,
            "nonstem_layers_mean_divergence": sum(nonstem_divs) / len(nonstem_divs) if nonstem_divs else 0.0,
            "overall_mean_divergence": sum(all_divs) / len(all_divs) if all_divs else 0.0,
            "amplification_ratio": amplification_ratio,
        }
        return results


# =============================================================================
# Metric B: STEM Embedding Effective Rank and Spectral Energy
# =============================================================================
class EmbeddingRankAnalyzer:
    """Analyzes the spectral properties of STEM embedding tables.

    Computes effective rank (exponential of normalized singular value entropy)
    and spectral energy concentration (fraction of variance in top-k SVs).

    If a vanilla model is provided, compares against the original w3 weight
    matrix at corresponding layers.

    Interpretation:
    - STEM effective rank << w3 effective rank -> dimensional collapse,
      STEM can't represent the feature diversity the original model had.
    - High spectral energy in top-1 -> representations dominated by a single
      direction (the "mean embedding" problem).
    """

    def __init__(
        self,
        stem_model: StemLMTransformer,
        vanilla_model: Optional[nn.Module] = None,
    ):
        self.stem_model = stem_model
        self.vanilla_model = vanilla_model
        self.stem_layer_indices = list(stem_model.stem_layers)

    @staticmethod
    def _effective_rank(svd_vals: torch.Tensor) -> float:
        """Compute effective rank from singular values.

        erank = exp(-sum(p_i * log(p_i))) where p_i = sigma_i / sum(sigma_j)
        """
        s = svd_vals / svd_vals.sum()
        s = s[s > 1e-30]  # filter near-zeros to avoid log(0)
        entropy = -(s * s.log()).sum()
        return entropy.exp().item()

    @staticmethod
    def _spectral_energy(svd_vals: torch.Tensor, ks: List[int]) -> Dict[str, float]:
        """Fraction of total spectral energy in top-k singular values."""
        total = (svd_vals ** 2).sum()
        result = {}
        for k in ks:
            k_eff = min(k, len(svd_vals))
            result[f"top_{k}"] = ((svd_vals[:k_eff] ** 2).sum() / total).item()
        return result

    @torch.no_grad()
    def compute(self) -> Dict[str, Any]:
        results = {}
        ks = [1, 5, 10, 50]

        for stem_idx, layer_idx in enumerate(self.stem_layer_indices):
            # Get STEM embedding weight: (V, d_ffn)
            emb_weight = self.stem_model.stem_embeddings[stem_idx].weight.data
            V, d_ffn = emb_weight.shape

            # SVD on CPU in float32 for numerical stability
            logger.info(f"  Computing SVD for stem_embeddings[{stem_idx}] (layer {layer_idx}), shape ({V}, {d_ffn})")
            svd_vals = torch.linalg.svdvals(emb_weight.float().cpu())
            # svd_vals has min(V, d_ffn) entries, sorted descending

            erank = self._effective_rank(svd_vals)
            max_rank = len(svd_vals)
            energies = self._spectral_energy(svd_vals, ks)

            entry = {
                "effective_rank": erank,
                "effective_rank_fraction": erank / max_rank,
                "max_possible_rank": max_rank,
                "spectral_energy": energies,
                "top_singular_value": svd_vals[0].item(),
                "smallest_singular_value": svd_vals[-1].item(),
                "condition_number": (svd_vals[0] / svd_vals[-1]).item() if svd_vals[-1] > 1e-30 else float("inf"),
                "shape": [V, d_ffn],
            }

            # Compare against vanilla w3 if available
            if self.vanilla_model is not None:
                w3 = self.vanilla_model.layers[layer_idx].feed_forward.w3.weight.data
                d_ffn_w3, d_model = w3.shape
                logger.info(f"  Computing SVD for vanilla w3 at layer {layer_idx}, shape ({d_ffn_w3}, {d_model})")
                w3_svd = torch.linalg.svdvals(w3.float().cpu())
                w3_erank = self._effective_rank(w3_svd)
                w3_max_rank = len(w3_svd)
                entry["w3_effective_rank"] = w3_erank
                entry["w3_effective_rank_fraction"] = w3_erank / w3_max_rank
                entry["w3_max_possible_rank"] = w3_max_rank
                entry["w3_spectral_energy"] = self._spectral_energy(w3_svd, ks)
                entry["w3_shape"] = [d_ffn_w3, d_model]
                entry["rank_ratio_stem_over_w3"] = (erank / max_rank) / (w3_erank / w3_max_rank) if w3_erank > 0 else 0.0

            results[f"stem_layer_{layer_idx}"] = entry

        # Summary
        eranks = [v["effective_rank_fraction"] for v in results.values() if isinstance(v, dict) and "effective_rank_fraction" in v]
        summary = {
            "mean_effective_rank_fraction": sum(eranks) / len(eranks) if eranks else 0.0,
        }
        if self.vanilla_model is not None:
            w3_eranks = [v["w3_effective_rank_fraction"] for v in results.values()
                         if isinstance(v, dict) and "w3_effective_rank_fraction" in v]
            rank_ratios = [v["rank_ratio_stem_over_w3"] for v in results.values()
                          if isinstance(v, dict) and "rank_ratio_stem_over_w3" in v]
            summary["mean_w3_effective_rank_fraction"] = sum(w3_eranks) / len(w3_eranks) if w3_eranks else 0.0
            summary["mean_rank_ratio_stem_over_w3"] = sum(rank_ratios) / len(rank_ratios) if rank_ratios else 0.0
        results["summary"] = summary
        return results


# =============================================================================
# Forward Pass Variants
# =============================================================================
@torch.no_grad()
def forward_stem_disabled(model: StemLMTransformer, token_values: torch.Tensor) -> torch.Tensor:
    """Forward with STEM embeddings zeroed. StemFeedForward: w2(silu(w1(x)) * 0) = 0."""
    bsz, seqlen = token_values.shape
    lm = model.lm_transformer
    h = lm.tok_embeddings(token_values)
    mask = lm._create_causal_mask(seqlen, "sdpa", lm.sliding_window)
    freq_cis = lm.rope_embeddings(seqlen=lm.max_seqlen, tok_idx=None)
    for i, layer in enumerate(lm.layers):
        if i in lm.stem_layers:
            y_zeros = torch.zeros(bsz, seqlen, layer.feed_forward.hidden_dim,
                                  device=h.device, dtype=h.dtype)
            h = layer(h, freq_cis, y=y_zeros, tok_idx=None, mask=mask, attn_impl="sdpa")
        else:
            h = layer(h, freq_cis, tok_idx=None, mask=mask, attn_impl="sdpa")
    return lm.output(lm.norm(h))


@torch.no_grad()
def forward_stem_enabled(model: StemLMTransformer, token_values: torch.Tensor) -> torch.Tensor:
    return model(token_values=token_values, target=None)


@torch.no_grad()
def forward_vanilla(model, token_values: torch.Tensor) -> torch.Tensor:
    return model(token_values=token_values, target=None)


# =============================================================================
# Data Loading
# =============================================================================
def build_eval_batches(cfg: DiagnosticsArgs, tokenizer):
    srcs = {}
    for src in (cfg.data_sources or []):
        srcs[os.path.join(cfg.data_root_dir, src)] = 1.0
    if not srcs:
        logger.warning("No data sources specified; generating random input for smoke test.")
        for _ in range(cfg.max_batches):
            ids = torch.randint(0, tokenizer.n_words, (cfg.batch_size, cfg.seq_len + 1))
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
            token_buf.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            while len(token_buf) >= (cfg.seq_len + 1) * cfg.batch_size:
                chunk = token_buf[: (cfg.seq_len + 1) * cfg.batch_size]
                token_buf = token_buf[(cfg.seq_len + 1) * cfg.batch_size :]
                t = torch.tensor(chunk, dtype=torch.long).reshape(cfg.batch_size, cfg.seq_len + 1)
                yield t[:, :-1].cuda(), t[:, 1:].cuda()
                batch_count += 1
                if batch_count >= cfg.max_batches:
                    return


# =============================================================================
# Checkpoint Resolution
# =============================================================================
def _resolve_checkpoint(ckpt_dir: str) -> Path:
    """Resolve a checkpoint directory to its consolidated path.

    Mirrors the logic in ``stem_eval.py`` / ``launch_stem_eval`` so that both
    scripts behave identically:

    1. If the directory itself already contains ``params.json`` and a ``.pth``
       file it is treated as a ready-to-use consolidated checkpoint.
    2. Otherwise look for the ``consolidated/consolidated.pth`` file (not just
       the directory — an empty ``consolidated/`` folder is not sufficient).
    3. If neither exists, run DCP-to-torch consolidation.
    """
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_dir}")

    if (
        (ckpt_path / "params.json").exists()
        and next(ckpt_path.glob("*.pth"), None) is not None
    ):
        return ckpt_path

    consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path = Path(consolidate_checkpoints(str(ckpt_path)))

    return consolidate_path


def _validate_stem_embeddings(model: StemLMTransformer, ckpt_path: Path) -> None:
    """Verify that every STEM embedding was loaded from the checkpoint.

    After ``load_consolidated_model_and_tokenizer`` returns, this function
    checks that:
      1. ``consolidated_stem.pth`` (or the shard directory) actually existed
         and was non-empty.
      2. Every ``stem_embeddings.*.weight`` parameter in the model was
         populated from the checkpoint (i.e. none remain at random init).

    Raises ``RuntimeError`` if any STEM embedding appears to be missing from
    the checkpoint so that diagnostics are never run on randomly-initialized
    weights.
    """
    stem_layer_indices = list(model.stem_layers)
    if not stem_layer_indices:
        return

    consolidated_stem_path = ckpt_path / CONSOLIDATE_STEM_NAME
    stem_shards_dir = ckpt_path.parent / "stem_shards"

    has_consolidated_stem = (
        consolidated_stem_path.exists()
        and consolidated_stem_path.stat().st_size > 0
    )
    has_stem_shards = (
        stem_shards_dir.exists()
        and any(stem_shards_dir.glob("stem_model_mp*.pt"))
    )

    if not has_consolidated_stem and not has_stem_shards:
        raise RuntimeError(
            f"STEM embeddings checkpoint not found. "
            f"Looked for '{consolidated_stem_path}' and shard files in "
            f"'{stem_shards_dir}'. Without these, STEM embedding layers "
            f"{stem_layer_indices} would remain randomly initialized."
        )

    loaded_keys = set()
    if has_consolidated_stem:
        stem_dict = torch.load(consolidated_stem_path, map_location="cpu", weights_only=True)
        loaded_keys = set(stem_dict.keys())
        del stem_dict

    expected_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, ParallelEmbedding)):
            weight_key = f"{module_name}.weight" if module_name else "weight"
            if weight_key.startswith("stem_embeddings."):
                expected_keys.add(weight_key)

    if has_consolidated_stem and expected_keys:
        missing = expected_keys - loaded_keys
        if missing:
            raise RuntimeError(
                f"STEM embedding weights missing from checkpoint: {sorted(missing)}. "
                f"These layers would remain randomly initialized. "
                f"Checkpoint at: {consolidated_stem_path}"
            )

    logger.info(
        f"STEM embedding validation passed: {len(expected_keys)} embedding(s) "
        f"verified for layers {stem_layer_indices}"
    )


# =============================================================================
# Main Diagnostic Pipeline
# =============================================================================
@torch.no_grad()
def run_diagnostics(cfg: DiagnosticsArgs):
    torch.manual_seed(cfg.seed)

    # --- Load STEM model ---
    stem_ckpt = _resolve_checkpoint(cfg.stem_ckpt_dir)
    if cfg.model_type not in STEM_MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{cfg.model_type}'. Available: {list(STEM_MODEL_REGISTRY.keys())}")
    stem_model_cls, stem_args_cls = STEM_MODEL_REGISTRY[cfg.model_type][:2]
    logger.info(f"Loading STEM model from {stem_ckpt}")
    _tok_kw = {}
    if cfg.tokenizer_path:
        _tok_kw["tokenizer_path"] = cfg.tokenizer_path
        if cfg.tokenizer_name:
            _tok_kw["tokenizer_name"] = cfg.tokenizer_name
    stem_model, tokenizer, train_cfg = load_stem_model(
        str(stem_ckpt), model_cls=stem_model_cls, model_args_cls=stem_args_cls, **_tok_kw
    )
    stem_model.eval()
    _validate_stem_embeddings(stem_model, stem_ckpt)
    logger.info(f"STEM model loaded: {len(stem_model.layers)} layers, stem_layers={list(stem_model.stem_layers)}")

    # --- Metric B: Embedding rank (no forward pass, run before loading vanilla) ---
    rank_analyzer = None
    rank_results = None
    if cfg.compute_embedding_rank:
        logger.info("Metric B: Computing STEM embedding spectral analysis...")
        rank_analyzer = EmbeddingRankAnalyzer(stem_model, vanilla_model=None)
        rank_results = rank_analyzer.compute()
        logger.info("Metric B: Spectral analysis complete (w3 comparison pending vanilla load)")

    # --- Load vanilla model (needed for Metrics 1-ablation, A, B-w3-comparison) ---
    vanilla_model = None
    needs_vanilla = (
        (cfg.compute_loss_decomposition and cfg.vanilla_ckpt_dir)
        or cfg.compute_hidden_divergence
        or (cfg.compute_embedding_rank and cfg.vanilla_ckpt_dir)
    )
    if needs_vanilla:
        if not cfg.vanilla_ckpt_dir:
            if cfg.compute_hidden_divergence:
                logger.warning("compute_hidden_divergence=True but vanilla_ckpt_dir not set. Skipping Metric A.")
                cfg.compute_hidden_divergence = False
        else:
            van_ckpt = _resolve_checkpoint(cfg.vanilla_ckpt_dir)
            van_model_cls, van_args_cls = MODEL_REGISTRY.get(cfg.model_type, MODEL_REGISTRY["llama"])
            logger.info(f"Loading vanilla model from {van_ckpt}")
            _tok_kw = {}
            if cfg.tokenizer_path:
                _tok_kw["tokenizer_path"] = cfg.tokenizer_path
                if cfg.tokenizer_name:
                    _tok_kw["tokenizer_name"] = cfg.tokenizer_name
            vanilla_model, _, _ = load_vanilla_model(
                str(van_ckpt), model_cls=van_model_cls, model_args_cls=van_args_cls, **_tok_kw)
            vanilla_model.eval()
            logger.info("Vanilla model loaded")

    # --- Metric B update: recompute with w3 comparison if vanilla available ---
    if cfg.compute_embedding_rank and vanilla_model is not None:
        logger.info("Metric B: Recomputing with vanilla w3 comparison...")
        rank_analyzer = EmbeddingRankAnalyzer(stem_model, vanilla_model=vanilla_model)
        rank_results = rank_analyzer.compute()
        s = rank_results["summary"]
        logger.info(
            f"Metric B — STEM erank fraction: {s['mean_effective_rank_fraction']:.4f}, "
            f"w3 erank fraction: {s.get('mean_w3_effective_rank_fraction', 'N/A')}, "
            f"ratio: {s.get('mean_rank_ratio_stem_over_w3', 'N/A')}"
        )

    # --- Resolve tokenizer from train config if needed ---
    if not cfg.tokenizer_path and hasattr(train_cfg, "data"):
        cfg.tokenizer_path = train_cfg.data.tokenizer.path
        cfg.tokenizer_name = train_cfg.data.tokenizer.name
        tokenizer = build_tokenizer(cfg.tokenizer_name, cfg.tokenizer_path)

    # --- Initialize forward-pass-based metrics ---
    loss_decomp = None
    residual_analyzer = None
    cka_analyzer = None
    divergence_analyzer = None

    if cfg.compute_loss_decomposition:
        vocab_size = stem_model.lm_transformer.tok_embeddings.weight.shape[0]
        loss_decomp = PerPositionLossDecomposition(vocab_size=vocab_size, seq_len=cfg.seq_len)

    if cfg.compute_residual_norms:
        residual_analyzer = ResidualStreamAnalyzer(stem_model)

    if cfg.compute_cka:
        cka_analyzer = CKAAnalyzer(stem_model)

    if cfg.compute_hidden_divergence and vanilla_model is not None:
        divergence_analyzer = HiddenStateDivergenceAnalyzer(stem_model, vanilla_model)

    # =================================================================
    # Phase 1: STEM-only hooks (Metrics 2 & 3)
    # =================================================================
    if residual_analyzer or cka_analyzer:
        if residual_analyzer:
            residual_analyzer.install_hooks()
        if cka_analyzer:
            cka_analyzer.install_hooks()
        logger.info("Phase 1: STEM-enabled forward with hooks (Metrics 2 & 3)")
        for bi, (input_ids, targets) in enumerate(build_eval_batches(cfg, tokenizer)):
            _ = forward_stem_enabled(stem_model, input_ids)
            if (bi + 1) % 10 == 0:
                logger.info(f"  Batch {bi + 1}/{cfg.max_batches}")
        if residual_analyzer:
            residual_analyzer.remove_hooks()
        if cka_analyzer:
            cka_analyzer.remove_hooks()
        torch.cuda.empty_cache()

    # =================================================================
    # Phase 2: Loss decomposition (Metric 1)
    # =================================================================
    if loss_decomp:
        logger.info("Phase 2: Loss decomposition (Metric 1)")
        for bi, (input_ids, targets) in enumerate(build_eval_batches(cfg, tokenizer)):
            stem_logits = forward_stem_enabled(stem_model, input_ids)
            base_logits = forward_vanilla(vanilla_model, input_ids) if vanilla_model is not None else forward_stem_disabled(stem_model, input_ids)
            loss_decomp.update(stem_logits, base_logits, targets, input_ids)
            if (bi + 1) % 10 == 0:
                logger.info(f"  Batch {bi + 1}/{cfg.max_batches}")
        torch.cuda.empty_cache()

    # =================================================================
    # Phase 3: Hidden state divergence (Metric A)
    # =================================================================
    if divergence_analyzer:
        divergence_analyzer.install_hooks()
        logger.info("Phase 3: Hidden state divergence (Metric A)")
        for bi, (input_ids, targets) in enumerate(build_eval_batches(cfg, tokenizer)):
            divergence_analyzer.update(input_ids)
            if (bi + 1) % 10 == 0:
                logger.info(f"  Batch {bi + 1}/{cfg.max_batches}")
        divergence_analyzer.remove_hooks()
        torch.cuda.empty_cache()

    # =================================================================
    # Compute and save
    # =================================================================
    results = {
        "config": {
            "stem_ckpt_dir": cfg.stem_ckpt_dir,
            "vanilla_ckpt_dir": cfg.vanilla_ckpt_dir,
            "model_type": cfg.model_type,
            "seq_len": cfg.seq_len,
            "max_batches": cfg.max_batches,
            "batch_size": cfg.batch_size,
            "ablation_mode": "vanilla_checkpoint" if vanilla_model is not None else "stem_zeroed",
        }
    }

    if loss_decomp:
        results["loss_decomposition"] = loss_decomp.compute()
        ld = results["loss_decomposition"]
        logger.info(f"Metric 1 — delta_ell: {ld['global_delta_ell']:.4f}, STEM PPL: {ld['global_stem_ppl']:.2f}, Base PPL: {ld['global_base_ppl']:.2f}")

    if residual_analyzer:
        results["residual_norms"] = residual_analyzer.compute()
        s = results["residual_norms"]["summary"]
        logger.info(f"Metric 2 — STEM FFN ratio: {s['stem_layers_mean_ffn_ratio']:.4f}, cos: {s['stem_layers_mean_ffn_cos']:.4f}")

    if cka_analyzer:
        results["cka"] = cka_analyzer.compute()
        s = results["cka"]["summary"]
        logger.info(f"Metric 3 — STEM CKA: {s['stem_layers_mean_cka']:.4f}, non-STEM: {s['nonstem_layers_mean_cka']:.4f}")

    if divergence_analyzer:
        results["hidden_divergence"] = divergence_analyzer.compute()
        s = results["hidden_divergence"]["summary"]
        logger.info(f"Metric A — STEM div: {s['stem_layers_mean_divergence']:.4f}, non-STEM: {s['nonstem_layers_mean_divergence']:.4f}, amplification: {s['amplification_ratio']:.2f}x")

    if rank_results:
        results["embedding_rank"] = rank_results
        s = rank_results["summary"]
        logger.info(f"Metric B — STEM erank fraction: {s['mean_effective_rank_fraction']:.4f}")

    # --- Save JSON ---
    out_dir = Path(cfg.dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostics.json"

    def _ser(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return str(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_ser(results), f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # --- Optional wandb ---
    if cfg.wandb is not None:
        try:
            import wandb as wb
            wb.init(**(cfg.wandb if isinstance(cfg.wandb, dict) else {}))
            flat = {}
            if loss_decomp:
                flat["diag/global_delta_ell"] = ld["global_delta_ell"]
                flat["diag/stem_ppl"] = ld["global_stem_ppl"]
                flat["diag/base_ppl"] = ld["global_base_ppl"]
            if residual_analyzer:
                for k, v in results["residual_norms"]["summary"].items():
                    flat[f"diag/residual/{k}"] = v
            if cka_analyzer:
                for k, v in results["cka"]["summary"].items():
                    flat[f"diag/cka/{k}"] = v
            if divergence_analyzer:
                for k, v in results["hidden_divergence"]["summary"].items():
                    flat[f"diag/divergence/{k}"] = v
            if rank_results:
                for k, v in rank_results["summary"].items():
                    flat[f"diag/rank/{k}"] = v
            wb.log(flat)
            wb.finish()
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return results


# =============================================================================
# CLI
# =============================================================================
def launch_diagnostics(cfg):
    if isinstance(cfg, dict):
        cfg = dataclass_from_dict(DiagnosticsArgs, cfg, strict=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    return run_diagnostics(cfg)


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_cfg = OmegaConf.structured(DiagnosticsArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_diagnostics(cfg)


if __name__ == "__main__":
    main()