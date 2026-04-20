# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Diagnostic toolkit for STEM-based language models.
# Computes eight metrics for diagnosing STEM behavior:
#   1.   Per-position / per-token-type loss decomposition (Delta-ell)
#   2.   Residual stream contribution norm analysis
#   3.   CKA between STEM FFN outputs and attention outputs (per-layer)
#   A.   Cross-model hidden state divergence (STEM vs vanilla, per-layer)
#   B.   STEM embedding effective rank and spectral energy
#   I.   W3 context-conditioned variance decomposition
#   II.  Gate-feature alignment score
#   III. Gated output three-component decomposition
#
# Metrics I/II/III require vanilla_ckpt_dir and run in two additional passes:
#   Phase 4a: vanilla model only → accumulate per-token w3 means
#   Phase 4b: both models → compute variance, alignment, and decomposition
#
# Usage (single GPU, consolidated checkpoint):
#   python -m apps.main.stem_diagnostics config=apps/main/configs/stem_diagnostics.yaml

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

    stem_ckpt_dir: str = ""
    vanilla_ckpt_dir: Optional[str] = None

    data_root_dir: str = ""
    data_sources: Optional[List[str]] = None
    tokenizer_name: str = "tiktoken"
    tokenizer_path: str = ""
    seq_len: int = 2048
    max_batches: int = 50
    batch_size: int = 4

    compute_loss_decomposition: bool = True
    compute_residual_norms: bool = True
    compute_cka: bool = True
    compute_hidden_divergence: bool = True
    compute_embedding_rank: bool = True
    compute_ffn_internals: bool = True   # Metrics I, II, III: requires vanilla_ckpt_dir

    seed: int = 42
    wandb: Optional[Any] = None


# =============================================================================
# Metric 1: Per-Position Loss Decomposition
# =============================================================================
class PerPositionLossDecomposition:
    """delta_ell(t) = ell_base(t) - ell_stem(t). Positive = STEM helps."""

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
            "per_position": position_bins, "per_freq_bin": freq_bins,
            "global_delta_ell": delta[mask].mean().item() if valid > 0 else 0.0,
            "global_stem_ppl": math.exp(stem_mean[mask].mean().item()) if valid > 0 else float("inf"),
            "global_base_ppl": math.exp(base_mean[mask].mean().item()) if valid > 0 else float("inf"),
            "total_tokens": int(valid),
        }


# =============================================================================
# Helpers
# =============================================================================
def _is_postnorm_block(block: nn.Module) -> bool:
    return hasattr(block, "post_attention_norm") and hasattr(block, "post_feedforward_norm")


# =============================================================================
# Metric 2: Residual Stream Contribution Norm Analysis
# =============================================================================
class ResidualStreamAnalyzer:
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
            attn_s = {"o": None}; ffn_s = {"o": None}
            is_pn = _is_postnorm_block(block)
            def _sh(s):
                def hook(m, a, o): s["o"] = o.detach()
                return hook
            h1 = block.attention.register_forward_hook(_sh(attn_s))
            h2 = block.feed_forward.register_forward_hook(_sh(ffn_s))
            def _bh(li, a_s, f_s, blk, pn):
                def hook(m, a, o):
                    if a_s["o"] is None or f_s["o"] is None: return
                    with torch.no_grad():
                        ao, fo = a_s["o"], f_s["o"]
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
                    a_s["o"] = None; f_s["o"] = None
                return hook
            h3 = block.register_forward_hook(_bh(layer_idx, attn_s, ffn_s, block, is_pn))
            self._hooks.extend([h1, h2, h3])

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    def compute(self):
        results = {}
        for li in range(self.n_layers):
            cnt = self.count.get(li, 0)
            if cnt == 0: continue
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
        _m = lambda es, k: sum(e[k] for e in es) / len(es) if es else 0.0
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
    def __init__(self, model: StemLMTransformer):
        self.model = model
        self.stem_layer_set = set(model.stem_layers)
        self.n_layers = len(model.layers)
        self._hooks = []
        self._accum: Dict[int, Dict[str, Any]] = {}

    def install_hooks(self):
        self._hooks = []
        for layer_idx, block in enumerate(self.model.layers):
            a_s = {"o": None}; f_s = {"o": None}
            is_pn = _is_postnorm_block(block)
            def _sh(s):
                def hook(m, a, o): s["o"] = o.detach()
                return hook
            h1 = block.attention.register_forward_hook(_sh(a_s))
            h2 = block.feed_forward.register_forward_hook(_sh(f_s))
            def _bh(li, a_st, f_st, blk, pn):
                def hook(m, a, o):
                    if a_st["o"] is None or f_st["o"] is None: return
                    with torch.no_grad():
                        ao, fo = a_st["o"], f_st["o"]
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
                                "sum_Y": torch.zeros(D, dtype=torch.float64), "N": 0,
                            }
                        X64, Y64 = X.double(), Y.double()
                        acc = self._accum[li]
                        acc["XtX"].add_((X64.T @ X64).cpu())
                        acc["YtY"].add_((Y64.T @ Y64).cpu())
                        acc["YtX"].add_((Y64.T @ X64).cpu())
                        acc["sum_X"].add_(X64.sum(dim=0).cpu())
                        acc["sum_Y"].add_(Y64.sum(dim=0).cpu())
                        acc["N"] += Nb
                    a_st["o"] = None; f_st["o"] = None
                return hook
            h3 = block.register_forward_hook(_bh(layer_idx, a_s, f_s, block, is_pn))
            self._hooks.extend([h1, h2, h3])

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    def compute(self):
        results = {}
        for li in range(self.n_layers):
            if li not in self._accum: continue
            acc = self._accum[li]; N = acc["N"]
            if N == 0: continue
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
                "cka_attn_vs_ffn": num / den if den > 1e-12 else 0.0, "n_tokens": N,
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
    def __init__(self, stem_model: StemLMTransformer, vanilla_model: nn.Module):
        self.stem_model = stem_model
        self.vanilla_model = vanilla_model
        self.stem_layer_set = set(stem_model.stem_layers)
        self.n_layers = len(stem_model.layers)
        assert len(vanilla_model.layers) == self.n_layers
        self._hooks = []
        self._stem_states: Dict[int, torch.Tensor] = {}
        self._vanilla_states: Dict[int, torch.Tensor] = {}
        self.divergence_sum: Dict[int, float] = defaultdict(float)
        self.count: Dict[int, int] = defaultdict(int)

    def install_hooks(self):
        self._hooks = []
        for li, block in enumerate(self.stem_model.layers):
            def _mh(storage, layer_idx):
                def hook(module, args, output): storage[layer_idx] = output.detach()
                return hook
            self._hooks.append(block.register_forward_hook(_mh(self._stem_states, li)))
        for li, block in enumerate(self.vanilla_model.layers):
            def _mh(storage, layer_idx):
                def hook(module, args, output): storage[layer_idx] = output.detach()
                return hook
            self._hooks.append(block.register_forward_hook(_mh(self._vanilla_states, li)))

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    @torch.no_grad()
    def update(self, input_ids):
        self._stem_states.clear(); self._vanilla_states.clear()
        _ = forward_stem_enabled(self.stem_model, input_ids)
        _ = forward_vanilla(self.vanilla_model, input_ids)
        for li in range(self.n_layers):
            hs = self._stem_states.get(li); hv = self._vanilla_states.get(li)
            if hs is None or hv is None: continue
            hs = hs.reshape(-1, hs.shape[-1]); hv = hv.reshape(-1, hv.shape[-1])
            N = hs.shape[0]
            normed_div = (hs - hv).norm(dim=-1) / hv.norm(dim=-1).clamp(min=1e-8)
            self.divergence_sum[li] += normed_div.sum().item()
            self.count[li] += N
        self._stem_states.clear(); self._vanilla_states.clear()

    def compute(self):
        results = {}
        all_divs = []
        for li in range(self.n_layers):
            cnt = self.count.get(li, 0)
            if cnt == 0: continue
            d = self.divergence_sum[li] / cnt
            results[f"layer_{li}"] = {"is_stem_layer": li in self.stem_layer_set, "mean_normalized_divergence": d}
            all_divs.append(d)
        sd = [v["mean_normalized_divergence"] for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is True]
        nd = [v["mean_normalized_divergence"] for v in results.values() if isinstance(v, dict) and v.get("is_stem_layer") is False]
        if len(all_divs) >= 4:
            q = len(all_divs) // 4
            amp = (sum(all_divs[-q:]) / q) / (sum(all_divs[:q]) / q) if sum(all_divs[:q]) > 1e-12 else 0.0
        else:
            amp = 0.0
        results["summary"] = {
            "stem_layers_mean_divergence": sum(sd) / len(sd) if sd else 0.0,
            "nonstem_layers_mean_divergence": sum(nd) / len(nd) if nd else 0.0,
            "overall_mean_divergence": sum(all_divs) / len(all_divs) if all_divs else 0.0,
            "amplification_ratio": amp,
        }
        return results


# =============================================================================
# Metric B: STEM Embedding Effective Rank and Spectral Energy
# =============================================================================
class EmbeddingRankAnalyzer:
    def __init__(self, stem_model: StemLMTransformer, vanilla_model: Optional[nn.Module] = None):
        self.stem_model = stem_model
        self.vanilla_model = vanilla_model
        self.stem_layer_indices = list(stem_model.stem_layers)

    @staticmethod
    def _effective_rank(svd_vals):
        s = svd_vals / svd_vals.sum()
        s = s[s > 1e-30]
        return (-(s * s.log()).sum()).exp().item()

    @staticmethod
    def _spectral_energy(svd_vals, ks):
        total = (svd_vals ** 2).sum()
        return {f"top_{k}": ((svd_vals[:min(k, len(svd_vals))] ** 2).sum() / total).item() for k in ks}

    @torch.no_grad()
    def compute(self):
        results = {}; ks = [1, 5, 10, 50]
        for si, li in enumerate(self.stem_layer_indices):
            w = self.stem_model.stem_embeddings[si].weight.data
            V, d = w.shape
            logger.info(f"  SVD stem_embeddings[{si}] (layer {li}), shape ({V}, {d})")
            sv = torch.linalg.svdvals(w.float().cpu())
            er = self._effective_rank(sv); mr = len(sv)
            entry = {"effective_rank": er, "effective_rank_fraction": er / mr, "max_possible_rank": mr,
                     "spectral_energy": self._spectral_energy(sv, ks), "shape": [V, d],
                     "condition_number": (sv[0] / sv[-1]).item() if sv[-1] > 1e-30 else float("inf")}
            if self.vanilla_model is not None:
                w3 = self.vanilla_model.layers[li].feed_forward.w3.weight.data
                logger.info(f"  SVD vanilla w3 layer {li}, shape {list(w3.shape)}")
                sv3 = torch.linalg.svdvals(w3.float().cpu())
                er3 = self._effective_rank(sv3); mr3 = len(sv3)
                entry["w3_effective_rank"] = er3
                entry["w3_effective_rank_fraction"] = er3 / mr3
                entry["w3_spectral_energy"] = self._spectral_energy(sv3, ks)
                entry["rank_ratio_stem_over_w3"] = (er / mr) / (er3 / mr3) if er3 > 0 else 0.0
            results[f"stem_layer_{li}"] = entry
        eranks = [v["effective_rank_fraction"] for v in results.values() if isinstance(v, dict) and "effective_rank_fraction" in v]
        summary = {"mean_effective_rank_fraction": sum(eranks) / len(eranks) if eranks else 0.0}
        if self.vanilla_model is not None:
            w3e = [v["w3_effective_rank_fraction"] for v in results.values() if isinstance(v, dict) and "w3_effective_rank_fraction" in v]
            rr = [v["rank_ratio_stem_over_w3"] for v in results.values() if isinstance(v, dict) and "rank_ratio_stem_over_w3" in v]
            summary["mean_w3_effective_rank_fraction"] = sum(w3e) / len(w3e) if w3e else 0.0
            summary["mean_rank_ratio_stem_over_w3"] = sum(rr) / len(rr) if rr else 0.0
        results["summary"] = summary
        return results


# =============================================================================
# Metrics I, II, III: FFN Internal Dynamics Analyzer
# =============================================================================
class FFNInternalAnalyzer:
    """Analyzes internal FFN dynamics to distinguish Hypothesis A (context-
    dependency loss) from Hypothesis B (w1/w2 inductive bias mismatch).

    Recall the computation:
      Vanilla:  output = w2( silu(w1(x)) * w3(x) )
      STEM:     output = w2( silu(w1(x)) * stem_emb[token_id] )

    Metric I  — W3 Context-Conditioned Variance Decomposition:
      Decomposes w3 output variance into between-token (capturable by lookup)
      and within-token (lost by any lookup). The ratio rho = within/total
      measures how much context matters. High rho → Hypothesis A.

    Metric II — Gate-Feature Alignment Score:
      Correlates per-dimension gate activation magnitude with feature magnitude.
      Low alignment for STEM vs high for vanilla → Hypothesis B (gate
      expects energy in dimensions STEM doesn't provide).

    Metric III — Gated Output Three-Component Decomposition:
      Decomposes the gated FFN input into:
        (a) g(x) * y_bar_t           — shared baseline (oracle lookup)
        (b) g_van(x) * (w3(x)-y_bar) — context residual (Hypothesis A)
        (c) g_stem(x) * (emb-y_bar)  — STEM deviation from oracle (Hypothesis B)
      The ratio of (c) to (b) directly attributes the loss gap.

    Two-phase execution:
      Phase 4a: vanilla-only → accumulate per-token w3 sums/counts → compute means
      Phase 4b: both models  → compute all three metrics using stored means
    """

    def __init__(self, stem_model: StemLMTransformer, vanilla_model: nn.Module):
        self.stem_model = stem_model
        self.vanilla_model = vanilla_model
        self.stem_layer_indices = list(stem_model.stem_layers)
        self._layer_to_stem_idx = {
            li: si for si, li in enumerate(self.stem_layer_indices)
        }

        # Dimensions
        first_block = stem_model.layers[self.stem_layer_indices[0]]
        self.d_ffn = first_block.feed_forward.hidden_dim
        self.vocab_size = stem_model.lm_transformer.tok_embeddings.weight.shape[0]

        # --- Phase 4a accumulators (CPU, float64 for precision) ---
        self.w3_sum = {}
        self.w3_count = {}
        for li in self.stem_layer_indices:
            self.w3_sum[li] = torch.zeros(self.vocab_size, self.d_ffn, dtype=torch.float64)
            self.w3_count[li] = torch.zeros(self.vocab_size, dtype=torch.int64)

        # Computed after Phase 4a
        self.w3_mean: Dict[int, torch.Tensor] = {}   # layer -> (V, d_ffn) float32 CPU
        self.between_var: Dict[int, float] = {}       # layer -> scalar

        # --- Phase 4b accumulators ---
        self.within_var_sum = {li: 0.0 for li in self.stem_layer_indices}

        # Metric II: per-dimension absolute value sums (CPU, float64)
        self.van_gate_abs = {li: torch.zeros(self.d_ffn, dtype=torch.float64) for li in self.stem_layer_indices}
        self.van_w3_abs = {li: torch.zeros(self.d_ffn, dtype=torch.float64) for li in self.stem_layer_indices}
        self.stem_gate_abs = {li: torch.zeros(self.d_ffn, dtype=torch.float64) for li in self.stem_layer_indices}
        self.stem_emb_abs = {li: torch.zeros(self.d_ffn, dtype=torch.float64) for li in self.stem_layer_indices}
        self.van_gated_abs = {li: torch.zeros(self.d_ffn, dtype=torch.float64) for li in self.stem_layer_indices}
        self.stem_gated_abs = {li: torch.zeros(self.d_ffn, dtype=torch.float64) for li in self.stem_layer_indices}

        # Metric III: three-component norm sums
        self.baseline_norm_sum = {li: 0.0 for li in self.stem_layer_indices}
        self.context_res_norm_sum = {li: 0.0 for li in self.stem_layer_indices}
        self.stem_dev_norm_sum = {li: 0.0 for li in self.stem_layer_indices}

        self.phase_4b_count = {li: 0 for li in self.stem_layer_indices}

        self._hooks = []
        self._stores_4a: Dict[int, Dict] = {}
        self._stores_4b: Dict[int, Dict] = {}
        self._w3_mean_gpu: Dict[int, torch.Tensor] = {}  # GPU copies for Phase 4b

        # Per-batch coverage tracking (for vocabulary coverage analysis)
        self._seen_tokens = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.coverage_per_batch: List[Dict[str, Any]] = []  # [{batch, unique, total, frac}, ...]
        self._total_tokens_seen = 0

    # -----------------------------------------------------------------
    # Phase 4a: accumulate per-token w3 statistics from vanilla model
    # -----------------------------------------------------------------
    def install_phase_4a_hooks(self):
        self._hooks = []
        self._stores_4a = {}
        for li in self.stem_layer_indices:
            store = {"w3_out": None}

            def _make_w3_hook(storage):
                def hook(module, args, output):
                    storage["w3_out"] = output.detach()
                return hook

            h = self.vanilla_model.layers[li].feed_forward.w3.register_forward_hook(
                _make_w3_hook(store)
            )
            self._stores_4a[li] = store
            self._hooks.append(h)

    @torch.no_grad()
    def update_phase_4a(self, input_ids: torch.Tensor):
        """Run vanilla forward and accumulate per-token w3 sums."""
        _ = forward_vanilla(self.vanilla_model, input_ids)

        token_ids_cpu = input_ids.flatten().cpu()  # (B*S,)

        # Track vocabulary coverage
        batch_unique = token_ids_cpu.unique()
        self._seen_tokens[batch_unique] = True
        self._total_tokens_seen += token_ids_cpu.shape[0]
        cumulative_unique = self._seen_tokens.sum().item()
        self.coverage_per_batch.append({
            "batch": len(self.coverage_per_batch) + 1,
            "cumulative_unique_tokens": int(cumulative_unique),
            "cumulative_total_tokens": int(self._total_tokens_seen),
            "coverage_fraction": cumulative_unique / self.vocab_size,
        })

        for li in self.stem_layer_indices:
            w3_out = self._stores_4a[li]["w3_out"]  # (B, S, d_ffn)
            w3_flat = w3_out.reshape(-1, self.d_ffn).cpu().double()  # (N, d_ffn)
            N = w3_flat.shape[0]

            # scatter_add for per-token sum: w3_sum[token_id] += w3(x)
            ids_exp = token_ids_cpu.unsqueeze(1).expand(N, self.d_ffn)
            self.w3_sum[li].scatter_add_(0, ids_exp, w3_flat)
            self.w3_count[li].scatter_add_(
                0, token_ids_cpu, torch.ones(N, dtype=torch.int64)
            )

            self._stores_4a[li]["w3_out"] = None

    def finalize_phase_4a(self):
        """Compute per-token means and between-token variance from accumulated sums."""
        for li in self.stem_layer_indices:
            count = self.w3_count[li]  # (V,)
            has_data = count > 0
            N_total = count.sum().item()

            # Safe division for mean
            count_safe = count.clone()
            count_safe[~has_data] = 1
            self.w3_mean[li] = (
                self.w3_sum[li] / count_safe.unsqueeze(1).double()
            ).float()  # (V, d_ffn) float32
            self.w3_mean[li][~has_data] = 0.0

            # Between-token variance (chunked to control peak memory)
            global_sum = self.w3_sum[li].sum(dim=0)  # (d_ffn,) float64
            global_mean = (global_sum / N_total).float() if N_total > 0 else torch.zeros(self.d_ffn)

            between_var = 0.0
            active_idx = has_data.nonzero(as_tuple=True)[0]
            chunk_sz = 8192
            for start in range(0, len(active_idx), chunk_sz):
                idx = active_idx[start:start + chunk_sz]
                means_c = self.w3_mean[li][idx].double()         # (chunk, d_ffn)
                counts_c = count[idx].double()                    # (chunk,)
                dev = means_c - global_mean.double().unsqueeze(0)  # (chunk, d_ffn)
                between_var += (counts_c.unsqueeze(1) * dev ** 2).sum().item()
            self.between_var[li] = between_var / N_total if N_total > 0 else 0.0

            logger.info(
                f"  Layer {li}: {has_data.sum().item()} active tokens, "
                f"between_var={self.between_var[li]:.4f}"
            )

        # Free sums to reclaim memory
        self.w3_sum.clear()

    def compute_coverage_analysis(self, dump_dir: str) -> Dict[str, Any]:
        """Analyze vocabulary coverage from Phase 4a and generate diagnostic plots.

        Produces:
        - Token occurrence distribution statistics
        - Coverage vs batch count curve (observed + projected)
        - Per-occurrence-count histogram
        - Recommended batch count for target coverage levels
        """
        # Use the first STEM layer's count (identical across layers since tokens are shared)
        li = self.stem_layer_indices[0]
        count = self.w3_count[li]  # (V,)

        V = self.vocab_size
        active_mask = count > 0
        n_active = active_mask.sum().item()
        n_zero = V - n_active
        coverage_frac = n_active / V
        total_tokens = count.sum().item()

        # Per-token occurrence counts for active tokens
        active_counts = count[active_mask].float()

        # Occurrence distribution statistics
        stats = {
            "vocab_size": V,
            "active_tokens": int(n_active),
            "zero_count_tokens": int(n_zero),
            "coverage_fraction": coverage_frac,
            "total_token_occurrences": int(total_tokens),
            "num_batches_used": len(self.coverage_per_batch),
            "tokens_per_batch": int(total_tokens / len(self.coverage_per_batch)) if self.coverage_per_batch else 0,
        }

        if n_active > 0:
            stats["min_occurrence"] = int(active_counts.min().item())
            stats["max_occurrence"] = int(active_counts.max().item())
            stats["mean_occurrence"] = active_counts.mean().item()
            stats["median_occurrence"] = active_counts.median().item()
            stats["std_occurrence"] = active_counts.std().item()

            # Occurrence quantiles
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                val = torch.quantile(active_counts, q / 100.0).item()
                stats[f"p{q}_occurrence"] = val

            # Tokens by occurrence bucket
            buckets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, None)]
            bucket_counts = {}
            for lo, hi in buckets:
                if hi is None:
                    mask_b = active_counts >= lo
                    label = f"{lo}+"
                else:
                    mask_b = (active_counts >= lo) & (active_counts <= hi)
                    label = f"{lo}-{hi}"
                bucket_counts[label] = int(mask_b.sum().item())
            stats["occurrence_buckets"] = bucket_counts

        # Coverage per batch (already tracked during Phase 4a)
        stats["coverage_per_batch"] = self.coverage_per_batch

        # --- Projected coverage at larger batch counts ---
        # Use Heaps' law: V(n) = K * n^beta, fit from observed data
        if len(self.coverage_per_batch) >= 5:
            obs_n = torch.tensor([e["cumulative_total_tokens"] for e in self.coverage_per_batch], dtype=torch.float64)
            obs_v = torch.tensor([e["cumulative_unique_tokens"] for e in self.coverage_per_batch], dtype=torch.float64)

            # Log-log linear regression: log(V) = log(K) + beta * log(n)
            log_n = obs_n.log()
            log_v = obs_v.log()
            # Filter valid entries
            valid = (log_n.isfinite()) & (log_v.isfinite())
            if valid.sum() >= 2:
                ln = log_n[valid]
                lv = log_v[valid]
                n_pts = ln.shape[0]
                # Least squares: [log(K), beta] = (X^T X)^{-1} X^T y
                X = torch.stack([torch.ones(n_pts, dtype=torch.float64), ln], dim=1)
                params = torch.linalg.lstsq(X, lv).solution
                log_K, beta = params[0].item(), params[1].item()
                K = math.exp(log_K)

                stats["heaps_law_K"] = K
                stats["heaps_law_beta"] = beta

                # Project to various batch counts
                tpb = stats["tokens_per_batch"]
                projections = {}
                for target_batches in [100, 200, 500, 1000, 2000, 5000]:
                    proj_tokens = target_batches * tpb
                    proj_unique = min(K * (proj_tokens ** beta), V)
                    projections[str(target_batches)] = {
                        "total_tokens": proj_tokens,
                        "projected_unique": int(proj_unique),
                        "projected_coverage": proj_unique / V,
                    }
                stats["coverage_projections"] = projections

                # Estimated batches for target coverage levels
                targets = {}
                for target_cov in [0.50, 0.75, 0.90, 0.95, 0.99]:
                    target_unique = target_cov * V
                    if target_unique <= K:
                        needed_tokens = 1
                    else:
                        needed_tokens = (target_unique / K) ** (1.0 / beta) if beta > 0 else float("inf")
                    needed_batches = math.ceil(needed_tokens / tpb) if tpb > 0 else float("inf")
                    targets[f"{int(target_cov*100)}%"] = {
                        "needed_batches": int(needed_batches) if needed_batches < 1e9 else "inf",
                        "needed_tokens": int(needed_tokens) if needed_tokens < 1e15 else "inf",
                    }
                stats["batches_for_target_coverage"] = targets

        # --- Generate plots ---
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._plot_coverage(count, stats, out_dir)

        return stats

    def _plot_coverage(self, count: torch.Tensor, stats: Dict, out_dir: Path):
        """Generate four coverage diagnostic plots."""
        V = self.vocab_size
        active_mask = count > 0
        active_counts = count[active_mask].numpy()
        all_counts = count.numpy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Vocabulary Coverage Analysis  |  {stats['num_batches_used']} batches, "
            f"seq_len={stats['tokens_per_batch'] // 4 if stats['tokens_per_batch'] > 0 else '?'}, "
            f"batch_size=4",
            fontsize=14, fontweight="bold",
        )

        # --- Plot 1: Sorted token occurrence (Zipf curve) ---
        ax = axes[0, 0]
        sorted_counts = np.sort(all_counts)[::-1]
        ranks = np.arange(1, len(sorted_counts) + 1)
        ax.semilogy(ranks, sorted_counts + 1, linewidth=0.5, color="#2563EB")
        ax.axhline(y=1, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="Zero occurrences (count=0)")
        ax.axvline(x=stats["active_tokens"], color="orange", linestyle="--", linewidth=0.8,
                   label=f"Coverage boundary: {stats['active_tokens']:,} / {V:,}")
        ax.set_xlabel("Token rank (sorted by frequency)")
        ax.set_ylabel("Occurrence count + 1 (log scale)")
        ax.set_title("Token Occurrence Rank Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Occurrence histogram (active tokens only) ---
        ax = axes[0, 1]
        if len(active_counts) > 0:
            log_counts = np.log10(active_counts.clip(min=1))
            ax.hist(log_counts, bins=80, color="#2563EB", alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.axvline(x=np.log10(max(np.median(active_counts), 1)), color="orange", linestyle="--",
                       linewidth=1.2, label=f"Median: {int(np.median(active_counts))}")
            ax.axvline(x=np.log10(max(np.mean(active_counts), 1)), color="red", linestyle="--",
                       linewidth=1.2, label=f"Mean: {np.mean(active_counts):.1f}")
        ax.set_xlabel("log₁₀(occurrence count)")
        ax.set_ylabel("Number of tokens")
        ax.set_title(f"Occurrence Distribution ({stats['active_tokens']:,} active tokens)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Plot 3: Cumulative coverage vs batch ---
        ax = axes[1, 0]
        if self.coverage_per_batch:
            batches = [e["batch"] for e in self.coverage_per_batch]
            coverages = [e["coverage_fraction"] * 100 for e in self.coverage_per_batch]
            ax.plot(batches, coverages, color="#2563EB", linewidth=2, label="Observed")

            # Add Heaps' law projection
            if "heaps_law_K" in stats and "heaps_law_beta" in stats:
                K = stats["heaps_law_K"]
                beta = stats["heaps_law_beta"]
                tpb = stats["tokens_per_batch"]
                proj_batches = np.arange(1, max(batches[-1] * 5, 500) + 1)
                proj_tokens = proj_batches * tpb
                proj_unique = np.minimum(K * (proj_tokens ** beta), V)
                proj_coverage = proj_unique / V * 100
                ax.plot(proj_batches, proj_coverage, color="red", linestyle="--",
                        linewidth=1.2, alpha=0.8, label=f"Heaps' law (β={beta:.3f})")

            # Target lines
            for target, color in [(50, "#94A3B8"), (75, "#F59E0B"), (90, "#EF4444"), (95, "#7C3AED")]:
                ax.axhline(y=target, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
                ax.text(batches[-1] * 0.02, target + 0.8, f"{target}%", fontsize=7, color=color)

        ax.set_xlabel("Number of batches")
        ax.set_ylabel("Vocabulary coverage (%)")
        ax.set_title("Cumulative Vocabulary Coverage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Plot 4: Coverage reliability assessment ---
        ax = axes[1, 1]
        if len(active_counts) > 0:
            # Tokens with >= N occurrences (reliability threshold)
            thresholds = [1, 2, 3, 5, 10, 20, 50, 100]
            reliable_counts = []
            for t in thresholds:
                n_reliable = (active_counts >= t).sum()
                reliable_counts.append(n_reliable)

            bars = ax.bar(
                range(len(thresholds)),
                [r / V * 100 for r in reliable_counts],
                color="#2563EB", alpha=0.8, edgecolor="white",
            )
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels([f"≥{t}" for t in thresholds])
            ax.set_xlabel("Minimum occurrence count threshold")
            ax.set_ylabel("Vocabulary coverage (%)")
            ax.set_title("Oracle Reliability: Coverage at Occurrence Thresholds")
            ax.grid(True, alpha=0.3, axis="y")

            # Annotate bars
            for bar, cnt, total in zip(bars, reliable_counts, [V] * len(thresholds)):
                pct = cnt / total * 100
                if pct > 2:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plot_path = out_dir / "vocabulary_coverage.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Coverage plots saved to {plot_path}")

    def _ensure_w3_mean_gpu(self):
        """Transfer w3 means to GPU once for Phase 4b lookups."""
        if self._w3_mean_gpu:
            return
        device = next(self.stem_model.parameters()).device
        for li in self.stem_layer_indices:
            self._w3_mean_gpu[li] = self.w3_mean[li].to(device)

    # -----------------------------------------------------------------
    # Phase 4b: compute Metrics I, II, III from both models
    # -----------------------------------------------------------------
    def install_phase_4b_hooks(self):
        self._hooks = []
        self._stores_4b = {}
        self._ensure_w3_mean_gpu()

        for li in self.stem_layer_indices:
            store = {"van_w1": None, "van_w3": None, "stem_w1": None}

            def _make_hook(storage, key):
                def hook(module, args, output):
                    storage[key] = output.detach()
                return hook

            h1 = self.vanilla_model.layers[li].feed_forward.w1.register_forward_hook(
                _make_hook(store, "van_w1"))
            h2 = self.vanilla_model.layers[li].feed_forward.w3.register_forward_hook(
                _make_hook(store, "van_w3"))
            h3 = self.stem_model.layers[li].feed_forward.w1.register_forward_hook(
                _make_hook(store, "stem_w1"))
            self._stores_4b[li] = store
            self._hooks.extend([h1, h2, h3])

    @torch.no_grad()
    def update_phase_4b(self, input_ids: torch.Tensor):
        """Run both models and compute Metrics I/II/III contributions."""
        _ = forward_vanilla(self.vanilla_model, input_ids)
        _ = forward_stem_enabled(self.stem_model, input_ids)

        token_ids_gpu = input_ids.flatten()  # (B*S,) on GPU
        N = token_ids_gpu.shape[0]

        for li in self.stem_layer_indices:
            store = self._stores_4b[li]

            # Gate and feature activations (GPU, float32)
            van_gate = F.silu(store["van_w1"].reshape(-1, self.d_ffn).float())
            van_w3 = store["van_w3"].reshape(-1, self.d_ffn).float()
            stem_gate = F.silu(store["stem_w1"].reshape(-1, self.d_ffn).float())

            # STEM embeddings for these tokens
            stem_idx = self._layer_to_stem_idx[li]
            stem_emb = self.stem_model.stem_embeddings[stem_idx](input_ids)
            stem_emb = stem_emb.reshape(-1, self.d_ffn).float()

            # Oracle means
            oracle = self._w3_mean_gpu[li][token_ids_gpu]  # (N, d_ffn)

            # --- Metric I: within-token variance ---
            w3_residual = van_w3 - oracle
            self.within_var_sum[li] += (w3_residual ** 2).sum().item()

            # --- Metric II: per-dimension absolute value sums ---
            self.van_gate_abs[li] += van_gate.abs().sum(dim=0).cpu().double()
            self.van_w3_abs[li] += van_w3.abs().sum(dim=0).cpu().double()
            self.stem_gate_abs[li] += stem_gate.abs().sum(dim=0).cpu().double()
            self.stem_emb_abs[li] += stem_emb.abs().sum(dim=0).cpu().double()
            self.van_gated_abs[li] += (van_gate * van_w3).abs().sum(dim=0).cpu().double()
            self.stem_gated_abs[li] += (stem_gate * stem_emb).abs().sum(dim=0).cpu().double()

            # --- Metric III: three-component norms ---
            baseline = stem_gate * oracle
            context_res = van_gate * w3_residual
            stem_dev = stem_gate * (stem_emb - oracle)

            self.baseline_norm_sum[li] += baseline.norm(dim=-1).sum().item()
            self.context_res_norm_sum[li] += context_res.norm(dim=-1).sum().item()
            self.stem_dev_norm_sum[li] += stem_dev.norm(dim=-1).sum().item()

            self.phase_4b_count[li] += N

            # Clear
            store["van_w1"] = None
            store["van_w3"] = None
            store["stem_w1"] = None

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute(self) -> Dict[str, Any]:
        results = {}

        for li in self.stem_layer_indices:
            cnt = self.phase_4b_count[li]
            if cnt == 0:
                continue

            # --- Metric I ---
            within_var = self.within_var_sum[li] / cnt
            between_var = self.between_var.get(li, 0.0)
            total_var = within_var + between_var
            rho = within_var / total_var if total_var > 1e-12 else 0.0

            # --- Metric II ---
            vg = self.van_gate_abs[li] / cnt     # E[|gate_d|] vanilla
            vw = self.van_w3_abs[li] / cnt       # E[|w3_d|] vanilla
            sg = self.stem_gate_abs[li] / cnt    # E[|gate_d|] STEM
            se = self.stem_emb_abs[li] / cnt     # E[|stem_emb_d|]

            def _corr(a, b):
                a = a - a.mean(); b = b - b.mean()
                n = a.norm() * b.norm()
                return ((a * b).sum() / n).item() if n > 1e-12 else 0.0

            alignment_vanilla = _corr(vg, vw)
            alignment_stem = _corr(sg, se)

            van_gated_mean = self.van_gated_abs[li] / cnt
            stem_gated_mean = self.stem_gated_abs[li] / cnt
            ratio_d = stem_gated_mean / (van_gated_mean + 1e-12)
            ratio_mean = ratio_d.mean().item()
            ratio_std = ratio_d.std().item()
            ratio_median = ratio_d.median().item()

            # --- Metric III ---
            base_norm = self.baseline_norm_sum[li] / cnt
            ctx_res_norm = self.context_res_norm_sum[li] / cnt
            stem_dev_norm = self.stem_dev_norm_sum[li] / cnt
            total_gap = ctx_res_norm + stem_dev_norm
            hyp_a_frac = ctx_res_norm / total_gap if total_gap > 1e-12 else 0.0
            hyp_b_frac = stem_dev_norm / total_gap if total_gap > 1e-12 else 0.0

            results[f"stem_layer_{li}"] = {
                # Metric I
                "context_dependency_ratio": rho,
                "within_variance": within_var,
                "between_variance": between_var,
                "total_variance": total_var,
                # Metric II
                "gate_feature_alignment_vanilla": alignment_vanilla,
                "gate_feature_alignment_stem": alignment_stem,
                "gated_energy_ratio_mean": ratio_mean,
                "gated_energy_ratio_std": ratio_std,
                "gated_energy_ratio_median": ratio_median,
                # Metric III
                "baseline_norm": base_norm,
                "context_residual_norm": ctx_res_norm,
                "stem_deviation_norm": stem_dev_norm,
                "hypothesis_a_fraction": hyp_a_frac,
                "hypothesis_b_fraction": hyp_b_frac,
            }

        # Summary across STEM layers
        entries = [v for v in results.values() if isinstance(v, dict)]
        if entries:
            _avg = lambda k: sum(e[k] for e in entries) / len(entries)
            results["summary"] = {
                "mean_context_dependency_ratio": _avg("context_dependency_ratio"),
                "mean_gate_alignment_vanilla": _avg("gate_feature_alignment_vanilla"),
                "mean_gate_alignment_stem": _avg("gate_feature_alignment_stem"),
                "mean_gated_energy_ratio": _avg("gated_energy_ratio_mean"),
                "mean_baseline_norm": _avg("baseline_norm"),
                "mean_context_residual_norm": _avg("context_residual_norm"),
                "mean_stem_deviation_norm": _avg("stem_deviation_norm"),
                "mean_hypothesis_a_fraction": _avg("hypothesis_a_fraction"),
                "mean_hypothesis_b_fraction": _avg("hypothesis_b_fraction"),
            }
        else:
            results["summary"] = {}

        return results


# =============================================================================
# Forward Pass Variants
# =============================================================================
@torch.no_grad()
def forward_stem_disabled(model: StemLMTransformer, token_values: torch.Tensor) -> torch.Tensor:
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
        logger.warning("No data sources; random input for smoke test.")
        for _ in range(cfg.max_batches):
            ids = torch.randint(0, tokenizer.n_words, (cfg.batch_size, cfg.seq_len + 1))
            yield ids[:, :-1].cuda(), ids[:, 1:].cuda()
        return
    multi_state = init_choice_state("", srcs, 0, 0, 1, "*.val.jsonl")
    path_to_iter = setup_sources(multi_state)
    batch_count = 0; token_buf = []
    for src in path_to_iter:
        for step, (content, state) in enumerate(path_to_iter[src]):
            if state["current_iter"] > 0 or batch_count >= cfg.max_batches: break
            token_buf.extend(tokenizer.encode(
                content.get("text", content.get("content", "")), add_bos=True, add_eos=True))
            while len(token_buf) >= (cfg.seq_len + 1) * cfg.batch_size:
                chunk = token_buf[:(cfg.seq_len + 1) * cfg.batch_size]
                token_buf = token_buf[(cfg.seq_len + 1) * cfg.batch_size:]
                t = torch.tensor(chunk, dtype=torch.long).reshape(cfg.batch_size, cfg.seq_len + 1)
                yield t[:, :-1].cuda(), t[:, 1:].cuda()
                batch_count += 1
                if batch_count >= cfg.max_batches: return


# =============================================================================
# Checkpoint Resolution
# =============================================================================
def _resolve_checkpoint(ckpt_dir: str) -> Path:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_dir}")
    if (ckpt_path / "params.json").exists() and next(ckpt_path.glob("*.pth"), None) is not None:
        return ckpt_path
    consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path = Path(consolidate_checkpoints(str(ckpt_path)))
    return consolidate_path


def _ensure_sorted_stem_consolidation(ckpt_path: Path) -> None:
    """Ensure consolidated_stem.pth exists with correctly SORTED shard ordering.

    The upstream consolidate_stem_shards() uses Path.glob() which returns
    files in inode order on Linux, NOT alphabetical. This causes
    stem_model_mp0..mp7 to be concatenated in random column order,
    producing embeddings with plausible norms but scrambled features —
    manifesting as PPL 5-10x higher than expected.

    This function:
    1. Locates the consolidated dir and stem_shards/ dir
    2. If consolidated_stem.pth already exists, verifies shard count matches
    3. If it doesn't exist or needs rebuild, creates it with explicit
       sorted-by-rank-index ordering
    """
    # Find the step dir (parent of consolidated/) and consolidated dir
    if ckpt_path.name == CONSOLIDATE_FOLDER:
        step_dir = ckpt_path.parent
        consolidate_dir = ckpt_path
    elif (ckpt_path / CONSOLIDATE_FOLDER).exists():
        step_dir = ckpt_path
        consolidate_dir = ckpt_path / CONSOLIDATE_FOLDER
    else:
        step_dir = ckpt_path
        consolidate_dir = ckpt_path  # might be the consolidated dir itself

    stem_shards_dir = step_dir / "stem_shards"
    if not stem_shards_dir.exists():
        # No shards at all — either not a STEM checkpoint or shards are elsewhere
        logger.debug(f"No stem_shards/ at {stem_shards_dir}, skipping sorted consolidation")
        return

    shard_files = list(stem_shards_dir.glob("stem_model_mp*.pt"))
    if not shard_files:
        return

    output_path = consolidate_dir / CONSOLIDATE_STEM_NAME

    # Always rebuild to guarantee correct ordering
    def _extract_rank(p: Path) -> int:
        return int(p.stem.split("mp")[-1])

    shard_files.sort(key=_extract_rank)
    n_shards = len(shard_files)
    logger.info(
        f"Consolidating {n_shards} STEM shards (sorted) from {stem_shards_dir}: "
        f"{[f.name for f in shard_files]}"
    )

    consolidated = {}
    for sf in shard_files:
        rank = _extract_rank(sf)
        sd = torch.load(sf, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            consolidated.setdefault(k, []).append((rank, v))

    result = {}
    for k, rank_tensor_pairs in consolidated.items():
        rank_tensor_pairs.sort(key=lambda x: x[0])
        ranks = [r for r, _ in rank_tensor_pairs]
        expected = list(range(len(ranks)))
        if ranks != expected:
            raise ValueError(
                f"Shard rank gap for '{k}': found {ranks}, expected {expected}"
            )
        result[k] = torch.cat([t for _, t in rank_tensor_pairs], dim=1)

    consolidate_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    logger.info(f"Sorted STEM consolidation saved to {output_path}")


def _validate_stem_embeddings(model: StemLMTransformer, ckpt_path: Path) -> None:
    stem_layer_indices = list(model.stem_layers)
    if not stem_layer_indices: return
    cstem = ckpt_path / CONSOLIDATE_STEM_NAME
    shards = ckpt_path.parent / "stem_shards"
    has_c = cstem.exists() and cstem.stat().st_size > 0
    has_s = shards.exists() and any(shards.glob("stem_model_mp*.pt"))
    if not has_c and not has_s:
        raise RuntimeError(f"STEM embeddings not found at {cstem} or {shards}")
    if has_c:
        sd = torch.load(cstem, map_location="cpu", weights_only=True)
        loaded = set(sd.keys()); del sd
        expected = set()
        for mn, mod in model.named_modules():
            if isinstance(mod, (nn.Embedding, ParallelEmbedding)):
                wk = f"{mn}.weight" if mn else "weight"
                if wk.startswith("stem_embeddings."): expected.add(wk)
        missing = expected - loaded
        if missing:
            raise RuntimeError(f"STEM weights missing: {sorted(missing)}")
    logger.info(f"STEM embedding validation passed for layers {stem_layer_indices}")


# =============================================================================
# Main Diagnostic Pipeline
# =============================================================================
@torch.no_grad()
def run_diagnostics(cfg: DiagnosticsArgs):
    torch.manual_seed(cfg.seed)

    # --- Load STEM model ---
    stem_ckpt = _resolve_checkpoint(cfg.stem_ckpt_dir)
    _ensure_sorted_stem_consolidation(stem_ckpt)
    if cfg.model_type not in STEM_MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{cfg.model_type}'")
    stem_model_cls, stem_args_cls = STEM_MODEL_REGISTRY[cfg.model_type][:2]
    logger.info(f"Loading STEM model from {stem_ckpt}")
    _tok_kw = {}
    if cfg.tokenizer_path:
        _tok_kw["tokenizer_path"] = cfg.tokenizer_path
        if cfg.tokenizer_name: _tok_kw["tokenizer_name"] = cfg.tokenizer_name
    stem_model, tokenizer, train_cfg = load_stem_model(
        str(stem_ckpt), model_cls=stem_model_cls, model_args_cls=stem_args_cls, **_tok_kw)
    stem_model.eval()
    _validate_stem_embeddings(stem_model, stem_ckpt)
    logger.info(f"STEM model loaded: {len(stem_model.layers)} layers, stem_layers={list(stem_model.stem_layers)}")

    # --- Metric B ---
    rank_results = None
    if cfg.compute_embedding_rank:
        logger.info("Metric B: Spectral analysis...")
        rank_results = EmbeddingRankAnalyzer(stem_model).compute()

    # --- Load vanilla model ---
    vanilla_model = None
    needs_vanilla = (
        (cfg.compute_loss_decomposition and cfg.vanilla_ckpt_dir)
        or cfg.compute_hidden_divergence
        or (cfg.compute_embedding_rank and cfg.vanilla_ckpt_dir)
        or cfg.compute_ffn_internals
    )
    if needs_vanilla:
        if not cfg.vanilla_ckpt_dir:
            if cfg.compute_hidden_divergence:
                logger.warning("vanilla_ckpt_dir not set. Skipping Metric A.")
                cfg.compute_hidden_divergence = False
            if cfg.compute_ffn_internals:
                logger.warning("vanilla_ckpt_dir not set. Skipping Metrics I/II/III.")
                cfg.compute_ffn_internals = False
        else:
            van_ckpt = _resolve_checkpoint(cfg.vanilla_ckpt_dir)
            van_model_cls, van_args_cls = MODEL_REGISTRY.get(cfg.model_type, MODEL_REGISTRY["llama"])
            logger.info(f"Loading vanilla model from {van_ckpt}")
            _tok_kw2 = {}
            if cfg.tokenizer_path:
                _tok_kw2["tokenizer_path"] = cfg.tokenizer_path
                if cfg.tokenizer_name: _tok_kw2["tokenizer_name"] = cfg.tokenizer_name
            vanilla_model, _, _ = load_vanilla_model(
                str(van_ckpt), model_cls=van_model_cls, model_args_cls=van_args_cls, **_tok_kw2)
            vanilla_model.eval()
            logger.info("Vanilla model loaded")

    # --- Metric B update with w3 comparison ---
    if cfg.compute_embedding_rank and vanilla_model is not None:
        logger.info("Metric B: Recomputing with w3 comparison...")
        rank_results = EmbeddingRankAnalyzer(stem_model, vanilla_model).compute()
        s = rank_results["summary"]
        logger.info(f"Metric B — erank: {s['mean_effective_rank_fraction']:.4f}, "
                    f"ratio: {s.get('mean_rank_ratio_stem_over_w3', 'N/A')}")

    # --- Tokenizer fallback ---
    if not cfg.tokenizer_path and hasattr(train_cfg, "data"):
        cfg.tokenizer_path = train_cfg.data.tokenizer.path
        cfg.tokenizer_name = train_cfg.data.tokenizer.name
        tokenizer = build_tokenizer(cfg.tokenizer_name, cfg.tokenizer_path)

    # --- Initialize metrics ---
    loss_decomp = None; residual_analyzer = None; cka_analyzer = None
    divergence_analyzer = None; ffn_analyzer = None; coverage_results = None

    if cfg.compute_loss_decomposition:
        vs = stem_model.lm_transformer.tok_embeddings.weight.shape[0]
        loss_decomp = PerPositionLossDecomposition(vocab_size=vs, seq_len=cfg.seq_len)
    if cfg.compute_residual_norms:
        residual_analyzer = ResidualStreamAnalyzer(stem_model)
    if cfg.compute_cka:
        cka_analyzer = CKAAnalyzer(stem_model)
    if cfg.compute_hidden_divergence and vanilla_model is not None:
        divergence_analyzer = HiddenStateDivergenceAnalyzer(stem_model, vanilla_model)
    if cfg.compute_ffn_internals and vanilla_model is not None:
        ffn_analyzer = FFNInternalAnalyzer(stem_model, vanilla_model)

    # === Phase 1: STEM-only hooks (Metrics 2 & 3) ===
    if residual_analyzer or cka_analyzer:
        if residual_analyzer: residual_analyzer.install_hooks()
        if cka_analyzer: cka_analyzer.install_hooks()
        logger.info("Phase 1: STEM forward with hooks (Metrics 2 & 3)")
        for bi, (iids, _) in enumerate(build_eval_batches(cfg, tokenizer)):
            _ = forward_stem_enabled(stem_model, iids)
            if (bi + 1) % 10 == 0: logger.info(f"  Batch {bi+1}/{cfg.max_batches}")
        if residual_analyzer: residual_analyzer.remove_hooks()
        if cka_analyzer: cka_analyzer.remove_hooks()
        torch.cuda.empty_cache()

    # === Phase 2: Loss decomposition (Metric 1) ===
    if loss_decomp:
        logger.info("Phase 2: Loss decomposition (Metric 1)")
        for bi, (iids, tgts) in enumerate(build_eval_batches(cfg, tokenizer)):
            sl = forward_stem_enabled(stem_model, iids)
            bl = forward_vanilla(vanilla_model, iids) if vanilla_model else forward_stem_disabled(stem_model, iids)
            loss_decomp.update(sl, bl, tgts, iids)
            if (bi + 1) % 10 == 0: logger.info(f"  Batch {bi+1}/{cfg.max_batches}")
        torch.cuda.empty_cache()

    # === Phase 3: Hidden state divergence (Metric A) ===
    if divergence_analyzer:
        divergence_analyzer.install_hooks()
        logger.info("Phase 3: Hidden state divergence (Metric A)")
        for bi, (iids, _) in enumerate(build_eval_batches(cfg, tokenizer)):
            divergence_analyzer.update(iids)
            if (bi + 1) % 10 == 0: logger.info(f"  Batch {bi+1}/{cfg.max_batches}")
        divergence_analyzer.remove_hooks()
        torch.cuda.empty_cache()

    # === Phase 4a: Vanilla w3 accumulation (Metrics I/II/III setup) ===
    if ffn_analyzer:
        ffn_analyzer.install_phase_4a_hooks()
        logger.info("Phase 4a: Vanilla w3 per-token accumulation")
        for bi, (iids, _) in enumerate(build_eval_batches(cfg, tokenizer)):
            ffn_analyzer.update_phase_4a(iids)
            if (bi + 1) % 10 == 0: logger.info(f"  Batch {bi+1}/{cfg.max_batches}")
        ffn_analyzer.remove_hooks()
        ffn_analyzer.finalize_phase_4a()
        coverage_results = ffn_analyzer.compute_coverage_analysis(cfg.dump_dir)
        torch.cuda.empty_cache()

    # === Phase 4b: Joint FFN analysis (Metrics I/II/III) ===
    if ffn_analyzer:
        ffn_analyzer.install_phase_4b_hooks()
        logger.info("Phase 4b: FFN internal analysis (Metrics I/II/III)")
        for bi, (iids, _) in enumerate(build_eval_batches(cfg, tokenizer)):
            ffn_analyzer.update_phase_4b(iids)
            if (bi + 1) % 10 == 0: logger.info(f"  Batch {bi+1}/{cfg.max_batches}")
        ffn_analyzer.remove_hooks()
        torch.cuda.empty_cache()

    # === Compute and save ===
    results = {"config": {
        "stem_ckpt_dir": cfg.stem_ckpt_dir, "vanilla_ckpt_dir": cfg.vanilla_ckpt_dir,
        "model_type": cfg.model_type, "seq_len": cfg.seq_len,
        "max_batches": cfg.max_batches, "batch_size": cfg.batch_size,
        "ablation_mode": "vanilla_checkpoint" if vanilla_model else "stem_zeroed",
    }}

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
        logger.info(f"Metric A — STEM div: {s['stem_layers_mean_divergence']:.4f}, amplification: {s['amplification_ratio']:.2f}x")
    if rank_results:
        results["embedding_rank"] = rank_results
        logger.info(f"Metric B — erank: {rank_results['summary']['mean_effective_rank_fraction']:.4f}")
    if ffn_analyzer:
        results["ffn_internals"] = ffn_analyzer.compute()
        s = results["ffn_internals"].get("summary", {})
        logger.info(
            f"Metric I — context_dependency_ratio: {s.get('mean_context_dependency_ratio', 0):.4f}")
        logger.info(
            f"Metric II — gate_align vanilla: {s.get('mean_gate_alignment_vanilla', 0):.4f}, "
            f"stem: {s.get('mean_gate_alignment_stem', 0):.4f}, "
            f"energy_ratio: {s.get('mean_gated_energy_ratio', 0):.4f}")
        logger.info(
            f"Metric III — hyp_A: {s.get('mean_hypothesis_a_fraction', 0):.4f}, "
            f"hyp_B: {s.get('mean_hypothesis_b_fraction', 0):.4f}")
    if coverage_results:
        results["vocabulary_coverage"] = coverage_results
        logger.info(
            f"Coverage — {coverage_results['active_tokens']:,}/{coverage_results['vocab_size']:,} "
            f"({coverage_results['coverage_fraction']:.1%}) tokens seen in "
            f"{coverage_results['num_batches_used']} batches")
        if "batches_for_target_coverage" in coverage_results:
            for tgt, info in coverage_results["batches_for_target_coverage"].items():
                logger.info(f"  {tgt} coverage needs ~{info['needed_batches']} batches")

    # --- Save ---
    out_dir = Path(cfg.dump_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostics.json"
    def _ser(obj):
        if isinstance(obj, torch.Tensor): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_ser(v) for v in obj]
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)): return str(obj)
        return obj
    with open(out_path, "w") as f:
        json.dump(_ser(results), f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # --- wandb ---
    if cfg.wandb is not None:
        try:
            import wandb as wb
            wb.init(**(cfg.wandb if isinstance(cfg.wandb, dict) else {}))
            flat = {}
            if loss_decomp:
                flat["diag/global_delta_ell"] = ld["global_delta_ell"]
                flat["diag/stem_ppl"] = ld["global_stem_ppl"]
            if residual_analyzer:
                for k, v in results["residual_norms"]["summary"].items(): flat[f"diag/residual/{k}"] = v
            if cka_analyzer:
                for k, v in results["cka"]["summary"].items(): flat[f"diag/cka/{k}"] = v
            if divergence_analyzer:
                for k, v in results["hidden_divergence"]["summary"].items(): flat[f"diag/divergence/{k}"] = v
            if rank_results:
                for k, v in rank_results["summary"].items(): flat[f"diag/rank/{k}"] = v
            if ffn_analyzer and "summary" in results.get("ffn_internals", {}):
                for k, v in results["ffn_internals"]["summary"].items(): flat[f"diag/ffn/{k}"] = v
            wb.log(flat); wb.finish()
        except Exception as e:
            logger.warning(f"wandb failed: {e}")
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