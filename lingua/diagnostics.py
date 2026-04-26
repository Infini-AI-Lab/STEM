"""Optional diagnostics for STEM, STEM+projection, and DAG-STEM models.

The helpers in this module are intentionally passive: nothing is registered or
written unless ``DiagnosticsArgs.enabled`` is true.  Collection is hook based and
keeps only streaming summaries or bounded samples in memory.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import types
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor

from lingua.diagnostic_records import (
    InterventionRecord,
    TokenEffectRecord,
    append_jsonl,
    classify_task_group,
    classify_token_role,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

TASK_ALIGNED_INTERVENTIONS_JSONL = "interventions_task_aligned.jsonl"
TOKEN_EFFECTS_JSONL = "token_effects.jsonl"
TOKEN_EFFECTS_BY_TASK_JSON = "token_effects_by_task.json"
TOKEN_EFFECTS_BY_ROLE_JSON = "token_effects_by_role.json"
PATH_RELATIONS_BY_TASK_LAYER_JSON = "path_relations_by_task_layer.json"


@dataclass
class DiagnosticsArgs:
    enabled: bool = False
    log_every_n_steps: int = 100
    sample_every_n_steps: int = 1000
    max_batches_per_collection: int = 1
    max_token_positions: int = 2048
    max_tokens_per_layer_geometry: int = 4096
    collect_train_stats: bool = False
    collect_eval_stats: bool = False
    collect_token_stats: bool = False
    collect_geometry: bool = False
    collect_optimizer_stats: bool = False
    collect_interventions: bool = False
    collect_code_error_taxonomy: bool = False
    save_raw_samples: bool = False
    output_dir: Optional[str] = None
    layers: Optional[List[int]] = None
    track_frequency_buckets: bool = False
    frequency_bucket_boundaries: List[int] = field(
        default_factory=lambda: [10, 100, 1000]
    )
    path_ablation_eval_every_n_steps: int = 0
    path_ablation_num_batches: int = 1
    token_topk: int = 50
    geometry_var_frac: float = 0.9
    enable_backward_hooks: bool = False
    enable_optimizer_moment_logging: bool = False
    code_tasks: Optional[List[str]] = field(default_factory=lambda: ["mbpp", "humaneval"])
    baseline_dense_checkpoint_path: Optional[str] = None
    gate_alpha_values: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    # ------------------------------------------------------------------
    # Train-time task/source labelling.
    # When ``train_task_label`` is set it is passed verbatim to
    # ``StemDiagnosticsCollector.start_batch(task=...)``.
    # When ``infer_task_from_data_path`` is True *and* a single data source
    # is configured, the base directory name of that source is used as the
    # label (e.g. ``"math_data"`` from ``"/datasets/math_data/"``).
    # If neither applies the label falls back to ``"train"``.
    # ------------------------------------------------------------------
    train_task_label: Optional[str] = None
    infer_task_from_data_path: bool = False

    # ------------------------------------------------------------------
    # Eval-time sample capture (Round 2: task-aligned lm-eval sample logging).
    # All defaults are conservative; nothing is captured unless ``enabled``
    # AND ``collect_eval_samples`` are both True.
    # ------------------------------------------------------------------
    collect_eval_samples: bool = False
    max_eval_samples_per_task: Optional[int] = None
    capture_prompts: bool = True
    capture_generations: bool = True
    capture_token_ids: bool = False
    max_text_chars: int = 4096
    tasks: Optional[List[str]] = None  # subset of task names; None = all
    rank0_only: bool = True
    run_id: Optional[str] = None  # propagated into DiagnosticSampleRecord.run_id

    # ------------------------------------------------------------------
    # Eval-time activation diagnostics (Round 3: per-task, per-layer
    # forward-only metrics on actual lm-eval samples).  All defaults are
    # off so existing eval flows are unchanged when these are absent.
    # ------------------------------------------------------------------
    collect_eval_activations: bool = False
    collect_eval_geometry: bool = False
    eval_activation_max_samples_per_task: Optional[int] = 64
    eval_activation_max_tokens_per_sample: int = 256
    eval_layers: Optional[List[int]] = None
    write_layer_path_records: bool = False
    eval_activation_max_layer_path_records_per_sample: int = 64
    eval_geometry_max_tokens_per_layer: int = 4096
    eval_geometry_save_npz: bool = False

    # ------------------------------------------------------------------
    # Task-aligned eval interventions (Round 4: causal loss deltas on the
    # actual lm-eval samples).  The master switch is separate from the older
    # validation-prompt ``collect_interventions`` path so existing eval runs
    # are unchanged unless this is explicitly enabled.
    # ------------------------------------------------------------------
    collect_eval_interventions: bool = False
    intervention_max_samples_per_task: Optional[int] = 8
    intervention_layers: Optional[List[int]] = None
    intervention_types: List[str] = field(
        default_factory=lambda: [
            "ablate_stem",
            "ablate_up",
            "ablate_combined",
            "ablate_layer_stem",
            "ablate_layer_up",
            "force_gate_0",
            "force_gate_0_25",
            "force_gate_0_5",
            "force_gate_0_75",
            "force_gate_1",
            "replace_stem_mean",
            "replace_up_mean",
        ]
    )
    compute_per_token_delta: bool = True
    update_token_effectiveness: bool = True


def _as_local_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, DTensor):
        x = x.to_local()
    if not isinstance(x, torch.Tensor):
        return None
    return x


def _scalar(x: Any) -> Optional[float]:
    x = _as_local_tensor(x)
    if x is None or x.numel() == 0:
        return None
    return float(x.detach().float().mean().item())


def _safe_norm(x: Any) -> Optional[torch.Tensor]:
    x = _as_local_tensor(x)
    if x is None or x.numel() == 0:
        return None
    return x.detach().float().flatten(start_dim=-1).norm(dim=-1)


def _finite(v: float) -> float:
    return v if math.isfinite(v) else 0.0


class OnlineStats:
    """Streaming scalar stats with a merge-friendly representation."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: Any) -> None:
        tensor = _as_local_tensor(value)
        if tensor is None:
            vals = torch.tensor([float(value)], dtype=torch.float32)
        else:
            vals = tensor.detach().float().reshape(-1).cpu()
        vals = vals[torch.isfinite(vals)]
        for val in vals.tolist():
            self.count += 1
            delta = val - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (val - self.mean)
            self.min = min(self.min, val)
            self.max = max(self.max, val)

    def merge(self, other: "OnlineStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.min = other.min
            self.max = other.max
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean = (self.mean * self.count + other.mean * other.count) / total
        self.count = total
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)

    def as_dict(self, prefix: str) -> Dict[str, float]:
        if self.count == 0:
            return {}
        var = self.m2 / max(1, self.count - 1)
        return {
            f"{prefix}_mean": _finite(self.mean),
            f"{prefix}_std": _finite(math.sqrt(max(var, 0.0))),
            f"{prefix}_min": _finite(self.min),
            f"{prefix}_max": _finite(self.max),
            f"{prefix}_count": float(self.count),
        }


class TensorReservoir:
    """Bounded deterministic-ish tensor sample used for geometry."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, max_items)
        self.chunks: List[torch.Tensor] = []
        self.count = 0

    def add(self, x: Any) -> None:
        if self.max_items <= 0:
            return
        x = _as_local_tensor(x)
        if x is None:
            return
        flat = x.detach().float().reshape(-1, x.shape[-1]).cpu()
        if flat.numel() == 0:
            return
        remaining = self.max_items - self.count
        if remaining <= 0:
            return
        if flat.shape[0] > remaining:
            idx = torch.linspace(0, flat.shape[0] - 1, remaining).long()
            flat = flat.index_select(0, idx)
        self.chunks.append(flat)
        self.count += flat.shape[0]

    def tensor(self) -> Optional[torch.Tensor]:
        if not self.chunks:
            return None
        return torch.cat(self.chunks, dim=0)


def pairwise_cosine_stats(x: torch.Tensor, max_pairs: int = 2048) -> Dict[str, float]:
    x = x.detach().float()
    if x.ndim != 2 or x.shape[0] < 2:
        return {}
    if x.shape[0] > max_pairs:
        idx = torch.linspace(0, x.shape[0] - 1, max_pairs).long()
        x = x.index_select(0, idx)
    xn = F.normalize(x, dim=-1, eps=1e-8)
    sim = xn @ xn.T
    tri = sim[torch.triu_indices(sim.shape[0], sim.shape[1], offset=1).unbind()]
    return {
        "pairwise_cos_mean": _finite(float(tri.mean().item())),
        "pairwise_cos_std": _finite(float(tri.std(unbiased=False).item())),
        "pairwise_cos_min": _finite(float(tri.min().item())),
        "pairwise_cos_max": _finite(float(tri.max().item())),
    }


def geometry_summary(
    x: torch.Tensor,
    *,
    var_frac: float = 0.9,
    compare_to: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    x = x.detach().float()
    if x.ndim != 2 or x.shape[0] == 0:
        return {}
    x = x[torch.isfinite(x).all(dim=-1)]
    if x.shape[0] == 0:
        return {}
    centroid = x.mean(dim=0, keepdim=True)
    centered = x - centroid
    norm_mean = float(x.norm(dim=-1).mean().item())
    centroid_cos = F.cosine_similarity(x, centroid.expand_as(x), dim=-1, eps=1e-8)
    out = {
        "norm_mean": _finite(norm_mean),
        "centroid_cos_mean": _finite(float(centroid_cos.mean().item())),
        "centroid_cos_std": _finite(float(centroid_cos.std(unbiased=False).item())),
        "anisotropy": _finite(float((centroid.norm() / (x.norm(dim=-1).mean() + 1e-8)).item())),
    }
    out.update(pairwise_cosine_stats(x))
    try:
        s = torch.linalg.svdvals(centered)
        eig = s.square()
        total = eig.sum().clamp_min(1e-12)
        probs = eig / total
        entropy = -(probs * (probs + 1e-12).log()).sum()
        out["effective_rank"] = _finite(float(entropy.exp().item()))
        out["top_pc_var_frac"] = _finite(float(probs[0].item())) if probs.numel() else 0.0
        csum = probs.cumsum(dim=0)
        out["explained_var_rank"] = float(int((csum < var_frac).sum().item()) + 1)
        if s.numel() > 0:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            pc = vh[0:1]
            out["top_pc_alignment"] = _finite(
                float(F.cosine_similarity(x, pc.expand_as(x), dim=-1, eps=1e-8).abs().mean().item())
            )
            if compare_to is not None:
                comp = compare_to.detach().float().reshape(-1, compare_to.shape[-1])
                comp = comp[: x.shape[0]]
                comp_mean = comp.mean(dim=0, keepdim=True)
                out["alignment_with_dense_mean"] = _finite(
                    float(F.cosine_similarity(centroid, comp_mean, dim=-1, eps=1e-8).item())
                )
    except RuntimeError:
        pass
    return out


def _iter_stem_layers(model: torch.nn.Module, layers: Optional[List[int]] = None) -> Iterator[Tuple[int, torch.nn.Module]]:
    lm = getattr(model, "lm_transformer", model)
    layer_modules = getattr(lm, "layers", [])
    allowed = set(layers) if layers is not None else None
    stem_layers = set(getattr(lm, "stem_layers", []))
    for idx, layer in enumerate(layer_modules):
        if allowed is not None and idx not in allowed:
            continue
        ff = getattr(layer, "feed_forward", None)
        if (
            idx in stem_layers
            or hasattr(ff, "alpha")
            or (allowed is not None and ff is not None)
        ):
            yield idx, layer


def _iter_all_ffn_layers(
    model: torch.nn.Module,
    layers: Optional[List[int]] = None,
) -> Iterator[Tuple[int, torch.nn.Module]]:
    """Iterate every block with a ``feed_forward`` attribute.

    Unlike :func:`_iter_stem_layers` this does not gate on STEM-specific
    attributes, so plain transformer blocks can also be tracked when the
    caller explicitly opts in via ``layers``.  Used by eval-mode activation
    diagnostics so plain dense FFNs can be measured alongside STEM blocks.
    """
    lm = getattr(model, "lm_transformer", model)
    layer_modules = getattr(lm, "layers", [])
    allowed = set(layers) if layers is not None else None
    for idx, layer in enumerate(layer_modules):
        if allowed is not None and idx not in allowed:
            continue
        ff = getattr(layer, "feed_forward", None)
        if ff is not None:
            yield idx, layer


def _stem_embedding_for_layer(model: torch.nn.Module, layer_idx: int) -> Optional[torch.nn.Module]:
    if not hasattr(model, "stem_embeddings"):
        return None
    stem_idx = getattr(model, "_layer_to_stem_idx", {}).get(layer_idx)
    if stem_idx is None:
        return None
    return model.stem_embeddings[stem_idx]


def _token_role(token_text: str) -> str:
    if token_text in {"\n", "\\n"} or token_text.strip() == "":
        return "whitespace"
    if token_text in {"(", ")", "[", "]", "{", "}", ":", ",", "."}:
        return "bracket" if token_text in {"(", ")", "[", "]", "{", "}"} else "punctuation"
    if token_text in {"+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">="}:
        return "operator"
    if token_text.isdigit() or re.fullmatch(r"\d+(\.\d+)?", token_text):
        return "numeral"
    if token_text in {
        "False", "None", "True", "and", "as", "assert", "async", "await", "break",
        "class", "continue", "def", "del", "elif", "else", "except", "finally",
        "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
        "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    }:
        return "python_keyword"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_text.strip()):
        return "identifier"
    if any(ch.isalpha() for ch in token_text):
        return "natural_language"
    return "unknown"


class TokenStatsAggregator:
    def __init__(self, boundaries: Iterable[int] = (10, 100, 1000), topk: int = 50) -> None:
        self.boundaries = list(boundaries)
        self.topk = topk
        self.freq: Counter[int] = Counter()
        self.by_layer: Dict[int, Dict[int, Dict[str, OnlineStats]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(OnlineStats))
        )
        self.loss_deltas: Dict[int, Dict[str, OnlineStats]] = defaultdict(lambda: defaultdict(OnlineStats))
        self.roles: Dict[str, Dict[str, OnlineStats]] = defaultdict(lambda: defaultdict(OnlineStats))
        self.loss_delta_effects: Dict[Tuple[str, str, int, int, str, str], Dict[str, OnlineStats]] = defaultdict(
            lambda: defaultdict(OnlineStats)
        )
        self.layer_loss_deltas: Dict[Tuple[str, int, str], OnlineStats] = defaultdict(OnlineStats)
        self.role_loss_deltas: Dict[Tuple[str, str, str], OnlineStats] = defaultdict(OnlineStats)
        self.token_strings: Dict[int, str] = {}
        self.token_roles: Dict[int, str] = {}

    def bucket(self, token_id: int) -> str:
        f = self.freq[token_id]
        if len(self.boundaries) >= 3:
            if f < self.boundaries[0]:
                return "rare"
            if f < self.boundaries[1]:
                return "mid"
            if f < self.boundaries[2]:
                return "frequent"
            return "very_frequent"
        return f"ge_{max([b for b in self.boundaries if f >= b], default=0)}"

    def update(
        self,
        layer_idx: int,
        token_ids: torch.Tensor,
        *,
        stem_norm: Optional[torch.Tensor] = None,
        up_norm: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
    ) -> None:
        ids = token_ids.detach().reshape(-1).cpu().tolist()
        self.freq.update(int(i) for i in ids)
        vals = {
            "stem_activation_norm": stem_norm,
            "dense_up_norm": up_norm,
            "gradient_norm": grad_norm,
        }
        for name, value in vals.items():
            if value is None:
                continue
            flat = value.detach().float().reshape(-1).cpu()
            for tok, val in zip(ids, flat.tolist()):
                self.by_layer[layer_idx][int(tok)][name].update(val)

    @staticmethod
    def _normalized_delta_name(name: str) -> str:
        aliases = {
            "stem_ablation_loss_delta": "stem_ablation_delta_loss",
            "up_ablation_loss_delta": "up_ablation_delta_loss",
            "dense_ablation_loss_delta": "up_ablation_delta_loss",
            "combined_ablation_loss_delta": "combined_ablation_delta_loss",
        }
        return aliases.get(name, name)

    def update_loss_delta(
        self,
        token_ids: torch.Tensor,
        name: str,
        deltas: torch.Tensor,
        *,
        task: Optional[str] = None,
        task_group: Optional[str] = None,
        layer_idx: Optional[int] = None,
        tokens: Optional[Iterable[str]] = None,
        token_roles: Optional[Iterable[str]] = None,
    ) -> None:
        """Merge per-token intervention loss deltas.

        Sign convention: ``delta = intervened_nll - original_nll``.  A
        positive STEM ablation delta means STEM made that token easier to
        predict; a negative delta means STEM was harmful for that token.
        """
        ids = token_ids.detach().reshape(-1).cpu().tolist()
        flat = deltas.detach().float().reshape(-1).cpu().tolist()
        token_list = list(tokens) if tokens is not None else []
        role_list = list(token_roles) if token_roles is not None else []
        norm_name = self._normalized_delta_name(name)
        layer_key = -1 if layer_idx is None else int(layer_idx)
        task_key = task or "global"
        group_key = task_group or "unknown"
        # Training/activation collection calls ``update`` first, which already
        # counts token frequency.  Task-aligned interventions may only call
        # ``update_loss_delta``, so count exposure here only for those
        # intervention-only aggregators.
        if not self.by_layer:
            self.freq.update(int(i) for i in ids)
        for pos, (tok, val) in enumerate(zip(ids, flat)):
            tok_i = int(tok)
            token_text = token_list[pos] if pos < len(token_list) else str(tok_i)
            role = role_list[pos] if pos < len(role_list) else "unknown"
            self.token_strings.setdefault(tok_i, token_text)
            self.token_roles.setdefault(tok_i, role)
            self.loss_deltas[tok_i][name].update(val)
            if norm_name != name:
                self.loss_deltas[tok_i][norm_name].update(val)
            if task is not None:
                key = (task_key, group_key, layer_key, tok_i, token_text, role)
                self.loss_delta_effects[key][norm_name].update(val)
                self.loss_delta_effects[key]["abs_delta_loss"].update(abs(float(val)))
                self.layer_loss_deltas[(task_key, layer_key, norm_name)].update(val)
                self.role_loss_deltas[(task_key, role, norm_name)].update(val)
                self.roles[role][norm_name].update(val)

    def update_role_delta(self, role: str, name: str, value: float) -> None:
        self.roles[role][name].update(value)

    def rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for layer_idx, layer_data in self.by_layer.items():
            for token_id, stats in layer_data.items():
                row: Dict[str, Any] = {
                    "layer": layer_idx,
                    "token_id": token_id,
                    "frequency": self.freq[token_id],
                    "frequency_bucket": self.bucket(token_id),
                }
                for name, stat in stats.items():
                    row.update(stat.as_dict(name))
                for name, stat in self.loss_deltas.get(token_id, {}).items():
                    row.update(stat.as_dict(name))
                rows.append(row)
        return rows

    @staticmethod
    def _score_from_stem_delta(delta: Optional[float], count: int) -> Tuple[float, float, float]:
        """Return ``(benefit, harm, ineffective)`` for a STEM-token row.

        Formula is deliberately transparent:
        ``benefit_score = max(mean_stem_ablation_delta, 0)`` and
        ``harm_score = max(-mean_stem_ablation_delta, 0)``.  For ineffective
        exposure, ``exposure = log1p(count)`` and
        ``ineffective_score = exposure / (1 + abs(mean_delta))`` only when the
        mean STEM benefit is zero or negative; otherwise it is ``0``.  This
        makes high-count tokens with no positive causal benefit easy to spot.
        """
        if delta is None:
            return 0.0, 0.0, 0.0
        benefit = max(float(delta), 0.0)
        harm = max(-float(delta), 0.0)
        exposure = math.log1p(max(int(count), 0))
        ineffective = exposure / (1.0 + abs(float(delta))) if delta <= 0.0 else 0.0
        return benefit, harm, ineffective

    def token_effect_records(self, *, run_id: str) -> List[TokenEffectRecord]:
        records: List[TokenEffectRecord] = []
        for (task, task_group, layer_key, token_id, token, role), stats in sorted(
            self.loss_delta_effects.items(),
            key=lambda item: (item[0][0], item[0][2], item[0][3], item[0][4], item[0][5]),
        ):
            count = max((stat.count for stat in stats.values()), default=0)
            stem_stat = stats.get("stem_ablation_delta_loss")
            up_stat = stats.get("up_ablation_delta_loss")
            combined_stat = stats.get("combined_ablation_delta_loss")
            stem_delta = stem_stat.mean if stem_stat and stem_stat.count else None
            up_delta = up_stat.mean if up_stat and up_stat.count else None
            combined_delta = combined_stat.mean if combined_stat and combined_stat.count else None
            benefit, harm, ineffective = self._score_from_stem_delta(stem_delta, count)
            extra_metrics = {
                name: stat.as_dict(name)
                for name, stat in stats.items()
                if name
                not in {
                    "stem_ablation_delta_loss",
                    "up_ablation_delta_loss",
                    "combined_ablation_delta_loss",
                }
            }
            records.append(
                TokenEffectRecord(
                    run_id=run_id,
                    task=task,
                    task_group=task_group,
                    token_id=int(token_id),
                    token=token,
                    token_role=role,
                    layer_idx=None if layer_key < 0 else int(layer_key),
                    frequency_bucket=self.bucket(int(token_id)),
                    count=int(count),
                    stem_ablation_delta_loss=stem_delta,
                    up_ablation_delta_loss=up_delta,
                    combined_ablation_delta_loss=combined_delta,
                    benefit_score=benefit,
                    harm_score=harm,
                    ineffective_score=ineffective,
                    metadata={
                        "score_formula": (
                            "delta = intervened_nll - original_nll; "
                            "benefit=max(stem_delta,0); harm=max(-stem_delta,0); "
                            "ineffective=log1p(count)/(1+abs(stem_delta)) when stem_delta<=0 else 0"
                        ),
                        "extra_delta_metrics": extra_metrics,
                    },
                )
            )
        return records

    def rankings(self) -> Dict[str, List[Dict[str, Any]]]:
        token_scores = []
        for token_id, stats in self.loss_deltas.items():
            stem = stats.get("stem_ablation_delta_loss") or stats.get("stem_ablation_loss_delta")
            benefit = stem.mean if stem and stem.count else 0.0
            token_scores.append(
                {
                    "token_id": token_id,
                    "token": self.token_strings.get(token_id, str(token_id)),
                    "token_role": self.token_roles.get(token_id, "unknown"),
                    "frequency": self.freq[token_id],
                    "score": benefit,
                    "benefit_score": max(benefit, 0.0),
                    "harm_score": max(-benefit, 0.0),
                }
            )
        return {
            "top_beneficial_tokens": sorted(token_scores, key=lambda r: r["score"], reverse=True)[: self.topk],
            "top_harmful_tokens": sorted(token_scores, key=lambda r: r["score"])[: self.topk],
            "high_frequency_negative_impact": sorted(
                [r for r in token_scores if r["score"] < 0],
                key=lambda r: (r["frequency"], -r["score"]),
                reverse=True,
            )[: self.topk],
        }


class StemDiagnosticsCollector:
    """Hook-based diagnostics for STEM / DAG / plain FFNs.

    Two modes are supported:

    * ``mode="train"`` (default): unchanged historical behaviour.  Forward
      and (optionally) backward hooks accumulate streaming stats and feed
      :func:`scalar_metrics` / :func:`write_artifacts`.  Optimizer state
      can be sampled via :func:`collect_param_metrics`.
    * ``mode="eval"``: forward-only.  No backward hooks, no optimizer
      probing, no training mutation.  Per-task / per-layer / per-role
      streaming aggregates are accumulated and detailed
      :class:`LayerPathMetricRecord`s can optionally be written.  Use
      :meth:`set_sample_context` before each forward, then call
      :meth:`flush_sample` after.

    Switching mode happens at construction time; ``register()`` always
    consults the chosen mode.
    """

    VALID_MODES = ("train", "eval")

    def __init__(
        self,
        model: torch.nn.Module,
        args: DiagnosticsArgs,
        *,
        output_dir: Optional[Path] = None,
        prefix: str = "diag/train",
        mode: str = "train",
        run_id: Optional[str] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}; got {mode!r}")
        self.model = model
        self.args = args
        self.mode = mode
        self.prefix = prefix
        self.run_id = run_id or getattr(args, "run_id", None) or "run"
        self.output_dir = Path(output_dir or args.output_dir or "diagnostics")
        self.handles: List[Any] = []
        self.stats: Dict[str, OnlineStats] = defaultdict(OnlineStats)
        self.geometry: Dict[int, Dict[str, TensorReservoir]] = defaultdict(
            lambda: {
                "stem": TensorReservoir(args.max_tokens_per_layer_geometry),
                "dense": TensorReservoir(args.max_tokens_per_layer_geometry),
            }
        )
        self.token_stats = TokenStatsAggregator(args.frequency_bucket_boundaries, args.token_topk)
        self.current_tokens: Optional[torch.Tensor] = None
        self.current_task: Optional[str] = None
        self.batches_collected = 0
        self._active = False
        self._raw_samples: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Eval-mode state.  Always present for simplicity but only populated
        # when self.mode == "eval".  Per-task aggregation is keyed on
        # ``(task, layer_idx, metric_name)`` so we can emit a compact
        # per-task summary at the end of eval.  Role-bucketed aggregation
        # mirrors that with an extra ``role`` axis.
        # ------------------------------------------------------------------
        self.current_sample_id: Optional[str] = None
        self.current_task_group: Optional[str] = None
        self.current_token_ids_list: Optional[List[int]] = None
        self.current_tokens_list: Optional[List[str]] = None
        self.current_token_roles: Optional[List[str]] = None
        self.current_sample_records: List[Any] = []  # LayerPathMetricRecord

        self.eval_task_layer_stats: Dict[str, Dict[int, Dict[str, OnlineStats]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(OnlineStats)))
        )
        self.eval_task_layer_role_stats: Dict[
            str, Dict[int, Dict[str, Dict[str, OnlineStats]]]
        ] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(OnlineStats)))
        )
        self.eval_task_layer_geometry: Dict[
            str, Dict[int, Dict[str, TensorReservoir]]
        ] = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "stem": TensorReservoir(args.eval_geometry_max_tokens_per_layer),
                    "dense": TensorReservoir(args.eval_geometry_max_tokens_per_layer),
                }
            )
        )
        # Counts of samples seen per task — drives the per-task sample cap.
        self.eval_task_sample_counts: Dict[str, int] = defaultdict(int)
        self.eval_task_layer_path_records_written: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.args.enabled)

    @property
    def is_eval_mode(self) -> bool:
        return self.mode == "eval"

    def __enter__(self) -> "StemDiagnosticsCollector":
        if self.enabled:
            self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def register(self) -> None:
        """Register hooks.

        In eval mode we deliberately register *only* forward hooks: no
        backward hooks, no parameter grad hooks, no optimizer touch points.
        The set of layers is determined by ``args.eval_layers`` in eval
        mode, falling back to ``args.layers``.
        """
        if self.handles or not self.enabled:
            return
        if self.is_eval_mode:
            layers_arg = self.args.eval_layers if self.args.eval_layers is not None else self.args.layers
            iterator = _iter_all_ffn_layers(self.model, layers_arg)
            for layer_idx, layer in iterator:
                ff = layer.feed_forward
                self.handles.append(ff.register_forward_hook(self._make_ffn_hook(layer_idx)))
            return
        for layer_idx, layer in _iter_stem_layers(self.model, self.args.layers):
            ff = layer.feed_forward
            self.handles.append(ff.register_forward_hook(self._make_ffn_hook(layer_idx)))
            if self.args.enable_backward_hooks:
                for name, param in ff.named_parameters(recurse=True):
                    self.handles.append(param.register_hook(self._make_param_grad_hook(layer_idx, name)))
            emb = _stem_embedding_for_layer(self.model, layer_idx)
            if emb is not None and self.args.enable_backward_hooks:
                for name, param in emb.named_parameters(recurse=True):
                    self.handles.append(param.register_hook(self._make_param_grad_hook(layer_idx, f"stem_{name}")))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_batch(
        self,
        token_ids: Optional[torch.Tensor],
        *,
        task: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            self._active = False
            return
        if self.batches_collected >= max(1, self.args.max_batches_per_collection) and not force:
            self._active = False
            return
        self.current_tokens = token_ids.detach() if token_ids is not None else None
        self.current_task = task
        self._active = True
        self.batches_collected += 1

    def end_batch(self) -> None:
        self._active = False
        self.current_tokens = None
        self.current_task = None

    def reset_window(self) -> None:
        self.stats = defaultdict(OnlineStats)
        self.geometry = defaultdict(
            lambda: {
                "stem": TensorReservoir(self.args.max_tokens_per_layer_geometry),
                "dense": TensorReservoir(self.args.max_tokens_per_layer_geometry),
            }
        )
        self.batches_collected = 0

    def _make_param_grad_hook(self, layer_idx: int, name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            norm = _safe_norm(grad)
            if norm is not None:
                self.stats[f"layer_{layer_idx}/{name}_grad_norm"].update(norm)
            return grad
        return hook

    def _sample_positions(self, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if not xs:
            return ()
        n = xs[0].reshape(-1, xs[0].shape[-1]).shape[0]
        limit = min(n, max(1, self.args.max_token_positions))
        if n <= limit:
            idx = torch.arange(n, device=xs[0].device)
        else:
            idx = torch.linspace(0, n - 1, limit, device=xs[0].device).long()
        out = []
        for x in xs:
            flat = x.reshape(-1, x.shape[-1])
            out.append(flat.index_select(0, idx))
        return tuple(out)

    def _make_ffn_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self._active:
                return None
            x = inputs[0] if inputs else None
            y = inputs[1] if len(inputs) > 1 else None
            x = _as_local_tensor(x)
            y = _as_local_tensor(y)
            if x is None:
                return None
            with torch.no_grad():
                xf = x.detach()
                x1 = module.w1(xf.view_as(xf))
                act = F.silu(x1)
                x3 = module.w3(xf.view_as(xf)) if hasattr(module, "w3") else None
                up = y
                alpha_sig = None
                if hasattr(module, "alpha"):
                    alpha_sig = torch.sigmoid(_as_local_tensor(module.alpha)).detach().float()
                    if not self.is_eval_mode:
                        self.stats[f"layer_{layer_idx}/alpha_sigmoid"].update(alpha_sig)
                    if x3 is not None and y is not None:
                        up = (1.0 - alpha_sig.to(x3.device).to(x3.dtype)) * x3 + alpha_sig.to(y.device).to(y.dtype) * y
                    elif x3 is not None and y is None:
                        # Sigmoid-gated DAG eval where the stem path was not
                        # provided (e.g. plain LM forward without stem_embeddings_fn).
                        up = (1.0 - alpha_sig.to(x3.device).to(x3.dtype)) * x3
                if up is None and x3 is not None:
                    # Plain dense FFN with no stem path: combined == dense up.
                    up = x3
                if up is not None:
                    out2 = module.w2(act * up)
                else:
                    out2 = _as_local_tensor(output)

                if self.is_eval_mode:
                    self._eval_record_layer(
                        layer_idx,
                        x=xf,
                        y=y,
                        x3=x3,
                        up=up,
                        out2=out2,
                        x1=x1,
                        act=act,
                        alpha_sig=alpha_sig,
                    )
                    return None

                stem_norm = _safe_norm(y)
                up_norm = _safe_norm(x3)
                combined_norm = _safe_norm(up)
                ffn_out_norm = _safe_norm(out2)
                act_norm = _safe_norm(act)
                if stem_norm is not None:
                    self.stats[f"layer_{layer_idx}/stem_norm"].update(stem_norm)
                if up_norm is not None:
                    self.stats[f"layer_{layer_idx}/up_norm"].update(up_norm)
                if combined_norm is not None:
                    self.stats[f"layer_{layer_idx}/combined_up_norm"].update(combined_norm)
                if ffn_out_norm is not None:
                    self.stats[f"layer_{layer_idx}/ffn_out_norm"].update(ffn_out_norm)
                if act_norm is not None:
                    self.stats[f"layer_{layer_idx}/ffn_activation_norm"].update(act_norm)
                if y is not None and x3 is not None:
                    sy, sx3 = self._sample_positions(y, x3)
                    cos = F.cosine_similarity(sy.float(), sx3.float(), dim=-1, eps=1e-8)
                    self.stats[f"layer_{layer_idx}/stem_up_cos"].update(cos)
                if self.args.collect_geometry:
                    if y is not None:
                        self.geometry[layer_idx]["stem"].add(y)
                    if x3 is not None:
                        self.geometry[layer_idx]["dense"].add(x3)
                if self.args.collect_token_stats and self.current_tokens is not None:
                    self.token_stats.update(
                        layer_idx,
                        self.current_tokens,
                        stem_norm=stem_norm,
                        up_norm=up_norm,
                    )
            return None
        return hook

    # ------------------------------------------------------------------
    # Eval-mode helpers
    # ------------------------------------------------------------------

    def set_sample_context(
        self,
        *,
        task: str,
        sample_id: str,
        task_group: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
        tokens: Optional[List[str]] = None,
        token_roles: Optional[List[str]] = None,
    ) -> None:
        """Bind the next forward pass to a specific lm-eval sample.

        Must be called from eval mode.  After the forward, call
        :meth:`flush_sample` to emit the per-sample detailed records (if
        enabled) and increment the per-task sample counter.
        """
        if not self.is_eval_mode:
            raise RuntimeError("set_sample_context only valid in eval mode")
        # Reset all sample-level state up front so a skipped sample does
        # not leak the previous sample's task / id into ``flush_sample``.
        self.current_task = None
        self.current_sample_id = None
        self.current_task_group = None
        self.current_token_ids_list = None
        self.current_tokens_list = None
        self.current_token_roles = None
        self.current_sample_records = []

        cap = self.args.eval_activation_max_samples_per_task
        if cap is not None and self.eval_task_sample_counts.get(task, 0) >= cap:
            self._active = False
            return
        self.current_task = task
        self.current_sample_id = sample_id
        self.current_task_group = task_group
        self.current_token_ids_list = token_ids
        self.current_tokens_list = tokens
        self.current_token_roles = token_roles
        self._active = True

    @staticmethod
    def _flatten_per_token(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if t.ndim < 2:
            return None
        return t.detach().float().reshape(-1, t.shape[-1])

    def _per_token_norms(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        flat = self._flatten_per_token(t)
        if flat is None or flat.numel() == 0:
            return None
        return flat.norm(dim=-1).cpu()

    def _eval_record_layer(
        self,
        layer_idx: int,
        *,
        x: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        x3: Optional[torch.Tensor],
        up: Optional[torch.Tensor],
        out2: Optional[torch.Tensor],
        x1: Optional[torch.Tensor],
        act: Optional[torch.Tensor],
        alpha_sig: Optional[torch.Tensor],
    ) -> None:
        """Update per-task / per-layer streaming aggregates."""
        from lingua.diagnostic_records import LayerPathMetricRecord  # lazy

        if self.current_task is None or self.current_sample_id is None:
            return
        task = self.current_task
        # Cap tokens per sample by sub-sampling positions consistently across
        # all per-token tensors so role/index alignment is preserved.
        max_tokens = max(1, int(self.args.eval_activation_max_tokens_per_sample))

        per_token = {
            "stem_norm": self._per_token_norms(y),
            "up_norm": self._per_token_norms(x3),
            "combined_norm": self._per_token_norms(up),
            "ffn_out_norm": self._per_token_norms(out2),
            "w1_act_norm": self._per_token_norms(x1),
            "silu_act_norm": self._per_token_norms(act),
        }
        # Pairwise stem/up cosine, per token.
        cos_per_token: Optional[torch.Tensor] = None
        flat_y = self._flatten_per_token(y)
        flat_x3 = self._flatten_per_token(x3)
        if flat_y is not None and flat_x3 is not None and flat_y.shape == flat_x3.shape:
            cos_per_token = F.cosine_similarity(flat_y, flat_x3, dim=-1, eps=1e-8).cpu()
        per_token["stem_up_cos"] = cos_per_token

        # Determine number of positions and sub-sample indices.
        n_pos = 0
        for v in per_token.values():
            if v is not None:
                n_pos = max(n_pos, int(v.shape[0]))
                break
        if n_pos == 0:
            n_pos = self._flatten_per_token(x).shape[0] if self._flatten_per_token(x) is not None else 0
        if n_pos == 0:
            return
        if n_pos > max_tokens:
            sample_idx = torch.linspace(0, n_pos - 1, max_tokens).long()
        else:
            sample_idx = torch.arange(n_pos)
        sub: Dict[str, Optional[torch.Tensor]] = {}
        for name, vals in per_token.items():
            if vals is None:
                sub[name] = None
            else:
                v = vals
                if v.shape[0] != n_pos:
                    # Length mismatch (e.g. some tensors broadcast to a different
                    # shape).  Skip alignment-dependent metrics in that case.
                    sub[name] = None
                else:
                    sub[name] = v.index_select(0, sample_idx)

        layer_stats = self.eval_task_layer_stats[task][layer_idx]
        token_count_added = 0
        for name, vals in sub.items():
            if vals is None:
                continue
            layer_stats[name].update(vals)
            token_count_added = max(token_count_added, int(vals.numel()))
        if alpha_sig is not None:
            try:
                alpha_val = float(alpha_sig.detach().float().mean().item())
            except Exception:
                alpha_val = None
            if alpha_val is not None and math.isfinite(alpha_val):
                layer_stats["gate_alpha"].update(float(alpha_val))
        # Track token_count separately for visibility (e.g. for debugging
        # when a layer received 0 positions).
        if token_count_added > 0:
            layer_stats["token_count"].update(float(token_count_added))

        # Role-bucketed stats.
        roles = self.current_token_roles
        if roles is not None and len(roles) >= n_pos:
            sampled_roles = [roles[i] for i in sample_idx.tolist()]
            for name, vals in sub.items():
                if vals is None:
                    continue
                vals_list = vals.tolist()
                for role, v in zip(sampled_roles, vals_list):
                    self.eval_task_layer_role_stats[task][layer_idx][role][name].update(float(v))

        # Geometry reservoirs (stem and dense).
        if self.args.collect_eval_geometry:
            geo = self.eval_task_layer_geometry[task][layer_idx]
            if y is not None:
                geo["stem"].add(y)
            if x3 is not None:
                geo["dense"].add(x3)

        # Detailed per-position records (optional).
        if self.args.write_layer_path_records:
            cap_records = max(1, int(self.args.eval_activation_max_layer_path_records_per_sample))
            tok_ids = self.current_token_ids_list or []
            tok_strs = self.current_tokens_list or []
            tok_roles = self.current_token_roles or []
            # Cap the number of detailed records we emit per sample/layer.
            record_idx = sample_idx[:cap_records].tolist()
            for pos_idx in record_idx:
                pos = int(pos_idx)
                rec = LayerPathMetricRecord(
                    run_id=self.run_id,
                    task=task,
                    sample_id=self.current_sample_id,
                    layer_idx=int(layer_idx),
                    token_position=pos,
                    token_id=int(tok_ids[pos]) if pos < len(tok_ids) else None,
                    token=tok_strs[pos] if pos < len(tok_strs) else None,
                    token_role=tok_roles[pos] if pos < len(tok_roles) else None,
                    stem_norm=self._opt_at(sub.get("stem_norm"), pos, sample_idx),
                    up_norm=self._opt_at(sub.get("up_norm"), pos, sample_idx),
                    combined_norm=self._opt_at(sub.get("combined_norm"), pos, sample_idx),
                    stem_up_cos=self._opt_at(sub.get("stem_up_cos"), pos, sample_idx),
                    ffn_out_norm=self._opt_at(sub.get("ffn_out_norm"), pos, sample_idx),
                    gate_alpha=float(alpha_sig.detach().float().mean().item()) if alpha_sig is not None else None,
                    w1_act_norm=self._opt_at(sub.get("w1_act_norm"), pos, sample_idx),
                    silu_act_norm=self._opt_at(sub.get("silu_act_norm"), pos, sample_idx),
                    metadata={
                        "task_group": self.current_task_group,
                    },
                )
                self.current_sample_records.append(rec)

    @staticmethod
    def _opt_at(vals: Optional[torch.Tensor], pos: int, sample_idx: torch.Tensor) -> Optional[float]:
        """Look up the value for absolute position *pos* in a sub-sampled tensor.

        ``sample_idx`` carries the original positions so we map *pos* back
        to its location in the sub-sampled tensor.
        """
        if vals is None:
            return None
        # sample_idx is sorted ascending; find the index where it equals pos.
        try:
            offsets = (sample_idx == pos).nonzero(as_tuple=False)
        except Exception:
            return None
        if offsets.numel() == 0:
            return None
        idx = int(offsets[0].item())
        if idx >= int(vals.shape[0]):
            return None
        v = float(vals[idx].item())
        return v if math.isfinite(v) else None

    def flush_sample(self) -> List[Any]:
        """Finalise the in-flight sample and return any detailed records.

        Returns the list of :class:`LayerPathMetricRecord` accumulated for
        this sample.  Callers should write them to a JSONL file (we do not
        write here so that all IO can be batched centrally).  The per-task
        sample counter is incremented exactly once per ``set_sample_context``
        / ``flush_sample`` pair.
        """
        if not self.is_eval_mode:
            return []
        records = list(self.current_sample_records)
        # Only count the sample if the forward actually ran (``_active``
        # was set true by ``set_sample_context``).  Skipped samples leave
        # the per-task counter alone so retries / re-attempts are safe.
        if self._active and self.current_task is not None:
            self.eval_task_sample_counts[self.current_task] += 1
        self.current_sample_records = []
        self.current_sample_id = None
        self.current_task = None
        self.current_task_group = None
        self.current_token_ids_list = None
        self.current_tokens_list = None
        self.current_token_roles = None
        self._active = False
        return records

    def collect_param_metrics(self, optimizers: Optional[Any] = None) -> Dict[str, float]:
        if not self.enabled:
            return {}
        metrics: Dict[str, float] = {}
        for layer_idx, layer in _iter_stem_layers(self.model, self.args.layers):
            ff = layer.feed_forward
            for pname in ("w1", "w2", "w3"):
                mod = getattr(ff, pname, None)
                if mod is None:
                    continue
                p = getattr(mod, "weight", None)
                if p is None:
                    continue
                g = _as_local_tensor(p.grad)
                param = _as_local_tensor(p)
                if g is not None:
                    metrics[f"{self.prefix}/layer_{layer_idx}/{pname}_grad_norm"] = _finite(float(g.detach().float().norm().item()))
                if param is not None and g is not None:
                    metrics[f"{self.prefix}/layer_{layer_idx}/{pname}_grad_param_ratio"] = _finite(
                        float(g.detach().float().norm().item() / (param.detach().float().norm().item() + 1e-12))
                    )
            emb = _stem_embedding_for_layer(self.model, layer_idx)
            if emb is not None:
                weight_param = getattr(emb, "weight", None)
                weight = _as_local_tensor(weight_param)
                grad = _as_local_tensor(getattr(weight_param, "grad", None))
                if grad is not None:
                    metrics[f"{self.prefix}/layer_{layer_idx}/stem_grad_norm"] = _finite(float(grad.detach().float().norm().item()))
                if weight is not None and grad is not None:
                    metrics[f"{self.prefix}/layer_{layer_idx}/stem_grad_param_ratio"] = _finite(
                        float(grad.detach().float().norm().item() / (weight.detach().float().norm().item() + 1e-12))
                    )
                    self.geometry[layer_idx]["stem"].add(weight)
                step = self._adam_step_for_param(weight_param, optimizers)
                if step is not None:
                    metrics[f"{self.prefix}/layer_{layer_idx}/stem_step_norm"] = _finite(
                        float(step.detach().float().norm().item())
                    )
                    if weight is not None:
                        metrics[f"{self.prefix}/layer_{layer_idx}/stem_step_param_ratio"] = _finite(
                            float(step.detach().float().norm().item() / (weight.detach().float().norm().item() + 1e-12))
                        )
        if self.args.collect_optimizer_stats or self.args.enable_optimizer_moment_logging:
            metrics.update(self._optimizer_metrics(optimizers))
        return metrics

    def _adam_step_for_param(self, param: Any, optimizers: Optional[Any]) -> Optional[torch.Tensor]:
        if param is None or optimizers is None:
            return None
        opt_items = optimizers.items() if isinstance(optimizers, dict) else [("optimizer", optimizers)]
        for _opt_name, opt in opt_items:
            state = getattr(opt, "state", {}).get(param, {})
            exp_avg = _as_local_tensor(state.get("exp_avg"))
            exp_avg_sq = _as_local_tensor(state.get("exp_avg_sq"))
            if exp_avg is None or exp_avg_sq is None:
                continue
            lr = 0.0
            eps = 1e-8
            for group in getattr(opt, "param_groups", []):
                if any(p is param for p in group.get("params", [])):
                    lr = group.get("lr", 0.0)
                    eps = group.get("eps", eps)
                    break
            return lr * exp_avg / (exp_avg_sq.sqrt() + eps)
        return None

    def _optimizer_metrics(self, optimizers: Optional[Any]) -> Dict[str, float]:
        if optimizers is None:
            return {}
        opt_items = optimizers.items() if isinstance(optimizers, dict) else [("optimizer", optimizers)]
        out: Dict[str, float] = {}
        for opt_name, opt in opt_items:
            exp_avg_norms = []
            exp_avg_sq_norms = []
            for group in getattr(opt, "param_groups", []):
                lr = group.get("lr", 0.0)
                for p in group.get("params", []):
                    state = opt.state.get(p, {})
                    param = _as_local_tensor(p)
                    if param is None:
                        continue
                    exp_avg = _as_local_tensor(state.get("exp_avg"))
                    exp_avg_sq = _as_local_tensor(state.get("exp_avg_sq"))
                    if exp_avg is not None:
                        exp_avg_norms.append(exp_avg.detach().float().norm())
                    if exp_avg_sq is not None:
                        exp_avg_sq_norms.append(exp_avg_sq.detach().float().norm())
                    if exp_avg is not None and exp_avg_sq is not None:
                        step = lr * exp_avg / (exp_avg_sq.sqrt() + group.get("eps", 1e-8))
                        out[f"{self.prefix}/optim/{opt_name}_step_param_ratio"] = _finite(
                            float(step.detach().float().norm().item() / (param.detach().float().norm().item() + 1e-12))
                        )
            if exp_avg_norms:
                out[f"{self.prefix}/optim/{opt_name}_adam_m1_norm_mean"] = _finite(float(torch.stack(exp_avg_norms).mean().item()))
            if exp_avg_sq_norms:
                out[f"{self.prefix}/optim/{opt_name}_adam_m2_norm_mean"] = _finite(float(torch.stack(exp_avg_sq_norms).mean().item()))
        return out

    # ------------------------------------------------------------------
    # Eval-mode aggregation outputs
    # ------------------------------------------------------------------

    # Compact per-layer metric names emitted into the metrics logger.
    EVAL_SCALAR_KEYS: Tuple[str, ...] = (
        "stem_norm",
        "up_norm",
        "combined_norm",
        "stem_up_cos",
        "ffn_out_norm",
        "gate_alpha",
        "w1_act_norm",
        "silu_act_norm",
    )

    def eval_scalar_metrics(self) -> Dict[str, float]:
        """Return compact metrics suitable for the metrics-logger JSONL.

        Keys follow the convention::

            diag/eval/{task}/layer_{L}/{metric}_mean
            diag/eval/{task}/layer_{L}/gate_alpha_std
            diag/eval/global/layer_{L}/{metric}_mean

        ``effective_rank`` is added under the per-task / global layer key
        when geometry is enabled.
        """
        if not self.is_eval_mode or not self.enabled:
            return {}
        out: Dict[str, float] = {}
        # Per-task per-layer.
        for task, by_layer in self.eval_task_layer_stats.items():
            for layer_idx, metrics in by_layer.items():
                for name, stats in metrics.items():
                    if stats.count == 0:
                        continue
                    out[f"diag/eval/{task}/layer_{layer_idx}/{name}_mean"] = _finite(stats.mean)
                    if name == "gate_alpha":
                        var = stats.m2 / max(1, stats.count - 1)
                        out[f"diag/eval/{task}/layer_{layer_idx}/gate_alpha_std"] = _finite(
                            math.sqrt(max(var, 0.0))
                        )
        # Global aggregate across tasks (mean of per-task means).
        global_by_layer: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for task, by_layer in self.eval_task_layer_stats.items():
            for layer_idx, metrics in by_layer.items():
                for name, stats in metrics.items():
                    if stats.count == 0:
                        continue
                    global_by_layer[layer_idx][name].append(stats.mean)
        for layer_idx, by_name in global_by_layer.items():
            for name, vals in by_name.items():
                if not vals:
                    continue
                m = float(sum(vals) / len(vals))
                out[f"diag/eval/global/layer_{layer_idx}/{name}_mean"] = _finite(m)
        # Geometry-derived scalars.
        if self.args.collect_eval_geometry:
            geo_metrics = self.eval_geometry_summary().get("scalar_metrics", {})
            out.update(geo_metrics)
        return out

    def eval_summary_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for ``eval_activation_summary.json``."""
        if not self.is_eval_mode:
            return {}
        per_task: Dict[str, Any] = {}
        for task, by_layer in self.eval_task_layer_stats.items():
            task_entry: Dict[str, Any] = {
                "task_layers": {},
                "samples_seen": int(self.eval_task_sample_counts.get(task, 0)),
            }
            for layer_idx, metrics in by_layer.items():
                layer_entry: Dict[str, Any] = {}
                for name, stats in metrics.items():
                    if stats.count == 0:
                        continue
                    var = stats.m2 / max(1, stats.count - 1)
                    layer_entry[f"{name}_mean"] = _finite(stats.mean)
                    layer_entry[f"{name}_std"] = _finite(math.sqrt(max(var, 0.0)))
                    layer_entry[f"{name}_count"] = int(stats.count)
                # Role-bucketed sub-aggregates.
                role_entry: Dict[str, Any] = {}
                for role, role_metrics in self.eval_task_layer_role_stats.get(task, {}).get(layer_idx, {}).items():
                    role_dict: Dict[str, Any] = {}
                    for rname, rstats in role_metrics.items():
                        if rstats.count == 0:
                            continue
                        role_dict[f"{rname}_mean"] = _finite(rstats.mean)
                        role_dict[f"{rname}_count"] = int(rstats.count)
                    if role_dict:
                        role_entry[role] = role_dict
                if role_entry:
                    layer_entry["by_role"] = role_entry
                if layer_entry:
                    task_entry["task_layers"][str(int(layer_idx))] = layer_entry
            per_task[task] = task_entry
        out = {
            "run_id": self.run_id,
            "mode": self.mode,
            "per_task": per_task,
            "scalar_metrics": self.eval_scalar_metrics(),
        }
        if self.args.collect_eval_geometry:
            out["geometry"] = self.eval_geometry_summary().get("per_task_layer", {})
        return out

    def eval_geometry_summary(self) -> Dict[str, Any]:
        """Compute geometry metrics from per-task / per-layer reservoirs.

        Returns a dict with two top-level keys: ``per_task_layer`` (rich
        per-cell measurements suitable for JSON dump) and ``scalar_metrics``
        (compact key/value pairs for the metrics logger).
        """
        if not self.is_eval_mode:
            return {}
        per_task_layer: Dict[str, Dict[str, Any]] = {}
        scalar_metrics: Dict[str, float] = {}
        # Aggregate dense up-path mean / top PC across tasks for the
        # alignment metrics on stem-only blocks.
        global_dense_mean: Dict[int, torch.Tensor] = {}
        global_dense_pc: Dict[int, torch.Tensor] = {}
        for task, by_layer in self.eval_task_layer_geometry.items():
            for layer_idx, reservoirs in by_layer.items():
                dense = reservoirs["dense"].tensor()
                if dense is not None and dense.shape[0] >= 2:
                    centroid = dense.mean(dim=0, keepdim=True)
                    global_dense_mean[layer_idx] = centroid.squeeze(0)
                    try:
                        _, _, vh = torch.linalg.svd(dense - centroid, full_matrices=False)
                        global_dense_pc[layer_idx] = vh[0]
                    except RuntimeError:
                        pass
        for task, by_layer in self.eval_task_layer_geometry.items():
            per_layer: Dict[str, Any] = {}
            for layer_idx, reservoirs in by_layer.items():
                stem = reservoirs["stem"].tensor()
                dense = reservoirs["dense"].tensor()
                cell: Dict[str, Any] = {}
                if stem is not None and stem.shape[0] >= 2:
                    geo = geometry_summary(stem, var_frac=self.args.geometry_var_frac, compare_to=dense)
                    for k, v in geo.items():
                        cell[f"stem_{k}"] = v
                    if "effective_rank" in geo:
                        scalar_metrics[
                            f"diag/eval/{task}/layer_{layer_idx}/effective_rank"
                        ] = _finite(float(geo["effective_rank"]))
                if dense is not None and dense.shape[0] >= 2:
                    geo = geometry_summary(dense, var_frac=self.args.geometry_var_frac)
                    for k, v in geo.items():
                        cell[f"dense_{k}"] = v
                # Alignment to *global* dense up-path mean / PC if available.
                if stem is not None and stem.shape[0] >= 1:
                    centroid = stem.mean(dim=0)
                    if layer_idx in global_dense_mean:
                        cos = F.cosine_similarity(
                            centroid.unsqueeze(0),
                            global_dense_mean[layer_idx].unsqueeze(0),
                            dim=-1,
                            eps=1e-8,
                        )
                        cell["alignment_with_global_dense_mean"] = _finite(float(cos.item()))
                    if layer_idx in global_dense_pc:
                        cos = F.cosine_similarity(
                            stem,
                            global_dense_pc[layer_idx].unsqueeze(0).expand_as(stem),
                            dim=-1,
                            eps=1e-8,
                        ).abs().mean()
                        cell["alignment_with_global_dense_top_pc"] = _finite(float(cos.item()))
                if cell:
                    per_layer[str(int(layer_idx))] = cell
            if per_layer:
                per_task_layer[task] = per_layer
        return {"per_task_layer": per_task_layer, "scalar_metrics": scalar_metrics}

    def write_eval_artifacts(
        self,
        *,
        output_dir: Optional[Path] = None,
        layer_path_records: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Write eval-mode artifacts to disk and return the summary dict.

        Always rank-0 only (delegates to :func:`append_jsonl` /
        :func:`write_json_atomic` which gate on rank).  When
        ``layer_path_records`` is provided it is appended to
        ``layer_path_metrics.jsonl``; otherwise the caller is assumed to
        have streamed them already.
        """
        from lingua.diagnostic_records import (  # lazy
            append_jsonl,
            write_json_atomic,
        )

        if not self.is_eval_mode or not self.enabled:
            return {}
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = self.eval_summary_dict()
        write_json_atomic(out_dir / "eval_activation_summary.json", summary)

        if self.args.write_layer_path_records and layer_path_records:
            append_jsonl(
                out_dir / "layer_path_metrics.jsonl",
                layer_path_records,
                rank0_only=True,
            )

        if self.args.collect_eval_geometry:
            geo = self.eval_geometry_summary()
            if geo.get("per_task_layer"):
                write_json_atomic(out_dir / "eval_geometry_summary.json", geo)
            if self.args.eval_geometry_save_npz:
                try:
                    import numpy as np  # type: ignore
                    # Aggregate per-layer across tasks (concatenate samples).
                    per_layer: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
                        lambda: {"stem": [], "dense": []}
                    )
                    for _task, by_layer in self.eval_task_layer_geometry.items():
                        for layer_idx, reservoirs in by_layer.items():
                            for path_name in ("stem", "dense"):
                                t = reservoirs[path_name].tensor()
                                if t is not None:
                                    per_layer[layer_idx][path_name].append(t)
                    for layer_idx, paths in per_layer.items():
                        arrays: Dict[str, Any] = {}
                        for path_name, chunks in paths.items():
                            if chunks:
                                arrays[path_name] = torch.cat(chunks, dim=0).cpu().numpy()
                        if arrays:
                            np.savez(
                                out_dir / f"eval_geometry_layer_{layer_idx}.npz",
                                **arrays,
                            )
                except Exception:
                    pass
        return summary

    def scalar_metrics(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        metrics: Dict[str, float] = {}
        for name, stat in self.stats.items():
            for key, val in stat.as_dict(f"{self.prefix}/{name}").items():
                if key.endswith("_mean") or key.endswith("_std") or key.endswith("_count"):
                    metrics[key] = val
        if self.args.collect_geometry:
            for layer_idx, reservoirs in self.geometry.items():
                stem = reservoirs["stem"].tensor()
                dense = reservoirs["dense"].tensor()
                if stem is not None:
                    geo = geometry_summary(stem, var_frac=self.args.geometry_var_frac, compare_to=dense)
                    for k, v in geo.items():
                        metrics[f"{self.prefix}/layer_{layer_idx}/stem_{k}"] = v
                if dense is not None:
                    geo = geometry_summary(dense, var_frac=self.args.geometry_var_frac)
                    for k, v in geo.items():
                        metrics[f"{self.prefix}/layer_{layer_idx}/dense_{k}"] = v
        return metrics

    def write_artifacts(self, extra_summary: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "diagnostics": asdict(self.args),
            "metrics": self.scalar_metrics(),
            "token_rankings": self.token_stats.rankings(),
        }
        if extra_summary:
            summary.update(extra_summary)
        with open(self.output_dir / "summary_train.json", "w") as f:
            json.dump(summary, f, indent=2)
        if self.args.collect_token_stats:
            with open(self.output_dir / "token_stats.jsonl", "w") as f:
                for row in self.token_stats.rows():
                    print(json.dumps(row), file=f)

        # ------------------------------------------------------------------
        # Canonical train-prefixed artifact files.  These are schema-
        # compatible with eval diagnostics so that offline tools can join
        # train and eval outputs by file name prefix without special-casing.
        # ------------------------------------------------------------------
        task_label: str = getattr(self.args, "train_task_label", None) or "train"

        # train_layer_path_summary.json — per-layer streaming stats, keyed as
        # {"task_label": ..., "per_layer": {"0": {"metric_mean": ...}, ...}}.
        per_layer_summary: Dict[str, Any] = {}
        for stat_key, stat in self.stats.items():
            m = re.match(r"layer_(\d+)/(.*)", stat_key)
            if m:
                layer_k, metric_k = m.group(1), m.group(2)
                per_layer_summary.setdefault(layer_k, {}).update(
                    stat.as_dict(metric_k)
                )
        write_json_atomic(
            self.output_dir / "train_layer_path_summary.json",
            {"task_label": task_label, "per_layer": per_layer_summary},
        )

        # train_token_effects.jsonl — token stats rows tagged with task_label.
        if self.args.collect_token_stats:
            with open(self.output_dir / "train_token_effects.jsonl", "w") as f:
                for row in self.token_stats.rows():
                    row.setdefault("task", task_label)
                    print(json.dumps(row), file=f)

        if self.args.collect_geometry:
            for layer_idx, reservoirs in self.geometry.items():
                arrays = {}
                stem = reservoirs["stem"].tensor()
                dense = reservoirs["dense"].tensor()
                if stem is not None:
                    arrays["stem"] = stem.numpy()
                if dense is not None:
                    arrays["dense"] = dense.numpy()
                if arrays:
                    import numpy as np

                    np.savez(self.output_dir / f"geometry_layer_{layer_idx}.npz", **arrays)
        readme = self.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# STEM Diagnostics Artifacts\n\n"
                "Generated by `lingua.diagnostics`. Compact scalar summaries are also "
                "logged into the existing metrics JSONL files under `diag/...` keys.\n",
                encoding="utf-8",
            )


@contextmanager
def stem_intervention(
    model: torch.nn.Module,
    *,
    kind: str,
    layer_idx: Optional[int] = None,
    layers: Optional[List[int]] = None,
    gate_value: Optional[float] = None,
    replacement: Optional[torch.Tensor] = None,
):
    """Temporarily patch STEM FFNs for causal path analysis."""

    _missing = object()
    kind_aliases = {
        "ablate_up": "ablate_dense",
        "replace_up_mean": "replace_dense_mean",
    }
    kind = kind_aliases.get(kind, kind)
    patches: List[Tuple[torch.nn.Module, Any]] = []
    target_layers = [layer_idx] if layer_idx is not None else layers
    for idx, layer in _iter_stem_layers(model, target_layers):
        ff = layer.feed_forward
        old_forward = ff.forward
        old_instance_forward = getattr(ff, "__dict__", {}).get("forward", _missing)

        def make_forward(module, original_forward):
            def forward(self, x, y=None):
                if kind == "none":
                    return original_forward(x, y)
                x1 = self.w1(x.view_as(x))
                stem = y
                dense = self.w3(x.view_as(x)) if hasattr(self, "w3") else None
                if kind in {"ablate_stem", "ablate_combined"} and stem is not None:
                    stem = torch.zeros_like(stem)
                elif kind == "replace_stem_mean" and stem is not None:
                    rep = replacement.to(device=stem.device, dtype=stem.dtype) if replacement is not None else stem.mean(dim=(0, 1), keepdim=True)
                    stem = rep.expand_as(stem)
                if kind in {"ablate_dense", "ablate_combined"} and dense is not None:
                    dense = torch.zeros_like(dense)
                elif kind == "replace_dense_mean" and dense is not None:
                    rep = replacement.to(device=dense.device, dtype=dense.dtype) if replacement is not None else dense.mean(dim=(0, 1), keepdim=True)
                    dense = rep.expand_as(dense)
                # Reproduce the module's real combination rule so that
                # ablations/replacements measure deltas against the actual
                # forward, not a mismatched one.  DAG blocks expose
                # ``alpha_mode`` which selects between the sigmoid-gated and
                # pure-sum variants; plain STEM blocks have neither.
                ff_alpha_mode = getattr(self, "alpha_mode", None)
                if (
                    ff_alpha_mode == "sigmoid_gated"
                    and hasattr(self, "alpha")
                    and dense is not None
                    and stem is not None
                ):
                    if kind == "force_gate" and gate_value is not None:
                        alpha_sig = torch.tensor(gate_value, device=x.device, dtype=x.dtype)
                    else:
                        alpha_sig = torch.sigmoid(_as_local_tensor(self.alpha)).to(device=x.device, dtype=x.dtype)
                    up = (1.0 - alpha_sig) * dense + alpha_sig * stem
                elif ff_alpha_mode == "sum" and dense is not None and stem is not None:
                    up = dense + stem
                elif stem is not None:
                    up = stem
                elif dense is not None:
                    up = dense
                else:
                    return original_forward(x, y)
                return self.w2(F.silu(x1) * up)

            return types.MethodType(forward, module)

        ff.forward = make_forward(ff, old_forward)
        patches.append((ff, old_instance_forward))
    try:
        yield
    finally:
        for module, old_instance_forward in patches:
            if old_instance_forward is _missing:
                try:
                    delattr(module, "forward")
                except AttributeError:
                    pass
            else:
                module.forward = old_instance_forward
            restored = getattr(module, "__dict__", {}).get("forward", _missing)
            assert restored is old_instance_forward


@torch.no_grad()
def compute_loss(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return model(input_ids, labels).detach()


@torch.no_grad()
def run_intervention_suite(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    args: DiagnosticsArgs,
    *,
    prefix: str = "diag/eval",
    token_ids_for_roles: Optional[torch.Tensor] = None,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not args.enabled or not args.collect_interventions:
        return {}, []
    base = compute_loss(model, input_ids, labels)
    metrics: Dict[str, float] = {f"{prefix}/intervention/base_loss": float(base.item())}
    rows: List[Dict[str, Any]] = []

    kinds = ["ablate_stem", "ablate_dense", "replace_stem_mean", "replace_dense_mean"]
    for layer_idx, _layer in _iter_stem_layers(model, args.layers):
        layer_deltas: Dict[str, float] = {}
        for kind in kinds:
            if "dense" in kind and not hasattr(_layer.feed_forward, "w3"):
                continue
            with stem_intervention(model, kind=kind, layer_idx=layer_idx):
                loss = compute_loss(model, input_ids, labels)
            delta = float((loss - base).item())
            metrics[f"{prefix}/intervention/layer_{layer_idx}/{kind}_delta_loss"] = delta
            layer_deltas[kind] = delta
            rows.append(
                {
                    "layer": layer_idx,
                    "kind": kind,
                    "base_loss": float(base.item()),
                    "loss": float(loss.item()),
                    "delta_loss": delta,
                }
            )
        if "ablate_stem" in layer_deltas and "ablate_dense" in layer_deltas:
            stem_delta = layer_deltas["ablate_stem"]
            dense_delta = layer_deltas["ablate_dense"]
            if stem_delta > 0 and dense_delta > 0:
                rel = "cooperative"
            elif stem_delta < 0 or dense_delta < 0:
                rel = "destructive"
            else:
                rel = "redundant"
            rows.append({"layer": layer_idx, "kind": "path_relation", "relation": rel})
        if hasattr(_layer.feed_forward, "alpha"):
            for gate in args.gate_alpha_values:
                with stem_intervention(model, kind="force_gate", layer_idx=layer_idx, gate_value=gate):
                    loss = compute_loss(model, input_ids, labels)
                delta = float((loss - base).item())
                metrics[f"{prefix}/intervention/layer_{layer_idx}/gate_{gate:g}_delta_loss"] = delta
                rows.append(
                    {
                        "layer": layer_idx,
                        "kind": "force_gate",
                        "gate_value": gate,
                        "base_loss": float(base.item()),
                        "loss": float(loss.item()),
                        "delta_loss": delta,
                    }
                )
    return metrics, rows


def write_intervention_rows(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "interventions.jsonl", "a") as f:
        for row in rows:
            print(json.dumps(row), file=f)


def write_train_intervention_rows(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    task_label: str = "train",
) -> None:
    """Write train-time intervention rows to the canonical train artifacts.

    Writes to both ``interventions.jsonl`` (backward-compatible) and to the
    schema-stable ``train_interventions.jsonl`` file.  Rows are augmented with
    a ``task`` field set to *task_label* when not already present.
    """
    if not rows:
        return
    augmented = [
        ({**r, "task": task_label} if "task" not in r else r) for r in rows
    ]
    write_intervention_rows(output_dir, augmented)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_interventions.jsonl", "a") as f:
        for row in augmented:
            print(json.dumps(row), file=f)


def _model_max_seqlen(model: torch.nn.Module, fallback: int = 2048) -> int:
    max_len = getattr(model, "max_seqlen", None)
    if max_len is None:
        lm = getattr(model, "lm_transformer", None)
        max_len = getattr(lm, "max_seqlen", None) if lm is not None else None
    try:
        return int(max_len) if max_len is not None else int(fallback)
    except Exception:
        return int(fallback)


def _selected_intervention_layers(model: torch.nn.Module, args: DiagnosticsArgs) -> List[int]:
    requested = args.intervention_layers
    if requested is None:
        requested = args.eval_layers if args.eval_layers is not None else args.layers
    return [idx for idx, _layer in _iter_stem_layers(model, requested)]


def _layer_has_up_path(layer: torch.nn.Module) -> bool:
    return hasattr(getattr(layer, "feed_forward", None), "w3")


def _layer_has_gate(layer: torch.nn.Module) -> bool:
    return hasattr(getattr(layer, "feed_forward", None), "alpha")


def _force_gate_value(intervention_type: str) -> Optional[float]:
    if not intervention_type.startswith("force_gate_"):
        return None
    suffix = intervention_type[len("force_gate_") :]
    try:
        return float(suffix.replace("_", "."))
    except ValueError:
        return None


def _build_intervention_specs(
    model: torch.nn.Module,
    args: DiagnosticsArgs,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    layers = _selected_intervention_layers(model, args)
    by_idx = {idx: layer for idx, layer in _iter_stem_layers(model, layers)}
    requested = list(args.intervention_types or [])
    specs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    def add_global(intervention_type: str, kind: str, target_path: str) -> None:
        if not layers:
            skipped.append({"intervention_type": intervention_type, "reason": "no_target_layers"})
            return
        if target_path == "up" and not any(_layer_has_up_path(by_idx[idx]) for idx in layers):
            skipped.append({"intervention_type": intervention_type, "reason": "missing_up_path"})
            return
        specs.append(
            {
                "intervention_name": f"{intervention_type}_all_layers",
                "intervention_type": intervention_type,
                "target_path": target_path,
                "kind": kind,
                "layer_idx": None,
                "layers": layers,
                "gate_value": None,
            }
        )

    for intervention_type in requested:
        if intervention_type == "ablate_stem":
            add_global(intervention_type, "ablate_stem", "stem")
        elif intervention_type == "ablate_up":
            add_global(intervention_type, "ablate_up", "up")
        elif intervention_type == "ablate_combined":
            add_global(intervention_type, "ablate_combined", "combined")
        elif intervention_type == "replace_stem_mean":
            add_global(intervention_type, "replace_stem_mean", "stem")
        elif intervention_type == "replace_up_mean":
            add_global(intervention_type, "replace_up_mean", "up")
        elif intervention_type == "ablate_layer_stem":
            if not layers:
                skipped.append({"intervention_type": intervention_type, "reason": "no_target_layers"})
            for layer_idx in layers:
                specs.append(
                    {
                        "intervention_name": f"{intervention_type}_layer_{layer_idx}",
                        "intervention_type": intervention_type,
                        "target_path": "stem",
                        "kind": "ablate_stem",
                        "layer_idx": int(layer_idx),
                        "layers": [int(layer_idx)],
                        "gate_value": None,
                    }
                )
        elif intervention_type == "ablate_layer_up":
            emitted = False
            for layer_idx in layers:
                if not _layer_has_up_path(by_idx[layer_idx]):
                    continue
                emitted = True
                specs.append(
                    {
                        "intervention_name": f"{intervention_type}_layer_{layer_idx}",
                        "intervention_type": intervention_type,
                        "target_path": "up",
                        "kind": "ablate_up",
                        "layer_idx": int(layer_idx),
                        "layers": [int(layer_idx)],
                        "gate_value": None,
                    }
                )
            if not emitted:
                skipped.append({"intervention_type": intervention_type, "reason": "missing_up_path"})
        elif intervention_type.startswith("force_gate_"):
            gate_value = _force_gate_value(intervention_type)
            if gate_value is None:
                skipped.append({"intervention_type": intervention_type, "reason": "invalid_gate_value"})
                continue
            emitted = False
            for layer_idx in layers:
                if not _layer_has_gate(by_idx[layer_idx]):
                    continue
                emitted = True
                specs.append(
                    {
                        "intervention_name": f"{intervention_type}_layer_{layer_idx}",
                        "intervention_type": intervention_type,
                        "target_path": "gate",
                        "kind": "force_gate",
                        "layer_idx": int(layer_idx),
                        "layers": [int(layer_idx)],
                        "gate_value": float(gate_value),
                    }
                )
            if not emitted:
                skipped.append({"intervention_type": intervention_type, "reason": "missing_gate"})
        else:
            skipped.append({"intervention_type": intervention_type, "reason": "unknown_intervention_type"})
    return specs, skipped


def _delta_metric_name(intervention_type: str) -> str:
    if intervention_type in {"ablate_stem", "ablate_layer_stem"}:
        return "stem_ablation_delta_loss"
    if intervention_type in {"ablate_up", "ablate_layer_up"}:
        return "up_ablation_delta_loss"
    if intervention_type == "ablate_combined":
        return "combined_ablation_delta_loss"
    return f"{intervention_type}_delta_loss"


def _classify_path_relation(
    stem_delta: Optional[float],
    up_delta: Optional[float],
    *,
    eps: float = 1e-4,
    dominance_ratio: float = 2.0,
) -> str:
    if stem_delta is None or up_delta is None:
        return "inconclusive"
    stem = float(stem_delta)
    up = float(up_delta)
    if abs(stem) <= eps and abs(up) <= eps:
        return "redundant"
    if stem < -eps and up < -eps:
        return "destructive_stem" if abs(stem) >= abs(up) else "destructive_up"
    if stem < -eps:
        return "destructive_stem"
    if up < -eps:
        return "destructive_up"
    if stem > eps and up > eps:
        if stem >= max(up * dominance_ratio, up + eps):
            return "stem_dominant"
        if up >= max(stem * dominance_ratio, stem + eps):
            return "up_dominant"
        return "cooperative"
    return "inconclusive"


@torch.no_grad()
def compute_loss_and_per_token_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute mean NLL and per-token NLL from the model's logits path."""
    try:
        logits = model(input_ids)
    except Exception:
        return compute_loss(model, input_ids, labels), None
    if isinstance(logits, (tuple, list)):
        logits = logits[0] if logits else None
    if not isinstance(logits, torch.Tensor) or logits.ndim < 3:
        return compute_loss(model, input_ids, labels), None
    steps = min(int(logits.shape[-2]), int(labels.shape[-1]))
    if steps <= 0:
        return compute_loss(model, input_ids, labels), None
    logits = logits[..., :steps, :]
    labels = labels[..., :steps]
    vocab = int(logits.shape[-1])
    per_token = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)
    return per_token.mean().detach(), per_token.detach()


def _safe_tensor(values: List[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)


def _mean_or_none(stat: Optional[OnlineStats]) -> Optional[float]:
    return stat.mean if stat is not None and stat.count else None


def _compact_ranked(items: List[Dict[str, Any]], key: str, *, reverse: bool, topk: int) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda row: row.get(key, 0.0), reverse=reverse)[:topk]


def _summarize_token_effect_records(
    records: List[TokenEffectRecord],
    *,
    topk: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = [asdict(r) for r in records]

    def summarize(scope_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        token_map: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        role_map: Dict[str, OnlineStats] = defaultdict(OnlineStats)
        layer_stem: Dict[int, OnlineStats] = defaultdict(OnlineStats)
        layer_up: Dict[int, OnlineStats] = defaultdict(OnlineStats)
        ineffective: Dict[Tuple[int, str, str], float] = defaultdict(float)
        counts: Counter[Tuple[int, str, str]] = Counter()
        for row in scope_rows:
            token_key = (int(row["token_id"]), row.get("token") or "", row.get("token_role") or "unknown")
            counts[token_key] += int(row.get("count") or 0)
            stem_delta = row.get("stem_ablation_delta_loss")
            up_delta = row.get("up_ablation_delta_loss")
            if stem_delta is not None:
                token_map.setdefault(token_key, {"stem": OnlineStats(), "up": OnlineStats()})["stem"].update(stem_delta)
                role_map[token_key[2]].update(stem_delta)
                if row.get("layer_idx") is not None:
                    layer_stem[int(row["layer_idx"])].update(stem_delta)
            if up_delta is not None:
                token_map.setdefault(token_key, {"stem": OnlineStats(), "up": OnlineStats()})["up"].update(up_delta)
                if row.get("layer_idx") is not None:
                    layer_up[int(row["layer_idx"])].update(up_delta)
            ineffective[token_key] = max(ineffective[token_key], float(row.get("ineffective_score") or 0.0))
        token_items: List[Dict[str, Any]] = []
        for (token_id, token, role), stats in token_map.items():
            stem_delta = _mean_or_none(stats.get("stem"))
            benefit, harm, _ineff = TokenStatsAggregator._score_from_stem_delta(
                stem_delta,
                counts[(token_id, token, role)],
            )
            token_items.append(
                {
                    "token_id": token_id,
                    "token": token,
                    "token_role": role,
                    "count": counts[(token_id, token, role)],
                    "stem_ablation_delta_loss": stem_delta,
                    "benefit_score": benefit,
                    "harm_score": harm,
                    "ineffective_score": ineffective[(token_id, token, role)],
                }
            )
        role_items = [
            {
                "token_role": role,
                "stem_ablation_delta_loss": stat.mean,
                "count": stat.count,
                "benefit_score": max(stat.mean, 0.0),
                "harm_score": max(-stat.mean, 0.0),
            }
            for role, stat in role_map.items()
            if stat.count
        ]
        stem_layer_items = [
            {
                "layer_idx": layer,
                "stem_ablation_delta_loss": stat.mean,
                "count": stat.count,
                "benefit_score": max(stat.mean, 0.0),
                "harm_score": max(-stat.mean, 0.0),
            }
            for layer, stat in layer_stem.items()
            if stat.count
        ]
        up_layer_items = [
            {
                "layer_idx": layer,
                "up_ablation_delta_loss": stat.mean,
                "count": stat.count,
                "benefit_score": max(stat.mean, 0.0),
                "harm_score": max(-stat.mean, 0.0),
            }
            for layer, stat in layer_up.items()
            if stat.count
        ]
        return {
            "top_beneficial_stem_tokens": [
                row for row in _compact_ranked(token_items, "benefit_score", reverse=True, topk=topk)
                if row["benefit_score"] > 0
            ],
            "top_harmful_stem_tokens": [
                row for row in _compact_ranked(token_items, "harm_score", reverse=True, topk=topk)
                if row["harm_score"] > 0
            ],
            "top_ineffective_stem_tokens": [
                row for row in _compact_ranked(token_items, "ineffective_score", reverse=True, topk=topk)
                if row["ineffective_score"] > 0
            ],
            "top_beneficial_token_roles": [
                row for row in _compact_ranked(role_items, "benefit_score", reverse=True, topk=topk)
                if row["benefit_score"] > 0
            ],
            "top_harmful_token_roles": [
                row for row in _compact_ranked(role_items, "harm_score", reverse=True, topk=topk)
                if row["harm_score"] > 0
            ],
            "stem_layers_most_beneficial": _compact_ranked(stem_layer_items, "benefit_score", reverse=True, topk=topk),
            "stem_layers_most_harmful": _compact_ranked(stem_layer_items, "harm_score", reverse=True, topk=topk),
            "up_layers_most_beneficial": _compact_ranked(up_layer_items, "benefit_score", reverse=True, topk=topk),
            "up_layers_most_harmful": _compact_ranked(up_layer_items, "harm_score", reverse=True, topk=topk),
        }

    by_task: Dict[str, Any] = {"global": summarize(rows)}
    for task in sorted({row.get("task") for row in rows if row.get("task")}):
        by_task[str(task)] = summarize([row for row in rows if row.get("task") == task])
    by_role: Dict[str, Any] = {
        "global": {
            role: summarize([row for row in rows if row.get("token_role") == role])
            for role in sorted({row.get("token_role") or "unknown" for row in rows})
        },
        "by_task": {},
    }
    for task in sorted({row.get("task") for row in rows if row.get("task")}):
        task_rows = [row for row in rows if row.get("task") == task]
        by_role["by_task"][str(task)] = {
            role: summarize([row for row in task_rows if row.get("token_role") == role])
            for role in sorted({row.get("token_role") or "unknown" for row in task_rows})
        }
    return by_task, by_role


def _update_relation_summary(
    summary: Dict[str, Any],
    *,
    task: str,
    layer_idx: int,
    sample_id: str,
    relation: str,
    stem_delta: Optional[float],
    up_delta: Optional[float],
) -> None:
    task_cell = summary.setdefault(task, {})
    layer_cell = task_cell.setdefault(
        str(int(layer_idx)),
        {
            "counts": {},
            "samples": [],
            "_stem": OnlineStats(),
            "_up": OnlineStats(),
        },
    )
    layer_cell["counts"][relation] = layer_cell["counts"].get(relation, 0) + 1
    if stem_delta is not None:
        layer_cell["_stem"].update(stem_delta)
    if up_delta is not None:
        layer_cell["_up"].update(up_delta)
    layer_cell["samples"].append(
        {
            "sample_id": sample_id,
            "relation": relation,
            "stem_delta_loss": stem_delta,
            "up_delta_loss": up_delta,
        }
    )


def _finalize_relation_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for task, layers in summary.items():
        out[task] = {}
        for layer, cell in layers.items():
            stem_stat = cell.get("_stem")
            up_stat = cell.get("_up")
            out[task][layer] = {
                "counts": cell.get("counts", {}),
                "mean_stem_delta_loss": _mean_or_none(stem_stat),
                "mean_up_delta_loss": _mean_or_none(up_stat),
                "samples": cell.get("samples", []),
            }
    return out


@torch.no_grad()
def run_task_aligned_interventions(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    args: DiagnosticsArgs,
    output_dir: Path,
    run_id: str,
    results: Optional[Dict[str, Any]] = None,
    sample_records: Optional[Iterable[Any]] = None,
    prefix: str = "diag/eval",
    rank: Optional[int] = None,
    model_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run causal interventions on actual lm-eval samples.

    This is the task-aligned counterpart to ``run_prompt_interventions``.  It
    computes the original per-token NLL, reruns each configured intervention,
    writes ``interventions_task_aligned.jsonl``, and optionally feeds signed
    token deltas into :class:`TokenStatsAggregator`.
    """
    if not args.enabled or not args.collect_eval_interventions:
        return {}
    rank0_only = bool(getattr(args, "rank0_only", True))
    if rank0_only and rank is not None and rank != 0:
        return {}
    if tokenizer is None:
        return {"missing_fields": ["no_tokenizer_available"]}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, skipped_specs = _build_intervention_specs(model, args)
    if not specs:
        summary = {
            "run_id": run_id,
            "samples_processed": 0,
            "skipped_interventions": skipped_specs,
            "scalar_metrics": {},
        }
        write_json_atomic(output_dir / "eval_intervention_summary.json", summary)
        return summary

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_seqlen = _model_max_seqlen(model)
    max_tokens_per_sample = int(getattr(args, "eval_activation_max_tokens_per_sample", 256))
    max_per_task = getattr(args, "intervention_max_samples_per_task", None)
    tasks_filter = getattr(args, "tasks", None)

    from lingua.eval_activations import _build_input_ids, disable_kv_cache
    from lingua.eval_sample_capture import (
        iter_lm_eval_samples,
        lm_eval_sample_to_record,
    )

    def iter_records() -> Iterator[Any]:
        if sample_records is not None:
            counts: Counter[str] = Counter()
            filt = set(t.lower() for t in tasks_filter) if tasks_filter else None
            for rec in sample_records:
                task = getattr(rec, "task", None)
                if task is None:
                    continue
                if filt is not None and task.lower() not in filt:
                    continue
                if max_per_task is not None and counts[task] >= max_per_task:
                    continue
                counts[task] += 1
                yield rec
            return
        if results is None:
            return
        for task_name, sample in iter_lm_eval_samples(
            results,
            tasks_filter=tasks_filter,
            max_per_task=max_per_task,
        ):
            yield lm_eval_sample_to_record(
                task_name,
                sample,
                run_id=run_id,
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                capture_prompts=True,
                capture_generations=True,
                capture_token_ids=False,
                tokenizer=tokenizer,
                max_text_chars=getattr(args, "max_text_chars", 4096),
                max_tokens=max_tokens_per_sample,
            )

    metrics_acc: Dict[str, OnlineStats] = defaultdict(OnlineStats)
    intervention_records: List[InterventionRecord] = []
    token_agg = TokenStatsAggregator(args.frequency_bucket_boundaries, args.token_topk)
    relation_summary_raw: Dict[str, Any] = {}
    largest_changes: List[Dict[str, Any]] = []
    samples_processed = 0
    samples_skipped = 0
    samples_failed = 0
    samples_per_task: Counter[str] = Counter()

    was_training = model.training
    model.eval()

    try:
        with disable_kv_cache(model):
            for rec in iter_records():
                task = getattr(rec, "task", None) or "unknown"
                prompt = getattr(rec, "prompt", None)
                completion = getattr(rec, "generation", None) or getattr(rec, "target", None)
                if not prompt and results is not None:
                    # Defensive fallback for records built elsewhere with only
                    # metadata; keep this best-effort and non-fatal.
                    completion = completion
                if not prompt and not completion:
                    samples_skipped += 1
                    continue
                _input_full, tok_ids, tok_strs, tok_roles, prompt_len = _build_input_ids(
                    tokenizer,
                    prompt,
                    completion,
                    max_seqlen=max_seqlen,
                    max_tokens_per_sample=max_tokens_per_sample,
                )
                if tok_ids is None or len(tok_ids) < 2:
                    samples_skipped += 1
                    continue
                label_ids = list(tok_ids[1:])
                label_tokens = list(tok_strs[1:]) if tok_strs is not None and len(tok_strs) >= len(tok_ids) else [
                    str(t) for t in label_ids
                ]
                label_roles = list(tok_roles[1:]) if tok_roles is not None and len(tok_roles) >= len(tok_ids) else [
                    classify_token_role(t) for t in label_tokens
                ]
                input_ids = _safe_tensor(tok_ids[:-1], device=device)
                labels = _safe_tensor(label_ids, device=device)
                try:
                    base_loss, base_nll = compute_loss_and_per_token_nll(model, input_ids, labels)
                except Exception as exc:
                    logger.warning(
                        "diagnostics: task-aligned base loss failed for task=%s sample=%s: %s",
                        task,
                        getattr(rec, "sample_id", "unknown"),
                        exc,
                    )
                    samples_failed += 1
                    continue

                task_group = getattr(rec, "task_group", None) or classify_task_group(task)
                sample_id = getattr(rec, "sample_id", None) or f"{task}:{samples_per_task[task]}"
                layer_deltas: Dict[int, Dict[str, float]] = defaultdict(dict)
                sample_rows: List[InterventionRecord] = []

                for spec in specs:
                    try:
                        with stem_intervention(
                            model,
                            kind=spec["kind"],
                            layer_idx=spec.get("layer_idx"),
                            layers=spec.get("layers"),
                            gate_value=spec.get("gate_value"),
                        ):
                            int_loss, int_nll = compute_loss_and_per_token_nll(model, input_ids, labels)
                    except Exception as exc:
                        logger.warning(
                            "diagnostics: intervention failed task=%s sample=%s intervention=%s: %s",
                            task,
                            sample_id,
                            spec["intervention_name"],
                            exc,
                        )
                        samples_failed += 1
                        continue

                    delta_loss = float((int_loss - base_loss).detach().float().item())
                    delta_per_token: Optional[List[float]] = None
                    delta_tensor: Optional[torch.Tensor] = None
                    if args.compute_per_token_delta and base_nll is not None and int_nll is not None:
                        steps = min(int(base_nll.numel()), int(int_nll.numel()), len(label_ids))
                        if steps > 0:
                            delta_tensor = (int_nll.reshape(-1)[:steps] - base_nll.reshape(-1)[:steps]).detach()
                            delta_per_token = [float(v) for v in delta_tensor.cpu().tolist()]
                    metadata = {
                        "task_group": task_group,
                        "prompt_token_count": int(prompt_len or 0),
                        "label_token_count": len(label_ids),
                        "intervention_layers": spec.get("layers"),
                    }
                    if spec.get("gate_value") is not None:
                        metadata["gate_value"] = spec["gate_value"]
                    row = InterventionRecord(
                        run_id=run_id,
                        task=task,
                        sample_id=sample_id,
                        intervention_name=spec["intervention_name"],
                        intervention_type=spec["intervention_type"],
                        target_path=spec["target_path"],
                        loss_original=float(base_loss.detach().float().item()),
                        loss_intervened=float(int_loss.detach().float().item()),
                        delta_loss=delta_loss,
                        layer_idx=spec.get("layer_idx"),
                        delta_per_token_nll=delta_per_token,
                        token_ids=label_ids[: len(delta_per_token)] if delta_per_token is not None else label_ids,
                        token_roles=label_roles[: len(delta_per_token)] if delta_per_token is not None else label_roles,
                        metadata=metadata,
                    )
                    sample_rows.append(row)
                    metric_key = (
                        f"{prefix}/intervention_task_aligned/{task}/"
                        f"{spec['intervention_name']}_delta_loss"
                    )
                    metrics_acc[metric_key].update(delta_loss)
                    largest_changes.append(
                        {
                            "task": task,
                            "sample_id": sample_id,
                            "layer_idx": spec.get("layer_idx"),
                            "intervention_type": spec["intervention_type"],
                            "delta_loss": delta_loss,
                            "abs_delta_loss": abs(delta_loss),
                        }
                    )
                    layer_idx = spec.get("layer_idx")
                    if layer_idx is not None and spec["intervention_type"] == "ablate_layer_stem":
                        layer_deltas[int(layer_idx)]["stem"] = delta_loss
                    if layer_idx is not None and spec["intervention_type"] == "ablate_layer_up":
                        layer_deltas[int(layer_idx)]["up"] = delta_loss
                    if args.update_token_effectiveness and delta_tensor is not None:
                        steps = int(delta_tensor.numel())
                        token_agg.update_loss_delta(
                            torch.tensor(label_ids[:steps], dtype=torch.long),
                            _delta_metric_name(spec["intervention_type"]),
                            delta_tensor[:steps].cpu(),
                            task=task,
                            task_group=task_group,
                            layer_idx=spec.get("layer_idx"),
                            tokens=label_tokens[:steps],
                            token_roles=label_roles[:steps],
                        )

                for layer_idx in _selected_intervention_layers(model, args):
                    stem_delta = layer_deltas.get(int(layer_idx), {}).get("stem")
                    up_delta = layer_deltas.get(int(layer_idx), {}).get("up")
                    relation = _classify_path_relation(stem_delta, up_delta)
                    if stem_delta is not None or up_delta is not None:
                        _update_relation_summary(
                            relation_summary_raw,
                            task=task,
                            layer_idx=int(layer_idx),
                            sample_id=sample_id,
                            relation=relation,
                            stem_delta=stem_delta,
                            up_delta=up_delta,
                        )
                    for row in sample_rows:
                        if row.layer_idx == int(layer_idx):
                            row.path_relation = relation
                            row.metadata["path_relation"] = relation
                            row.metadata["path_relation_stem_delta_loss"] = stem_delta
                            row.metadata["path_relation_up_delta_loss"] = up_delta

                intervention_records.extend(sample_rows)
                samples_processed += 1
                samples_per_task[task] += 1
    finally:
        if was_training:
            model.train()

    append_jsonl(
        output_dir / TASK_ALIGNED_INTERVENTIONS_JSONL,
        intervention_records,
        rank0_only=True,
        max_prompt_chars=getattr(args, "max_text_chars", 4096),
        max_generation_chars=getattr(args, "max_text_chars", 4096),
    )
    if not intervention_records:
        (output_dir / TASK_ALIGNED_INTERVENTIONS_JSONL).touch(exist_ok=True)

    token_effect_records = token_agg.token_effect_records(run_id=run_id) if args.update_token_effectiveness else []
    if token_effect_records:
        append_jsonl(output_dir / TOKEN_EFFECTS_JSONL, token_effect_records, rank0_only=True)
        by_task, by_role = _summarize_token_effect_records(token_effect_records, topk=args.token_topk)
    else:
        (output_dir / TOKEN_EFFECTS_JSONL).touch(exist_ok=True)
        by_task, by_role = {}, {}
    write_json_atomic(output_dir / TOKEN_EFFECTS_BY_TASK_JSON, by_task)
    write_json_atomic(output_dir / TOKEN_EFFECTS_BY_ROLE_JSON, by_role)

    relation_summary = _finalize_relation_summary(relation_summary_raw)
    write_json_atomic(output_dir / PATH_RELATIONS_BY_TASK_LAYER_JSON, relation_summary)

    scalar_metrics = {f"{key}_mean": stat.mean for key, stat in metrics_acc.items() if stat.count}
    largest_changes = sorted(largest_changes, key=lambda row: row["abs_delta_loss"], reverse=True)[: args.token_topk]
    summary = {
        "run_id": run_id,
        "samples_processed": samples_processed,
        "samples_per_task": dict(samples_per_task),
        "samples_skipped": samples_skipped,
        "samples_failed": samples_failed,
        "intervention_types_requested": list(args.intervention_types or []),
        "intervention_types_implemented": sorted({spec["intervention_type"] for spec in specs}),
        "skipped_interventions": skipped_specs,
        "compute_per_token_delta": bool(args.compute_per_token_delta),
        "update_token_effectiveness": bool(args.update_token_effectiveness),
        "scalar_metrics": scalar_metrics,
        "largest_loss_changes": largest_changes,
        "token_effects_by_task": by_task,
        "token_effects_by_role": by_role,
        "path_relations_by_task_layer": relation_summary,
        "artifacts": {
            "interventions": TASK_ALIGNED_INTERVENTIONS_JSONL,
            "token_effects": TOKEN_EFFECTS_JSONL,
            "token_effects_by_task": TOKEN_EFFECTS_BY_TASK_JSON,
            "token_effects_by_role": TOKEN_EFFECTS_BY_ROLE_JSON,
            "path_relations_by_task_layer": PATH_RELATIONS_BY_TASK_LAYER_JSON,
        },
    }
    write_json_atomic(output_dir / "eval_intervention_summary.json", summary)
    return summary


@torch.no_grad()
def run_prompt_interventions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    args: DiagnosticsArgs,
    *,
    prefix: str = "diag/eval",
    max_prompts: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run intervention analysis on plain text prompts without lm-eval coupling."""
    if not args.enabled or not args.collect_interventions:
        return {}, []
    metrics_acc: Dict[str, OnlineStats] = defaultdict(OnlineStats)
    rows: List[Dict[str, Any]] = []
    max_items = max_prompts or max(1, args.path_ablation_num_batches)
    device = next(model.parameters()).device
    for prompt_id, prompt in enumerate(prompts[:max_items]):
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) < 2:
            continue
        max_len = getattr(model, "max_seqlen", len(ids))
        ids = ids[:max_len]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        metrics, item_rows = run_intervention_suite(
            model,
            input_ids,
            labels,
            args,
            prefix=prefix,
        )
        for key, val in metrics.items():
            metrics_acc[key].update(val)
        for row in item_rows:
            row["prompt_id"] = prompt_id
            row["prompt"] = prompt[:1000] if args.save_raw_samples else None
            rows.append(row)
    merged = {}
    for key, stat in metrics_acc.items():
        if stat.count:
            merged[key] = stat.mean
    return merged, rows


def analyze_code_failure(generated: str, reference: Optional[str] = None) -> Dict[str, Any]:
    """Heuristic single-pass code failure classifier.

    Delegates to :func:`lingua.code_diagnostics.classify_static_failure` for
    the full taxonomy, falling back to a lightweight inline path so this
    function remains importable without the new module.
    """
    try:
        from lingua.code_diagnostics import classify_static_failure
        result = classify_static_failure(
            generated or "",
            prompt=None,
            correct=None,
        )
        return {"category": result["static_category"], "detail": result["detail"]}
    except ImportError:
        pass

    # Minimal inline fallback (kept for import-time safety)
    text = generated or ""
    category = "unknown_failure"
    detail = ""
    try:
        ast.parse(text)
    except SyntaxError as exc:
        msg = str(exc).lower()
        detail = str(exc)
        if "indent" in msg:
            category = "indentation_error"
        elif "eol while scanning string literal" in msg or "unterminated string" in msg:
            category = "unmatched_bracket_or_quote"
        else:
            category = "syntax_error"
    if category == "unknown_failure":
        if _has_unbalanced_delimiters(text):
            category = "unmatched_bracket_or_quote"
        elif re.search(r"import\s+\*|from\s+\w+\s+import", text) and reference and "import" not in reference:
            category = "import_error_or_api_misuse"
        elif "def " not in text and reference and "def " in reference:
            category = "prompt_noncompliance"
        elif re.search(r"\b(for|while|if|return)\b", text):
            category = "logic_error_likely"
    return {"category": category, "detail": detail}


def _has_unbalanced_delimiters(text: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: List[str] = []
    quote: Optional[str] = None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return True
    return bool(stack or quote)


def analyze_eval_samples(results: Dict[str, Any], args: DiagnosticsArgs, output_dir: Path) -> Dict[str, Any]:
    """Analyse code generations from lm-eval results.

    Produces ``code_failure_analysis.json`` with per-task failure counts.
    When ``diagnostics_eval_samples.jsonl`` and
    ``interventions_task_aligned.jsonl`` are present in *output_dir* the
    full causal analysis is also run via
    :func:`lingua.code_diagnostics.run_code_causal_analysis`, which writes
    ``code_causal_failure_analysis.json``.
    """
    if not args.enabled or not args.collect_code_error_taxonomy:
        return {}
    code_tasks = [t.lower() for t in (args.code_tasks or [])]
    samples = results.get("samples", {}) if isinstance(results, dict) else {}
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    examples: List[Dict[str, Any]] = []
    for task_name, task_samples in samples.items():
        if not any(t in task_name.lower() for t in code_tasks):
            continue
        for sample in task_samples:
            generated = (
                sample.get("resps", [""])[0][0]
                if isinstance(sample.get("resps"), list) and sample.get("resps")
                else sample.get("filtered_resps", [""])[0]
                if isinstance(sample.get("filtered_resps"), list) and sample.get("filtered_resps")
                else sample.get("prediction", "")
            )
            reference = sample.get("target") or sample.get("doc", {}).get("target")
            result = analyze_code_failure(str(generated), str(reference) if reference is not None else None)
            counts[task_name][result["category"]] += 1
            if len(examples) < 100:
                examples.append(
                    {
                        "task": task_name,
                        "category": result["category"],
                        "generated": str(generated)[:2000],
                        "reference": str(reference)[:2000] if reference is not None else None,
                    }
                )
    summary = {
        "code_failure_counts": {task: dict(counter) for task, counter in counts.items()},
        "examples": examples if args.save_raw_samples else [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "code_failure_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Run the full causal analysis if the artifact files are present.
    samples_jsonl = output_dir / "diagnostics_eval_samples.jsonl"
    if samples_jsonl.exists():
        try:
            from lingua.code_diagnostics import run_code_causal_analysis
            run_id = getattr(args, "run_id", None) or "unknown"
            run_code_causal_analysis(
                output_dir=output_dir,
                run_id=str(run_id),
                code_tasks=list(args.code_tasks) if args.code_tasks else None,
            )
        except Exception as exc:
            logger.warning("code_diagnostics: causal analysis failed (non-fatal): %s", exc)

    return summary


def diagnostics_output_dir(base_dump_dir: Optional[str], args: DiagnosticsArgs) -> Path:
    return Path(args.output_dir) if args.output_dir else Path(base_dump_dir or ".") / "diagnostics"


def resolve_train_task_label(
    diagnostics_args: DiagnosticsArgs,
    data_sources: Optional[Dict[str, float]] = None,
) -> str:
    """Determine the task label for train-time ``start_batch`` calls.

    Priority order:

    1. ``diagnostics_args.train_task_label`` if explicitly set (non-empty).
    2. If ``diagnostics_args.infer_task_from_data_path`` is ``True`` *and*
       exactly one data source is configured in *data_sources*, the base
       directory name of that source path is used
       (e.g. ``"math_data"`` from ``"/datasets/math_data/"``).
    3. Falls back to ``"train"``.

    This function is side-effect-free; nothing is written to disk.
    """
    explicit = getattr(diagnostics_args, "train_task_label", None)
    if explicit:
        return explicit
    infer = getattr(diagnostics_args, "infer_task_from_data_path", False)
    if infer and data_sources and len(data_sources) == 1:
        source_path = next(iter(data_sources.keys()))
        label = Path(source_path).name
        if label:
            return label
    return "train"


def build_train_diagnostics_collector(
    model: torch.nn.Module,
    args: DiagnosticsArgs,
    *,
    output_dir: Optional[Path] = None,
    prefix: str = "diag/train",
    mode: str = "train",
) -> StemDiagnosticsCollector:
    """Build a :class:`StemDiagnosticsCollector` for training scripts.

    This is the shared factory used by all training entry-points
    (``stem_train``, ``stem_dag_train``, ``stem_distill_train``, and
    ``stem_projection_finetune``) so that collector setup is not duplicated.

    Parameters
    ----------
    model:
        The model being trained (before or after FSDP wrapping).
    args:
        ``DiagnosticsArgs`` from the training config.
    output_dir:
        Directory for diagnostics artifacts.  When ``None`` the collector
        creates a ``diagnostics/`` sub-directory relative to its own
        ``args.output_dir`` or defaults gracefully.
    prefix:
        Metric key prefix used in ``scalar_metrics()`` output (default
        ``"diag/train"``).
    mode:
        Collector mode; must be ``"train"`` for training scripts.  Do not
        pass ``"eval"`` here.

    Returns
    -------
    StemDiagnosticsCollector
        Ready to use as a context manager::

            collector = build_train_diagnostics_collector(model, args.diagnostics,
                                                          output_dir=output_dir)
            with collector:
                ...  # training loop
            if args.diagnostics.enabled and get_is_master():
                collector.write_artifacts({"global_step": step})
    """
    return StemDiagnosticsCollector(
        model,
        args,
        output_dir=output_dir,
        prefix=prefix,
        mode=mode,
    )
