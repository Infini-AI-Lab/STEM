"""Dashboard-style diagnostics summarizer for STEM finetune-ej diagnostics pipeline.

Reads artifacts produced by earlier pipeline rounds and writes three outputs:

* ``diagnostics/path_interference_dashboard.json`` — per-(task, task_group,
  layer, token_role) path-norm and intervention statistics.
* ``diagnostics/debuggability_report.json`` — per-task and global
  ``DebuggabilityRecord`` classifications with evidence strings.
* ``diagnostics/diagnostics_summary.md`` — human-readable Markdown summary.

All I/O is post-processing only (no model loading).  Missing artifact files are
tolerated and cause the corresponding sections to be marked with
``available=false`` and ``reason`` strings rather than crashing.

Typical standalone use::

    from lingua.diagnostic_dashboard import run_diagnostic_dashboard
    run_diagnostic_dashboard(diagnostics_dir="path/to/diagnostics")

Or via the CLI::

    python -m apps.main.diagnostic_dashboard --diagnostics-dir path/to/diagnostics
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lingua.diagnostic_records import (
    DebuggabilityRecord,
    sanitize_for_json,
    write_json_atomic,
    read_jsonl,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output file name constants
# ---------------------------------------------------------------------------

PATH_INTERFERENCE_DASHBOARD_JSON = "path_interference_dashboard.json"
DEBUGGABILITY_REPORT_JSON = "debuggability_report.json"
DIAGNOSTICS_SUMMARY_MD = "diagnostics_summary.md"

# ---------------------------------------------------------------------------
# Running-statistics accumulator (Welford / sum-of-squares)
# ---------------------------------------------------------------------------


class _StatAcc:
    """Accumulates mean and stdev without materialising all values."""

    __slots__ = ("n", "_total", "_sq_total", "_sat_near0", "_sat_near1")

    def __init__(self) -> None:
        self.n: int = 0
        self._total: float = 0.0
        self._sq_total: float = 0.0
        # For gate-saturation counts
        self._sat_near0: int = 0
        self._sat_near1: int = 0

    def update(self, v: Optional[float], *, is_gate: bool = False) -> None:
        if v is None or not math.isfinite(v):
            return
        self.n += 1
        self._total += v
        self._sq_total += v * v
        if is_gate:
            if v < 0.1:
                self._sat_near0 += 1
            elif v > 0.9:
                self._sat_near1 += 1

    def mean(self) -> Optional[float]:
        return self._total / self.n if self.n > 0 else None

    def stdev(self) -> Optional[float]:
        if self.n < 2:
            return None
        mean = self._total / self.n
        var = self._sq_total / self.n - mean * mean
        return math.sqrt(max(var, 0.0))

    def saturation_frac_near0(self) -> Optional[float]:
        return self._sat_near0 / self.n if self.n > 0 else None

    def saturation_frac_near1(self) -> Optional[float]:
        return self._sat_near1 / self.n if self.n > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean(),
            "stdev": self.stdev(),
            "count": self.n,
        }

    def to_gate_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean(),
            "stdev": self.stdev(),
            "count": self.n,
            "saturation_near0_frac": self.saturation_frac_near0(),
            "saturation_near1_frac": self.saturation_frac_near1(),
        }


# ---------------------------------------------------------------------------
# Artifact bundle — loads all inputs once, tolerates missing files
# ---------------------------------------------------------------------------

# Canonical artifact filenames (relative to diagnostics_dir)
_ARTIFACT_LAYER_PATH_METRICS = "layer_path_metrics.jsonl"
_ARTIFACT_INTERVENTIONS = "interventions_task_aligned.jsonl"
_ARTIFACT_TOKEN_EFFECTS = "token_effects.jsonl"
_ARTIFACT_CODE_FAILURES = "code_failures.jsonl"
_ARTIFACT_CODE_CAUSAL = "code_causal_failure_analysis.json"
_ARTIFACT_EVAL_SAMPLES = "diagnostics_eval_samples.jsonl"
_ARTIFACT_EVAL_GEOMETRY = "eval_geometry_summary.json"
_ARTIFACT_RICHER_GEOMETRY = "richer_geometry_summary.json"
_ARTIFACT_GEOMETRY_BY_TASK_LAYER_ROLE = "geometry_by_task_layer_role.json"
_ARTIFACT_GEOMETRY_BY_FREQUENCY_BUCKET = "geometry_by_frequency_bucket.json"
_ARTIFACT_BASELINE_COMPARISON_CKA = "baseline_comparison_cka.json"
_ARTIFACT_TOKEN_EFFECTS_BY_TASK = "token_effects_by_task.json"
_ARTIFACT_PATH_RELATIONS = "path_relations_by_task_layer.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("diagnostic_dashboard: failed to read %s: %s", path, exc)
        return None


@dataclass
class ArtifactBundle:
    """Container for all loaded diagnostic artifacts."""

    diagnostics_dir: Path
    run_id: str

    # JSONL rows (empty list when file absent/unreadable)
    layer_path_rows: List[Dict[str, Any]] = field(default_factory=list)
    intervention_rows: List[Dict[str, Any]] = field(default_factory=list)
    token_effect_rows: List[Dict[str, Any]] = field(default_factory=list)
    code_failure_rows: List[Dict[str, Any]] = field(default_factory=list)
    eval_sample_rows: List[Dict[str, Any]] = field(default_factory=list)

    # JSON blobs (None when file absent/unreadable)
    code_causal: Optional[Dict[str, Any]] = None
    eval_geometry: Optional[Dict[str, Any]] = None
    richer_geometry: Optional[Dict[str, Any]] = None
    geometry_by_task_layer_role: Optional[Dict[str, Any]] = None
    geometry_by_frequency_bucket: Optional[Dict[str, Any]] = None
    baseline_comparison_cka: Optional[Dict[str, Any]] = None
    token_effects_by_task: Optional[Dict[str, Any]] = None
    path_relations: Optional[Dict[str, Any]] = None

    # Availability flags
    available: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, diagnostics_dir: "os.PathLike[str] | str", run_id: str = "unknown") -> "ArtifactBundle":
        d = Path(diagnostics_dir)
        bundle = cls(diagnostics_dir=d, run_id=run_id)

        def _load_jsonl(name: str) -> List[Dict[str, Any]]:
            rows = read_jsonl(d / name)
            bundle.available[name] = len(rows) > 0
            if not rows:
                logger.info("diagnostic_dashboard: artifact absent or empty: %s", d / name)
            return rows

        def _load_json(name: str) -> Optional[Dict[str, Any]]:
            blob = _read_json(d / name)
            bundle.available[name] = blob is not None
            if blob is None:
                logger.info("diagnostic_dashboard: artifact absent or unreadable: %s", d / name)
            return blob

        bundle.layer_path_rows = _load_jsonl(_ARTIFACT_LAYER_PATH_METRICS)
        bundle.intervention_rows = _load_jsonl(_ARTIFACT_INTERVENTIONS)
        bundle.token_effect_rows = _load_jsonl(_ARTIFACT_TOKEN_EFFECTS)
        bundle.code_failure_rows = _load_jsonl(_ARTIFACT_CODE_FAILURES)
        bundle.eval_sample_rows = _load_jsonl(_ARTIFACT_EVAL_SAMPLES)
        bundle.code_causal = _load_json(_ARTIFACT_CODE_CAUSAL)
        bundle.eval_geometry = _load_json(_ARTIFACT_EVAL_GEOMETRY)
        bundle.richer_geometry = _load_json(_ARTIFACT_RICHER_GEOMETRY)
        bundle.geometry_by_task_layer_role = _load_json(_ARTIFACT_GEOMETRY_BY_TASK_LAYER_ROLE)
        bundle.geometry_by_frequency_bucket = _load_json(_ARTIFACT_GEOMETRY_BY_FREQUENCY_BUCKET)
        bundle.baseline_comparison_cka = _load_json(_ARTIFACT_BASELINE_COMPARISON_CKA)
        bundle.token_effects_by_task = _load_json(_ARTIFACT_TOKEN_EFFECTS_BY_TASK)
        bundle.path_relations = _load_json(_ARTIFACT_PATH_RELATIONS)

        logger.info(
            "diagnostic_dashboard: loaded bundle for run_id=%s from %s — available: %s",
            run_id, d,
            {k: v for k, v in bundle.available.items() if v},
        )
        return bundle

    def tasks(self) -> List[str]:
        """Return all unique task names seen across all artifacts."""
        tasks: set = set()
        for rows in (
            self.layer_path_rows,
            self.intervention_rows,
            self.token_effect_rows,
            self.code_failure_rows,
            self.eval_sample_rows,
        ):
            for row in rows:
                t = row.get("task")
                if t:
                    tasks.add(str(t))
        if self.code_causal:
            for t in (self.code_causal.get("failure_counts_by_task") or {}):
                tasks.add(str(t))
        if self.geometry_by_task_layer_role:
            for t in (self.geometry_by_task_layer_role.get("per_task_layer_role") or {}):
                tasks.add(str(t))
        if self.geometry_by_frequency_bucket:
            for t in (self.geometry_by_frequency_bucket.get("per_task_layer_bucket") or {}):
                tasks.add(str(t))
        return sorted(tasks)

    def task_group(self, task: str) -> str:
        from lingua.diagnostic_records import classify_task_group
        return classify_task_group(task)


# ---------------------------------------------------------------------------
# Path-interference dashboard
# ---------------------------------------------------------------------------

# Key type: (task, task_group, layer_idx_str, token_role)
_PathKey = Tuple[str, str, str, str]


def build_path_interference_dashboard(bundle: ArtifactBundle) -> Dict[str, Any]:
    """Aggregate layer-path metrics and intervention deltas per group.

    Groups are (task, task_group, layer_idx, token_role).

    Returns a JSON-safe dict with keys:
    - ``available``: bool
    - ``groups``: list of group dicts with aggregated stats
    - ``top_harmful_tokens``: global top harmful tokens across tasks
    - ``top_beneficial_tokens``: global top beneficial tokens across tasks
    - ``top_ineffective_tokens``: global top ineffective tokens across tasks
    - ``path_relation_counts``: global and per-task path_relation tallies
    """
    from lingua.diagnostic_records import classify_task_group

    # ---- Layer-path norm stats grouped by (task, task_group, layer, role) ----

    # acc[(task, task_group, layer_str, token_role)][field] -> _StatAcc
    acc: Dict[_PathKey, Dict[str, _StatAcc]] = defaultdict(
        lambda: defaultdict(_StatAcc)
    )

    for row in bundle.layer_path_rows:
        task = str(row.get("task") or "unknown")
        tg = str(row.get("task_group") or classify_task_group(task))
        layer = str(row.get("layer_idx") if row.get("layer_idx") is not None else "all")
        role = str(row.get("token_role") or "unknown")
        key: _PathKey = (task, tg, layer, role)
        for field_name in (
            "stem_norm", "up_norm", "combined_norm",
            "stem_up_cos", "ffn_out_norm",
        ):
            v = row.get(field_name)
            if v is not None:
                try:
                    acc[key][field_name].update(float(v))
                except (TypeError, ValueError):
                    pass
        gate = row.get("gate_alpha")
        if gate is not None:
            try:
                acc[key]["gate_alpha"].update(float(gate), is_gate=True)
            except (TypeError, ValueError):
                pass

    # ---- Intervention deltas grouped by (task, layer_str) ----

    # intv_acc[(task, layer_str)][itype] -> _StatAcc
    intv_acc: Dict[Tuple[str, str], Dict[str, _StatAcc]] = defaultdict(
        lambda: defaultdict(_StatAcc)
    )
    # path_relation counts by (task, layer_str)
    path_rel_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    global_path_rel: Counter = Counter()

    for row in bundle.intervention_rows:
        task = str(row.get("task") or "unknown")
        layer = str(row.get("layer_idx") if row.get("layer_idx") is not None else "all")
        itype = str(row.get("intervention_type") or "unknown")
        delta = row.get("delta_loss")
        if delta is not None:
            try:
                intv_acc[(task, layer)][itype].update(float(delta))
            except (TypeError, ValueError):
                pass
        pr = row.get("path_relation") or row.get("metadata", {}).get("path_relation")
        if pr:
            path_rel_counts[(task, layer)][str(pr)] += 1
            global_path_rel[str(pr)] += 1

    # ---- Token effects for top-token lists ----

    harmful_tokens: List[Tuple[float, str, str, str]] = []   # (score, token, role, task)
    beneficial_tokens: List[Tuple[float, str, str, str]] = []
    ineffective_tokens: List[Tuple[float, str, str, str]] = []

    for row in bundle.token_effect_rows:
        tok = str(row.get("token") or "")
        role = str(row.get("token_role") or "unknown")
        task = str(row.get("task") or "unknown")
        harm = row.get("harm_score")
        benefit = row.get("benefit_score")
        ineff = row.get("ineffective_score")
        if harm is not None:
            try:
                harmful_tokens.append((float(harm), tok, role, task))
            except (TypeError, ValueError):
                pass
        if benefit is not None:
            try:
                beneficial_tokens.append((float(benefit), tok, role, task))
            except (TypeError, ValueError):
                pass
        if ineff is not None:
            try:
                ineffective_tokens.append((float(ineff), tok, role, task))
            except (TypeError, ValueError):
                pass

    harmful_tokens.sort(reverse=True)
    beneficial_tokens.sort(reverse=True)
    ineffective_tokens.sort(reverse=True)

    def _tok_list(lst: List[Tuple[float, str, str, str]], n: int = 20) -> List[Dict[str, Any]]:
        return [
            {"score": s, "token": t, "token_role": r, "task": task}
            for s, t, r, task in lst[:n]
        ]

    # ---- Assemble groups ----

    groups: List[Dict[str, Any]] = []
    for (task, tg, layer, role), fields in sorted(acc.items()):
        group: Dict[str, Any] = {
            "task": task,
            "task_group": tg,
            "layer_idx": None if layer == "all" else (int(layer) if layer.isdigit() else layer),
            "token_role": role,
        }
        for fname in ("stem_norm", "up_norm", "combined_norm", "stem_up_cos", "ffn_out_norm"):
            stat = fields.get(fname)
            group[fname] = stat.to_dict() if stat else {"mean": None, "stdev": None, "count": 0}
        gate_stat = fields.get("gate_alpha")
        if gate_stat and gate_stat.n > 0:
            group["gate_alpha"] = gate_stat.to_gate_dict()
            group["gate_available"] = True
        else:
            group["gate_alpha"] = None
            group["gate_available"] = False

        # Attach intervention deltas for this (task, layer)
        intv_key = (task, layer)
        if intv_key in intv_acc:
            group["intervention_deltas"] = {
                itype: stat.to_dict()
                for itype, stat in intv_acc[intv_key].items()
            }
        else:
            group["intervention_deltas"] = {}

        group["path_relation_counts"] = dict(path_rel_counts.get(intv_key, Counter()))
        groups.append(group)

    return sanitize_for_json({
        "available": len(bundle.layer_path_rows) > 0 or len(bundle.intervention_rows) > 0,
        "groups": groups,
        "global_path_relation_counts": dict(global_path_rel),
        "top_harmful_tokens": _tok_list(harmful_tokens),
        "top_beneficial_tokens": _tok_list(beneficial_tokens),
        "top_ineffective_tokens": _tok_list(ineffective_tokens),
        "artifacts_used": {
            "layer_path_metrics": bundle.available.get(_ARTIFACT_LAYER_PATH_METRICS, False),
            "interventions": bundle.available.get(_ARTIFACT_INTERVENTIONS, False),
            "token_effects": bundle.available.get(_ARTIFACT_TOKEN_EFFECTS, False),
        },
    })


# ---------------------------------------------------------------------------
# Gate analysis
# ---------------------------------------------------------------------------

def build_gate_analysis(bundle: ArtifactBundle) -> Dict[str, Any]:
    """Summarise gate alpha values by task / layer / token_role.

    Returns a JSON-safe dict with:
    - ``gate_available``: bool
    - ``by_task_layer_role``: nested dict of gate stats
    - ``code_vs_nl_comparison``: whether code tokens differ from NL tokens
    - ``failure_category_gate_correlation``: gate means by failure category (if
      code_failures available)
    """
    code_roles = frozenset(
        {"python_keyword", "identifier", "numeral", "operator", "bracket", "punctuation"}
    )
    nl_roles = frozenset({"natural_language", "whitespace", "newline"})

    # by (task, layer_str, token_role) -> _StatAcc
    acc: Dict[Tuple[str, str, str], _StatAcc] = defaultdict(_StatAcc)
    global_code_gate = _StatAcc()
    global_nl_gate = _StatAcc()

    any_gate = False
    for row in bundle.layer_path_rows:
        gate = row.get("gate_alpha")
        if gate is None:
            continue
        try:
            gv = float(gate)
        except (TypeError, ValueError):
            continue
        any_gate = True
        task = str(row.get("task") or "unknown")
        layer = str(row.get("layer_idx") if row.get("layer_idx") is not None else "all")
        role = str(row.get("token_role") or "unknown")
        acc[(task, layer, role)].update(gv, is_gate=True)
        if role in code_roles:
            global_code_gate.update(gv, is_gate=True)
        elif role in nl_roles:
            global_nl_gate.update(gv, is_gate=True)

    if not any_gate:
        return sanitize_for_json({
            "gate_available": False,
            "reason": "no gate_alpha values in layer_path_metrics.jsonl",
            "by_task_layer_role": {},
            "code_vs_nl_comparison": None,
            "failure_category_gate_correlation": None,
        })

    # Nested dict: task -> layer -> role -> stats
    by_tlr: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (task, layer, role), stat in acc.items():
        by_tlr[task][layer][role] = stat.to_gate_dict()

    # Code vs NL comparison
    code_mean = global_code_gate.mean()
    nl_mean = global_nl_gate.mean()
    code_vs_nl: Dict[str, Any] = {
        "code_gate_mean": code_mean,
        "nl_gate_mean": nl_mean,
        "code_gate_count": global_code_gate.n,
        "nl_gate_count": global_nl_gate.n,
        "code_gate_saturation_near0": global_code_gate.saturation_frac_near0(),
        "code_gate_saturation_near1": global_code_gate.saturation_frac_near1(),
        "nl_gate_saturation_near0": global_nl_gate.saturation_frac_near0(),
        "nl_gate_saturation_near1": global_nl_gate.saturation_frac_near1(),
    }
    if code_mean is not None and nl_mean is not None:
        diff = abs(code_mean - nl_mean)
        code_vs_nl["mean_difference"] = diff
        code_vs_nl["systematic_difference"] = diff > 0.05
        code_vs_nl[
            "interpretation"
        ] = (
            "code tokens show systematically lower gate values (more STEM-path usage)"
            if code_mean < nl_mean - 0.05
            else (
                "code tokens show systematically higher gate values (more up-path usage)"
                if code_mean > nl_mean + 0.05
                else "code and NL tokens have similar gate distributions"
            )
        )
    else:
        code_vs_nl["systematic_difference"] = None
        code_vs_nl["interpretation"] = "insufficient data"

    # Failure category gate correlation
    fail_gate_corr: Optional[Dict[str, Any]] = None
    if bundle.code_failure_rows and bundle.layer_path_rows:
        # Map sample_id -> failure_category from code_failure_rows
        sample_cat: Dict[str, str] = {}
        for row in bundle.code_failure_rows:
            sid = row.get("sample_id")
            cat = row.get("failure_category")
            if sid and cat:
                sample_cat[str(sid)] = str(cat)
        # Match gate values to failure categories
        cat_gate: Dict[str, _StatAcc] = defaultdict(_StatAcc)
        for row in bundle.layer_path_rows:
            gate = row.get("gate_alpha")
            sid = row.get("sample_id")
            if gate is None or sid is None:
                continue
            cat = sample_cat.get(str(sid))
            if cat:
                try:
                    cat_gate[cat].update(float(gate), is_gate=True)
                except (TypeError, ValueError):
                    pass
        if cat_gate:
            fail_gate_corr = {
                cat: stat.to_gate_dict() for cat, stat in sorted(cat_gate.items())
            }

    return sanitize_for_json({
        "gate_available": True,
        "by_task_layer_role": dict(by_tlr),
        "code_vs_nl_comparison": code_vs_nl,
        "failure_category_gate_correlation": fail_gate_corr,
    })


# ---------------------------------------------------------------------------
# Richer geometry dashboard
# ---------------------------------------------------------------------------

def build_richer_geometry_analysis(bundle: ArtifactBundle) -> Dict[str, Any]:
    """Summarise richer geometry artifacts and warning categories."""

    rich = bundle.richer_geometry or {}
    by_role = bundle.geometry_by_task_layer_role or {}
    by_freq = bundle.geometry_by_frequency_bucket or {}
    baseline = bundle.baseline_comparison_cka or {}

    available = bool(rich or by_role or by_freq or baseline)
    if not available:
        return sanitize_for_json({
            "available": False,
            "reason": "no richer geometry artifacts found",
            "warnings": [],
            "warning_counts": {},
        })

    warnings: List[Dict[str, Any]] = []
    for warning in rich.get("warnings") or []:
        if isinstance(warning, dict):
            warnings.append(dict(warning))

    # Derive warnings as a fallback if the summary file was not produced but
    # the detailed role/frequency artifacts are present.
    if not warnings and by_role:
        per_task_layer_role = by_role.get("per_task_layer_role") or {}
        for task, by_layer in per_task_layer_role.items():
            for layer_idx, roles in (by_layer or {}).items():
                cell = (roles or {}).get("all") or {}
                er = cell.get("stem_effective_rank")
                ani = cell.get("stem_anisotropy")
                top_pc = cell.get("stem_top_pc_var_frac")
                if isinstance(er, (int, float)) and er < 4:
                    warnings.append({
                        "type": "low_effective_rank",
                        "severity": "warning",
                        "task": task,
                        "layer_idx": layer_idx,
                        "value": er,
                        "message": "STEM representation has low effective rank.",
                    })
                if isinstance(ani, (int, float)) and ani > 0.9:
                    warnings.append({
                        "type": "high_anisotropy",
                        "severity": "warning",
                        "task": task,
                        "layer_idx": layer_idx,
                        "value": ani,
                        "message": "STEM representation is highly anisotropic.",
                    })
                if isinstance(top_pc, (int, float)) and top_pc > 0.8:
                    warnings.append({
                        "type": "representation_collapse",
                        "severity": "warning",
                        "task": task,
                        "layer_idx": layer_idx,
                        "value": top_pc,
                        "message": "Top principal component explains most STEM variance.",
                    })

    warning_counts = Counter(str(w.get("type", "unknown")) for w in warnings)

    # Compact code-vs-non-code geometry comparison for dashboard readers.
    group_rank: Dict[str, List[float]] = defaultdict(list)
    group_aniso: Dict[str, List[float]] = defaultdict(list)
    per_task_layer_role = by_role.get("per_task_layer_role") or {}
    for task, by_layer in per_task_layer_role.items():
        tg = bundle.task_group(str(task))
        for _layer_idx, roles in (by_layer or {}).items():
            cell = (roles or {}).get("all") or {}
            er = cell.get("stem_effective_rank")
            ani = cell.get("stem_anisotropy")
            if isinstance(er, (int, float)):
                group_rank[tg].append(float(er))
            if isinstance(ani, (int, float)):
                group_aniso[tg].append(float(ani))

    def _avg(vals: List[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    code_vs_reasoning = {
        "code_effective_rank_mean": _avg(group_rank.get("code", [])),
        "non_code_effective_rank_mean": _avg([
            v for g, vals in group_rank.items() if g != "code" for v in vals
        ]),
        "code_anisotropy_mean": _avg(group_aniso.get("code", [])),
        "non_code_anisotropy_mean": _avg([
            v for g, vals in group_aniso.items() if g != "code" for v in vals
        ]),
    }
    if (
        code_vs_reasoning["code_effective_rank_mean"] is not None
        and code_vs_reasoning["non_code_effective_rank_mean"] is not None
    ):
        code_vs_reasoning["effective_rank_difference"] = (
            code_vs_reasoning["code_effective_rank_mean"]
            - code_vs_reasoning["non_code_effective_rank_mean"]
        )

    rare_token_differences: List[Dict[str, Any]] = []
    per_task_layer_bucket = by_freq.get("per_task_layer_bucket") or {}
    for task, by_layer in per_task_layer_bucket.items():
        for layer_idx, buckets in (by_layer or {}).items():
            rare = (buckets or {}).get("rare") or {}
            frequent = (buckets or {}).get("frequent") or {}
            for metric in ("stem_effective_rank", "stem_anisotropy", "stem_centroid_cos_mean"):
                rv = rare.get(metric)
                fv = frequent.get(metric)
                if isinstance(rv, (int, float)) and isinstance(fv, (int, float)):
                    rare_token_differences.append({
                        "task": task,
                        "layer_idx": layer_idx,
                        "metric": metric,
                        "rare_value": rv,
                        "frequent_value": fv,
                        "difference": float(rv) - float(fv),
                    })
    rare_token_differences.sort(key=lambda r: abs(float(r.get("difference", 0.0))), reverse=True)

    baseline_summary = {
        "available": bool(baseline.get("available")),
        "requested": bool(baseline.get("requested")),
        "reason": baseline.get("reason"),
        "status": baseline.get("status") or {},
    }
    if baseline.get("per_task_layer"):
        cka_vals: List[float] = []
        for by_layer in (baseline.get("per_task_layer") or {}).values():
            for by_path in (by_layer or {}).values():
                for metrics in (by_path or {}).values():
                    val = metrics.get("linear_cka") if isinstance(metrics, dict) else None
                    if isinstance(val, (int, float)):
                        cka_vals.append(float(val))
        baseline_summary["linear_cka_mean"] = _avg(cka_vals)
        baseline_summary["linear_cka_min"] = min(cka_vals) if cka_vals else None

    return sanitize_for_json({
        "available": True,
        "warnings": warnings,
        "warning_counts": dict(warning_counts),
        "code_vs_reasoning_geometry": code_vs_reasoning,
        "rare_token_geometry_differences": rare_token_differences[:20],
        "baseline_comparison": baseline_summary,
        "artifacts_used": {
            "richer_geometry_summary": bundle.available.get(_ARTIFACT_RICHER_GEOMETRY, False),
            "geometry_by_task_layer_role": bundle.available.get(_ARTIFACT_GEOMETRY_BY_TASK_LAYER_ROLE, False),
            "geometry_by_frequency_bucket": bundle.available.get(_ARTIFACT_GEOMETRY_BY_FREQUENCY_BUCKET, False),
            "baseline_comparison_cka": bundle.available.get(_ARTIFACT_BASELINE_COMPARISON_CKA, False),
        },
    })


# ---------------------------------------------------------------------------
# Debuggability classifier
# ---------------------------------------------------------------------------

# Thresholds
_LAYER_CONCENTRATION_THRESHOLD = 2       # harmful in ≤ this many layers → concentrated
_ROLE_CONCENTRATION_THRESHOLD = 3        # harmful in ≤ this many roles → concentrated
_BROAD_LAYER_FRAC = 0.4                  # harmful in ≥ this fraction of total layers → broad
_BROAD_ROLE_MIN = 5                      # harmful in ≥ this many roles → broad
_MEANINGFUL_DELTA = 0.01                 # |delta_loss| ≥ this to be "meaningful"
_GATE_DIFFERENCE_THRESHOLD = 0.05        # force_gate_0 vs force_gate_1 delta difference
_MIN_SAMPLES_FOR_CLASSIFICATION = 3      # minimum samples to make a non-inconclusive call


def _classify_task(
    task: str,
    bundle: ArtifactBundle,
    run_id: str,
    total_layers: Optional[int] = None,
) -> DebuggabilityRecord:
    """Apply rule-based debuggability classification for a single task."""

    evidence_debuggable: List[str] = []
    evidence_architectural: List[str] = []
    evidence_inconclusive: List[str] = []

    from lingua.diagnostic_records import classify_task_group
    tg = classify_task_group(task)

    # ---- Gather task-specific data ----

    # Samples for this task from eval samples
    task_samples = [r for r in bundle.eval_sample_rows if str(r.get("task") or "") == task]
    n_total = len(task_samples)
    n_correct = sum(1 for r in task_samples if r.get("correct") is True)
    n_incorrect = sum(1 for r in task_samples if r.get("correct") is False)

    # Intervention rows for this task
    task_intv = [r for r in bundle.intervention_rows if str(r.get("task") or "") == task]

    # Code failures for this task
    task_failures = [r for r in bundle.code_failure_rows if str(r.get("task") or "") == task]

    # Token effects for this task
    task_effects = [r for r in bundle.token_effect_rows if str(r.get("task") or "") == task]

    # Causal analysis per-task section
    causal_by_task: Optional[Dict[str, Any]] = None
    if bundle.code_causal:
        causal_by_task = (bundle.code_causal.get("failure_counts_by_task") or {}).get(task)

    # ---- Rule A: likely_debuggable ----

    # A1: Failures concentrated in 1-2 layers
    harmful_layers_for_task: List[int] = []
    for row in task_intv:
        li = row.get("layer_idx")
        delta = row.get("delta_loss")
        itype = str(row.get("intervention_type") or "")
        if (
            li is not None
            and delta is not None
            and "ablate_stem" in itype
            and float(delta) < -_MEANINGFUL_DELTA
        ):
            harmful_layers_for_task.append(int(li))
    unique_harmful_layers = sorted(set(harmful_layers_for_task))
    n_harmful_layers = len(unique_harmful_layers)
    if n_harmful_layers >= 1 and n_harmful_layers <= _LAYER_CONCENTRATION_THRESHOLD:
        evidence_debuggable.append(
            f"A1: STEM harmful concentrated in {n_harmful_layers} layer(s): {unique_harmful_layers}"
        )
    elif n_harmful_layers == 0 and task_intv:
        evidence_inconclusive.append("A1: no harmful layers found in intervention data")

    # A2: Forcing gate toward up path improves code loss
    gate_up_helpful = (bundle.code_causal or {}).get("gate_up_helpful_examples") or []
    gate_up_task = [ex for ex in gate_up_helpful if str(ex.get("task") or "") == task]
    if gate_up_task:
        evidence_debuggable.append(
            f"A2: {len(gate_up_task)} sample(s) improved when forcing gate toward up path"
        )

    # A3: Ablating STEM improves failed code samples
    stem_helpful = (bundle.code_causal or {}).get("stem_helpful_examples") or []
    stem_task = [ex for ex in stem_helpful if str(ex.get("task") or "") == task]
    if stem_task:
        evidence_debuggable.append(
            f"A3: {len(stem_task)} sample(s) where ablating STEM lowered loss"
        )

    # A4: Harmful tokens concentrated in small number of token roles
    code_roles_set = {"python_keyword", "identifier", "numeral", "operator", "bracket", "punctuation"}
    harmful_roles_for_task: set = set()
    for row in task_intv:
        roles = row.get("token_roles")
        per_tok = row.get("delta_per_token_nll")
        if not isinstance(roles, list) or not isinstance(per_tok, list):
            continue
        for role, td in zip(roles, per_tok):
            if isinstance(role, str) and isinstance(td, (int, float)):
                if float(td) < -_MEANINGFUL_DELTA:
                    harmful_roles_for_task.add(role)
    n_harmful_roles = len(harmful_roles_for_task)
    if n_harmful_roles >= 1 and n_harmful_roles <= _ROLE_CONCENTRATION_THRESHOLD:
        evidence_debuggable.append(
            f"A4: harmful token roles concentrated in {n_harmful_roles} role(s): "
            f"{sorted(harmful_roles_for_task)}"
        )
    elif n_harmful_roles == 0 and task_intv:
        # Try from causal analysis
        causal_roles_by_cat = (bundle.code_causal or {}).get("harmful_token_roles_by_category") or {}
        all_causal_roles: set = set()
        for roles_list in causal_roles_by_cat.values():
            all_causal_roles.update(roles_list)
        if 1 <= len(all_causal_roles) <= _ROLE_CONCENTRATION_THRESHOLD:
            evidence_debuggable.append(
                f"A4: harmful token roles (from causal analysis) concentrated: {sorted(all_causal_roles)}"
            )

    # A5: Path relation is destructive for specific layers/token roles
    destructive_count = sum(
        1 for r in task_intv
        if str(r.get("path_relation") or r.get("metadata", {}).get("path_relation") or "")
        == "destructive_stem"
    )
    total_with_relation = sum(
        1 for r in task_intv
        if (r.get("path_relation") or r.get("metadata", {}).get("path_relation"))
    )
    if total_with_relation >= _MIN_SAMPLES_FOR_CLASSIFICATION and destructive_count / total_with_relation > 0.2:
        evidence_debuggable.append(
            f"A5: {destructive_count}/{total_with_relation} ({100*destructive_count//total_with_relation}%) "
            "of path relations are destructive_stem"
        )

    # A6: STEM-only and STEM+up behave differently (gate forcing shows difference)
    gate0_deltas: List[float] = []
    gate1_deltas: List[float] = []
    for row in task_intv:
        itype = str(row.get("intervention_type") or "")
        delta = row.get("delta_loss")
        if delta is None:
            continue
        try:
            dv = float(delta)
        except (TypeError, ValueError):
            continue
        if itype == "force_gate_0":
            gate0_deltas.append(dv)
        elif itype == "force_gate_1":
            gate1_deltas.append(dv)
    if gate0_deltas and gate1_deltas:
        mean0 = sum(gate0_deltas) / len(gate0_deltas)
        mean1 = sum(gate1_deltas) / len(gate1_deltas)
        diff = abs(mean0 - mean1)
        if diff > _GATE_DIFFERENCE_THRESHOLD:
            evidence_debuggable.append(
                f"A6: gate forcing shows {diff:.3f} mean delta difference "
                f"(force_gate_0={mean0:.3f}, force_gate_1={mean1:.3f}) — "
                "STEM-only and STEM+up paths behave differently"
            )
    elif not gate0_deltas and not gate1_deltas and task_intv:
        evidence_inconclusive.append("A6: no gate-forcing interventions found for this task")

    # ---- Rule B: possibly_architectural ----

    # B1: Harmful STEM effect is broad across layers
    est_total_layers = total_layers or 32
    if n_harmful_layers > est_total_layers * _BROAD_LAYER_FRAC:
        evidence_architectural.append(
            f"B1: harmful STEM effect spans {n_harmful_layers}/{est_total_layers} layers "
            f"({100*n_harmful_layers//est_total_layers}%) — broad layer coverage"
        )

    # B2: Harmful effect is broad across token roles
    if n_harmful_roles >= _BROAD_ROLE_MIN:
        evidence_architectural.append(
            f"B2: harmful STEM effect spans {n_harmful_roles} token roles — broad role coverage"
        )

    # B3: Both STEM-only and STEM+up fail similarly
    if gate0_deltas and gate1_deltas:
        mean0 = sum(gate0_deltas) / len(gate0_deltas)
        mean1 = sum(gate1_deltas) / len(gate1_deltas)
        diff = abs(mean0 - mean1)
        if diff < _GATE_DIFFERENCE_THRESHOLD:
            evidence_architectural.append(
                f"B3: gate forcing shows minimal difference ({diff:.3f}) — "
                "STEM-only and STEM+up paths fail similarly"
            )

    # B4: No intervention improves loss meaningfully
    all_deltas: List[float] = []
    for row in task_intv:
        delta = row.get("delta_loss")
        if delta is not None:
            try:
                all_deltas.append(float(delta))
            except (TypeError, ValueError):
                pass
    if all_deltas:
        max_improvement = min(all_deltas)  # most negative = most improvement
        if max_improvement > -_MEANINGFUL_DELTA:
            evidence_architectural.append(
                f"B4: no intervention produced meaningful loss improvement "
                f"(best delta={max_improvement:.4f})"
            )
        elif max_improvement < -0.1:
            evidence_debuggable.append(
                f"B4_inv: some interventions show large improvement "
                f"(best delta={max_improvement:.4f}) — issue may be fixable"
            )

    # B5: Rare/identifier/code tokens consistently harmed across tasks
    code_harm_roles = code_roles_set & harmful_roles_for_task
    if len(code_harm_roles) >= 3:
        evidence_architectural.append(
            f"B5: code-specific token roles consistently harmed: {sorted(code_harm_roles)}"
        )
    else:
        # Check across all token_effect_rows for this task
        harmful_code_roles_from_effects = set()
        for row in task_effects:
            role = str(row.get("token_role") or "")
            hs = row.get("harm_score")
            if role in code_roles_set and hs is not None:
                try:
                    if float(hs) > _MEANINGFUL_DELTA:
                        harmful_code_roles_from_effects.add(role)
                except (TypeError, ValueError):
                    pass
        if len(harmful_code_roles_from_effects) >= 3:
            evidence_architectural.append(
                f"B5: code-specific token roles with positive harm_score: "
                f"{sorted(harmful_code_roles_from_effects)}"
            )

    # B6: Geometry shows broad collapse or severe anisotropy across layers
    if bundle.eval_geometry:
        per_task_layer = bundle.eval_geometry.get("per_task_layer") or {}
        task_geo = per_task_layer.get(task) or {}
        low_rank_layers: List[str] = []
        high_anisotropy_layers: List[str] = []
        for layer_key, cell in task_geo.items():
            if not isinstance(cell, dict):
                continue
            er = cell.get("stem_effective_rank")
            ani = cell.get("stem_anisotropy")
            if isinstance(er, (int, float)) and er < 4:
                low_rank_layers.append(layer_key)
            if isinstance(ani, (int, float)) and ani > 0.9:
                high_anisotropy_layers.append(layer_key)
        if low_rank_layers:
            evidence_architectural.append(
                f"B6: geometry collapse (effective_rank<4) in layers: {low_rank_layers[:5]}"
            )
        if high_anisotropy_layers:
            evidence_architectural.append(
                f"B6: high anisotropy (>0.9) in layers: {high_anisotropy_layers[:5]}"
            )
    if bundle.richer_geometry:
        task_warnings = [
            w for w in (bundle.richer_geometry.get("warnings") or [])
            if isinstance(w, dict) and str(w.get("task") or "") == task
        ]
        collapse_like = [
            w for w in task_warnings
            if str(w.get("type") or "") in {
                "representation_collapse",
                "low_effective_rank",
                "high_anisotropy",
            }
        ]
        if collapse_like:
            evidence_architectural.append(
                f"B6: richer geometry warning(s): "
                f"{[w.get('type') for w in collapse_like[:5]]}"
            )

    # ---- Insufficient data check ----

    has_intv_data = len(task_intv) >= _MIN_SAMPLES_FOR_CLASSIFICATION
    has_sample_data = n_total >= _MIN_SAMPLES_FOR_CLASSIFICATION
    has_geometry_data = bool(bundle.eval_geometry or bundle.richer_geometry)

    if not has_intv_data and not has_sample_data and not has_geometry_data:
        return DebuggabilityRecord(
            run_id=run_id,
            task=task,
            classification="inconclusive",
            evidence=["insufficient data: no intervention rows and no eval sample rows for this task"],
            confidence=0.0,
        )

    # ---- Classify ----

    n_debug = len(evidence_debuggable)
    n_arch = len(evidence_architectural)

    all_evidence = evidence_debuggable + evidence_architectural + evidence_inconclusive

    if n_debug >= 2 and n_debug > n_arch:
        classification = "likely_debuggable"
        confidence = min(0.5 + 0.1 * (n_debug - 2), 0.95)
    elif n_arch >= 2 and n_arch > n_debug:
        classification = "possibly_architectural"
        confidence = min(0.5 + 0.1 * (n_arch - 2), 0.95)
    elif n_debug >= 2 and n_arch >= 2:
        # Tied — mixed evidence
        classification = "inconclusive"
        confidence = 0.3
        all_evidence.insert(0, f"mixed evidence: {n_debug} debuggable signals vs {n_arch} architectural signals")
    else:
        classification = "inconclusive"
        confidence = 0.2
        if not all_evidence:
            all_evidence.append(
                f"insufficient evidence: {n_debug} debuggable signal(s), {n_arch} architectural signal(s)"
            )

    return DebuggabilityRecord(
        run_id=run_id,
        task=task,
        classification=classification,
        evidence=all_evidence,
        confidence=round(confidence, 3),
        metadata={
            "task_group": tg,
            "n_total_samples": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_intervention_rows": len(task_intv),
            "n_debuggable_signals": n_debug,
            "n_architectural_signals": n_arch,
            "harmful_layers": unique_harmful_layers,
            "harmful_roles": sorted(harmful_roles_for_task),
        },
    )


def build_debuggability_report(
    bundle: ArtifactBundle,
    *,
    total_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify each task and build a global aggregate.

    Returns a JSON-safe dict with:
    - ``per_task``: list of DebuggabilityRecord dicts
    - ``global_classification``: the most common verdict
    - ``global_evidence``: merged evidence strings
    - ``summary_counts``: counts by classification
    - ``tasks``: list of all tasks considered
    """
    tasks = bundle.tasks()
    records: List[DebuggabilityRecord] = []

    # Estimate total layers from layer_path rows if not given
    if total_layers is None and bundle.layer_path_rows:
        layers_seen = {r.get("layer_idx") for r in bundle.layer_path_rows if r.get("layer_idx") is not None}
        if layers_seen:
            total_layers = max(int(l) for l in layers_seen) + 1

    for task in tasks:
        rec = _classify_task(task, bundle, run_id=bundle.run_id, total_layers=total_layers)
        records.append(rec)
        logger.info(
            "diagnostic_dashboard: %s → %s (confidence=%.2f) evidence=%s",
            task, rec.classification, rec.confidence, rec.evidence,
        )

    # Global aggregate
    classification_counts: Counter = Counter(r.classification for r in records)
    if not records:
        global_cls = "inconclusive"
        global_evidence: List[str] = ["no tasks found in any artifact"]
    else:
        global_cls = classification_counts.most_common(1)[0][0]
        global_evidence = []
        for cls in ("likely_debuggable", "possibly_architectural", "inconclusive"):
            cls_tasks = [r.task for r in records if r.classification == cls]
            if cls_tasks:
                global_evidence.append(f"{cls}: {cls_tasks}")

    return sanitize_for_json({
        "run_id": bundle.run_id,
        "global_classification": global_cls,
        "global_evidence": global_evidence,
        "summary_counts": dict(classification_counts),
        "tasks": tasks,
        "per_task": [asdict(r) for r in records],
    })


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    bundle: ArtifactBundle,
    dashboard: Dict[str, Any],
    debug_report: Dict[str, Any],
    gate_analysis: Dict[str, Any],
    out_path: Path,
) -> None:
    """Write a human-readable Markdown diagnostics summary."""

    lines: List[str] = []

    def h(level: int, text: str) -> None:
        lines.append("#" * level + " " + text)
        lines.append("")

    def p(text: str) -> None:
        lines.append(text)
        lines.append("")

    def ul(items: List[str]) -> None:
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    # =========================================================================
    h(1, "STEM Diagnostics Summary")
    p(f"**Run ID:** `{bundle.run_id}`  \n**Diagnostics directory:** `{bundle.diagnostics_dir}`")

    # ---- Executive summary ----
    h(2, "Executive Summary")
    global_cls = debug_report.get("global_classification", "inconclusive")
    counts = debug_report.get("summary_counts") or {}
    n_debug = counts.get("likely_debuggable", 0)
    n_arch = counts.get("possibly_architectural", 0)
    n_incon = counts.get("inconclusive", 0)
    total_tasks = len(debug_report.get("tasks") or [])

    verdict_emoji = {
        "likely_debuggable": "[DEBUGGABLE]",
        "possibly_architectural": "[ARCHITECTURAL]",
        "inconclusive": "[INCONCLUSIVE]",
    }
    p(
        f"**Overall verdict:** {verdict_emoji.get(global_cls, global_cls)}  \n"
        f"Across {total_tasks} task(s): "
        f"{n_debug} likely-debuggable, {n_arch} possibly-architectural, "
        f"{n_incon} inconclusive."
    )

    artifacts_available = [k for k, v in bundle.available.items() if v]
    artifacts_missing = [k for k, v in bundle.available.items() if not v]
    if artifacts_available:
        p(f"**Artifacts available:** {', '.join(artifacts_available)}")
    if artifacts_missing:
        p(f"**Artifacts missing / empty:** {', '.join(artifacts_missing)}")

    # ---- Tasks most helped / hurt by STEM ----
    h(2, "Tasks Most Helped / Hurt by STEM")

    # Derive from token_effects_by_task if available
    tbt = bundle.token_effects_by_task
    if tbt:
        global_rankings = tbt.get("global") or {}
        top_benefit_layers = global_rankings.get("top_stem_benefit_layers") or []
        top_harm_layers = global_rankings.get("top_stem_harm_layers") or []
        per_task_ranks = tbt.get("per_task") or {}
        if top_benefit_layers:
            p(f"**Layers where STEM most benefits:** {top_benefit_layers[:5]}")
        if top_harm_layers:
            p(f"**Layers where STEM most harms:** {top_harm_layers[:5]}")
        if per_task_ranks:
            rows_tbt = []
            for task, stats in sorted(per_task_ranks.items()):
                if not isinstance(stats, dict):
                    continue
                rows_tbt.append([
                    task,
                    _fmt(stats.get("mean_stem_ablation_delta_loss")),
                    stats.get("top_harm_role", "N/A"),
                    stats.get("top_benefit_role", "N/A"),
                ])
            if rows_tbt:
                lines.append(_md_table(
                    ["Task", "Mean Ablate-STEM Δloss", "Top Harm Role", "Top Benefit Role"],
                    rows_tbt,
                ))
                lines.append("")
    else:
        # Derive from intervention rows
        task_delta: Dict[str, List[float]] = defaultdict(list)
        for row in bundle.intervention_rows:
            itype = str(row.get("intervention_type") or "")
            if "ablate_stem" not in itype:
                continue
            task = str(row.get("task") or "unknown")
            delta = row.get("delta_loss")
            if delta is not None:
                try:
                    task_delta[task].append(float(delta))
                except (TypeError, ValueError):
                    pass
        if task_delta:
            task_means = sorted(
                ((t, sum(d) / len(d)) for t, d in task_delta.items()),
                key=lambda x: x[1],
            )
            rows_td = [[t, _fmt(m)] for t, m in task_means]
            lines.append(_md_table(["Task", "Mean Ablate-STEM Δloss (negative = STEM harmful)"], rows_td))
            lines.append("")
        else:
            p("_No intervention data available._")

    # ---- MBPP / HumanEval diagnosis ----
    h(2, "MBPP / HumanEval Diagnosis")
    cc = bundle.code_causal
    if cc and not cc.get("skipped"):
        n_total_cc = cc.get("total_samples", 0)
        cat_counts = cc.get("failure_counts_by_category") or {}
        n_pass = cat_counts.get("pass", 0)
        pass_rate = n_pass / n_total_cc if n_total_cc > 0 else None
        p(
            f"Total code samples analysed: **{n_total_cc}**.  "
            f"Pass rate: **{_fmt(pass_rate, 2)}**.  "
            f"Harness data available: {cc.get('harness_data_available', False)}."
        )
        # Failure category breakdown
        rows_cc = sorted(
            [(cat, cnt) for cat, cnt in cat_counts.items() if cat != "pass"],
            key=lambda x: -x[1],
        )
        if rows_cc:
            lines.append(_md_table(
                ["Failure Category", "Count"],
                rows_cc,
            ))
            lines.append("")

        n_stem_help = len(cc.get("stem_helpful_examples") or [])
        n_gate_help = len(cc.get("gate_up_helpful_examples") or [])
        ul([
            f"Samples where ablating STEM lowered loss (STEM was harmful): **{n_stem_help}**",
            f"Samples where forcing gate→up path was better: **{n_gate_help}**",
        ])

        avg_delta = cc.get("avg_delta_loss_by_category") or {}
        if avg_delta:
            rows_ad = []
            for cat, itypes in sorted(avg_delta.items()):
                if not isinstance(itypes, dict):
                    continue
                ablate = itypes.get("ablate_stem") or itypes.get("ablate_layer_stem")
                rows_ad.append([
                    cat,
                    _fmt(ablate),
                ])
            if rows_ad:
                lines.append(_md_table(
                    ["Failure Category", "Mean Ablate-STEM Δloss"],
                    rows_ad,
                ))
                lines.append("")
    elif cc and cc.get("skipped"):
        p(f"_Causal analysis skipped: {cc.get('reason', 'unknown reason')}_")
    else:
        p("_No `code_causal_failure_analysis.json` available._")

    # ---- Path interference summary ----
    h(2, "Path Interference Summary")
    global_pr = dashboard.get("global_path_relation_counts") or {}
    if global_pr:
        total_pr = sum(global_pr.values())
        rows_pr = sorted(
            [(rel, cnt, f"{100*cnt//total_pr}%") for rel, cnt in global_pr.items()],
            key=lambda x: -x[1],
        )
        lines.append(_md_table(["Path Relation", "Count", "Fraction"], rows_pr))
        lines.append("")
    else:
        p("_No path relation data available._")

    # Layer/role table (summarised — show top 10 harmful groups)
    groups = dashboard.get("groups") or []
    harmful_groups = []
    for g in groups:
        intv_deltas = g.get("intervention_deltas") or {}
        ablate_stem = intv_deltas.get("ablate_stem") or intv_deltas.get("ablate_layer_stem")
        if ablate_stem:
            mean_delta = ablate_stem.get("mean")
            if isinstance(mean_delta, float) and mean_delta < -_MEANINGFUL_DELTA:
                harmful_groups.append((mean_delta, g))
    harmful_groups.sort(key=lambda x: x[0])
    if harmful_groups:
        rows_hg = []
        for mean_delta, g in harmful_groups[:10]:
            rows_hg.append([
                g.get("task", "?"),
                g.get("layer_idx", "?"),
                g.get("token_role", "?"),
                _fmt(mean_delta),
                _fmt((g.get("stem_norm") or {}).get("mean")),
                _fmt((g.get("stem_up_cos") or {}).get("mean")),
            ])
        lines.append(_md_table(
            ["Task", "Layer", "Token Role", "Ablate-STEM Δloss", "STEM Norm", "STEM-Up Cos"],
            rows_hg,
        ))
        lines.append("")
    else:
        p("_No harmful path groups found._")

    # ---- Token-effectiveness summary ----
    h(2, "Token-Effectiveness Summary")
    top_harm = dashboard.get("top_harmful_tokens") or []
    top_benefit = dashboard.get("top_beneficial_tokens") or []
    if top_harm:
        h(3, "Top Harmful Tokens (STEM hurts these tokens)")
        rows_th = [[e["token"], e["token_role"], e["task"], _fmt(e["score"])] for e in top_harm[:10]]
        lines.append(_md_table(["Token", "Role", "Task", "Harm Score"], rows_th))
        lines.append("")
    if top_benefit:
        h(3, "Top Beneficial Tokens (STEM helps these tokens)")
        rows_tb = [[e["token"], e["token_role"], e["task"], _fmt(e["score"])] for e in top_benefit[:10]]
        lines.append(_md_table(["Token", "Role", "Task", "Benefit Score"], rows_tb))
        lines.append("")

    # ---- Gate behaviour summary ----
    h(2, "Gate Behaviour Summary")
    if not gate_analysis.get("gate_available"):
        p(f"_Gate data not available: {gate_analysis.get('reason', 'no gate_alpha values found')}_")
    else:
        cvn = gate_analysis.get("code_vs_nl_comparison") or {}
        p(
            f"**Code token gate mean:** {_fmt(cvn.get('code_gate_mean'))}  \n"
            f"**NL token gate mean:** {_fmt(cvn.get('nl_gate_mean'))}  \n"
            f"**Systematic difference:** {cvn.get('systematic_difference')}  \n"
            f"**Interpretation:** {cvn.get('interpretation', 'N/A')}"
        )
        sat_code_0 = cvn.get("code_gate_saturation_near0")
        sat_code_1 = cvn.get("code_gate_saturation_near1")
        sat_nl_0 = cvn.get("nl_gate_saturation_near0")
        sat_nl_1 = cvn.get("nl_gate_saturation_near1")
        if any(v is not None for v in (sat_code_0, sat_code_1, sat_nl_0, sat_nl_1)):
            rows_gs = [
                ["code", _fmt(sat_code_0, 3), _fmt(sat_code_1, 3)],
                ["natural_language", _fmt(sat_nl_0, 3), _fmt(sat_nl_1, 3)],
            ]
            lines.append(_md_table(["Token Type", "Saturation near 0", "Saturation near 1"], rows_gs))
            lines.append("")

        fcg = gate_analysis.get("failure_category_gate_correlation")
        if fcg:
            h(3, "Gate Values by Failure Category")
            rows_fcg = []
            for cat, stats in sorted(fcg.items()):
                if not isinstance(stats, dict):
                    continue
                rows_fcg.append([
                    cat,
                    _fmt(stats.get("mean")),
                    _fmt(stats.get("stdev")),
                    stats.get("count", 0),
                    _fmt(stats.get("saturation_near0_frac"), 3),
                    _fmt(stats.get("saturation_near1_frac"), 3),
                ])
            if rows_fcg:
                lines.append(_md_table(
                    ["Failure Category", "Gate Mean", "Gate Std", "Count", "Sat.~0", "Sat.~1"],
                    rows_fcg,
                ))
                lines.append("")

    # ---- Richer geometry summary ----
    h(2, "Richer Geometry Summary")
    geom = dashboard.get("richer_geometry_analysis") or {}
    if not geom.get("available"):
        p(f"_Richer geometry not available: {geom.get('reason', 'no richer geometry artifacts found')}_")
    else:
        counts = geom.get("warning_counts") or {}
        if counts:
            p(
                "**Geometry warnings:** "
                + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )
        warnings = geom.get("warnings") or []
        if warnings:
            rows_gw = []
            for w in warnings[:10]:
                rows_gw.append([
                    w.get("type", "?"),
                    w.get("task", "global"),
                    w.get("layer_idx", "?"),
                    w.get("metric", "?"),
                    _fmt(w.get("value", w.get("difference"))),
                ])
            lines.append(_md_table(
                ["Warning", "Task", "Layer", "Metric", "Value"],
                rows_gw,
            ))
            lines.append("")
        cvr = geom.get("code_vs_reasoning_geometry") or {}
        if any(v is not None for v in cvr.values()):
            p(
                f"**Code effective-rank mean:** {_fmt(cvr.get('code_effective_rank_mean'))}  \n"
                f"**Non-code effective-rank mean:** {_fmt(cvr.get('non_code_effective_rank_mean'))}  \n"
                f"**Code anisotropy mean:** {_fmt(cvr.get('code_anisotropy_mean'))}  \n"
                f"**Non-code anisotropy mean:** {_fmt(cvr.get('non_code_anisotropy_mean'))}"
            )
        rare_diffs = geom.get("rare_token_geometry_differences") or []
        if rare_diffs:
            rows_rd = []
            for row in rare_diffs[:8]:
                rows_rd.append([
                    row.get("task", "?"),
                    row.get("layer_idx", "?"),
                    row.get("metric", "?"),
                    _fmt(row.get("rare_value")),
                    _fmt(row.get("frequent_value")),
                    _fmt(row.get("difference")),
                ])
            lines.append(_md_table(
                ["Task", "Layer", "Metric", "Rare", "Frequent", "Δ"],
                rows_rd,
            ))
            lines.append("")
        bcka = geom.get("baseline_comparison") or {}
        if bcka.get("requested"):
            p(
                f"**Baseline CKA available:** {bcka.get('available')}  \n"
                f"**Mean linear CKA:** {_fmt(bcka.get('linear_cka_mean'))}  \n"
                f"**Min linear CKA:** {_fmt(bcka.get('linear_cka_min'))}  \n"
                f"**Status:** {bcka.get('reason') or (bcka.get('status') or {}).get('reason') or 'ok'}"
            )

    # ---- Debuggable vs architectural verdict ----
    h(2, "Debuggable vs Architectural Verdict")
    per_task = debug_report.get("per_task") or []
    if per_task:
        rows_verdict = []
        for rec in per_task:
            rows_verdict.append([
                rec.get("task", "?"),
                rec.get("classification", "?"),
                _fmt(rec.get("confidence")),
                str((rec.get("evidence") or [""])[:2])[1:-1],  # first 2 evidence items
            ])
        lines.append(_md_table(
            ["Task", "Classification", "Confidence", "Top Evidence (first 2)"],
            rows_verdict,
        ))
        lines.append("")

    # Full evidence per task
    for rec in per_task:
        task = rec.get("task", "?")
        cls = rec.get("classification", "?")
        conf = rec.get("confidence", 0.0)
        ev = rec.get("evidence") or []
        h(3, f"Task: `{task}` → {cls} (confidence {_fmt(conf, 2)})")
        ul(ev) if ev else p("_No evidence collected._")
        meta = rec.get("metadata") or {}
        if meta.get("harmful_layers"):
            p(f"Harmful layers: `{meta['harmful_layers']}`")
        if meta.get("harmful_roles"):
            p(f"Harmful roles: `{meta['harmful_roles']}`")

    # ---- Recommended next experiments ----
    h(2, "Recommended Next Experiments")

    recommended: List[str] = []

    if global_cls == "likely_debuggable":
        recommended += [
            "Identify the 1-2 harmful layers and try layer-selective STEM masking or re-initialisation.",
            "Investigate the specific token roles where STEM is harmful — consider role-conditioned gate training.",
            "Run extended gate-forcing experiments (force_gate_1) on failed code samples to quantify the routing fix gap.",
            "Fine-tune gate or routing head specifically on code task failures.",
        ]
    elif global_cls == "possibly_architectural":
        recommended += [
            "Investigate whether the base model capacity for code generation is sufficient before STEM.",
            "Compare STEM-ablated model performance vs baseline on code tasks to isolate the capability ceiling.",
            "Consider architectural changes: wider hidden dim, deeper STEM, or task-specific STEM heads.",
            "Profile whether STEM is simply too small relative to the up-path to steer code generation.",
        ]
    else:
        recommended += [
            "Collect more complete intervention and layer-path data (enable collect_eval_interventions=true).",
            "Run with collect_geometry=true and eval_geometry_save_npz=true to get representation geometry.",
            "Add token_ids capture (capture_token_ids=true) for richer token-level analysis.",
        ]

    # Task-specific
    for rec in per_task:
        cls = rec.get("classification")
        task = rec.get("task", "?")
        meta = rec.get("metadata") or {}
        if cls == "likely_debuggable" and meta.get("harmful_layers"):
            recommended.append(
                f"[{task}] Focus debugging on layers {meta['harmful_layers']} "
                "(identified as STEM-harmful)."
            )
        if cls == "possibly_architectural" and not meta.get("harmful_layers"):
            recommended.append(
                f"[{task}] No focal harmful layers — consider architecture or capacity changes."
            )

    ul(recommended) if recommended else p("_No specific recommendations available._")

    # =========================================================================
    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("diagnostic_dashboard: wrote markdown summary → %s", out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostic_dashboard(
    diagnostics_dir: "os.PathLike[str] | str",
    *,
    run_id: str = "unknown",
    output_dir: Optional["os.PathLike[str] | str"] = None,
    total_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Load artifacts and write all three dashboard outputs.

    Parameters
    ----------
    diagnostics_dir:
        Directory containing the input JSONL / JSON artifacts.
    run_id:
        Identifier propagated into :class:`DebuggabilityRecord` rows.
    output_dir:
        Directory to write output files.  Defaults to *diagnostics_dir*.
    total_layers:
        Hint for the total number of transformer layers (used in B1 rule).
        If ``None`` it is estimated from the layer_path_metrics data.

    Returns
    -------
    Dict with keys ``path_interference_dashboard``, ``debuggability_report``,
    ``markdown_path``, and ``artifacts_available``.
    """
    d_in = Path(diagnostics_dir)
    d_out = Path(output_dir) if output_dir else d_in
    d_out.mkdir(parents=True, exist_ok=True)

    bundle = ArtifactBundle.load(d_in, run_id=run_id)

    dashboard = build_path_interference_dashboard(bundle)
    gate_analysis = build_gate_analysis(bundle)
    richer_geometry_analysis = build_richer_geometry_analysis(bundle)
    debug_report = build_debuggability_report(bundle, total_layers=total_layers)

    # Merge gate analysis into dashboard for convenience
    dashboard["gate_analysis"] = gate_analysis
    dashboard["richer_geometry_analysis"] = richer_geometry_analysis

    # Write JSON outputs
    pi_path = d_out / PATH_INTERFERENCE_DASHBOARD_JSON
    dr_path = d_out / DEBUGGABILITY_REPORT_JSON
    md_path = d_out / DIAGNOSTICS_SUMMARY_MD

    write_json_atomic(pi_path, dashboard, rank0_only=False)
    logger.info("diagnostic_dashboard: wrote path interference dashboard → %s", pi_path)

    write_json_atomic(dr_path, debug_report, rank0_only=False)
    logger.info("diagnostic_dashboard: wrote debuggability report → %s", dr_path)

    write_markdown_summary(bundle, dashboard, debug_report, gate_analysis, md_path)

    return {
        "path_interference_dashboard": dashboard,
        "debuggability_report": debug_report,
        "richer_geometry_analysis": richer_geometry_analysis,
        "markdown_path": str(md_path),
        "artifacts_available": bundle.available,
    }
