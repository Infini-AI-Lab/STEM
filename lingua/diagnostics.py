"""Optional diagnostics for STEM, STEM+projection, and DAG-STEM models.

The helpers in this module are intentionally passive: nothing is registered or
written unless ``DiagnosticsArgs.enabled`` is true.  Collection is hook based and
keeps only streaming summaries or bounded samples in memory.
"""

from __future__ import annotations

import ast
import json
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

    def update_loss_delta(self, token_ids: torch.Tensor, name: str, deltas: torch.Tensor) -> None:
        ids = token_ids.detach().reshape(-1).cpu().tolist()
        flat = deltas.detach().float().reshape(-1).cpu().tolist()
        for tok, val in zip(ids, flat):
            self.loss_deltas[int(tok)][name].update(val)

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

    def rankings(self) -> Dict[str, List[Dict[str, Any]]]:
        token_scores = []
        for token_id, stats in self.loss_deltas.items():
            stem = stats.get("stem_ablation_loss_delta")
            benefit = stem.mean if stem and stem.count else 0.0
            token_scores.append(
                {"token_id": token_id, "frequency": self.freq[token_id], "score": benefit}
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
    gate_value: Optional[float] = None,
    replacement: Optional[torch.Tensor] = None,
):
    """Temporarily patch STEM FFNs for causal path analysis."""

    patches: List[Tuple[torch.nn.Module, Any]] = []
    for idx, layer in _iter_stem_layers(model, [layer_idx] if layer_idx is not None else None):
        ff = layer.feed_forward
        old_forward = ff.forward

        def make_forward(module, original_forward):
            def forward(self, x, y=None):
                if kind == "none":
                    return original_forward(x, y)
                x1 = self.w1(x.view_as(x))
                stem = y
                dense = self.w3(x.view_as(x)) if hasattr(self, "w3") else None
                if kind == "ablate_stem" and stem is not None:
                    stem = torch.zeros_like(stem)
                elif kind == "replace_stem_mean" and stem is not None:
                    rep = replacement.to(device=stem.device, dtype=stem.dtype) if replacement is not None else stem.mean(dim=(0, 1), keepdim=True)
                    stem = rep.expand_as(stem)
                if kind == "ablate_dense" and dense is not None:
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
        patches.append((ff, old_forward))
    try:
        yield
    finally:
        for module, old_forward in patches:
            module.forward = old_forward


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
    text = generated or ""
    category = "unknown"
    detail = ""
    try:
        ast.parse(text)
    except SyntaxError as exc:
        msg = str(exc).lower()
        detail = str(exc)
        if "indent" in msg:
            category = "indentation_formatting_error"
        elif "eol while scanning string literal" in msg or "unterminated string" in msg:
            category = "unmatched_bracket_or_quote"
        else:
            category = "syntax_parse_error"
    if category == "unknown":
        if _has_unbalanced_delimiters(text):
            category = "unmatched_bracket_or_quote"
        elif re.search(r"NameError|undefined|not defined", text):
            category = "identifier_mismatch_missing_symbol"
        elif re.search(r"import\s+\*|from\s+\w+\s+import", text) and reference and "import" not in reference:
            category = "api_or_import_misuse"
        elif "def " not in text and reference and "def " in reference:
            category = "prompt_non_compliance_wrong_output_format"
        elif re.search(r"\b(for|while|if|return)\b", text):
            category = "arithmetic_logic_mismatch"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "code_failure_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def diagnostics_output_dir(base_dump_dir: Optional[str], args: DiagnosticsArgs) -> Path:
    return Path(args.output_dir) if args.output_dir else Path(base_dump_dir or ".") / "diagnostics"
