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
    def __init__(
        self,
        model: torch.nn.Module,
        args: DiagnosticsArgs,
        *,
        output_dir: Optional[Path] = None,
        prefix: str = "diag/train",
    ) -> None:
        self.model = model
        self.args = args
        self.prefix = prefix
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

    @property
    def enabled(self) -> bool:
        return bool(self.args.enabled)

    def __enter__(self) -> "StemDiagnosticsCollector":
        if self.enabled:
            self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def register(self) -> None:
        if self.handles or not self.enabled:
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
                    self.stats[f"layer_{layer_idx}/alpha_sigmoid"].update(alpha_sig)
                    if x3 is not None and y is not None:
                        up = (1.0 - alpha_sig.to(x3.device).to(x3.dtype)) * x3 + alpha_sig.to(y.device).to(y.dtype) * y
                if up is not None:
                    out2 = module.w2(act * up)
                else:
                    out2 = _as_local_tensor(output)

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
                if hasattr(self, "alpha") and dense is not None and stem is not None:
                    if kind == "force_gate" and gate_value is not None:
                        alpha_sig = torch.tensor(gate_value, device=x.device, dtype=x.dtype)
                    else:
                        alpha_sig = torch.sigmoid(_as_local_tensor(self.alpha)).to(device=x.device, dtype=x.dtype)
                    up = (1.0 - alpha_sig) * dense + alpha_sig * stem
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
