"""End-to-end smoke test for the STEM diagnostics workflow.

This script deliberately uses a tiny local torch model and fake lm-eval
samples.  It avoids checkpoints, tokenizers, distributed setup, CUDA, and
optional model dependencies while exercising the real diagnostics pipeline:

    python -m apps.main.diagnostics_smoke_e2e

Artifacts are written to a temporary directory unless ``--output-dir`` is
provided.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from lingua.code_diagnostics import run_code_causal_analysis
from lingua.diagnostic_dashboard import (
    DEBUGGABILITY_REPORT_JSON,
    DIAGNOSTICS_SUMMARY_MD,
    PATH_INTERFERENCE_DASHBOARD_JSON,
    run_diagnostic_dashboard,
    validate_diagnostics_artifacts,
)
from lingua.diagnostic_records import DiagnosticSampleRecord, read_jsonl
from lingua.diagnostics import (
    DiagnosticsArgs,
    TOKEN_EFFECTS_BY_ROLE_JSON,
    TOKEN_EFFECTS_BY_TASK_JSON,
    TOKEN_EFFECTS_JSONL,
    TASK_ALIGNED_INTERVENTIONS_JSONL,
    StemDiagnosticsCollector,
    TokenStatsAggregator,
    run_task_aligned_interventions,
)
from lingua.eval_activations import (
    EVAL_ACTIVATION_SUMMARY,
    LAYER_PATH_METRICS_JSONL,
    capture_eval_activations,
)
from lingua.eval_sample_capture import (
    EVAL_SAMPLES_JSONL,
    capture_eval_samples,
)


class TinyTokenizer:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {"<bos>": 0, "<eos>": 1, "<unk>": 2}
        self.id_to_token: Dict[int, str] = {0: "<bos>", 1: "<eos>", 2: "<unk>"}

    def _pieces(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%=(){}\\[\\]:,.]|\\S", text)

    def _id_for(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        next_id = len(self.token_to_id)
        if next_id >= self.vocab_size:
            return 2
        self.token_to_id[token] = next_id
        self.id_to_token[next_id] = token
        return next_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self._id_for(piece) for piece in self._pieces(text or "")]
        if add_bos:
            ids = [0] + ids
        if add_eos:
            ids.append(1)
        return ids

    def decode(self, ids) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self.id_to_token.get(int(i), "<unk>") for i in ids)

    def get_token_offsets(self, text: str, tokens=None):
        toks = [self.id_to_token.get(int(i), "<unk>") for i in (tokens or [])]
        return toks, list(range(len(toks)))


class TinyStemFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha_mode = "sigmoid_gated"

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.w1(x)
        dense = self.w3(x)
        stem = torch.zeros_like(dense) if y is None else y
        alpha = torch.sigmoid(self.alpha).to(device=x.device, dtype=x.dtype)
        up = (1.0 - alpha) * dense + alpha * stem
        return self.w2(F.silu(x1) * up)


class TinyLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feed_forward = TinyStemFeedForward(dim, hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + self.feed_forward(x, y)


class TinyDiagnosticsModel(nn.Module):
    def __init__(self, vocab_size: int = 64, dim: int = 12, hidden_dim: int = 16) -> None:
        super().__init__()
        self.max_seqlen = 64
        self.stem_layers = [0]
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.stem_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TinyLayer(dim, hidden_dim)])
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_values: torch.Tensor, target: Optional[torch.Tensor] = None):
        h = self.tok_embeddings(token_values)
        y = self.stem_embedding(token_values)
        for layer in self.layers:
            h = layer(h, y)
        logits = self.output(h)
        if target is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def _fake_lm_eval_results() -> Dict[str, object]:
    mbpp_prompt = "def add(a, b):\n    "
    mbpp_generation = "def add(a, b):\n    return a - b"
    humaneval_prompt = "def square(x):\n    "
    humaneval_generation = "def square(x):\n    return x * x"
    return {
        "samples": {
            "mbpp": [
                {
                    "doc_id": 0,
                    "doc": {"prompt": mbpp_prompt, "target": "return a + b", "split": "test"},
                    "arguments": [[mbpp_prompt, {}]],
                    "target": "return a + b",
                    "filtered_resps": [mbpp_generation],
                    "resps": [[mbpp_generation]],
                    "metrics": ["pass@1"],
                    "pass@1": 0.0,
                }
            ],
            "humaneval": [
                {
                    "doc_id": 1,
                    "doc": {"prompt": humaneval_prompt, "target": "return x * x", "split": "test"},
                    "arguments": [[humaneval_prompt, {}]],
                    "target": "return x * x",
                    "filtered_resps": [humaneval_generation],
                    "resps": [[humaneval_generation]],
                    "metrics": ["pass@1"],
                    "pass@1": 1.0,
                }
            ],
        }
    }


def _exercise_schema() -> None:
    rec = DiagnosticSampleRecord(
        run_id="smoke",
        task="mbpp",
        sample_id="schema",
        prompt="def f(): pass",
        generation="def f():\n    return 1",
        correct=False,
        token_ids=[0, 1],
        token_roles=["python_keyword", "identifier"],
    )
    assert rec.task == "mbpp"
    assert rec.metadata == {}


def _exercise_forward_only_collector(model: TinyDiagnosticsModel) -> None:
    args = DiagnosticsArgs(
        enabled=True,
        collect_eval_activations=True,
        collect_eval_geometry=True,
        eval_layers=[0],
        eval_activation_max_tokens_per_sample=16,
    )
    collector = StemDiagnosticsCollector(model, args, mode="eval", run_id="collector-smoke")
    collector.register()
    try:
        ids = torch.tensor([[0, 3, 4, 5, 6]], dtype=torch.long)
        collector.set_sample_context(
            task="mbpp",
            sample_id="collector",
            task_group="code",
            token_ids=ids.reshape(-1).tolist(),
            tokens=["<bos>", "def", "add", "(", ")"],
            token_roles=["unknown", "python_keyword", "identifier", "bracket", "bracket"],
        )
        model(ids)
        records = collector.flush_sample()
        assert collector.eval_summary_dict()["per_task"]["mbpp"]["samples_seen"] == 1
        assert records == []
    finally:
        collector.close()


def _exercise_token_aggregation() -> None:
    agg = TokenStatsAggregator(boundaries=[1, 3, 5], topk=5)
    ids = torch.tensor([[1, 2, 1, 3]])
    vals = torch.tensor([[0.5, -0.25, 0.75, 0.0]])
    agg.update_loss_delta(
        ids,
        "stem_ablation_delta_loss",
        vals,
        task="mbpp",
        task_group="code",
        layer_idx=0,
        tokens=["def", "add", "def", "return"],
        token_roles=["python_keyword", "identifier", "python_keyword", "python_keyword"],
    )
    records = agg.token_effect_records(run_id="smoke")
    assert records
    assert agg.rankings()["top_beneficial_tokens"]


def run_smoke(output_dir: Path) -> Dict[str, object]:
    torch.manual_seed(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TinyTokenizer(vocab_size=64)
    model = TinyDiagnosticsModel(vocab_size=64)
    results = _fake_lm_eval_results()

    _exercise_schema()
    _exercise_forward_only_collector(model)
    _exercise_token_aggregation()

    args = DiagnosticsArgs(
        enabled=True,
        collect_eval_samples=True,
        collect_eval_activations=True,
        collect_eval_geometry=True,
        collect_eval_interventions=True,
        collect_code_error_taxonomy=True,
        capture_token_ids=True,
        max_eval_samples_per_task=2,
        eval_activation_max_samples_per_task=2,
        eval_activation_max_tokens_per_sample=32,
        eval_layers=[0],
        write_layer_path_records=True,
        eval_activation_max_layer_path_records_per_sample=8,
        intervention_max_samples_per_task=2,
        intervention_layers=[0],
        intervention_types=[
            "ablate_layer_stem",
            "ablate_layer_up",
            "ablate_combined",
            "force_gate_0",
            "force_gate_1",
        ],
        compute_per_token_delta=True,
        update_token_effectiveness=True,
        code_tasks=["mbpp", "humaneval"],
        rank0_only=True,
        run_id="diagnostics-e2e-smoke",
    )

    sample_summary = capture_eval_samples(
        results=results,
        args=args,
        output_dir=output_dir,
        run_id="diagnostics-e2e-smoke",
        checkpoint_path="tiny://checkpoint",
        model_id="tiny-diagnostics-model",
        tokenizer=tokenizer,
        rank=0,
    )
    assert sample_summary["total_records"] == 2
    assert read_jsonl(output_dir / EVAL_SAMPLES_JSONL)

    activation_summary = capture_eval_activations(
        model=model,
        tokenizer=tokenizer,
        results=results,
        args=args,
        output_dir=output_dir,
        run_id="diagnostics-e2e-smoke",
        rank=0,
        model_max_seqlen=model.max_seqlen,
    )
    assert activation_summary["samples_processed"] == 2
    assert read_jsonl(output_dir / LAYER_PATH_METRICS_JSONL)

    intervention_summary = run_task_aligned_interventions(
        model=model,
        tokenizer=tokenizer,
        results=results,
        args=args,
        output_dir=output_dir,
        run_id="diagnostics-e2e-smoke",
        rank=0,
        model_id="tiny-diagnostics-model",
        checkpoint_path="tiny://checkpoint",
    )
    assert intervention_summary["samples_processed"] == 2
    assert read_jsonl(output_dir / TASK_ALIGNED_INTERVENTIONS_JSONL)
    assert read_jsonl(output_dir / TOKEN_EFFECTS_JSONL)

    code_summary = run_code_causal_analysis(
        output_dir=output_dir,
        run_id="diagnostics-e2e-smoke",
        code_tasks=["mbpp", "humaneval"],
        save_examples=True,
    )
    assert not code_summary.get("skipped")
    assert read_jsonl(output_dir / "code_failures.jsonl")

    dashboard_result = run_diagnostic_dashboard(
        diagnostics_dir=output_dir,
        run_id="diagnostics-e2e-smoke",
        output_dir=output_dir,
        total_layers=1,
    )
    assert (output_dir / PATH_INTERFERENCE_DASHBOARD_JSON).exists()
    assert (output_dir / DEBUGGABILITY_REPORT_JSON).exists()
    assert (output_dir / DIAGNOSTICS_SUMMARY_MD).exists()

    validation = validate_diagnostics_artifacts(
        diagnostics_dir=output_dir,
        output_dir=output_dir,
        run_id="diagnostics-e2e-smoke",
        write_report=True,
    )
    assert validation["dashboard_generated"]
    assert validation["record_counts"]["eval_samples"] == 2
    assert validation["record_counts"]["interventions"] > 0
    assert validation["record_counts"]["token_effects"] > 0
    assert validation["record_counts"]["code_failures"] == 2

    required = [
        EVAL_SAMPLES_JSONL,
        LAYER_PATH_METRICS_JSONL,
        EVAL_ACTIVATION_SUMMARY,
        TASK_ALIGNED_INTERVENTIONS_JSONL,
        TOKEN_EFFECTS_JSONL,
        TOKEN_EFFECTS_BY_TASK_JSON,
        TOKEN_EFFECTS_BY_ROLE_JSON,
        "code_failures.jsonl",
        "code_causal_failure_analysis.json",
        PATH_INTERFERENCE_DASHBOARD_JSON,
        DEBUGGABILITY_REPORT_JSON,
        DIAGNOSTICS_SUMMARY_MD,
        "diagnostics_validation.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    assert not missing, f"missing artifacts: {missing}"

    return {
        "output_dir": str(output_dir),
        "sample_summary": sample_summary,
        "activation_samples": activation_summary["samples_processed"],
        "intervention_samples": intervention_summary["samples_processed"],
        "code_failure_categories": code_summary.get("failure_counts_by_category", {}),
        "validation": validation,
        "dashboard_keys": sorted(dashboard_result.keys()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the diagnostics end-to-end smoke workflow.")
    parser.add_argument("--output-dir", default=None, help="Optional directory to keep smoke artifacts.")
    args = parser.parse_args()

    if args.output_dir:
        summary = run_smoke(Path(args.output_dir))
        print(json.dumps(summary, indent=2, default=str))
        return 0

    with TemporaryDirectory() as tmp:
        summary = run_smoke(Path(tmp))
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
