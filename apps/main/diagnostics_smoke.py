"""Executable smoke checks for ``lingua.diagnostics``.

These checks avoid checkpoints, tokenizers, FSDP, and CUDA.  Run in an
environment with torch installed:

    python -m apps.main.diagnostics_smoke
"""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from lingua.diagnostics import (
    DiagnosticsArgs,
    StemDiagnosticsCollector,
    TokenStatsAggregator,
    geometry_summary,
    run_intervention_suite,
    run_task_aligned_interventions,
)
from lingua.diagnostic_records import read_jsonl
from lingua.stem import StemFeedForward
from lingua.stem_dag import STEMDagFeedForward


class _Layer(nn.Module):
    def __init__(self, ff):
        super().__init__()
        self.feed_forward = ff

    def forward(self, x, y=None):
        return x + self.feed_forward(x, y)


class _LM(nn.Module):
    def __init__(self, parent, ff):
        super().__init__()
        object.__setattr__(self, "_stem_parent", parent)
        self.stem_layers = [0]
        self.layers = nn.ModuleList([_Layer(ff)])
        self.tok_embeddings = nn.Embedding(17, 8)
        self.output = nn.Linear(8, 17, bias=False)
        self.max_seqlen = 16

    def forward(self, token_values, target=None):
        h = self.tok_embeddings(token_values)
        y = self._stem_parent.stem_embeddings[0](token_values)
        for layer in self.layers:
            h = layer(h, y)
        logits = self.output(h)
        if target is None:
            return logits
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
        )


class _StemToy(nn.Module):
    def __init__(self, dag=False):
        super().__init__()
        ff_cls = STEMDagFeedForward if dag else StemFeedForward
        ff = ff_cls(dim=8, hidden_dim=16, multiple_of=1, ffn_dim_multiplier=None)
        self.stem_embeddings = nn.ModuleList([nn.Embedding(17, ff.hidden_dim)])
        self._layer_to_stem_idx = {0: 0}
        self.lm_transformer = _LM(self, ff)

    @property
    def max_seqlen(self):
        return self.lm_transformer.max_seqlen

    def forward(self, token_values, target=None):
        return self.lm_transformer(token_values, target)


def _assert_hooks_register_and_cleanup_plain_stem_without_w3(tmp_path: Path):
    model = _StemToy(dag=False)
    args = DiagnosticsArgs(
        enabled=True,
        collect_train_stats=True,
        collect_token_stats=True,
        collect_geometry=True,
        output_dir=str(tmp_path),
    )
    collector = StemDiagnosticsCollector(model, args)
    collector.register()
    assert collector.handles
    tokens = torch.randint(0, 17, (2, 5))
    collector.start_batch(tokens)
    loss = model(tokens, tokens)
    loss.backward()
    collector.end_batch()
    metrics = collector.scalar_metrics()
    assert any("stem_norm_mean" in key for key in metrics)
    assert not any(key.endswith("/up_norm_mean") for key in metrics)
    collector.close()
    assert collector.handles == []


def _assert_dag_gate_metric_logged():
    model = _StemToy(dag=True)
    args = DiagnosticsArgs(enabled=True, collect_train_stats=True)
    with StemDiagnosticsCollector(model, args) as collector:
        tokens = torch.randint(0, 17, (2, 5))
        collector.start_batch(tokens)
        model(tokens, tokens)
        collector.end_batch()
        metrics = collector.scalar_metrics()
    gate_keys = [key for key in metrics if "alpha_sigmoid_mean" in key]
    assert gate_keys
    assert 0.0 <= metrics[gate_keys[0]] <= 1.0


def _assert_intervention_engine_changes_loss_on_toy_batch():
    torch.manual_seed(0)
    model = _StemToy(dag=True)
    tokens = torch.randint(0, 17, (2, 5))
    labels = torch.randint(0, 17, (2, 5))
    args = DiagnosticsArgs(enabled=True, collect_interventions=True)
    metrics, rows = run_intervention_suite(model, tokens, labels, args)
    assert rows
    deltas = [v for k, v in metrics.items() if k.endswith("delta_loss")]
    assert deltas
    assert any(abs(v) > 1e-8 for v in deltas)


def _assert_token_stats_aggregator_merges_batches():
    agg = TokenStatsAggregator(boundaries=[1, 3, 5], topk=5)
    ids = torch.tensor([[1, 2, 1], [3, 2, 1]])
    vals = torch.ones_like(ids, dtype=torch.float32)
    agg.update(0, ids, stem_norm=vals)
    agg.update_loss_delta(ids, "stem_ablation_loss_delta", vals * 0.5)
    rows = agg.rows()
    assert {row["token_id"] for row in rows} == {1, 2, 3}
    assert agg.freq[1] == 3
    assert agg.rankings()["top_beneficial_tokens"][0]["score"] == 0.5


class _SmokeTokenizer:
    def encode(self, text, add_bos=False, add_eos=False):
        ids = [int(tok) % 17 for tok in text.split()] if text else []
        if add_bos:
            ids = [0] + ids
        if add_eos:
            ids.append(1)
        return ids

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def get_token_offsets(self, text, tokens=None):
        toks = self.decode(tokens or []).split()
        return toks, list(range(len(toks)))


def _assert_task_aligned_interventions_write_artifacts(tmp_path: Path):
    torch.manual_seed(0)
    model = _StemToy(dag=True)
    args = DiagnosticsArgs(
        enabled=True,
        collect_eval_interventions=True,
        intervention_max_samples_per_task=1,
        intervention_layers=[0],
        intervention_types=["ablate_layer_stem", "ablate_layer_up", "ablate_combined"],
        compute_per_token_delta=True,
        update_token_effectiveness=True,
        output_dir=str(tmp_path),
    )
    results = {
        "samples": {
            "mbpp": [
                {
                    "doc_id": 0,
                    "doc": {"prompt": "2 3"},
                    "target": "4 5",
                    "arguments": [["2 3", {}]],
                    "filtered_resps": ["4 5"],
                    "metrics": ["pass@1"],
                    "pass@1": 1.0,
                }
            ]
        }
    }
    summary = run_task_aligned_interventions(
        model=model,
        tokenizer=_SmokeTokenizer(),
        results=results,
        args=args,
        output_dir=tmp_path,
        run_id="smoke",
        rank=0,
    )
    assert summary["samples_processed"] == 1
    assert read_jsonl(tmp_path / "interventions_task_aligned.jsonl")
    assert read_jsonl(tmp_path / "token_effects.jsonl")


def _assert_geometry_utilities_are_finite():
    torch.manual_seed(0)
    summary = geometry_summary(torch.randn(32, 8))
    assert summary
    for value in summary.values():
        assert math.isfinite(value)


def main():
    with TemporaryDirectory() as tmp:
        _assert_hooks_register_and_cleanup_plain_stem_without_w3(Path(tmp))
    _assert_dag_gate_metric_logged()
    _assert_intervention_engine_changes_loss_on_toy_batch()
    _assert_token_stats_aggregator_merges_batches()
    with TemporaryDirectory() as tmp:
        _assert_task_aligned_interventions_write_artifacts(Path(tmp))
    _assert_geometry_utilities_are_finite()
    print("diagnostics smoke checks passed")


if __name__ == "__main__":
    main()
