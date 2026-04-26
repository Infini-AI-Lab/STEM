# STEM Diagnostics

<!-- ====================================================================== -->
## Task-Aligned Eval Sample Capture (`lingua.eval_sample_capture`)

Round 2 of the diagnostics pipeline.  Captures one
`DiagnosticSampleRecord` per lm-eval sample so failure analysis can be done
at the **task / sample / token** level rather than only at scalar metric
level.  This wiring is **opt-in** and adds no compute when disabled.

### Enable

```yaml
diagnostics:
  enabled: true
  collect_eval_samples: true            # default false
  capture_prompts: true                 # default true
  capture_generations: true             # default true
  capture_token_ids: false              # default false (heavier)
  max_eval_samples_per_task: 200        # default null (no cap)
  max_text_chars: 4096                  # truncates prompt/generation per record
  tasks: null                           # null = all; or e.g. [mbpp, humaneval, gsm8k]
  rank0_only: true                      # default true
  output_dir: null                      # default <dump_dir>/diagnostics
  run_id: null                          # default cfg.name
harness:
  log_samples: true                     # required by lm-eval to surface sample dicts
```

CLI:

```bash
python -m apps.main.stem_eval \
  config=apps/main/configs/stem_eval.yaml \
  diagnostics.enabled=true \
  diagnostics.collect_eval_samples=true \
  diagnostics.tasks='[mbpp,gsm8k]' \
  harness.log_samples=true
```

### Output files

| Path | Description |
|------|-------------|
| `<dump_dir>/diagnostics/diagnostics_eval_samples.jsonl` | One `DiagnosticSampleRecord` per line (rank 0 / single-process) |
| `<dump_dir>/diagnostics/diagnostics_eval_samples.shard{NN}.jsonl` | Per-DP-rank shard files when `rank0_only=false` |
| `<dump_dir>/diagnostics/eval_sample_summary.json` | Aggregated counts and missing-field bookkeeping |

The summary JSON contains:

- `total_records`, `total_correct`, `total_incorrect`, `total_correctness_unknown`
- `total_with_generation`, `total_with_token_ids`
- `missing_field_counts` — counts of records that flagged each missing field (`prompt_unavailable`, `generation_unavailable`, `target_unavailable`, `correctness_unknown`, `tokenization_failed`, `no_tokenizer_available`)
- `by_task[<task>]` — per-task `count`, `correct`, `incorrect`, `correctness_unknown`, `with_generation`, `with_token_ids`, `avg_sequence_length`, `task_group`

### Fields populated for each record

For an lm-eval sample produced via `harness.log_samples=true`, the adapter
fills:

- `task`, `task_group` (from `classify_task_group`)
- `sample_id` — stable SHA-256 hash of `(task, prompt, target, doc_id)`
- `doc_id` (from sample `doc_id`)
- `split` (from `doc.split` when present)
- `prompt` — first element of `arguments[0]`, falling back to `doc.prompt|text|question|input|passage`
- `target` — `sample.target`, falling back to `doc.target|answer|output`
- `generation` — `filtered_resps[0]` then `resps[0][0]` then `prediction`
- `correct` — bool derived from any of `acc`, `acc_norm`, `exact_match`, `pass@1`, `f1`, `rouge1`
- `metric_name` / `metric_value` — first matching metric from a preferred list
- `model_id`, `checkpoint_path`, `run_id` — passed in by the eval driver
- `metadata` — preserves lm-eval `doc_hash`, `prompt_hash`, `target_hash`, `filter`, `metrics`; also lists missing fields if any

If `capture_token_ids=true` and a tokenizer is available, the adapter also
fills:

- `token_ids` (bounded, default cap 2048)
- `tokens` (via `tokenizer.get_token_offsets` when supported, else
  `tokenizer.decode([id])` per id)
- `token_roles` (via `classify_token_role`)
- `sequence_length`

### Fields not populated in this round

- `baseline_loss` and `model_loss` — require running an extra forward pass;
  reserved for a later round that wires loss capture into the generator.
- `per_token_nll` — same reason.

When a field is unavailable the record sets it to `null` and adds a string
note to `metadata.missing_fields` (e.g., `"prompt_unavailable"`,
`"tokenization_failed"`).  The pipeline never crashes on a malformed
sample; conversion errors are recorded under `metadata.conversion_error`.

### Distributed / DP behaviour

- `eval.py` (vanilla, non-DP): lm-eval gathers samples to rank 0; rank 0
  writes `diagnostics_eval_samples.jsonl`; non-zero ranks are no-ops by
  default (`rank0_only=true`).
- `stem_eval.py` (STEM with DP > 1): each DP group has its own slice of
  samples (lm-eval's internal gather is bypassed in `lingua.lm_eval_dp`).
  When `rank0_only=true` (default) only the global-rank-0 DP shard is
  written.  Set `rank0_only=false` to write per-DP-shard files; an offline
  union of these shards reconstructs the full sample set.  The capture
  call is placed **before** the DP scalar-metric merge, which strips
  `samples` from the gathered dict.

### Backward compatibility with `analyze_eval_samples`

The legacy heuristic code-failure analysis in `lingua.diagnostics` still
reads from the in-memory lm-eval `results["samples"]` dict.  To run it
against the captured JSONL offline, use:

```python
from lingua.eval_sample_capture import (
    load_eval_sample_records,
    records_to_results_samples_dict,
)
from lingua.diagnostics import analyze_eval_samples, DiagnosticsArgs

rows = load_eval_sample_records(diag_dir)
fake_results = {"samples": records_to_results_samples_dict(rows)}
analyze_eval_samples(fake_results, DiagnosticsArgs(enabled=True, collect_code_error_taxonomy=True), diag_dir)
```

### Limitations

1. `harness.log_samples` must be true (it already defaults to true in
   `LMHarnessArgs`); without it lm-eval does not expose per-sample dicts.
2. Some lm-eval tasks do not include a meaningful `target` in `sample.target`
   (e.g., loglikelihood-only tasks).  The adapter falls back to `doc.target /
   answer / output` and otherwise leaves the field `null` and notes
   `target_unavailable` in metadata.
3. Tokenisation captures `prompt + generation` (or `prompt + target` if
   generation is missing).  We do not currently align positions to the
   `prompt_len` boundary; later rounds doing per-token NLL will record
   that boundary explicitly.
4. With STEM DP, the canonical `diagnostics_eval_samples.jsonl` contains
   only rank 0's slice unless `rank0_only=false`.  Each shard file is
   self-contained JSONL and can be unioned offline.

<!-- ====================================================================== -->
## Shared Diagnostic Schema (`lingua.diagnostic_records`)

`lingua/diagnostic_records.py` is the **foundation layer** for the task-aligned
STEM diagnostics pipeline.  It is pure Python (stdlib + optional torch/numpy)
and has no effect on model behaviour.  Later rounds will wire these records into
`stem_eval.py`, `stem_train.py`, and `apps/main/stem_diagnostics.py`.

### Record types

#### `DiagnosticSampleRecord`

One row per eval/generation sample.  Primary key: `(run_id, task, sample_id)`.

| Field | Type | Purpose |
|-------|------|---------|
| `run_id` | `str` | Identifies the eval/train run |
| `model_id` | `Optional[str]` | Model name or tag |
| `checkpoint_path` | `Optional[str]` | Path used for this run |
| `task` | `str` | lm-eval task name |
| `task_group` | `Optional[str]` | Coarse group: `code`, `math`, `knowledge_reasoning`, `commonsense`, `other` |
| `sample_id` | `str` | Unique sample identifier (upstream id or `make_sample_id()` hash) |
| `doc_id` | `Optional[str]` | Upstream document id |
| `split` | `Optional[str]` | Dataset split |
| `prompt` | `Optional[str]` | Prompt text (may be truncated; see `prompt_truncated`) |
| `prompt_truncated` | `bool` | True when prompt was trimmed by `record_to_dict()` |
| `target` | `Optional[str]` | Gold target |
| `generation` | `Optional[str]` | Model generation |
| `correct` | `Optional[bool]` | Whether the sample was scored correct |
| `metric_name` | `Optional[str]` | Name of the primary metric |
| `metric_value` | `Optional[float]` | Metric value |
| `baseline_loss` | `Optional[float]` | Dense/baseline model NLL |
| `model_loss` | `Optional[float]` | STEM model NLL |
| `per_token_nll` | `Optional[List[float]]` | Token-level NLL values |
| `token_ids` | `Optional[List[int]]` | Token IDs for the sequence |
| `tokens` | `Optional[List[str]]` | Decoded token strings |
| `token_roles` | `Optional[List[str]]` | Role per token (see `classify_token_role`) |
| `sequence_length` | `Optional[int]` | Number of tokens |
| `metadata` | `Dict[str, Any]` | Catch-all for extra fields |

#### `LayerPathMetricRecord`

Per-layer, optionally per-token-position, norms and geometry.  Join on
`(run_id, task, sample_id, layer_idx)`.

| Field | Type | Purpose |
|-------|------|---------|
| `layer_idx` | `int` | Transformer layer index |
| `token_position` | `Optional[int]` | Position within the sequence |
| `stem_norm` | `Optional[float]` | L2 norm of the STEM embedding output |
| `up_norm` | `Optional[float]` | L2 norm of the dense up-proj output |
| `combined_norm` | `Optional[float]` | L2 norm of the combined up vector |
| `stem_up_cos` | `Optional[float]` | Cosine similarity between stem and up vectors |
| `ffn_out_norm` | `Optional[float]` | L2 norm of the FFN output |
| `gate_alpha` | `Optional[float]` | Sigmoid gate value α |
| `w1_act_norm` | `Optional[float]` | L2 norm after w1 (pre-SiLU) |
| `silu_act_norm` | `Optional[float]` | L2 norm after SiLU activation |

#### `InterventionRecord`

Result of a causal-path intervention (ablation / mean-replacement / gate forcing).
Richer than the flat dicts currently written by `write_intervention_rows()`.

| Field | Type | Purpose |
|-------|------|---------|
| `intervention_type` | `str` | `ablate_stem`, `ablate_up`, `ablate_layer`, `force_gate`, `replace_mean`, … |
| `target_path` | `str` | `stem`, `up`, `gate`, `combined`, `layer` |
| `loss_original` | `float` | Unmodified model loss on this sample |
| `loss_intervened` | `float` | Loss after intervention |
| `delta_loss` | `float` | `loss_intervened - loss_original` |
| `delta_per_token_nll` | `Optional[List[float]]` | Per-token NLL delta |
| `path_relation` | `Optional[str]` | `cooperative`, `redundant`, `destructive_stem`, `destructive_up`, `stem_dominant`, `up_dominant`, `inconclusive` |

#### `TokenEffectRecord`

Aggregated per-token-id statistics.  Produced offline from
`DiagnosticSampleRecord` lists or from `TokenStatsAggregator.rows()`.

| Field | Type | Purpose |
|-------|------|---------|
| `token_id` | `int` | Vocabulary token id |
| `token` | `str` | Decoded string |
| `token_role` | `str` | Role category |
| `task_group` | `Optional[str]` | Coarse group such as `code`, `math`, or `knowledge_reasoning` |
| `frequency_bucket` | `Optional[str]` | `rare`, `mid`, `frequent`, `very_frequent` |
| `count` | `int` | Occurrence count |
| `stem_activation_norm_mean` | `Optional[float]` | Mean STEM activation norm for this token |
| `stem_ablation_delta_loss` | `Optional[float]` | Mean loss delta when STEM is ablated |
| `up_ablation_delta_loss` | `Optional[float]` | Mean loss delta when the up path is ablated |
| `combined_ablation_delta_loss` | `Optional[float]` | Mean loss delta when STEM and up are ablated together |
| `benefit_score` | `Optional[float]` | Positive = STEM helps this token |
| `harm_score` | `Optional[float]` | Positive = STEM hurts this token |
| `ineffective_score` | `Optional[float]` | High-count token has no positive STEM benefit |

#### `CodeFailureRecord`

Failure taxonomy for one code-generation sample.  Extends
`lingua.diagnostics.analyze_code_failure()` output with typed boolean flags.

| Field | Type | Purpose |
|-------|------|---------|
| `failure_category` | `str` | Heuristic category string |
| `parse_ok` | `bool` | Whether the generation parsed without a SyntaxError |
| `syntax_error` | `bool` | SyntaxError raised |
| `runtime_error` | `bool` | Runtime error during execution |
| `signature_error` | `bool` | Function signature mismatch |
| `test_failure` | `bool` | Unit tests failed |
| `timeout` | `bool` | Execution timed out |
| `traceback_type` | `Optional[str]` | Exception class name |
| `function_name_expected` | `Optional[str]` | Expected function name |
| `function_name_found` | `Optional[str]` | Found function name |

#### `DebuggabilityRecord`

High-level debuggability verdict for a sample or whole task.

| Field | Type | Purpose |
|-------|------|---------|
| `classification` | `str` | `likely_debuggable`, `possibly_architectural`, `inconclusive` |
| `evidence` | `List[str]` | Supporting observations |
| `confidence` | `float` | Score in [0, 1] |

---

### Helper functions

```python
# IO — safe from any rank; only writes from rank 0 when dist is initialised
append_jsonl(path, rows, *, rank0_only=True, max_prompt_chars=4096)
read_jsonl(path, *, skip_errors=True)
write_json_atomic(path, data, *, rank0_only=True, indent=2)

# Serialisation
record_to_dict(record, *, max_prompt_chars=4096, max_generation_chars=4096)
sanitize_for_json(obj)   # handles tensors, numpy, NaN/Inf, bfloat16, Path

# Stable deterministic sample id
make_sample_id(task, prompt=None, target=None, doc_idx=None) -> str  # 16-char hex

# Classification
classify_token_role(token: str) -> str    # newline | whitespace | python_keyword | ...
classify_task_group(task_name: str) -> str  # code | math | knowledge_reasoning | commonsense | other
```

### Token role categories

| Category | Examples |
|----------|---------|
| `newline` | `"\n"`, `"\\n"` |
| `whitespace` | `"   "`, `"\t"` |
| `python_keyword` | `def`, `return`, `import`, `class` |
| `identifier` | `foo`, `_bar`, `CamelCase` |
| `numeral` | `42`, `3.14`, `0xFF`, `1e-3` |
| `bracket` | `(`, `)`, `[`, `]`, `{`, `}` |
| `operator` | `+`, `==`, `<=`, `:=` |
| `punctuation` | `:`, `,`, `.`, `;` |
| `string_like` | `"hello`, `f"`, `b'x` |
| `natural_language` | `world!`, `isn't` |
| `unknown` | `@#$` |

### Task groups

| Group | Task-name substrings matched |
|-------|------------------------------|
| `code` | mbpp, humaneval, humaneval_plus, apps, code, python |
| `math` | gsm8k, math, minerva, amc, aime |
| `knowledge_reasoning` | mmlu, bbh, triviaqa, naturalqa, nq_open |
| `commonsense` | arc, hellaswag, piqa, winogrande, boolq, openbookqa, race |
| `other` | (fallback) |

---

> **Note:** These record types are the stable interchange layer used by eval,
> train, code-failure, and dashboard diagnostics.  No train/eval behaviour
> changes when diagnostics are disabled.

<!-- ====================================================================== -->

The diagnostics framework is optional and config driven.  With
`diagnostics.enabled: false` it registers no hooks, performs no extra forwards,
and does not change training, checkpointing, or eval behavior.

Scalar summaries are logged into the existing `metrics.jsonl` /
`metrics.eval.jsonl` files under `diag/...` keys.  Larger artifacts are written
under `<dump_dir>/diagnostics` unless `diagnostics.output_dir` is set.

## End-to-End Recipes

These overlays live under `configs/diagnostics/` and can be merged with the
normal app configs via OmegaConf `config=...` arguments.

### A. Minimal eval sample capture

Config overlay: `configs/diagnostics/eval_light.yaml`

```yaml
harness:
  log_samples: true
  limit: 5
diagnostics:
  enabled: true
  collect_eval_samples: true
  max_eval_samples_per_task: 5
  capture_prompts: true
  capture_generations: true
  capture_token_ids: false
```

Exact command:

```bash
python -m apps.main.stem_eval \
  config='[apps/main/configs/stem_eval.yaml,configs/diagnostics/eval_light.yaml]' \
  ckpt_dir=/path/to/stem/checkpoint \
  harness.tasks='[mbpp]' \
  dump_dir=logs/diag_eval_light
```

Expected first artifact: `logs/diag_eval_light/diagnostics/diagnostics_eval_samples.jsonl`.

### B. Full task-aligned causal code eval

Config overlay: `configs/diagnostics/eval_causal_code.yaml`

```yaml
harness:
  tasks: [mbpp, humaneval]
  log_samples: true
  limit: 20
diagnostics:
  enabled: true
  collect_eval_samples: true
  collect_eval_activations: true
  collect_eval_geometry: true
  write_layer_path_records: true
  collect_eval_interventions: true
  compute_per_token_delta: true
  update_token_effectiveness: true
  collect_code_error_taxonomy: true
  code_tasks: [mbpp, humaneval]
```

Exact commands:

```bash
python -m apps.main.stem_eval \
  config='[apps/main/configs/stem_eval.yaml,configs/diagnostics/eval_causal_code.yaml]' \
  ckpt_dir=/path/to/stem/checkpoint \
  dump_dir=logs/diag_causal_code

python -m apps.main.diagnostic_dashboard \
  --diagnostics-dir logs/diag_causal_code/diagnostics \
  --run-id diag_causal_code \
  --validate \
  --print-summary
```

This produces sample records, layer-path records, task-aligned interventions,
token-effect summaries, code-failure analysis, dashboard JSON, Markdown, and
`diagnostics_validation.json`.

### C. DAG train diagnostics

Config overlay: `configs/diagnostics/train_dag_light.yaml`

```yaml
diagnostics:
  enabled: true
  collect_train_stats: true
  collect_token_stats: true
  collect_optimizer_stats: true
  enable_backward_hooks: true
  enable_optimizer_moment_logging: true
  collect_interventions: false
  path_ablation_eval_every_n_steps: 2000
```

Exact command:

```bash
torchrun --nproc-per-node=4 -m apps.main.stem_dag_train \
  config='[apps/main/configs/stem_dag_llama3_1B.yaml,configs/diagnostics/train_dag_light.yaml]' \
  dump_dir=logs/diag_dag_train
```

For occasional train-time prompt interventions, add:

```bash
diagnostics.collect_interventions=true \
diagnostics.path_ablation_eval_every_n_steps=2000 \
diagnostics.path_ablation_num_batches=1
```

### D. Rich geometry

Config overlay: `configs/diagnostics/geometry_reference.yaml`

```yaml
diagnostics:
  enabled: true
  collect_eval_samples: true
  collect_eval_activations: true
  collect_eval_geometry: true
  collect_richer_geometry: true
  reference_checkpoint_path: null
  reference_model_type: null
  compute_cka: true
  compute_svcca: false
  max_geometry_samples_per_task: 8
  max_geometry_tokens_per_bucket: 512
```

Without a reference checkpoint:

```bash
python -m apps.main.stem_eval \
  config='[apps/main/configs/stem_eval.yaml,configs/diagnostics/geometry_reference.yaml]' \
  ckpt_dir=/path/to/stem/checkpoint \
  diagnostics.reference_checkpoint_path=null \
  dump_dir=logs/diag_geometry
```

With a reference checkpoint:

```bash
python -m apps.main.stem_eval \
  config='[apps/main/configs/stem_eval.yaml,configs/diagnostics/geometry_reference.yaml]' \
  ckpt_dir=/path/to/stem/checkpoint \
  diagnostics.reference_checkpoint_path=/path/to/baseline/checkpoint \
  diagnostics.reference_model_type=llama \
  dump_dir=logs/diag_geometry_ref
```

CKA/SVCCA can be expensive because the reference path loads a second model and
runs the same sampled prompts through it. Keep `eval_layers`,
`max_geometry_samples_per_task`, and `max_geometry_tokens_per_bucket` small.

### E. Expected artifacts

Full causal eval artifacts:

- `diagnostics_eval_samples.jsonl`
- `eval_sample_summary.json`
- `eval_activation_summary.json`
- `layer_path_metrics.jsonl`
- `eval_geometry_summary.json`
- `interventions_task_aligned.jsonl`
- `eval_intervention_summary.json`
- `token_effects.jsonl`
- `token_effects_by_task.json`
- `token_effects_by_role.json`
- `path_relations_by_task_layer.json`
- `code_failures.jsonl`
- `code_causal_failure_analysis.json`
- `code_failure_examples.jsonl`
- `path_interference_dashboard.json`
- `debuggability_report.json`
- `diagnostics_summary.md`
- `diagnostics_validation.json`

Geometry-specific artifacts:

- `geometry_layer_{L}.npz` for train geometry when `collect_geometry=true`
- `eval_geometry_layer_{L}.npz` when `eval_geometry_save_npz=true`
- `richer_geometry_summary.json`
- `geometry_by_task_layer_role.json`
- `geometry_by_frequency_bucket.json`
- `baseline_comparison_cka.json` when `reference_checkpoint_path` is provided

### F. Interpretation guide

- `delta_loss = intervened_loss - original_loss`.
- Positive STEM/up ablation delta means the ablated path was beneficial:
  removing it increased loss.
- Negative STEM/up ablation delta means the path was harmful:
  removing it decreased loss.
- `benefit_score = max(mean_stem_ablation_delta_loss, 0)`.
- `harm_score = max(-mean_stem_ablation_delta_loss, 0)`.
- `ineffective_score` is high when a high-count token has no positive STEM
  benefit, using `log1p(count)/(1 + abs(mean_delta))` when `mean_delta <= 0`.
- `cooperative`: ablating both STEM and up paths hurts.
- `redundant`: both path deltas are close to zero.
- `destructive_stem`: ablating STEM helps, so STEM was counterproductive.
- `destructive_up`: ablating the up path helps, so the up path was
  counterproductive.
- `likely_debuggable`: dashboard rules found localized, intervention-sensitive
  failure signals such as a small set of harmful layers/roles or helpful gate
  forcing.
- `possibly_architectural`: dashboard rules found broad failures, severe
  geometry collapse/anisotropy, or no meaningful intervention improvement.

### Smoke and validation

Run the self-contained e2e smoke test:

```bash
python -m apps.main.diagnostics_smoke_e2e
```

Validate an existing diagnostics directory and regenerate dashboard outputs:

```bash
python -m apps.main.diagnostic_dashboard \
  --diagnostics-dir logs/diag_causal_code/diagnostics \
  --run-id diag_causal_code \
  --validate
```

## Lightweight Train-Time Diagnostics

```yaml
diagnostics:
  enabled: true
  collect_train_stats: true
  collect_token_stats: true
  collect_geometry: false
  collect_optimizer_stats: false
  sample_every_n_steps: 500
  max_batches_per_collection: 1
  max_token_positions: 1024
  layers: [1, 3, 5, 7]
```

Run with an existing STEM or DAG train config:

```bash
torchrun --nproc-per-node=4 -m apps.main.stem_train \
  config=apps/main/configs/stem_llama3_1B.yaml \
  diagnostics.enabled=true \
  diagnostics.collect_train_stats=true \
  diagnostics.sample_every_n_steps=500
```

For DAG models use the DAG wrapper so the registry includes `llama_dag`,
`qwen3_dag`, and `olmo3_dag`:

```bash
torchrun --nproc-per-node=4 -m apps.main.stem_dag_train \
  config=apps/main/configs/stem_dag_llama3_1B.yaml \
  diagnostics.enabled=true \
  diagnostics.collect_train_stats=true \
  diagnostics.collect_optimizer_stats=true
```

## Legacy Prompt Path Ablations

This older `diagnostics.collect_interventions` path runs on validation prompts
or fallback strings, not on exact lm-eval samples.  Prefer the task-aligned
`collect_eval_interventions` recipe above for causal eval analysis.  This path
is kept for backward compatibility and lightweight prompt probing.

```yaml
diagnostics:
  enabled: true
  collect_eval_stats: true
  collect_interventions: true
  path_ablation_num_batches: 4
  layers: [1, 3, 5, 7]
  gate_alpha_values: [0.0, 0.25, 0.5, 0.75, 1.0]
```

Run:

```bash
python -m apps.main.stem_eval \
  config=apps/main/configs/stem_eval.yaml \
  diagnostics.enabled=true \
  diagnostics.collect_interventions=true \
  diagnostics.path_ablation_num_batches=4
```

For DAG checkpoints:

```bash
python -m apps.main.stem_dag_eval \
  config=apps/main/configs/stem_eval.yaml \
  model_type=llama_dag \
  diagnostics.enabled=true \
  diagnostics.collect_interventions=true
```

Interventions currently include STEM ablation, dense-path ablation when `w3`
exists, layerwise ablations, replacing paths with their running batch mean, and
DAG gate forcing.

### Task-Aligned Eval Interventions

The older `diagnostics.collect_interventions` path above runs on validation
prompts.  For causal answers about the exact MBPP, HumanEval, MMLU, GSM8K, or
other lm-eval sample, enable the task-aligned path:

```yaml
diagnostics:
  enabled: true
  collect_eval_interventions: true
  intervention_max_samples_per_task: 4
  intervention_layers: [1, 3, 5, 7]  # optional; defaults to eval_layers/layers/all STEM layers
  compute_per_token_delta: true
  update_token_effectiveness: true
  intervention_types:
    - ablate_stem
    - ablate_up
    - ablate_combined
    - ablate_layer_stem
    - ablate_layer_up
    - force_gate_0
    - force_gate_0_25
    - force_gate_0_5
    - force_gate_0_75
    - force_gate_1
    - replace_stem_mean
    - replace_up_mean
```

This reruns bounded extra forwards on the same prompt plus generation/target
tokens recovered from `results["samples"]`.  The default sample cap is small
(`intervention_max_samples_per_task: 8`) and the feature is off unless
`collect_eval_interventions` is true.

Sign convention:

- `delta_loss = intervened_loss - original_loss`
- `delta_per_token_nll = intervened_token_nll - original_token_nll`
- Positive STEM/up ablation delta means the ablated path was beneficial because
  removing it increased loss.
- Negative ablation delta means the path was harmful for that sample/token
  because removing it decreased loss.

Token-effect scores are intentionally simple:

- `benefit_score = max(mean_stem_ablation_delta_loss, 0)`
- `harm_score = max(-mean_stem_ablation_delta_loss, 0)`
- `ineffective_score = log1p(count) / (1 + abs(mean_stem_ablation_delta_loss))`
  when the mean STEM benefit is zero or negative, otherwise `0`

Path relations are classified per task/sample/layer from layerwise STEM and
up-path ablations:

- `cooperative`: ablating STEM hurts and ablating up hurts
- `destructive_stem`: ablating STEM helps
- `destructive_up`: ablating up helps
- `stem_dominant`: STEM ablation hurts much more than up ablation
- `up_dominant`: up ablation hurts much more than STEM ablation
- `redundant`: both deltas are close to zero
- `inconclusive`: one of the required metrics is missing

## Richer Eval Geometry

Richer geometry is a separate opt-in layer on top of eval activation capture.
It reruns bounded forward passes on lm-eval samples and writes task/layer/token
role geometry, observed-frequency bucket geometry drift, gate-conditioned
geometry, and optional baseline-vs-STEM CKA.

### Enable without a reference checkpoint

```yaml
diagnostics:
  enabled: true
  collect_richer_geometry: true
  collect_eval_activations: false       # optional; richer geometry can trigger the capture loop
  collect_eval_geometry: false          # old compact eval_geometry_summary remains independent
  max_geometry_samples_per_task: 8
  max_geometry_tokens_per_bucket: 512
  eval_activation_max_tokens_per_sample: 256
  eval_layers: [1, 3, 5, 7]             # optional cap; null = all FFN layers
  geometry_by_token_role: true
  geometry_by_frequency_bucket: true
harness:
  log_samples: true
```

CLI:

```bash
python -m apps.main.stem_eval \
  config=apps/main/configs/stem_eval.yaml \
  harness.log_samples=true \
  diagnostics.enabled=true \
  diagnostics.collect_richer_geometry=true \
  diagnostics.max_geometry_samples_per_task=8 \
  diagnostics.max_geometry_tokens_per_bucket=512
```

### Enable with a baseline/reference checkpoint

```yaml
diagnostics:
  enabled: true
  collect_richer_geometry: true
  reference_checkpoint_path: /path/to/baseline/0000200000
  reference_model_type: llama       # optional; otherwise read from params.json when possible
  compute_cka: true
  compute_svcca: false              # optional simplified PCA+CCA-style score
```

`reference_checkpoint_path` may point at a consolidated checkpoint directory
or at a checkpoint root containing a `consolidated/params.json`. If the path is
missing or unsupported, the run writes a skipped baseline-comparison status and
continues. No reference model is required unless you explicitly provide this
path.

CKA/SVCCA can be expensive because they require loading another model and
running the same sampled eval prompts through it. Keep
`max_geometry_samples_per_task`, `max_geometry_tokens_per_bucket`, and
`eval_layers` small for exploratory runs.

### Output artifacts

| Path | Description |
|------|-------------|
| `diagnostics/richer_geometry_summary.json` | Compact run metadata, metric notes, gate-bucket summary, and geometry warnings |
| `diagnostics/geometry_by_task_layer_role.json` | Task → layer → token-role geometry for hidden, STEM, up, combined, and FFN-output vectors |
| `diagnostics/geometry_by_frequency_bucket.json` | Task → layer → rare/mid/frequent observed-frequency geometry and drift vs all observed tokens |
| `diagnostics/baseline_comparison_cka.json` | Optional baseline-vs-current linear CKA, plus simplified SVCCA when requested |

### Metric meanings

| Metric | Meaning |
|--------|---------|
| `effective_rank` | Entropy effective rank of the centered singular-value spectrum; low values indicate dimensional collapse. |
| `explained_var_rank` | Smallest PCA rank explaining `diagnostics.geometry_var_frac` variance. |
| `anisotropy` | `||centroid|| / mean(||token_vector||)`; high values mean vectors point in a shared direction. |
| `pairwise_cos_mean/std` | Mean and standard deviation of sampled off-diagonal pairwise cosine similarities. |
| `centroid_cos_mean/std` | Cosine from each token vector to that cell's centroid. |
| `alignment_to_stem_centroid` | Mean cosine alignment to the STEM-path centroid for the same task/layer/role or bucket. |
| `alignment_to_up_path_mean` | Mean cosine alignment to the dense/up-path mean. |
| `alignment_to_up_path_top_pc` | Mean absolute cosine alignment to the up-path top principal component. |
| `norm_ratio_stem_to_up` | Mean STEM-path norm divided by mean up-path norm. |
| `linear_cka` | Centered linear CKA between row-aligned current and reference representations. |
| `simplified_svcca` | Optional lightweight PCA+CCA-style similarity; use a dedicated SVCCA package for publication-grade analysis. |

Frequency buckets are `rare`, `mid`, and `frequent`. The current eval-time
schema does not include corpus-level token frequencies, so buckets are marked
`frequency_source: observed_eval_subset` and derived as rank tertiles over
observed token counts in the sampled eval records. Treat them as approximate
observed-frequency buckets, not corpus-frequency buckets.

## MBPP / HumanEval Causal Failure Analysis (`lingua.code_diagnostics`)

The full causal analysis connects each sample's outcome to a failure category and
then joins that category with the per-sample intervention deltas, producing
queryable summaries.

### Enable

```yaml
diagnostics:
  enabled: true
  collect_eval_samples: true          # writes diagnostics_eval_samples.jsonl
  collect_eval_interventions: true    # writes interventions_task_aligned.jsonl
  collect_code_error_taxonomy: true   # triggers analyze_eval_samples wrapper
  code_tasks: [mbpp, humaneval]
  save_raw_samples: true
harness:
  log_samples: true
```

CLI:

```bash
python -m apps.main.stem_eval \
  config=apps/main/configs/stem_eval.yaml \
  harness.tasks='[mbpp,humaneval]' \
  harness.log_samples=true \
  diagnostics.enabled=true \
  diagnostics.collect_eval_samples=true \
  diagnostics.collect_eval_interventions=true \
  diagnostics.collect_code_error_taxonomy=true
```

### Output artifacts

| Path | Description |
|------|-------------|
| `diagnostics/code_failures.jsonl` | One `CodeFailureRecord` per code sample |
| `diagnostics/code_causal_failure_analysis.json` | Summary with counts, delta_loss averages, harmful layers/roles/tokens, examples |
| `diagnostics/code_failure_examples.jsonl` | Bounded prompt/generation snippets per sample (up to 200) |

### Failure categories

Samples are classified into exactly one of:

| Category | Source | Meaning |
|----------|--------|---------|
| `pass` | static + harness | Structurally valid and harness-confirmed correct |
| `syntax_error` | static | `ast.parse` raises `SyntaxError` (non-indentation) |
| `indentation_error` | static | `ast.parse` raises `IndentationError` |
| `unmatched_bracket_or_quote` | static | `SyntaxError` with EOL/EOF message, or unbalanced `()[]{}` |
| `missing_function` | static | No `def` found when task prompt contains `def` or `expected_fn_name` is set |
| `wrong_function_name` | static | Has a `def` but function name does not match expected |
| `signature_error` | static | Function found but positional arg count differs from prompt |
| `import_error_or_api_misuse` | static + harness | Wildcard import, or harness reports `ImportError` |
| `runtime_error` | harness | Harness traceback contains a non-timeout non-assertion exception |
| `timeout` | harness | Harness traceback contains `TimeoutError` or "timed out" |
| `test_failure` | harness | Harness traceback contains `AssertionError` or "test failed" |
| `logic_error_likely` | static | Parses OK, structurally valid, but `correct=False` |
| `prompt_noncompliance` | static | Prose-only response when code was expected |
| `empty_or_truncated_generation` | static | Generation is empty or fewer than 5 characters |
| `unknown_failure` | fallback | None of the above could be determined |

**Priority order:** harness execution results override static results when
available.  Within static analysis: empty → parse errors → delimiter errors →
missing/wrong function → signature → import → logic → pass.

### Static analysis limitations

1. **`logic_error_likely` is broad.** A generation that parses without error
   but the harness marks incorrect could be wrong for any reason: semantic bug,
   off-by-one, wrong algorithm, subtle API misuse not caught by static checks.
   It cannot distinguish these without execution.

2. **`import_error_or_api_misuse` is conservative.** Only wildcard
   (`from x import *`) imports are flagged statically.  Non-wildcard bad imports
   (`import nonexistent_module`) require execution to detect.

3. **Signature comparison is positional-only.** Parameters declared as
   `*args`, `**kwargs`, or keyword-only are not counted against the expected
   positional count.  The expected count is extracted from the prompt's
   function stub using a simple regex + comma-count heuristic.

4. **Fenced code extraction is best-effort.** The extractor tries
   ` ```python `, ` ```py `, then any backtick fence.  Inline code snippets
   without fences are treated as raw code.

5. **No code is executed.** Static analysis cannot detect `RuntimeError`,
   `NameError`, or `AttributeError` that only manifest at runtime.

### How to interpret intervention deltas with failure categories

From `code_causal_failure_analysis.json`:

```
avg_delta_loss_by_category["logic_error_likely"]["ablate_stem"]
```

A **positive** value means STEM was *beneficial* on average for
`logic_error_likely` samples (ablating it raised loss).  A **negative** value
means STEM was *harmful* (ablating it lowered loss — the model would have done
better without STEM on these samples).

- `stem_helpful_examples` — samples where `ablate_stem` delta was negative
  (STEM hurt; removing it helped).  If logic-error samples cluster here, STEM
  may be actively steering generations away from correct logic.
- `gate_up_helpful_examples` — samples where `force_gate_1` delta was lower
  than `force_gate_0`.  Forcing the gate toward the up-path was more helpful
  for these samples than STEM-path dominance.
- `harmful_layers_by_category` — layers where the mean per-layer STEM ablation
  delta was negative for samples of that category.  Points to specific layers
  where STEM is counterproductive for that failure mode.
- `harmful_token_roles_by_category` — token roles (e.g. `python_keyword`,
  `identifier`) where per-token STEM ablation delta was negative.  Indicates
  which syntactic positions STEM hurts most for that category.

### Standalone / offline use

```python
from lingua.code_diagnostics import run_code_causal_analysis
summary = run_code_causal_analysis(
    output_dir="path/to/diagnostics",
    run_id="my_run",
    code_tasks=["mbpp", "humaneval"],
)
```

The function reads `diagnostics_eval_samples.jsonl` and
`interventions_task_aligned.jsonl` from `output_dir` and writes the three
output artifacts.  It is safe to call when either file is absent; it returns
`{"skipped": True, "reason": ...}` and writes no files.

### Legacy `collect_code_error_taxonomy` path

The older `diagnostics.collect_code_error_taxonomy=true` path still works.
It calls `analyze_eval_samples` which writes `code_failure_analysis.json` with
simple per-task failure counts.  When `diagnostics_eval_samples.jsonl` is
present in the output directory, `analyze_eval_samples` also automatically
triggers `run_code_causal_analysis` for the richer analysis.  Both outputs are
included in `summary_eval.json`.

<!-- ====================================================================== -->
## Diagnostics Dashboard (`lingua.diagnostic_dashboard`)

Round 5 of the diagnostics pipeline.  Post-processing only — no model loading
required.  Reads all earlier-round artifacts and produces three summary files.

### Command

```bash
python -m apps.main.diagnostic_dashboard \
  --diagnostics-dir path/to/diagnostics \
  --run-id my_run \
  [--output-dir path/to/out] \
  [--total-layers 32] \
  [--print-summary] \
  [--validate]
```

`--diagnostics-dir` must point to the directory containing the JSONL / JSON
artifact files written by earlier rounds.  `--output-dir` defaults to
`--diagnostics-dir`.  `--total-layers` is used in the B1 breadth rule to
compute the fraction of layers that are STEM-harmful; if omitted it is
estimated from the `layer_path_metrics.jsonl` data.  `--print-summary` prints
the per-task verdicts to stdout.  `--validate` writes
`diagnostics_validation.json` and reports present/missing artifacts, record
counts, tasks, intervention types, token effects, code failures, and whether the
dashboard files were generated.

### Output files

| Path | Description |
|------|-------------|
| `diagnostics/path_interference_dashboard.json` | Per-(task, task_group, layer, token_role) path-norm and intervention statistics |
| `diagnostics/debuggability_report.json` | Per-task and global `DebuggabilityRecord` classifications with evidence strings |
| `diagnostics/diagnostics_summary.md` | Human-readable Markdown summary with tables, verdict, richer-geometry warnings, and recommended experiments |
| `diagnostics/diagnostics_validation.json` | Validation report when `--validate` is passed |

### `path_interference_dashboard.json` — key fields

```jsonc
{
  "available": true,
  "groups": [                              // one entry per (task, layer, token_role) combination
    {
      "task": "mbpp",
      "task_group": "code",
      "layer_idx": 2,
      "token_role": "python_keyword",
      "stem_norm":     {"mean": 1.23, "stdev": 0.05, "count": 480},
      "up_norm":       {"mean": 0.91, "stdev": 0.03, "count": 480},
      "combined_norm": {"mean": 1.40, "stdev": 0.06, "count": 480},
      "stem_up_cos":   {"mean": -0.18, "stdev": 0.12, "count": 480},
      "gate_alpha":    {"mean": 0.31, "stdev": 0.07, "count": 480,
                        "saturation_near0_frac": 0.12, "saturation_near1_frac": 0.03},
      "gate_available": true,
      "intervention_deltas": {             // mean Δloss per intervention type
        "ablate_stem": {"mean": -0.14, "stdev": 0.06, "count": 12}
      },
      "path_relation_counts": {"destructive_stem": 8, "cooperative": 2}
    }
  ],
  "global_path_relation_counts": {"destructive_stem": 40, "cooperative": 120, ...},
  "top_harmful_tokens":    [...],          // sorted by harm_score descending
  "top_beneficial_tokens": [...],
  "top_ineffective_tokens":[...],
  "gate_analysis": { ... },                // embedded gate analysis section
  "richer_geometry_analysis": {            // present when richer geometry artifacts exist
    "warning_counts": {"low_effective_rank": 2},
    "code_vs_reasoning_geometry": {...},
    "rare_token_geometry_differences": [...],
    "baseline_comparison": {...}
  }
}
```

`delta_loss = intervened_loss - original_loss`.  A **negative** `ablate_stem`
mean means STEM was *harmful* (removing it lowered loss).  A **positive** value
means STEM was *beneficial*.

### `debuggability_report.json` — key fields

```jsonc
{
  "run_id": "my_run",
  "global_classification": "likely_debuggable",
  "global_evidence": ["likely_debuggable: ['mbpp', 'humaneval']"],
  "summary_counts": {"likely_debuggable": 2, "inconclusive": 1},
  "tasks": ["humaneval", "mbpp", "gsm8k"],
  "per_task": [
    {
      "run_id": "my_run",
      "task": "mbpp",
      "classification": "likely_debuggable",   // likely_debuggable | possibly_architectural | inconclusive
      "evidence": [
        "A1: STEM harmful concentrated in 2 layer(s): [1, 2]",
        "A3: 3 sample(s) where ablating STEM lowered loss",
        "A6: gate forcing shows 0.30 mean delta difference ..."
      ],
      "confidence": 0.7,
      "metadata": {
        "task_group": "code",
        "n_total_samples": 50,
        "harmful_layers": [1, 2],
        "harmful_roles": ["identifier", "python_keyword"]
      }
    }
  ]
}
```

### Classifier rules

**`likely_debuggable`** (≥ 2 of these must hold):

| ID | Condition |
|----|-----------|
| A1 | STEM harmful concentrated in ≤ 2 layers |
| A2 | Forcing gate → up path improves code loss (`gate_up_helpful_examples` non-empty) |
| A3 | Ablating STEM improves failed code samples (`stem_helpful_examples` non-empty) |
| A4 | Harmful tokens concentrated in ≤ 3 token roles |
| A5 | `destructive_stem` path-relation fraction > 20% of all classified rows |
| A6 | Gate forcing shows > 0.05 Δloss difference between `force_gate_0` and `force_gate_1` |

**`possibly_architectural`** (≥ 2 of these must hold):

| ID | Condition |
|----|-----------|
| B1 | Harmful STEM layers span ≥ 40% of total layers |
| B2 | Harmful token roles span ≥ 5 distinct roles |
| B3 | Gate forcing shows < 0.05 difference — STEM-only and STEM+up fail similarly |
| B4 | No ablation produces a loss improvement > 0.01 |
| B5 | ≥ 3 code-specific token roles (identifier, numeral, python_keyword, …) consistently harmed |
| B6 | Geometry collapse (effective_rank < 4) or high anisotropy (> 0.9) in multiple layers |

When both sides have ≥ 2 signals, the verdict is `inconclusive` with a note
about the tied evidence.

### Standalone Python use

```python
from lingua.diagnostic_dashboard import run_diagnostic_dashboard

result = run_diagnostic_dashboard(
    diagnostics_dir="path/to/diagnostics",
    run_id="my_run",
    output_dir="path/to/out",  # optional; defaults to diagnostics_dir
    total_layers=32,            # optional hint
)

# result keys: path_interference_dashboard, debuggability_report,
#              markdown_path, artifacts_available
dr = result["debuggability_report"]
print(dr["global_classification"])
```

### Tolerance for missing artifacts

The dashboard never crashes on absent files.  When an artifact is missing:

- `layer_path_metrics.jsonl` absent → `path_interference_dashboard.groups` is empty,
  `gate_analysis.gate_available=false`.
- `interventions_task_aligned.jsonl` absent → all intervention deltas are empty;
  rules A1/A2/A3/A5/A6/B1/B3/B4 cannot fire; verdict defaults to `inconclusive`.
- `code_causal_failure_analysis.json` absent → rules A2/A3 cannot use
  `stem_helpful_examples` / `gate_up_helpful_examples`.
- `eval_geometry_summary.json` absent → rule B6 does not fire.
- `richer_geometry_summary.json` and related richer geometry artifacts absent →
  richer dashboard warnings are marked unavailable; the rest of the dashboard
  still runs.
- All absent → global verdict is `inconclusive` with an evidence note.

Availability of each artifact is reported in the `artifacts_available` dict in
the return value and printed by `--print-summary`.

---

## Train Diagnostics

Train-time diagnostics are supported in all three training entry-points:

| Script | How diagnostics are wired |
|--------|--------------------------|
| `stem_train.py` | Native — full diagnostics loop, optimizer stats, interventions |
| `stem_dag_train.py` | Delegates to `stem_train.train()` — identical wiring, no extra code |
| `stem_distill_train.py` | Delegates to `stem_train.train()` — identical wiring |
| `stem_projection_finetune.py` | Own training loop — uses `build_train_diagnostics_collector`; forward stats and param metrics supported; train-time interventions not wired (see limitations) |

### Enabling train diagnostics

```yaml
diagnostics:
  enabled: true
  collect_train_stats: true         # forward-pass hook metrics each sample step
  sample_every_n_steps: 500         # how often to collect (default 1000)
  collect_interventions: true       # path ablation suite (expensive; default false)
  path_ablation_eval_every_n_steps: 2000
  collect_optimizer_stats: true     # optimizer moment norms (default false)
  enable_backward_hooks: false      # gradient norm hooks (default false)
  layers: [0, 4, 8, 12]            # restrict to specific layers; null = all STEM layers
  train_task_label: null            # explicit label; overrides infer logic
  infer_task_from_data_path: false  # infer label from single-source data path
```

For DAG training add the same block under `diagnostics:` in your
`stem_dag_*.yaml` config — nothing else is needed.

### Task / source label

The `task` field in train-time records and artifacts is determined in this order:

1. `diagnostics.train_task_label` — explicit string, highest priority.
2. If `diagnostics.infer_task_from_data_path: true` and the data config has
   exactly **one** source, the base directory name of that source path is used
   (e.g. `"math_data"` from `data.sources: {"/datasets/math_data": 1.0}`).
3. Falls back to `"train"`.

For multi-source mixture training the label always falls back to `"train"` because
there is no per-batch source metadata in the tensor batch.  Per-source diagnostics
would require changes to the data loader to propagate source information.

### Train artifact files

| File | Description |
|------|-------------|
| `diagnostics/summary_train.json` | Scalar metrics summary + token rankings |
| `diagnostics/train_layer_path_summary.json` | Per-layer streaming stats, keyed `{task_label, per_layer: {layer_idx: {metric_mean, …}}}` |
| `diagnostics/train_token_effects.jsonl` | Token stats rows tagged with `task` field (only if `collect_token_stats: true`) |
| `diagnostics/interventions.jsonl` | Raw intervention rows (backward-compatible) |
| `diagnostics/train_interventions.jsonl` | Same rows plus `task` field injection (canonical train prefix) |
| `diagnostics/geometry_layer_{L}.npz` | Stem/dense geometry arrays per layer (if `collect_geometry: true`) |
| `diagnostics/token_stats.jsonl` | Low-level token stats rows |

### Known limitations

- **Per-batch source label not available**: the batch tensor carries only
  token IDs and labels.  Source/domain labels from a multi-source mixture
  are not propagated.  To get per-source diagnostics you would need the data
  loader to emit a per-batch source index; this is not yet implemented.
- **`stem_projection_finetune.py`** does not run the path-intervention suite
  during training (it only trains projection weights, not the base model).
  Forward stats and optimizer-param metrics work normally.
- **Backward hooks** (`enable_backward_hooks: true`) may have higher overhead
  with FSDP-sharded models — gradient hooks fire on the local shard.
- **No extra backward passes** are added by diagnostics.  Interventions run
  their own `torch.no_grad()` forwards; they never add a backward pass.

---

## Artifacts

Common outputs:

- `diagnostics/summary_train.json`
- `diagnostics/train_layer_path_summary.json`   ← train-specific (Round 6)
- `diagnostics/train_token_effects.jsonl`        ← train-specific (Round 6)
- `diagnostics/train_interventions.jsonl`        ← train-specific (Round 6)
- `diagnostics/summary_eval.json`
- `diagnostics/token_stats.jsonl`
- `diagnostics/interventions.jsonl`
- `diagnostics/interventions_task_aligned.jsonl`
- `diagnostics/token_effects.jsonl`
- `diagnostics/token_effects_by_task.json`
- `diagnostics/token_effects_by_role.json`
- `diagnostics/path_relations_by_task_layer.json`
- `diagnostics/geometry_layer_{L}.npz`
- `diagnostics/diagnostics_eval_samples.jsonl`
- `diagnostics/eval_sample_summary.json`
- `diagnostics/eval_activation_summary.json`
- `diagnostics/layer_path_metrics.jsonl`
- `diagnostics/eval_geometry_summary.json`
- `diagnostics/code_failure_analysis.json`
- `diagnostics/code_failures.jsonl`
- `diagnostics/code_causal_failure_analysis.json`
- `diagnostics/code_failure_examples.jsonl`
- `diagnostics/README.md`
- `diagnostics/path_interference_dashboard.json`  ← Round 5 dashboard
- `diagnostics/debuggability_report.json`         ← Round 5 classifier verdicts
- `diagnostics/diagnostics_summary.md`            ← Round 5 human-readable summary
- `diagnostics/diagnostics_validation.json`
- `diagnostics/richer_geometry_summary.json`
- `diagnostics/geometry_by_task_layer_role.json`
- `diagnostics/geometry_by_frequency_bucket.json`
- `diagnostics/baseline_comparison_cka.json`

Task-aligned artifact interpretation:

- `interventions_task_aligned.jsonl` has one row per sample and intervention,
  including original loss, intervened loss, signed loss delta, optional
  per-token NLL deltas, token ids, token roles, and `metadata.path_relation`.
- `token_effects.jsonl` aggregates signed per-token deltas by task, task group,
  layer, token id, token string, role, and frequency bucket.
- `token_effects_by_task.json` contains global and per-task rankings for
  beneficial, harmful, and ineffective STEM tokens/roles, plus STEM/up layers
  with the largest signed effects.
- `path_relations_by_task_layer.json` counts cooperative, destructive,
  dominant, redundant, and inconclusive sample/layer cases and stores the mean
  STEM/up ablation deltas behind each count.

## Caveats

Geometry and interventions add extra computation only when enabled. Keep
`max_batches_per_collection`, `max_token_positions`,
`max_tokens_per_layer_geometry`, `max_geometry_samples_per_task`, and
`max_geometry_tokens_per_bucket` small for routine runs. Baseline-checkpoint
CKA/SVCCA requires an explicit `reference_checkpoint_path` and may be expensive
because it loads a second model and runs the same sampled prompts through it.
