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

> **Note:** This module is infrastructure only.  Later rounds will wire these
> record types into `stem_eval.py` (sample-level JSONL output), `stem_train.py`
> (train-step records), and `apps/main/stem_diagnostics.py` (standalone
> analysis).  No train/eval behaviour changes when diagnostics are disabled.

<!-- ====================================================================== -->

The diagnostics framework is optional and config driven.  With
`diagnostics.enabled: false` it registers no hooks, performs no extra forwards,
and does not change training, checkpointing, or eval behavior.

Scalar summaries are logged into the existing `metrics.jsonl` /
`metrics.eval.jsonl` files under `diag/...` keys.  Larger artifacts are written
under `<dump_dir>/diagnostics` unless `diagnostics.output_dir` is set.

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

## Eval-Time Path Ablations

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

## Artifacts

Common outputs:

- `diagnostics/summary_train.json`
- `diagnostics/summary_eval.json`
- `diagnostics/token_stats.jsonl`
- `diagnostics/interventions.jsonl`
- `diagnostics/interventions_task_aligned.jsonl`
- `diagnostics/token_effects.jsonl`
- `diagnostics/token_effects_by_task.json`
- `diagnostics/token_effects_by_role.json`
- `diagnostics/path_relations_by_task_layer.json`
- `diagnostics/geometry_layer_{L}.npz`
- `diagnostics/code_failure_analysis.json`
- `diagnostics/code_failures.jsonl`
- `diagnostics/code_causal_failure_analysis.json`
- `diagnostics/code_failure_examples.jsonl`
- `diagnostics/README.md`

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

Geometry and interventions add extra computation only when enabled.  Keep
`max_batches_per_collection`, `max_token_positions`, and
`max_tokens_per_layer_geometry` small for routine runs.  Baseline-checkpoint CKA
and dense-path swapping are reserved behind config fields and not run unless
explicitly implemented for a specific comparison checkpoint.
