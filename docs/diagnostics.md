# STEM Diagnostics

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

## MBPP / HumanEval Failure Analysis

Enable lm-eval sample logging and code taxonomy:

```bash
python -m apps.main.stem_eval \
  config=apps/main/configs/stem_eval.yaml \
  harness.tasks='[mbpp,humaneval]' \
  harness.log_samples=true \
  diagnostics.enabled=true \
  diagnostics.collect_code_error_taxonomy=true \
  diagnostics.save_raw_samples=true
```

The taxonomy is deliberately offline-safe and heuristic.  It uses Python parsing
and regex checks to bucket generations into syntax/parse, indentation,
unmatched delimiter, missing symbol, API/import misuse, logic mismatch,
long-range dependency, prompt-format, or unknown categories.  It does not execute
generated code.

## Artifacts

Common outputs:

- `diagnostics/summary_train.json`
- `diagnostics/summary_eval.json`
- `diagnostics/token_stats.jsonl`
- `diagnostics/interventions.jsonl`
- `diagnostics/geometry_layer_{L}.npz`
- `diagnostics/code_failure_analysis.json`
- `diagnostics/README.md`

## Caveats

Geometry and interventions add extra computation only when enabled.  Keep
`max_batches_per_collection`, `max_token_positions`, and
`max_tokens_per_layer_geometry` small for routine runs.  Baseline-checkpoint CKA
and dense-path swapping are reserved behind config fields and not run unless
explicitly implemented for a specific comparison checkpoint.

