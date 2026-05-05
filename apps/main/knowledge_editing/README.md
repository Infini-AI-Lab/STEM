# STEM Knowledge Editing

Standalone experiment package for Section 3.3 / Figure 7 style STEM knowledge
editing. The intervention leaves the prompt text and token IDs unchanged and
only swaps STEM embedding lookup vectors at the source entity token positions.

Dry-run tokenization:

```bash
python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
  --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
  --source-entity Spain \
  --target-entity Germany \
  --dry-run-tokenization
```

Full experiment:

```bash
python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
  --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
  --checkpoint-dir /path/to/dump/checkpoints/0000100000 \
  --output-dir /tmp/stem_knowledge_editing \
  --prompt-type country-capital \
  --source-entity Spain \
  --target-entity Germany \
  --top-k 4 \
  --max-new-tokens 100 \
  --temperature 0.0 \
  --seed 0 \
  --device cuda \
  --dtype bfloat16 \
  --edit-mode auto
```

Math text operator-edit experiment:

```bash
python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
  --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
  --checkpoint-dir /path/to/dump/checkpoints/0000100000 \
  --output-dir /tmp/stem_knowledge_editing \
  --prompt-type math-text \
  --source-operator add \
  --target-operator subtract \
  --top-k 4 \
  --max-new-tokens 100 \
  --temperature 0.0 \
  --seed 0 \
  --device cuda \
  --dtype bfloat16 \
  --edit-mode auto
```

Additional prompt types:

| Prompt type | Alias | Default source -> target | Edited field |
| --- | --- | --- | --- |
| `country-capital-zero-shot` | | `Spain` -> `Germany` | country |
| `math-text-zero-shot` | | `add` -> `subtract` | operator |
| `math-unary-op` | `math-prompt-1` | `square` -> `cube` | operation |
| `math-unary-op-zero-shot` | `math-prompt-1-zero-shot` | `square` -> `cube` | operation |
| `math-binary-op` | `math-prompt-2` | `add` -> `multiply` | operation |
| `math-binary-op-zero-shot` | `math-prompt-2-zero-shot` | `add` -> `multiply` | operation |
| `math-derivative` | `math-prompt-3` | `sin(x)` -> `cos(x)` | function |
| `math-derivative-zero-shot` | `math-prompt-3-zero-shot` | `sin(x)` -> `cos(x)` | function |
| `math-prime-composite` | `math-prompt-4` | `13` -> `21` | number |
| `math-prime-composite-zero-shot` | `math-prompt-4-zero-shot` | `13` -> `21` | number |
| `math-area` | `math-prompt-5` | `circle` -> `square` | shape |
| `math-area-zero-shot` | `math-prompt-5-zero-shot` | `circle` -> `square` | shape |
| `coding-sort-reverse` | `coding-prompt-1` | `sort` -> `reverse` | action |
| `coding-sort-reverse-zero-shot` | `coding-prompt-1-zero-shot` | `sort` -> `reverse` | action |
| `coding-builtin-call` | `coding-prompt-2` | `count` -> `sum` | operation |
| `coding-builtin-call-zero-shot` | `coding-prompt-2-zero-shot` | `count` -> `sum` | operation |
| `coding-array-constructor` | `coding-prompt-3` | `zeros` -> `ones` | array type |
| `coding-array-constructor-zero-shot` | `coding-prompt-3-zero-shot` | `zeros` -> `ones` | array type |
| `coding-pandas-method` | `coding-prompt-4` | `first rows` -> `last rows` | request |
| `coding-pandas-method-zero-shot` | `coding-prompt-4-zero-shot` | `first rows` -> `last rows` | request |
| `coding-sql-aggregate` | `coding-prompt-5` | `COUNT` -> `SUM` | aggregate |
| `coding-sql-aggregate-zero-shot` | `coding-prompt-5-zero-shot` | `COUNT` -> `SUM` | aggregate |

If `--source-entity`/`--target-entity` are omitted, the selected prompt type
uses the default source -> target pair from this table. The
`--source-operator` and `--target-operator` flags are aliases for
`--source-entity` and `--target-entity`.

Example using a numbered alias:

```bash
python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
  --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
  --checkpoint-dir /path/to/dump/checkpoints/0000100000 \
  --output-dir /tmp/stem_knowledge_editing \
  --prompt-type coding-prompt-5 \
  --source-entity COUNT \
  --target-entity SUM \
  --top-k 4 \
  --max-new-tokens 100 \
  --temperature 0.0 \
  --seed 0 \
  --device cuda \
  --dtype bfloat16
```

Example zero-shot run:

```bash
python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
  --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
  --checkpoint-dir /path/to/dump/checkpoints/0000100000 \
  --output-dir /tmp/stem_knowledge_editing \
  --prompt-type math-prompt-1-zero-shot \
  --top-k 4 \
  --max-new-tokens 100 \
  --temperature 0.0 \
  --seed 0 \
  --device cuda \
  --dtype bfloat16
```

The `math-text` default prompt edits the operator in the final problem:

```text
Problem: three subtract by one
Answer: two

Problem: one add by ten
Answer: eleven

Problem: two multiply by four
Answer: eight

Problem: six divide by three
Answer: two

Problem: nine add by two
Answer:
```

By default, the CLI first runs an intervention diagnostic gate and refuses to
continue if the gate fails. The gate verifies that the intervened prompt text
and token IDs are identical to the original prompt, each STEM layer receives
the override, only the source entity positions differ from a normal STEM
embedding lookup, those positions equal the target-vector replacement strategy,
and selected STEM embedding weight rows remain unchanged.

Diagnostics-only:

```bash
python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
  --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
  --checkpoint-dir /path/to/dump/checkpoints/0000100000 \
  --output-dir /tmp/stem_knowledge_editing \
  --source-entity Spain \
  --target-entity Germany \
  --device cuda \
  --dtype bfloat16 \
  --diagnostics-only
```

`--checkpoint-dir` can point to a 10-digit checkpoint step directory, a
`checkpoints/` directory, a dump directory containing `checkpoints/`, or an
already-created `consolidated/` directory. STEM shard loading follows the repo
checkpoint layout: `stem_shards/stem_model_mp*.pt` is consolidated into
`consolidated/consolidated_stem.pth` in rank order when needed.

Each run writes a timestamped directory containing:

- `metadata.json`
- `intervention_diagnostics.json`
- `prompt_original.txt`
- `prompt_target.txt`
- `topk_next_token_probs.json`
- `full_results.json`
- `generations.jsonl`
- `generations.txt`
- `knowledge_edit_topk_probs.png`
- `knowledge_edit_topk_probs.pdf`
- `run.log`
