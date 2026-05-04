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

`--checkpoint-dir` can point to a 10-digit checkpoint step directory, a
`checkpoints/` directory, a dump directory containing `checkpoints/`, or an
already-created `consolidated/` directory. STEM shard loading follows the repo
checkpoint layout: `stem_shards/stem_model_mp*.pt` is consolidated into
`consolidated/consolidated_stem.pth` in rank order when needed.

Each run writes a timestamped directory containing:

- `metadata.json`
- `prompt_original.txt`
- `prompt_target.txt`
- `topk_next_token_probs.json`
- `full_results.json`
- `generations.jsonl`
- `generations.txt`
- `knowledge_edit_topk_probs.png`
- `knowledge_edit_topk_probs.pdf`
- `run.log`
