#!/usr/bin/env bash
# ============================================================================
# download_eval_datasets.sh
#
# Downloads all HuggingFace datasets used by lm-eval-harness tasks in
# apps/main/configs/eval_qwen3.yaml to a local root directory.
#
# Usage:
#   bash setup/download_eval_datasets.sh /raid/user_data/rsadhukh/test_data
#
# Each dataset is cloned from its HuggingFace repo (with LFS data), and any
# legacy .py dataset scripts are removed (newer `datasets` versions reject
# them with "Dataset scripts are no longer supported").
# ============================================================================

set -euo pipefail

ROOT_DIR="${1:?Usage: $0 <local_root_dir>}"
mkdir -p "$ROOT_DIR"

# ── Mapping: local_dir_name  →  HuggingFace repo ID ──────────────────────
#
# Task              HF dataset_path          dataset_name      Local dir
# ─────────────     ─────────────────────    ──────────────    ──────────
# hellaswag         Rowan/hellaswag          (none)            hellaswag
# boolq             aps/super_glue           boolq             super_glue
# piqa              baber/piqa               (none)            piqa
# winogrande        allenai/winogrande       winogrande_xl     winogrande
# openbookqa        allenai/openbookqa       main              openbookqa
# arc_easy          allenai/ai2_arc          ARC-Easy          ai2_arc
# arc_challenge     allenai/ai2_arc          ARC-Challenge     ai2_arc  (same)
# race              EleutherAI/race          high              race
# gsm8k             openai/gsm8k             main              gsm8k
# mmlu              cais/mmlu                (per-subject)     mmlu

declare -A DATASETS=(
    ["hellaswag"]="Rowan/hellaswag"
    ["super_glue"]="aps/super_glue"
    ["piqa"]="baber/piqa"
    ["winogrande"]="allenai/winogrande"
    ["openbookqa"]="allenai/openbookqa"
    ["ai2_arc"]="allenai/ai2_arc"
    ["race"]="EleutherAI/race"
    ["gsm8k"]="openai/gsm8k"
    ["mmlu"]="cais/mmlu"
)

clone_dataset() {
    local name="$1"
    local repo="$2"
    local dest="$ROOT_DIR/$name"

    if [ -d "$dest" ]; then
        echo "[SKIP]  $name  — already exists at $dest"
        return
    fi

    echo "[CLONE] $name  ← https://huggingface.co/datasets/$repo"
    GIT_LFS_SKIP_SMUDGE=0 git clone "https://huggingface.co/datasets/$repo" "$dest"

    # Remove legacy .py dataset scripts that cause:
    #   RuntimeError: Dataset scripts are no longer supported
    local py_scripts
    py_scripts=$(find "$dest" -maxdepth 1 -name "*.py" ! -name "__*" 2>/dev/null || true)
    if [ -n "$py_scripts" ]; then
        echo "        Removing legacy dataset scripts: $(echo $py_scripts | xargs -n1 basename)"
        echo "$py_scripts" | xargs rm -f
    fi

    echo "        Done."
}

echo "=== Downloading eval datasets to: $ROOT_DIR ==="
echo ""

for name in "${!DATASETS[@]}"; do
    clone_dataset "$name" "${DATASETS[$name]}"
    echo ""
done

echo "=== All datasets ready ==="
echo ""
echo "Local dataset layout:"
for name in "${!DATASETS[@]}"; do
    count=$(find "$ROOT_DIR/$name" -name "*.parquet" 2>/dev/null | wc -l)
    echo "  $ROOT_DIR/$name/  ($count parquet files)"
done

