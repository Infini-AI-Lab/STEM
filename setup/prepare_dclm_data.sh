set -euo pipefail

LOCAL_TMP="/dev/shm/data/dclm_tmp"
S3_URI="s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dclm_baseline_1.0_4prct_raw"

LOCAL_SHARDS=(1 2 3)

command -v huggingface-cli >/dev/null
command -v aws >/dev/null

for sid in "${LOCAL_SHARDS[@]}"; do
  echo "Processing local-shard_${sid}_of_10"

  rm -rf "$LOCAL_TMP"
  mkdir -p "$LOCAL_TMP"

  huggingface-cli download mlfoundations/dclm-baseline-1.0 \
    --repo-type dataset \
    --include "global-shard_01_of_10/local-shard_${sid}_of_10/*.jsonl.zst" \
    --local-dir "$LOCAL_TMP"

  aws s3 sync "$LOCAL_TMP/global-shard_01_of_10/local-shard_${sid}_of_10" "$S3_URI/global-shard_01_of_10/local-shard_${sid}_of_10/" --region us-east-2

  rm -rf "$LOCAL_TMP"
done