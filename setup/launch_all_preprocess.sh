#!/bin/bash
# ---------------------------------------------------------------------------
# launch_all_preprocess.sh
#
# Launches 32 preprocessing jobs.  Each job streams 1/32 of the raw
# compressed data from S3, produces one shuffled chunk, and uploads it
# back to S3.  Disk usage per job is just 1 chunk (~10 GB for 100B tokens).
#
# Run modes
# ---------
# 1. SEQUENTIAL (safest, minimal disk):
#      bash setup/launch_all_preprocess.sh
#
# 2. PARALLEL (N at a time — uses N× disk):
#      PARALLEL=8 bash setup/launch_all_preprocess.sh
#
# After all 32 jobs finish, this script also merges the 32 validation
# fragments into a single val.jsonl and uploads it.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Configuration (edit these) ------------------------------------------
TOTAL_JOBS=32
S3_SRC="s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolma3_dolmino_mix-100B-1025"
S3_DST="s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolmino-mix-32chunks"
REGION="us-east-2"
DATASET="dolmino-mix"
K_VALIDATION=10000

# How many jobs to run concurrently (default: 1 = sequential)
PARALLEL=${PARALLEL:-1}
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Launching ${TOTAL_JOBS} preprocessing jobs"
echo "  Parallelism : ${PARALLEL}"
echo "  S3 source   : ${S3_SRC}"
echo "  S3 dest     : ${S3_DST}"
echo "============================================================"

running=0
for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    echo ""
    echo ">>> Launching job ${i}/${TOTAL_JOBS} ..."
    bash "${SCRIPT_DIR}/preprocess_one_chunk.sh" \
        "${i}" "${TOTAL_JOBS}" "${S3_SRC}" "${S3_DST}" \
        "${REGION}" "${DATASET}" "${K_VALIDATION}" &

    running=$((running + 1))

    # Throttle: wait when we hit the parallelism limit
    if [ "${running}" -ge "${PARALLEL}" ]; then
        wait -n  # wait for any one background job to finish
        running=$((running - 1))
    fi
done

# Wait for all remaining jobs
wait
echo ""
echo "=== All ${TOTAL_JOBS} preprocessing jobs complete ==="

# ---- Merge validation fragments into a single val.jsonl ------------------
echo ""
echo "Merging validation fragments ..."
VAL_MERGE_DIR=$(mktemp -d "/tmp/val_merge_XXXXXX")
trap "rm -rf '${VAL_MERGE_DIR}'" EXIT

aws s3 sync "${S3_DST%/}/val_parts/" "${VAL_MERGE_DIR}/" --region "${REGION}"

# Concatenate all parts in order
VAL_MERGED="${VAL_MERGE_DIR}/${DATASET}.val.jsonl"
for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    PART=$(printf "${DATASET}.val.part%02d.jsonl" "${i}")
    if [ -f "${VAL_MERGE_DIR}/${PART}" ]; then
        cat "${VAL_MERGE_DIR}/${PART}" >> "${VAL_MERGED}"
    fi
done

echo "Uploading merged ${DATASET}.val.jsonl ..."
aws s3 cp "${VAL_MERGED}" "${S3_DST%/}/${DATASET}.val.jsonl" --region "${REGION}"

echo ""
echo "============================================================"
echo "  Done!  Processed data is at:"
echo "    ${S3_DST}"
echo ""
echo "  Files:"
echo "    ${DATASET}.chunk.00.jsonl  ..  ${DATASET}.chunk.31.jsonl"
echo "    ${DATASET}.val.jsonl"
echo "============================================================"

