#!/bin/bash
# ---------------------------------------------------------------------------
# preprocess_one_chunk.sh
#
# Processes 1/N of the raw .jsonl.zst data from S3, producing a single
# shuffled .jsonl chunk, then uploads it back to S3.
#
# Disk usage is minimal: only the *streaming decompression buffer* and the
# single output chunk file exist at any time — no need to store the full
# dataset.
#
# Usage:
#   bash setup/preprocess_one_chunk.sh <JOB_INDEX> <TOTAL_JOBS> \
#       <S3_SRC> <S3_DST> [REGION] [DATASET] [K_VALIDATION]
#
# Example (job 5 of 32):
#   bash setup/preprocess_one_chunk.sh 5 32 \
#       s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolma3_dolmino_mix-100B-1025 \
#       s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolmino-mix-32chunks \
#       us-east-2 dolmino-mix 10000
# ---------------------------------------------------------------------------
set -euo pipefail

JOB_INDEX=${1:?"Usage: $0 <JOB_INDEX> <TOTAL_JOBS> <S3_SRC> <S3_DST> [REGION] [DATASET] [K_VALIDATION]"}
TOTAL_JOBS=${2:?"Usage: $0 <JOB_INDEX> <TOTAL_JOBS> <S3_SRC> <S3_DST> [REGION] [DATASET] [K_VALIDATION]"}
S3_SRC=${3:?"Usage: $0 <JOB_INDEX> <TOTAL_JOBS> <S3_SRC> <S3_DST> [REGION] [DATASET] [K_VALIDATION]"}
S3_DST=${4:?"Usage: $0 <JOB_INDEX> <TOTAL_JOBS> <S3_SRC> <S3_DST> [REGION] [DATASET] [K_VALIDATION]"}
REGION=${5:-us-east-2}
DATASET=${6:-dolmino-mix}
K_VALIDATION=${7:-10000}

WORK_DIR=$(mktemp -d "/tmp/preprocess_chunk_${JOB_INDEX}_XXXXXX")
trap "rm -rf '${WORK_DIR}'" EXIT

echo "============================================================"
echo "  Job ${JOB_INDEX} / ${TOTAL_JOBS}"
echo "  Source : ${S3_SRC}"
echo "  Dest   : ${S3_DST}"
echo "  Workdir: ${WORK_DIR}"
echo "============================================================"

# --- 1. Stream from S3, decompress, write one local chunk -----------------
python3 setup/aws_prepare_hf_dataset.py \
    --s3_uri "${S3_SRC}" \
    --region "${REGION}" \
    --out_dir "${WORK_DIR}" \
    --dataset "${DATASET}" \
    --num_nodes "${TOTAL_JOBS}" \
    --node_rank "${JOB_INDEX}" \
    --nchunks 1 \
    --seed 42 \
    --k_validation "${K_VALIDATION}"

# The script produces:
#   ${WORK_DIR}/${DATASET}.chunk.00.jsonl   (the data chunk)
#   ${WORK_DIR}/${DATASET}.val.jsonl        (validation carved from this chunk)

# --- 2. Rename chunk.00 → chunk.<JOB_INDEX> for global uniqueness ---------
CHUNK_NAME=$(printf "${DATASET}.chunk.%02d.jsonl" "${JOB_INDEX}")
mv "${WORK_DIR}/${DATASET}.chunk.00.jsonl" "${WORK_DIR}/${CHUNK_NAME}"

echo "Renamed chunk.00 -> ${CHUNK_NAME}"

# --- 3. Upload chunk to S3 -----------------------------------------------
echo "Uploading ${CHUNK_NAME} ..."
aws s3 cp "${WORK_DIR}/${CHUNK_NAME}" "${S3_DST%/}/${CHUNK_NAME}" --region "${REGION}"

# Upload validation piece (named by job index so jobs don't overwrite each other)
VAL_PART_NAME="${DATASET}.val.part$(printf '%02d' "${JOB_INDEX}").jsonl"
mv "${WORK_DIR}/${DATASET}.val.jsonl" "${WORK_DIR}/${VAL_PART_NAME}"
echo "Uploading ${VAL_PART_NAME} ..."
aws s3 cp "${WORK_DIR}/${VAL_PART_NAME}" "${S3_DST%/}/val_parts/${VAL_PART_NAME}" --region "${REGION}"

echo "=== Job ${JOB_INDEX} complete ==="

