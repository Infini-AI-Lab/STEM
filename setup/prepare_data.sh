#!/bin/bash

export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

apt-get update 
apt-get install -y zip 
apt install -y zstd 
cd /dev/shm/ 
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 
unzip -q awscliv2.zip 
./aws/install 
cd /code-fsx/beidchen-sandbox/STEM

set -euxo pipefail

TOTAL_JOBS=16
NCHUNKS=2          # chunks per job → 16 × 2 = 32 global chunks
S3_SRC="s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolma3_dolmino_mix-100B-1025"
S3_DST="s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolmino-mix-32chunks"
REGION="us-east-2"
DATASET="dolmino-mix"
K_VALIDATION=10000

for JOB_INDEX in {0..15}; do

    WORK_DIR=/dev/shm/dolmino-mix_shuffled
    mkdir -p "${WORK_DIR}"

    # --- 1. Stream 1/16 of raw data from S3, decompress, write 2 local chunks --
    python3 setup/aws_prepare_hf_dataset.py \
        --s3_uri "${S3_SRC}" \
        --region "${REGION}" \
        --out_dir "${WORK_DIR}" \
        --dataset "${DATASET}" \
        --num_nodes "${TOTAL_JOBS}" \
        --node_rank "${JOB_INDEX}" \
        --nchunks "${NCHUNKS}" \
        --seed 42 \
        --k_validation "${K_VALIDATION}"

    # Produces:
    #   ${WORK_DIR}/${DATASET}.chunk.00.jsonl
    #   ${WORK_DIR}/${DATASET}.chunk.01.jsonl
    #   ${WORK_DIR}/${DATASET}.val.jsonl

    # --- 2. Rename chunk.{00,01} → chunk.{GLOBAL_INDEX} for global uniqueness ---
    CHUNK_OFFSET=$((JOB_INDEX * NCHUNKS))   # job 0 → offset 0, job 1 → offset 2, ...
    for i in $(seq 0 $((NCHUNKS - 1))); do
        LOCAL_NAME=$(printf "${DATASET}.chunk.%02d.jsonl" "${i}")
        GLOBAL_NAME=$(printf "${DATASET}.chunk.%02d.jsonl" "$((CHUNK_OFFSET + i))")
        mv "${WORK_DIR}/${LOCAL_NAME}" "${WORK_DIR}/${GLOBAL_NAME}"
        echo "Renamed ${LOCAL_NAME} -> ${GLOBAL_NAME}"
    done

    # --- 3. Upload chunks to S3 -------------------------------------------------
    for i in $(seq 0 $((NCHUNKS - 1))); do
        GLOBAL_NAME=$(printf "${DATASET}.chunk.%02d.jsonl" "$((CHUNK_OFFSET + i))")
        echo "Uploading ${GLOBAL_NAME} ..."
        aws s3 cp "${WORK_DIR}/${GLOBAL_NAME}" "${S3_DST%/}/${GLOBAL_NAME}" --region "${REGION}"
    done

    # Upload validation fragment (named by job index so jobs don't overwrite each other)
    VAL_PART_NAME="${DATASET}.val.part$(printf '%02d' "${JOB_INDEX}").jsonl"
    mv "${WORK_DIR}/${DATASET}.val.jsonl" "${WORK_DIR}/${VAL_PART_NAME}"
    echo "Uploading ${VAL_PART_NAME} ..."
    aws s3 cp "${WORK_DIR}/${VAL_PART_NAME}" "${S3_DST%/}/val_parts/${VAL_PART_NAME}" --region "${REGION}"

    # --- 4. Clean up local disk --------------------------------------------------
    rm -rf "${WORK_DIR}"

    echo "=== Job ${JOB_INDEX} complete: uploaded chunks $((CHUNK_OFFSET))..$(( CHUNK_OFFSET + NCHUNKS - 1 )) ==="
done