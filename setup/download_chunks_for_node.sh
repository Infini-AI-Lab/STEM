#!/bin/bash
# ---------------------------------------------------------------------------
# download_chunks_for_node.sh
#
# Downloads the chunk slice assigned to this node from S3 into a local dir.
# Designed to be sourced or called from a training launch script.
#
# With 32 total chunks and 4 nodes:
#   Node 0 → chunk.00 .. chunk.07
#   Node 1 → chunk.08 .. chunk.15
#   Node 2 → chunk.16 .. chunk.23
#   Node 3 → chunk.24 .. chunk.31
#
# Usage:
#   bash setup/download_chunks_for_node.sh <TOTAL_CHUNKS> <NNODES> \
#       <S3_DATA> <LOCAL_DIR> [REGION] [DATASET]
#
# The NODE_RANK is auto-detected from $HOSTNAME (K8s pod name ending in
# -worker-<N>), or can be overridden with the NODE_RANK env var.
# ---------------------------------------------------------------------------
set -euo pipefail

TOTAL_CHUNKS=${1:?"Usage: $0 <TOTAL_CHUNKS> <NNODES> <S3_DATA> <LOCAL_DIR> [REGION] [DATASET]"}
NNODES=${2:?"Usage: $0 <TOTAL_CHUNKS> <NNODES> <S3_DATA> <LOCAL_DIR> [REGION] [DATASET]"}
S3_DATA=${3:?"Usage: $0 <TOTAL_CHUNKS> <NNODES> <S3_DATA> <LOCAL_DIR> [REGION] [DATASET]"}
LOCAL_DIR=${4:?"Usage: $0 <TOTAL_CHUNKS> <NNODES> <S3_DATA> <LOCAL_DIR> [REGION] [DATASET]"}
REGION=${5:-us-east-2}
DATASET=${6:-dolmino-mix}

# Auto-detect node rank from K8s hostname (e.g. "job-worker-2" → 2)
if [ -z "${NODE_RANK:-}" ]; then
    NODE_RANK="${HOSTNAME##*-}"
fi

CHUNKS_PER_NODE=$((TOTAL_CHUNKS / NNODES))
START=$((NODE_RANK * CHUNKS_PER_NODE))
END=$((START + CHUNKS_PER_NODE - 1))

echo "============================================================"
echo "  Node ${NODE_RANK} / ${NNODES}"
echo "  Downloading chunks ${START}..${END} (${CHUNKS_PER_NODE} chunks)"
echo "  From: ${S3_DATA}"
echo "  To  : ${LOCAL_DIR}"
echo "============================================================"

mkdir -p "${LOCAL_DIR}"

for i in $(seq "${START}" "${END}"); do
    CHUNK=$(printf "${DATASET}.chunk.%02d.jsonl" "${i}")
    echo "  Downloading ${CHUNK} ..."
    aws s3 cp "${S3_DATA%/}/${CHUNK}" "${LOCAL_DIR}/${CHUNK}" --region "${REGION}"
done

# Download validation file (small, needed by every node)
echo "  Downloading ${DATASET}.val.jsonl ..."
aws s3 cp "${S3_DATA%/}/${DATASET}.val.jsonl" "${LOCAL_DIR}/${DATASET}.val.jsonl" --region "${REGION}" || \
    echo "  (no val.jsonl found, skipping)"

echo "=== Node ${NODE_RANK}: download complete ==="
ls -lh "${LOCAL_DIR}"

