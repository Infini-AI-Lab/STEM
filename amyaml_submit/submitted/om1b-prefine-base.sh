export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="olmo2-1b-base-warmup100B-1_1"
NNODES=4

export TORCHINDUCTOR_CACHE_DIR=/scratch/scratch/beidchen/torchinductor_cache/${HOSTNAME} 
WANDB_DIR=/scratch/scratch/beidchen/projects/stem_wandb 
export WANDB_API_KEY="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"
export WANDB_DIR
export WANDB_MODE=offline

echo "$HOSTNAME $(hostname -I)"
echo "$HOSTNAME $(hostname -I | awk '{print $2}')"

if [ "${HOSTNAME##*-}" -eq 0 ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi


NODE_RANK=${HOSTNAME##*-}
echo "NODE_RANK: $NODE_RANK"
echo "WANDB_MODE: $WANDB_MODE"

S3_GLOBAL_SHARD_URI="s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dclm_baseline_1.0_4prct_raw/global-shard_01_of_10"
LOCAL_S3_SHARD_NAME="local-shard_${NODE_RANK}_of_10"
LOCAL_S3_SHARD_URI="${S3_GLOBAL_SHARD_URI}/${LOCAL_S3_SHARD_NAME}/"
LOCAL_RAW_DIR="/dev/shm/${LOCAL_S3_SHARD_NAME}"
LOCAL_PREPARED_DIR="/dev/shm/dclm-baseline_shuffled"

if [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "Error: NODE_RANK (${NODE_RANK}) must be < NNODES (${NNODES})"
    exit 1
fi

echo "Syncing node-local shard from ${LOCAL_S3_SHARD_URI}"
rm -rf "${LOCAL_RAW_DIR}" "${LOCAL_PREPARED_DIR}"
cmd="aws s3 sync ${LOCAL_S3_SHARD_URI} ${LOCAL_RAW_DIR} --region us-east-2 --only-show-errors"
echo "Running: ${cmd}"
eval ${cmd}

python3 setup/aws_prepare_hf_dataset.py \
    --local_dir "${LOCAL_RAW_DIR}" \
    --out_dir "${LOCAL_PREPARED_DIR}" \
    --dataset dclm-baseline \
    --num_nodes 1 \
    --node_rank 0 \
    --nchunks 8 

empty_chunks=$(find /dev/shm/dclm-baseline_shuffled -type f -name "*.chunk.*.jsonl" -empty)
if [ -n "${empty_chunks}" ]; then
    echo "ERROR: Found empty chunk files. Aborting before training."
    echo "${empty_chunks}"
    exit 1
fi
echo "Chunk validation passed: no empty chunk files found."

rm -rf "${LOCAL_RAW_DIR}"

echo "########################################################"
echo "Training starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.train \
    config=apps/main/configs/olmo2_1B_prefine.yaml \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/checkpoints-fsx/beidchen-sandbox/stem/olmo2-1b-stage1-token1T/ \
    checkpoint.dump.every=100000 \
    checkpoint.dump.keep=2 \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/olmo2-1b-stage1-token1T/ \
    logging.wandb.name=${experiment_name} 