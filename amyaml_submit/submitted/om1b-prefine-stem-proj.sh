export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="olmo2-1b-stem-proj250M-warmup100B"
NNODES=1

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

hf download Rano23/olmo2-1b-stage1-token1T --local-dir /dev/shm/olmo2-1b-stage1-token1T 


# echo "########################################################"
# echo "Projection warmup Training starting"
# echo "########################################################"

# torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.stem_projection_warmup \
#     config=apps/main/configs/stem_olmo2_1B_projection.yaml \
#     dump_dir=/dev/shm/logs/stem_projection_warmup_olmo2_1B \
#     checkpoint.init_ckpt_path=/dev/shm/olmo2-1b-stage1-token1T \
#     data.tokenizer.path=/dev/shm/olmo2-1b-stage1-token1T/ \
#     logging.wandb.name=stem_projection_warmup_olmo2_1B \
#     stem_layers=[1,2,3,4] \
#     steps=10000


# # confirm the directory exists
# if [ ! -d "/dev/shm/logs/stem_projection_warmup_olmo2_1B" ]; then
#     echo "Error: /dev/shm/logs/stem_projection_warmup_olmo2_1B directory does not exist"
#     exit 1
# fi

# mkdir -p /data-fsx/beidchen-sandbox/data/checkpoints

# python3 -m apps.main.prepare_reparam_init_checkpoint  \
#     --base-init-ckpt-path /dev/shm/olmo2-1b-stage1-token1T   \
#     --warmup-ckpt-path /dev/shm/logs/stem_projection_warmup_olmo2_1B/checkpoints/0000010000  \
#     --output-dir /data-fsx/beidchen-sandbox/data/checkpoints/olmo2-1b-reparam-init \
#     --stem-parallel-size 2 

echo "########################################################"
echo "Training starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.stem_reparam_train \
    config=apps/main/configs/stem_olmo3_1B_reparam_stage2.yaml \
    dump_dir=/data-fsx/beidchen-sandbox/data/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/data-fsx/beidchen-sandbox/data/checkpoints/olmo2-1b-reparam-init \
    checkpoint.dump.every=100000 \
    checkpoint.dump.keep=2 \
    data.tokenizer.path=/dev/shm/olmo2-1b-stage1-token1T/ \
    logging.wandb.name=${experiment_name} \
    model.stem_layers=[1,2,3,4] \
    stem_lr=8e-4 \
    proj_lr=8e-4 