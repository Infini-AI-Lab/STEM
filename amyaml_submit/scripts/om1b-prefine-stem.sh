export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="olmo2-1b-stem-warmup-1_1"
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

python3 -m apps.main.prepare_stem_checkpoint \
    --ckpt-path /checkpoints-fsx/beidchen-sandbox/stem/olmo2-1b-stage1-token1T/ \
    --output-dir /dev/shm/olmo2-1b-1T-stem-init \
    --stem-layers 1 2 3 4 \
    --stem-parallel-size 8 \
    --tokenizer-name huggingface \
    --tokenizer-path /checkpoints-fsx/beidchen-sandbox/stem/olmo2-1b-stage1-token1T/

# confirm the directory exists
if [ ! -d "/dev/shm/olmo2-1b-1T-stem-init" ]; then
    echo "Error: /dev/shm/olmo2-1b-1T-stem-init directory does not exist"
    exit 1
fi

python3 setup/aws_prepare_hf_dataset.py \
    --local_dir /dev/shm/global-shard_01_of_10 \
    --out_dir /dev/shm/dclm-baseline_shuffled \
    --dataset dclm-baseline \
    --num_nodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nchunks 8 

empty_chunks=$(find /dev/shm/dclm-baseline_shuffled -type f -name "*.chunk.*.jsonl" -empty)
if [ -n "${empty_chunks}" ]; then
    echo "ERROR: Found empty chunk files. Aborting before training."
    echo "${empty_chunks}"
    exit 1
fi
echo "Chunk validation passed: no empty chunk files found."

rm -rf /dev/shm/global-shard_01_of_10

echo "########################################################"
echo "Training starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.stem_train \
    config=apps/main/configs/stem_olmo2_1B_prefine.yaml \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/dev/shm/olmo2-1b-1T-stem-init \
    checkpoint.dump.every=100000 \
    checkpoint.dump.keep=2 \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/olmo2-1b-stage1-token1T/ \
    logging.wandb.name=${experiment_name} \
    model.stem_layers=[1,2,3,4] \
    stem_lr=8e-4 \
    stem_weight_decay=0.01 \
    stem_warmup=5000 \
    stem_lr_min_ratio=0.01 \
    eval.validation.max_steps=8000