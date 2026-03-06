export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="lm1b-dclm100B-midtrain100B-base"
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

python3 apps/main/prepare_init_checkpoint.py \
    --input-dir /checkpoints-fsx/beidchen-sandbox/STEM/logs/lm1b-dclm-base-100B/checkpoints/0000200000 \
    --output-dir /dev/shm/Llama-1B-dclm-base \
    --no-drop-optim \
    --overwrite

# confirm the directory exists
if [ ! -d "/dev/shm/Llama-1B-dclm-base" ]; then
    echo "Error: /dev/shm/Llama-1B-dclm-base directory does not exist"
    exit 1
fi

echo "########################################################"
echo "Data preparation starting"
echo "########################################################"


python3 setup/prepare_hf_dataset_by_source.py \
    --local_dir /dev/shm/data \
    --out_dir /dev/shm/dolmino-mix_shuffled \
    --num_nodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nchunks 8 \
    --group_yaml setup/source_groups_reasoning.yaml

empty_chunks=$(find /dev/shm/dolmino-mix_shuffled -type f -name "*.chunk.*.jsonl" -empty)
if [ -n "${empty_chunks}" ]; then
    echo "ERROR: Found empty chunk files. Aborting before training."
    echo "${empty_chunks}"
    exit 1
fi
echo "Chunk validation passed: no empty chunk files found."

rm -rf /dev/shm/data

echo "########################################################"
echo "Training starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.train \
    config=apps/main/configs/llama3_1B_midfine.yaml \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/dev/shm/Llama-1B-dclm-base \
    checkpoint.continue_training_from_init=true \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/original/tokenizer.model \
    optim.lr=1e-4 \
    optim.weight_decay=0.1 \
    optim.warmup=0 \
    optim.scheduler="linear" \
    optim.lr_min_ratio=0.0 \
    logging.wandb.name=${experiment_name} 