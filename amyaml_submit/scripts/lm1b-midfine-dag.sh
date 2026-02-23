export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="lm1b-midfine-warmup-s2l4-100B"
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

# Space-efficient data prep: each node streams only its 1/N share from S3
# and writes decompressed chunks directly (no 2x storage needed).
# NOTE: remove or comment out the "aws s3 sync" line in template.yaml when
#       using --s3_uri mode, since this script streams directly from S3.
python3 setup/aws_prepare_hf_dataset.py \
    --local_dir /dev/shm/data \
    --out_dir /dev/shm/dolmino-mix_shuffled \
    --dataset dolmino-mix \
    --num_nodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nchunks 8 

rm -rf /dev/shm/data

python3 apps/main/prepare_dag_stem_checkpoint.py \
    --ckpt-path /checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/distcp \
    --output-dir /dev/shm/Llama-1B-dag-stem-init \
    --stem-layers 2 6 10 14 \
    --stem-parallel-size 8 \
    --alpha-init -5.0

# confirm the directory exists
if [ ! -d "/dev/shm/Llama-1B-dag-stem-init" ]; then
    echo "Error: /dev/shm/Llama-1B-dag-stem-init directory does not exist"
    exit 1
fi

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.stem_dag_train \
    config=apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
    data.root_dir=/dev/shm \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/dev/shm/Llama-1B-dag-stem-init \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/original/tokenizer.model \
    logging.wandb.name=${experiment_name} \
    model.stem_layers=[2,6,10,14] \
    stem_lr=1e-3 \
    stem_weight_decay=0.0