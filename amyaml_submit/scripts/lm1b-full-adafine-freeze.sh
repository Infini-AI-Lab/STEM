export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="lm1b-full-adafine-freeze"
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

# Space-efficient data prep: each node streams only its 1/N share from S3
# and writes decompressed chunks directly (no 2x storage needed).
# NOTE: remove or comment out the "aws s3 sync" line in template.yaml when
#       using --s3_uri mode, since this script streams directly from S3.
python setup/aws_prepare_hf_dataset.py \
    --s3_uri s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolma3_dolmino_mix-100B-1125/ \
    --region us-east-2 \
    --out_dir /dev/shm/dolmino-mix_shuffled \
    --dataset dolmino-mix \
    --num_nodes ${NNODES} \
    --nchunks 8 \
    --seed 42

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.stem_adapter_finetune \
    config=apps/main/configs/stem_adapter_finetune_freeze.yaml \
    data.root_dir=/dev/shm \
    cpu_offload_stem_optim=true \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/lm1b-full-adafine-freeze \
    checkpoint.init_ckpt_path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/distcp \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/original/tokenizer.model \
    logging.wandb.name=${experiment_name}
