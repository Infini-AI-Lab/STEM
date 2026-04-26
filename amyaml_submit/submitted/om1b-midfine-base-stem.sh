export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="olmo2-1b-base-4T-midfine100B-code-stem"
NNODES=4

export TORCHINDUCTOR_CACHE_DIR=/scratch/scratch/beidchen/torchinductor_cache/${HOSTNAME} 
WANDB_DIR=/scratch/scratch/beidchen/projects/stem_wandb 
export WANDB_API_KEY="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"
export WANDB_DIR
export WANDB_MODE=offline
export HF_ALLOW_CODE_EVAL=1

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

aws s3 sync \
  s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolma3_dolmino_mix-100B-1025/ \
  /dev/shm \
  --region us-east-2 \
  --exclude "*" \
  --include "data/*common_crawl-high-quality_*/*" \
  --include "data/*stem-heavy-crawl/*" \
  --include "data/*olmocr_science_pdfs-high_quality-/*"


python3 setup/prepare_hf_dataset_by_source.py \
    --local_dir /dev/shm/data \
    --out_dir /dev/shm/dolmino-mix_shuffled \
    --num_nodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nchunks 8 \
    --group_yaml setup/source_groups_stem.yaml \
    --k_validation 1000

empty_chunks=$(find /dev/shm/dolmino-mix_shuffled -type f -name "*.chunk.*.jsonl" -empty)
if [ -n "${empty_chunks}" ]; then
    echo "ERROR: Found empty chunk files. Aborting before training."
    echo "${empty_chunks}"
    exit 1
fi
echo "Chunk validation passed: no empty chunk files found."

rm -rf /dev/shm/data

hf download Rano23/olmo2-1b-base-token4T --local-dir /dev/shm/olmo2-1b-base-token4T

echo "########################################################"
echo "Training starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.train \
    config=apps/main/configs/olmo2_1B_midfine_stem.yaml \
    dump_dir=/data-fsx/beidchen-sandbox/data/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/data-fsx/beidchen-sandbox/data/logs/olmo2-1b-base-4T-midfine100B-code/checkpoints/0000025000 \
    checkpoint.continue_training_from_init=true \
    checkpoint.dump.keep=2 \
    data.tokenizer.path=/dev/shm/olmo2-1b-base-token4T/ \
    logging.wandb.name=${experiment_name} 