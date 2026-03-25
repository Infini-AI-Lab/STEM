export PYTHONPATH=/code-fsx/beidchen-sandbox/STEM:$PYTHONPATH

set -x

project_name="stem"
experiment_name="lm1b-midtrain-base-100B-continual"
stage1_name="lm1b-midtrain-base-100B-math"
stage2_name="lm1b-midtrain-base-100B-code"
stage3_name="lm1b-midtrain-base-100B-stem"
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

export HF_ALLOW_CODE_EVAL=1

NODE_RANK=${HOSTNAME##*-}
echo "NODE_RANK: $NODE_RANK"
echo "WANDB_MODE: $WANDB_MODE"

echo "########################################################"
echo "Evaluation starting"
echo "########################################################"

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

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.eval \
    config=apps/main/configs/continual_eval.yaml \
    ckpt_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/lm1b-dclm-base-100B/checkpoints/0000200000 \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}-init \
    wandb.project=stem \
    wandb.name=lm1b-midtrain-base-100B-continual-init