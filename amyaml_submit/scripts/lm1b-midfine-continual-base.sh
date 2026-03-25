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
echo "Math training starting"
echo "########################################################"

# math training stage
torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.train \
    config=apps/main/configs/llama3_1B_midfine_math.yaml \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/dev/shm/Llama-1B-dclm-base \
    checkpoint.continue_training_from_init=true \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/original/tokenizer.model \
    stage_steps=40000 \
    logging.wandb.name=${stage1_name} 

echo "########################################################"
echo "Post-math Evaluation starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.eval \
    config=apps/main/configs/continual_eval.yaml \
    ckpt_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}/checkpoints/0000040000 \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}-math \
    wandb.project=stem \
    wandb.name=lm1b-midtrain-base-100B-continual-math

echo "########################################################"
echo "Code training starting"
echo "########################################################"

# code training stage
torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.train \
    config=apps/main/configs/llama3_1B_midfine_code.yaml \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/dev/shm/Llama-1B-dclm-base \
    checkpoint.continue_training_from_init=true \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/original/tokenizer.model \
    stage_steps=40000 \
    logging.wandb.name=${stage2_name} 

echo "########################################################"
echo "Post-code Evaluation starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.eval \
    config=apps/main/configs/continual_eval.yaml \
    ckpt_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}/checkpoints/0000080000 \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}-code \
    wandb.project=stem \
    wandb.name=lm1b-midtrain-base-100B-continual-code

echo "########################################################"
echo "Stem training starting"
echo "########################################################"

# stem training stage
torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.train \
    config=apps/main/configs/llama3_1B_midfine_stem.yaml \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name} \
    checkpoint.init_ckpt_path=/dev/shm/Llama-1B-dclm-base \
    checkpoint.continue_training_from_init=true \
    data.tokenizer.path=/checkpoints-fsx/beidchen-sandbox/stem/Llama-3.2-1B/original/tokenizer.model \
    stage_steps=70000 \
    logging.wandb.name=${stage3_name} 

echo "########################################################"
echo "Post-stem Evaluation starting"
echo "########################################################"

torchrun --nproc-per-node=8 --nnodes=${NNODES} -m apps.main.eval \
    config=apps/main/configs/continual_eval.yaml \
    ckpt_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}/checkpoints/0000150000 \
    dump_dir=/checkpoints-fsx/beidchen-sandbox/STEM/logs/${experiment_name}-stem \
    wandb.project=stem \
    wandb.name=lm1b-midtrain-base-100B-continual-stem