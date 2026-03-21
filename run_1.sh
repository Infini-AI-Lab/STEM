export WANDB_DIR=/raid/user_data/rsadhukh/wandb 
export WANDB_API_KEY="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"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=4,7

torchrun --nnodes=2 --nproc-per-node=2 --node_rank=1 \
    --master_addr=127.0.0.1 --master_port=29512 \
    -m apps.main.train \
    config=apps/main/configs/llama3_1B_1.yaml \
    profiling.run=false \
    data.batch_size=2