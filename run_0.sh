export WANDB_DIR=/raid/user_data/rsadhukh/wandb 
export WANDB_API_KEY="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"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1,2

# python3 apps/main/prepare_stem_checkpoint.py \
#     --ckpt-path checkpoints/Llama-3.2-1B/distcp \
#     --output-dir /raid/user_data/rsadhukh/checkpoints/Llama-3.2-1B-stem-init \
#     --stem-layers 2 6 10 14 \
#     --stem-parallel-size 2 

# torchrun --nnodes=2 --nproc-per-node=2 --node_rank=0 \
#     --master_addr=127.0.0.1 --master_port=29512 \
# torchrun  --nproc-per-node=2 
python3 -m apps.main.train \
    config=apps/main/configs/llama3_1B_0.yaml \
    profiling.run=false \
    data.batch_size=2