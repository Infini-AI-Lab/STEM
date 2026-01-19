import contextlib
import time

import torch

import argparse
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import TensorDataset
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from Engine.backend import LMBackend
from Engine.utils import setup_seed


def convert_pg19_dataset(tokenizer, seq_len=4096, nitem=100):
    dataset = load_dataset("tests/Data/pg19/", split="train")
    tokenized_prompts = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            max_length=seq_len,
            padding=True,
            truncation=True,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_prompts.set_format(type="torch", columns=["input_ids"])
    data = torch.stack([x["input_ids"] for x in tokenized_prompts]*5, dim=0)
    return TensorDataset(data)

parser = argparse.ArgumentParser(
    description="Process model configuration and partitions."
)
parser.add_argument(
    "--model",
    type=Path,
    default=Path("/home/rsadhukh/STEM/inference/checkpoints/meta-llama/Llama-3.2-1B/model.pth"),
    help="model",
)
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="model name"
)

parser.add_argument("--B", type=int, default=4, help="Batch size.")
parser.add_argument("--prefix_len", type=int, default=8065, help="Prefix length")
parser.add_argument("--max_len", type=int, default=8192, help="Generate length")

parser.add_argument("--seed", type=int, default=123, help="Random seed.")

parser.add_argument(
    "--compile", action="store_true", help="Whether to compile the model."
)
parser.add_argument(
    "--compile_prefill", action="store_true", help="Whether to compile the prefill."
)
parser.add_argument("--rank_group", nargs="+", type=int, default=[-1], help="Target group of ranks")
parser.add_argument(
    "--printoutput", action="store_true", help="Whether to compile the model."
)
parser.add_argument(
    "--profile", action="store_true", help="Whether to profile the model."
)
parser.add_argument(
    "--num_eval_steps", type=int, default=15, help="Number of evaluation steps."
)

args = parser.parse_args()
assert args.prefix_len < args.max_len
assert args.max_len % 128 == 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
global print

use_tp = len(args.rank_group) > 1
assert not use_tp, "TP is not supported yet"
global_group = None
rank = -1

setup_seed(args.seed)
print(f"Using device={DEVICE}")

MAX_LEN = args.max_len
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
checkpoint_path = args.model
model_name = args.model_name

engine = LMBackend(dtype=DTYPE, device=DEVICE)
engine.load_model(
    checkpoint_path,
    model_name,
    use_tp=use_tp,
    rank_group=args.rank_group,
    group=global_group,
)
if args.compile:
    engine.compile(compile_prefill=args.compile_prefill)
    
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)
tokenizer = AutoTokenizer.from_pretrained(args.model.parent, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

dataset = convert_pg19_dataset(
    tokenizer=tokenizer, seq_len=args.prefix_len, nitem=BATCH_SIZE * args.num_eval_steps
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(args.num_eval_steps, len(dataloader))

total_time = 0.0
prefill_time = 0.0
model_steps = 0
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps-1):
    if step >= num_eval_steps-1:
        break
    input_ids = batch[0].to(DEVICE)
    terminate = False
    output = input_ids.clone()

    start_event.record()
    next_tokens = engine.encode(input_ids=input_ids)[:, -1:]
    end_event.record()
    end_event.synchronize()
    current_prefill_time = start_event.elapsed_time(end_event)
    prefill_time += current_prefill_time
    model_steps += 1
    print(f"Prefill latency: {current_prefill_time} ms")
    if step < 5:
        prefill_time = 0.0
        model_steps = 0
    if use_tp:
        dist.barrier()
        
    """
    output = torch.cat((output, next_tokens), dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    while output.size(1) < MAX_LEN and not terminate:
        input_ids = next_tokens.clone()
        next_tokens = engine.inference(input_ids=input_ids)
        output = torch.cat((output, next_tokens), dim=-1)
        model_steps += 1
        if (next_tokens[:, -1] == eot_1)._is_any_true() or (
            next_tokens[:, -1] == eot_2
        )._is_any_true():
            terminate = True
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    total_time += t2 - t1
    
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, args.prefix_len :]))
    print(f"Tokens per second :{BATCH_SIZE*(model_steps/total_time)}")
    if step < 10:
        total_time = 0.0
        model_steps = 0
    if use_tp:
        dist.barrier()
        
print(f"Final tokens per second :{BATCH_SIZE*(model_steps/total_time)}")
    """
    
print(f"Average prefill latency: {prefill_time/model_steps} ms")

# profile
batch = next(iter(dataloader))
input_ids = batch[0].to(DEVICE)

torch.cuda.synchronize()
if rank <= 0:
    torch.profiler._utils._init_for_cuda_graphs()
    prof = torch.profiler.profile()
else:
    prof = contextlib.nullcontext()
    
with prof:
    next_tokens = engine.encode(input_ids=input_ids)[:, -1:]
    
torch.cuda.synchronize()
if rank <= 0:
    prof.export_chrome_trace("baseline_profile.json")
    
if use_tp:
    dist.barrier()
