import os
import torch
from torch.utils.cpp_extension import load

_src_path = os.path.join(os.path.dirname(__file__), "src", "stem_gather.cpp")

stem_gather = load(
    name="stem_gather_ext",
    sources=[_src_path],
    # extra_cflags=["-O3"],
    # If you really want OpenMP, add:
    extra_cflags=["-O3", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    with_cuda=False,
    verbose=False,
)

def stem_gather_cpu(token_ids_cpu, cpu_tbl, stage, inv_stage, unique_out, seen, slot, epoch: int):
    # returns (U, T)
    return stem_gather.stem_gather_cpu(
        token_ids_cpu, cpu_tbl, stage, inv_stage, unique_out, seen, slot, int(epoch)
    )
