# Copyright (c) Meta Platforms, Inc. and affiliates.

from logging import getLogger, Logger
from typing import Callable, Optional, List, Dict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter

logger = getLogger(__name__)

_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_DATA_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_DATA_PARALLEL_RANKS: Optional[List[int]] = None

def divide_and_check_no_remainder(x: int, y: int) -> int:
    assert x % y == 0, f"x ({x}) must be divisible by y ({y})"
    return x // y

def ensure_divisibility(x: int, y: int) -> int:
    assert x % y == 0, f"x ({x}) must be divisible by y ({y})"
    
def initialize_stem_process_group(
    model_parallel_size: int,
    mp_backend: str = "nccl",
    ddp_backend: str = "nccl",
    timeout: Optional[timedelta] = None,
) -> None:
    if not dist.is_initialized():
        raise RuntimeError("Torch distributed is not initialized")
    
    world_size = dist.get_world_size()
    model_parallel_size = int(min(model_parallel_size, world_size))
    rank = dist.get_rank()
    
    data_parallel_size = divide_and_check_no_remainder(world_size, model_parallel_size)
    
    if rank == 0:
        logger.info(f"Initializing stem process group with model parallel size {model_parallel_size} and data parallel size {data_parallel_size}")
        
    groups = torch.LongTensor(range(world_size)).reshape(
        data_parallel_size, model_parallel_size
    )
    
    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]
    
    # build data parallel groups
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_RANKS
    
    for j in range(model_parallel_size):
        ranks = groups[:, j].tolist()
        group = dist.new_group(ranks, backend=ddp_backend, timeout=timeout)
        if j == found[1]:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_RANKS = ranks
        
    global _MODEL_PARALLEL_GROUP
    for i in range(data_parallel_size):
        group = dist.new_group(groups[i].tolist(), backend=mp_backend, timeout=timeout)
        if i == found[0]:
            _MODEL_PARALLEL_GROUP = group
            
    if rank == 0:
        logger.info(f"Initialized stem process group with groups: {groups}")
        

def get_stem_data_parallel_group() -> dist.ProcessGroup:
    if _DATA_PARALLEL_GROUP is None:
        raise RuntimeError("Stem data parallel group is not initialized")
    return _DATA_PARALLEL_GROUP

def get_stem_model_parallel_group() -> dist.ProcessGroup:
    if _MODEL_PARALLEL_GROUP is None:
        raise RuntimeError("Stem model parallel group is not initialized")
    return _MODEL_PARALLEL_GROUP

def get_stem_model_parallel_world_size() -> int:
    return dist.get_world_size(group=get_stem_model_parallel_group())

def get_stem_model_parallel_rank() -> int:
    return dist.get_rank(group=get_stem_model_parallel_group())

def get_stem_data_parallel_rank() -> int:
    return dist.get_rank(group=get_stem_data_parallel_group())


# Low-level communication primitives (similar to mp_utils.py)


def _gather_along_first_dim_stem(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors along first dimension within Stem process group.
    Similar to _gather_along_first_dim in mp_utils.py but uses Stem process group.
    """
    assert x.is_contiguous()
    stem_size = get_stem_model_parallel_world_size()
    if stem_size == 1:
        return x

    # Output shape: first dim multiplied by stem_size
    shape = list(x.shape)
    shape[0] = shape[0] * stem_size

    # Gather using _all_gather_base
    output = torch.empty(shape, dtype=x.dtype, device=torch.cuda.current_device())
    dist._all_gather_base(
        output_tensor=output,
        input_tensor=x,
        group=get_stem_model_parallel_group(),
    )

    return output


def _split_along_first_dim_stem(x: torch.Tensor) -> torch.Tensor:
    """
    Split tensor along first dimension and keep the slice for this Stem rank.
    Similar to _split_along_first_dim in mp_utils.py.
    """
    assert x.is_contiguous()
    stem_size = get_stem_model_parallel_world_size()
    if stem_size == 1:
        return x

    # Split based on Stem rank
    dim_size = x.size(0)
    assert (
        dim_size % stem_size == 0
    ), f"dim_size={dim_size} not divisible by stem_size={stem_size}"
    local_dim = dim_size // stem_size
    rank = get_stem_model_parallel_rank()
    offset = rank * local_dim

    # Split and return this rank's slice
    output = x[offset : offset + local_dim].contiguous()
    assert output.size(0) == local_dim

    return output


def _gather_along_last_dim_stem(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors along last dimension within Stem process group.
    Similar to _gather_along_last_dim in mp_utils.py.
    """
    assert x.is_contiguous()
    stem_size = get_stem_model_parallel_world_size()
    if stem_size == 1:
        return x

    rank = get_stem_model_parallel_rank()
    group = get_stem_model_parallel_group()

    # Gather from all ranks
    tensor_list = [torch.empty_like(x) for _ in range(stem_size)]
    tensor_list[rank] = x
    dist.all_gather(tensor_list, x, group=group)

    # Concatenate along last dimension
    output = torch.cat(tensor_list, dim=-1)

    return output


def _split_along_last_dim_stem(x: torch.Tensor) -> torch.Tensor:
    """
    Split tensor along last dimension and keep the slice for this Stem rank.
    Similar to _split_along_last_dim in mp_utils.py.
    """
    assert x.is_contiguous()
    stem_size = get_stem_model_parallel_world_size()
    if stem_size == 1:
        return x

    # Split based on Stem rank
    dim_size = x.size(-1)
    assert (
        dim_size % stem_size == 0
    ), f"dim_size={dim_size} not divisible by stem_size={stem_size}"
    local_dim = dim_size // stem_size
    rank = get_stem_model_parallel_rank()
    offset = rank * local_dim

    # Split and return this rank's slice
    output = x[..., offset : offset + local_dim].contiguous()
    assert output.size(-1) == local_dim

    return output


def _reduce_scatter_along_first_dim_stem(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce-scatter along first dimension within Stem process group.
    Similar to _reduce_scatter_along_first_dim in mp_utils.py.
    """
    assert x.is_contiguous()    
    stem_size = get_stem_model_parallel_world_size()
    if stem_size == 1:
        return x

    assert x.size(0) % stem_size == 0

    # Output shape: first dim divided by stem_size
    shape = list(x.shape)
    shape[0] = shape[0] // stem_size

    # Reduce scatter
    output = torch.empty(shape, dtype=x.dtype, device=torch.cuda.current_device())
    dist._reduce_scatter_base(
        output=output,
        input=x,
        op=dist.ReduceOp.SUM,
        group=get_stem_model_parallel_group(),
    )

    return output


def _all_to_all_stem(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    rescale: bool = False,
) -> torch.Tensor:
    stem_size = get_stem_model_parallel_world_size()
    if stem_size == 1:
        return input_

    # Ensure dimensions are positive indices
    scatter_dim = scatter_dim if scatter_dim >= 0 else len(input_.shape) + scatter_dim
    gather_dim = gather_dim if gather_dim >= 0 else len(input_.shape) + gather_dim

    # Split input along scatter dimension
    input_list = [
        _chunk.contiguous() for _chunk in torch.chunk(input_, stem_size, dim=scatter_dim)
    ]

    # Output list to receive scattered chunks
    output_list = [torch.empty_like(input_list[0]) for _ in range(stem_size)]

    # Perform all-to-all
    dist.all_to_all(output_list, input_list, group=get_stem_model_parallel_group())

    # Concatenate along gather dimension
    output = torch.cat(output_list, dim=gather_dim).contiguous()
    if rescale:
        output.div_(stem_size)

    return output

# Autograd Functions for Stem operations


class _GatherTokensForStem(torch.autograd.Function):
    """
    Gather tokens from all GPUs in Stem group along batch dimension.
    Forward: gather along first dim
    Backward: reduce-scatter gradients along first dim
    """

    @staticmethod
    def forward(ctx, tokens: torch.Tensor) -> torch.Tensor:
        return _gather_along_first_dim_stem(tokens)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Gradients need to be reduced and scattered back
        return _reduce_scatter_along_first_dim_stem(grad_output)


class _ScatterTokensFromStem(torch.autograd.Function):
    """
    Scatter tokens back to original GPUs along batch dimension.
    Forward: split along first dim
    Backward: gather gradients along first dim
    """

    @staticmethod
    def forward(ctx, tokens: torch.Tensor) -> torch.Tensor:
        return _split_along_first_dim_stem(tokens)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Gradients need to be gathered from all ranks
        return _gather_along_first_dim_stem(grad_output)


class _GatherEmbeddingsForStem(torch.autograd.Function):
    """
    Gather embedding shards from all GPUs in Stem group along hidden dimension.
    Forward: gather along last dim
    Backward: split gradients along last dim
    """

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor) -> torch.Tensor:
        return _gather_along_last_dim_stem(embeddings)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Each rank only needs gradients for its shard
        return _split_along_last_dim_stem(grad_output)


class _ScatterEmbeddingsFromStem(torch.autograd.Function):
    """
    Scatter embeddings back to original GPUs along batch dimension.
    Forward: split along first dim
    Backward: gather gradients along first dim
    """

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor) -> torch.Tensor:
        return _split_along_first_dim_stem(embeddings)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _gather_along_first_dim_stem(grad_output)


class _AllToAllForStem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings: torch.Tensor) -> torch.Tensor:
        return _all_to_all_stem(embeddings, scatter_dim=0, gather_dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _all_to_all_stem(grad_output, scatter_dim=-1, gather_dim=0, rescale=True)


# High-level API functions


def gather_tokens_for_stem(tokens: torch.Tensor) -> torch.Tensor:
    """
    Gather tokens from all GPUs within the node for Stem embedding lookup.

    Args:
        tokens: [batch_size_local, seq_len]

    Returns:
        gathered_tokens: [batch_size_total, seq_len] where
                        batch_size_total = batch_size_local * stem_world_size
    """
    return _GatherTokensForStem.apply(tokens)


def gather_embeddings_for_stem(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Gather embedding shards from all GPUs to reconstruct full embeddings.

    Args:
        embeddings: [batch_size, seq_len, hidden_dim / stem_world_size]

    Returns:
        full_embeddings: [batch_size, seq_len, hidden_dim]
    """
    return _GatherEmbeddingsForStem.apply(embeddings)


def scatter_embeddings_for_stem(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Scatter full embeddings back to original GPUs along batch dimension.

    Args:
        embeddings: [batch_size_total, seq_len, hidden_dim]

    Returns:
        local_embeddings: [batch_size_local, seq_len, hidden_dim]
    """
    return _ScatterEmbeddingsFromStem.apply(embeddings)


def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    stride: int = 1,
    return_master_weight: bool = False,
) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # if no_init, skip the initialization.
    if torch.equal(init_method(torch.zeros_like(weight)), torch.zeros_like(weight)):
        return None

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_stem_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight on the same device as weight
    master_weight = torch.empty(
        out_features, in_features, dtype=weight.dtype, device=weight.device, requires_grad=False
    )
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide_and_check_no_remainder(
        per_partition_size, stride
    )
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_stem_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class ParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[
            [torch.Tensor], torch.Tensor
        ] = torch.nn.init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # pyre-fixme[4]: Attribute must be annotated.
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        world_size = get_stem_model_parallel_world_size()
        # pyre-fixme[4]: Attribute must be annotated.
        self.embedding_dim_per_partition = divide_and_check_no_remainder(
            self.embedding_dim, world_size
        )

        # Allocate weights on the specified device (or default if not specified)
        self.weight = Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim_per_partition, device=device)
        )
        # And initialize.
        self.init_method = init_method
        self.reset_parameters()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_parallel = gather_tokens_for_stem(input_)
        output_parallel = F.embedding(
            input_parallel,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # output = scatter_embeddings_for_stem(gather_embeddings_for_stem(output_parallel))
        output = _AllToAllForStem.apply(output_parallel)

        return output
    
    def reset_parameters(self):
        if self.weight.device.type == "meta":
            return
        # Ensure weight is actually on a real device before initializing
        if not self.weight.is_cuda and self.weight.device.type != "cpu":
            return
        
        try:
            _initialize_affine_weight(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.embedding_dim_per_partition,
                1,
                self.init_method,
                stride=1,
                return_master_weight=False,
            )
            # Verify initialization succeeded
            if self.weight.numel() > 0 and self.weight.abs().max() == 0:
                # Fallback: if initialization resulted in zeros, use direct initialization
                import logging
                logger = logging.getLogger()
                logger.warning(
                    f"ParallelEmbedding initialization resulted in zeros, using fallback initialization"
                )
                with torch.no_grad():
                    self.init_method(self.weight)
        except Exception as e:
            # Fallback: if model parallel initialization fails, use direct initialization
            import logging
            logger = logging.getLogger()
            logger.warning(
                f"ParallelEmbedding model parallel initialization failed: {e}, using fallback initialization"
            )
            with torch.no_grad():
                self.init_method(self.weight)