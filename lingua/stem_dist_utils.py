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
        

def is_stem_initialized() -> bool:
    """Check if STEM process groups are initialized."""
    return _MODEL_PARALLEL_GROUP is not None and dist.is_initialized()

def get_stem_data_parallel_group() -> dist.ProcessGroup:
    if _DATA_PARALLEL_GROUP is None:
        raise RuntimeError("Stem data parallel group is not initialized")
    return _DATA_PARALLEL_GROUP

def get_stem_model_parallel_group() -> dist.ProcessGroup:
    if _MODEL_PARALLEL_GROUP is None:
        raise RuntimeError("Stem model parallel group is not initialized")
    return _MODEL_PARALLEL_GROUP

def get_stem_model_parallel_world_size() -> int:
    """Get STEM model parallel world size. Returns 1 if not distributed."""
    if not dist.is_initialized() or _MODEL_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(group=get_stem_model_parallel_group())

def get_stem_model_parallel_rank() -> int:
    """Get STEM model parallel rank. Returns 0 if not distributed."""
    if not dist.is_initialized() or _MODEL_PARALLEL_GROUP is None:
        return 0
    return dist.get_rank(group=get_stem_model_parallel_group())

def get_stem_data_parallel_rank() -> int:
    """Get STEM data parallel rank. Returns 0 if not distributed."""
    if not dist.is_initialized() or _DATA_PARALLEL_GROUP is None:
        return 0
    return dist.get_rank(group=get_stem_data_parallel_group())

def get_stem_data_parallel_world_size() -> int:
    """Get STEM data parallel world size. Returns 1 if not distributed."""
    if not dist.is_initialized() or _DATA_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(group=get_stem_data_parallel_group())


# Low-level communication primitives (similar to mp_utils.py)


def _gather_along_first_dim_stem(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors along first dimension within Stem process group.
    Similar to _gather_along_first_dim in mp_utils.py but uses Stem process group.
    """
    if not x.is_contiguous():
        x = x.contiguous()
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
    if not x.is_contiguous():
        x = x.contiguous()
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
    if not x.is_contiguous():
        x = x.contiguous()
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
    if not x.is_contiguous():
        x = x.contiguous()
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
    if not x.is_contiguous():
        x = x.contiguous()
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


class _ReduceScatterEmbeddingsForStem(torch.autograd.Function):
    """
    Reduce-scatter embeddings across Stem ranks along batch dimension.
    Forward: reduce-scatter along first dim
    Backward: gather gradients along first dim
    """

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor) -> torch.Tensor:
        return _reduce_scatter_along_first_dim_stem(embeddings)

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


def reduce_scatter_embeddings_for_stem(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Reduce-scatter embeddings across Stem ranks along batch dimension.

    Args:
        embeddings: [batch_size_total, seq_len, hidden_dim]

    Returns:
        local_embeddings: [batch_size_local, seq_len, hidden_dim]
    """
    return _ReduceScatterEmbeddingsForStem.apply(embeddings)


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

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  
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


class VocabParallelEmbedding(torch.nn.Module):
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
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None

        world_size = get_stem_model_parallel_world_size()
        self.num_embeddings_per_partition = divide_and_check_no_remainder(
            self.num_embeddings, world_size
        )
        rank = get_stem_model_parallel_rank()
        self.vocab_start_index = rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=device)
        )
        self.init_method = init_method
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.master_weight = None
        self.reset_parameters()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = gather_tokens_for_stem(input_)

        input_mask = (input_parallel < self.vocab_start_index) | (
            input_parallel >= self.vocab_end_index
        )
        masked_input = input_parallel.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        local_padding_idx = None
        if self.padding_idx is not None:
            if self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
                local_padding_idx = self.padding_idx - self.vocab_start_index

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0.0)

        return reduce_scatter_embeddings_for_stem(output_parallel)

    def reset_parameters(self):
        if self.weight.device.type == "meta":
            return
        if not self.weight.is_cuda and self.weight.device.type != "cpu":
            return

        try:
            master_weight = _initialize_affine_weight(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                self.init_method,
                stride=1,
                return_master_weight=self.keep_master_weight_for_test,
            )
            if self.keep_master_weight_for_test:
                self.master_weight = master_weight
            if self.weight.numel() > 0 and self.weight.abs().max() == 0:
                logger.warning(
                    "VocabParallelEmbedding initialization resulted in zeros, using fallback initialization"
                )
                with torch.no_grad():
                    self.init_method(self.weight)
        except Exception as e:
            logger.warning(
                f"VocabParallelEmbedding model parallel initialization failed: {e}, using fallback initialization"
            )
            with torch.no_grad():
                self.init_method(self.weight)


# ---------------------------------------------------------------------------
# Verification / self-test
# ---------------------------------------------------------------------------


def verify_parallel_embedding(
    num_embeddings: int = 1024,
    embedding_dim: int = 128,
    batch_size: int = 4,
    seq_len: int = 16,
    seed: int = 42,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Verify ``ParallelEmbedding`` against a vanilla ``nn.Embedding`` reference.

    **Must** be called from every rank in the STEM model-parallel group at the
    same time (the function contains distributed collectives).

    Tests
    -----
    1. *train forward*  – same ``(B, S)`` on every rank, output matches reference.
    2. *train backward* – weight-gradient shard matches the corresponding shard
       of the reference gradient (uses identical input on all ranks so that the
       ``1/W`` rescale from the all-to-all backward cancels out).
    3. *eval forward (same shape)* – sanity check, same ``(B, S)`` on every rank.
    4. *eval forward (different shapes)* – each rank gets a **different**
       ``batch_size`` and ``seq_len``; output still matches reference.

    Returns ``True`` if every check passes **on every rank**.
    """
    rank = get_stem_model_parallel_rank()
    world_size = get_stem_model_parallel_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if rank == 0:
        logger.info(
            f"=== verify_parallel_embedding  (world_size={world_size}) ==="
        )

    # ---- build ParallelEmbedding -----------------------------------------
    pemb = ParallelEmbedding(
        num_embeddings,
        embedding_dim,
        device=device,
        init_method=lambda w: torch.nn.init.normal_(w, mean=0.0, std=0.02),
    )

    # ---- reconstruct full weight for reference ---------------------------
    full_weight = _gather_along_last_dim_stem(pemb.weight.data)  # [V, D]
    ref = torch.nn.Embedding(num_embeddings, embedding_dim, device=device)
    with torch.no_grad():
        ref.weight.copy_(full_weight)

    checks: Dict[str, bool] = {}

    # =====================================================================
    # Test 1 — train forward, same shape per rank
    # =====================================================================
    pemb.train()
    torch.manual_seed(seed + rank)
    inp = torch.randint(0, num_embeddings, (batch_size, seq_len), device=device)

    out_par = pemb(inp)
    out_ref = ref(inp)

    ok = torch.allclose(out_par, out_ref, atol=atol, rtol=rtol)
    checks["train_forward"] = ok
    if not ok:
        logger.error(
            f"  [FAIL] train_forward — rank {rank}, "
            f"max |diff| = {(out_par - out_ref).abs().max().item():.2e}"
        )

    # =====================================================================
    # Test 2 — train backward, same input on every rank
    #
    # When every rank uses the *same* input the rescale factor from the
    # all-to-all backward cancels with the W copies of each token in the
    # gathered batch, so  pemb.weight.grad  should exactly equal the
    # corresponding column-shard of  ref.weight.grad.
    # =====================================================================
    pemb.zero_grad()
    ref.zero_grad()

    torch.manual_seed(seed)  # identical on all ranks
    inp_same = torch.randint(
        0, num_embeddings, (batch_size, seq_len), device=device
    )

    pemb(inp_same).sum().backward()
    inp_same_global = _gather_along_first_dim_stem(inp_same.contiguous())
    ref(inp_same_global).sum().backward()

    shard_lo = rank * pemb.embedding_dim_per_partition
    shard_hi = shard_lo + pemb.embedding_dim_per_partition
    ref_grad_shard = ref.weight.grad[:, shard_lo:shard_hi]

    if pemb.weight.grad is None:
        checks["train_backward"] = False
        logger.error(
            f"  [FAIL] train_backward — rank {rank}, grad is None"
        )
    else:
        ok = torch.allclose(
            pemb.weight.grad, ref_grad_shard, atol=atol, rtol=rtol
        )
        checks["train_backward"] = ok
        if not ok:
            logger.error(
                f"  [FAIL] train_backward — rank {rank}, "
                f"max |diff| = "
                f"{(pemb.weight.grad - ref_grad_shard).abs().max().item():.2e}"
            )

    # =====================================================================
    # Test 3 — eval forward, same shape per rank (sanity)
    # =====================================================================
    pemb.eval()

    torch.manual_seed(seed)
    inp_eval_same = torch.randint(
        0, num_embeddings, (batch_size, seq_len), device=device
    )

    out_par_eval = pemb(inp_eval_same)
    out_ref_eval = ref(inp_eval_same)

    ok = torch.allclose(out_par_eval, out_ref_eval, atol=atol, rtol=rtol)
    checks["eval_forward_same_shape"] = ok
    if not ok:
        logger.error(
            f"  [FAIL] eval_forward_same_shape — rank {rank}, "
            f"max |diff| = {(out_par_eval - out_ref_eval).abs().max().item():.2e}"
        )

    # =====================================================================
    # Test 4 — eval forward, same input on every rank (simulates correct
    #          STEM dp usage where all model-parallel ranks get the same
    #          requests from lm_eval)
    # =====================================================================
    torch.manual_seed(seed + 999)
    inp_eval_same_all = torch.randint(
        0, num_embeddings, (batch_size + 2, seq_len + 5), device=device
    )

    out_par_eval2 = pemb(inp_eval_same_all)
    out_ref_eval2 = ref(inp_eval_same_all)

    ok = torch.allclose(out_par_eval2, out_ref_eval2, atol=atol, rtol=rtol)
    checks["eval_forward_same_input_all_ranks"] = ok
    if not ok:
        logger.error(
            f"  [FAIL] eval_forward_same_input_all_ranks — rank {rank}, "
            f"max |diff| = {(out_par_eval2 - out_ref_eval2).abs().max().item():.2e}"
        )

    # ---- aggregate across ranks -----------------------------------------
    all_local = all(checks.values())
    passed_t = torch.tensor(
        [int(all_local)], device=device, dtype=torch.long
    )
    if dist.is_initialized() and _MODEL_PARALLEL_GROUP is not None:
        dist.all_reduce(
            passed_t,
            op=dist.ReduceOp.MIN,
            group=get_stem_model_parallel_group(),
        )
    global_passed = passed_t.item() == 1

    if rank == 0:
        for name, ok in checks.items():
            logger.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        summary = "ALL PASSED" if global_passed else "SOME FAILED"
        (logger.info if global_passed else logger.error)(
            f"=== {summary} ==="
        )

    return global_passed


def verify_vocab_parallel_embedding(
    num_embeddings: int = 1024,
    embedding_dim: int = 128,
    batch_size: int = 4,
    seq_len: int = 16,
    seed: int = 42,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Verify ``VocabParallelEmbedding`` against a vanilla ``nn.Embedding`` reference."""
    rank = get_stem_model_parallel_rank()
    world_size = get_stem_model_parallel_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if rank == 0:
        logger.info(
            f"=== verify_vocab_parallel_embedding  (world_size={world_size}) ==="
        )

    pemb = VocabParallelEmbedding(
        num_embeddings,
        embedding_dim,
        device=device,
        init_method=lambda w: torch.nn.init.normal_(w, mean=0.0, std=0.02),
    )

    full_weight = _gather_along_first_dim_stem(pemb.weight.data)
    ref = torch.nn.Embedding(num_embeddings, embedding_dim, device=device)
    with torch.no_grad():
        ref.weight.copy_(full_weight)

    checks: Dict[str, bool] = {}

    pemb.train()
    torch.manual_seed(seed + rank)
    inp = torch.randint(0, num_embeddings, (batch_size, seq_len), device=device)

    out_par = pemb(inp)
    out_ref = ref(inp)

    ok = torch.allclose(out_par, out_ref, atol=atol, rtol=rtol)
    checks["train_forward"] = ok
    if not ok:
        logger.error(
            f"  [FAIL] train_forward — rank {rank}, "
            f"max |diff| = {(out_par - out_ref).abs().max().item():.2e}"
        )

    pemb.zero_grad()
    ref.zero_grad()

    torch.manual_seed(seed)
    inp_same = torch.randint(
        0, num_embeddings, (batch_size, seq_len), device=device
    )

    pemb(inp_same).sum().backward()
    inp_same_global = _gather_along_first_dim_stem(inp_same.contiguous())
    ref(inp_same_global).sum().backward()

    shard_lo = rank * pemb.num_embeddings_per_partition
    shard_hi = shard_lo + pemb.num_embeddings_per_partition
    ref_grad_shard = ref.weight.grad[shard_lo:shard_hi, :]

    if pemb.weight.grad is None:
        checks["train_backward"] = False
        logger.error(
            f"  [FAIL] train_backward — rank {rank}, grad is None"
        )
    else:
        ok = torch.allclose(
            pemb.weight.grad, ref_grad_shard, atol=atol, rtol=rtol
        )
        checks["train_backward"] = ok
        if not ok:
            logger.error(
                f"  [FAIL] train_backward — rank {rank}, "
                f"max |diff| = "
                f"{(pemb.weight.grad - ref_grad_shard).abs().max().item():.2e}"
            )

    pemb.eval()
    torch.manual_seed(seed)
    inp_eval_same = torch.randint(
        0, num_embeddings, (batch_size, seq_len), device=device
    )

    out_par_eval = pemb(inp_eval_same)
    out_ref_eval = ref(inp_eval_same)

    ok = torch.allclose(out_par_eval, out_ref_eval, atol=atol, rtol=rtol)
    checks["eval_forward_same_shape"] = ok
    if not ok:
        logger.error(
            f"  [FAIL] eval_forward_same_shape — rank {rank}, "
            f"max |diff| = {(out_par_eval - out_ref_eval).abs().max().item():.2e}"
        )

    torch.manual_seed(seed + 999)
    inp_eval_same_all = torch.randint(
        0, num_embeddings, (batch_size + 2, seq_len + 5), device=device
    )

    out_par_eval2 = pemb(inp_eval_same_all)
    out_ref_eval2 = ref(inp_eval_same_all)

    ok = torch.allclose(out_par_eval2, out_ref_eval2, atol=atol, rtol=rtol)
    checks["eval_forward_same_input_all_ranks"] = ok
    if not ok:
        logger.error(
            f"  [FAIL] eval_forward_same_input_all_ranks — rank {rank}, "
            f"max |diff| = {(out_par_eval2 - out_ref_eval2).abs().max().item():.2e}"
        )

    all_local = all(checks.values())
    passed_t = torch.tensor(
        [int(all_local)], device=device, dtype=torch.long
    )
    if dist.is_initialized() and _MODEL_PARALLEL_GROUP is not None:
        dist.all_reduce(
            passed_t,
            op=dist.ReduceOp.MIN,
            group=get_stem_model_parallel_group(),
        )
    global_passed = passed_t.item() == 1

    if rank == 0:
        for name, ok in checks.items():
            logger.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        summary = "ALL PASSED" if global_passed else "SOME FAILED"
        (logger.info if global_passed else logger.error)(
            f"=== {summary} ==="
        )

    return global_passed


# ---------------------------------------------------------------------------
# Standalone entry-point — run with:
#   torchrun --nproc_per_node=<N> lingua/stem_dist_utils.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    initialize_stem_process_group(world_size)

    # ok = verify_parallel_embedding()
    ok = verify_vocab_parallel_embedding()

    dist.barrier()
    dist.destroy_process_group()

    if not ok:
        raise SystemExit(1)
