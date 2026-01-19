import random
from pathlib import Path

import flashinfer
import numpy as np
import torch

from Engine.model import ModelArgs, Transformer
from Engine.stem_model import StemTransformer



torch.library.define(
    "mylib::update_kv",
    "(Tensor k, Tensor v, Tensor kv_append_indptr, Tensor(a!) kv_cache, Tensor kv_page_indices, Tensor kv_page_indptr, Tensor cachelen) -> ()",
)


@torch.library.impl("mylib::update_kv", "cuda")
def update_kv(
    k,
    v,
    kv_append_indptr,
    kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_page_last_len,
):
    flashinfer.append_paged_kv_cache(
        k,
        v,
        kv_append_indptr,
        kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_page_last_len,
    )


@torch.library.register_fake("mylib::update_kv")
def update_kv_abstract(
    k,
    v,
    kv_append_indptr,
    kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_page_last_len,
):
    return None


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def load_model(
    checkpoint_path,
    model_name,
    device,
    precision,
    use_tp=False,
    rank_group=None,
    group=None,
):
    print("Loading model ...")
    with torch.device("meta"):
        model = Transformer.from_name(model_name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)
    
    # TODO: add support for tp
    
    print("Moving model to device ...")
    model = model.to(device=device, dtype=precision)
    return model.eval()


def load_stem_model(
    checkpoint_path,
    model_name,
    device,
    precision,
    max_batched_tokens: int,
    use_tp=False,
    rank_group=None,
    group=None,
):
    print("Loading model ...")
    with torch.device("meta"):
        model = StemTransformer.from_name(model_name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True, strict=False)
    
    if "stem_embeddings" in checkpoint:
        with torch.device(device):
            model.setup_stem(max_batched_tokens=max_batched_tokens)
        ckpt_stem = checkpoint["stem_embeddings"]
        model.cpu_stem_embeddings.copy_(ckpt_stem)
    
    # TODO: add support for tp
    
    print("Moving model to device ...")
    model = model.to(device=device, dtype=precision)
    return model.eval()