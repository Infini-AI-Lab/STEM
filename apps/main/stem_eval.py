# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, List, Optional, Tuple, Union
from lm_eval import simple_evaluate
from omegaconf import OmegaConf
import torch
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.eval import LMHarnessArgs, ValidationArgs, EvalArgs
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.stem_checkpoint import consolidate_stem_checkpoints
from lingua.data import init_choice_state, setup_sources
from lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()