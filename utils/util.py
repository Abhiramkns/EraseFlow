# utils.py

import os
import random
import logging
import numpy as np
import torch
from diffusers.utils.torch_utils import is_compiled_module

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    logging.info(f"Using seed: {seed}")

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    If a model is compiled (e.g. with torch.compile()), unwrap back to the original module.
    """
    return model._orig_mod if is_compiled_module(model) else model

def image_postprocess(x):
    # [-1, 1] -> [0, 1]
    return torch.clamp((x + 1) / 2, 0, 1)  # x / 2 + 0.5

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)