"""Reproducibility utilities for seeding random number generators."""

import os
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (incl. CUDA) RNGs; force deterministic cuDNN."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False