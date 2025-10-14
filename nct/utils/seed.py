"""Reproducibility utilities for seeding random number generators."""

import os
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """
    Sets the seed for all relevant random number generators to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Using deterministic algorithms can aid reproducibility but may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False