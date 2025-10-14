"""Data loading utilities."""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional


def create_data_loader(
    data: Dataset,
    is_training: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader from a PyTorch Dataset.

    Args:
        data: PyTorch Dataset object.
        is_training: Whether this is training data (affects shuffling).
        batch_size: Batch size for data loader.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for shuffling.
        generator: Optional persistent generator for reproducible shuffling.
        **kwargs: Additional arguments (ignored, kept for backward compatibility).

    Returns:
        DataLoader: Configured data loader.
    """
    if generator is None:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if is_training else None
    )