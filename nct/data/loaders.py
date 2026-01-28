"""Data loading utilities."""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def create_data_loader(
    data: Dataset,
    is_training: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader from a PyTorch Dataset.

    Args:
        data: PyTorch Dataset object.
        is_training: Whether this is training data (affects shuffling and drop_last).
        batch_size: Batch size for data loader.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for shuffling.
        generator: Optional persistent generator for reproducible shuffling.
        **kwargs: Additional arguments (ignored, kept for backward compatibility).

    Returns:
        DataLoader: Configured data loader with optimizations:
            - persistent_workers=True: Workers persist across epochs (faster)
            - prefetch_factor=4: Each worker prefetches 4 batches (reduces GPU idle)
            - drop_last=True (training only): Drops incomplete last batch for stability
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
        generator=generator if is_training else None,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=is_training,
    )


def create_node_balanced_sampler(node_labels: torch.Tensor) -> WeightedRandomSampler:
    """
    Create sampler that balances across nodes for DFR-style training.

    Each node gets equal sampling probability, so minority nodes
    (hard samples) appear as often as majority nodes (easy samples).
    """
    num_nodes = node_labels.max().item() + 1
    node_counts = torch.bincount(node_labels, minlength=num_nodes).float()
    sample_weights = 1.0 / node_counts[node_labels]
    return WeightedRandomSampler(sample_weights, len(node_labels), replacement=True)
