"""Data loading utilities."""

import math
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler


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
        generator=generator if is_training else None,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=is_training,
    )


def create_node_upsample_sampler(
    node_labels: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """
    Create sampler that balances across nodes via weighted oversampling.

    Each node gets equal sampling probability, so minority nodes appear
    as often as majority nodes. Epoch size = full dataset size.
    Pass a generator for reproducible but varying draws across epochs/runs.
    """
    num_nodes = node_labels.max().item() + 1
    node_counts = torch.bincount(node_labels, minlength=num_nodes).float()
    sample_weights = 1.0 / node_counts[node_labels]
    return WeightedRandomSampler(
        sample_weights, len(node_labels), replacement=True, generator=generator
    )


class NodeDownsampleSampler(Sampler):
    """
    Node-balanced sampler that draws a fresh random subsample each epoch
    by downsampling all nodes to the smallest node's count.

    # DataLoader calls iter(sampler) once per epoch — fresh subsample each call.
    Epoch size = min_count * num_non_empty_nodes.
    """

    def __init__(
        self,
        node_labels: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.generator = generator

        num_nodes = node_labels.max().item() + 1
        self._node_indices = [
            (node_labels == n).nonzero(as_tuple=True)[0] for n in range(num_nodes)
        ]
        self._nonempty = [idx for idx in self._node_indices if len(idx) > 0]
        self._min_count = min(len(idx) for idx in self._nonempty)

    def __iter__(self):
        """Fresh balanced subsample every epoch."""
        selected = []
        for idx in self._nonempty:
            perm = torch.randperm(len(idx), generator=self.generator)
            selected.append(idx[perm[: self._min_count]])
        selected = torch.cat(selected)
        final_perm = torch.randperm(len(selected), generator=self.generator)
        yield from selected[final_perm].tolist()

    def __len__(self) -> int:
        return self._min_count * len(self._nonempty)


def create_node_downsample_sampler(
    node_labels: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> NodeDownsampleSampler:
    """Create a node-balanced sampler that downsamples to the smallest node."""
    return NodeDownsampleSampler(node_labels, generator=generator)


class NodeHybridSampler(Sampler):
    """
    Hybrid node-balanced sampler for iterations 2+:
      - Easy nodes (even idx): downsample each to min(easy node sizes).
      - Difficult nodes (odd idx): upsample each to max(difficult node sizes).
    Fresh resampling every epoch.
    """

    def __init__(
        self,
        node_labels: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        easy_target: Optional[int] = None,
        diff_target: Optional[int] = None,
    ) -> None:
        self.generator = generator
        num_nodes = int(node_labels.max().item()) + 1
        all_indices = [
            (node_labels == n).nonzero(as_tuple=True)[0] for n in range(num_nodes)
        ]
        easy_node_ids = [n for n in range(num_nodes) if n % 2 == 0 and len(all_indices[n]) > 0]
        diff_node_ids = [n for n in range(num_nodes) if n % 2 == 1 and len(all_indices[n]) > 0]
        easy = [all_indices[n] for n in easy_node_ids]
        diff = [all_indices[n] for n in diff_node_ids]

        # Edge case: one group is empty → fall back to downsample behavior
        if not easy or not diff:
            nonempty = easy if easy else diff
            nonempty_ids = easy_node_ids if easy else diff_node_ids
            target = min(len(idx) for idx in nonempty)
            self._easy = nonempty
            self._diff = []
            self._easy_node_ids: list = nonempty_ids
            self._diff_node_ids: list = []
            self._easy_target = target
            self._diff_target = 0
        else:
            self._easy = easy
            self._diff = diff
            self._easy_node_ids = easy_node_ids
            self._diff_node_ids = diff_node_ids
            natural_easy = min(len(idx) for idx in easy)
            natural_diff = max(len(idx) for idx in diff)
            self._easy_target = easy_target if easy_target is not None else natural_easy
            self._diff_target = diff_target if diff_target is not None else natural_diff

        self._easy_contributions = [min(len(idx), self._easy_target) for idx in self._easy]
        self._epoch_len = sum(self._easy_contributions) + len(self._diff) * self._diff_target

    def __iter__(self):
        selected = []
        for idx, contrib in zip(self._easy, self._easy_contributions):
            perm = torch.randperm(len(idx), generator=self.generator)
            selected.append(idx[perm[:contrib]])
        for idx in self._diff:
            if len(idx) >= self._diff_target:
                perm = torch.randperm(len(idx), generator=self.generator)
                selected.append(idx[perm[: self._diff_target]])
            else:
                rep = torch.randint(len(idx), (self._diff_target,), generator=self.generator)
                selected.append(idx[rep])
        selected = torch.cat(selected)
        final_perm = torch.randperm(len(selected), generator=self.generator)
        yield from selected[final_perm].tolist()

    def __len__(self) -> int:
        return self._epoch_len


def create_node_hybrid_sampler(
    node_labels: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    easy_target: Optional[int] = None,
    diff_target: Optional[int] = None,
) -> NodeHybridSampler:
    """Create a hybrid node sampler (downsample easy, upsample difficult)."""
    return NodeHybridSampler(node_labels, generator=generator,
                              easy_target=easy_target, diff_target=diff_target)


class NodeGeomeanSampler(Sampler):
    """
    Geometric-mean node sampler: every non-empty node contributes N_target
    = geometric mean of node sizes per epoch.
    """

    def __init__(
        self,
        node_labels: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.generator = generator

        num_nodes = int(node_labels.max().item()) + 1
        all_indices = [
            (node_labels == n).nonzero(as_tuple=True)[0] for n in range(num_nodes)
        ]
        self._nonempty = [idx for idx in all_indices if len(idx) > 0]

        sizes = [len(idx) for idx in self._nonempty]
        log_mean = sum(math.log(s) for s in sizes) / len(sizes)
        self._target: int = max(1, int(round(math.exp(log_mean))))

    def __iter__(self):
        """Fresh per-epoch sample: downsample large nodes, oversample small ones."""
        selected = []
        for idx in self._nonempty:
            n = len(idx)
            if n >= self._target:
                perm = torch.randperm(n, generator=self.generator)
                selected.append(idx[perm[: self._target]])
            else:
                rep = torch.randint(n, (self._target,), generator=self.generator)
                selected.append(idx[rep])
        selected = torch.cat(selected)
        final_perm = torch.randperm(len(selected), generator=self.generator)
        yield from selected[final_perm].tolist()

    def __len__(self) -> int:
        return self._target * len(self._nonempty)


def create_node_geomean_sampler(
    node_labels: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> NodeGeomeanSampler:
    """Create a geometric-mean node sampler."""
    return NodeGeomeanSampler(node_labels, generator=generator)
