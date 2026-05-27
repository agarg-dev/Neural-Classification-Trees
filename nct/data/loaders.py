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
    """Create a DataLoader from a PyTorch Dataset."""
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
    """Create sampler that balances across nodes via weighted oversampling."""
    num_nodes = node_labels.max().item() + 1
    node_counts = torch.bincount(node_labels, minlength=num_nodes).float()
    sample_weights = 1.0 / node_counts[node_labels]
    return WeightedRandomSampler(
        sample_weights, len(node_labels), replacement=True, generator=generator
    )


class NodeDownsampleSampler(Sampler):
    """Node-balanced sampler that draws a fresh random subsample each epoch
    by downsampling all nodes to the smallest node's count.

    # DataLoader calls iter() once per epoch — fresh subsample each call.
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
    """Create a node-balanced sampler that downsamples all nodes to the smallest node's count."""
    return NodeDownsampleSampler(node_labels, generator=generator)


class NodeGeomeanSampler(Sampler):
    """Geometric-mean node sampler: all non-empty nodes contribute N_target samples
    per epoch, where N_target = geometric mean of non-empty node sizes."""

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
