"""Dataset classes for image data."""

import torch
from torch.utils.data import Dataset
from typing import Tuple


class IterativeDataset(Dataset):
    """
    Wrapper dataset for iterative training with dynamic labels.
    Works with any base dataset - overrides labels for hierarchical training.
    """
    def __init__(self, base_dataset: Dataset, hier_labels: torch.Tensor):
        self.base_dataset = base_dataset
        self.hier_labels = hier_labels
        assert len(self.base_dataset) == len(self.hier_labels), \
            "Base dataset and labels must have the same length."

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, _ = self.base_dataset[idx]
        hier_label = self.hier_labels[idx]
        if isinstance(hier_label, torch.Tensor):
            hier_label = hier_label.item()
        return image, hier_label