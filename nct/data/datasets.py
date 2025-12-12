"""Dataset classes for image data."""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Any, Callable, Optional


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


class GenericImageDataset(Dataset):
    """Generic dataset for loading images from metadata DataFrame."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        root_dir: str,
        img_filename_col: str = 'img_filename',
        label_col: str = 'y',
        transform: Optional[Callable] = None
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.img_filename_col = img_filename_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx][self.img_filename_col])

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            image = Image.new('RGB', (256, 256), color='black')

        label = int(self.metadata.iloc[idx][self.label_col])

        if self.transform:
            image = self.transform(image)

        return image, label


def extract_labels_from_dataset(dataset: Any) -> torch.Tensor:
    """
    Extract all labels directly from a dataset.

    This function handles multiple dataset formats:
    - Datasets with metadata DataFrame (e.g., WILDS datasets)
    - Datasets with a labels attribute
    - Generic datasets via iteration

    Args:
        dataset: PyTorch Dataset to extract labels from.

    Returns:
        Tensor containing all labels from the dataset.
    """
    if hasattr(dataset, 'metadata') and 'y' in dataset.metadata.columns:
        return torch.tensor(dataset.metadata['y'].values, dtype=torch.long)
    elif hasattr(dataset, 'labels'):
        return torch.tensor(dataset.labels, dtype=torch.long) if not isinstance(dataset.labels, torch.Tensor) else dataset.labels
    else:
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return torch.tensor(labels, dtype=torch.long)