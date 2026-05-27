"""Dataset classes for image data."""

import os
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class IterativeDataset(Dataset):
    """
    Wrapper dataset for iterative training with dynamic labels.
    Works with any base dataset - overrides labels for hierarchical training.
    """

    def __init__(self, base_dataset: Dataset, hier_labels: torch.Tensor):
        self.base_dataset = base_dataset
        self.hier_labels = hier_labels
        assert len(self.base_dataset) == len(self.hier_labels), (
            "Base dataset and labels must have the same length."
        )

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
        img_filename_col: str = "img_filename",
        label_col: str = "y",
        transform: Optional[Callable] = None,
        cache_images_in_memory: bool = False,
        num_channels: int = 3,
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.img_filename_col = img_filename_col
        self.label_col = label_col
        self.transform = transform
        self.cache_images_in_memory = cache_images_in_memory
        self.num_channels = num_channels
        self._image_cache = {} if cache_images_in_memory else None

        if cache_images_in_memory:
            self._preload_cache()

    def _preload_cache(self):
        """Pre-load all images into cache."""
        num_images = len(self.metadata)
        mode = "L" if self.num_channels == 1 else "RGB"
        print(f"Pre-loading {num_images} images into memory (mode={mode})...")
        for idx in range(num_images):
            img_path = os.path.join(
                self.root_dir, self.metadata.iloc[idx][self.img_filename_col]
            )
            try:
                img = Image.open(img_path)
                converted_img = img.convert(mode)
                converted_img.load()
                self._image_cache[idx] = converted_img
                img.close()
            except FileNotFoundError:
                print(f"Warning: Image not found at {img_path}")
                self._image_cache[idx] = Image.new(mode, (256, 256), color="black" if mode == "RGB" else 0)
            if (idx + 1) % 1000 == 0:
                print(f"  Loaded {idx + 1} / {num_images}")
        print(f"Cache complete: {len(self._image_cache)} images")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = int(self.metadata.iloc[idx][self.label_col])
        mode = "L" if self.num_channels == 1 else "RGB"

        if self._image_cache is not None and idx in self._image_cache:
            image = self._image_cache[idx]
        else:
            img_path = os.path.join(
                self.root_dir, self.metadata.iloc[idx][self.img_filename_col]
            )
            try:
                with Image.open(img_path) as img:
                    image = img.convert(mode)
            except FileNotFoundError:
                print(f"Error: Image not found at {img_path}")
                image = Image.new(mode, (256, 256), color="black" if mode == "RGB" else 0)

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
    if hasattr(dataset, "metadata") and "y" in dataset.metadata.columns:
        return torch.tensor(dataset.metadata["y"].values, dtype=torch.long)
    elif hasattr(dataset, "labels"):
        return (
            torch.tensor(dataset.labels, dtype=torch.long)
            if not isinstance(dataset.labels, torch.Tensor)
            else dataset.labels
        )
    else:
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return torch.tensor(labels, dtype=torch.long)
