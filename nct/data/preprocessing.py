"""Dataset preparation and metadata loading."""

import os
from typing import Dict, Optional

import pandas as pd

from .configs import DATASET_REGISTRY, DatasetConfig
from .datasets import GenericImageDataset
from .transforms import create_transforms


def load_waterbird_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load Waterbird dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Waterbird dataset not found at {dataset_path}")
    metadata_df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))

    train_df = metadata_df[metadata_df["split"] == 0]
    val_df = metadata_df[metadata_df["split"] == 1]
    test_df = metadata_df[metadata_df["split"] == 2]

    group_test = test_df["y"].values * 2 + test_df["place"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "root_dir": dataset_path,
    }


def load_celeba_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load CelebA dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CelebA dataset not found at {dataset_path}")

    attr_df = pd.read_csv(os.path.join(dataset_path, "list_attr_celeba.csv"))
    split_df = pd.read_csv(os.path.join(dataset_path, "list_eval_partition.csv"))

    metadata_df = pd.merge(attr_df, split_df, on="image_id")

    # Convert labels from -1/1 to 0/1
    metadata_df["y"] = ((metadata_df["Blond_Hair"] + 1) // 2).astype(int)
    metadata_df["male"] = ((metadata_df["Male"] + 1) // 2).astype(int)
    metadata_df["img_filename"] = "img_align_celeba/" + metadata_df["image_id"]

    train_df = metadata_df[metadata_df["partition"] == 0]
    val_df = metadata_df[metadata_df["partition"] == 1]
    test_df = metadata_df[metadata_df["partition"] == 2]

    group_test = test_df["y"].values * 2 + test_df["male"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "root_dir": dataset_path,
    }


def load_isic_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load ISIC dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ISIC dataset not found at {dataset_path}")

    metadata_df = pd.read_csv(os.path.join(dataset_path, "labels.csv"))

    # Filter to only benign/malignant (exclude indeterminate)
    metadata_df = metadata_df[
        metadata_df["benign_malignant"].isin(["benign", "malignant"])
    ]

    metadata_df["y"] = metadata_df["Diagnosis"].astype(int)
    metadata_df["img_filename"] = (
        "ISIC_2019_Training_Input/" + metadata_df["Image Index"]
    )

    # Histopathology-confirmed diagnosis indicator
    metadata_df["histopathology"] = (
        metadata_df["diagnosis_confirm_type"] == "histopathology"
    ).astype(int)

    train_df = metadata_df[metadata_df["fold"] == "train"]
    val_df = metadata_df[metadata_df["fold"] == "val"]
    test_df = metadata_df[metadata_df["fold"] == "test"]

    group_test = test_df["y"].values * 2 + test_df["patch"].values

    # Subgroups for AUROC evaluation (boolean masks)
    val_subgroups = {
        "non_patch": val_df["patch"].values == 0,
        "histopath": val_df["histopathology"].values == 1,
    }
    test_subgroups = {
        "non_patch": test_df["patch"].values == 0,
        "histopath": test_df["histopathology"].values == 1,
    }

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "val_subgroups": val_subgroups,
        "test_subgroups": test_subgroups,
        "root_dir": dataset_path,
    }


def load_umnist_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load UMNIST (Undersampled MNIST) dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"UMNIST dataset not found at {dataset_path}")

    metadata_df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))

    train_df = metadata_df[metadata_df["split"] == "train"]
    val_df = metadata_df[metadata_df["split"] == "val"]
    test_df = metadata_df[metadata_df["split"] == "test"]

    # Group: y * 2 + is_digit_8
    group_test = test_df["y"].values * 2 + test_df["is_digit_8"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "root_dir": dataset_path,
    }


METADATA_LOADERS = {
    "waterbird": load_waterbird_metadata,
    "celeba": load_celeba_metadata,
    "isic": load_isic_metadata,
    "umnist": load_umnist_metadata,
}


def prepare_dataset(
    data_dir: str,
    dataset_name: str,
    cache_images_in_memory: bool = False,
):
    """
    Prepare dataset with appropriate transforms.

    Args:
        data_dir: Root data directory.
        dataset_name: Dataset name ('waterbird', 'celeba', 'isic').
        cache_images_in_memory: If True, cache PIL images in RAM (causes OOM on large datasets).

    Returns:
        Dictionary with train/val/test datasets, group labels, and config.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[dataset_name]
    loader_fn = METADATA_LOADERS[dataset_name]

    metadata = loader_fn(data_dir, config)

    # Use PIL-based transforms
    train_transform = create_transforms(config, is_training=True)
    val_test_transform = create_transforms(config, is_training=False)

    train_dataset = GenericImageDataset(
        metadata["train_df"],
        metadata["root_dir"],
        transform=train_transform,
        cache_images_in_memory=cache_images_in_memory,
        num_channels=config.num_channels,
    )
    val_dataset = GenericImageDataset(
        metadata["val_df"],
        metadata["root_dir"],
        transform=val_test_transform,
        cache_images_in_memory=cache_images_in_memory,
        num_channels=config.num_channels,
    )
    test_dataset = GenericImageDataset(
        metadata["test_df"],
        metadata["root_dir"],
        transform=val_test_transform,
        cache_images_in_memory=cache_images_in_memory,
        num_channels=config.num_channels,
    )

    print(f"\n{config.name.upper()} Dataset Loaded:")
    print(
        f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "group_test": metadata["group_test"],
        "val_subgroups": metadata.get("val_subgroups"),
        "test_subgroups": metadata.get("test_subgroups"),
        "config": config,
        "num_channels": config.num_channels,
        "num_classes": config.num_classes,
        "class_names": config.class_names,
    }
