"""Dataset preparation and metadata loading."""

import os
from typing import Dict, Tuple

import pandas as pd
from PIL import Image

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
    group_val = val_df["y"].values * 2 + val_df["place"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "group_val": group_val,
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

    metadata_df["y"] = ((metadata_df["Blond_Hair"] + 1) // 2).astype(int)
    metadata_df["male"] = ((metadata_df["Male"] + 1) // 2).astype(int)
    metadata_df["img_filename"] = "img_align_celeba/" + metadata_df["image_id"]

    train_df = metadata_df[metadata_df["partition"] == 0]
    val_df = metadata_df[metadata_df["partition"] == 1]
    test_df = metadata_df[metadata_df["partition"] == 2]

    group_test = test_df["y"].values * 2 + test_df["male"].values
    group_val = val_df["y"].values * 2 + val_df["male"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "group_val": group_val,
        "root_dir": dataset_path,
    }


def load_isic_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load ISIC dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ISIC dataset not found at {dataset_path}")

    metadata_df = pd.read_csv(os.path.join(dataset_path, "labels.csv"))

    metadata_df = metadata_df[
        metadata_df["benign_malignant"].isin(["benign", "malignant"])
    ]

    metadata_df["y"] = metadata_df["Diagnosis"].astype(int)
    metadata_df["img_filename"] = (
        "ISIC_2019_Training_Input/" + metadata_df["Image Index"]
    )

    metadata_df["histopathology"] = (
        metadata_df["diagnosis_confirm_type"] == "histopathology"
    ).astype(int)

    train_df = metadata_df[metadata_df["fold"] == "train"]
    val_df = metadata_df[metadata_df["fold"] == "val"]
    test_df = metadata_df[metadata_df["fold"] == "test"]

    group_test = test_df["y"].values * 2 + test_df["patch"].values
    group_val = val_df["y"].values * 2 + val_df["patch"].values

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
        "group_val": group_val,
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

    group_test = test_df["y"].values * 2 + test_df["is_digit_8"].values
    group_val = val_df["y"].values * 2 + val_df["is_digit_8"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "group_val": group_val,
        "root_dir": dataset_path,
    }


def load_cmnist_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load CMNIST (Colored MNIST) dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CMNIST dataset not found at {dataset_path}")

    metadata_df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))

    train_df = metadata_df[metadata_df["split"] == "train"]
    val_df = metadata_df[metadata_df["split"] == "val"]
    test_df = metadata_df[metadata_df["split"] == "test"]

    num_colors = config.num_classes
    if "color" in metadata_df.columns:
        group_test = test_df["y"].values * num_colors + test_df["color"].values
        group_val = val_df["y"].values * num_colors + val_df["color"].values
    else:
        group_test = test_df["y"].values
        group_val = val_df["y"].values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": group_test,
        "group_val": group_val,
        "root_dir": dataset_path,
    }


def _load_urbancars_split(
    dataset_path: str,
    split: str,
    correlation_folder: str,
    target_resolution: Tuple[int, int],
) -> pd.DataFrame:
    """Build deterministic metadata for one official UrbanCars split."""
    category_names = ("urban", "country")
    split_root = os.path.join(dataset_path, correlation_folder, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(
            f"UrbanCars {split} split not found at {split_root}"
        )

    rows = []
    for object_id, object_name in enumerate(category_names):
        for background_id, background_name in enumerate(category_names):
            for coobject_id, coobject_name in enumerate(category_names):
                group_dir_name = (
                    f"obj-{object_name}_bg-{background_name}_"
                    f"co_occur_obj-{coobject_name}"
                )
                group_dir = os.path.join(split_root, group_dir_name)
                if not os.path.isdir(group_dir):
                    raise FileNotFoundError(
                        f"UrbanCars group directory not found: {group_dir}"
                    )

                image_paths = sorted(
                    os.path.join(group_dir, filename)
                    for filename in os.listdir(group_dir)
                    if filename.lower().endswith(".jpg")
                )
                if not image_paths:
                    raise ValueError(
                        f"UrbanCars group directory contains no JPG images: {group_dir}"
                    )

                group_id = object_id * 4 + background_id * 2 + coobject_id
                for image_path in image_paths:
                    with Image.open(image_path) as image:
                        if image.size != tuple(target_resolution):
                            raise ValueError(
                                "UrbanCars images must preserve the official "
                                f"{tuple(target_resolution)} resolution; found "
                                f"{image.size} at {image_path}"
                            )

                    rows.append(
                        {
                            "img_filename": os.path.relpath(
                                image_path, dataset_path
                            ),
                            "y": object_id,
                            "background": background_id,
                            "co_occur_obj": coobject_id,
                            "bg_aligned": int(background_id == object_id),
                            "co_occur_obj_aligned": int(
                                coobject_id == object_id
                            ),
                            "group": group_id,
                            "split": split,
                        }
                    )

    metadata_df = pd.DataFrame(rows)
    decoded_y = metadata_df["group"].to_numpy() // 4
    if not (decoded_y == metadata_df["y"].to_numpy()).all():
        raise RuntimeError("UrbanCars group encoding is inconsistent with targets")
    if set(metadata_df["group"].unique()) != set(range(8)):
        raise RuntimeError(
            f"UrbanCars {split} split must contain all 8 joint groups"
        )

    return metadata_df


def load_urbancars_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load UrbanCars using the official Whac-A-Mole directory layout."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"UrbanCars dataset not found at {dataset_path}")

    train_df = _load_urbancars_split(
        dataset_path,
        split="train",
        correlation_folder="bg-0.95_co_occur_obj-0.95",
        target_resolution=config.target_resolution,
    )
    val_df = _load_urbancars_split(
        dataset_path,
        split="val",
        correlation_folder="bg-0.5_co_occur_obj-0.5",
        target_resolution=config.target_resolution,
    )
    test_df = _load_urbancars_split(
        dataset_path,
        split="test",
        correlation_folder="bg-0.5_co_occur_obj-0.5",
        target_resolution=config.target_resolution,
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "group_test": test_df["group"].to_numpy(),
        "group_val": val_df["group"].to_numpy(),
        "root_dir": dataset_path,
    }


METADATA_LOADERS = {
    "waterbird": load_waterbird_metadata,
    "celeba": load_celeba_metadata,
    "isic": load_isic_metadata,
    "umnist": load_umnist_metadata,
    "cmnist": load_cmnist_metadata,
    "urbancars": load_urbancars_metadata,
}


def prepare_dataset(
    data_dir: str,
    dataset_name: str,
    cache_images_in_memory: bool = False,
):
    """Prepare train/val/test datasets with the per-dataset transforms applied."""
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[dataset_name]
    loader_fn = METADATA_LOADERS[dataset_name]

    metadata = loader_fn(data_dir, config)

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
        "group_val": metadata.get("group_val"),
        "val_subgroups": metadata.get("val_subgroups"),
        "test_subgroups": metadata.get("test_subgroups"),
        "config": config,
        "num_channels": config.num_channels,
        "num_classes": config.num_classes,
        "class_names": config.class_names,
    }
