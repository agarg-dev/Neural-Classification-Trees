"""Dataset preparation and metadata loading."""

import os
import pandas as pd
from typing import Dict

from .datasets import GenericImageDataset
from .transforms import create_transforms
from .configs import DATASET_REGISTRY, DatasetConfig


def load_waterbird_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load Waterbird dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Waterbird dataset not found at {dataset_path}")

    metadata_df = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))

    train_df = metadata_df[metadata_df['split'] == 0]
    val_df = metadata_df[metadata_df['split'] == 1]
    test_df = metadata_df[metadata_df['split'] == 2]

    group_test = test_df['y'].values * 2 + test_df['place'].values

    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'group_test': group_test,
        'root_dir': dataset_path,
    }


def load_celeba_metadata(data_dir: str, config: DatasetConfig) -> Dict:
    """Load CelebA dataset metadata."""
    dataset_path = os.path.join(data_dir, config.data_subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CelebA dataset not found at {dataset_path}")

    attr_df = pd.read_csv(os.path.join(dataset_path, 'list_attr_celeba.csv'))
    split_df = pd.read_csv(os.path.join(dataset_path, 'list_eval_partition.csv'))

    metadata_df = pd.merge(attr_df, split_df, on='image_id')

    # Convert labels from -1/1 to 0/1
    metadata_df['y'] = ((metadata_df['Blond_Hair'] + 1) // 2).astype(int)
    metadata_df['male'] = ((metadata_df['Male'] + 1) // 2).astype(int)
    metadata_df['img_filename'] = 'img_align_celeba/' + metadata_df['image_id']

    train_df = metadata_df[metadata_df['partition'] == 0]
    val_df = metadata_df[metadata_df['partition'] == 1]
    test_df = metadata_df[metadata_df['partition'] == 2]

    group_test = test_df['y'].values * 2 + test_df['male'].values

    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'group_test': group_test,
        'root_dir': dataset_path,
    }


METADATA_LOADERS = {
    'waterbird': load_waterbird_metadata,
    'celeba': load_celeba_metadata,
}


def prepare_dataset(data_dir: str, dataset_name: str):
    """
    Prepare dataset with appropriate transforms.

    Args:
        data_dir: Root data directory.
        dataset_name: Dataset name ('waterbird', 'celeba').

    Returns:
        Dictionary with train/val/test datasets, group labels, and config.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    config = DATASET_REGISTRY[dataset_name]
    loader_fn = METADATA_LOADERS[dataset_name]

    metadata = loader_fn(data_dir, config)

    train_transform = create_transforms(config, is_training=True)
    val_test_transform = create_transforms(config, is_training=False)

    train_dataset = GenericImageDataset(metadata['train_df'], metadata['root_dir'], transform=train_transform)
    val_dataset = GenericImageDataset(metadata['val_df'], metadata['root_dir'], transform=val_test_transform)
    test_dataset = GenericImageDataset(metadata['test_df'], metadata['root_dir'], transform=val_test_transform)

    print(f"\n{config.name.upper()} Dataset Loaded:")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'group_test': metadata['group_test'],
        'config': config,
    }
