#!/usr/bin/env python3
"""
Generate CMNIST (Colored MNIST) dataset - TEST ONLY version.
This script only processes the test split and generates metadata.csv
for all splits (assuming train/val images already exist).
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image
from torchvision import datasets
from tqdm import tqdm


class CMNISTGenerator:
    """Generate CMNIST dataset from standard MNIST."""

    def __init__(self, output_dir: str, p_corr: float = 0.995, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.p_corr = p_corr
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)

        # 5 colors sampled from the 'hsv' colormap at [0, 0.2, 0.4, 0.6, 0.8],
        # matching CnC reference (datasets/colored_mnist.py:_init_colors).
        cmap = cm.get_cmap("hsv")
        cmap_vals = np.arange(0, 1, step=1 / 5)
        self.colors_rgb = [
            tuple(int(x) for x in (np.array(cmap(v)[:3]) * 255))
            for v in cmap_vals
        ]
        self.colors_hex = [
            "#" + "".join(f"{c:02x}" for c in rgb) for rgb in self.colors_rgb
        ]

    def download_mnist(self) -> tuple:
        """Download MNIST using torchvision."""
        print("Downloading MNIST...")
        train_data = datasets.MNIST(
            root=str(self.output_dir / "raw"),
            train=True,
            download=True,
        )
        test_data = datasets.MNIST(
            root=str(self.output_dir / "raw"),
            train=False,
            download=True,
        )
        return train_data, test_data

    def create_splits(
        self,
        train_data,
        test_data,
        val_ratio: float = 0.1,
    ) -> dict:
        """Create train/val/test splits."""
        print("\nCreating data splits...")

        train_labels = np.array(train_data.targets)
        test_labels = np.array(test_data.targets)

        # Split original train into train/val
        n_train_orig = len(train_labels)
        all_train_indices = list(range(n_train_orig))
        random.shuffle(all_train_indices)

        val_size = int(n_train_orig * val_ratio)
        val_indices = all_train_indices[:val_size]
        train_indices = all_train_indices[val_size:]

        test_indices = list(range(len(test_labels)))

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_indices)}")
        print(f"  Val: {len(val_indices)}")
        print(f"  Test: {len(test_indices)}")

        return {
            "train": (train_data, train_indices),
            "val": (train_data, val_indices),
            "test": (test_data, test_indices),
        }

    def get_color(self, digit_class: int, split: str):
        if split == 'train':
            if random.random() < self.p_corr:
                return self.colors_rgb[digit_class]
            else:
                return random.choice([c for i, c in enumerate(self.colors_rgb) if i != digit_class])
        else: # val and test
            return random.choice(self.colors_rgb)

    def color_image(self, image: Image.Image, color: tuple) -> Image.Image:
        """Color a grayscale image.

        Matches CnC reference (datasets/colored_mnist.py:172-188): hard-threshold
        digit pixels at >=120, replace them with the solid uint8 color, leave
        sub-threshold (dark) pixels as grayscale replicated into RGB.
        """
        img_arr = np.array(image)
        img_arr_rgb = np.stack([img_arr] * 3, axis=-1).astype(np.uint8)

        pixels_r = np.where(np.logical_and(img_arr >= 120, img_arr <= 255))
        img_arr_rgb[pixels_r[0], pixels_r[1], :] = np.array(color, dtype=np.uint8)

        return Image.fromarray(img_arr_rgb)

    def generate_metadata_for_existing(self, dataset, indices, split_name) -> list:
        """Generate metadata records for already existing images (train/val)."""
        print(f"  Generating metadata for existing {split_name} images...")
        records = []

        for idx in tqdm(indices, desc=f"  {split_name}", leave=False):
            img, digit = dataset[idx]
            y = digit // 2
            color = self.get_color(y, split_name)
            color_idx = self.colors_rgb.index(color)
            img_filename = f"{split_name}_{idx:06d}.png"

            records.append({
                "img_filename": f"images/{img_filename}",
                "y": y,
                "digit": digit,
                "color": color_idx,
                "split": split_name,
            })

        return records

    def save_test_images_and_metadata(self, splits: dict) -> pd.DataFrame:
        """Save test images to disk and create metadata DataFrame for all splits."""
        print("\nProcessing splits...")

        records = []

        # For train and val, just generate metadata (images already exist)
        for split_name in ["train", "val"]:
            dataset, indices = splits[split_name]
            records.extend(self.generate_metadata_for_existing(dataset, indices, split_name))

        # For test, actually generate images
        split_name = "test"
        dataset, indices = splits[split_name]
        print(f"  Processing {split_name} (generating images)...")

        for idx in tqdm(indices, desc=f"  {split_name}", leave=False):
            img, digit = dataset[idx]

            # 5-class label: (0,1)->0, (2,3)->1, (4,5)->2, (6,7)->3, (8,9)->4
            y = digit // 2

            color = self.get_color(y, split_name)
            color_idx = self.colors_rgb.index(color)
            colored_img = self.color_image(img, color)

            img_filename = f"{split_name}_{idx:06d}.png"
            img_path = self.image_dir / img_filename
            colored_img.save(img_path)

            records.append({
                "img_filename": f"images/{img_filename}",
                "y": y,
                "digit": digit,
                "color": color_idx,
                "split": split_name,
            })

        df = pd.DataFrame(records)
        return df

    def print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics."""
        print("\n" + "=" * 50)
        print("CMNIST Dataset Statistics")
        print("=" * 50)

        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            print(f"\n{split.upper()} ({len(split_df)} samples):")
            class_counts = split_df["y"].value_counts().sort_index()
            print(f"  Classes: {dict(class_counts)}")

    def run(self):
        """Run the generation pipeline (test only)."""
        print("=" * 50)
        print("CMNIST Dataset Generator (TEST ONLY)")
        print("=" * 50)

        # Download MNIST
        train_data, test_data = self.download_mnist()

        # Create splits
        splits = self.create_splits(train_data, test_data)

        # Save test images and create metadata for all
        df = self.save_test_images_and_metadata(splits)

        # Save metadata
        csv_path = self.output_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nMetadata saved to: {csv_path}")

        # Print statistics
        self.print_statistics(df)

        print("\n" + "=" * 50)
        print("Generation complete!")
        print(f"Images: {self.image_dir}")
        print(f"Metadata: {csv_path}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMNIST dataset (test split only)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/cmnist",
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--p_corr",
        type=float,
        default=0.995,
        help="Correlation between class and color in training set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    generator = CMNISTGenerator(args.output_dir, args.p_corr, args.seed)
    generator.run()


if __name__ == "__main__":
    main()
