#!/usr/bin/env python3
"""
Generate CMNIST (Colored MNIST) dataset.
Based on "Correct-N-Contrast: A Contrastive Approach for Improving Robustness to
Spurious Correlations" paper (https://arxiv.org/pdf/2203.01517).
Digits are colored based on their class, with a specified correlation.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
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

        # 5 colors from the paper: {#ff0000, #85ff00, #00fff3, #6e00ff, #ff0018}
        self.colors_hex = ["#ff0000", "#85ff00", "#00fff3", "#6e00ff", "#ff0018"]
        self.colors_rgb = [
            tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            for h in self.colors_hex
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
        val_ratio: float = 0.2,
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
        """Color a grayscale image."""
        img_arr = np.array(image)
        img_arr_rgb = np.stack([img_arr]*3, axis=-1)

        # Create a color mask
        color_mask = np.array(color).reshape(1, 1, 3)

        # Apply color. The digit part will be colored, background remains black.
        colored_img_arr = (img_arr_rgb / 255.0 * color_mask).astype(np.uint8)

        return Image.fromarray(colored_img_arr)


    def save_images_and_metadata(self, splits: dict) -> pd.DataFrame:
        """Save images to disk and create metadata DataFrame."""
        print("\nSaving images and creating metadata...")

        records = []

        for split_name, (dataset, indices) in splits.items():
            print(f"  Processing {split_name}...")

            for idx in tqdm(indices, desc=f"  {split_name}", leave=False):
                img, digit = dataset[idx]

                # 5-class label: (0,1)->0, (2,3)->1, (4,5)->2, (6,7)->3, (8,9)->4
                y = digit // 2

                # Get color based on correlation rule
                color = self.get_color(y, split_name)

                # Color the image
                colored_img = self.color_image(img, color)

                # Save image
                img_filename = f"{split_name}_{idx:06d}.png"
                img_path = self.image_dir / img_filename
                colored_img.save(img_path)

                records.append({
                    "img_filename": f"images/{img_filename}",
                    "y": y,
                    "digit": digit,
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
        """Run the full generation pipeline."""
        print("=" * 50)
        print("CMNIST Dataset Generator")
        print("=" * 50)

        # Download MNIST
        train_data, test_data = self.download_mnist()

        # Create splits
        splits = self.create_splits(train_data, test_data)

        # Save images and create metadata
        df = self.save_images_and_metadata(splits)

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
        description="Generate CMNIST dataset from MNIST"
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
