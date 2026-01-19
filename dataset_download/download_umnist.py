#!/usr/bin/env python3
"""
Generate UMNIST (Undersampled MNIST) dataset.
Based on GEORGE paper: digits 0-4 vs 5-9, with digit 8 undersampled to 5%.
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


class UMNISTGenerator:
    """Generate UMNIST dataset from standard MNIST."""

    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)

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

    def subsample_digit(
        self,
        indices: list,
        labels: np.ndarray,
        digit: int,
        proportion: float,
    ) -> list:
        """Subsample a specific digit to create spurious correlation."""
        digit_indices = [i for i in indices if labels[i] == digit]
        other_indices = [i for i in indices if labels[i] != digit]

        n_keep = int(len(digit_indices) * proportion)
        kept_digit_indices = random.sample(digit_indices, n_keep)

        print(f"  Digit {digit}: {len(digit_indices)} -> {n_keep} ({proportion:.0%})")
        return other_indices + kept_digit_indices

    def create_splits(
        self,
        train_data,
        test_data,
        val_ratio: float = 0.2,
        subsample_digit_8: float = 0.05,
    ) -> dict:
        """Create train/val/test splits with undersampling."""
        print("\nCreating data splits...")

        train_labels = np.array(train_data.targets)
        test_labels = np.array(test_data.targets)

        # Split original train into train/val (stratified)
        n_train_orig = len(train_labels)
        all_train_indices = list(range(n_train_orig))
        random.shuffle(all_train_indices)

        val_size = int(n_train_orig * val_ratio)
        val_indices = all_train_indices[:val_size]
        train_indices = all_train_indices[val_size:]

        # Undersample digit 8 in training set only
        print(f"\nUndersampling digit 8 to {subsample_digit_8:.0%} in training set...")
        train_indices = self.subsample_digit(
            train_indices, train_labels, digit=8, proportion=subsample_digit_8
        )

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

    def save_images_and_metadata(self, splits: dict) -> pd.DataFrame:
        """Save images to disk and create metadata DataFrame."""
        print("\nSaving images and creating metadata...")

        records = []

        for split_name, (dataset, indices) in splits.items():
            print(f"  Processing {split_name}...")

            for idx in tqdm(indices, desc=f"  {split_name}", leave=False):
                img, digit = dataset[idx]

                # Binary label: 0 for digits 0-4, 1 for digits 5-9
                y = 0 if digit < 5 else 1

                # Subclass is the original digit
                subclass = digit

                # Group: y * 10 + subclass (for fine-grained evaluation)
                # But for spurious correlation style: focus on digit 8
                # Group 0: class 0 (digits 0-4), non-8
                # Group 1: class 0 (digits 0-4), digit 8 -> N/A (8 is class 1)
                # Group 2: class 1 (digits 5-9), non-8
                # Group 3: class 1 (digits 5-9), digit 8 (minority)
                is_digit_8 = 1 if digit == 8 else 0
                group = y * 2 + is_digit_8

                # Save image
                img_filename = f"{split_name}_{idx:06d}.png"
                img_path = self.image_dir / img_filename
                img.save(img_path)

                records.append({
                    "img_filename": f"images/{img_filename}",
                    "y": y,
                    "digit": digit,
                    "subclass": subclass,
                    "is_digit_8": is_digit_8,
                    "group": group,
                    "split": split_name,
                })

        df = pd.DataFrame(records)
        return df

    def print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics."""
        print("\n" + "=" * 50)
        print("UMNIST Dataset Statistics")
        print("=" * 50)

        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            print(f"\n{split.upper()} ({len(split_df)} samples):")

            # Class distribution
            class_counts = split_df["y"].value_counts().sort_index()
            print(f"  Classes: {dict(class_counts)}")

            # Digit 8 distribution
            digit_8_count = (split_df["digit"] == 8).sum()
            class_1_count = (split_df["y"] == 1).sum()
            print(f"  Digit 8: {digit_8_count}/{class_1_count} of class 1 "
                  f"({100*digit_8_count/class_1_count:.1f}%)")

    def run(self):
        """Run the full generation pipeline."""
        print("=" * 50)
        print("UMNIST Dataset Generator")
        print("=" * 50)

        # Download MNIST
        train_data, test_data = self.download_mnist()

        # Create splits with undersampling
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
        description="Generate UMNIST dataset from MNIST"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/umnist",
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    generator = UMNISTGenerator(args.output_dir, args.seed)
    generator.run()


if __name__ == "__main__":
    main()
