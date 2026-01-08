#!/usr/bin/env python3
"""
Download ISIC dataset from the official API.
Based on the paper's methodology: extract images with 'benign' or 'malignant' labels.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class ISICDownloader:
    """Download ISIC dataset from official API."""

    BASE_URL = "https://isic-archive.com/api/v1"

    def __init__(self, output_dir: str, max_workers: int = 8):
        """
        Initialize ISIC downloader.

        Args:
            output_dir: Directory to save images and metadata
            max_workers: Number of parallel download threads
        """
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "ISIC_2019_Training_Input"
        self.max_workers = max_workers

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def get_all_images(self, limit: int = 50000):
        """
        Fetch all image metadata from ISIC API.

        Args:
            limit: Maximum number of images to fetch

        Returns:
            List of image metadata dictionaries
        """
        print(f"Fetching image metadata from {self.BASE_URL}/image...")

        url = f"{self.BASE_URL}/image"
        params = {
            "limit": limit,
            "offset": 0,
            "sort": "name",
            "sortdir": 1,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        images = response.json()
        print(f"Found {len(images)} total images")

        return images

    def filter_benign_malignant(self, images: list) -> list:
        """
        Filter images to only include those with 'benign' or 'malignant' labels.

        Args:
            images: List of image metadata

        Returns:
            Filtered list of images
        """
        print("\nFiltering images with 'benign' or 'malignant' labels...")

        filtered = []
        for img in tqdm(images, desc="Filtering"):
            # Get detailed metadata for each image
            img_id = img["_id"]
            detail_url = f"{self.BASE_URL}/image/{img_id}"

            try:
                response = requests.get(detail_url)
                response.raise_for_status()
                metadata = response.json()

                # Check benign_malignant field
                benign_malignant = (
                    metadata.get("meta", {}).get("clinical", {}).get("benign_malignant")
                )

                if benign_malignant in ["benign", "malignant"]:
                    filtered.append(metadata)

                # Rate limiting
                time.sleep(0.05)

            except Exception as e:
                print(f"\nWarning: Failed to fetch metadata for {img_id}: {e}")
                continue

        print(f"Filtered to {len(filtered)} images with benign/malignant labels")
        return filtered

    def create_metadata_csv(self, images: list) -> pd.DataFrame:
        """
        Create metadata DataFrame from image list.

        Args:
            images: List of image metadata dictionaries

        Returns:
            DataFrame with processed metadata
        """
        print("\nCreating metadata CSV...")

        records = []
        for img in images:
            meta = img.get("meta", {})
            clinical = meta.get("clinical", {})
            acquisition = meta.get("acquisition", {})

            record = {
                "Image Index": img["name"] + ".jpg",
                "_id": img["_id"],
                "age_approx": clinical.get("age_approx"),
                "anatom_site_general": clinical.get("anatom_site_general"),
                "benign_malignant": clinical.get("benign_malignant"),
                "diagnosis": clinical.get("diagnosis"),
                "sex": clinical.get("sex"),
                "image_type": acquisition.get("image_type"),
                "updated": img.get("updated"),
            }

            records.append(record)

        df = pd.DataFrame(records)

        # Create binary diagnosis label (1 = malignant, 0 = benign)
        df["Diagnosis"] = (df["benign_malignant"] == "malignant").astype(int)

        print(f"Created metadata for {len(df)} images")
        print(f"  Benign: {(df['Diagnosis'] == 0).sum()}")
        print(f"  Malignant: {(df['Diagnosis'] == 1).sum()}")

        return df

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create train/val/test splits as described in the paper.

        Args:
            df: Metadata DataFrame
            train_ratio: Training set ratio (default 0.8)
            val_ratio: Validation set ratio (default 0.1)
            seed: Random seed

        Returns:
            DataFrame with 'fold' column added
        """
        print(
            f"\nCreating data splits (train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio})..."
        )

        df = df.copy()
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        df["fold"] = "test"
        df.loc[:train_end, "fold"] = "train"
        df.loc[train_end:val_end, "fold"] = "val"

        print(f"  Train: {(df['fold'] == 'train').sum()}")
        print(f"  Val: {(df['fold'] == 'val').sum()}")
        print(f"  Test: {(df['fold'] == 'test').sum()}")

        return df

    def download_image(self, image_id: str, image_name: str) -> bool:
        """
        Download a single image.

        Args:
            image_id: ISIC image ID
            image_name: Filename to save as

        Returns:
            True if successful, False otherwise
        """
        image_path = self.image_dir / image_name

        # Skip if already exists
        if image_path.exists():
            return True

        try:
            url = f"{self.BASE_URL}/image/{image_id}/download"
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            print(f"\nFailed to download {image_name}: {e}")
            return False

    def download_images(self, df: pd.DataFrame):
        """
        Download all images in parallel.

        Args:
            df: Metadata DataFrame with Image Index and _id columns
        """
        print(f"\nDownloading {len(df)} images using {self.max_workers} workers...")

        tasks = [(row["_id"], row["Image Index"]) for _, row in df.iterrows()]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_image, img_id, img_name): img_name
                for img_id, img_name in tasks
            }

            successful = 0
            failed = 0

            with tqdm(total=len(tasks), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                    pbar.update(1)

        print(f"\nDownload complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

    def run(self, skip_download: bool = False):
        """
        Run the full download pipeline.

        Args:
            skip_download: If True, only create metadata without downloading images
        """
        print("=" * 60)
        print("ISIC Dataset Downloader")
        print("=" * 60)

        # Step 1: Get all images
        all_images = self.get_all_images()

        # Step 2: Filter to benign/malignant only
        filtered_images = self.filter_benign_malignant(all_images)

        # Step 3: Create metadata DataFrame
        df = self.create_metadata_csv(filtered_images)

        # Step 4: Create train/val/test splits
        df = self.create_train_val_test_splits(df)

        # Step 5: Save metadata
        csv_path = self.output_dir / "labels.csv"
        df.to_csv(csv_path, index=True)
        print(f"\nMetadata saved to: {csv_path}")

        # Step 6: Download images
        if not skip_download:
            self.download_images(df)
        else:
            print("\nSkipping image download (--skip-download flag set)")

        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"Images: {self.image_dir}")
        print(f"Metadata: {csv_path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download ISIC dataset from official API"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ankur.garg1/NCT/data/isic",
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel download threads",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Only create metadata CSV without downloading images",
    )

    args = parser.parse_args()

    downloader = ISICDownloader(args.output_dir, args.max_workers)
    downloader.run(skip_download=args.skip_download)


if __name__ == "__main__":
    main()
