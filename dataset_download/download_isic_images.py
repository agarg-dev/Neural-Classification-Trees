#!/usr/bin/env python3
"""
Download ISIC images using ISIC API v2.
This version uses the 'isic_id' (e.g., ISIC_0026370) directly,
which is supported by v2 (unlike v1).
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class ISICImageDownloaderV2:
    # API v2 Base URL
    BASE_URL = "https://api.isic-archive.com/api/v2"

    def __init__(self, csv_path: str, output_dir: str, max_workers: int = 4):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_single_image(self, image_name: str) -> tuple:
        """
        Download a single image using API v2.
        1. GET metadata for 'ISIC_xxxx'
        2. Extract distinct download URL
        3. Download bytes
        """
        # Clean the name (remove extension if present)
        clean_name = image_name.split(".")[0]  # ISIC_0026370.jpg -> ISIC_0026370

        # Determine save path (ensure .jpg extension)
        save_name = f"{clean_name}.jpg"
        save_path = self.output_dir / save_name

        # Skip if exists
        if save_path.exists() and save_path.stat().st_size > 0:
            return (image_name, "SKIPPED")

        try:
            # STEP 1: Get Metadata by ISIC ID
            # v2 allows direct lookup: /api/v2/images/ISIC_0026370/
            meta_url = f"{self.BASE_URL}/images/{clean_name}/"
            resp = requests.get(meta_url, timeout=10)

            if resp.status_code == 404:
                return (image_name, "NOT_FOUND_IN_ARCHIVE")
            resp.raise_for_status()

            data = resp.json()

            # STEP 2: Extract Download URL
            # The structure is usually data['files']['full']['url'] or similar
            # We look for the 'files' object
            if "files" in data and "full" in data["files"]:
                download_url = data["files"]["full"]["url"]
            elif "url" in data:
                download_url = data["url"]
            else:
                # Fallback: some endpoints return a direct redirect or different structure
                # Let's try the direct download endpoint if metadata fails to provide a link
                # But standard v2 provides 'files'
                return (image_name, "NO_FILE_URL_IN_JSON")

            # STEP 3: Download the actual image content
            with requests.get(download_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            return (image_name, "SUCCESS")

        except Exception as e:
            return (image_name, f"ERROR: {str(e)}")

    def run(self):
        print(f"Reading {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        # Robust column search
        if "Image Index" in df.columns:
            col_name = "Image Index"
        elif "image" in df.columns:
            col_name = "image"
        elif "image_id" in df.columns:
            col_name = "image_id"
        else:
            print(
                "Error: CSV must have a column for image names (e.g., 'Image Index', 'image', 'image_id')"
            )
            return

        image_names = df[col_name].dropna().unique().tolist()
        print(f"Found {len(image_names)} unique images.")

        print(f"Starting download with {self.max_workers} workers...")

        results = {"SUCCESS": 0, "SKIPPED": 0, "FAILED": 0}
        failed_list = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self.process_single_image, name): name
                for name in image_names
            }

            with tqdm(total=len(image_names), unit="img") as pbar:
                for future in as_completed(future_to_name):
                    name, status = future.result()

                    if status == "SUCCESS":
                        results["SUCCESS"] += 1
                    elif status == "SKIPPED":
                        results["SKIPPED"] += 1
                    else:
                        results["FAILED"] += 1
                        failed_list.append(f"{name}: {status}")

                    pbar.update(1)

        print("\n" + "=" * 30)
        print(f"Results:")
        print(f"  Downloaded: {results['SUCCESS']}")
        print(f"  Skipped:    {results['SKIPPED']}")
        print(f"  Failed:     {results['FAILED']}")
        print("=" * 30)

        if failed_list:
            log_path = self.output_dir.parent / "download_errors.log"
            with open(log_path, "w") as f:
                f.write("\n".join(failed_list))
            print(f"Error log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    downloader = ISICImageDownloaderV2(args.csv_path, args.output_dir, args.max_workers)
    downloader.run()


if __name__ == "__main__":
    main()
