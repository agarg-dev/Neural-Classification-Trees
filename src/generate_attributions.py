#!/usr/bin/env python3
"""Regenerate attribution visualizations from a saved NCT checkpoint."""

import argparse
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct import NCTClassifier
from nct.data.preprocessing import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="waterbird",
        choices=["waterbird", "celeba", "isic", "umnist", "cmnist", "urbancars"],
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--attribution_dir", type=str, default="./attributions")
    parser.add_argument("--num_per_node", type=int, default=5)
    parser.add_argument("--num_transitions", type=int, default=5)
    parser.add_argument(
        "--methods",
        type=str,
        default="all_layer",
        help='Comma-separated list or one of "all", "all_layer", "all_pixel".',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = os.path.join(args.attribution_dir, f"{job_id}_{timestamp}")
    os.makedirs(base_run_dir, exist_ok=True)
    print(f"Output directory: {base_run_dir}")

    raw = args.methods.lower().strip()
    if raw in {"all", "all_layer", "all_pixel"}:
        methods_arg = raw
    else:
        methods_arg = [m.strip() for m in raw.split(",")]

    data = prepare_dataset(args.data_dir, dataset_name=args.dataset)
    test_dataset = data["test_dataset"]
    dataset_config = data["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {args.checkpoint}")
    classifier = NCTClassifier.from_checkpoint(args.checkpoint, device=device)

    classifier.generate_attribution_visualizations(
        X_test=test_dataset,
        save_dir=base_run_dir,
        num_per_node=args.num_per_node,
        num_transitions=args.num_transitions,
        class_names=dataset_config.class_names,
        methods=methods_arg,
    )

    print(f"Done. Output: {base_run_dir}")


if __name__ == "__main__":
    main()
