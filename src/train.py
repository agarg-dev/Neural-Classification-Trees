#!/usr/bin/env python3
"""NCT training entrypoint."""

import argparse
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct import NCTClassifier, NCTConfig
from nct.data.preprocessing import prepare_dataset
from nct.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NCT Training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["waterbird", "celeba", "isic", "umnist", "cmnist"],
    )
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--attribution_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generate_attributions", action="store_true")
    parser.add_argument("--num_per_node", type=int, default=5)
    parser.add_argument("--num_transitions", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n--- Loading Configuration from YAML ---")
    print(f"Preset: {args.preset}")

    config = NCTConfig.from_yaml(
        args.preset,
        checkpoint_dir=args.checkpoint_dir,
        attribution_dir=args.attribution_dir,
        seed=args.seed,
    )

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    config.wandb_run_name = f"nct_{job_id}"

    print("\n--- Experiment Configuration ---")
    for key, value in sorted(config.to_dict().items()):
        print(f"  - {key:<25}: {value}")
    print("--------------------------------\n")

    if config.seed is not None:
        set_seed(config.seed)

    print(f"Loading {args.dataset} dataset...")
    data = prepare_dataset(
        args.data_dir,
        dataset_name=args.dataset,
        cache_images_in_memory=config.cache_images_in_memory,
    )
    dataset_config = data["config"]

    if config.num_channels != dataset_config.num_channels:
        config.num_channels = dataset_config.num_channels
    if config.num_classes != dataset_config.num_classes:
        config.num_classes = dataset_config.num_classes

    classifier = NCTClassifier(config=config)

    print("\n--- Starting NCT Training ---")
    train_start = time.time()
    classifier.fit(
        X_train=data["train_dataset"],
        X_val=data["val_dataset"],
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        val_group_labels=data.get("group_val"),
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
        test_subgroups=data.get("test_subgroups"),
    )
    train_elapsed = time.time() - train_start

    print("\n--- Final Evaluation ---")
    inference_start = time.time()
    results = classifier.evaluate(
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
        test_subgroups=data.get("test_subgroups"),
    )
    inference_elapsed = time.time() - inference_start
    n_test = len(data["test_dataset"])

    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    classifier.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}")

    if args.generate_attributions:
        print("\n--- Generating Attribution Visualizations ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{job_id}_{timestamp}"
        run_attribution_dir = os.path.join(config.attribution_dir, run_id)
        print(f"Saving attributions to: {run_attribution_dir}")

        classifier.generate_attribution_visualizations(
            X_test=data["test_dataset"],
            save_dir=run_attribution_dir,
            num_per_node=args.num_per_node,
            num_transitions=args.num_transitions,
            class_names=dataset_config.class_names,
        )

    print("\n--- Experiment Complete ---")
    basic_metrics = results["basic_metrics"]
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {basic_metrics['accuracy']:.2%}")
    print(f"  Test F1 Score: {basic_metrics['f1_score']:.4f}")
    if "auroc" in basic_metrics and not math.isnan(basic_metrics["auroc"]):
        print(f"  Test AUROC: {basic_metrics['auroc']:.4f}")
    if "non_patch_auroc" in basic_metrics and not math.isnan(basic_metrics["non_patch_auroc"]):
        print(f"  Non-Patch AUROC: {basic_metrics['non_patch_auroc']:.4f}")
    if "histopath_auroc" in basic_metrics and not math.isnan(basic_metrics["histopath_auroc"]):
        print(f"  Histopath AUROC: {basic_metrics['histopath_auroc']:.4f}")

    if "group_metrics" in results:
        group_metrics = results["group_metrics"]
        print(f"  Worst Group Accuracy: {group_metrics['summary']['worst_group_accuracy']:.2%}")

    print(f"\n  Wall-Clock Time:")
    print(f"    total_train_s: {train_elapsed:.1f}s ({train_elapsed/60:.1f}m)")
    print(f"  Inference: {inference_elapsed:.2f}s ({n_test/max(inference_elapsed, 1e-6):.0f} samples/s)")


if __name__ == "__main__":
    main()
