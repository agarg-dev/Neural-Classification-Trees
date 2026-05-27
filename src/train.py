#!/usr/bin/env python3
"""
NCT Training Script
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for nct package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct import NCTClassifier, NCTConfig
from nct.data.preprocessing import prepare_dataset
from nct.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NCT Training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["waterbird", "isic", "celeba", "umnist", "cmnist"],
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the dataset is stored.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides YAML).",
    )
    parser.add_argument(
        "--attribution_dir",
        type=str,
        default=None,
        help="Directory to save attributions (overrides YAML).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides YAML).",
    )
    parser.add_argument(
        "--generate_attributions",
        action="store_true",
        help="Generate Grad-CAM attribution visualizations.",
    )
    parser.add_argument(
        "--num_per_node",
        type=int,
        default=5,
        help="Number of random samples per node for attributions.",
    )
    parser.add_argument(
        "--num_transitions",
        type=int,
        default=5,
        help="Number of transition samples for attributions.",
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=None,
        help="Override aux_loss_weight from YAML (for ablation studies).",
    )
    parser.add_argument(
        "--split_node_head_type",
        type=str,
        default=None,
        choices=["true", "false", "reversed"],
        help="Override split_node_head_type from YAML (for ablation studies).",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default=None,
        choices=["linear", "mlp"],
        help="Override head_type from YAML (for ablation studies).",
    )
    parser.add_argument(
        "--phase1_uniform",
        type=float,
        default=None,
        help="Override phase1_epochs_ratio with [value]*num_iterations (for ablation studies).",
    )
    parser.add_argument(
        "--min_samples_per_node",
        type=int,
        default=None,
        help="Override min_samples_per_node from YAML (for ablation studies).",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default=None,
        choices=["none", "upsample", "downsample", "geomean"],
        help="Override sampling_strategy from YAML (for ablation studies).",
    )
    parser.add_argument(
        "--use_class_weights",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override use_class_weights from YAML (for ablation studies).",
    )

    return parser.parse_args()


def main():
    """Main function to run NCT training and evaluation."""
    args = parse_args()

    print("\n--- Loading Configuration from YAML ---")
    print(f"Preset: {args.preset}")

    # Resolve string-typed CLI overrides into the types NCTConfig expects.
    split_override = None
    if args.split_node_head_type is not None:
        split_override = {"true": True, "false": False, "reversed": "reversed"}[args.split_node_head_type]

    # phase1_uniform: peek at num_iterations from the yaml to expand into a list.
    phase1_override = None
    if args.phase1_uniform is not None:
        import yaml as _yaml
        with open(args.preset, "r") as _f:
            _n = int(_yaml.safe_load(_f).get("num_iterations", 3))
        phase1_override = [args.phase1_uniform] * _n

    use_cw_override = None
    if args.use_class_weights is not None:
        use_cw_override = args.use_class_weights == "true"

    # Load configuration from YAML with CLI overrides
    config = NCTConfig.from_yaml(
        args.preset,
        checkpoint_dir=args.checkpoint_dir,
        attribution_dir=args.attribution_dir,
        seed=args.seed,
        aux_loss_weight=args.aux_loss_weight,
        split_node_head_type=split_override,
        head_type=args.head_type,
        phase1_epochs_ratio=phase1_override,
        min_samples_per_node=args.min_samples_per_node,
        sampling_strategy=args.sampling_strategy,
        use_class_weights=use_cw_override,
    )

    # Get job ID for wandb run name
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    config.wandb_run_name = f"arc_job_{job_id}"

    print("\n--- Experiment Configuration ---")
    config_dict = config.to_dict()
    for key, value in sorted(config_dict.items()):
        print(f"  - {key:<25}: {value}")
    print("--------------------------------\n")

    # Set seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)

    # Prepare data
    print(f"Loading {args.dataset} dataset...")
    data = prepare_dataset(
        args.data_dir,
        dataset_name=args.dataset,
        cache_images_in_memory=config.cache_images_in_memory,
    )
    dataset_config = data["config"]

    # Sync num_channels and num_classes from dataset config to model config
    # This ensures the model architecture matches the data format
    if config.num_channels != dataset_config.num_channels:
        print(
            f"Note: Overriding num_channels from {config.num_channels} to {dataset_config.num_channels} (from dataset config)"
        )
        config.num_channels = dataset_config.num_channels

    if config.num_classes != dataset_config.num_classes:
        print(
            f"Note: Overriding num_classes from {config.num_classes} to {dataset_config.num_classes} (from dataset config)"
        )
        config.num_classes = dataset_config.num_classes

    # Create NCT classifier with config
    classifier = NCTClassifier(config=config)

    # Train the model - pass Dataset objects directly
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

    # Evaluate on test set
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

    # Save final checkpoint with iteration results for future attribution generation
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    classifier.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}")

    # Generate attribution visualizations if requested
    if args.generate_attributions:
        print("\n--- Generating Attribution Visualizations ---")

        # Create unique run directory: {attribution_dir}/{job_id}_{timestamp}
        # Note: config.attribution_dir already includes dataset and seed from fit()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
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

    # Print summary results
    basic_metrics = results["basic_metrics"]
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {basic_metrics['accuracy']:.2%}")
    print(f"  Test F1 Score: {basic_metrics['f1_score']:.4f}")
    if "auroc" in basic_metrics and not math.isnan(basic_metrics["auroc"]):
        print(f"  Test AUROC: {basic_metrics['auroc']:.4f}")
    if "non_patch_auroc" in basic_metrics and not math.isnan(
        basic_metrics["non_patch_auroc"]
    ):
        print(f"  Non-Patch AUROC: {basic_metrics['non_patch_auroc']:.4f}")
    if "histopath_auroc" in basic_metrics and not math.isnan(
        basic_metrics["histopath_auroc"]
    ):
        print(f"  Histopath AUROC: {basic_metrics['histopath_auroc']:.4f}")

    if "group_metrics" in results:
        group_metrics = results["group_metrics"]
        print(
            f"  Worst Group Accuracy: {group_metrics['summary']['worst_group_accuracy']:.2%}"
        )

    print(f"\n  Wall-Clock Time:")
    print(f"    total_train_s: {train_elapsed:.1f}s ({train_elapsed/60:.1f}m)")
    print(f"\n  Inference: {inference_elapsed:.2f}s ({n_test/max(inference_elapsed, 1e-6):.0f} samples/s)")


if __name__ == "__main__":
    main()
