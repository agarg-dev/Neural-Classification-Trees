#!/usr/bin/env python3
"""
Main script for running NCT on CelebA dataset
Uses YAML presets for configuration (consistent with Waterbird script)
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for nct package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct import NCTClassifier, NCTConfig
from nct.data.preprocessing import prepare_dataset
from nct.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NCT hierarchical training on CelebA data using YAML presets."
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Path to YAML preset file (e.g., script/presets/celeba_nct.yaml)",
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

    return parser.parse_args()


def main():
    """Main function to run NCT training and evaluation."""
    args = parse_args()

    print("\n--- Loading Configuration from YAML ---")
    print(f"Preset: {args.preset}")

    # Load configuration from YAML with CLI overrides
    config = NCTConfig.from_yaml(
        args.preset,
        checkpoint_dir=args.checkpoint_dir,
        attribution_dir=args.attribution_dir,
        seed=args.seed,
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
    print("Loading CelebA dataset...")
    data = prepare_dataset(
        args.data_dir,
        dataset_name="celeba",
        cache_images_in_memory=config.cache_images_in_memory,
    )
    dataset_config = data["config"]

    # Create NCT classifier with config
    classifier = NCTClassifier(config=config)

    # Train the model - pass Dataset objects directly
    print("\n--- Starting NCT Training ---")
    classifier.fit(
        X_train=data["train_dataset"],
        X_val=data["val_dataset"],
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
    )

    # Evaluate on test set
    print("\n--- Final Evaluation ---")
    results = classifier.evaluate(
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
    )

    # Save final checkpoint with iteration results for future attribution generation
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    classifier.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}")

    # Generate attribution visualizations if requested
    if args.generate_attributions:
        print("\n--- Generating Attribution Visualizations ---")

        # Create unique run directory using job_id and timestamp
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

    if "group_metrics" in results:
        group_metrics = results["group_metrics"]
        print(
            f"  Worst Group Accuracy: {group_metrics['summary']['worst_group_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
