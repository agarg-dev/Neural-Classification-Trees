#!/usr/bin/env python3
"""
Optuna hyperparameter tuning script for NCT - Version 2 (Research-Based)
Based on Group DRO paper findings and spurious correlation literature.

Optimizes iteration 2 pseudo-label worst-group accuracy (WGA) on validation set,
then evaluates best config on test set.

Tuned hyperparameters (14 total):
Learning Rates:
- initial_base_lr: Learning rate for base model (1e-6 to 1e-3)
- initial_head_lr: Learning rate for classification heads (1e-6 to 1e-2)
- lr_decay_factor: Factor for decaying learning rates between iterations [1.0, 1.5, 2.0, 2.5]

Optimizer & Scheduler:
- optimizer_type: Optimizer type [adamw, sgd]
- scheduler_type: LR scheduler type [none, plateau, cosine]

Regularization:
- weight_decay: Weight decay for optimizer (focused on 1e-4 to 1.0 - critical for WGA)

Model Architecture:
- head_hidden_dim: Hidden dimension for classification heads [32, 64, 128]
- head_dropout: Dropout rate for classification heads (0.0 to 0.5, step 0.1)

Training Configuration:
- phase1_ratio_iter2: Phase 1 ratio for iteration 2 (0.2 to 0.8, step 0.1)
- max_class_weight: Maximum class weight cap [5, 10, 20, 30, 50]
- skip_class_weights_iter1: Skip class weights in iteration 1 (True/False)
- aux_loss_weight: Auxiliary loss weight (0.0 to 2.0)

Fixed hyperparameters:
- num_iterations: 2
- epochs: 100 (fixed for both iterations)
- batch_size: 128
- use_class_weights: True
- phase1_ratio_iter1: 1.0 (only phase 1 in iteration 1)
- phase1_patience: 15
- phase2_patience: 15
- model_selection_criterion: 'auto'
- regularization_type: 'l2' (no L1, based on Group DRO paper)
- routing_strategy: 'correctness'
- hierarchical_architecture: 'correct_label_routing'

Research basis:
- Group DRO paper uses fixed LR (no scheduler) and high weight decay (0.1-1.0)
- AdamW outperforms SGD on OOD by ~4% on vision models (but SGD still used in some papers)
- Scheduler interaction with WGA is unclear, so we tune it

Note: Optimizes for pseudo-label worst-group accuracy (WGA) at iteration 2.
      After optimization, runs final evaluation with best parameters and generates attributions.
"""

import argparse
import gc
import os
import sys
from typing import Dict

import optuna
import torch

try:
    import wandb
    from wandb.integration.optuna import WeightsAndBiasesCallback

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add project root directory to path for nct package import
# From src/tune/, we need to go up 2 levels to reach project root
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from nct import NCTClassifier, NCTConfig
from nct.core.config import OptimizerConfig, SchedulerConfig
from nct.data.preprocessing import prepare_dataset
from nct.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna narrow hyperparameter tuning for NCT (v2 - research-based)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the dataset is stored.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="waterbird",
        choices=["waterbird", "celeba", "isic"],
        help="Dataset to use for optimization (default: waterbird).",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="nct_tune_narrow_v2",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///script/tune/databases/tune_narrow_v2.db",
        help="Optuna storage (SQLite database file).",
    )
    parser.add_argument(
        "--fresh_study",
        action="store_true",
        help="Start a fresh study (ignore previous results).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="wga",
        choices=["wga", "dna"],
        help="Optimization metric: 'wga' (worst-group accuracy) or 'dna' (difficult node accuracy). Default: wga",
    )
    return parser.parse_args()


def objective(
    trial: optuna.Trial, data: Dict, seed: int, dataset: str, metric: str = "wga"
) -> float:
    """
    Optuna objective function to maximize pseudo-label metric at iteration 2.

    Args:
        trial: Optuna trial object
        data: Dictionary containing train/val/test datasets
        seed: Random seed
        dataset: Dataset name
        metric: Optimization metric

    Returns:
        Pseudo-label WGA or DNA (routing accuracy to be maximized)
    """

    # ========== TUNED HYPERPARAMETERS (14 total) ==========

    # Learning rates
    initial_base_lr = trial.suggest_categorical(
        "initial_base_lr",
        [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    )

    initial_head_lr = trial.suggest_categorical(
        "initial_head_lr",
        [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    )

    # Simplified lr_decay_factor
    lr_decay_factor = trial.suggest_categorical(
        "lr_decay_factor", [1.0, 1.5, 2.0, 2.5]
    )

    # Optimizer type (AdamW vs SGD)
    optimizer_type = trial.suggest_categorical("optimizer_type", ["adamw", "sgd"])

    # Scheduler type (none = fixed LR as in Group DRO paper)
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["none", "plateau", "cosine"]
    )

    # Regularization - focused on higher values per Group DRO findings
    # High weight decay (0.1-1.0) is critical for worst-group accuracy
    weight_decay = trial.suggest_categorical(
        "weight_decay", [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0]
    )

    # Model architecture - simplified
    head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [32, 64, 128])
    head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5, step=0.1)

    # Training configuration
    phase1_ratio_iter2 = trial.suggest_float("phase1_ratio_iter2", 0.2, 0.8, step=0.1)

    # Class weighting
    max_class_weight = trial.suggest_categorical(
        "max_class_weight", [5, 10, 20, 30, 50]
    )
    skip_class_weights_iter1 = trial.suggest_categorical(
        "skip_class_weights_iter1", [True, False]
    )

    # Auxiliary loss
    aux_loss_weight = trial.suggest_float("aux_loss_weight", 0.0, 2.0, step=0.2)

    # ========== FIXED HYPERPARAMETERS ==========
    # Training configuration
    batch_size = 128
    use_class_weights = True
    epochs = 100
    epochs_per_iteration = [epochs, epochs]
    phase1_ratio_iter1 = 1.0  # Fixed to 1.0 (only phase 1 in iteration 1)
    phase1_epochs_ratio = [phase1_ratio_iter1, phase1_ratio_iter2]
    phase1_patience = 15
    phase2_patience = 15
    model_selection_criterion = "auto"
    num_workers = 16

    # Regularization - L2 only (no elastic net, per Group DRO paper)
    regularization_type = "l2"
    lambda_l1 = 0.0

    # Routing
    routing_strategy = "correctness"

    # Hierarchical architecture
    hierarchical_architecture = "correct_label_routing"

    # Set seed for reproducibility
    set_seed(seed)

    # Create optimizer config
    if optimizer_type == "adamw":
        optimizer_config = OptimizerConfig(
            type="adamw",
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    else:  # sgd
        optimizer_config = OptimizerConfig(
            type="sgd",
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    # Create scheduler config based on scheduler_type
    if scheduler_type == "none":
        scheduler_config = None  # Fixed learning rate
    elif scheduler_type == "plateau":
        scheduler_config = SchedulerConfig(
            type="plateau", patience=10, factor=0.1, eta_min=1e-6
        )
    elif scheduler_type == "cosine":
        scheduler_config = SchedulerConfig(type="cosine", eta_min=1e-6)
    else:
        scheduler_config = None

    # Create NCT classifier for 2 iterations with hierarchical architecture
    classifier = NCTClassifier(
        num_iterations=2,  # 2 iterations with hierarchical routing
        epochs_per_iteration=epochs_per_iteration,
        batch_size=batch_size,
        initial_base_lr=initial_base_lr,
        initial_head_lr=initial_head_lr,
        lr_decay_factor=lr_decay_factor,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        phase1_epochs_ratio=phase1_epochs_ratio,
        phase1_patience=phase1_patience,
        phase2_patience=phase2_patience,
        model_selection_criterion=model_selection_criterion,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        regularization_type=regularization_type,
        lambda_l1=lambda_l1,
        routing_strategy=routing_strategy,
        hierarchical_architecture=hierarchical_architecture,
        aux_loss_weight=aux_loss_weight,
        max_class_weight=max_class_weight,
        skip_class_weights_iter1=skip_class_weights_iter1,
        num_workers=num_workers,
        seed=seed,
        use_class_weights=use_class_weights,
        verbose=1,  # Silent mode for Optuna
        wandb_enabled=False,  # Disable wandb for trials (prevents 100+ runs)
        save_iteration_checkpoints=False,  # Disable to avoid race conditions in parallel Optuna
    )

    try:
        # Train the model
        training_results = classifier.fit(
            X_train=data["train_dataset"],
            X_val=data["val_dataset"],
            X_test=None,  # Don't use test set during tuning
            group_labels=None,
            dataset_name=data["config"].name,
            class_names=data["config"].class_names,
            group_names=data["config"].group_names,
            test_subgroups=data.get("test_subgroups"),
        )

        # Extract validation pseudo-label WGA from iteration 2
        val_pseudo_metrics = training_results.get("val_pseudo_metrics")

        if val_pseudo_metrics is None:
            raise ValueError(
                "Validation pseudo-label metrics not available. Ensure num_iterations >= 2."
            )

        # Select metric based on argument
        if metric == "dna":
            val_metric = val_pseudo_metrics["difficult_node_accuracy"]
        else:  # default: wga
            val_metric = val_pseudo_metrics["worst_group_accuracy"]

        return val_metric

    finally:
        # Clean up GPU memory between trials (runs even if exception occurs)
        del classifier
        gc.collect()
        torch.cuda.empty_cache()


def main():
    """Main function to run Optuna hyperparameter tuning."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("NCT Narrow Hyperparameter Tuning v2 (Research-Based)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Number of iterations: 2")
    print(f"  Number of trials: {args.n_trials}")
    print(f"  Epochs: 100 (fixed)")
    print(f"  Batch size: 128 (fixed)")
    print(f"  Study name: {args.study_name}")
    print(f"  Random seed: {args.seed}")
    print(f"  Fresh study: {args.fresh_study}")
    print(f"\nKey changes from v1:")
    print(f"  - Optimizer: tunable [adamw, sgd]")
    print(f"  - Scheduler: tunable [none, plateau, cosine]")
    print(f"  - Weight decay: focused on 1e-4 to 1.0 (per Group DRO)")
    print(f"  - Regularization: L2 only (no elastic net)")
    print("=" * 60 + "\n")

    # Prepare data
    print(f"Loading {args.dataset} dataset...")
    data = prepare_dataset(args.data_dir, dataset_name=args.dataset)
    dataset_config = data["config"]

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=20, multivariate=True
    )

    # Ensure database directory exists
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    # Create Optuna study (fresh study or resume based on flag)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize validation pseudo-label WGA
        load_if_exists=not args.fresh_study,
        sampler=sampler,
    )

    if args.fresh_study:
        print(f"Starting fresh study: {args.study_name}")
    else:
        print(f"Resuming or creating study: {args.study_name}")

    # Set up wandb callback for Optuna (lightweight trial metadata logging)
    wandb_callback = None
    if WANDB_AVAILABLE:
        # Configure wandb for offline mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = "script/wandb_logs/optuna/nct"
        os.environ["WANDB_CACHE_DIR"] = "script/.cache/wandb"

        metric_name = "val_pseudo_dna" if args.metric == "dna" else "val_pseudo_wga"
        wandb_callback = WeightsAndBiasesCallback(
            metric_name=metric_name,
            wandb_kwargs={
                "project": "nct-optuna",
                "entity": "ankurgarg-university-of-calgary-in-alberta",
                "name": f"{args.study_name}",
                "tags": [
                    "hierarchical",
                    "2-iterations",
                    "tune-narrow-v2",
                    "research-based",
                    "group-dro-inspired",
                ],
            },
        )
        print("Wandb callback enabled - trial metadata will be logged")

    # Run optimization
    print(f"\nStarting hyperparameter optimization for {args.n_trials} trials...")
    print(
        f"Optimization metric: {args.metric.upper()} ({'Difficult Node Accuracy' if args.metric == 'dna' else 'Worst-Group Accuracy'})\n"
    )
    callbacks = [wandb_callback] if wandb_callback else []
    study.optimize(
        lambda trial: objective(trial, data, args.seed, args.dataset, args.metric),
        n_trials=args.n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nOverall best trial: {study.best_trial.number}")
    print(f"Overall best validation pseudo-label WGA: {study.best_trial.value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key:25}: {value}")

    # Train final model with best hyperparameters on full data and evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60 + "\n")

    best_params = study.best_trial.params

    # Set seed for final training
    set_seed(args.seed)

    # Create optimizer config based on best params
    if best_params["optimizer_type"] == "adamw":
        optimizer_config = OptimizerConfig(
            type="adamw",
            weight_decay=best_params["weight_decay"],
            betas=(0.9, 0.999),
        )
    else:  # sgd
        optimizer_config = OptimizerConfig(
            type="sgd",
            weight_decay=best_params["weight_decay"],
            momentum=0.9,
            nesterov=True,
        )

    # Create scheduler config based on best params
    if best_params["scheduler_type"] == "none":
        scheduler_config = None
    elif best_params["scheduler_type"] == "plateau":
        scheduler_config = SchedulerConfig(
            type="plateau", patience=10, factor=0.1, eta_min=1e-6
        )
    elif best_params["scheduler_type"] == "cosine":
        scheduler_config = SchedulerConfig(type="cosine", eta_min=1e-6)
    else:
        scheduler_config = None

    # Reconstruct lists from best parameters
    epochs_per_iteration = [100, 100]
    phase1_epochs_ratio = [1.0, best_params["phase1_ratio_iter2"]]

    # Organize checkpoint and attribution directories by trial number
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_id = f"trial{study.best_trial.number}_{timestamp}"
    checkpoint_dir = f"script/checkpoints/tune_narrow_v2/{trial_id}"
    attribution_dir = f"script/attributions/tune_narrow_v2/{trial_id}"

    # Create final classifier
    final_classifier = NCTClassifier(
        num_iterations=2,
        epochs_per_iteration=epochs_per_iteration,
        batch_size=128,
        initial_base_lr=best_params["initial_base_lr"],
        initial_head_lr=best_params["initial_head_lr"],
        lr_decay_factor=best_params["lr_decay_factor"],
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        phase1_epochs_ratio=phase1_epochs_ratio,
        phase1_patience=15,
        phase2_patience=15,
        model_selection_criterion="auto",
        head_hidden_dim=best_params["head_hidden_dim"],
        head_dropout=best_params["head_dropout"],
        regularization_type="l2",
        lambda_l1=0.0,
        routing_strategy="correctness",
        hierarchical_architecture="correct_label_routing",
        aux_loss_weight=best_params["aux_loss_weight"],
        max_class_weight=best_params["max_class_weight"],
        skip_class_weights_iter1=best_params["skip_class_weights_iter1"],
        seed=args.seed,
        use_class_weights=True,
        checkpoint_dir=checkpoint_dir,
        verbose=2,  # Detailed output for final training
        wandb_enabled=True,  # Enable wandb for final best model
        wandb_run_name=f"tune_narrow_v2_best_trial_{study.best_trial.number}",
        wandb_tags=[
            "hierarchical",
            "2-iterations",
            "tune-narrow-v2",
            "best-model",
            "final-evaluation",
        ],
    )

    # Train final model
    print("Training final model with best hyperparameters...")
    training_results = final_classifier.fit(
        X_train=data["train_dataset"],
        X_val=data["val_dataset"],
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
        test_subgroups=data.get("test_subgroups"),
    )

    print(f"\nTraining complete. Checkpoints saved to: {checkpoint_dir}")

    # Final evaluation on test set
    print("\n--- Final Test Set Evaluation ---")
    test_results = final_classifier.evaluate(
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
        test_subgroups=data.get("test_subgroups"),
    )

    print(f"\nFinal Test Results:")
    print(f"  Test Accuracy: {test_results['basic_metrics']['accuracy']:.2%}")
    print(f"  Test F1 Score: {test_results['basic_metrics']['f1_score']:.4f}")

    if "group_metrics" in test_results:
        group_metrics = test_results["group_metrics"]
        print(
            f"  Worst Group Accuracy: {group_metrics['summary']['worst_group_accuracy']:.2%}"
        )

    # Generate attributions for the best model
    print("\n" + "=" * 60)
    print("GENERATING ATTRIBUTIONS")
    print("=" * 60 + "\n")

    print("Generating attributions for best model...")
    final_classifier.generate_attribution_visualizations(
        X_test=data["test_dataset"],
        save_dir=attribution_dir,
        methods="all_layer",
        class_names=dataset_config.class_names,
    )
    print(f"Attributions saved to: {attribution_dir}")

    # Save best hyperparameters to file
    output_file = f"script/wandb_logs/results/{args.study_name}_best_params.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write("Best Hyperparameters from Optuna Study (tune_narrow_v2)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Research basis: Group DRO paper, spurious correlation literature\n\n")
        f.write(f"Validation Pseudo-Label WGA: {study.best_trial.value:.4f}\n")
        f.write(f"Test Accuracy: {test_results['basic_metrics']['accuracy']:.4f}\n")
        if "group_metrics" in test_results:
            f.write(
                f"Test Worst Group Accuracy: {group_metrics['summary']['worst_group_accuracy']:.4f}\n"
            )
        f.write("\nTuned Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nFixed Hyperparameters:\n")
        f.write(f"  batch_size: 128\n")
        f.write(f"  epochs: 100\n")
        f.write(f"  use_class_weights: True\n")
        f.write(f"  phase1_ratio_iter1: 1.0\n")
        f.write(f"  phase1_patience: 15\n")
        f.write(f"  phase2_patience: 15\n")
        f.write(f"  model_selection_criterion: auto\n")
        f.write(f"  regularization_type: l2\n")
        f.write(f"  lambda_l1: 0.0\n")
        f.write(f"  hierarchical_architecture: correct_label_routing\n")

    print(f"\nBest hyperparameters saved to: {output_file}")
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
