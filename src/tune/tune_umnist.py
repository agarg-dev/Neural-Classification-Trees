#!/usr/bin/env python3
"""
Optuna hyperparameter tuning script for NCT on UMNIST dataset.
Optimizes for pseudo-label worst-group accuracy (WGA) at iteration 2.

UMNIST-specific considerations:
- Uses LeNet-5 backbone (84 output features, no pretrained weights)
- Grayscale 28x28 images - smaller model capacity needed
- Higher learning rates acceptable (training from scratch)
- Lower weight decay (smaller model needs less regularization)
- AdamW optimizer with elastic net regularization

Tuned hyperparameters (~12 total):

Learning Rates:
- initial_base_lr: Learning rate for backbone (1e-4 to 5e-2)
- initial_head_lr: Learning rate for heads (1e-4 to 5e-2)
- lr_decay_factor: LR decay between iterations (1.0 to 4.0)

Regularization:
- weight_decay: Weight decay for optimizer (1e-6 to 1e-2)
- lambda_l1: L1 regularization weight for elastic net (0 to 1e-3)
- head_dropout: Dropout rate for heads (0.0 to 0.5)

Model Architecture:
- head_hidden_dim: Hidden dimension for heads (16 to 64)

Training Configuration:
- batch_size: Batch size (64, 128, 256)
- use_class_weights: Whether to use class weights (True/False)
- max_class_weight: Maximum class weight cap [5, 10, 15, 20, 30, 40, 50]
- skip_class_weights_iter1: Skip class weights in iteration 1 (True/False)
- aux_loss_weight: Weight for auxiliary loss (0.0 to 2.0, step 0.1)
- phase1_ratio_iter1: Phase 1 ratio for iteration 1 (0.3 to 1.0)
- phase1_ratio_iter2: Phase 1 ratio for iteration 2 (0.3 to 0.8)

Fixed hyperparameters:
- num_iterations: 2
- epochs: 100 per iteration
- backbone: lenet5
- optimizer: AdamW
- regularization_type: elastic_net
- scheduler: plateau (patience=5, factor=0.5)
- routing_strategy: correctness
- hierarchical_architecture: correct_label_routing
- phase1_patience: 10
- phase2_patience: 5
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
        description="Optuna hyperparameter tuning for NCT on UMNIST (LeNet-5 backbone)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the dataset is stored.",
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
        default="nct_tune_umnist",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///outputs/optuna/tune_umnist.db",
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


def objective(trial: optuna.Trial, data: Dict, seed: int, metric: str = "wga") -> float:
    """
    Optuna objective function to maximize pseudo-label metric at iteration 2.

    Args:
        trial: Optuna trial object
        data: Dictionary containing train/val/test datasets
        seed: Random seed

    Returns:
        Pseudo-label WGA (routing accuracy to be maximized)
    """

    # ========== TUNED HYPERPARAMETERS (~12 total) ==========

    # Learning rates (higher range for training from scratch)
    initial_base_lr = trial.suggest_categorical(
        "initial_base_lr",
        [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2],
    )

    initial_head_lr = trial.suggest_categorical(
        "initial_head_lr",
        [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2],
    )

    lr_decay_factor = trial.suggest_float("lr_decay_factor", 1.0, 4.0, step=0.5)

    # Regularization (elastic net: L2 via weight_decay + L1 via lambda_l1)
    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    )

    lambda_l1 = trial.suggest_categorical(
        "lambda_l1",
        [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    )

    head_dropout = trial.suggest_categorical(
        "head_dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    # Model architecture (smaller range for LeNet-5's 84 output features)
    head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [16, 32, 48, 64])

    # Training configuration
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])

    # Class weighting (same ranges as tune_narrow.py)
    max_class_weight = trial.suggest_categorical(
        "max_class_weight", [5, 10, 15, 20, 30, 40, 50]
    )
    skip_class_weights_iter1 = trial.suggest_categorical(
        "skip_class_weights_iter1", [True, False]
    )

    # Auxiliary loss (same range as tune_narrow.py)
    aux_loss_weight = trial.suggest_float("aux_loss_weight", 0.0, 2.0, step=0.1)

    # Fixed epochs
    epochs = 100
    epochs_per_iteration = [epochs, epochs]

    # Phase ratios (tuned for both iterations)
    phase1_ratio_iter1 = trial.suggest_float("phase1_ratio_iter1", 0.3, 1.0, step=0.1)
    phase1_ratio_iter2 = trial.suggest_float("phase1_ratio_iter2", 0.3, 0.8, step=0.1)
    phase1_epochs_ratio = [phase1_ratio_iter1, phase1_ratio_iter2]

    # Fixed early stopping patience
    phase1_patience = 10
    phase2_patience = 5

    # ========== FIXED HYPERPARAMETERS ==========
    regularization_type = "elastic_net"  # Combined L1 + L2
    routing_strategy = "correctness"
    optimizer_type = "adamw"  # AdamW for UMNIST

    # Set seed for reproducibility
    set_seed(seed)

    # Create optimizer config (AdamW)
    optimizer_config = OptimizerConfig(
        type=optimizer_type,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Fixed scheduler config (plateau works well for iterative training)
    scheduler_config = SchedulerConfig(
        type="plateau",
        patience=5,
        factor=0.5,
        eta_min=1e-6,
    )

    # Add data-specific arguments. For UMNIST (grayscale), input channels should be 1.
    model_kwargs = {}
    if "mnist" in data["config"].name:
        model_kwargs["num_channels"] = 1

    # Create NCT classifier for 2 iterations with hierarchical architecture
    classifier = NCTClassifier(
        num_iterations=2,
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
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        regularization_type=regularization_type,
        lambda_l1=lambda_l1,
        routing_strategy=routing_strategy,
        hierarchical_architecture="correct_label_routing",
        aux_loss_weight=aux_loss_weight,
        max_class_weight=max_class_weight,
        skip_class_weights_iter1=skip_class_weights_iter1,
        seed=seed,
        use_class_weights=use_class_weights,
        backbone="lenet5",  # LeNet-5 for UMNIST
        pretrained=False,  # No pretrained weights for LeNet-5
        verbose=1,  # Silent mode for Optuna
        wandb_enabled=False,  # Disable wandb for trials
        save_iteration_checkpoints=False,  # Disable to avoid race conditions in parallel Optuna
        **model_kwargs,
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
        # Clean up GPU memory between trials
        del classifier
        gc.collect()
        torch.cuda.empty_cache()


def main():
    """Main function to run Optuna hyperparameter tuning."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("NCT Hyperparameter Tuning - UMNIST (LeNet-5)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset: umnist")
    print(f"  Backbone: LeNet-5 (training from scratch)")
    print(f"  Number of iterations: 2")
    print(f"  Number of trials: {args.n_trials}")
    print(f"  Epochs per iteration: 100")
    print(f"  Study name: {args.study_name}")
    print(f"  Random seed: {args.seed}")
    print(f"  Fresh study: {args.fresh_study}")
    print("=" * 60 + "\n")

    # Prepare data
    print("Loading UMNIST dataset...")
    data = prepare_dataset(args.data_dir, dataset_name="umnist")
    dataset_config = data["config"]

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=15, multivariate=True
    )

    # Ensure database directory exists
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    # If a fresh study is requested, delete the existing study first
    if args.fresh_study:
        try:
            optuna.delete_study(study_name=args.study_name, storage=args.storage)
            print(f"Deleted existing study '{args.study_name}' to start fresh.")
        except (KeyError, ValueError):  # Catches if study doesn't exist
            pass

    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize validation pseudo-label WGA
        load_if_exists=True,  # Let Optuna handle creation/loading
        sampler=sampler,
    )

    if args.fresh_study:
        print(f"Starting fresh study: {args.study_name}")
    else:
        print(f"Resuming or creating study: {args.study_name}")

    # Set up wandb callback for Optuna
    wandb_callback = None
    if WANDB_AVAILABLE:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = "outputs/wandb/optuna/umnist"
        os.environ["WANDB_CACHE_DIR"] = "outputs/.cache/wandb"

        # Create wandb directory
        os.makedirs("outputs/wandb/optuna/umnist", exist_ok=True)

        metric_name = "val_pseudo_dna" if args.metric == "dna" else "val_pseudo_wga"
        wandb_callback = WeightsAndBiasesCallback(
            metric_name=metric_name,
            wandb_kwargs={
                "project": "nct-optuna-umnist",
                "entity": "ankurgarg-university-of-calgary-in-alberta",
                "name": f"{args.study_name}",
                "tags": ["umnist", "lenet5", "hierarchical", "2-iterations"],
            },
        )
        print("Wandb callback enabled - trial metadata will be logged")

    # Run optimization
    print(f"\nStarting hyperparameter optimization for {args.n_trials} trials...")
    print(f"Optimization metric: {args.metric.upper()} ({'Difficult Node Accuracy' if args.metric == 'dna' else 'Worst-Group Accuracy'})\n")
    callbacks = [wandb_callback] if wandb_callback else []
    study.optimize(
        lambda trial: objective(trial, data, args.seed, args.metric),
        n_trials=args.n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation pseudo-label WGA: {study.best_trial.value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key:25}: {value}")

    # Compare best with and without class_weights
    print("\n" + "=" * 60)
    print("CLASS WEIGHTS COMPARISON")
    print("=" * 60)

    trials_with_weights = [
        t
        for t in study.trials
        if t.params.get("use_class_weights") == True
        and t.state == optuna.trial.TrialState.COMPLETE
    ]
    trials_without_weights = [
        t
        for t in study.trials
        if t.params.get("use_class_weights") == False
        and t.state == optuna.trial.TrialState.COMPLETE
    ]

    if trials_with_weights:
        best_with_weights = max(trials_with_weights, key=lambda t: t.value)
        print(f"\nBest WITH class_weights:")
        print(f"  Trial: {best_with_weights.number}")
        print(f"  Val Pseudo-Label WGA: {best_with_weights.value:.4f}")

    if trials_without_weights:
        best_without_weights = max(trials_without_weights, key=lambda t: t.value)
        print(f"\nBest WITHOUT class_weights:")
        print(f"  Trial: {best_without_weights.number}")
        print(f"  Val Pseudo-Label WGA: {best_without_weights.value:.4f}")

    # Train final model with best hyperparameters
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60 + "\n")

    best_params = study.best_trial.params

    set_seed(args.seed)

    optimizer_config = OptimizerConfig(
        type="adamw",
        weight_decay=best_params["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler_config = SchedulerConfig(
        type="plateau",
        patience=5,
        factor=0.5,
        eta_min=1e-6,
    )

    epochs_per_iteration = [100, 100]
    phase1_epochs_ratio = [
        best_params["phase1_ratio_iter1"],
        best_params["phase1_ratio_iter2"],
    ]

    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_id = f"trial{study.best_trial.number}_{timestamp}"
    checkpoint_dir = f"outputs/checkpoints/tune_umnist/{trial_id}"
    attribution_dir = f"outputs/attributions/tune_umnist/{trial_id}"

    final_classifier = NCTClassifier(
        num_iterations=2,
        epochs_per_iteration=epochs_per_iteration,
        batch_size=best_params["batch_size"],
        initial_base_lr=best_params["initial_base_lr"],
        initial_head_lr=best_params["initial_head_lr"],
        lr_decay_factor=best_params["lr_decay_factor"],
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        phase1_epochs_ratio=phase1_epochs_ratio,
        phase1_patience=10,
        phase2_patience=5,
        head_hidden_dim=best_params["head_hidden_dim"],
        head_dropout=best_params["head_dropout"],
        regularization_type="elastic_net",
        lambda_l1=best_params["lambda_l1"],
        routing_strategy="correctness",
        hierarchical_architecture="correct_label_routing",
        aux_loss_weight=best_params["aux_loss_weight"],
        max_class_weight=best_params["max_class_weight"],
        skip_class_weights_iter1=best_params["skip_class_weights_iter1"],
        seed=args.seed,
        use_class_weights=best_params["use_class_weights"],
        backbone="lenet5",
        pretrained=False,
        checkpoint_dir=checkpoint_dir,
        verbose=2,
        wandb_enabled=True,
        wandb_run_name=f"umnist_best_trial_{study.best_trial.number}",
        wandb_tags=["umnist", "lenet5", "best-model", "final-evaluation"],
    )

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

    # Final evaluation
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

    # Generate attributions
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

    # Save best hyperparameters
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = f"{results_dir}/{args.study_name}_best_params.txt"

    with open(output_file, "w") as f:
        f.write("Best Hyperparameters from Optuna Study - UMNIST\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Validation Pseudo-Label WGA: {study.best_trial.value:.4f}\n")
        f.write(f"Test Accuracy: {test_results['basic_metrics']['accuracy']:.4f}\n")
        if "group_metrics" in test_results:
            f.write(
                f"Test Worst Group Accuracy: {test_results['group_metrics']['summary']['worst_group_accuracy']:.4f}\n"
            )
        f.write("\nHyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"\nBest hyperparameters saved to: {output_file}")
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
