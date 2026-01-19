#!/usr/bin/env python3
"""
Optuna hyperparameter tuning script for NCT - 2 Iterations with Hierarchical Architecture
Optimizes iteration 2 pseudo-label worst-group accuracy (WGA) on validation set, then evaluates best config on test set.

Tuned hyperparameters (17 total):
Learning Rates:
- initial_base_lr: Learning rate for base model (1e-6 to 1e-3)
- initial_head_lr: Learning rate for classification heads (1e-6 to 1e-2)
- lr_decay_factor: Factor for decaying learning rates between iterations (1.0 to 5.0)

Regularization:
- weight_decay: Weight decay for optimizer (1e-5 to 10.0)
- head_dropout: Dropout rate for classification heads (0.0 to 0.5)

Model Architecture:
- head_hidden_dim: Hidden dimension for classification heads (64 to 512)

Training Configuration:
- batch_size: Batch size for training (32, 64, 128, 256)
- use_class_weights: Whether to use class weights for imbalanced data (True/False)
- aux_loss_weight: Weight for auxiliary parent layer loss (0.0 to 1.0)
- phase1_ratio_iter1: Phase 1 ratio for iteration 1 (0.5 to 1.0)
- phase1_ratio_iter2: Phase 1 ratio for iteration 2 (0.2 to 0.8)
- phase1_patience: Early stopping patience for phase 1 (5, 10, 15, 20)
- phase2_patience: Early stopping patience for phase 2 (5, 10, 15, 20)

LR Scheduler (NEW - most important!):
- scheduler_type: Type of LR scheduler (plateau, cosine, cosine_warmup)
- eta_min: Minimum learning rate (0.0, 1e-7, 1e-6, 1e-5)
- plateau_patience: Patience for plateau scheduler (2 to 15) [conditional]
- plateau_factor: LR reduction factor for plateau (0.1, 0.2, 0.3, 0.5, 0.7) [conditional]
- warmup_ratio: Warmup ratio for cosine_warmup (0.05 to 0.20) [conditional]

Fixed hyperparameters:
- num_iterations: 2
- epochs: 100 (fixed for both iterations)
- regularization_type: 'l2'
- lambda_l1: 0.0
- routing_strategy: 'correctness'
- optimizer: AdamW
- hierarchical_architecture: 'correct_label_routing' (true label-based hierarchical routing)

Note: Optimizes for pseudo-label worst-group accuracy (WGA) at iteration 2.
      After optimization, runs final evaluation with/without class weights and generates attributions.
      Scheduler hyperparameters are conditional based on scheduler_type selected.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nct import NCTClassifier, NCTConfig
from nct.core.config import OptimizerConfig, SchedulerConfig
from nct.data.preprocessing import prepare_dataset
from nct.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for NCT (2 iterations with hierarchical architecture)"
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
        default="nct_tune_broad",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///script/tune/databases/tune_broad.db",
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
        metric: Optimization metric - 'wga' (worst-group accuracy) or 'dna' (difficult node accuracy)

    Returns:
        Selected metric value (routing accuracy to be maximized)
    """

    # ========== TUNED HYPERPARAMETERS (17 total) ==========

    # Learning rates
    initial_base_lr = trial.suggest_categorical(
        "initial_base_lr",
        [1e-6, 2e-6, 5e-6, 8e-6, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 2e-4, 5e-4, 8e-4, 1e-3],
    )

    initial_head_lr = trial.suggest_categorical(
        "initial_head_lr",
        [
            1e-6,
            2e-6,
            5e-6,
            8e-6,
            1e-5,
            2e-5,
            5e-5,
            8e-5,
            1e-4,
            2e-4,
            5e-4,
            8e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
        ],
    )

    lr_decay_factor = trial.suggest_float("lr_decay_factor", 1.0, 5.0, step=0.25)

    # Regularization
    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            5e-1,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
    )

    head_dropout = trial.suggest_categorical("head_dropout", [0.0, 0.2, 0.4, 0.6, 0.8])

    # Model architecture
    head_hidden_dim = trial.suggest_int("head_hidden_dim", 64, 512, step=64)

    # Training configuration
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])

    aux_loss_weight = trial.suggest_categorical(
        "aux_loss_weight", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    )

    # Fixed epochs to 100
    epochs = 100
    epochs_per_iteration = [epochs, epochs]

    # Phase ratios
    phase1_ratio_iter1 = 1.0  # Fixed to 1.0 (only phase 1 in iteration 1)
    phase1_ratio_iter2 = trial.suggest_float("phase1_ratio_iter2", 0.2, 0.8, step=0.05)
    phase1_epochs_ratio = [phase1_ratio_iter1, phase1_ratio_iter2]

    # Early stopping patience
    phase1_patience = trial.suggest_categorical("phase1_patience", [5, 10, 15, 20])
    phase2_patience = trial.suggest_categorical("phase2_patience", [5, 10, 15, 20])

    # LR Scheduler
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["plateau", "cosine", "cosine_warmup"]
    )

    # Shared scheduler hyperparameter
    eta_min = trial.suggest_categorical("eta_min", [0.0, 1e-7, 1e-6, 1e-5])

    # Scheduler-specific hyperparameters (conditional)
    if scheduler_type == "plateau":
        scheduler_patience = trial.suggest_int("plateau_patience", 2, 15, step=1)
        scheduler_factor = trial.suggest_categorical(
            "plateau_factor", [0.1, 0.2, 0.3, 0.5, 0.7]
        )
        scheduler_config = SchedulerConfig(
            type="plateau",
            patience=scheduler_patience,
            factor=scheduler_factor,
            eta_min=eta_min,
        )
    elif scheduler_type == "cosine":
        # T_max auto-calculated to epochs=100
        scheduler_config = SchedulerConfig(type="cosine", eta_min=eta_min)
    elif scheduler_type == "cosine_warmup":
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.20, step=0.05)
        scheduler_config = SchedulerConfig(
            type="cosine_warmup", warmup_ratio=warmup_ratio, eta_min=eta_min
        )

    # ========== FIXED HYPERPARAMETERS ==========
    regularization_type = "l2"
    lambda_l1 = 0.0
    routing_strategy = "correctness"
    optimizer_type = "adamw"
    momentum = 0.9
    nesterov = True

    # Set seed for reproducibility
    set_seed(seed)

    # Create optimizer config
    optimizer_config = OptimizerConfig(
        type=optimizer_type,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
    )

    # Scheduler config already created above based on scheduler_type

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
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        regularization_type=regularization_type,
        lambda_l1=lambda_l1,
        routing_strategy=routing_strategy,
        hierarchical_architecture="correct_label_routing",
        aux_loss_weight=aux_loss_weight,
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

        # Extract validation pseudo-label metric from iteration 2
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
    print("NCT Hyperparameter Tuning with Optuna (2 Iterations - Hierarchical)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Number of iterations: 2")
    print(f"  Number of trials: {args.n_trials}")
    print(f"  Epochs range: 50-300 (tuned)")
    print(f"  Study name: {args.study_name}")
    print(f"  Random seed: {args.seed}")
    print(f"  Fresh study: {args.fresh_study}")
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
        direction="maximize",  # Maximize validation pseudo-label WGA (iteration 2 routing accuracy)
        load_if_exists=not args.fresh_study,  # Start fresh if flag is set
        sampler=sampler,
    )

    if args.fresh_study:
        print(f"Starting fresh study: {args.study_name}")
    else:
        print(f"Resuming or creating study: {args.study_name}")

    # Set up wandb callback for Optuna (lightweight trial metadata logging)
    metric_name = "val_pseudo_dna" if args.metric == "dna" else "val_pseudo_wga"
    wandb_callback = None
    if WANDB_AVAILABLE:
        # Configure wandb for offline mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = "script/wandb_logs/optuna/nct"
        os.environ["WANDB_CACHE_DIR"] = "script/.cache/wandb"

        wandb_callback = WeightsAndBiasesCallback(
            metric_name=metric_name,
            wandb_kwargs={
                "project": "nct-optuna",
                "entity": "ankurgarg-university-of-calgary-in-alberta",
                "name": f"{args.study_name}",
                "tags": [
                    "hierarchical",
                    "2-iterations",
                    f"iteration2-pseudo-{args.metric}-optimization",
                ],
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

    print(f"\nOverall best trial: {study.best_trial.number}")
    print(f"Overall best validation pseudo-label WGA: {study.best_trial.value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key:25}: {value}")

    # Compare best with and without class_weights
    print("\n" + "=" * 60)
    print("CLASS WEIGHTS COMPARISON")
    print("=" * 60)

    # Find best with class_weights=True
    trials_with_weights = [
        t
        for t in study.trials
        if t.params.get("use_class_weights") == True
        and t.state == optuna.trial.TrialState.COMPLETE
    ]
    # Find best without class_weights (False)
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
        print(f"  Params: {best_with_weights.params}")

    if trials_without_weights:
        best_without_weights = max(trials_without_weights, key=lambda t: t.value)
        print(f"\nBest WITHOUT class_weights:")
        print(f"  Trial: {best_without_weights.number}")
        print(f"  Val Pseudo-Label WGA: {best_without_weights.value:.4f}")
        print(f"  Params: {best_without_weights.params}")

    print("=" * 60)

    # Train final model with best hyperparameters on full data and evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60 + "\n")

    best_params = study.best_trial.params

    # Set seed for final training
    set_seed(args.seed)

    # Create optimizer config (using defaults + tuned weight_decay)
    optimizer_config = OptimizerConfig(
        type="adamw",
        weight_decay=best_params["weight_decay"],
        momentum=0.9,
        nesterov=True,
    )

    # Create scheduler config (using plateau with eta_min)
    scheduler_config = SchedulerConfig(type="plateau", patience=3, eta_min=1e-6)

    # Reconstruct lists from best parameters
    # epochs is fixed at 100 (not tuned), phase1_ratio_iter1 is fixed at 1.0
    epochs_per_iteration = [100, 100]
    phase1_epochs_ratio = [
        1.0,  # phase1_ratio_iter1 is fixed at 1.0
        best_params["phase1_ratio_iter2"],
    ]

    # Organize checkpoint and attribution directories by trial number
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_id = f"trial{study.best_trial.number}_{timestamp}"
    checkpoint_dir = f"script/checkpoints/tune_broad/{trial_id}"
    attribution_dir = f"script/attributions/tune_broad/{trial_id}"

    # Create final classifier (using defaults + tuned hyperparameters)
    final_classifier = NCTClassifier(
        num_iterations=2,
        epochs_per_iteration=epochs_per_iteration,
        batch_size=64,  # Default
        initial_base_lr=best_params["initial_base_lr"],
        initial_head_lr=best_params["initial_head_lr"],
        lr_decay_factor=best_params["lr_decay_factor"],
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        phase1_epochs_ratio=phase1_epochs_ratio,
        phase1_patience=10,  # Default
        phase2_patience=10,  # Default
        head_hidden_dim=best_params["head_hidden_dim"],
        head_dropout=0.3,  # Fixed
        regularization_type="l2",  # Fixed
        lambda_l1=0.0,  # Fixed
        routing_strategy="correctness",  # Fixed
        hierarchical_architecture="correct_label_routing",
        aux_loss_weight=best_params["aux_loss_weight"],
        seed=args.seed,
        use_class_weights=best_params["use_class_weights"],
        checkpoint_dir=checkpoint_dir,
        verbose=2,  # Detailed output for final training
        wandb_enabled=True,  # Enable wandb for final best model
        wandb_run_name=f"hierarchical_best_trial_{study.best_trial.number}",
        wandb_tags=["hierarchical", "2-iterations", "best-model", "final-evaluation"],
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

    # Second final evaluation with opposite class_weights setting
    print("\n" + "=" * 60)
    print(
        f"SECOND EVALUATION WITH use_class_weights={not best_params['use_class_weights']}"
    )
    print("=" * 60 + "\n")

    # Create second classifier with opposite class_weights
    second_checkpoint_dir = (
        f"script/checkpoints/tune_broad/{trial_id}_opposite_weights"
    )
    second_attribution_dir = (
        f"script/attributions/tune_broad/{trial_id}_opposite_weights"
    )

    second_classifier = NCTClassifier(
        num_iterations=2,
        epochs_per_iteration=epochs_per_iteration,
        batch_size=64,
        initial_base_lr=best_params["initial_base_lr"],
        initial_head_lr=best_params["initial_head_lr"],
        lr_decay_factor=best_params["lr_decay_factor"],
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        phase1_epochs_ratio=phase1_epochs_ratio,
        phase1_patience=10,
        phase2_patience=10,
        head_hidden_dim=best_params["head_hidden_dim"],
        head_dropout=0.3,
        regularization_type="l2",
        lambda_l1=0.0,
        routing_strategy="correctness",
        hierarchical_architecture="correct_label_routing",
        aux_loss_weight=best_params["aux_loss_weight"],
        seed=args.seed,
        use_class_weights=not best_params["use_class_weights"],  # Opposite setting
        checkpoint_dir=second_checkpoint_dir,
        verbose=2,
        wandb_enabled=True,
        wandb_run_name=f"hierarchical_best_trial_{study.best_trial.number}_opposite_weights",
        wandb_tags=[
            "hierarchical",
            "2-iterations",
            "best-model",
            "opposite-class-weights",
        ],
    )

    print(f"Training with use_class_weights={not best_params['use_class_weights']}...")
    second_training_results = second_classifier.fit(
        X_train=data["train_dataset"],
        X_val=data["val_dataset"],
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
        test_subgroups=data.get("test_subgroups"),
    )

    second_test_results = second_classifier.evaluate(
        X_test=data["test_dataset"],
        group_labels=data["group_test"],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
        test_subgroups=data.get("test_subgroups"),
    )

    print(f"\nSecond Evaluation Test Results:")
    print(f"  Test Accuracy: {second_test_results['basic_metrics']['accuracy']:.2%}")
    print(f"  Test F1 Score: {second_test_results['basic_metrics']['f1_score']:.4f}")
    if "group_metrics" in second_test_results:
        print(
            f"  Worst Group Accuracy: {second_test_results['group_metrics']['summary']['worst_group_accuracy']:.2%}"
        )

    # Generate attributions for both models using method='all'
    print("\n" + "=" * 60)
    print("GENERATING ATTRIBUTIONS")
    print("=" * 60 + "\n")

    print("Generating attributions for first model (best trial)...")
    final_classifier.generate_attribution_visualizations(
        X_test=data["test_dataset"],
        save_dir=attribution_dir,
        methods="all_layer",
        class_names=dataset_config.class_names,
    )
    print(f"Attributions saved to: {attribution_dir}")

    print("\nGenerating attributions for second model (opposite class weights)...")
    second_classifier.generate_attribution_visualizations(
        X_test=data["test_dataset"],
        save_dir=second_attribution_dir,
        methods="all_layer",
        class_names=dataset_config.class_names,
    )
    print(f"Attributions saved to: {second_attribution_dir}")

    # Save best hyperparameters to file
    output_file = f"script/wandb_logs/results/{args.study_name}_best_params.txt"
    with open(output_file, "w") as f:
        f.write("Best Hyperparameters from Optuna Study\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Validation Pseudo-Label WGA: {study.best_trial.value:.4f}\n")
        f.write(
            f"Test Accuracy (with best class_weights={best_params['use_class_weights']}): {test_results['basic_metrics']['accuracy']:.4f}\n"
        )
        f.write(
            f"Test Accuracy (with opposite class_weights={not best_params['use_class_weights']}): {second_test_results['basic_metrics']['accuracy']:.4f}\n\n"
        )
        f.write("Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"\nBest hyperparameters saved to: {output_file}")
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
