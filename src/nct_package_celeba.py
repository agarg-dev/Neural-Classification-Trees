#!/usr/bin/env python3
"""
Main script for running NCT on CelebA dataset
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct import NCTClassifier, NCTConfig
from nct.core.config import OptimizerConfig, SchedulerConfig
from nct.data.preprocessing import prepare_dataset
from nct.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NCT hierarchical training on CelebA data.")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory where the dataset is stored.')
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of training iterations.')
    parser.add_argument('--epochs_per_iteration', type=int, nargs='+', default=[20, 20, 20],
                       help='List of epochs for each iteration.')
    parser.add_argument('--initial_base_lr', type=float, default=1e-4, help='Initial learning rate for the CNN base.')
    parser.add_argument('--initial_head_lr', type=float, default=1e-3, help='Initial learning rate for the output heads.')
    parser.add_argument('--lr_decay_factor', type=float, default=2.0, help='Factor to DIVIDE base LR by between iterations.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for L2 regularization.')
    parser.add_argument('--regularization_type', type=str, default='l2',
                       choices=['l1', 'l2', 'elastic_net', 'none'],
                       help='Type of regularization: l1, l2, elastic_net, or none.')
    parser.add_argument('--lambda_l1', type=float, default=0.0, help='L1 regularization coefficient.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--phase1_epochs_ratio', type=float, nargs='+', default=[0.5],
                       help='Ratio of epochs for phase 1. Single value or list (0.0=skip phase1, 1.0=skip phase2).')
    parser.add_argument('--phase1_patience', type=int, default=10,
                       help='Early stopping patience for phase 1.')
    parser.add_argument('--phase2_patience', type=int, default=10,
                       help='Early stopping patience for phase 2.')
    parser.add_argument('--optimizer_type', type=str, default='adamw', help='Type of optimizer to use.')
    parser.add_argument('--scheduler_type', type=str, default='plateau', help='Type of scheduler to use.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for learning rate scheduler.')
    parser.add_argument('--save_iteration_checkpoints', action='store_true', default=True,
                       help='Save model checkpoints after each iteration.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save iteration checkpoints.')
    parser.add_argument('--generate_attributions', action='store_true', help='Generate Grad-CAM attribution visualizations.')
    parser.add_argument('--attribution_dir', type=str, default='./attributions', help='Directory to save attribution visualizations.')
    parser.add_argument('--num_per_node', type=int, default=5, help='Number of random samples per node for attributions.')
    parser.add_argument('--num_transitions', type=int, default=5, help='Number of transition samples for attributions.')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6, help='Minimum learning rate for scheduler.')
    parser.add_argument('--routing_strategy', type=str, default='argmax', choices=['argmax', 'confidence'],
                       help='Routing strategy for sample assignment.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for routing.')
    parser.add_argument('--aux_loss_weight', type=float, default=0.3,
                       help='Weight for auxiliary parent loss (0.0 = disabled).')
    parser.add_argument('--hierarchical_architecture', type=str, default='none',
                        choices=['none', 'prediction_routing', 'correct_label_routing'],
                        help='Hierarchical architecture mode')

    return parser.parse_args()


def main():
    """Main function to run NCT training and evaluation."""
    args = parse_args()

    print("\n--- Experiment Configuration ---")
    for key, value in sorted(vars(args).items()):
        print(f"  - {key:<25}: {value}")
    print("--------------------------------\n")

    if args.seed is not None:
        set_seed(args.seed)

    print("Loading CelebA dataset...")
    data = prepare_dataset(args.data_dir, dataset_name='celeba')
    dataset_config = data['config']

    optimizer_config = OptimizerConfig(
        type=args.optimizer_type,
        weight_decay=args.weight_decay
    )

    scheduler_config = SchedulerConfig(
        type=args.scheduler_type,
        patience=args.patience,
    )

    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    classifier = NCTClassifier(
        num_iterations=args.num_iterations,
        epochs_per_iteration=args.epochs_per_iteration,
        batch_size=args.batch_size,
        initial_base_lr=args.initial_base_lr,
        initial_head_lr=args.initial_head_lr,
        lr_decay_factor=args.lr_decay_factor,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        phase1_epochs_ratio=args.phase1_epochs_ratio,
        phase1_patience=args.phase1_patience,
        phase2_patience=args.phase2_patience,
        seed=args.seed,
        use_class_weights=True,
        regularization_type=args.regularization_type,
        lambda_l1=args.lambda_l1,
        aux_loss_weight=args.aux_loss_weight,
        save_iteration_checkpoints=args.save_iteration_checkpoints,
        checkpoint_dir=args.checkpoint_dir,
        routing_strategy=args.routing_strategy,
        confidence_threshold=args.confidence_threshold,
        verbose=2,
        wandb_enabled=True,
        wandb_project="nct-celeba",
        wandb_entity="ankurgarg-university-of-calgary-in-alberta",
        wandb_run_name=f"arc_job_{job_id}",
        wandb_tags=["arc", "celeba", "resnet50"],
        log_attributions=True,
        log_epoch_metrics=True,
        hierarchical_architecture=args.hierarchical_architecture
    )

    print("\n--- Starting NCT Training ---")
    classifier.fit(
        X_train=data['train_dataset'],
        X_val=data['val_dataset'],
        X_test=data['test_dataset'],
        group_labels=data['group_test'],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
    )

    print("\n--- Final Evaluation ---")
    results = classifier.evaluate(
        X_test=data['test_dataset'],
        group_labels=data['group_test'],
        dataset_name=dataset_config.name,
        class_names=dataset_config.class_names,
        group_names=dataset_config.group_names,
    )

    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    classifier.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}")

    if args.generate_attributions:
        print("\n--- Generating Attribution Visualizations ---")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{job_id}_{timestamp}"
        run_attribution_dir = os.path.join(args.attribution_dir, run_id)

        print(f"Saving attributions to: {run_attribution_dir}")

        classifier.generate_attribution_visualizations(
            X_test=data['test_dataset'],
            save_dir=run_attribution_dir,
            num_per_node=args.num_per_node,
            num_transitions=args.num_transitions,
            class_names=dataset_config.class_names,
        )

    print("\n--- Experiment Complete ---")

    basic_metrics = results['basic_metrics']
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {basic_metrics['accuracy']:.2%}")
    print(f"  Test F1 Score: {basic_metrics['f1_score']:.4f}")

    if 'group_metrics' in results:
        group_metrics = results['group_metrics']
        print(f"  Worst Group Accuracy: {group_metrics['summary']['worst_group_accuracy']:.2%}")


if __name__ == '__main__':
    main()
