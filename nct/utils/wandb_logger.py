"""Weights & Biases logging utilities for NCT training."""

import math
import os
import time
from typing import Any, Dict, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Handles all wandb logging for NCT training."""

    def __init__(
        self, config: Dict[str, Any], wandb_config: Dict[str, Any], verbose: int = 1
    ):
        """
        Initialize wandb logger.

        Args:
            config: Training configuration to log
            wandb_config: Wandb-specific config (project, entity, run_name, tags)
            verbose: Verbosity level for warnings
        """
        self.verbose = verbose
        self.wandb_config = wandb_config
        self.run = None
        self.enabled = WANDB_AVAILABLE

        if not WANDB_AVAILABLE:
            if verbose >= 1:
                print("Warning: wandb not installed. Install with: pip install wandb")
            self.enabled = False
            return

        # Initialize wandb run
        self.run = wandb.init(
            project=wandb_config.get("project", "nct-experiments"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("run_name"),
            tags=wandb_config.get("tags", []),
            config=config,
            reinit=True,
        )

    def log_epoch_metrics(
        self,
        iteration: int,
        phase: int,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        worst_group_acc: float,
        group_accuracies: Dict[int, float],
        base_lr: float,
        head_lr: float,
        train_aux_loss: float = 0.0,
        val_aux_loss: float = 0.0,
        difficult_node_acc: float = 0.0,
        difficult_node_f1: float = 0.0,
    ):
        """Log per-epoch metrics to wandb including auxiliary loss, DNA, and DNF1."""
        if not self.enabled or not self.run:
            return

        metrics = {
            f"iteration_{iteration}/phase_{phase}/train_loss": train_loss,
            f"iteration_{iteration}/phase_{phase}/val_loss": val_loss,
            f"iteration_{iteration}/phase_{phase}/val_accuracy": val_acc,
            f"iteration_{iteration}/phase_{phase}/val_f1": val_f1,
            f"iteration_{iteration}/phase_{phase}/worst_group_accuracy": worst_group_acc,
            f"iteration_{iteration}/phase_{phase}/difficult_node_accuracy": difficult_node_acc,
            f"iteration_{iteration}/phase_{phase}/difficult_node_f1": difficult_node_f1,
            f"iteration_{iteration}/phase_{phase}/base_lr": base_lr,
            f"iteration_{iteration}/phase_{phase}/head_lr": head_lr,
            "epoch": epoch,
            "iteration": iteration,
        }

        # Add auxiliary loss metrics if non-zero (iteration > 1 with hierarchical architecture)
        if train_aux_loss > 0 or val_aux_loss > 0:
            metrics[f"iteration_{iteration}/phase_{phase}/train_aux_loss"] = (
                train_aux_loss
            )
            metrics[f"iteration_{iteration}/phase_{phase}/val_aux_loss"] = val_aux_loss

        # Add per-group accuracies
        for group_id, acc in group_accuracies.items():
            metrics[
                f"iteration_{iteration}/phase_{phase}/group_{group_id}_accuracy"
            ] = acc

        wandb.log(metrics)

    def log_iteration_summary(
        self,
        iteration: int,
        test_metrics: Dict[str, Any],
        group_metrics: Dict[str, Any],
        node_metrics: Dict[str, Any],
        training_time: float,
        num_epochs_trained: int,
        early_stopped: bool,
    ):
        """Log summary metrics after iteration completes."""
        if not self.enabled or not self.run:
            return

        metrics = {
            f"iteration_{iteration}/test_accuracy": test_metrics.get("accuracy", 0),
            f"iteration_{iteration}/test_f1": test_metrics.get("f1_score", 0),
            f"iteration_{iteration}/test_precision": test_metrics.get("precision", 0),
            f"iteration_{iteration}/test_recall": test_metrics.get("recall", 0),
            f"iteration_{iteration}/test_auroc": test_metrics.get("auroc", 0),
            f"iteration_{iteration}/training_time": training_time,
            f"iteration_{iteration}/epochs_trained": num_epochs_trained,
            f"iteration_{iteration}/early_stopped": early_stopped,
            f"iteration_{iteration}/num_nodes": node_metrics.get("total_nodes", 0),
            f"iteration_{iteration}/node_utilization": node_metrics.get(
                "node_utilization", 0
            ),
            f"iteration_{iteration}/active_nodes": node_metrics.get("used_nodes", 0),
            "iteration": iteration,
        }

        # Add subgroup AUROC metrics if available (e.g., ISIC dataset)
        if "non_patch_auroc" in test_metrics and not math.isnan(test_metrics["non_patch_auroc"]):
            metrics[f"iteration_{iteration}/non_patch_auroc"] = test_metrics["non_patch_auroc"]
        if "histopath_auroc" in test_metrics and not math.isnan(test_metrics["histopath_auroc"]):
            metrics[f"iteration_{iteration}/histopath_auroc"] = test_metrics["histopath_auroc"]

        # Add pseudo-label metrics if available (iteration 2+)
        if "pseudo_difficult_node_acc" in test_metrics:
            metrics[f"iteration_{iteration}/pseudo_difficult_node_acc"] = test_metrics["pseudo_difficult_node_acc"]
        if "pseudo_worst_group_acc" in test_metrics:
            metrics[f"iteration_{iteration}/pseudo_worst_group_acc"] = test_metrics["pseudo_worst_group_acc"]

        # Add group metrics
        if group_metrics is not None and "summary" in group_metrics:
            summary = group_metrics["summary"]
            metrics[f"iteration_{iteration}/worst_group_accuracy"] = summary.get(
                "worst_group_accuracy", 0
            )
            metrics[f"iteration_{iteration}/best_group_accuracy"] = summary.get(
                "best_group_accuracy", 0
            )
            metrics[f"iteration_{iteration}/mean_group_accuracy"] = summary.get(
                "mean_group_accuracy", 0
            )
            metrics[f"iteration_{iteration}/group_accuracy_std"] = summary.get(
                "group_accuracy_std", 0
            )
            # Add AUROC group metrics if available
            if "worst_group_auroc" in summary:
                metrics[f"iteration_{iteration}/worst_group_auroc"] = summary.get(
                    "worst_group_auroc", 0
                )
                metrics[f"iteration_{iteration}/best_group_auroc"] = summary.get(
                    "best_group_auroc", 0
                )
                metrics[f"iteration_{iteration}/mean_group_auroc"] = summary.get(
                    "mean_group_auroc", 0
                )

        # Add per-group test accuracies and AUROC
        if group_metrics is not None and "group_metrics" in group_metrics:
            for group_name, group_data in group_metrics["group_metrics"].items():
                group_id = group_data.get("group_id", 0)
                metrics[f"iteration_{iteration}/test_group_{group_id}_accuracy"] = (
                    group_data.get("accuracy", 0)
                )
                metrics[f"iteration_{iteration}/test_group_{group_id}_f1"] = (
                    group_data.get("f1_score", 0)
                )
                if "auroc" in group_data:
                    auroc_val = group_data.get("auroc", float("nan"))
                    if not math.isnan(auroc_val):
                        metrics[
                            f"iteration_{iteration}/test_group_{group_id}_auroc"
                        ] = auroc_val

        # Add per-node metrics
        if "node_distribution" in node_metrics:
            for node_name, node_data in node_metrics["node_distribution"].items():
                metrics[f"iteration_{iteration}/{node_name}/population"] = (
                    node_data.get("count", 0)
                )
                metrics[f"iteration_{iteration}/{node_name}/percentage"] = (
                    node_data.get("percentage", 0)
                )
                if "accuracy" in node_data:
                    metrics[f"iteration_{iteration}/{node_name}/accuracy"] = node_data[
                        "accuracy"
                    ]

        wandb.log(metrics)

    def log_transition_statistics(
        self, prev_iteration: int, curr_iteration: int, transition_stats: Dict[str, Any]
    ):
        """Log transition statistics between iterations."""
        if not self.enabled or not self.run:
            return

        metrics = {}
        transition_key = f"transition_{prev_iteration}_to_{curr_iteration}"

        # Overall transitions
        if "overall" in transition_stats:
            overall = transition_stats["overall"]
            metrics[f"{transition_key}/forgetting_rate"] = overall.get(
                "forgetting_rate", 0
            )
            metrics[f"{transition_key}/stability_rate"] = overall.get(
                "stability_rate", 0
            )
            metrics[f"{transition_key}/learning_rate"] = overall.get("learning_rate", 0)
            metrics[f"{transition_key}/forgetting_count"] = overall[
                "correct_to_incorrect"
            ]
            metrics[f"{transition_key}/learning_count"] = overall[
                "incorrect_to_correct"
            ]
            metrics[f"{transition_key}/consistent_correct"] = overall[
                "correct_to_correct"
            ]
            metrics[f"{transition_key}/consistent_incorrect"] = overall[
                "incorrect_to_incorrect"
            ]

        # Per-group transitions
        if "per_group" in transition_stats:
            for group_id, group_stats in transition_stats["per_group"].items():
                metrics[f"{transition_key}/group_{group_id}_forgetting_rate"] = (
                    group_stats.get("forgetting_rate", 0)
                )
                metrics[f"{transition_key}/group_{group_id}_learning_rate"] = (
                    group_stats.get("learning_rate", 0)
                )
                metrics[f"{transition_key}/group_{group_id}_forgetting_count"] = (
                    group_stats["correct_to_incorrect"]
                )
                metrics[f"{transition_key}/group_{group_id}_learning_count"] = (
                    group_stats["incorrect_to_correct"]
                )

        if metrics:
            wandb.log(metrics)

    def log_attribution_image(
        self, image_path: str, caption: str, iteration: Optional[int] = None
    ):
        """Log attribution visualization image to wandb."""
        if not self.enabled or not self.run:
            return

        if not os.path.exists(image_path):
            if self.verbose >= 1:
                print(f"Warning: Attribution image not found: {image_path}")
            return

        key = f"attributions/iteration_{iteration}" if iteration else "attributions"
        wandb.log({key: wandb.Image(image_path, caption=caption)})

    def log_weight_statistics(self, iteration: int, weight_stats: Dict[str, float]):
        """Log pre-computed weight change statistics."""
        if not self.enabled or not self.run:
            return

        metrics = {
            f"weight_changes/iteration_{iteration}/{k}": v
            for k, v in weight_stats.items()
        }
        wandb.log(metrics)

    def log_routing_statistics(self, routing_stats: Dict[str, int]):
        """Log routing statistics for correctness_certainty routing."""
        if not self.enabled or not self.run:
            return

        wandb.log(
            {
                "routing/num_wrong": routing_stats["num_wrong"],
                "routing/num_uncertain": routing_stats["num_uncertain"],
                "routing/num_difficult": routing_stats["num_difficult"],
                "routing/difficult_ratio": routing_stats["num_difficult"]
                / routing_stats["total"],
            }
        )

    def finish(self, total_training_time: Optional[float] = None):
        """Finish wandb run and set summary metrics for easy dashboard access."""
        if not self.enabled or not self.run:
            return

        # Log total training time before finishing
        if total_training_time is not None:
            wandb.log({"total_training_time": total_training_time})
            wandb.run.summary["total_training_time"] = total_training_time

        wandb.finish()
