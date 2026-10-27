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
        self.verbose = verbose
        self.wandb_config = wandb_config
        self.run = None
        self.enabled = WANDB_AVAILABLE

        if not WANDB_AVAILABLE:
            if verbose >= 1:
                print("Warning: wandb not installed. Install with: pip install wandb")
            self.enabled = False
            return

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
        iter2_level_wga: Optional[float] = None,
        iter2_level_auroc: Optional[float] = None,
        iter2_level_dna: Optional[float] = None,
        iter2_level_dnf1: Optional[float] = None,
    ):
        """Log per-epoch metrics to wandb."""
        if not self.enabled or not self.run:
            return

        metrics = {
            f"iteration_{iteration}/phase_{phase}/train_loss": train_loss,
            f"iteration_{iteration}/phase_{phase}/val_loss": val_loss,
            f"iteration_{iteration}/phase_{phase}/val_accuracy": val_acc,
            f"iteration_{iteration}/phase_{phase}/val_f1": val_f1,
            f"iteration_{iteration}/phase_{phase}/pseudo_worst_group_accuracy": worst_group_acc,
            f"iteration_{iteration}/phase_{phase}/pseudo_difficult_node_accuracy": difficult_node_acc,
            f"iteration_{iteration}/phase_{phase}/pseudo_difficult_node_f1": difficult_node_f1,
            f"iteration_{iteration}/phase_{phase}/base_lr": base_lr,
            f"iteration_{iteration}/phase_{phase}/head_lr": head_lr,
            "epoch": epoch,
            "iteration": iteration,
        }

        if train_aux_loss > 0 or val_aux_loss > 0:
            metrics[f"iteration_{iteration}/phase_{phase}/train_aux_loss"] = (
                train_aux_loss
            )
            metrics[f"iteration_{iteration}/phase_{phase}/val_aux_loss"] = val_aux_loss

        if iteration >= 3 and iter2_level_wga is not None:
            metrics[
                f"iteration_{iteration}/phase_{phase}/iter2_pseudo_worst_group_accuracy"
            ] = iter2_level_wga
        if iteration >= 3 and iter2_level_dna is not None:
            metrics[
                f"iteration_{iteration}/phase_{phase}/iter2_pseudo_difficult_node_accuracy"
            ] = iter2_level_dna
        if iteration >= 3 and iter2_level_dnf1 is not None:
            metrics[
                f"iteration_{iteration}/phase_{phase}/iter2_pseudo_difficult_node_f1"
            ] = iter2_level_dnf1

        if iteration >= 2 and iter2_level_auroc is not None and not math.isnan(iter2_level_auroc):
            metrics[
                f"iteration_{iteration}/phase_{phase}/iter2_level_worst_group_auroc"
            ] = iter2_level_auroc

        for group_id, acc in group_accuracies.items():
            metrics[
                f"iteration_{iteration}/phase_{phase}/group_{group_id}_accuracy"
            ] = acc

        wandb.log(metrics)

    def log_wga_spearman(self, spearman_r: float, n_epochs: int, variant: str = "full"):
        """Log Spearman correlation of pseudo metric vs true metric."""
        if not self.enabled or not self.run:
            return
        suffix = "" if variant == "full" else f"_{variant}"
        wandb.log({
            f"wga_correlation/spearman_r{suffix}": spearman_r,
            f"wga_correlation/n_epochs{suffix}": n_epochs,
        })
        if self.run:
            self.run.summary[f"wga_spearman_r{suffix}"] = spearman_r

    def log_iteration_summary(
        self,
        iteration: int,
        test_metrics: Dict[str, Any],
        group_metrics: Dict[str, Any],
        node_metrics: Dict[str, Any],
        training_time: float,
        num_epochs_trained: int,
        early_stopped: bool,
        consolidated_group_metrics: Optional[Dict[str, Any]] = None,
        multi_spurious_metrics: Optional[Dict[str, Any]] = None,
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

        if "non_patch_auroc" in test_metrics and not math.isnan(test_metrics["non_patch_auroc"]):
            metrics[f"iteration_{iteration}/non_patch_auroc"] = test_metrics["non_patch_auroc"]
        if "histopath_auroc" in test_metrics and not math.isnan(test_metrics["histopath_auroc"]):
            metrics[f"iteration_{iteration}/histopath_auroc"] = test_metrics["histopath_auroc"]

        if "pseudo_difficult_node_acc" in test_metrics:
            metrics[f"iteration_{iteration}/pseudo_difficult_node_acc"] = test_metrics["pseudo_difficult_node_acc"]
        if "pseudo_worst_group_acc" in test_metrics:
            metrics[f"iteration_{iteration}/pseudo_worst_group_acc"] = test_metrics["pseudo_worst_group_acc"]
        if "pseudo_iter2_level_worst_group_acc" in test_metrics:
            metrics[f"iteration_{iteration}/pseudo_iter2_level_worst_group_acc"] = test_metrics[
                "pseudo_iter2_level_worst_group_acc"
            ]

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

        if consolidated_group_metrics is not None:
            metrics[f"iteration_{iteration}/consolidated_worst_group_accuracy"] = (
                consolidated_group_metrics.get("worst_group_accuracy", 0)
            )
            per_class = consolidated_group_metrics.get("per_class", {})
            for class_id, class_metrics in per_class.items():
                metrics[f"iteration_{iteration}/consolidated_class_{class_id}_aligned_accuracy"] = (
                    class_metrics.get("aligned_accuracy", 0)
                )
                metrics[f"iteration_{iteration}/consolidated_class_{class_id}_conflicting_accuracy"] = (
                    class_metrics.get("conflicting_accuracy", 0)
                )

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

        if multi_spurious_metrics is not None:
            metrics[f"iteration_{iteration}/id_accuracy"] = multi_spurious_metrics["id_accuracy"]
            for name in [*multi_spurious_metrics["correlator_names"], "joint"]:
                metrics[f"iteration_{iteration}/{name}_gap"] = multi_spurious_metrics["gaps"][name]
                metrics[f"iteration_{iteration}/{name}_wga"] = multi_spurious_metrics["worst_group_accuracies"][name]

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

    def log_depth_selection_summary(
        self,
        enabled: bool,
        selected_iter: Optional[int],
        headline_iter: Optional[int],
        stop_reason: str,
        per_iter_stats: Dict[int, Dict[str, Any]],
        last_trained_iter: int,
        num_iterations: int,
        early_stopped: bool,
        metric: str = "wga",
    ) -> None:
        if not self.enabled or not self.run:
            return

        summary = self.run.summary
        value_key = f"pseudo_{metric}"
        summary["depth_selection/enabled"] = enabled
        summary["depth_selection/metric"] = metric
        summary["depth_selection/early_stopped"] = early_stopped
        summary["depth_selection/last_trained_iter"] = last_trained_iter
        summary["depth_selection/num_iterations"] = num_iterations
        summary["depth_selection/stop_reason"] = stop_reason
        if selected_iter is not None:
            summary["depth_selection/selected_iter"] = selected_iter
        if headline_iter is not None:
            summary["depth_selection/headline_iter"] = headline_iter

        for it, st in per_iter_stats.items():
            prefix = f"depth_selection/iter_{it}"
            if st.get("value") is not None:
                summary[f"{prefix}/{value_key}"] = st["value"]
            if st.get("n_worst") is not None:
                summary[f"{prefix}/n_worst"] = st["n_worst"]
            if st.get("epsilon") is not None:
                summary[f"{prefix}/epsilon"] = st["epsilon"]
            if st.get("decision") is not None:
                summary[f"{prefix}/decision"] = st["decision"]

    def finish(self, total_training_time: Optional[float] = None):
        """Finish wandb run and set summary metrics for easy dashboard access."""
        if not self.enabled or not self.run:
            return

        if total_training_time is not None:
            wandb.log({"total_training_time": total_training_time})
            wandb.run.summary["total_training_time"] = total_training_time

        wandb.finish()
