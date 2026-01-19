"""Logging utilities for NCT training."""

import math
import time
from typing import Any, Dict, Optional

import numpy as np


class NCTLogger:
    """Logger for NCT training with different verbosity levels and optional wandb logging."""

    def __init__(self, verbose: int, config):
        """
        Initialize logger.

        Args:
            verbose: 0=silent, 1=progress, 2=detailed
            config: NCTConfig object (wandb settings extracted internally)
        """
        self.verbose = verbose
        self.config = config
        self.start_time = None
        self.iteration_start_time = None

        # Extract wandb config internally
        self.wandb_logger = None
        self._wandb_enabled = config.wandb_enabled
        self._wandb_config = {}

        if self._wandb_enabled:
            # Extract only wandb-specific settings
            self._wandb_config = {
                "project": config.wandb_project,
                "entity": config.wandb_entity,
                "run_name": config.wandb_run_name,
                "tags": config.wandb_tags,
            }

    def training_start(self, config: dict):
        """Print training start message and initialize wandb."""
        self.start_time = time.time()

        # Initialize wandb logger if enabled
        if self._wandb_enabled:
            from .wandb_logger import WandbLogger

            self.wandb_logger = WandbLogger(config, self._wandb_config, self.verbose)

        if self.verbose == 0:
            return

        if self.verbose >= 1:
            print("\nNCT Classifier Training")
            print("=" * 50)
            print(
                f"Iterations: {config['num_iterations']}  |  "
                f"Backbone: {config['backbone']}  |  "
                f"Device: {config.get('device', 'auto')}"
            )

            # Log architecture mode
            arch_mode = config.get("hierarchical_architecture", "none")
            if arch_mode == "correct_label_routing":
                print("Architecture: Hierarchical (correct label routing)")
            else:  # "none"
                print("Architecture: Simple (position-based split)")

            if self.wandb_logger and self.wandb_logger.enabled:
                print(f"Wandb: {self._wandb_config.get('project', 'nct-experiments')}")
            print()

    def iteration_start(
        self,
        iteration: int,
        total_iterations: int,
        num_nodes: int,
        epochs: int,
        class_weights=None,
        class_counts=None,
    ):
        """Print iteration start message."""
        if self.verbose == 0:
            return

        self.iteration_start_time = time.time()

        if self.verbose == 1:
            print(
                f"\nIteration {iteration}/{total_iterations} [{num_nodes} nodes, {epochs} epochs]"
            )
        elif self.verbose >= 2:
            print(
                f"\n--- Iteration {iteration}/{total_iterations}: Training with {num_nodes} nodes for {epochs} epochs ---\n"
            )

            # Print hierarchical architecture info
            if self.config.hierarchical_architecture == "correct_label_routing":
                if iteration > 1:
                    num_parent_layers = iteration - 1
                    print(
                        f"  Tree Structure: {num_parent_layers} parent layer(s) → {num_nodes} output heads"
                    )
                else:
                    print(f"  Tree Structure: Base iteration (no parent layers yet)")

            if class_counts is not None:
                counts_array = self._to_numpy(class_counts)
                total = np.sum(counts_array)
                print(f"  Sample Distribution: {counts_array} (total: {total})")
            if class_weights is not None:
                weights_array = self._to_numpy(class_weights)
                print(f"  Class Weights: {np.round(weights_array, 2)}")
            elif self.config.use_class_weights:
                # Class weights enabled but not applied - explain why
                if iteration == 1 and self.config.skip_class_weights_iter1:
                    print(f"  Class Weights: Skipped for iteration 1 (skip_class_weights_iter1=True)")

    def phase_start(self, phase_name: str, phase_num: int, epochs: int):
        """Print phase start message."""
        if self.verbose == 0:
            return

        if self.verbose == 1:
            phase_label = "frozen backbone" if phase_num == 1 else "full model"
            print(
                f"  Phase {phase_num} ({phase_label}): {epochs} epochs",
                end="",
                flush=True,
            )
        elif self.verbose >= 2:
            print(f"  {phase_name}")

    def epoch_update(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        train_aux_loss: float = 0.0,
        val_aux_loss: float = 0.0,
        worst_group_acc: float = 0.0,
        difficult_node_acc: float = 0.0,
        difficult_node_f1: float = 0.0,
    ):
        """Print epoch update (only for verbose=2)."""
        if self.verbose >= 2:
            aux_info = ""
            if train_aux_loss > 0 or val_aux_loss > 0:
                aux_info = (
                    f", Train Aux: {train_aux_loss:.4f}, Val Aux: {val_aux_loss:.4f}"
                )
            wga_info = ""
            if worst_group_acc > 0:
                wga_info = f", WGA: {worst_group_acc:.3f}"
            dna_info = ""
            if difficult_node_acc > 0:
                dna_info = f", DNA: {difficult_node_acc:.3f}"
            dnf1_info = ""
            if difficult_node_f1 > 0:
                dnf1_info = f", DNF1: {difficult_node_f1:.3f}"
            print(
                f"    Epoch {epoch}/{total_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}{aux_info}{wga_info}{dna_info}{dnf1_info}"
            )

    def early_stop(self, phase_name: str, phase_num: int):
        """Print early stopping message."""
        if self.verbose == 0:
            return

        if self.verbose == 1:
            print(" → early stop", end="", flush=True)
        elif self.verbose >= 2:
            print(f"    Early stopping triggered in {phase_name}.")

    def phase_end(self, phase_num: int, completed_all: bool = False):
        """Print phase end message (only for verbose=1)."""
        if self.verbose == 1:
            if completed_all:
                print(" → completed all epochs")
            else:
                print()  # New line after early stop or normal completion

    def weight_transfer(self, hierarchical="none"):
        """Print weight transfer message."""
        if self.verbose >= 2:
            if hierarchical == "none":
                print(
                    "  Transferring weights from previous iteration (parallel heads mode)"
                )

    def iteration_end(self):
        """Print iteration end message."""
        if self.verbose >= 1 and self.iteration_start_time:
            duration = time.time() - self.iteration_start_time
            if self.verbose == 1:
                print(f"  ✓ Completed in {duration / 60:.1f} min\n")
            # For verbose=2, just add extra spacing
            elif self.verbose >= 2:
                print()

    def training_end(
        self,
        test_accuracy: Optional[float] = None,
        worst_group_acc: Optional[float] = None,
        finish_wandb: bool = True,
    ):
        """Print training end message and optionally finish logging.

        Args:
            test_accuracy: Optional final test accuracy to display
            worst_group_acc: Optional worst group accuracy to display
            finish_wandb: Whether to finish WandB run now (False if attributions will be logged later)
        """
        if self.verbose >= 1:
            if self.start_time:
                total_duration = time.time() - self.start_time
                print(f"\nTraining Complete ({total_duration / 60:.1f} min total)")
                if test_accuracy is not None:
                    metrics_str = f"Final Test Accuracy: {test_accuracy:.2%}"
                    if worst_group_acc is not None:
                        metrics_str += f" | Worst Group: {worst_group_acc:.2%}"
                    print(metrics_str)
                print()

        # Finish wandb run now if requested (or defer if attributions need to be logged)
        if finish_wandb:
            self.finish_wandb()

    def log(self, message: str, level: int = 1):
        """
        Generic log message.

        Args:
            message: Message to print
            level: Minimum verbosity level to print (default 1)
        """
        if self.verbose >= level:
            print(message)

    @staticmethod
    def _to_numpy(tensor):
        """Convert tensor/array to numpy array."""
        if hasattr(tensor, "cpu"):
            return tensor.cpu().numpy()
        elif hasattr(tensor, "numpy"):
            return tensor.numpy()
        return np.asarray(tensor)

    # =========================================================================
    # Wandb Logging Methods (Delegate to WandbLogger)
    # =========================================================================

    def log_epoch_metrics(
        self,
        iteration: int,
        phase: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        base_lr: float,
        head_lr: float,
        train_acc: float = 0.0,
        worst_group_acc: float = 0.0,
        group_accuracies: Optional[Dict[int, float]] = None,
        train_aux_loss: float = 0.0,
        val_aux_loss: float = 0.0,
        difficult_node_acc: float = 0.0,
        difficult_node_f1: float = 0.0,
    ):
        """Log per-epoch metrics including auxiliary loss, DNA, and DNF1."""
        if self.wandb_logger:
            self.wandb_logger.log_epoch_metrics(
                iteration,
                phase,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1,
                worst_group_acc,
                group_accuracies or {},
                base_lr,
                head_lr,
                train_aux_loss,
                val_aux_loss,
                difficult_node_acc,
                difficult_node_f1,
            )

    def log_iteration_summary(
        self,
        iteration: int,
        test_metrics: Dict[str, Any],
        group_metrics: Dict[str, Any],
        node_metrics: Dict[str, Any],
        training_time: float,
        num_epochs_trained: int = 0,
        early_stopped: bool = False,
    ):
        """Log summary metrics after iteration completes."""
        if self.wandb_logger:
            self.wandb_logger.log_iteration_summary(
                iteration,
                test_metrics,
                group_metrics,
                node_metrics,
                training_time,
                num_epochs_trained,
                early_stopped,
            )

    def log_transition_statistics(
        self, prev_iteration: int, curr_iteration: int, transition_stats: Dict[str, Any]
    ):
        """Log transition statistics between iterations."""
        if self.wandb_logger:
            self.wandb_logger.log_transition_statistics(
                prev_iteration, curr_iteration, transition_stats
            )

    def log_attribution_image(
        self, image_path: str, caption: str, iteration: Optional[int] = None
    ):
        """Log attribution visualization image to wandb."""
        if self.wandb_logger:
            self.wandb_logger.log_attribution_image(image_path, caption, iteration)

    def log_weight_statistics(self, iteration: int, weight_stats: Dict[str, float]):
        """Log weight change statistics to wandb and print to console."""
        # Print to console
        if self.verbose >= 1 and weight_stats:
            if self.verbose == 1:
                # Compact format for verbose=1
                parts = []
                if "backbone_mean_abs_change" in weight_stats:
                    parts.append(
                        f"Backbone: {weight_stats['backbone_relative_change'] * 100:.2f}%"
                    )
                if "heads_mean_abs_change" in weight_stats:
                    parts.append(
                        f"Heads: {weight_stats['heads_relative_change'] * 100:.2f}%"
                    )
                if parts:
                    print(f"  Weight drift: {', '.join(parts)}")

            elif self.verbose >= 2:
                # Detailed format for verbose>=2
                print("  Weight Changes:")
                if "backbone_mean_abs_change" in weight_stats:
                    print(
                        f"    Backbone: mean Δ={weight_stats['backbone_mean_abs_change']:.6f}, "
                        f"relative={weight_stats['backbone_relative_change'] * 100:.2f}%"
                    )
                if "heads_mean_abs_change" in weight_stats:
                    print(
                        f"    Heads: mean Δ={weight_stats['heads_mean_abs_change']:.6f}, "
                        f"relative={weight_stats['heads_relative_change'] * 100:.2f}%"
                    )

        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log_weight_statistics(iteration, weight_stats)

    def log_routing_statistics(self, routing_stats: Dict[str, int]):
        """Log sample routing statistics for correctness_certainty routing."""
        if self.verbose >= 2:
            num_wrong = routing_stats["num_wrong"]
            num_uncertain = routing_stats["num_uncertain"]
            num_difficult = routing_stats["num_difficult"]
            total = routing_stats["total"]
            ratio = num_difficult / total if total > 0 else 0
            print(
                f"  Routing: {num_wrong} wrong + {num_uncertain} uncertain → "
                f"{num_difficult} difficult ({ratio * 100:.1f}%)"
            )

        if self.wandb_logger:
            self.wandb_logger.log_routing_statistics(routing_stats)

    def log_evaluation_results(
        self,
        test_metrics: Dict[str, Any],
        pseudo_metrics: Optional[Dict[str, Any]] = None,
        group_results: Optional[Dict[str, Any]] = None,
    ):
        """Print evaluation results to console based on verbosity level."""
        if self.verbose == 0:
            return

        # Print basic metrics
        self.log(f"Test Accuracy: {test_metrics['accuracy']:.2%}", level=1)
        self.log(f"Test Precision: {test_metrics['precision']:.4f}", level=1)
        self.log(f"Test Recall: {test_metrics['recall']:.4f}", level=1)
        self.log(f"Test F1 Score: {test_metrics['f1_score']:.4f}", level=1)
        if "auroc" in test_metrics and not math.isnan(test_metrics["auroc"]):
            self.log(f"Test AUROC: {test_metrics['auroc']:.4f}", level=1)
        if "non_patch_auroc" in test_metrics and not math.isnan(test_metrics["non_patch_auroc"]):
            self.log(f"Non-Patch AUROC: {test_metrics['non_patch_auroc']:.4f}", level=1)
        if "histopath_auroc" in test_metrics and not math.isnan(test_metrics["histopath_auroc"]):
            self.log(f"Histopath AUROC: {test_metrics['histopath_auroc']:.4f}", level=1)

        # Print pseudo label metrics if available
        if pseudo_metrics:
            self.log(
                f"Pseudo Label Accuracy: {pseudo_metrics['accuracy']:.2%}", level=1
            )
            self.log(f"Pseudo Label F1: {pseudo_metrics['f1_score']:.4f}", level=1)
            self.log(
                f"Worst Pseudo Group Accuracy (All): {pseudo_metrics['worst_group_accuracy']:.2%}",
                level=1,
            )
            self.log(
                f"Worst Pseudo Group Accuracy (Filled): {pseudo_metrics['worst_group_accuracy_filled']:.2%}",
                level=1,
            )
            if "difficult_node_accuracy" in pseudo_metrics:
                self.log(
                    f"Difficult Node Accuracy (DNA): {pseudo_metrics['difficult_node_accuracy']:.2%}",
                    level=1,
                )
            if "difficult_node_f1" in pseudo_metrics:
                self.log(
                    f"Difficult Node F1 (DNF1): {pseudo_metrics['difficult_node_f1']:.4f}",
                    level=1,
                )
            self.log(
                f"Filled Classes: {pseudo_metrics['num_filled_classes']}/{pseudo_metrics['num_expected_classes']}",
                level=1,
            )
            # Log node sizes if available
            if "node_sizes" in pseudo_metrics:
                sizes_str = ", ".join(
                    f"N{k}:{v}" for k, v in sorted(pseudo_metrics["node_sizes"].items())
                )
                self.log(f"Node Sizes: {sizes_str}", level=2)

        # Print group-specific performance
        if group_results is not None:
            self.log("\n--- Group-Specific Performance ---", level=1)
            for group_name, metrics in group_results["group_metrics"].items():
                acc_str = f"Acc: {metrics['accuracy']:.2%}"
                if "auroc" in metrics and not math.isnan(metrics["auroc"]):
                    acc_str += f", AUROC: {metrics['auroc']:.4f}"
                self.log(f"  {group_name}: {acc_str}", level=1)

            self.log(
                f"\nWorst-Group Accuracy: {group_results['summary']['worst_group_accuracy']:.2%}",
                level=1,
            )
            if "worst_group_auroc" in group_results["summary"]:
                self.log(
                    f"Worst-Group AUROC: {group_results['summary']['worst_group_auroc']:.4f}",
                    level=1,
                )

        # Print trailing newline (matches old behavior)
        self.log("", level=1)

    def log_transitions(
        self,
        iteration: int,
        prev_iteration: int,
        transitions: Dict[str, Any],
        group_labels=None,
        dataset_name: str = "",
        group_names: Optional[Dict[int, str]] = None,
    ):
        """Print transition statistics to console and log to wandb."""
        if self.verbose >= 2:
            self.log(
                f"\nIteration {prev_iteration} to {iteration} Transitions: "
                f"Forgetting={transitions['overall']['correct_to_incorrect']}, "
                f"Learning={transitions['overall']['incorrect_to_correct']}",
                level=2,
            )

            if "per_group" in transitions and group_labels is not None:
                self.log("  Per-Group Transitions:", level=2)
                for group_id, stats in transitions["per_group"].items():
                    group_name = (
                        group_names.get(group_id, f"Group {group_id}")
                        if group_names
                        else f"Group {group_id}"
                    )
                    self.log(
                        f"    {group_name}: Forgetting={stats['correct_to_incorrect']}, "
                        f"Learning={stats['incorrect_to_correct']}",
                        level=2,
                    )

        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log_transition_statistics(
                prev_iteration, iteration, transitions
            )
            self.wandb_logger.log_transition_statistics(
                prev_iteration, iteration, transitions
            )

    def finish_wandb(self):
        """Finish wandb run."""
        if self.wandb_logger:
            total_time = time.time() - self.start_time if self.start_time else None
            self.wandb_logger.finish(total_time)
