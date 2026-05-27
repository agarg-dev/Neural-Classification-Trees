"""Logging utilities for NCT training."""

import math
import time
from typing import Any, Dict, Optional

import numpy as np


class NCTLogger:
    """Logger for NCT training with different verbosity levels and optional wandb logging."""

    def __init__(self, verbose: int, config):
        self.verbose = verbose
        self.config = config
        self.start_time = None
        self.iteration_start_time = None

        self.wandb_logger = None
        self._wandb_enabled = config.wandb_enabled
        self._wandb_config = {}

        if self._wandb_enabled:
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
            phase_label = "frozen base" if phase_num == 1 else "full model"
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
        iter2_level_wga: Optional[float] = None,
        iter2_level_auroc: Optional[float] = None,
        iter2_level_dna: Optional[float] = None,
        iter2_level_dnf1: Optional[float] = None,
        iteration_num: int = 1,
    ):
        """Print epoch update (only for verbose=2)."""
        if self.verbose >= 2:
            aux_info = ""
            if train_aux_loss > 0 or val_aux_loss > 0:
                aux_info = (
                    f", Train Aux: {train_aux_loss:.4f}, Val Aux: {val_aux_loss:.4f}"
                )
            # At iter3+: suppress current-node metrics, show iter2-level instead.
            # At iter1/iter2: show current-node metrics normally.
            if iteration_num >= 3:
                wga_info = ""
                dna_info = ""
                dnf1_info = ""
                iter2_wga_info = ""
                if iter2_level_wga is not None:
                    iter2_wga_info = f", Val Iter2-Pseudo-WGA: {iter2_level_wga:.3f}"
                iter2_dna_info = ""
                if iter2_level_dna is not None:
                    iter2_dna_info = f", Val Iter2-Pseudo-DNA: {iter2_level_dna:.3f}"
                iter2_dnf1_info = ""
                if iter2_level_dnf1 is not None:
                    iter2_dnf1_info = f", Val Iter2-Pseudo-DNF1: {iter2_level_dnf1:.3f}"
            else:
                wga_info = ""
                if worst_group_acc > 0:
                    wga_info = f", Val Pseudo-WGA: {worst_group_acc:.3f}"
                dna_info = ""
                if difficult_node_acc > 0:
                    dna_info = f", Val Pseudo-DNA: {difficult_node_acc:.3f}"
                dnf1_info = ""
                if difficult_node_f1 > 0:
                    dnf1_info = f", Val Pseudo-DNF1: {difficult_node_f1:.3f}"
                iter2_wga_info = ""
                iter2_dna_info = ""
                iter2_dnf1_info = ""
            iter2_auroc_info = ""
            if iteration_num >= 2 and iter2_level_auroc is not None:
                import math as _math
                if not _math.isnan(iter2_level_auroc):
                    iter2_auroc_info = f", Val Iter2-AUROC: {iter2_level_auroc:.3f}"
            print(
                f"    Epoch {epoch}/{total_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                f"{aux_info}{wga_info}{iter2_wga_info}{iter2_dna_info}{iter2_dnf1_info}"
                f"{iter2_auroc_info}{dna_info}{dnf1_info}"
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
        iter2_level_wga: Optional[float] = None,
        iter2_level_auroc: Optional[float] = None,
        iter2_level_dna: Optional[float] = None,
        iter2_level_dnf1: Optional[float] = None,
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
                iter2_level_wga,
                iter2_level_auroc,
                iter2_level_dna,
                iter2_level_dnf1,
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
        consolidated_group_metrics: Optional[Dict[str, Any]] = None,
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
                consolidated_group_metrics,
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

    def log_evaluation_results(
        self,
        test_metrics: Dict[str, Any],
        pseudo_metrics: Optional[Dict[str, Any]] = None,
        group_results: Optional[Dict[str, Any]] = None,
        consolidated_group_results: Optional[Dict[str, Any]] = None,
        class_names: Optional[tuple] = None,
    ):
        """Print evaluation results to console based on verbosity level.

        Order: Basic metrics → Worst-Group Accuracy → Group breakdown → Pseudo-label metrics
        """
        if self.verbose == 0:
            return

        # 1. Print basic metrics
        self.log(f"Test Accuracy: {test_metrics['accuracy']:.2%}", level=1)
        if "auroc" in test_metrics and not math.isnan(test_metrics["auroc"]):
            self.log(f"Test AUROC: {test_metrics['auroc']:.4f}", level=1)

        # 2. Print Worst-Group Accuracy (key metric for spurious correlation)
        if group_results is not None:
            num_groups = len(group_results.get("group_metrics", {}))
            self.log(
                f"Worst-Group Accuracy ({num_groups} groups): {group_results['summary']['worst_group_accuracy']:.2%}",
                level=1,
            )
            if "worst_group_auroc" in group_results["summary"]:
                self.log(
                    f"Worst-Group AUROC: {group_results['summary']['worst_group_auroc']:.4f}",
                    level=1,
                )

        if consolidated_group_results is not None:
            num_consolidated = len(consolidated_group_results.get("per_class", {})) * 2
            self.log(
                f"Worst-Group Accuracy ({num_consolidated} consolidated): {consolidated_group_results['worst_group_accuracy']:.2%}",
                level=1,
            )

        # 3. Print additional basic metrics (less prominent)
        self.log(f"Test Precision: {test_metrics['precision']:.4f}", level=2)
        self.log(f"Test Recall: {test_metrics['recall']:.4f}", level=2)
        self.log(f"Test F1 Score: {test_metrics['f1_score']:.4f}", level=2)
        if "non_patch_auroc" in test_metrics and not math.isnan(test_metrics["non_patch_auroc"]):
            self.log(f"Non-Patch AUROC: {test_metrics['non_patch_auroc']:.4f}", level=2)
        if "histopath_auroc" in test_metrics and not math.isnan(test_metrics["histopath_auroc"]):
            self.log(f"Histopath AUROC: {test_metrics['histopath_auroc']:.4f}", level=2)

        # 4. Print consolidated group breakdown if available
        if consolidated_group_results is not None:
            self.log("\n--- Consolidated Group Performance (Easy/Hard) ---", level=1)
            per_class = consolidated_group_results.get("per_class", {})
            for class_id in sorted(per_class.keys()):
                metrics = per_class[class_id]
                easy_acc = metrics["aligned_accuracy"]
                hard_acc = metrics["conflicting_accuracy"]
                easy_support = metrics["aligned_support"]
                hard_support = metrics["conflicting_support"]

                class_label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                easy_str = f"Easy: {easy_acc:.2%}" if not np.isnan(easy_acc) else "Easy: N/A"
                hard_str = f"Hard: {hard_acc:.2%}" if not np.isnan(hard_acc) else "Hard: N/A"
                self.log(f"  {class_label}:  {easy_str} (n={easy_support}),  {hard_str} (n={hard_support})", level=1)

        # 5. Print group-specific performance breakdown, grouped by class
        if group_results is not None:
            num_groups = len(group_results.get("group_metrics", {}))
            self.log(f"\n--- Group-Specific Performance ({num_groups} groups) ---", level=2)

            num_classes = self.config.num_classes
            num_attrs_per_class = max(1, num_groups // num_classes)

            # Sort groups by y_class (true class) then group_id, so headers match
            # asymmetric datasets (e.g. ISIC: Benign has 2 subgroups, Malignant 1).
            sorted_metrics = sorted(
                group_results["group_metrics"].items(),
                key=lambda kv: (
                    kv[1].get("y_class", kv[1].get("group_id", 0) // num_attrs_per_class),
                    kv[1].get("group_id", 0),
                ),
            )

            current_class_id = -1
            for group_name, metrics in sorted_metrics:
                group_id = metrics.get("group_id", 0)
                class_id = metrics.get("y_class", group_id // num_attrs_per_class)

                # Print class header when class changes
                if class_id != current_class_id:
                    current_class_id = class_id
                    class_label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                    self.log(f"\n  {class_label}:", level=2)

                support = metrics.get("support", "?")
                acc_str = f"{metrics['accuracy']:.2%}  (n={support})"
                if "auroc" in metrics and not math.isnan(metrics["auroc"]):
                    acc_str += f",  AUROC: {metrics['auroc']:.4f}"
                self.log(f"    {group_name}:  {acc_str}", level=2)

        # 6. Print pseudo label metrics (routing quality - less important)
        if pseudo_metrics:
            self.log("\n--- Pseudo-Label Routing Metrics ---", level=2)
            self.log(
                f"Pseudo Label Accuracy: {pseudo_metrics['accuracy']:.2%}", level=2
            )
            self.log(f"Pseudo Label F1: {pseudo_metrics['f1_score']:.4f}", level=2)
            self.log(
                f"Worst Pseudo Group Accuracy (All): {pseudo_metrics['worst_group_accuracy']:.2%}",
                level=2,
            )
            self.log(
                f"Worst Pseudo Group Accuracy (Filled): {pseudo_metrics['worst_group_accuracy_filled']:.2%}",
                level=2,
            )
            if "iter2_level_worst_group_accuracy" in pseudo_metrics:
                self.log(
                    f"Iteration-2-Level Pseudo WGA (All): {pseudo_metrics['iter2_level_worst_group_accuracy']:.2%}",
                    level=2,
                )
            if "iter2_level_worst_group_accuracy_filled" in pseudo_metrics:
                self.log(
                    f"Iteration-2-Level Pseudo WGA (Filled): {pseudo_metrics['iter2_level_worst_group_accuracy_filled']:.2%}",
                    level=2,
                )
            if "iter2_level_worst_group_auroc" in pseudo_metrics:
                auroc_val = pseudo_metrics["iter2_level_worst_group_auroc"]
                self.log(
                    f"Iteration-2-Level Pseudo AUROC (Worst Group): {auroc_val:.4f}",
                    level=2,
                )
            if "difficult_node_accuracy" in pseudo_metrics:
                self.log(
                    f"Difficult Node Accuracy (DNA): {pseudo_metrics['difficult_node_accuracy']:.2%}",
                    level=2,
                )
            if "difficult_node_f1" in pseudo_metrics:
                self.log(
                    f"Difficult Node F1 (DNF1): {pseudo_metrics['difficult_node_f1']:.4f}",
                    level=2,
                )
            if "iter2_level_difficult_node_accuracy" in pseudo_metrics:
                self.log(
                    f"Iteration-2-Level DNA: {pseudo_metrics['iter2_level_difficult_node_accuracy']:.2%}",
                    level=2,
                )
            if "iter2_level_difficult_node_f1" in pseudo_metrics:
                self.log(
                    f"Iteration-2-Level DNF1: {pseudo_metrics['iter2_level_difficult_node_f1']:.4f}",
                    level=2,
                )
            self.log(
                f"Filled Classes: {pseudo_metrics['num_filled_classes']}/{pseudo_metrics['num_expected_classes']}",
                level=2,
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

    def log_depth_decision(
        self,
        iteration: int,
        value: Optional[float],
        n_worst: Optional[int],
        best_value: Optional[float],
        best_iter: Optional[int],
        epsilon: Optional[float],
        decision: str,
        metric: str = "wga",
    ) -> None:
        if self.verbose < 1:
            return

        key = f"pseudo_{metric}"
        if decision == "bootstrap":
            print(f"\n[DepthSel] iter {iteration}: bootstrap (not eligible for selection)")
            return
        if decision == "skip":
            v = f"{value:.4f}" if value is not None else "n/a"
            nw = n_worst if n_worst is not None else "n/a"
            print(f"\n[DepthSel] iter {iteration}: {key}={v} n_worst={nw} (skipped — no valid node)")
            return
        if decision == "initial_best":
            print(
                f"\n[DepthSel] iter {iteration}: {key}={value:.4f} n_worst={n_worst} "
                f"-> initial best"
            )
            return

        v = f"{value:.4f}" if value is not None else "n/a"
        bv = f"{best_value:.4f}" if best_value is not None else "n/a"
        eps = f"{epsilon:.4f}" if epsilon is not None else "n/a"
        tag = {
            "new_best": "-> NEW BEST",
            "within_epsilon": f"-> within eps of best (iter {best_iter}, {key}={bv})",
            "exceeded_epsilon": f"-> DROPPED beyond eps from best (iter {best_iter}, {key}={bv})",
        }.get(decision, f"-> {decision}")
        print(f"\n[DepthSel] iter {iteration}: {key}={v} n_worst={n_worst} eps={eps}  {tag}")

    def log_depth_final_summary(
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
        if self.verbose >= 1:
            mode = "ENABLED" if enabled else "DISABLED"
            key = f"pseudo_{metric}"
            print(f"\n========== Depth Selection Summary ({mode}, metric={key}) ==========")

            for it in sorted(per_iter_stats.keys()):
                st = per_iter_stats[it]
                value = st.get("value")
                n_worst = st.get("n_worst")
                decision = st.get("decision", "")
                if value is None:
                    row = f"  iter {it}: {key}=n/a        ({decision})"
                else:
                    row = f"  iter {it}: {key}={value:.4f}  n_worst={n_worst}  ({decision})"

                markers = []
                if enabled and it == selected_iter:
                    markers.append("SELECTED")
                if it == headline_iter:
                    markers.append("HEADLINE")
                if markers:
                    row += "  <- " + " / ".join(markers)
                print(row)

            if not enabled and selected_iter is not None and selected_iter != headline_iter:
                print(f"  [passive] selected_iter_if_enabled = {selected_iter}")

            print(f"  stop reason:   {stop_reason}")
            print(
                f"  headline iter: {headline_iter} "
                f"(last trained: {last_trained_iter}, max: {num_iterations})"
            )
            print("=" * 56)

        if self.wandb_logger:
            self.wandb_logger.log_depth_selection_summary(
                enabled=enabled,
                selected_iter=selected_iter,
                headline_iter=headline_iter,
                stop_reason=stop_reason,
                per_iter_stats=per_iter_stats,
                last_trained_iter=last_trained_iter,
                num_iterations=num_iterations,
                early_stopped=early_stopped,
                metric=metric,
            )

    def log_wga_spearman(
        self, spearman_r: float, n_epochs: int, variant: str = "full",
        pseudo_metric: str = "WGA", true_metric: str = "WGA",
    ):
        """Log Spearman of val pseudo metric vs true metric ('full' or 'consolidated')."""
        rhs = f"True {true_metric}" + ("" if variant == "full" else " (consolidated)")
        if self.verbose >= 1:
            print(
                f"  Val Pseudo-{pseudo_metric} ↔ {rhs} Spearman r = {spearman_r:.4f} "
                f"(over {n_epochs} epochs)"
            )
        if self.wandb_logger:
            self.wandb_logger.log_wga_spearman(spearman_r, n_epochs, variant=variant)

    def finish_wandb(self):
        """Finish wandb run."""
        if self.wandb_logger:
            total_time = time.time() - self.start_time if self.start_time else None
            self.wandb_logger.finish(total_time)
