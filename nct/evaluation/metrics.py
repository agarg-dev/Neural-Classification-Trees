"""Evaluation metrics for hierarchical classification."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.hierarchical import get_num_nodes


def calculate_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Calculate basic classification metrics.

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_prob: Predicted probabilities for positive class (numpy array, optional).
                Required for AUROC computation.
        average: Averaging strategy for multi-class metrics

    Returns:
        Dict[str, float]: Dictionary of metrics including AUROC if y_prob provided
    """

    # Auto-detect if multiclass and adjust averaging
    # Check both y_true and y_pred to handle edge cases where model
    # predicts only a subset of classes or predicts extra classes
    num_classes_true = len(np.unique(y_true))
    num_classes_pred = len(np.unique(y_pred))
    num_classes = max(num_classes_true, num_classes_pred)

    if num_classes > 2 and average == "binary":
        average = "weighted"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Calculate AUROC if probabilities are provided
    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary classification: use probability of positive class
                metrics["auroc"] = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class: use one-vs-rest with weighted averaging
                metrics["auroc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
        except ValueError:
            # AUROC undefined if only one class present in y_true
            metrics["auroc"] = float("nan")

    return metrics


def calculate_subgroup_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroups: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calculate AUROC for different subgroups.

    For ISIC dataset, computes:
    - non_patch_auroc: AUROC on samples without colored patches
    - histopath_auroc: AUROC on samples with histopathology confirmation

    Args:
        y_true: True binary labels (numpy array)
        y_prob: Predicted probabilities for positive class (numpy array)
        subgroups: Dictionary of boolean masks (e.g., {"non_patch": ..., "histopath": ...})

    Returns:
        Dict[str, float]: Subgroup AUROC metrics
    """
    if subgroups is None:
        return {}

    results = {}

    # Non-patch AUROC: samples without the spurious patch attribute
    if "non_patch" in subgroups:
        mask = subgroups["non_patch"]
        if np.sum(mask) > 0 and len(np.unique(y_true[mask])) > 1:
            try:
                results["non_patch_auroc"] = roc_auc_score(y_true[mask], y_prob[mask])
            except ValueError:
                results["non_patch_auroc"] = float("nan")

    # Histopathology AUROC: samples with histopathology-confirmed diagnosis
    if "histopath" in subgroups:
        mask = subgroups["histopath"]
        if np.sum(mask) > 0 and len(np.unique(y_true[mask])) > 1:
            try:
                results["histopath_auroc"] = roc_auc_score(y_true[mask], y_prob[mask])
            except ValueError:
                results["histopath_auroc"] = float("nan")

    return results


def calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    group_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Calculate metrics for different groups/subpopulations.

    Args:
        y_true: True binary labels (numpy array)
        y_pred: Predicted binary labels (numpy array)
        group_labels: Group/subpopulation labels (numpy array)
        y_prob: Predicted probabilities for positive class (numpy array, optional).
                Required for per-group AUROC computation.
        group_names: Optional mapping from group IDs to names

    Returns:
        Dict[str, Any]: Group-specific metrics including per-group AUROC if y_prob provided
    """

    group_metrics = {}
    group_accuracies = []
    group_aurocs = []

    # Detect if multi-class based on unique labels
    num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
    average = "binary" if num_classes <= 2 else "weighted"

    unique_groups = np.unique(group_labels)

    for group_id in unique_groups:
        group_mask = group_labels == group_id

        if np.sum(group_mask) == 0:
            continue

        group_true = y_true[group_mask]
        group_pred = y_pred[group_mask]

        group_name = (
            group_names.get(group_id, f"Group_{group_id}")
            if group_names
            else f"Group_{group_id}"
        )

        # Calculate metrics for this group
        accuracy = accuracy_score(group_true, group_pred)
        group_accuracies.append(accuracy)

        group_metrics[group_name] = {
            "accuracy": accuracy,
            "precision": precision_score(group_true, group_pred, average=average, zero_division=0),
            "recall": recall_score(group_true, group_pred, average=average, zero_division=0),
            "f1_score": f1_score(group_true, group_pred, average=average, zero_division=0),
            "support": len(group_true),
            "group_id": int(group_id),
        }

        # Calculate per-group AUROC if probabilities provided
        if y_prob is not None:
            group_prob = y_prob[group_mask]
            # AUROC requires at least 2 classes present in the group
            unique_in_group = np.unique(group_true)
            if len(unique_in_group) > 1:
                try:
                    if num_classes <= 2:
                        # Binary classification
                        auroc = roc_auc_score(group_true, group_prob)
                    else:
                        # Multi-class: use one-vs-rest with weighted averaging
                        auroc = roc_auc_score(
                            group_true, group_prob, multi_class="ovr", average="weighted"
                        )
                    group_metrics[group_name]["auroc"] = auroc
                    group_aurocs.append(auroc)
                except ValueError:
                    group_metrics[group_name]["auroc"] = float("nan")
            else:
                # Only one class in group - AUROC undefined
                group_metrics[group_name]["auroc"] = float("nan")

    # Calculate worst-group accuracy
    worst_group_accuracy = min(group_accuracies) if group_accuracies else 0.0

    # Summary statistics
    summary = {
        "worst_group_accuracy": worst_group_accuracy,
        "best_group_accuracy": max(group_accuracies) if group_accuracies else 0.0,
        "mean_group_accuracy": np.mean(group_accuracies) if group_accuracies else 0.0,
        "group_accuracy_std": np.std(group_accuracies) if group_accuracies else 0.0,
    }

    # Add AUROC summary if available
    if group_aurocs:
        summary["worst_group_auroc"] = min(group_aurocs)
        summary["best_group_auroc"] = max(group_aurocs)
        summary["mean_group_auroc"] = float(np.mean(group_aurocs))

    return {"group_metrics": group_metrics, "summary": summary}


def calculate_hierarchical_metrics(
    hierarchical_predictions: np.ndarray,
    num_iterations: int,
    y_true: Optional[np.ndarray] = None,
    class_predictions: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """
    Calculate metrics specific to hierarchical classification.

    Args:
        hierarchical_predictions: Final node predictions from hierarchy (numpy array)
        num_iterations: Number of iterations in the hierarchy
        y_true: Optional true labels for computing per-node accuracy (numpy array)
        class_predictions: Optional class predictions for computing per-node accuracy (numpy array)
        num_classes: Number of base classes (default 2 for binary)

    Returns:
        Dict[str, Any]: Hierarchical-specific metrics (node distribution, utilization, and per-node accuracy)
    """

    # Hierarchical node distribution
    final_num_nodes = get_num_nodes(num_iterations, num_classes)
    node_distribution = {}

    for node_idx in range(final_num_nodes):
        node_mask = hierarchical_predictions == node_idx
        node_count = np.sum(node_mask)

        node_info = {
            "count": int(node_count),
            "percentage": float(node_count / len(hierarchical_predictions) * 100),
        }

        # Add per-node accuracy if labels and predictions are provided
        if y_true is not None and class_predictions is not None and node_count > 0:
            node_correct = np.sum((class_predictions[node_mask] == y_true[node_mask]))
            node_info["accuracy"] = float(node_correct / node_count)

        node_distribution[f"node_{node_idx}"] = node_info

    # Node utilization
    used_nodes = np.sum([dist["count"] > 0 for dist in node_distribution.values()])
    node_utilization = used_nodes / final_num_nodes

    return {
        "node_distribution": node_distribution,
        "node_utilization": node_utilization,
        "total_nodes": final_num_nodes,
        "used_nodes": int(used_nodes),
    }


def _compute_transition_stats(
    prev_correct: np.ndarray, curr_correct: np.ndarray
) -> Dict[str, Any]:
    """
    Helper to compute transition statistics and rates.

    Args:
        prev_correct: Boolean array of correct predictions from previous iteration
        curr_correct: Boolean array of correct predictions from current iteration

    Returns:
        Dictionary with transition counts and rates
    """
    correct_to_incorrect = int(np.sum(prev_correct & ~curr_correct))
    incorrect_to_correct = int(np.sum(~prev_correct & curr_correct))
    correct_to_correct = int(np.sum(prev_correct & curr_correct))
    incorrect_to_incorrect = int(np.sum(~prev_correct & ~curr_correct))

    total_prev_correct = correct_to_incorrect + correct_to_correct
    total_prev_incorrect = incorrect_to_correct + incorrect_to_incorrect

    forgetting_rate = (
        (correct_to_incorrect / total_prev_correct) if total_prev_correct > 0 else 0.0
    )
    stability_rate = (
        (correct_to_correct / total_prev_correct) if total_prev_correct > 0 else 0.0
    )
    learning_rate = (
        (incorrect_to_correct / total_prev_incorrect)
        if total_prev_incorrect > 0
        else 0.0
    )

    return {
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct,
        "correct_to_correct": correct_to_correct,
        "incorrect_to_incorrect": incorrect_to_incorrect,
        "prev_accuracy": float(np.mean(prev_correct)),
        "curr_accuracy": float(np.mean(curr_correct)),
        "forgetting_rate": float(forgetting_rate),
        "stability_rate": float(stability_rate),
        "learning_rate": float(learning_rate),
    }


def compute_iteration_transitions(
    prev_predictions: np.ndarray,
    curr_predictions: np.ndarray,
    labels: np.ndarray,
    group_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute prediction transitions between consecutive iterations.

    Args:
        prev_predictions: Predictions from previous iteration (numpy array)
        curr_predictions: Predictions from current iteration (numpy array)
        labels: Ground truth labels (numpy array)
        group_labels: Optional group labels for per-group analysis (numpy array)

    Returns:
        Dictionary containing transition statistics with counts and rates
    """

    prev_correct = prev_predictions == labels
    curr_correct = curr_predictions == labels

    # Compute overall transitions
    result = {"overall": _compute_transition_stats(prev_correct, curr_correct)}

    # Compute per-group transitions
    if group_labels is not None:
        per_group = {}
        unique_groups = np.unique(group_labels)

        for group_id in unique_groups:
            mask = group_labels == group_id
            if np.sum(mask) == 0:
                continue

            group_stats = _compute_transition_stats(
                prev_correct[mask], curr_correct[mask]
            )
            group_stats["size"] = int(np.sum(mask))
            per_group[int(group_id)] = group_stats

        result["per_group"] = per_group

    return result


def compute_difficult_node_accuracy(
    true_pseudo_labels: np.ndarray,
    predicted_pseudo_labels: np.ndarray,
) -> float:
    """
    Compute accuracy on difficult nodes only (odd-indexed nodes).

    Difficult nodes are those where samples were routed due to incorrect predictions
    in the previous iteration. This metric is robust to small noisy nodes as they
    contribute proportionally to their size rather than dominating like in WGA.

    Args:
        true_pseudo_labels: Ground truth pseudo labels from previous iteration
        predicted_pseudo_labels: Predicted routing from current iteration

    Returns:
        Accuracy on difficult nodes (float between 0 and 1).
        Returns 1.0 if no difficult samples exist.
    """
    # Difficult nodes have odd indices (1, 3, 5, 7, ...)
    difficult_mask = (true_pseudo_labels % 2 == 1)

    if np.sum(difficult_mask) == 0:
        return 1.0  # No difficult samples = perfect routing

    return float(accuracy_score(
        true_pseudo_labels[difficult_mask],
        predicted_pseudo_labels[difficult_mask]
    ))


def compute_difficult_node_f1(
    true_pseudo_labels: np.ndarray,
    predicted_pseudo_labels: np.ndarray,
) -> float:
    """
    Compute F1 score on difficult nodes only (odd-indexed nodes).

    Difficult nodes are those where samples were routed due to incorrect predictions
    in the previous iteration. F1 score provides a balance between precision and recall
    for difficult node classification.

    Args:
        true_pseudo_labels: Ground truth pseudo labels from previous iteration
        predicted_pseudo_labels: Predicted routing from current iteration

    Returns:
        F1 score on difficult nodes (float between 0 and 1).
        Returns 1.0 if no difficult samples exist.
    """
    # Difficult nodes have odd indices (1, 3, 5, 7, ...)
    difficult_mask = (true_pseudo_labels % 2 == 1)

    if np.sum(difficult_mask) == 0:
        return 1.0  # No difficult samples = perfect routing

    # Only include difficult classes (odd indices) in macro average
    # to avoid F1=0 for easy classes dragging down the metric
    difficult_labels = np.unique(true_pseudo_labels[difficult_mask])
    return float(f1_score(
        true_pseudo_labels[difficult_mask],
        predicted_pseudo_labels[difficult_mask],
        labels=difficult_labels,
        average="macro",
        zero_division=1.0
    ))


def compute_pseudo_label_metrics(
    true_pseudo_labels: np.ndarray,
    predicted_pseudo_labels: np.ndarray,
    num_expected_classes: int,
    min_samples_threshold: int = 0,
) -> Dict[str, Any]:
    """
    Compute pseudo label metrics (multi-class routing accuracy).

    Works for any number of iterations (2^n classes for n iterations).

    Args:
        true_pseudo_labels: Ground truth pseudo labels from previous iteration
        predicted_pseudo_labels: Predicted routing from current iteration
        num_expected_classes: Expected number of classes (should be 2^iteration)
        min_samples_threshold: Minimum samples for a class to be considered "filled"
                               (default 0 means any class with samples counts)

    Returns:
        Dict with accuracy, F1, worst pseudo-group accuracy (all classes),
        worst pseudo-group accuracy (filled classes only), difficult node accuracy,
        per-node sizes, and per-node accuracies.
    """
    from sklearn.metrics import accuracy_score, f1_score

    # Multi-class accuracy
    accuracy = accuracy_score(true_pseudo_labels, predicted_pseudo_labels)

    # Multi-class F1 (macro average)
    f1 = f1_score(true_pseudo_labels, predicted_pseudo_labels, average="macro")

    # Per-class accuracy and sizes for ALL expected classes
    group_accs_all = {}
    group_accs_filled = {}
    node_sizes = {}
    per_node_accuracy = {}

    for gid in range(num_expected_classes):
        mask = true_pseudo_labels == gid
        node_size = int(np.sum(mask))
        node_sizes[int(gid)] = node_size

        if node_size > 0:
            # Class exists in data
            acc = float(
                accuracy_score(true_pseudo_labels[mask], predicted_pseudo_labels[mask])
            )
            group_accs_all[int(gid)] = acc
            per_node_accuracy[int(gid)] = acc
            # Only count as "filled" if meets minimum threshold
            if node_size >= max(1, min_samples_threshold):
                group_accs_filled[int(gid)] = acc
        else:
            # Class missing from data = 0% accuracy
            group_accs_all[int(gid)] = 0.0
            per_node_accuracy[int(gid)] = 0.0

    # Compute difficult node accuracy (DNA) and F1
    difficult_node_accuracy = compute_difficult_node_accuracy(
        true_pseudo_labels, predicted_pseudo_labels
    )
    difficult_node_f1 = compute_difficult_node_f1(
        true_pseudo_labels, predicted_pseudo_labels
    )

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "worst_group_accuracy": float(min(group_accs_all.values()))
        if group_accs_all
        else 0.0,
        "worst_group_accuracy_filled": float(min(group_accs_filled.values()))
        if group_accs_filled
        else 0.0,
        "difficult_node_accuracy": difficult_node_accuracy,
        "difficult_node_f1": difficult_node_f1,
        "group_accuracies": group_accs_all,
        "node_sizes": node_sizes,
        "per_node_accuracy": per_node_accuracy,
        "num_filled_classes": len(group_accs_filled),
        "num_expected_classes": num_expected_classes,
    }


def print_evaluation_report(
    metrics: Dict[str, Any], title: str = "Evaluation Results"
) -> None:
    """Print a clean, professional evaluation report like scikit-learn."""

    # Basic metrics - clean, single line format
    if "binary_metrics" in metrics:
        basic = metrics["binary_metrics"]
        print(f"              precision    recall  f1-score   support")
        print(f"")
        print(
            f"           0     {basic.get('precision', 0):.2f}      {basic.get('recall', 0):.2f}      {basic.get('f1_score', 0):.2f}      -"
        )
        print(
            f"           1     {basic.get('precision', 0):.2f}      {basic.get('recall', 0):.2f}      {basic.get('f1_score', 0):.2f}      -"
        )
        print(f"")
        print(
            f"    accuracy                          {basic.get('accuracy', 0):.2f}      -"
        )

    # Group-specific metrics - clean table format
    if "group_metrics" in metrics:
        print(f"\nGroup Performance:")
        summary = metrics["group_metrics"]["summary"]
        print(f"  Worst-group accuracy: {summary['worst_group_accuracy']:.3f}")
        print(f"  Mean group accuracy:  {summary['mean_group_accuracy']:.3f}")

    # Hierarchical metrics - minimal
    if "node_utilization" in metrics:
        print(
            f"  Node utilization:     {metrics['node_utilization']:.3f} ({metrics['used_nodes']}/{metrics['total_nodes']} nodes)"
        )
