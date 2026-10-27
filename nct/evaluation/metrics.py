"""Evaluation metrics for hierarchical classification."""

from itertools import product
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

from ..data.configs import DATASET_REGISTRY, MultiSpuriousCorrelationConfig
from ..utils.hierarchical import get_num_nodes


def calculate_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, F1, and (if y_prob given) AUROC."""

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

    if y_prob is not None:
        try:
            if num_classes == 2:
                metrics["auroc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["auroc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
        except ValueError:
            # AUROC undefined when only one class is present
            metrics["auroc"] = float("nan")

    return metrics


def calculate_subgroup_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroups: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Calculate AUROC for each subgroup mask in `subgroups`."""
    if subgroups is None:
        return {}

    results = {}

    if "non_patch" in subgroups:
        mask = subgroups["non_patch"]
        if np.sum(mask) > 0 and len(np.unique(y_true[mask])) > 1:
            try:
                results["non_patch_auroc"] = roc_auc_score(y_true[mask], y_prob[mask])
            except ValueError:
                results["non_patch_auroc"] = float("nan")

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
    """Calculate per-group accuracy, precision, recall, F1 (and AUROC if y_prob given)."""

    group_metrics = {}
    group_accuracies = []
    group_aurocs = []

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

        accuracy = accuracy_score(group_true, group_pred)
        group_accuracies.append(accuracy)

        # Dominant true class in group (robust to asymmetric class/subgroup splits)
        vals, counts = np.unique(group_true, return_counts=True)
        y_class = int(vals[np.argmax(counts)]) if len(vals) > 0 else -1

        group_metrics[group_name] = {
            "accuracy": accuracy,
            "precision": precision_score(group_true, group_pred, average=average, zero_division=0),
            "recall": recall_score(group_true, group_pred, average=average, zero_division=0),
            "f1_score": f1_score(group_true, group_pred, average=average, zero_division=0),
            "support": len(group_true),
            "group_id": int(group_id),
            "y_class": y_class,
        }

        if y_prob is not None:
            group_prob = y_prob[group_mask]
            # AUROC requires at least 2 classes present in the group
            unique_in_group = np.unique(group_true)
            if len(unique_in_group) > 1:
                try:
                    if num_classes <= 2:
                        auroc = roc_auc_score(group_true, group_prob)
                    else:
                        auroc = roc_auc_score(
                            group_true, group_prob, multi_class="ovr", average="weighted"
                        )
                    group_metrics[group_name]["auroc"] = auroc
                    group_aurocs.append(auroc)
                except ValueError:
                    group_metrics[group_name]["auroc"] = float("nan")
            else:
                group_metrics[group_name]["auroc"] = float("nan")

    worst_group_accuracy = min(group_accuracies) if group_accuracies else 0.0

    summary = {
        "worst_group_accuracy": worst_group_accuracy,
        "best_group_accuracy": max(group_accuracies) if group_accuracies else 0.0,
        "mean_group_accuracy": np.mean(group_accuracies) if group_accuracies else 0.0,
        "group_accuracy_std": np.std(group_accuracies) if group_accuracies else 0.0,
    }

    if group_aurocs:
        summary["worst_group_auroc"] = min(group_aurocs)
        summary["best_group_auroc"] = max(group_aurocs)
        summary["mean_group_auroc"] = float(np.mean(group_aurocs))

    return {"group_metrics": group_metrics, "summary": summary}


def compute_consolidated_group_metrics(
    group_labels: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    num_groups: int = 25,
) -> Optional[Dict[str, Any]]:
    """Consolidate multi-attribute groups to per-class aligned/conflicting cells.

    Returns dict with WGA and per-class aligned/conflicting accuracies, or None
    if num_groups // num_classes <= 2.
    """
    num_spurious = num_groups // num_classes

    if num_spurious <= 2:
        return None

    consolidated_groups = []
    y_true_from_groups = []
    for group_id in group_labels:
        y_from_group = group_id // num_spurious
        attr_id = group_id % num_spurious
        is_conflicting = 1 if attr_id != y_from_group else 0
        consolidated_group = y_from_group * 2 + is_conflicting
        consolidated_groups.append(consolidated_group)
        y_true_from_groups.append(y_from_group)

    consolidated_groups = np.array(consolidated_groups)
    y_true_from_groups = np.array(y_true_from_groups)

    per_class_metrics = {}
    consolidated_accuracies = []

    for class_id in range(num_classes):
        aligned_mask = consolidated_groups == (class_id * 2)
        conflicting_mask = consolidated_groups == (class_id * 2 + 1)

        if np.sum(aligned_mask) > 0:
            aligned_acc = float(np.mean(y_pred[aligned_mask] == y_true_from_groups[aligned_mask]))
        else:
            aligned_acc = np.nan

        if np.sum(conflicting_mask) > 0:
            conflicting_acc = float(np.mean(y_pred[conflicting_mask] == y_true_from_groups[conflicting_mask]))
        else:
            conflicting_acc = np.nan

        per_class_metrics[class_id] = {
            "aligned_accuracy": aligned_acc,
            "conflicting_accuracy": conflicting_acc,
            "aligned_support": int(np.sum(aligned_mask)),
            "conflicting_support": int(np.sum(conflicting_mask)),
        }

        if not np.isnan(aligned_acc):
            consolidated_accuracies.append(aligned_acc)
        if not np.isnan(conflicting_acc):
            consolidated_accuracies.append(conflicting_acc)

    worst_group_accuracy = float(np.min(consolidated_accuracies)) if consolidated_accuracies else 0.0

    return {
        "worst_group_accuracy": worst_group_accuracy,
        "per_class": per_class_metrics,
        "consolidated_groups": consolidated_groups.tolist(),
    }


def calculate_multi_spurious_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    config: MultiSpuriousCorrelationConfig,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Evaluate multiple binary correlators from a joint group encoding.

    The expected encoding is class-major with correlators in the order given
    by ``config.correlator_names``. For two correlators this is
    ``y * 4 + correlator_0 * 2 + correlator_1``.

    Alignment-cell accuracies match the UrbanCars reference implementation:
    accuracy is computed in every target-class cell and then macro-averaged
    across target classes. Marginal worst-group accuracies are sample-weighted
    after marginalizing the other correlators.
    """
    if num_classes != 2:
        raise ValueError(
            "Multi-spurious alignment metrics currently require binary targets"
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_labels = np.asarray(group_labels)

    for name, values in (
        ("y_true", y_true),
        ("y_pred", y_pred),
        ("group_labels", group_labels),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
    if not (len(y_true) == len(y_pred) == len(group_labels)):
        raise ValueError(
            "y_true, y_pred, and group_labels must have the same length"
        )
    if len(y_true) == 0:
        raise ValueError("Multi-spurious metrics require at least one sample")
    if not np.all(np.isin(y_true, np.arange(num_classes))):
        raise ValueError("y_true contains labels outside the configured classes")
    if not np.all(np.equal(group_labels, np.floor(group_labels))):
        raise ValueError("group_labels must contain integer-valued group IDs")

    group_labels = group_labels.astype(np.int64, copy=False)
    num_correlators = len(config.correlator_names)
    combinations_per_class = 2**num_correlators
    expected_num_groups = num_classes * combinations_per_class

    if np.any(group_labels < 0) or np.any(group_labels >= expected_num_groups):
        raise ValueError(
            "Multi-spurious group IDs must be in "
            f"[0, {expected_num_groups - 1}]"
        )

    decoded_targets = group_labels // combinations_per_class
    if not np.array_equal(decoded_targets, y_true.astype(np.int64, copy=False)):
        raise ValueError(
            "Joint group IDs are inconsistent with y_true under the configured "
            "class-major encoding"
        )

    shortcut_code = group_labels % combinations_per_class
    correlator_values = np.empty(
        (len(group_labels), num_correlators), dtype=np.int64
    )
    for index in range(num_correlators):
        divisor = 2 ** (num_correlators - index - 1)
        correlator_values[:, index] = (shortcut_code // divisor) % 2

    correct = y_pred == y_true
    aligned = correlator_values == y_true[:, None]

    alignment_accuracies: Dict[str, Dict[str, Any]] = {}
    alignment_accuracy_by_pattern: Dict[Tuple[bool, ...], float] = {}
    alignment_support_by_pattern: Dict[Tuple[bool, ...], int] = {}

    for alignment_pattern in product((False, True), repeat=num_correlators):
        cell_name = "__".join(
            f"{name}_{'present' if is_aligned else 'absent'}"
            for name, is_aligned in zip(
                config.correlator_names, alignment_pattern
            )
        )
        cell_mask = np.all(
            aligned == np.asarray(alignment_pattern, dtype=bool), axis=1
        )
        per_class_accuracy: Dict[int, float] = {}
        for class_id in range(num_classes):
            class_mask = cell_mask & (y_true == class_id)
            per_class_accuracy[class_id] = (
                float(np.mean(correct[class_mask]))
                if np.any(class_mask)
                else float("nan")
            )

        class_accuracies = np.asarray(
            list(per_class_accuracy.values()), dtype=np.float64
        )
        cell_accuracy = (
            float(np.mean(class_accuracies))
            if np.all(np.isfinite(class_accuracies))
            else float("nan")
        )
        cell_support = int(np.sum(cell_mask))
        alignment_accuracy_by_pattern[alignment_pattern] = cell_accuracy
        alignment_support_by_pattern[alignment_pattern] = cell_support
        alignment_accuracies[cell_name] = {
            "accuracy": cell_accuracy,
            "support": cell_support,
            "per_class_accuracy": per_class_accuracy,
        }

    id_accuracy = 0.0
    id_accuracy_is_defined = True
    for alignment_pattern, cell_accuracy in alignment_accuracy_by_pattern.items():
        if not np.isfinite(cell_accuracy):
            id_accuracy_is_defined = False
            break
        cell_weight = 1.0
        for is_aligned, probability in zip(
            alignment_pattern, config.train_alignment_probabilities
        ):
            cell_weight *= probability if is_aligned else 1.0 - probability
        id_accuracy += cell_weight * cell_accuracy
    if not id_accuracy_is_defined:
        id_accuracy = float("nan")

    all_present = tuple(True for _ in range(num_correlators))
    all_absent = tuple(False for _ in range(num_correlators))
    gaps: Dict[str, float] = {}
    for correlator_index, correlator_name in enumerate(config.correlator_names):
        single_absent = list(all_present)
        single_absent[correlator_index] = False
        cell_accuracy = alignment_accuracy_by_pattern[tuple(single_absent)]
        gaps[correlator_name] = (
            float(cell_accuracy - id_accuracy)
            if np.isfinite(cell_accuracy) and np.isfinite(id_accuracy)
            else float("nan")
        )
    all_absent_accuracy = alignment_accuracy_by_pattern[all_absent]
    gaps["joint"] = (
        float(all_absent_accuracy - id_accuracy)
        if np.isfinite(all_absent_accuracy) and np.isfinite(id_accuracy)
        else float("nan")
    )

    marginal_group_accuracies: Dict[str, Dict[str, Dict[str, Any]]] = {}
    worst_group_accuracies: Dict[str, float] = {}
    for correlator_index, correlator_name in enumerate(config.correlator_names):
        correlator_groups: Dict[str, Dict[str, Any]] = {}
        cell_accuracies = []
        complete_grid = True
        for class_id in range(num_classes):
            for correlator_value in range(2):
                mask = (
                    (y_true == class_id)
                    & (correlator_values[:, correlator_index] == correlator_value)
                )
                accuracy = (
                    float(np.mean(correct[mask]))
                    if np.any(mask)
                    else float("nan")
                )
                support = int(np.sum(mask))
                correlator_groups[
                    f"class_{class_id}__value_{correlator_value}"
                ] = {
                    "accuracy": accuracy,
                    "support": support,
                }
                if np.isfinite(accuracy):
                    cell_accuracies.append(accuracy)
                else:
                    complete_grid = False

        marginal_group_accuracies[correlator_name] = correlator_groups
        worst_group_accuracies[correlator_name] = (
            float(min(cell_accuracies))
            if complete_grid and cell_accuracies
            else float("nan")
        )

    joint_group_accuracies: Dict[int, Dict[str, Any]] = {}
    joint_accuracies = []
    complete_joint_grid = True
    for group_id in range(expected_num_groups):
        mask = group_labels == group_id
        accuracy = (
            float(np.mean(correct[mask])) if np.any(mask) else float("nan")
        )
        support = int(np.sum(mask))
        joint_group_accuracies[group_id] = {
            "accuracy": accuracy,
            "support": support,
        }
        if np.isfinite(accuracy):
            joint_accuracies.append(accuracy)
        else:
            complete_joint_grid = False

    worst_group_accuracies["joint"] = (
        float(min(joint_accuracies))
        if complete_joint_grid and joint_accuracies
        else float("nan")
    )

    return {
        "correlator_names": list(config.correlator_names),
        "train_alignment_probabilities": {
            name: float(probability)
            for name, probability in zip(
                config.correlator_names,
                config.train_alignment_probabilities,
            )
        },
        "id_accuracy": float(id_accuracy),
        "gaps": gaps,
        "worst_group_accuracies": worst_group_accuracies,
        "alignment_accuracies": alignment_accuracies,
        "marginal_group_accuracies": marginal_group_accuracies,
        "joint_group_accuracies": joint_group_accuracies,
        "complete_alignment_grid": all(
            support > 0 for support in alignment_support_by_pattern.values()
        ),
        "complete_joint_grid": complete_joint_grid,
    }


def calculate_dataset_multi_spurious_metrics(
    dataset_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    num_classes: int = 2,
) -> Optional[Dict[str, Any]]:
    """Run multi-spurious metrics only for datasets declaring such a schema."""
    dataset_config = DATASET_REGISTRY.get(dataset_name.lower())
    if (
        dataset_config is None
        or not dataset_config.has_multiple_spurious_correlators
    ):
        return None
    return calculate_multi_spurious_metrics(
        y_true,
        y_pred,
        group_labels,
        dataset_config.multi_spurious,
        num_classes=num_classes,
    )


def calculate_hierarchical_metrics(
    hierarchical_predictions: np.ndarray,
    num_iterations: int,
    y_true: Optional[np.ndarray] = None,
    class_predictions: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Calculate hierarchical-specific metrics (node distribution, utilization, per-node accuracy)."""

    final_num_nodes = get_num_nodes(num_iterations, num_classes)
    node_distribution = {}

    for node_idx in range(final_num_nodes):
        node_mask = hierarchical_predictions == node_idx
        node_count = np.sum(node_mask)

        node_info = {
            "count": int(node_count),
            "percentage": float(node_count / len(hierarchical_predictions) * 100),
        }

        if y_true is not None and class_predictions is not None and node_count > 0:
            node_correct = np.sum((class_predictions[node_mask] == y_true[node_mask]))
            node_info["accuracy"] = float(node_correct / node_count)

        node_distribution[f"node_{node_idx}"] = node_info

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
    """Compute transition counts and rates between iterations."""
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
    """Compute prediction transitions between consecutive iterations."""

    prev_correct = prev_predictions == labels
    curr_correct = curr_predictions == labels

    result = {"overall": _compute_transition_stats(prev_correct, curr_correct)}

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
    """Accuracy on difficult nodes (odd-indexed). Returns 1.0 if none exist."""
    difficult_mask = (true_pseudo_labels % 2 == 1)

    if np.sum(difficult_mask) == 0:
        return 1.0

    return float(accuracy_score(
        true_pseudo_labels[difficult_mask],
        predicted_pseudo_labels[difficult_mask]
    ))


def compute_difficult_node_f1(
    true_pseudo_labels: np.ndarray,
    predicted_pseudo_labels: np.ndarray,
) -> float:
    """Macro F1 on difficult nodes (odd-indexed). Returns 1.0 if none exist."""
    difficult_mask = (true_pseudo_labels % 2 == 1)

    if np.sum(difficult_mask) == 0:
        return 1.0

    # Restrict labels to difficult ones so easy classes don't drag macro F1 to 0
    difficult_labels = np.unique(true_pseudo_labels[difficult_mask])
    return float(f1_score(
        true_pseudo_labels[difficult_mask],
        predicted_pseudo_labels[difficult_mask],
        labels=difficult_labels,
        average="macro",
        zero_division=1.0
    ))


def _compute_per_group_routing_stats(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    num_expected_groups: int,
    min_samples_threshold: int = 0,
) -> Dict[str, Any]:
    """Compute per-group routing stats used by pseudo-label metrics."""
    group_accs_all = {}
    group_accs_filled = {}
    node_sizes = {}
    per_node_accuracy = {}

    min_required = max(1, min_samples_threshold)

    for gid in range(num_expected_groups):
        mask = true_labels == gid
        node_size = int(np.sum(mask))
        node_sizes[int(gid)] = node_size

        if node_size > 0:
            acc = float(accuracy_score(true_labels[mask], predicted_labels[mask]))
            group_accs_all[int(gid)] = acc
            per_node_accuracy[int(gid)] = acc
            if node_size >= min_required:
                group_accs_filled[int(gid)] = acc
        else:
            group_accs_all[int(gid)] = 0.0
            per_node_accuracy[int(gid)] = 0.0

    return {
        "group_accuracies": group_accs_all,
        "group_accuracies_filled": group_accs_filled,
        "node_sizes": node_sizes,
        "per_node_accuracy": per_node_accuracy,
        "worst_group_accuracy": float(min(group_accs_all.values()))
        if group_accs_all
        else 0.0,
        "worst_group_accuracy_filled": float(min(group_accs_filled.values()))
        if group_accs_filled
        else 0.0,
        "num_filled_classes": len(group_accs_filled),
        "num_expected_classes": num_expected_groups,
    }


def compute_pseudo_label_metrics(
    true_pseudo_labels: np.ndarray,
    predicted_pseudo_labels: np.ndarray,
    num_expected_classes: int,
    min_samples_threshold: int = 0,
    base_num_classes: Optional[int] = None,
    routing_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Pseudo-label routing metrics (accuracy, F1, WGA, DNA, iter2-level collapsed)."""
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(true_pseudo_labels, predicted_pseudo_labels)
    f1 = f1_score(true_pseudo_labels, predicted_pseudo_labels, average="macro")

    node_level_stats = _compute_per_group_routing_stats(
        true_pseudo_labels,
        predicted_pseudo_labels,
        num_expected_classes,
        min_samples_threshold,
    )

    # Compute difficult node accuracy (DNA) and F1
    difficult_node_accuracy = compute_difficult_node_accuracy(
        true_pseudo_labels, predicted_pseudo_labels
    )
    difficult_node_f1 = compute_difficult_node_f1(
        true_pseudo_labels, predicted_pseudo_labels
    )

    results = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "worst_group_accuracy": node_level_stats["worst_group_accuracy"],
        "worst_group_accuracy_filled": node_level_stats[
            "worst_group_accuracy_filled"
        ],
        "difficult_node_accuracy": difficult_node_accuracy,
        "difficult_node_f1": difficult_node_f1,
        "group_accuracies": node_level_stats["group_accuracies"],
        "node_sizes": node_level_stats["node_sizes"],
        "per_node_accuracy": node_level_stats["per_node_accuracy"],
        "num_filled_classes": node_level_stats["num_filled_classes"],
        "num_expected_classes": node_level_stats["num_expected_classes"],
    }

    if base_num_classes is not None and base_num_classes > 0:
        iter2_expected_classes = int(base_num_classes) * 2
        if (
            num_expected_classes >= iter2_expected_classes
            and num_expected_classes % iter2_expected_classes == 0
        ):
            collapse_factor = num_expected_classes // iter2_expected_classes
            collapsed_true = (true_pseudo_labels // collapse_factor).astype(int)
            collapsed_pred = (predicted_pseudo_labels // collapse_factor).astype(int)

            iter2_level_stats = _compute_per_group_routing_stats(
                collapsed_true,
                collapsed_pred,
                iter2_expected_classes,
                min_samples_threshold,
            )

            results.update(
                {
                    "iter2_level_worst_group_accuracy": iter2_level_stats[
                        "worst_group_accuracy"
                    ],
                    "iter2_level_worst_group_accuracy_filled": iter2_level_stats[
                        "worst_group_accuracy_filled"
                    ],
                    "iter2_level_group_accuracies": iter2_level_stats[
                        "group_accuracies"
                    ],
                    "iter2_level_node_sizes": iter2_level_stats["node_sizes"],
                    "iter2_level_per_node_accuracy": iter2_level_stats[
                        "per_node_accuracy"
                    ],
                    "iter2_level_num_filled_classes": iter2_level_stats[
                        "num_filled_classes"
                    ],
                    "iter2_level_num_expected_classes": iter2_level_stats[
                        "num_expected_classes"
                    ],
                }
            )

            # Iter-2-level DNA/DNF1
            iter2_dna = compute_difficult_node_accuracy(collapsed_true, collapsed_pred)
            iter2_dnf1 = compute_difficult_node_f1(collapsed_true, collapsed_pred)
            results["iter2_level_difficult_node_accuracy"] = iter2_dna
            results["iter2_level_difficult_node_f1"] = iter2_dnf1

            # Iter-2-level AUROC (one-vs-rest per group), computed when routing
            # probabilities are provided.
            if routing_probs is not None:
                from sklearn.metrics import roc_auc_score

                collapsed_probs = np.zeros(
                    (len(collapsed_true), iter2_expected_classes), dtype=np.float64
                )
                for j in range(iter2_expected_classes):
                    start = j * collapse_factor
                    end = (j + 1) * collapse_factor
                    collapsed_probs[:, j] = routing_probs[:, start:end].sum(axis=1)

                iter2_auroc_per_group: Dict[int, float] = {}
                for j in range(iter2_expected_classes):
                    binary_labels = (collapsed_true == j).astype(int)
                    if 0 < binary_labels.sum() < len(binary_labels):
                        try:
                            auroc = roc_auc_score(binary_labels, collapsed_probs[:, j])
                            iter2_auroc_per_group[int(j)] = float(auroc)
                        except ValueError:
                            pass

                results["iter2_level_per_group_auroc"] = iter2_auroc_per_group
                results["iter2_level_worst_group_auroc"] = (
                    float(min(iter2_auroc_per_group.values()))
                    if iter2_auroc_per_group
                    else float("nan")
                )

    return results


def get_worst_pseudo_node_stats(
    pseudo_metrics: Optional[Dict[str, Any]],
    metric: str = "wga",
) -> Tuple[Optional[float], Optional[int]]:
    """Return (value, n_worst) for the worst iter-2-level pseudo group."""
    if pseudo_metrics is None:
        return None, None

    if metric == "auroc":
        per_group = pseudo_metrics.get("iter2_level_per_group_auroc")
        node_sizes = pseudo_metrics.get("iter2_level_node_sizes")
        if not per_group or not node_sizes:
            return None, None
        valid = [(gid, val) for gid, val in per_group.items() if node_sizes.get(gid, 0) > 0]
        if not valid:
            return None, None
        worst_gid, worst_val = min(valid, key=lambda x: x[1])
        return float(worst_val), int(node_sizes[worst_gid])

    per_node_acc = pseudo_metrics.get("iter2_level_per_node_accuracy")
    node_sizes = pseudo_metrics.get("iter2_level_node_sizes")
    if not per_node_acc or not node_sizes:
        per_node_acc = pseudo_metrics.get("per_node_accuracy", {})
        node_sizes = pseudo_metrics.get("node_sizes", {})
    if not per_node_acc or not node_sizes:
        return None, None

    valid = [(gid, acc) for gid, acc in per_node_acc.items() if node_sizes.get(gid, 0) > 0]
    if not valid:
        return None, None

    worst_gid, worst_acc = min(valid, key=lambda x: x[1])
    return float(worst_acc), int(node_sizes[worst_gid])


def print_evaluation_report(
    metrics: Dict[str, Any], title: str = "Evaluation Results"
) -> None:
    """Print an evaluation report."""

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

    if "group_metrics" in metrics:
        print(f"\nGroup Performance:")
        summary = metrics["group_metrics"]["summary"]
        print(f"  Worst-group accuracy: {summary['worst_group_accuracy']:.3f}")
        print(f"  Mean group accuracy:  {summary['mean_group_accuracy']:.3f}")

    if "multi_spurious_metrics" in metrics:
        multi = metrics["multi_spurious_metrics"]
        print(f"  ID accuracy:          {multi['id_accuracy']:.3f}")
        for name in [*multi["correlator_names"], "joint"]:
            print(
                f"  {name} gap / WGA: {multi['gaps'][name]:+.3f} / "
                f"{multi['worst_group_accuracies'][name]:.3f}"
            )

    if "node_utilization" in metrics:
        print(
            f"  Node utilization:     {metrics['node_utilization']:.3f} ({metrics['used_nodes']}/{metrics['total_nodes']} nodes)"
        )
        print(f"  Mean group accuracy:  {summary['mean_group_accuracy']:.3f}")

    if "node_utilization" in metrics:
        print(
            f"  Node utilization:     {metrics['node_utilization']:.3f} ({metrics['used_nodes']}/{metrics['total_nodes']} nodes)"
        )
        print(f"  Mean group accuracy:  {summary['mean_group_accuracy']:.3f}")

    if "node_utilization" in metrics:
        print(
            f"  Node utilization:     {metrics['node_utilization']:.3f} ({metrics['used_nodes']}/{metrics['total_nodes']} nodes)"
        )
