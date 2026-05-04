"""Analysis utilities for hierarchical classification results."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from ..utils.hierarchical import (
    get_num_nodes,
    get_nodes_per_class,
)

# Export get_node_label for use in other modules
__all__ = ['get_node_label', 'analyze_hierarchical_results', 'compute_head_calibration_diagnostics']


def get_node_label(
    node_index: int,
    num_iterations: int,
    class_names: Optional[List[str]] = None,
    hierarchical: str = "none",
    num_classes: int = 2
) -> str:
    """Build a human-readable hierarchical label for a given node.

    Examples:
        >>> get_node_label(3, 2, ['Landbird', 'Waterbird'])
        'Waterbird → Difficult'
        >>> get_node_label(5, 3, ['Cat', 'Dog'])
        'Dog → Easy → Difficult'
    """
    if num_iterations <= 0:
        return f"Node {node_index}"

    nodes_per_class = get_nodes_per_class(num_iterations)
    pred_class = node_index // nodes_per_class

    if class_names is not None and pred_class < len(class_names):
        class_name = class_names[pred_class]
    else:
        class_name = f"Class {pred_class}"

    if num_iterations == 1:
        return class_name

    path_parts = [class_name]

    nodes_per_class = get_nodes_per_class(num_iterations)
    node_within_class = node_index % nodes_per_class

    # High-to-low bit position gives chronological split order
    for bit_pos in range(num_iterations - 2, -1, -1):
        is_difficult = (node_within_class >> bit_pos) & 1
        path_parts.append("Difficult" if is_difficult else "Easy")

    return " → ".join(path_parts)


def get_waterbird_group_names() -> Dict[int, str]:
    """Get standard Waterbird dataset group names."""
    return {
        0: "Land Bird on Land (Easy)",
        1: "Land Bird on Water (Difficult)",
        2: "Water Bird on Land (Difficult)",
        3: "Water Bird on Water (Easy)"
    }


def detect_waterbird_dataset(group_labels: Optional[torch.Tensor], group_names: Optional[Dict[int, str]]) -> bool:
    """Detect if this is likely a Waterbird dataset from its group structure."""
    if group_names:
        names_str = str(group_names).lower()
        waterbird_terms = ['bird', 'land', 'water', 'easy', 'difficult']
        return any(term in names_str for term in waterbird_terms)

    if group_labels is not None:
        unique_groups = torch.unique(group_labels) if isinstance(group_labels, torch.Tensor) else np.unique(group_labels)
        return len(unique_groups) == 4 and set(unique_groups.tolist()) == {0, 1, 2, 3}

    return False


def analyze_hierarchical_nodes(
    predicted_hier_classes: torch.Tensor,
    true_labels: torch.Tensor,
    group_labels: Optional[torch.Tensor] = None,
    num_iterations: int = 1,
    group_names: Optional[Dict[int, str]] = None,
    verbose: int = 1,
    hierarchical: str = "none",
    class_names: Optional[Tuple[str, ...]] = None,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Analyze the performance of each node in the hierarchical classifier."""
    if isinstance(predicted_hier_classes, torch.Tensor):
        predicted_hier_classes = predicted_hier_classes.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if group_labels is not None and isinstance(group_labels, torch.Tensor):
        group_labels = group_labels.cpu().numpy()

    final_num_nodes = get_num_nodes(num_iterations, num_classes)
    node_analysis = {}

    if verbose:
        print(f"\n--- Detailed Analysis of Hierarchical Nodes ---")
        print(f"Number of iterations: {num_iterations}")
        print(f"Total nodes: {final_num_nodes}")

    for node_idx in range(final_num_nodes):
        node_info = _analyze_single_node(
            node_idx, predicted_hier_classes, true_labels, group_labels,
            num_iterations, group_names, verbose, hierarchical, class_names, num_classes
        )
        node_analysis[f'node_{node_idx}'] = node_info

    used_nodes = sum(1 for info in node_analysis.values() if info['population'] > 0)
    node_utilization = used_nodes / final_num_nodes

    summary = {
        'total_nodes': final_num_nodes,
        'used_nodes': used_nodes,
        'node_utilization': node_utilization,
        'num_iterations': num_iterations
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Summary:")
        print(f"  Used nodes: {used_nodes}/{final_num_nodes} ({node_utilization*100:.1f}%)")
        print(f"  Hierarchy depth: {num_iterations} iterations")

    return {
        'node_analysis': node_analysis,
        'summary': summary
    }


def _analyze_single_node(
    node_idx: int, predicted_hier_classes: np.ndarray, true_labels: np.ndarray,
    group_labels: Optional[np.ndarray], num_iterations: int, group_names: Optional[Dict[int, str]],
    verbose: int, hierarchical: str = "none", class_names: Optional[Tuple[str, ...]] = None,
    num_classes: int = 2
) -> Dict[str, Any]:
    """Analyze a single node."""
    mask = (predicted_hier_classes == node_idx)
    population = np.sum(mask)

    path_str = get_node_label(node_idx, num_iterations, class_names, hierarchical, num_classes)
    nodes_per_class = get_nodes_per_class(num_iterations)
    node_predicts_class = node_idx // nodes_per_class

    if class_names and node_predicts_class < len(class_names):
        prediction_str = class_names[node_predicts_class]
    else:
        prediction_str = f"Class {node_predicts_class}"

    node_info = {
        'node_index': node_idx,
        'path_description': path_str,
        'predicted_class': node_predicts_class,
        'population': int(population),
        'percentage': float(population / len(predicted_hier_classes) * 100)
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Node {node_idx}: {path_str}")

    if population == 0:
        node_info.update({'accuracy': 0.0, 'group_breakdown': {}})
        if verbose:
            print(f"  Predicts: {prediction_str}  |  Population: 0 samples  |  Status: UNUSED")
    else:
        true_labels_in_node = true_labels[mask]
        accuracy = np.mean(true_labels_in_node == node_predicts_class)
        node_info['accuracy'] = float(accuracy)

        if verbose:
            print(f"  Predicts: {prediction_str}  |  Population: {population} samples ({node_info['percentage']:.1f}%)  |  Accuracy: {accuracy:.2%}")

        group_breakdown = {}
        if group_labels is not None:
            groups_in_node = group_labels[mask]
            if verbose:
                print("  Group Breakdown:")

            # group = class * num_attrs + attr_value (e.g. 4 groups -> 2 attrs/class; 25 -> 5)
            num_groups = len(group_names) if group_names else int(np.max(groups_in_node)) + 1
            num_attrs_per_class = max(1, num_groups // num_classes)

            for group_id in np.unique(groups_in_node):
                group_mask = (groups_in_node == group_id)
                group_population = np.sum(group_mask)

                if group_population > 0:
                    group_name = group_names.get(group_id, f"Group_{group_id}") if group_names else f"Group_{group_id}"
                    expected_label = group_id // num_attrs_per_class
                    is_correct_prediction = (node_predicts_class == expected_label)

                    group_breakdown[group_name] = {
                        'population': int(group_population),
                        'group_id': int(group_id),
                        'expected_label': expected_label,
                        'prediction_correctness': 'Correct' if is_correct_prediction else 'Wrong'
                    }

                    if verbose:
                        correctness = '[Correct]' if is_correct_prediction else '[Wrong]  '
                        print(f"    {group_name:<34}: {group_population:>4} samples  {correctness}")

        node_info['group_breakdown'] = group_breakdown

    return node_info


def compute_head_calibration_diagnostics(
    logits: np.ndarray,
    hierarchical_predictions: np.ndarray,
    num_iterations: int,
    class_names: Optional[Tuple[str, ...]] = None,
    num_classes: int = 2,
    true_node_labels: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Diagnostics for cross-head logit scale: per-head stats, decision margins, per-node recall."""
    num_nodes = logits.shape[1]
    results = {}

    logit_stats = {}
    for h in range(num_nodes):
        col = logits[:, h]
        # Exclude -inf logits (masked/untrained nodes)
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            logit_stats[h] = {"mean": float(np.mean(valid)), "std": float(np.std(valid))}
        else:
            logit_stats[h] = {"mean": float("nan"), "std": float("nan")}
    results["logit_stats"] = logit_stats

    # Per-sample decision margin: logit[argmax] - second-best logit
    n_samples = logits.shape[0]
    margins = np.full(n_samples, np.nan)
    for i in range(n_samples):
        row = logits[i]
        pred = hierarchical_predictions[i]
        best = row[pred]
        # Second best: max of all other finite logits
        others = np.where(np.arange(num_nodes) != pred, row, -np.inf)
        second = np.max(others[np.isfinite(others)]) if np.any(np.isfinite(others)) else np.nan
        if np.isfinite(best) and np.isfinite(second):
            margins[i] = best - second

    margin_stats = {}
    for h in range(num_nodes):
        mask = hierarchical_predictions == h
        if np.sum(mask) > 0:
            m = margins[mask]
            valid_m = m[np.isfinite(m)]
            margin_stats[h] = {
                "count": int(np.sum(mask)),
                "mean_margin": float(np.mean(valid_m)) if len(valid_m) > 0 else float("nan"),
                "std_margin": float(np.std(valid_m)) if len(valid_m) > 0 else float("nan"),
            }
        else:
            margin_stats[h] = {"count": 0, "mean_margin": float("nan"), "std_margin": float("nan")}
    results["margin_stats"] = margin_stats

    if true_node_labels is not None:
        recall_stats = {}
        for h in range(num_nodes):
            true_mask = true_node_labels == h
            n_true = int(np.sum(true_mask))
            if n_true > 0:
                n_correct = int(np.sum(hierarchical_predictions[true_mask] == h))
                recall_stats[h] = {"true_count": n_true, "recall": float(n_correct / n_true)}
            else:
                recall_stats[h] = {"true_count": 0, "recall": float("nan")}
        results["per_node_recall"] = recall_stats

    if verbose:
        print("\n--- Head Calibration Diagnostics ---")

        print("\n  Logit scale per head (all test samples):")
        print(f"  {'Head':<5}  {'Node Label':<35}  {'Mean':>7}  {'Std':>7}")
        print(f"  {'─'*5}  {'─'*35}  {'─'*7}  {'─'*7}")
        for h in range(num_nodes):
            label = get_node_label(h, num_iterations, list(class_names) if class_names else None,
                                   "correct_label_routing", num_classes)
            s = logit_stats[h]
            mean_str = f"{s['mean']:+.3f}" if np.isfinite(s['mean']) else "  n/a "
            std_str  = f"{s['std']:.3f}"   if np.isfinite(s['std'])  else "  n/a "
            print(f"  {h:<5}  {label:<35}  {mean_str:>7}  {std_str:>7}")

        print("\n  Decision margin per predicted node (winning logit − second-best):")
        print(f"  {'Head':<5}  {'Node Label':<35}  {'Samples':>8}  {'Mean Margin':>12}  {'Std':>7}")
        print(f"  {'─'*5}  {'─'*35}  {'─'*8}  {'─'*12}  {'─'*7}")
        for h in range(num_nodes):
            label = get_node_label(h, num_iterations, list(class_names) if class_names else None,
                                   "correct_label_routing", num_classes)
            ms = margin_stats[h]
            if ms["count"] > 0 and np.isfinite(ms["mean_margin"]):
                print(f"  {h:<5}  {label:<35}  {ms['count']:>8}  {ms['mean_margin']:>+12.3f}  {ms['std_margin']:>7.3f}")
            else:
                print(f"  {h:<5}  {label:<35}  {ms['count']:>8}  {'—':>12}  {'—':>7}")

        if "per_node_recall" in results:
            print("\n  Per-node routing recall (correctly routed / true node samples):")
            print(f"  {'Head':<5}  {'Node Label':<35}  {'True N':>7}  {'Recall':>7}")
            print(f"  {'─'*5}  {'─'*35}  {'─'*7}  {'─'*7}")
            for h in range(num_nodes):
                label = get_node_label(h, num_iterations, list(class_names) if class_names else None,
                                       "correct_label_routing", num_classes)
                rs = results["per_node_recall"][h]
                if rs["true_count"] > 0:
                    print(f"  {h:<5}  {label:<35}  {rs['true_count']:>7}  {rs['recall']:>7.1%}")
                else:
                    print(f"  {h:<5}  {label:<35}  {rs['true_count']:>7}  {'—':>7}")

    return results


def analyze_prediction_paths(
    predicted_hier_classes: torch.Tensor,
    true_binary_labels: torch.Tensor,
    num_iterations: int
) -> Dict[str, Any]:
    """Analyze prediction paths per hierarchy level (binary-only)."""
    if isinstance(predicted_hier_classes, torch.Tensor):
        predicted_hier_classes = predicted_hier_classes.cpu().numpy()
    if isinstance(true_binary_labels, torch.Tensor):
        true_binary_labels = true_binary_labels.cpu().numpy()

    level_analysis = {}

    for level in range(num_iterations):
        level_nodes = predicted_hier_classes // (2 ** (num_iterations - level - 1))
        unique_nodes = np.unique(level_nodes)

        level_info = {}
        for node in unique_nodes:
            mask = (level_nodes == node)
            node_true_labels = true_binary_labels[mask]

            predicted_class = 1 if node >= (2 ** level) // 2 else 0

            accuracy = np.mean(node_true_labels == predicted_class) if len(node_true_labels) > 0 else 0.0

            level_info[f'node_{int(node)}'] = {
                'population': int(np.sum(mask)),
                'predicted_class': predicted_class,
                'accuracy': float(accuracy)
            }

        level_analysis[f'level_{level}'] = level_info

    return {
        'level_analysis': level_analysis,
        'num_levels': num_iterations
    }


def generate_hierarchy_visualization_data(
    predicted_hier_classes: torch.Tensor,
    true_labels: torch.Tensor,
    num_iterations: int,
    hierarchical: str = "none",
    num_classes: int = 2
) -> Dict[str, Any]:
    """Generate node/edge data for visualizing the hierarchical classification tree."""
    if isinstance(predicted_hier_classes, torch.Tensor):
        predicted_hier_classes = predicted_hier_classes.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    final_num_nodes = get_num_nodes(num_iterations, num_classes)

    tree_data = {
        'nodes': [],
        'edges': []
    }

    for level in range(num_iterations + 1):
        level_nodes = 2 ** level
        for node_idx in range(level_nodes):
            if level == num_iterations:
                node_id = f"L{level}_N{node_idx}"
                mask = (predicted_hier_classes == node_idx)
                population = np.sum(mask)
                nodes_per_class = get_nodes_per_class(num_iterations)
                node_class = node_idx // nodes_per_class
                accuracy = np.mean(true_labels[mask] == node_class) if population > 0 else 0.0
            else:
                node_id = f"L{level}_N{node_idx}"
                population = 0
                accuracy = 0.0

            tree_data['nodes'].append({
                'id': node_id,
                'level': level,
                'index': node_idx,
                'population': int(population),
                'accuracy': float(accuracy)
            })

            if level < num_iterations:
                child1_id = f"L{level+1}_N{node_idx*2}"
                child2_id = f"L{level+1}_N{node_idx*2+1}"
                tree_data['edges'].extend([
                    {'parent': node_id, 'child': child1_id, 'label': 'easy'},
                    {'parent': node_id, 'child': child2_id, 'label': 'hard'}
                ])

    return tree_data


def compare_iterations(
    results_by_iteration: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare results across iterations."""
    comparison = {
        'accuracy_progression': {},
        'node_utilization_progression': {},
        'worst_group_accuracy_progression': {}
    }

    for iteration, results in results_by_iteration.items():
        if 'binary_metrics' in results:
            comparison['accuracy_progression'][iteration] = results['binary_metrics']['accuracy']

        if 'node_utilization' in results:
            comparison['node_utilization_progression'][iteration] = results['node_utilization']

        if 'group_metrics' in results and 'summary' in results['group_metrics']:
            comparison['worst_group_accuracy_progression'][iteration] = results['group_metrics']['summary']['worst_group_accuracy']

    return comparison