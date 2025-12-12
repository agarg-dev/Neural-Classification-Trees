"""Analysis utilities for hierarchical classification results."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from ..utils.hierarchical import get_node_class, get_node_class_correct_label

# Export get_node_label for use in other modules
__all__ = ['get_node_label', 'analyze_hierarchical_results']


def get_node_label(
    node_index: int, 
    num_iterations: int, 
    class_names: Optional[List[str]] = None,
    hierarchical: str = "none"
) -> str:
    """
    Generate a hierarchical label for a given node.
    
    This is a general-purpose label generator used across evaluation, visualization,
    and logging. Supports custom class names.
    
    Args:
        node_index: Index of the final node
        num_iterations: Number of iterations in the hierarchy
        class_names: Optional list of class names (e.g., ['Landbird', 'Waterbird']).
                    If None, uses 'Class 0', 'Class 1', etc.
        hierarchical: Routing mode ('none', 'prediction_routing', 'correct_label_routing')
    
    Returns:
        str: Human-readable hierarchical label with arrow separators
        
    Examples:
        >>> get_node_label(3, 2, ['Landbird', 'Waterbird'])
        'Waterbird → Difficult'
        >>> get_node_label(1, 1, ['Cat', 'Dog'])
        'Dog'
        >>> get_node_label(5, 3, ['Cat', 'Dog'])
        'Dog → Easy → Difficult'
    """
    if num_iterations <= 0:
        return f"Node {node_index}"

    final_num_nodes = 2 ** num_iterations
    
    # Determine which class this node predicts
    if hierarchical == "prediction_routing":
        pred_class = get_node_class(node_index, num_iterations)
    elif hierarchical == "correct_label_routing":
        pred_class = get_node_class_correct_label(node_index, num_iterations)
    else:  # "none"
        pred_class = 1 if node_index >= final_num_nodes // 2 else 0

    # Get class name
    if class_names is not None and pred_class < len(class_names):
        class_name = class_names[pred_class]
    else:
        class_name = f"Class {pred_class}"

    # Iteration 1: Just return class name
    if num_iterations == 1:
        return class_name

    # Iteration 2+: Add Easy/Difficult hierarchy using bit extraction
    path_parts = [class_name]
    
    # Extract difficulty bits in chronological order (high to low bit positions)
    # This gives: first split, second split, third split, etc.
    for bit_pos in range(num_iterations - 2, -1, -1):
        is_difficult = (node_index >> bit_pos) & 1
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
    """
    Detect if this is likely a Waterbird dataset based on group structure.

    Args:
        group_labels: Group labels tensor
        group_names: Optional group names dict

    Returns:
        bool: True if likely Waterbird dataset
    """
    if group_names:
        # Check if group names contain Waterbird-specific terms
        names_str = str(group_names).lower()
        waterbird_terms = ['bird', 'land', 'water', 'easy', 'difficult']
        return any(term in names_str for term in waterbird_terms)

    if group_labels is not None:
        # Check if we have exactly 4 groups (0, 1, 2, 3)
        unique_groups = torch.unique(group_labels) if isinstance(group_labels, torch.Tensor) else np.unique(group_labels)
        return len(unique_groups) == 4 and set(unique_groups.tolist()) == {0, 1, 2, 3}

    return False


def analyze_hierarchical_nodes(
    predicted_hier_classes: torch.Tensor,
    true_binary_labels: torch.Tensor,
    group_labels: Optional[torch.Tensor] = None,
    num_iterations: int = 1,
    group_names: Optional[Dict[int, str]] = None,
    verbose: int = 1,
    hierarchical: str = "none",
    class_names: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    Analyze the performance of each node in the hierarchical classifier.
    Args:
        predicted_hier_classes: Predicted hierarchical node indices
        true_binary_labels: True binary class labels
        group_labels: Optional group labels for subgroup analysis
        num_iterations: Number of iterations in the hierarchy
        group_names: Optional mapping from group IDs to names
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        hierarchical: Routing mode ('none', 'prediction_routing', 'correct_label_routing')
        class_names: Optional binary class names
    Returns:
        Dict[str, Any]: Analysis results
    """
    # Convert to numpy for easier processing
    if isinstance(predicted_hier_classes, torch.Tensor):
        predicted_hier_classes = predicted_hier_classes.cpu().numpy()
    if isinstance(true_binary_labels, torch.Tensor):
        true_binary_labels = true_binary_labels.cpu().numpy()
    if group_labels is not None and isinstance(group_labels, torch.Tensor):
        group_labels = group_labels.cpu().numpy()

    final_num_nodes = 2 ** num_iterations
    node_analysis = {}

    if verbose:
        print(f"\n--- Detailed Analysis of Hierarchical Nodes ---")
        print(f"Number of iterations: {num_iterations}")
        print(f"Total nodes: {final_num_nodes}")

    for node_idx in range(final_num_nodes):
        node_info = _analyze_single_node(
            node_idx, predicted_hier_classes, true_binary_labels, group_labels,
            num_iterations, group_names, verbose, hierarchical, class_names
        )
        node_analysis[f'node_{node_idx}'] = node_info

    # Summary statistics
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
    node_idx: int, predicted_hier_classes: np.ndarray, true_binary_labels: np.ndarray,
    group_labels: Optional[np.ndarray], num_iterations: int, group_names: Optional[Dict[int, str]],
    verbose: int, hierarchical: str = "none", class_names: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """Analyze a single node - simple helper to reduce the monster function."""
    mask = (predicted_hier_classes == node_idx)
    population = np.sum(mask)
    final_num_nodes = 2 ** num_iterations

    # Basic node info
    path_str = get_node_label(node_idx, num_iterations, class_names, hierarchical)
    if hierarchical == "prediction_routing":
        node_predicts_binary = get_node_class(node_idx, num_iterations)
    elif hierarchical == "correct_label_routing":
        node_predicts_binary = get_node_class_correct_label(node_idx, num_iterations)
    else:  # "none"
        node_predicts_binary = 1 if node_idx >= final_num_nodes / 2 else 0

    if class_names and node_predicts_binary < len(class_names):
        prediction_str = class_names[node_predicts_binary]
    else:
        prediction_str = f"Class {node_predicts_binary}"

    node_info = {
        'node_index': node_idx,
        'path_description': path_str,
        'predicted_class': node_predicts_binary,
        'population': int(population),
        'percentage': float(population / len(predicted_hier_classes) * 100)
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Node {node_idx}: {path_str}")
        print(f"  Predicts: {prediction_str}")
        print(f"  Population: {population} samples ({node_info['percentage']:.2f}%)")

    if population == 0:
        node_info.update({'accuracy': 0.0, 'group_breakdown': {}})
        if verbose:
            print("  Status: UNUSED (no samples routed to this node)")
    else:
        # Calculate accuracy
        true_labels_in_node = true_binary_labels[mask]
        accuracy = np.mean(true_labels_in_node == node_predicts_binary)
        node_info['accuracy'] = float(accuracy)

        if verbose:
            print(f"  Accuracy: {accuracy * 100:.2f}%")

        # Group analysis
        group_breakdown = {}
        if group_labels is not None:
            groups_in_node = group_labels[mask]
            if verbose:
                print("  Group Breakdown:")

            for group_id in np.unique(groups_in_node):
                group_mask = (groups_in_node == group_id)
                group_population = np.sum(group_mask)

                if group_population > 0:
                    group_name = group_names.get(group_id, f"Group_{group_id}") if group_names else f"Group_{group_id}"
                    expected_label = 1 if group_id >= 2 else 0
                    is_correct_prediction = (node_predicts_binary == expected_label)

                    group_breakdown[group_name] = {
                        'population': int(group_population),
                        'group_id': int(group_id),
                        'expected_label': expected_label,
                        'prediction_correctness': 'CORRECT' if is_correct_prediction else 'INCORRECT'
                    }

                    if verbose:
                        correctness = 'CORRECTLY' if is_correct_prediction else 'INCORRECTLY'
                        print(f"    {group_name:<28}: {group_population:>4} samples (Predicts {correctness})")

        node_info['group_breakdown'] = group_breakdown

    return node_info


def analyze_prediction_paths(
    predicted_hier_classes: torch.Tensor,
    true_binary_labels: torch.Tensor,
    num_iterations: int
) -> Dict[str, Any]:
    """
    Analyze the prediction paths taken by different samples.

    Args:
        predicted_hier_classes: Predicted hierarchical classes
        true_binary_labels: True binary labels
        num_iterations: Number of iterations

    Returns:
        Dict[str, Any]: Path analysis results
    """
    # Convert to numpy
    if isinstance(predicted_hier_classes, torch.Tensor):
        predicted_hier_classes = predicted_hier_classes.cpu().numpy()
    if isinstance(true_binary_labels, torch.Tensor):
        true_binary_labels = true_binary_labels.cpu().numpy()

    # Analyze correctness at each level of the hierarchy
    level_analysis = {}

    for level in range(num_iterations):
        # Calculate node assignments at this level
        level_nodes = predicted_hier_classes // (2 ** (num_iterations - level - 1))
        unique_nodes = np.unique(level_nodes)

        level_info = {}
        for node in unique_nodes:
            mask = (level_nodes == node)
            node_true_labels = true_binary_labels[mask]

            # Determine what this level-node predicts
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
    true_binary_labels: torch.Tensor,
    num_iterations: int,
    hierarchical: str = "none"
) -> Dict[str, Any]:
    """
    Generate data for visualizing the hierarchical classification results.

    Args:
        predicted_hier_classes: Predicted hierarchical classes
        true_binary_labels: True binary labels
        num_iterations: Number of iterations
        hierarchical: Routing mode ('none', 'prediction_routing', 'correct_label_routing')

    Returns:
        Dict[str, Any]: Visualization data
    """
    # Convert to numpy
    if isinstance(predicted_hier_classes, torch.Tensor):
        predicted_hier_classes = predicted_hier_classes.cpu().numpy()
    if isinstance(true_binary_labels, torch.Tensor):
        true_binary_labels = true_binary_labels.cpu().numpy()

    final_num_nodes = 2 ** num_iterations

    # Create tree structure data
    tree_data = {
        'nodes': [],
        'edges': []
    }

    # Add all nodes
    for level in range(num_iterations + 1):
        level_nodes = 2 ** level
        for node_idx in range(level_nodes):
            if level == num_iterations:
                # Final level - use actual predictions
                node_id = f"L{level}_N{node_idx}"
                mask = (predicted_hier_classes == node_idx)
                population = np.sum(mask)
                if hierarchical == "prediction_routing":
                    node_class = get_node_class(node_idx, num_iterations)
                elif hierarchical == "correct_label_routing":
                    node_class = get_node_class_correct_label(node_idx, num_iterations)
                else:  # "none"
                    node_class = 1 if node_idx >= final_num_nodes / 2 else 0
                accuracy = np.mean(true_binary_labels[mask] == node_class) if population > 0 else 0.0
            else:
                # Intermediate levels
                node_id = f"L{level}_N{node_idx}"
                population = 0  # Would need to calculate based on tree traversal
                accuracy = 0.0

            tree_data['nodes'].append({
                'id': node_id,
                'level': level,
                'index': node_idx,
                'population': int(population),
                'accuracy': float(accuracy)
            })

            # Add edges to children (except for final level)
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
    """
    Compare results across different iterations of the hierarchy.

    Args:
        results_by_iteration: Results for each iteration

    Returns:
        Dict[str, Any]: Comparison analysis
    """
    comparison = {
        'accuracy_progression': {},
        'node_utilization_progression': {},
        'worst_group_accuracy_progression': {}
    }

    for iteration, results in results_by_iteration.items():
        # Extract key metrics
        if 'binary_metrics' in results:
            comparison['accuracy_progression'][iteration] = results['binary_metrics']['accuracy']

        if 'node_utilization' in results:
            comparison['node_utilization_progression'][iteration] = results['node_utilization']

        if 'group_metrics' in results and 'summary' in results['group_metrics']:
            comparison['worst_group_accuracy_progression'][iteration] = results['group_metrics']['summary']['worst_group_accuracy']

    return comparison