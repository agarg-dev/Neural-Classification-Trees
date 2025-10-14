"""Evaluation metrics for hierarchical classification."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple, Any


def calculate_basic_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Calculate basic classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics

    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    return metrics


def calculate_group_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    group_labels: torch.Tensor,
    group_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Calculate metrics for different groups/subpopulations.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        group_labels: Group/subpopulation labels
        group_names: Optional mapping from group IDs to names

    Returns:
        Dict[str, Any]: Group-specific metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(group_labels, torch.Tensor):
        group_labels = group_labels.cpu().numpy()

    group_metrics = {}
    group_accuracies = []

    unique_groups = np.unique(group_labels)

    for group_id in unique_groups:
        group_mask = (group_labels == group_id)

        if np.sum(group_mask) == 0:
            continue

        group_true = y_true[group_mask]
        group_pred = y_pred[group_mask]

        group_name = group_names.get(group_id, f"Group_{group_id}") if group_names else f"Group_{group_id}"

        # Calculate metrics for this group
        accuracy = accuracy_score(group_true, group_pred)
        group_accuracies.append(accuracy)

        group_metrics[group_name] = {
            'accuracy': accuracy,
            'precision': precision_score(group_true, group_pred, zero_division=0),
            'recall': recall_score(group_true, group_pred, zero_division=0),
            'f1_score': f1_score(group_true, group_pred, zero_division=0),
            'support': len(group_true),
            'group_id': int(group_id)
        }

    # Calculate worst-group accuracy
    worst_group_accuracy = min(group_accuracies) if group_accuracies else 0.0

    # Summary statistics
    summary = {
        'worst_group_accuracy': worst_group_accuracy,
        'best_group_accuracy': max(group_accuracies) if group_accuracies else 0.0,
        'mean_group_accuracy': np.mean(group_accuracies) if group_accuracies else 0.0,
        'group_accuracy_std': np.std(group_accuracies) if group_accuracies else 0.0
    }

    return {
        'group_metrics': group_metrics,
        'summary': summary
    }


def calculate_hierarchical_metrics(
    hierarchical_predictions: torch.Tensor,
    binary_predictions: torch.Tensor,
    true_binary_labels: torch.Tensor,
    num_iterations: int
) -> Dict[str, Any]:
    """
    Calculate metrics specific to hierarchical classification.

    Args:
        hierarchical_predictions: Final node predictions from hierarchy
        binary_predictions: Binary predictions derived from hierarchy
        true_binary_labels: True binary labels
        num_iterations: Number of iterations in the hierarchy

    Returns:
        Dict[str, Any]: Hierarchical-specific metrics
    """
    # Convert to numpy
    if isinstance(hierarchical_predictions, torch.Tensor):
        hierarchical_predictions = hierarchical_predictions.cpu().numpy()
    if isinstance(binary_predictions, torch.Tensor):
        binary_predictions = binary_predictions.cpu().numpy()
    if isinstance(true_binary_labels, torch.Tensor):
        true_binary_labels = true_binary_labels.cpu().numpy()

    # Basic binary classification metrics
    binary_metrics = calculate_basic_metrics(true_binary_labels, binary_predictions)

    # Hierarchical node distribution
    final_num_nodes = 2 ** num_iterations
    node_distribution = {}

    for node_idx in range(final_num_nodes):
        node_count = np.sum(hierarchical_predictions == node_idx)
        node_distribution[f'node_{node_idx}'] = {
            'count': int(node_count),
            'percentage': float(node_count / len(hierarchical_predictions) * 100)
        }

    # Node utilization
    used_nodes = np.sum([dist['count'] > 0 for dist in node_distribution.values()])
    node_utilization = used_nodes / final_num_nodes

    return {
        'binary_metrics': binary_metrics,
        'node_distribution': node_distribution,
        'node_utilization': node_utilization,
        'total_nodes': final_num_nodes,
        'used_nodes': int(used_nodes)
    }


def print_evaluation_report(
    metrics: Dict[str, Any],
    title: str = "Evaluation Results"
) -> None:
    """Print a clean, professional evaluation report like scikit-learn."""

    # Basic metrics - clean, single line format
    if 'binary_metrics' in metrics:
        basic = metrics['binary_metrics']
        print(f"              precision    recall  f1-score   support")
        print(f"")
        print(f"           0     {basic.get('precision', 0):.2f}      {basic.get('recall', 0):.2f}      {basic.get('f1_score', 0):.2f}      -")
        print(f"           1     {basic.get('precision', 0):.2f}      {basic.get('recall', 0):.2f}      {basic.get('f1_score', 0):.2f}      -")
        print(f"")
        print(f"    accuracy                          {basic.get('accuracy', 0):.2f}      -")

    # Group-specific metrics - clean table format
    if 'group_metrics' in metrics:
        print(f"\nGroup Performance:")
        summary = metrics['group_metrics']['summary']
        print(f"  Worst-group accuracy: {summary['worst_group_accuracy']:.3f}")
        print(f"  Mean group accuracy:  {summary['mean_group_accuracy']:.3f}")

    # Hierarchical metrics - minimal
    if 'node_utilization' in metrics:
        print(f"  Node utilization:     {metrics['node_utilization']:.3f} ({metrics['used_nodes']}/{metrics['total_nodes']} nodes)")