"""
Utilities for hierarchical classification routing strategies.

This module provides functions to map hierarchical node IDs to binary classes
for different routing strategies:
- prediction_routing: Prediction-based tree routing
- correct_label_routing: Per-class hierarchy routing
"""

import torch
import numpy as np


def get_node_class(node_id: int, iteration: int) -> int:
    """
    Compute the predicted class for a hierarchical node using prediction-based routing.

    With prediction-based routing:
    - Even children inherit parent's predicted class (correct predictions)
    - Odd children get the opposite class (incorrect predictions)

    Args:
        node_id: Hierarchical node ID
        iteration: Current iteration number (1-indexed)

    Returns:
        Binary predicted class (0 or 1)
    """
    if iteration == 1:
        return node_id

    parent_id = node_id // 2
    parent_class = get_node_class(parent_id, iteration - 1)

    if node_id % 2 == 1:
        return 1 - parent_class
    else:
        return parent_class


def node_to_class(node_ids: torch.Tensor, iteration: int) -> torch.Tensor:
    """
    Convert hierarchical node IDs to binary class predictions using prediction-based routing.

    Args:
        node_ids: Tensor of hierarchical node IDs
        iteration: Current iteration number (1-indexed)

    Returns:
        Tensor of binary class predictions (0 or 1)
    """
    node_ids_np = node_ids.cpu().numpy()
    binary_classes = np.zeros_like(node_ids_np)

    for i, node_id in enumerate(node_ids_np.flat):
        binary_classes.flat[i] = get_node_class(int(node_id), iteration)

    return torch.from_numpy(binary_classes).to(node_ids.device)


# =============================================================================
# Correct Label Routing Functions
# =============================================================================


def get_node_class_correct_label(node_id: int, iteration: int) -> int:
    """
    Compute the predicted class for a node using correct label routing.

    With correct label routing:
    - First half of nodes always predict class 0 (regardless of easy/hard)
    - Second half of nodes always predict class 1 (regardless of easy/hard)
    - Each class maintains its own independent hierarchy

    Args:
        node_id: Hierarchical node ID
        iteration: Current iteration number (1-indexed)

    Returns:
        Binary predicted class (0 or 1)

    Examples:
        Iteration 1: [0] -> 0, [1] -> 1
        Iteration 2: [0,1] -> 0, [2,3] -> 1
        Iteration 3: [0,1,2,3] -> 0, [4,5,6,7] -> 1
    """
    num_nodes_per_class = 2 ** (iteration - 1)
    return 1 if node_id >= num_nodes_per_class else 0


def node_to_class_correct_label(node_ids: torch.Tensor, iteration: int) -> torch.Tensor:
    """
    Convert node IDs to binary class predictions using correct label routing.

    Args:
        node_ids: Tensor of hierarchical node IDs
        iteration: Current iteration number (1-indexed)

    Returns:
        Tensor of binary class predictions (0 or 1)
    """
    num_nodes_per_class = 2 ** (iteration - 1)
    return (node_ids >= num_nodes_per_class).long()
