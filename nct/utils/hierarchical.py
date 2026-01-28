"""
Utilities for hierarchical classification.

This module provides functions to map hierarchical node IDs to class predictions.
Supports both binary and multi-class classification.

Node organization:
- Nodes are grouped by class, with each class having 2^(iteration-1) nodes
- Nodes 0 to (nodes_per_class - 1): Class 0
- Nodes nodes_per_class to (2*nodes_per_class - 1): Class 1
- etc.
"""

import torch


def get_num_nodes(iteration: int, num_classes: int = 2) -> int:
    """
    Calculate total number of nodes for a given iteration.

    Args:
        iteration: Iteration number (1-indexed)
        num_classes: Number of base classes (default 2 for binary)

    Returns:
        Number of nodes: num_classes * 2^(iteration-1)

    Examples:
        Binary (num_classes=2): iter 1 -> 2, iter 2 -> 4, iter 3 -> 8
        5-class (num_classes=5): iter 1 -> 5, iter 2 -> 10, iter 3 -> 20
    """
    return num_classes * (2 ** (iteration - 1))


def get_nodes_per_class(iteration: int) -> int:
    """
    Get number of nodes per class at a given iteration.

    Each class gets 2^(iteration-1) nodes, independent of num_classes.

    Args:
        iteration: Iteration number (1-indexed)

    Returns:
        Number of nodes per class: 2^(iteration-1)
    """
    return 2 ** (iteration - 1)


def node_to_class(
    node_ids: torch.Tensor, iteration: int, num_classes: int = 2
) -> torch.Tensor:
    """
    Convert node IDs to class predictions.

    Works for both binary and multi-class classification.

    Args:
        node_ids: Tensor of hierarchical node IDs
        iteration: Current iteration number (1-indexed)
        num_classes: Number of base classes (default 2 for binary)

    Returns:
        Tensor of class predictions (0 to num_classes-1)
    """
    nodes_per_class = get_nodes_per_class(iteration)
    return (node_ids // nodes_per_class).long()


def get_class_node_range(class_id: int, iteration: int) -> tuple:
    """
    Get the range of node IDs belonging to a specific class.

    Args:
        class_id: Class index (0 to num_classes-1)
        iteration: Current iteration number (1-indexed)

    Returns:
        Tuple of (start_node, end_node) - end is exclusive
    """
    nodes_per_class = get_nodes_per_class(iteration)
    start_node = class_id * nodes_per_class
    end_node = start_node + nodes_per_class
    return start_node, end_node
