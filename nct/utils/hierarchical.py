"""Helpers mapping hierarchical node IDs to class predictions.

Node layout: nodes are grouped by class. Class c owns node indices
[c * nodes_per_class, (c+1) * nodes_per_class), where nodes_per_class = 2^(iteration-1).
"""

import torch


def get_num_nodes(iteration: int, num_classes: int = 2) -> int:
    """Total number of nodes at this iteration: num_classes * 2^(iteration-1)."""
    return num_classes * (2 ** (iteration - 1))


def get_nodes_per_class(iteration: int) -> int:
    """Nodes per class at this iteration: 2^(iteration-1) (independent of num_classes)."""
    return 2 ** (iteration - 1)


def node_to_class(
    node_ids: torch.Tensor, iteration: int, num_classes: int = 2
) -> torch.Tensor:
    """Map a tensor of hierarchical node IDs to class IDs (0 .. num_classes-1)."""
    nodes_per_class = get_nodes_per_class(iteration)
    return (node_ids // nodes_per_class).long()


def get_class_node_range(class_id: int, iteration: int) -> tuple:
    """Half-open (start, end) range of node IDs belonging to class_id at this iteration."""
    nodes_per_class = get_nodes_per_class(iteration)
    start_node = class_id * nodes_per_class
    end_node = start_node + nodes_per_class
    return start_node, end_node
