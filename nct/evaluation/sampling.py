"""Sample collection utilities for evaluation and visualization."""

import numpy as np
from typing import Dict, List, Tuple, Optional


def collect_node_samples(
    predictions: np.ndarray,
    num_per_node: int,
    seed: Optional[int] = None
) -> Dict[int, List[int]]:
    """Collect random sample indices for each predicted node."""
    if seed is not None:
        np.random.seed(seed)

    node_samples = {}
    num_nodes = int(predictions.max()) + 1

    for node_idx in range(num_nodes):
        indices = np.where(predictions == node_idx)[0]
        if len(indices) > 0:
            selected = np.random.choice(
                indices,
                size=min(num_per_node, len(indices)),
                replace=False
            )
            node_samples[node_idx] = selected.tolist()
        else:
            node_samples[node_idx] = []

    return node_samples


def collect_transition_samples(
    prev_correct: np.ndarray,
    curr_correct: np.ndarray,
    num_samples: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (forgetting_indices, learning_indices) sample sets between iterations."""
    if seed is not None:
        np.random.seed(seed)

    # Forgetting: correct -> incorrect
    forgetting_mask = prev_correct & ~curr_correct
    forgetting_indices = np.where(forgetting_mask)[0]

    if len(forgetting_indices) > num_samples:
        forgetting_indices = np.random.choice(
            forgetting_indices,
            size=num_samples,
            replace=False
        )

    # Learning: incorrect -> correct
    learning_mask = ~prev_correct & curr_correct
    learning_indices = np.where(learning_mask)[0]

    if len(learning_indices) > num_samples:
        learning_indices = np.random.choice(
            learning_indices,
            size=num_samples,
            replace=False
        )

    return forgetting_indices, learning_indices


def stratified_sample(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """Collect stratified sample across classes."""
    if seed is not None:
        np.random.seed(seed)

    unique_labels = np.unique(labels)
    samples_per_class = num_samples // len(unique_labels)

    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) > 0:
            selected = np.random.choice(
                label_indices,
                size=min(samples_per_class, len(label_indices)),
                replace=False
            )
            selected_indices.extend(selected)

    return np.array(selected_indices)
