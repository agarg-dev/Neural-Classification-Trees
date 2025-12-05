"""Loss functions for hierarchical classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def weighted_bce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculates binary cross-entropy loss with optional class-specific weights.
    Args:
        outputs: Model outputs of shape (batch_size, num_classes)
        targets: Target labels of shape (batch_size,)
        class_weights: Optional class weights of shape (num_classes,)
    Returns:
        torch.Tensor: Weighted BCE loss
    """
    # Convert targets to one-hot encoding
    target_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()

    # Calculate unreduced BCE loss
    unreduced_loss = F.binary_cross_entropy_with_logits(
        outputs, target_one_hot, reduction='none'
    )

    if class_weights is not None:
        # Apply weights corresponding to each sample's target class
        weights_for_batch = class_weights[targets].unsqueeze(1)
        weighted_loss = unreduced_loss * weights_for_batch
        return weighted_loss.mean()
    else:
        return unreduced_loss.mean()



def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 regularization penalty on trainable model parameters only.

    Args:
        model: Neural network model
    Returns:
        torch.Tensor: L1 penalty (sum of absolute values of trainable parameters)
    """
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:  # Only penalize trainable parameters
            l1_penalty += torch.sum(torch.abs(param))
    return l1_penalty


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    return_counts: bool = False
):
    """
    Compute balanced class weights for imbalanced datasets.

    Uses the balanced weighting formula: total_samples / (num_classes * class_counts)
    This ensures minority classes receive higher weights proportional to their rarity.

    Args:
        labels: Training labels
        num_classes: Number of classes
        return_counts: If True, return (weights, counts) tuple
    Returns:
        torch.Tensor or tuple: Balanced class weights, or (weights, counts) if return_counts=True
    """
    # Count occurrences of each class
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Avoid division by zero
    class_counts = class_counts.clamp(min=1)

    # Balanced class weights
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)

    if return_counts:
        return class_weights, class_counts.int()
    return class_weights


def compute_auxiliary_loss(
    aux_outputs: list,
    labels: torch.Tensor,
    iteration: int,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute auxiliary loss for parent layer classifications.

    Uses only the immediate parent layer (most recent) to avoid over-constraining.
    Parent labels are derived from current labels using: parent_label = label // 2

    Args:
        aux_outputs: List of auxiliary outputs from parent layers (indexed by layer depth)
        labels: Current iteration's labels (range: 0 to 2^iteration - 1)
        iteration: Current iteration number (1-indexed)
        class_weights: Optional class weights for current iteration

    Returns:
        Auxiliary loss from immediate parent layer (0.0 if no parent exists)
    """
    if len(aux_outputs) == 0 or iteration <= 1:
        return torch.tensor(0.0, device=labels.device)

    # Use only the most recent parent layer (immediate parent)
    parent_outputs = aux_outputs[-1]  # Last layer = most recent parent

    # Derive parent labels from current labels
    # For current label in range [0, 2^iteration), parent label is in [0, 2^(iteration-1))
    parent_labels = labels // 2

    # Compute parent class weights if main weights provided
    parent_class_weights = None
    if class_weights is not None:
        num_parent_classes = 2 ** (iteration - 1)
        parent_class_weights = compute_class_weights(parent_labels, num_parent_classes)
        parent_class_weights = parent_class_weights.to(class_weights.device)

    # Compute loss for parent's classification task
    aux_loss = weighted_bce_loss(parent_outputs, parent_labels, parent_class_weights)

    return aux_loss
