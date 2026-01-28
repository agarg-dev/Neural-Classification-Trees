"""Loss functions for hierarchical classification."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nct.models.hierarchical import HierarchicalClassifier


def weighted_bce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
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
        outputs, target_one_hot, reduction="none"
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


def compute_l1_penalty_heads(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 regularization penalty on classification layers only.

    Applies L1 to the final Linear layer of each head, not the feature
    transformation layers which are reused by child heads.

    Args:
        model: HierarchicalClassifier model
    Returns:
        torch.Tensor: L1 penalty on classifier weights only
    """
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    # Get classifier index based on head_type (mlp=3, linear=1)
    head_type = getattr(model, "head_type", "mlp")  # Default to mlp for backwards compatibility
    indices = HierarchicalClassifier.get_head_indices(head_type)
    classifier_idx = indices["classifier"]

    # Output heads: classifier layer
    for head in model.output_heads:
        if len(head) > classifier_idx:
            for param in head[classifier_idx].parameters():
                if param.requires_grad:
                    l1_penalty += torch.sum(torch.abs(param))

    # Parent heads: classifier layer
    if hasattr(model, "parent_layers"):
        for parent_layer in model.parent_layers:
            for head in parent_layer:
                if len(head) > classifier_idx:
                    for param in head[classifier_idx].parameters():
                        if param.requires_grad:
                            l1_penalty += torch.sum(torch.abs(param))

    return l1_penalty


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    return_counts: bool = False,
    max_weight: Optional[float] = None,
):
    """
    Compute balanced class weights for imbalanced datasets.

    Uses the balanced weighting formula: total_samples / (num_classes * class_counts)
    This ensures minority classes receive higher weights proportional to their rarity.

    Args:
        labels: Training labels
        num_classes: Number of classes
        return_counts: If True, return (weights, counts) tuple
        max_weight: Optional maximum weight cap to prevent extreme imbalance effects
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

    # Apply weight clamping if specified
    if max_weight is not None:
        class_weights = class_weights.clamp(max=max_weight)

    if return_counts:
        return class_weights, class_counts.int()
    return class_weights


def compute_auxiliary_loss(
    aux_outputs: list,
    labels: torch.Tensor,
    iteration: int,
    class_weights: Optional[torch.Tensor] = None,
    max_weight: Optional[float] = None,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Compute auxiliary loss for parent layer classifications.

    Uses only the immediate parent layer (most recent) to avoid over-constraining.
    Parent labels are derived from current labels using: parent_label = label // 2

    Args:
        aux_outputs: List of auxiliary outputs from parent layers (indexed by layer depth)
        labels: Current iteration's labels (range: 0 to num_nodes - 1)
        iteration: Current iteration number (1-indexed)
        class_weights: Optional class weights for current iteration
        num_classes: Number of base classes (default 2 for binary)

    Returns:
        Auxiliary loss from immediate parent layer (0.0 if no parent exists)
    """
    if len(aux_outputs) == 0 or iteration <= 1:
        return torch.tensor(0.0, device=labels.device)

    # Use only the most recent parent layer (immediate parent)
    parent_outputs = aux_outputs[-1]  # Last layer = most recent parent

    # Derive parent labels from current labels
    # Parent label = label // 2 (easy/hard split is always binary)
    parent_labels = labels // 2

    # Compute parent class weights if main weights provided
    parent_class_weights = None
    if class_weights is not None:
        # Parent has num_classes * 2^(iteration-2) nodes = get_num_nodes(iteration-1, num_classes)
        num_parent_nodes = num_classes * (2 ** (iteration - 2)) if iteration > 1 else num_classes
        parent_class_weights = compute_class_weights(
            parent_labels,
            num_parent_nodes,
            max_weight=max_weight,
        )
        parent_class_weights = parent_class_weights.to(class_weights.device)

    # Compute loss for parent's classification task
    aux_loss = weighted_bce_loss(parent_outputs, parent_labels, parent_class_weights)

    return aux_loss
