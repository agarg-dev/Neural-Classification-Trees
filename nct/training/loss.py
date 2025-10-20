"""Loss functions for hierarchical classification."""

import torch
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



def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute balanced class weights for imbalanced datasets.

    Uses the balanced weighting formula: total_samples / (num_classes * class_counts)
    This ensures minority classes receive higher weights proportional to their rarity.

    Args:
        labels: Training labels
        num_classes: Number of classes
    Returns:
        torch.Tensor: Balanced class weights
    """
    # Count occurrences of each class
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Avoid division by zero
    class_counts = class_counts.clamp(min=1)

    # Balanced class weights
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights
