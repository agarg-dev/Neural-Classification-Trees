"""Loss functions for hierarchical classification."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy loss with optional class-specific weights."""
    target_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()
    unreduced_loss = F.binary_cross_entropy_with_logits(
        outputs, target_one_hot, reduction="none"
    )

    if class_weights is not None:
        weights_for_batch = class_weights[targets].unsqueeze(1)
        weighted_loss = unreduced_loss * weights_for_batch
        return weighted_loss.mean()
    else:
        return unreduced_loss.mean()


def weighted_ce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy loss with optional per-sample class weights."""
    if class_weights is not None:
        unreduced_loss = F.cross_entropy(outputs, targets, reduction="none")
        weights_for_batch = class_weights[targets]
        weighted_loss = unreduced_loss * weights_for_batch
        return weighted_loss.mean()
    else:
        return F.cross_entropy(outputs, targets)


def weighted_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    num_classes: int = 2,
    loss_type: str = "bce",
) -> torch.Tensor:
    if loss_type == "ce":
        return weighted_ce_loss(outputs, targets, class_weights)
    else:
        return weighted_bce_loss(outputs, targets, class_weights)


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    """L1 regularization penalty on trainable parameters."""
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l1_penalty += torch.sum(torch.abs(param))
    return l1_penalty


def compute_l1_penalty_heads(model: nn.Module) -> torch.Tensor:
    """L1 regularization penalty on classifier layers of heads only."""
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    for head in model.output_heads:
        for param in head.classifier.parameters():
            if param.requires_grad:
                l1_penalty += torch.sum(torch.abs(param))

    if hasattr(model, "parent_layers"):
        for parent_layer in model.parent_layers:
            for head in parent_layer:
                for param in head.classifier.parameters():
                    if param.requires_grad:
                        l1_penalty += torch.sum(torch.abs(param))

    return l1_penalty


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    return_counts: bool = False,
    max_weight: Optional[float] = None,
):
    """Compute balanced class weights: total_samples / (num_classes * class_counts)."""
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_counts = class_counts.clamp(min=1)

    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)

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
    loss_type: str = "bce",
) -> torch.Tensor:
    """Auxiliary loss on the immediate parent layer (parent_label = label // 2)."""
    if len(aux_outputs) == 0 or iteration <= 1:
        return torch.tensor(0.0, device=labels.device)

    parent_outputs = aux_outputs[-1]
    parent_labels = labels // 2

    parent_class_weights = None
    if class_weights is not None:
        num_parent_nodes = num_classes * (2 ** (iteration - 2)) if iteration > 1 else num_classes
        parent_class_weights = compute_class_weights(
            parent_labels,
            num_parent_nodes,
            max_weight=max_weight,
        )
        parent_class_weights = parent_class_weights.to(class_weights.device)

    aux_loss = weighted_loss(parent_outputs, parent_labels, parent_class_weights, num_classes, loss_type)

    return aux_loss
