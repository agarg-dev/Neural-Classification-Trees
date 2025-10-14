"""Model utilities and helper functions."""

import torch
import torch.nn as nn
from typing import Optional
from .hierarchical import HierarchicalClassifier


def initialize_or_update_model(
    num_nodes: int,
    prev_model: Optional[nn.Module] = None,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    verbose: int = 0,
    **model_kwargs
) -> HierarchicalClassifier:
    """
    Initialize a new hierarchical model or transfer weights from a previous one.
    Args:
        num_nodes: Number of nodes/heads for the new model
        prev_model: Previous model to transfer weights from (optional)
        backbone: Backbone architecture name
        pretrained: Whether to use pretrained backbone weights
        device: Device to place the model on
        **model_kwargs: Additional arguments for HierarchicalClassifier
    Returns:
        HierarchicalClassifier: New model with optionally transferred weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create new model
    new_model = HierarchicalClassifier(
        num_heads=num_nodes,
        backbone=backbone,
        pretrained=pretrained,
        **model_kwargs
    ).to(device)

    if prev_model is not None:
        # Transfer the learned backbone weights (silently unless verbose >= 2)
        try:
            new_model.base.load_state_dict(prev_model.base.state_dict())
        except Exception as e:
            if verbose >= 1:
                print(f"Backbone transfer failed: {e}")

        # Transfer head weights to their new "easy" children
        for old_head_idx, old_head in enumerate(prev_model.output_heads):
            # Each old node splits into two new nodes (easy and difficult)
            new_head_idx = old_head_idx * 2  # The "easy" child gets the old weights

            if new_head_idx < len(new_model.output_heads):
                try:
                    new_model.output_heads[new_head_idx].load_state_dict(
                        old_head.state_dict()
                    )
                except Exception as e:
                    if verbose >= 1:
                        print(f"Head {old_head_idx} → {new_head_idx} failed: {e}")

    return new_model


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count the number of parameters in a model.
    Args:
        model: PyTorch model
        only_trainable: If True, count only trainable parameters
    Returns:
        int: Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print information about a model.
    Args:
        model: PyTorch model
        model_name: Name to display for the model
    """
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)

    print(f"\n--- {model_name} Information ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        for key, value in info.items():
            if key not in ['total_parameters', 'trainable_parameters']:
                print(f"{key.replace('_', ' ').title()}: {value}")

    print("---" + "-" * len(model_name) + "---" + "-" * 13)
