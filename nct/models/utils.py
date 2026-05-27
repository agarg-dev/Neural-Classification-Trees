"""Model utilities and helper functions."""

import copy
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
    iteration: int = 1,
    hierarchical: str = "none",
    num_channels: int = 3,
    **model_kwargs
) -> HierarchicalClassifier:
    """Build a HierarchicalClassifier with num_nodes heads, optionally transferring from prev_model.

    In hierarchical mode, prev_model's output heads become the new model's parent
    layer (children are initialised fresh). In flat mode, each old head is copied
    into both of its child heads.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_tree_structure = (hierarchical != "none")

    parent_layers = None
    if prev_model is not None and use_tree_structure and iteration > 1:
        parent_layers = [layer for layer in prev_model.parent_layers]

        # Old output heads become the new parent layer. All layers are kept so
        # the auxiliary loss can score the parent's classification.
        new_parent_layer = nn.ModuleList()
        for old_head in prev_model.output_heads:
            new_parent_layer.append(copy.deepcopy(old_head))
        parent_layers.append(new_parent_layer)

    new_model = HierarchicalClassifier(
        num_heads=num_nodes,
        backbone=backbone,
        pretrained=pretrained,
        iteration=iteration,
        parent_layers=parent_layers,
        hierarchical=hierarchical,
        num_channels=num_channels,
        **model_kwargs
    ).to(device)

    if prev_model is not None:
        try:
            new_model.base.load_state_dict(prev_model.base.state_dict())
            if verbose >= 2:
                print("Transferred backbone weights")
        except Exception as e:
            if verbose >= 1:
                print(f"Backbone transfer failed: {e}")

        # Flat mode: copy each parent head's weights into both its children.
        # Tree mode has already moved the heads into parent_layers above.
        if not use_tree_structure:
            for old_head_idx, old_head in enumerate(prev_model.output_heads):
                easy_idx = old_head_idx * 2
                hard_idx = old_head_idx * 2 + 1

                for new_head_idx in [easy_idx, hard_idx]:
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
    """Count parameters; if only_trainable, restricts to requires_grad=True."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """Print parameter counts and model-specific info for `model`."""
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
