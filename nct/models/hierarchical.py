"""
Hierarchical Classifier Model Architecture.

This module defines the HierarchicalClassifier neural network that combines
a shared backbone feature extractor with multiple classification heads.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST/UMNIST (grayscale 28x28 images).

    Based on GEORGE paper configuration for MNIST experiments.
    Architecture: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84)
    Output features: 84 dimensions
    """

    def __init__(self, num_channels: int = 1):
        super().__init__()
        self.num_channels = num_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully connected layers (feature extractor part)
        # After conv layers on 28x28: 16 * 5 * 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        # Placeholder for compatibility with ResNet interface
        self.fc = nn.Linear(84, 84)  # Will be replaced with Identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers (feature extraction)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class HierarchicalClassifier(nn.Module):
    """
    Neural network with shared backbone and multiple classification heads.

    This model architecture consists of:
        - A shared convolutional backbone (e.g., ResNet) for feature extraction
        - Multiple independent classification heads, one per hierarchical node

    The backbone can be frozen or trained, and heads can have different architectures.
    """

    def __init__(
        self,
        num_heads: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_base: bool = False,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.5,
        head_type: str = "mlp",
        num_classes_per_head: int = 1,
        iteration: int = 1,
        parent_layers: Optional[list] = None,
        hierarchical: str = "none",
        num_channels: int = 3,
    ):
        """
        Initialize the hierarchical classifier.
        Args:
            num_heads: Number of output heads (nodes in current iteration)
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            freeze_base: Whether to freeze backbone parameters
            head_hidden_dim: Hidden dimension for output heads
            head_dropout: Dropout rate for output heads
            head_type: Head architecture type ("mlp" or "linear")
            num_classes_per_head: Number of output classes per head
            iteration: Current iteration number (1-indexed)
            parent_layers: List of parent layers from previous iterations (for tree mode)
            hierarchical: Hierarchical architecture mode ("none" or "correct_label_routing")
        """
        super().__init__()

        self.num_heads = num_heads
        self.backbone_name = backbone
        self.iteration = iteration
        self.hierarchical = hierarchical
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.head_type = head_type
        self.num_channels = num_channels

        # Initialize backbone
        self.base = self._create_backbone(backbone, pretrained, num_channels)
        num_features = self._get_backbone_features(self.base)
        self.num_features = num_features

        # Remove original classifier
        # ResNet has .fc, LeNet5 already outputs features directly
        if hasattr(self.base, "fc") and not isinstance(self.base, LeNet5):
            self.base.fc = nn.Identity()

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # Store parent layers (empty list for simple mode, ModuleList for hierarchical)
        self.parent_layers = (
            nn.ModuleList(parent_layers) if parent_layers else nn.ModuleList()
        )

        # Create output heads based on mode
        if hierarchical != "none" and iteration > 1:
            # Hierarchical tree structure (correct_label_routing)
            # Iteration 2+: heads connect to parent layer output
            self.output_heads = nn.ModuleList(
                [
                    self._create_head(
                        head_hidden_dim,
                        head_hidden_dim,
                        head_dropout,
                        num_classes_per_head,
                    )
                    for _ in range(num_heads)
                ]
            )
        else:
            # Non-hierarchical mode (none) OR iteration 1: heads connect to backbone
            self.output_heads = nn.ModuleList(
                [
                    self._create_head(
                        num_features,
                        head_hidden_dim,
                        head_dropout,
                        num_classes_per_head,
                    )
                    for _ in range(num_heads)
                ]
            )

    def _create_backbone(
        self, backbone: str, pretrained: bool, num_channels: int = 3
    ) -> nn.Module:
        """Create backbone network."""
        if backbone == "resnet50":
            weights = "IMAGENET1K_V1" if pretrained else None
            return models.resnet50(weights=weights)
        elif backbone == "resnet18":
            weights = "IMAGENET1K_V1" if pretrained else None
            return models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = "IMAGENET1K_V1" if pretrained else None
            return models.resnet34(weights=weights)
        elif backbone == "lenet5":
            # LeNet-5 for grayscale MNIST-like datasets (no pretrained weights)
            return LeNet5(num_channels=num_channels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _get_backbone_features(self, backbone: nn.Module) -> int:
        """Get number of features from backbone."""
        if isinstance(backbone, LeNet5):
            return 84  # LeNet-5 outputs 84-dimensional features
        return backbone.fc.in_features

    @staticmethod
    def get_head_indices(head_type: str) -> dict:
        """
        Get layer indices for different head types.

        Args:
            head_type: "mlp" or "linear"

        Returns:
            dict with 'feature_end' (slice end for feature layers) and 'classifier' (classifier layer index)
            - feature_end: used as [:feature_end] to get feature layers
            - classifier: used as [classifier] to get the classification layer
        """
        if head_type == "mlp":
            # MLP: Linear[0] → ReLU[1] → Dropout[2] → Linear[3]
            # Feature layers: [:3] → layers 0,1,2; Classifier: [3] → layer 3
            return {"feature_end": 3, "classifier": 3}
        else:  # linear
            # Linear: Linear[0] → Linear[1]
            # Feature layers: [:1] → layer 0; Classifier: [1] → layer 1
            return {"feature_end": 1, "classifier": 1}

    def _create_head(
        self, in_features: int, hidden_dim: int, dropout: float, num_classes: int
    ) -> nn.Module:
        """Create a single output head based on head_type."""
        if self.head_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:  # linear
            return nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor, return_aux_outputs: bool = False):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            return_aux_outputs: If True, return (main_output, aux_outputs) for auxiliary loss

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads)
            OR tuple (main_output, aux_outputs) if return_aux_outputs=True
        """
        features = self.base(x)
        aux_outputs = []  # Store auxiliary outputs from parent layers

        if self.hierarchical != "none" and self.iteration > 1:
            # Hierarchical tree structure: route through parent layers
            # Tree structure: route through parent layers
            current_features = [features] * len(self.parent_layers[0])

            # Get indices based on head_type for parent layer access
            indices = self.get_head_indices(self.head_type)

            for parent_layer in self.parent_layers:
                next_features = []
                layer_aux_outputs = []

                for parent_head in parent_layer:
                    parent_input = current_features[len(next_features) // 2]

                    # Compute features for children (first layers up to feature_end)
                    parent_features = parent_head[:indices["feature_end"]](parent_input)

                    # Compute logits for auxiliary loss if requested
                    # Check if classifier layer exists
                    if return_aux_outputs and len(parent_head) > indices["classifier"]:
                        parent_logits = parent_head[indices["classifier"]](parent_features)
                        layer_aux_outputs.append(parent_logits)

                    next_features.append(parent_features)
                    next_features.append(parent_features)

                current_features = next_features

                # Collect auxiliary outputs from this parent layer
                if return_aux_outputs and len(layer_aux_outputs) > 0:
                    aux_outputs.append(torch.cat(layer_aux_outputs, dim=1))

            outputs = [
                head(current_features[i]) for i, head in enumerate(self.output_heads)
            ]
        else:
            # Non-hierarchical mode (none) OR iteration 1: direct connection
            outputs = [head(features) for head in self.output_heads]

        main_output = torch.cat(outputs, dim=1)

        if return_aux_outputs:
            return main_output, aux_outputs
        return main_output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        Args:
            x: Input tensor
        Returns:
            torch.Tensor: Feature tensor
        """
        return self.base(x)

    def freeze_base(self):
        """Freeze backbone parameters."""
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze backbone parameters."""
        for param in self.base.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> Dict[str, list]:
        """
        Get trainable parameters grouped by component.

        Returns:
            Dict[str, list]: Dictionary with 'base' and 'heads' parameter lists
                - 'base': backbone + parent_layers (frozen in Phase 1, base_lr in Phase 2)
                - 'heads': output_heads only (trained in both phases with head_lr)
        """
        # Base includes backbone and parent layers (both use base_lr in Phase 2)
        base_params = [p for p in self.base.parameters() if p.requires_grad]
        if self.hierarchical != "none" and len(self.parent_layers) > 0:
            for parent_layer in self.parent_layers:
                base_params.extend(
                    p for p in parent_layer.parameters() if p.requires_grad
                )

        # Heads are only the current iteration's output heads
        head_params = [p for p in self.output_heads.parameters() if p.requires_grad]

        return {"base": base_params, "heads": head_params}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            "backbone": self.backbone_name,
            "num_heads": self.num_heads,
            "hierarchical_mode": self.hierarchical,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "base_frozen": not any(p.requires_grad for p in self.base.parameters()),
        }

        if self.hierarchical != "none":
            # Both correct_label_routing use tree structure
            info["iteration"] = self.iteration
            info["num_parent_layers"] = len(self.parent_layers)

        return info
