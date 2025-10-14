"""
Hierarchical Classifier Model Architecture.

This module defines the HierarchicalClassifier neural network that combines
a shared backbone feature extractor with multiple classification heads.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any


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
        backbone: str = 'resnet50',
        pretrained: bool = True,
        freeze_base: bool = False,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.5,
        num_classes_per_head: int = 1
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
            num_classes_per_head: Number of output classes per head
        """
        super().__init__()

        self.num_heads = num_heads
        self.backbone_name = backbone

        # Initialize backbone
        self.base = self._create_backbone(backbone, pretrained)
        num_features = self._get_backbone_features(self.base)

        # Remove original classifier
        if hasattr(self.base, 'fc'):
            self.base.fc = nn.Identity()
        elif hasattr(self.base, 'classifier'):
            self.base.classifier = nn.Identity()
        elif hasattr(self.base, 'head'):
            self.base.head = nn.Identity()

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # Create output heads
        self.output_heads = nn.ModuleList([
            self._create_head(
                num_features,
                head_hidden_dim,
                head_dropout,
                num_classes_per_head
            )
            for _ in range(num_heads)
        ])


    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create backbone network."""
        if backbone == 'resnet50':
            weights = 'IMAGENET1K_V1' if pretrained else None
            return models.resnet50(weights=weights)
        elif backbone == 'resnet18':
            weights = 'IMAGENET1K_V1' if pretrained else None
            return models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = 'IMAGENET1K_V1' if pretrained else None
            return models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _get_backbone_features(self, backbone: nn.Module) -> int:
        """Get number of features from backbone."""
        if hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'):
            return backbone.fc.in_features
        elif hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'):
            return backbone.classifier.in_features
        elif hasattr(backbone, 'head') and hasattr(backbone.head, 'in_features'):
            return backbone.head.in_features
        else:
            # Try to infer from a forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                try:
                    output = backbone(dummy_input)
                    return output.shape[1]
                except:
                    raise ValueError("Could not determine backbone feature size")


    def _create_head(self, in_features: int, hidden_dim: int, dropout: float, num_classes: int) -> nn.Module:
        """Create a single output head."""
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads)
        """
        features = self.base(x)
        outputs = [head(features) for head in self.output_heads]
        return torch.cat(outputs, dim=1)


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
        """
        base_params = [p for p in self.base.parameters() if p.requires_grad]
        head_params = [p for p in self.output_heads.parameters() if p.requires_grad]

        return {
            'base': base_params,
            'heads': head_params
        }


    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'backbone': self.backbone_name,
            'num_heads': self.num_heads,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'base_frozen': not any(p.requires_grad for p in self.base.parameters())
        }