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


class LeNet4(nn.Module):
    """
    LeNet-4 backbone for U-MNIST (32x32 input). Output features: 120 dimensions.

    Architecture: Conv(6) -> ReLU -> Pool -> Conv(16) -> ReLU -> Pool -> Conv(120) -> ReLU.
    """

    def __init__(self, num_channels: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.convnet = nn.Sequential(
            nn.Conv2d(num_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        self.fc = nn.Linear(120, 120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnet(x)
        return x.view(x.size(0), -1)


class LeNet5(nn.Module):
    """LeNet-5 backbone for MNIST-scale grayscale inputs (28x28 or 32x32).

    Architecture: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84).
    Output features: 84 dimensions.
    """

    def __init__(self, num_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=6, kernel_size=5
        )
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # 28x28 -> 16*4*4=256; 32x32 -> 16*5*5=400
        conv_out = ((image_size - 4) // 2 - 4) // 2
        self._flat_size = 16 * conv_out * conv_out

        self.fc1 = nn.Linear(self._flat_size, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc = nn.Linear(84, 84)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class HeadModule(nn.Module):
    """A classification head with explicit feature_extractor and classifier components."""

    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


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
        iter1_head_type: Optional[str] = None,
        split_node_head_type: bool = False,
        head_bn_affine: bool = False,
        num_classes_per_head: int = 1,
        iteration: int = 1,
        parent_layers: Optional[list] = None,
        hierarchical: str = "none",
        num_channels: int = 3,
        image_size: int = 224,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.backbone_name = backbone
        self.iteration = iteration
        self.hierarchical = hierarchical
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.head_type = head_type
        self.num_channels = num_channels
        self.image_size = image_size
        self.split_node_head_type = split_node_head_type
        self.head_bn_affine = head_bn_affine
        self.effective_head_type = (iter1_head_type if iteration == 1 and iter1_head_type is not None else head_type)

        self.base = self._create_backbone(backbone, pretrained, num_channels, image_size)
        num_features = self._get_backbone_features(self.base)
        self.num_features = num_features

        # ResNet has .fc; LeNet variants already output features directly
        if hasattr(self.base, "fc") and not isinstance(self.base, (LeNet5, LeNet4)):
            self.base.fc = nn.Identity()

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.parent_layers = (
            nn.ModuleList(parent_layers) if parent_layers else nn.ModuleList()
        )

        if hierarchical != "none" and iteration > 1:
            # Iteration 2+: heads connect to parent layer output
            self.output_heads = nn.ModuleList(
                [
                    self._create_head(
                        head_hidden_dim,
                        head_hidden_dim,
                        head_dropout,
                        num_classes_per_head,
                        head_idx=i,
                    )
                    for i in range(num_heads)
                ]
            )
        else:
            # Iteration 1 or non-hierarchical: heads connect to backbone
            self.output_heads = nn.ModuleList(
                [
                    self._create_head(
                        num_features,
                        head_hidden_dim,
                        head_dropout,
                        num_classes_per_head,
                        head_idx=i,
                    )
                    for i in range(num_heads)
                ]
            )

    def _create_backbone(
        self, backbone: str, pretrained: bool, num_channels: int = 3, image_size: int = 224
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
            return LeNet5(num_channels=num_channels, image_size=image_size)
        elif backbone == "lenet4":
            return LeNet4(num_channels=num_channels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _get_backbone_features(self, backbone: nn.Module) -> int:
        """Get number of features from backbone."""
        if isinstance(backbone, LeNet5):
            return 84
        if isinstance(backbone, LeNet4):
            return 120
        return backbone.fc.in_features

    def _create_head(
        self, in_features: int, hidden_dim: int, dropout: float, num_classes: int, head_idx: int = 0
    ) -> HeadModule:
        """Create a single output head based on head_type."""
        if self.split_node_head_type and self.iteration > 1:
            # Even idx = easy node, odd idx = difficult node.
            # split_node_head_type=True: easy=linear, hard=mlp; "reversed" swaps them.
            if self.split_node_head_type == "reversed":
                head_type = "mlp" if head_idx % 2 == 0 else "linear"
            else:
                head_type = "linear" if head_idx % 2 == 0 else "mlp"
        else:
            head_type = self.effective_head_type

        norm1 = nn.Identity() if self.iteration == 1 else nn.BatchNorm1d(in_features, affine=self.head_bn_affine)
        norm2 = nn.Identity() if self.iteration == 1 else nn.BatchNorm1d(hidden_dim, affine=self.head_bn_affine)
        if head_type == "mlp":
            feature_extractor = nn.Sequential(
                norm1,
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            feature_extractor = nn.Sequential(
                norm1,
                nn.Linear(in_features, hidden_dim),
            )

        classifier = nn.Sequential(
                norm2,
                nn.Linear(hidden_dim, num_classes)
        )

        return HeadModule(feature_extractor, classifier)

    def forward(self, x: torch.Tensor, return_aux_outputs: bool = False):
        """Forward pass. If return_aux_outputs, also returns parent-layer logits."""
        features = self.base(x)
        aux_outputs = []

        if self.hierarchical != "none" and self.iteration > 1:
            current_features = [features] * len(self.parent_layers[0])

            for parent_layer in self.parent_layers:
                next_features = []
                layer_aux_outputs = []

                for parent_head in parent_layer:
                    parent_input = current_features[len(next_features) // 2]

                    parent_features = parent_head.feature_extractor(parent_input)

                    if return_aux_outputs:
                        parent_logits = parent_head.classifier(parent_features)
                        layer_aux_outputs.append(parent_logits)

                    next_features.append(parent_features)
                    next_features.append(parent_features)

                current_features = next_features

                if return_aux_outputs and len(layer_aux_outputs) > 0:
                    aux_outputs.append(torch.cat(layer_aux_outputs, dim=1))

            outputs = [
                head(current_features[i]) for i, head in enumerate(self.output_heads)
            ]
        else:
            outputs = [head(features) for head in self.output_heads]

        main_output = torch.cat(outputs, dim=1)

        if return_aux_outputs:
            return main_output, aux_outputs
        return main_output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone."""
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
        """Return {'base': backbone+parent_layers, 'heads': output_heads}."""
        base_params = [p for p in self.base.parameters() if p.requires_grad]
        if self.hierarchical != "none" and len(self.parent_layers) > 0:
            for parent_layer in self.parent_layers:
                base_params.extend(
                    p for p in parent_layer.parameters() if p.requires_grad
                )

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
            info["iteration"] = self.iteration
            info["num_parent_layers"] = len(self.parent_layers)

        return info
