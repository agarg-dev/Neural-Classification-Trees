"""HierarchicalClassifier: shared backbone + one classification head per hierarchical node."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LeNet4(nn.Module):
    """
    LeNet-4 architecture for U-MNIST.

    Architecture: Conv(6) -> ReLU -> Pool -> Conv(16) -> ReLU -> Pool -> Conv(120) -> ReLU
    Requires 32x32 input. Output features: 120 dimensions.
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
        # Placeholder for compatibility with ResNet interface
        self.fc = nn.Linear(120, 120)  # Will be replaced with Identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnet(x)
        return x.view(x.size(0), -1)


class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST/UMNIST (grayscale 28x28 images).

    Architecture: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84)
    Output features: 84 dimensions
    """

    def __init__(self, num_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.num_channels = num_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=6, kernel_size=5
        )
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Compute flatten size dynamically based on image_size
        # 28x28 → 16*4*4=256; 32x32 → 16*5*5=400
        conv_out = ((image_size - 4) // 2 - 4) // 2
        self._flat_size = 16 * conv_out * conv_out

        # Fully connected layers (feature extractor part)
        self.fc1 = nn.Linear(self._flat_size, 120)
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


class HeadModule(nn.Module):
    """
    A classification head with explicit feature_extractor and classifier components.

    Separates the head into two named parts so the forward pass and L1 penalty
    can access them by name rather than by fragile positional indices.
    """

    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


class HierarchicalClassifier(nn.Module):
    """Shared backbone + one classification head per hierarchical node.

    In hierarchical mode (iter >= 2) heads connect to a parent layer's features
    instead of the backbone directly, building a tree across iterations.
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
        # Effective head type for this iteration (used when split_node_head_type is False)
        self.effective_head_type = (iter1_head_type if iteration == 1 and iter1_head_type is not None else head_type)

        # Initialize backbone
        self.base = self._create_backbone(backbone, pretrained, num_channels, image_size)
        num_features = self._get_backbone_features(self.base)
        self.num_features = num_features

        # Remove original classifier
        # ResNet has .fc, LeNet5/LeNet4 already output features directly
        if hasattr(self.base, "fc") and not isinstance(self.base, (LeNet5, LeNet4)):
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
                        head_idx=i,
                    )
                    for i in range(num_heads)
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
            # LeNet-5 for grayscale MNIST-like datasets (no pretrained weights)
            return LeNet5(num_channels=num_channels, image_size=image_size)
        elif backbone == "lenet4":
            # LeNet-4 for U-MNIST (requires 32x32 input)
            return LeNet4(num_channels=num_channels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _get_backbone_features(self, backbone: nn.Module) -> int:
        """Get number of features from backbone."""
        if isinstance(backbone, LeNet5):
            return 84  # LeNet-5 outputs 84-dimensional features
        if isinstance(backbone, LeNet4):
            return 120  # LeNet-4 outputs 120-dimensional features
        return backbone.fc.in_features

    def _create_head(
        self, in_features: int, hidden_dim: int, dropout: float, num_classes: int, head_idx: int = 0
    ) -> HeadModule:
        """Create a single output head based on head_type."""
        if self.split_node_head_type and self.iteration > 1:
            # Even index = easy node, odd index = difficult node.
            # split_node_head_type can be True (easy=linear, hard=mlp) or
            # the string "reversed" (easy=mlp, hard=linear).
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
        else:  # linear
            feature_extractor = nn.Sequential(
                norm1,
                nn.Linear(in_features, hidden_dim),
            )
        
        classifier = nn.Sequential(
                norm2,
                nn.Linear(hidden_dim, num_classes)
        )

        # classifier = nn.Linear(hidden_dim, num_classes)
        return HeadModule(feature_extractor, classifier)

    def forward(self, x: torch.Tensor, return_aux_outputs: bool = False):
        """Forward pass. Returns (batch, num_heads), or (output, aux_outputs) if return_aux_outputs."""
        features = self.base(x)
        aux_outputs = []  # per-parent-layer logits used by the auxiliary loss

        if self.hierarchical != "none" and self.iteration > 1:
            # Tree mode: route through parent layers, branching at each level.
            current_features = [features] * len(self.parent_layers[0])

            for parent_layer in self.parent_layers:
                next_features = []
                layer_aux_outputs = []

                for parent_head in parent_layer:
                    parent_input = current_features[len(next_features) // 2]

                    # Children consume the parent's feature_extractor output.
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
            # Iteration 1 or hierarchical="none": heads consume backbone features directly.
            outputs = [head(features) for head in self.output_heads]

        main_output = torch.cat(outputs, dim=1)

        if return_aux_outputs:
            return main_output, aux_outputs
        return main_output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features for x."""
        return self.base(x)

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> Dict[str, list]:
        """Trainable params grouped as 'base' (backbone + parent_layers) and 'heads'.

        'base' params use base_lr in Phase 2; 'heads' always use head_lr.
        """
        base_params = [p for p in self.base.parameters() if p.requires_grad]
        if self.hierarchical != "none" and len(self.parent_layers) > 0:
            for parent_layer in self.parent_layers:
                base_params.extend(
                    p for p in parent_layer.parameters() if p.requires_grad
                )

        head_params = [p for p in self.output_heads.parameters() if p.requires_grad]

        return {"base": base_params, "heads": head_params}

    def get_model_info(self) -> Dict[str, Any]:
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
