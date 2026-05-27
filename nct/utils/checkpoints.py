"""Checkpoint management utilities for NCT models."""

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..models.hierarchical import HierarchicalClassifier, HeadModule


class CheckpointManager:
    """Manages saving and loading of NCT model checkpoints."""

    @staticmethod
    def save_iteration_checkpoint(
        model: nn.Module, iteration: int, config: Any, checkpoint_dir: str
    ) -> str:
        """Save a per-iteration checkpoint; returns the saved path."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_iter_{iteration}.pt"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_heads": len(model.output_heads),
            "iteration": iteration,
            "hierarchical": config.hierarchical_architecture,
            "backbone": config.backbone,
            "num_features": model.num_features,
            "head_hidden_dim": config.head_hidden_dim,
            "head_dropout": config.head_dropout,
            "head_type": config.head_type,
            "head_bn_affine": config.head_bn_affine,
            "iter1_head_type": config.iter1_head_type,
            "split_node_head_type": config.split_node_head_type,
            "num_channels": config.num_channels,
            "image_size": config.image_size,
            "num_classes": config.num_classes,
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def save_final_checkpoint(
        model: nn.Module,
        config: Any,
        iteration_results: list,
        filepath: str,
        verbose: int = 0,
        trained_nodes_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Save the final model + full config + per-iteration results to one file."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "num_heads": len(model.output_heads),
            "iteration_results": iteration_results,
            "iteration": config.num_iterations,
            "hierarchical": config.hierarchical_architecture,
            "backbone": config.backbone,
            "num_features": model.num_features,
            "head_hidden_dim": config.head_hidden_dim,
            "head_dropout": config.head_dropout,
            "head_type": config.head_type,
            "head_bn_affine": config.head_bn_affine,
            "iter1_head_type": config.iter1_head_type,
            "split_node_head_type": config.split_node_head_type,
            "num_channels": config.num_channels,
            "num_classes": config.num_classes,
        }

        if trained_nodes_mask is not None:
            checkpoint["trained_nodes_mask"] = trained_nodes_mask

        torch.save(checkpoint, filepath)

        if verbose:
            print(f"Checkpoint saved to {filepath}")

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: str, config: Any, device: torch.device
    ) -> nn.Module:
        """Load just the model (in eval mode) from an iteration checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = CheckpointManager._reconstruct_model(checkpoint, config, device)
        model.eval()
        return model

    @staticmethod
    def load_classifier_from_checkpoint(
        filepath: str,
        classifier_class: type,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        """Reconstruct a fitted NCTClassifier from a final checkpoint.

        classifier_class is passed in to avoid a circular import.
        """
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        if "config" not in checkpoint or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Invalid checkpoint: missing required keys ('config', 'model_state_dict')"
            )

        # Override device if specified
        if device is not None:
            checkpoint["config"]["device"] = device

        # Create classifier with saved config
        classifier = classifier_class(**checkpoint["config"])

        # Reconstruct and load model
        classifier.model = CheckpointManager._reconstruct_model(
            checkpoint, classifier.config, classifier.device
        )
        classifier._is_fitted = True
        classifier._iteration_results = checkpoint.get("iteration_results", [])

        # Restore trained nodes mask if present (for min_samples_per_node feature)
        if "trained_nodes_mask" in checkpoint:
            classifier._trained_nodes_mask = checkpoint["trained_nodes_mask"]

        if classifier.config.verbose:
            print(f"Checkpoint loaded from {filepath}")

        return classifier

    @staticmethod
    def _reconstruct_model(
        checkpoint: Dict, config: Any, device: torch.device
    ) -> nn.Module:
        """Rebuild a HierarchicalClassifier from a checkpoint dict; config is the fallback for missing keys."""
        if "num_heads" not in checkpoint:
            raise RuntimeError("Invalid checkpoint: missing 'num_heads'")

        # Extract architecture parameters (with fallbacks to config)
        hierarchical = checkpoint.get("hierarchical", config.hierarchical_architecture)
        iteration = checkpoint.get("iteration", config.num_iterations)
        head_hidden_dim = checkpoint.get("head_hidden_dim", config.head_hidden_dim)
        head_dropout = checkpoint.get("head_dropout", config.head_dropout)
        head_type = checkpoint.get("head_type", getattr(config, "head_type", "mlp"))
        head_bn_affine = checkpoint.get("head_bn_affine", getattr(config, "head_bn_affine", False))
        iter1_head_type = checkpoint.get("iter1_head_type", getattr(config, "iter1_head_type", None))
        split_node_head_type = checkpoint.get("split_node_head_type", getattr(config, "split_node_head_type", False))
        backbone_name = checkpoint.get("backbone", config.backbone)
        num_features = checkpoint["num_features"]
        num_channels = checkpoint.get("num_channels", config.num_channels)
        image_size = checkpoint.get("image_size", config.image_size)

        # Reconstruct parent_layers for hierarchical modes (correct_label_routing)
        num_classes = checkpoint.get("num_classes", config.num_classes)
        parent_layers = CheckpointManager._reconstruct_parent_layers(
            hierarchical,
            iteration,
            num_features,
            head_hidden_dim,
            head_dropout,
            num_classes,
            head_type,
            head_bn_affine,
            iter1_head_type,
            split_node_head_type,
        )

        # Create model with reconstructed architecture
        model = HierarchicalClassifier(
            num_heads=checkpoint["num_heads"],
            backbone=backbone_name,
            pretrained=False,
            iteration=iteration,
            parent_layers=parent_layers,
            hierarchical=hierarchical,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            head_type=head_type,
            head_bn_affine=head_bn_affine,
            iter1_head_type=iter1_head_type,
            split_node_head_type=split_node_head_type,
            num_channels=num_channels,
            image_size=image_size,
        ).to(device)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def _reconstruct_parent_layers(
        hierarchical: str,
        iteration: int,
        num_features: int,
        head_hidden_dim: int,
        head_dropout: float,
        num_classes: int = 2,
        head_type: str = "mlp",
        head_bn_affine: bool = False,
        iter1_head_type: Optional[str] = None,
        split_node_head_type: bool = False,
    ) -> list:
        """Rebuild the tree of parent layers used in hierarchical mode (empty list for flat mode).

        Level 0 has num_classes modules; level k has num_classes * 2^k modules.
        """
        if hierarchical == "none" or iteration <= 1:
            return []

        parent_layers = []

        for level in range(iteration - 1):
            # Level 0: num_classes parents (iter 1 heads → Identity norm)
            # Level k: num_classes * 2^k parents (iter k+1 heads → BatchNorm)
            num_parents = num_classes * (2**level)
            in_dim = num_features if level == 0 else head_hidden_dim
            use_identity = (level == 0)  # iter 1 heads use Identity, iter 2+ use BatchNorm

            # Determine base head type for this level
            if level == 0:
                # Level 0 = iteration 1 output heads
                base_head_type = iter1_head_type if iter1_head_type is not None else head_type
            else:
                base_head_type = head_type

            def _make_norm(dim, use_id=use_identity):
                return nn.Identity() if use_id else nn.BatchNorm1d(dim, affine=head_bn_affine)

            def _make_single_head(ht, in_d=in_dim, hd=head_hidden_dim, do=head_dropout, ui=use_identity):
                if ht == "mlp":
                    return HeadModule(
                        feature_extractor=nn.Sequential(
                            _make_norm(in_d, ui),
                            nn.Linear(in_d, hd),
                            nn.ReLU(),
                            nn.Dropout(do),
                        ),
                        classifier=nn.Sequential(
                            _make_norm(hd, ui),
                            nn.Linear(hd, 1),
                        ),
                    )
                else:  # linear
                    return HeadModule(
                        feature_extractor=nn.Sequential(
                            _make_norm(in_d, ui),
                            nn.Linear(in_d, hd),
                        ),
                        classifier=nn.Sequential(
                            _make_norm(hd, ui),
                            nn.Linear(hd, 1),
                        ),
                    )

            parent_heads = []
            for i in range(num_parents):
                # split_node_head_type only applies to iter 2+ heads (level > 0)
                # Accepts True (easy=linear, hard=mlp) or "reversed" (easy=mlp, hard=linear).
                if split_node_head_type and level > 0:
                    if split_node_head_type == "reversed":
                        ht = "mlp" if i % 2 == 0 else "linear"
                    else:
                        ht = "linear" if i % 2 == 0 else "mlp"
                else:
                    ht = base_head_type
                parent_heads.append(_make_single_head(ht))

            parent_layers.append(nn.ModuleList(parent_heads))

        return parent_layers
