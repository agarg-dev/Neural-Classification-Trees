"""Checkpoint management utilities for NCT models."""

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..models.hierarchical import HierarchicalClassifier


class CheckpointManager:
    """Manages saving and loading of NCT model checkpoints."""

    @staticmethod
    def save_iteration_checkpoint(
        model: nn.Module, iteration: int, config: Any, checkpoint_dir: str
    ) -> str:
        """
        Save model checkpoint after an iteration.

        Args:
            model: Model to save
            iteration: Current iteration number (1-indexed)
            config: NCTConfig instance
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Path to saved checkpoint
        """
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
            "num_channels": config.num_channels,
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
        """
        Save the final trained model with full configuration and results.

        Args:
            model: Trained model
            config: NCTConfig instance
            iteration_results: List of per-iteration results
            filepath: Path where checkpoint will be saved
            verbose: Verbosity level
            trained_nodes_mask: Optional boolean mask indicating which nodes have training samples
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "num_heads": len(model.output_heads),
            "iteration_results": iteration_results,
            # Architecture parameters (same as save_iteration_checkpoint for consistency)
            "iteration": config.num_iterations,
            "hierarchical": config.hierarchical_architecture,
            "backbone": config.backbone,
            "num_features": model.num_features,
            "head_hidden_dim": config.head_hidden_dim,
            "head_dropout": config.head_dropout,
            "head_type": config.head_type,
            "num_channels": config.num_channels,
            "num_classes": config.num_classes,
        }

        # Save trained nodes mask if provided (for min_samples_per_node feature)
        if trained_nodes_mask is not None:
            checkpoint["trained_nodes_mask"] = trained_nodes_mask

        torch.save(checkpoint, filepath)

        if verbose:
            print(f"Checkpoint saved to {filepath}")

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: str, config: Any, device: torch.device
    ) -> nn.Module:
        """
        Load a model from iteration checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: NCTConfig instance (used as fallback for missing params)
            device: Device to load model on

        Returns:
            Loaded model in eval mode
        """
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
        """
        Load a complete NCTClassifier from final checkpoint.

        Args:
            filepath: Path to checkpoint file
            classifier_class: NCTClassifier class (passed to avoid circular import)
            device: Device to load model on (None = use saved device)

        Returns:
            Loaded NCTClassifier instance
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
        """
        Reconstruct model architecture from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
            config: NCTConfig instance (fallback for missing params)
            device: Device to load model on

        Returns:
            Reconstructed model with loaded weights
        """
        if "num_heads" not in checkpoint:
            raise RuntimeError("Invalid checkpoint: missing 'num_heads'")

        # Extract architecture parameters (with fallbacks to config)
        hierarchical = checkpoint.get("hierarchical", config.hierarchical_architecture)
        iteration = checkpoint.get("iteration", config.num_iterations)
        head_hidden_dim = checkpoint.get("head_hidden_dim", config.head_hidden_dim)
        head_dropout = checkpoint.get("head_dropout", config.head_dropout)
        head_type = checkpoint.get("head_type", getattr(config, "head_type", "mlp"))
        backbone_name = checkpoint.get("backbone", config.backbone)
        num_features = checkpoint["num_features"]
        num_channels = checkpoint.get("num_channels", config.num_channels)

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
            num_channels=num_channels,
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
    ) -> list:
        """
        Reconstruct parent_layers architecture for hierarchical models.

        In hierarchical modes (correct_label_routing),
        parent layers form a tree structure:
        - Level 0: num_classes modules (from iteration 1 -> 2)
        - Level k: num_classes * 2^k modules

        Args:
            hierarchical: Architecture mode ('none' or 'correct_label_routing')
            iteration: Iteration number
            num_features: Backbone feature dimension
            head_hidden_dim: Hidden dimension for heads
            head_dropout: Dropout rate
            num_classes: Number of base classes (default 2 for binary)
            head_type: Head architecture type ('mlp' or 'linear')

        Returns:
            List of parent layer ModuleLists (empty for non-hierarchical mode)
        """
        # Both correct_label_routing use tree structure
        if hierarchical == "none" or iteration <= 1:
            return []

        parent_layers = []

        for level in range(iteration - 1):
            # Level 0: num_classes parents, Level k: num_classes * 2^k parents
            num_parents = num_classes * (2**level)
            in_dim = num_features if level == 0 else head_hidden_dim

            # Create parent head based on head_type
            if head_type == "mlp":
                parent_layer = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(in_dim, head_hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(head_dropout),
                            nn.Linear(head_hidden_dim, 1),  # Classification layer for aux loss
                        )
                        for _ in range(num_parents)
                    ]
                )
            else:  # linear
                parent_layer = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(in_dim, head_hidden_dim),
                            nn.Linear(head_hidden_dim, 1),  # Classification layer for aux loss
                        )
                        for _ in range(num_parents)
                    ]
                )

            parent_layers.append(parent_layer)

        return parent_layers
