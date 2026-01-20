"""
Neural Classification Tree (NCT) Classifier.

Main interface for hierarchical image classification using iterative training.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ..data.datasets import IterativeDataset, extract_labels_from_dataset
from ..data.loaders import create_data_loader, create_node_balanced_sampler
from ..evaluation.analysis import analyze_hierarchical_nodes
from ..evaluation.metrics import (
    calculate_basic_metrics,
    calculate_group_metrics,
    calculate_hierarchical_metrics,
    calculate_subgroup_auroc,
    compute_iteration_transitions,
    compute_pseudo_label_metrics,
    print_evaluation_report,
)
from ..models.utils import initialize_or_update_model
from ..training.loss import compute_class_weights
from ..training.trainer import NCTTrainer
from ..utils.checkpoints import CheckpointManager
from ..utils.device import get_device
from ..utils.hierarchical import (
    get_class_node_range,
    get_nodes_per_class,
    get_num_nodes,
    node_to_class,
)
from ..utils.logger import NCTLogger
from ..utils.seed import set_seed
from ..visualization.attribution_captum import ALL_METHODS, LAYER_METHODS, PIXEL_METHODS

try:
    from ..visualization.attribution_captum import CaptumAttribution

    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    CaptumAttribution = None
from .config import NCTConfig


class NCTClassifier:
    """
    Neural Classification Tree for hierarchical classification.

    Builds a tree hierarchy of classifiers through iterative training,
    routing difficult samples to specialized nodes. Supports both binary
    (2-class) and multi-class (e.g., 5-class CMNIST) classification.

    Node structure per iteration:
        - Iteration 1: num_classes nodes (one per class)
        - Iteration 2: num_classes * 2 nodes (easy/hard split per class)
        - Iteration N: num_classes * 2^(N-1) nodes

    Attributes:
        config: Configuration with all hyperparameters.
        device: Torch device for training and inference.
        model: The HierarchicalClassifier model.
        generator: RNG for reproducible shuffling.
    """

    def __init__(self, config: NCTConfig = None, **kwargs):
        """
        Initialize the NCT classifier.

        Args:
            config: NCTConfig object. If None, builds from **kwargs.
            **kwargs: Used to build NCTConfig if config is None (backwards compatible).
        """

        if config is not None:
            self.config = config
        else:
            self.config = NCTConfig(**kwargs)

        self.device = get_device(self.config.device)
        if self.config.seed is not None:
            set_seed(self.config.seed)

        # Persistent generator for reproducible data shuffling
        self.generator = torch.Generator()
        if self.config.seed is not None:
            self.generator.manual_seed(self.config.seed)

        self.model = None
        self._is_fitted = False

        # Initialize logger
        self.logger = NCTLogger(verbose=self.config.verbose, config=self.config)
        self._iteration_results = []
        self._prev_weights = None  # For weight change tracking
        self._trained_nodes_mask = (
            None  # Mask for nodes with training samples (used at inference)
        )

    # ============================================================================
    # PUBLIC API
    # ============================================================================

    def fit(
        self,
        X_train: Any,
        X_val: Any,
        X_test: Optional[Any] = None,
        group_labels: Optional[Any] = None,
        dataset_name: str = "generic",
        class_names: Optional[Tuple[str, str]] = None,
        group_names: Optional[Dict[int, str]] = None,
        test_subgroups: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Train the hierarchical classifier on the provided data.
        Args:
            X_train: Training data. Must be a PyTorch Dataset object.
            X_val: Validation data. Must be a PyTorch Dataset object.
            X_test: Optional test data for per-iteration evaluation.
            group_labels: Optional group labels for test data evaluation.
            dataset_name: Name of dataset for evaluation (e.g., "waterbird").
            class_names: Optional binary class names (e.g., ("Landbird", "Waterbird")).
            group_names: Optional group ID to name mapping for display.
            test_subgroups: Optional subgroup masks for AUROC metrics (e.g., ISIC patch/histopathology).
        Returns:
            Dictionary with training results including:
                - 'val_pseudo_metrics': Validation pseudo-label metrics from final iteration (if available)
                - 'classifier': The fitted NCTClassifier instance (for backward compatibility)
        """

        if not isinstance(X_train, TorchDataset):
            raise NotImplementedError(
                "Direct tensor/array inputs not supported. Please pass PyTorch Dataset objects."
            )

        # Store class and group names for later use
        self._class_names = class_names
        self._group_names = group_names
        self._test_subgroups = test_subgroups

        y_train = extract_labels_from_dataset(X_train)
        y_val = extract_labels_from_dataset(X_val)
        self.model = self._run_iterative_training(
            X_train,
            X_val,
            y_train,
            y_val,
            X_test,
            group_labels,
            dataset_name,
            test_subgroups,
        )
        self._is_fitted = True

        # Finish WandB run unless we're going to generate attributions (which need to be logged first)
        finish_wandb_now = not self.config.log_attributions
        self.logger.training_end(finish_wandb=finish_wandb_now)

        # Extract validation metrics from final iteration for Optuna optimization
        val_pseudo_metrics = None
        if len(self._iteration_results) > 0:
            final_iter_results = self._iteration_results[-1]
            val_pseudo_metrics = final_iter_results.get("val_pseudo_metrics")

        return {
            "classifier": self,
            "val_pseudo_metrics": val_pseudo_metrics,
        }

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict binary class labels for samples.

        Args:
            X: Input data. Must be a PyTorch Dataset object.

        Returns:
            Binary predictions as a numpy array of shape (n_samples,).

        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        test_loader = self._create_iterative_loader(X, is_training=False)
        hierarchical_preds = self._get_hierarchical_predictions(test_loader)
        class_preds = node_to_class(
            hierarchical_preds, self.config.num_iterations, self.config.num_classes
        )
        return class_preds.numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for samples.
        Args:
            X: Input data. Must be a PyTorch Dataset object.
        Returns:
            Class probabilities as a numpy array of shape (n_samples, num_classes).
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        test_loader = self._create_iterative_loader(X, is_training=False)
        hierarchical_probs = self._get_hierarchical_probabilities(test_loader)

        # Sum probabilities for nodes belonging to each class
        class_probs = []
        for class_id in range(self.config.num_classes):
            start_node, end_node = get_class_node_range(
                class_id, self.config.num_iterations
            )
            class_raw = hierarchical_probs[:, start_node:end_node].sum(dim=1)
            class_probs.append(class_raw)

        # Normalize
        total = sum(class_probs)
        class_probs = [p / total for p in class_probs]

        return torch.stack(class_probs, dim=1).numpy()

    # ============================================================================
    # DATA UTILITIES - DataLoader Creation
    # ============================================================================

    def _create_iterative_loader(
        self, dataset: Any, is_training: bool, use_node_balanced: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader for iterative dataset.
        Args:
            dataset: Input dataset (IterativeDataset).
            is_training: Whether this is for training or evaluation.
            use_node_balanced: If True, use node-balanced sampling (DFR-style).
        Returns:
            DataLoader configured for iterative training/evaluation.
        """
        if is_training:
            if use_node_balanced:
                # DFR-style: balance across nodes, not just classes
                sampler = create_node_balanced_sampler(dataset.hier_labels)
                return DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                )
            else:
                return create_data_loader(
                    dataset,
                    is_training=True,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    generator=self.generator,
                )
        else:
            return create_data_loader(
                dataset,
                is_training=False,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                seed=self.config.seed,
            )

    # def _create_iterative_loader(self, dataset: Any, is_training: bool) -> DataLoader:
    #     """
    #     Create a DataLoader for iterative dataset.
    #     Args:
    #         dataset: Input dataset (IterativeDataset).
    #         is_training: Whether this is for training or evaluation.
    #     Returns:
    #         DataLoader configured for iterative training/evaluation.
    #     """
    #     if is_training:
    #         return create_data_loader(
    #             dataset,
    #             is_training=True,
    #             batch_size=self.config.batch_size,
    #             num_workers=self.config.num_workers,
    #             pin_memory=self.config.pin_memory,
    #             generator=self.generator,
    #         )
    #     else:
    #         return create_data_loader(
    #             dataset,
    #             is_training=False,
    #             batch_size=self.config.batch_size,
    #             num_workers=self.config.num_workers,
    #             pin_memory=self.config.pin_memory,
    #             seed=self.config.seed,
    #         )

    def evaluate(
        self,
        X_test: Any,
        group_labels: Optional[List[int]] = None,
        group_names: Optional[Dict[int, str]] = None,
        dataset_name: str = "generic",
        class_names: Optional[Tuple[str, str]] = None,
        test_subgroups: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        Args:
            X_test: Test data. Must be a PyTorch Dataset object.
            group_labels: Optional group/subpopulation labels for fairness analysis.
            group_names: Optional mapping from group IDs to readable names.
            dataset_name: Name of the dataset.
            class_names: Optional binary class names.
            test_subgroups: Optional subgroup masks for AUROC metrics.
        Returns:
            Dictionary containing evaluation metrics:
                - basic_metrics: Accuracy, precision, recall, F1.
                - pseudo_label_metrics: Routing quality metrics (if applicable).
                - group_metrics: Per-group performance (if group_labels provided).
                - node_analysis: Hierarchical node utilization.
        """

        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        # Use stored values if not provided
        if class_names is None and hasattr(self, "_class_names"):
            class_names = self._class_names
        if group_names is None and hasattr(self, "_group_names"):
            group_names = self._group_names
        if test_subgroups is None and hasattr(self, "_test_subgroups"):
            test_subgroups = self._test_subgroups

        binary_predictions = self.predict(X_test)
        binary_probabilities = self.predict_proba(X_test)
        test_loader = self._create_iterative_loader(X_test, is_training=False)
        hierarchical_predictions = self._get_hierarchical_predictions(test_loader)

        # Calculate basic metrics (with AUROC using probability of positive class)
        y_test = extract_labels_from_dataset(X_test)
        y_test_np = y_test.numpy()
        y_prob_positive = binary_probabilities[:, 1]  # Probability of class 1
        basic_metrics = calculate_basic_metrics(
            y_test_np, binary_predictions, y_prob=y_prob_positive
        )

        # Compute subgroup AUROC if test_subgroups provided
        subgroup_auroc = calculate_subgroup_auroc(
            y_test_np, y_prob_positive, test_subgroups
        )
        if subgroup_auroc:
            basic_metrics.update(subgroup_auroc)

        results = {"basic_metrics": basic_metrics}

        # # Calculate pseudo label metrics (ISSUE- labels are always for test set while we can pass any dataset)
        # if len(self._iteration_results) >= 2 and self._iteration_results[-2].get("test_pseudo_labels") is not None:
        #     prev_pseudo_labels = self._iteration_results[-2]["test_pseudo_labels"]
        #     curr_routing = hierarchical_predictions.cpu().numpy()
        #     pseudo_metrics = compute_pseudo_label_metrics(prev_pseudo_labels, curr_routing)
        #     results["pseudo_label_metrics"] = pseudo_metrics

        # Calculate group metrics if group labels provided
        if group_labels is not None:
            group_labels_np = np.array(group_labels)
            group_metrics = calculate_group_metrics(
                y_test_np,
                binary_predictions,
                group_labels_np,
                y_prob_positive,
                group_names,
            )
            results["group_metrics"] = group_metrics

            # Analyze hierarchical nodes
            node_analysis = analyze_hierarchical_nodes(
                hierarchical_predictions,
                torch.from_numpy(y_test_np),
                torch.from_numpy(group_labels_np),
                self.config.num_iterations,
                group_names,
                verbose=self.config.verbose,
                hierarchical=self.config.hierarchical_architecture,
                class_names=class_names,
            )
            results["node_analysis"] = node_analysis

        if self.config.verbose:
            print_evaluation_report(results, "NCT Evaluation Results")

        return results

    def generate_attribution_visualizations(
        self,
        X_test: Any,
        save_dir: str = "./attributions",
        num_per_node: int = 5,
        num_transitions: int = 5,
        methods: Union[str, List[str]] = "layer_gradcam",
        class_names: Optional[Tuple[str, str]] = None,
    ) -> None:
        """
        Generate attribution visualizations for hierarchical nodes using Captum.

        Must be called after fit(). Generates two types of visualizations:
        1. Random samples per node - shows general attention patterns
        2. Transition samples - shows forgetting/learning mechanisms

        Args:
            X_test: Test dataset (same as used in fit())
            save_dir: Directory to save visualizations
            num_per_node: Number of random samples per node
            num_transitions: Number of transition samples to visualize
            methods: Attribution method(s) to use. Can be:
                - Single method string: 'layer_gradcam', 'saliency', 'deeplift', etc.
                - List of methods: ['layer_gradcam', 'saliency', 'deeplift']
                - 'all': All available methods
                - 'all_layer': All layer-based methods (layer_gradcam, layer_conductance, etc.)
                - 'all_pixel': All pixel-based methods (saliency, guided_gradcam, deeplift, etc.)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before generating attributions")

        if len(self._iteration_results) == 0:
            # Iteration results should be available if loaded from final checkpoint.
            # They're only missing if loaded from iteration checkpoint or old format.
            if self.model is None:
                raise ValueError(
                    "No iteration results found. Pass X_test to fit() to enable attributions"
                )

        if self.model is None:
            raise ValueError("Model is not available for attribution generation")

        if not CAPTUM_AVAILABLE:
            raise ImportError(
                "Captum is required for attribution generation. "
                "Install with: pip install captum"
            )

        # Initialize visualizer with default method (will be overridden by methods parameter)
        visualizer = CaptumAttribution(
            model=self.model,
            device=self.device,
            checkpoint_loader=lambda path: CheckpointManager.load_model_from_checkpoint(
                path, self.config, self.device
            ),
            class_names=class_names
            if class_names
            else (
                self._class_names
                if hasattr(self, "_class_names")
                else ["Class 0", "Class 1"]
            ),
            num_iterations=self.config.num_iterations,
            hierarchical=self.config.hierarchical_architecture,
            attribution_method="layer_gradcam",  # Default, overridden by methods parameter
        )

        visualizer.generate_all_visualizations(
            dataset=X_test,
            iteration_results=self._iteration_results,
            save_dir=save_dir,
            num_per_node=num_per_node,
            num_transitions=num_transitions,
            seed=self.config.seed,
            methods=methods,  # Pass through methods parameter
        )

        # Log attribution images if enabled
        if self.config.log_attributions:
            # Determine which methods were actually run
            if isinstance(methods, list):
                methods_to_log = methods
            elif isinstance(methods, str):
                if methods == "all":
                    methods_to_log = ALL_METHODS
                elif methods == "all_layer":
                    methods_to_log = LAYER_METHODS
                elif methods == "all_pixel":
                    methods_to_log = PIXEL_METHODS
                else:
                    methods_to_log = [methods]
            else:
                methods_to_log = ["layer_gradcam"]  # fallback

            # Log for each method that was run
            for method_name in methods_to_log:
                method_save_dir = os.path.join(save_dir, method_name)

                # Log per-iteration node visualizations
                for iter_idx in range(len(self._iteration_results)):
                    iteration_num = iter_idx + 1
                    node_viz_path = os.path.join(
                        method_save_dir, f"iteration_{iteration_num}_nodes.png"
                    )
                    if os.path.exists(node_viz_path):
                        caption = (
                            f"Iter {iteration_num} - {method_name} - Node Attributions"
                        )
                        self.logger.log_attribution_image(
                            node_viz_path, caption, iteration_num
                        )

                # Log transition visualizations
                for i in range(len(self._iteration_results) - 1):
                    forgetting_path = os.path.join(
                        method_save_dir,
                        f"transitions_{i + 1}_to_{i + 2}_forgetting.png",
                    )
                    learning_path = os.path.join(
                        method_save_dir, f"transitions_{i + 1}_to_{i + 2}_learning.png"
                    )

                    if os.path.exists(forgetting_path):
                        caption = f"{method_name}: Forgetting {i + 1} → {i + 2}"
                        self.logger.log_attribution_image(forgetting_path, caption)

                    if os.path.exists(learning_path):
                        caption = f"{method_name}: Learning {i + 1} → {i + 2}"
                        self.logger.log_attribution_image(learning_path, caption)

            # Finish WandB run after all attributions are logged
            self.logger.finish_wandb()

    # ============================================================================
    # CHECKPOINT MANAGEMENT
    # ============================================================================

    def save_checkpoint(self, filepath: str) -> None:
        """Save the trained model and configuration to disk."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        if self.model is None:
            raise RuntimeError("Model is None despite being fitted - internal error")

        CheckpointManager.save_final_checkpoint(
            self.model,
            self.config,
            self._iteration_results,
            filepath,
            self.config.verbose,
            trained_nodes_mask=self._trained_nodes_mask,
        )

    @classmethod
    def from_checkpoint(
        cls, filepath: str, device: Optional[Union[str, torch.device]] = None
    ) -> "NCTClassifier":
        """Load a trained classifier from a checkpoint file."""
        return CheckpointManager.load_classifier_from_checkpoint(filepath, cls, device)

    # ============================================================================
    # TRAINING ORCHESTRATION - Iterative Training Loop
    # ============================================================================

    def _run_iterative_training(
        self,
        base_train_dataset: Any,
        base_val_dataset: Any,
        current_train_labels: torch.Tensor,
        current_val_labels: torch.Tensor,
        X_test: Optional[Any] = None,
        group_labels: Optional[Any] = None,
        dataset_name: str = "generic",
        test_subgroups: Optional[Dict[str, np.ndarray]] = None,
    ) -> nn.Module:
        """
        Execute the main iterative training loop.
        This method implements the core NCT algorithm by iteratively building
        a hierarchical tree of classifiers, routing samples based on correctness.
        Args:
            base_train_dataset: Base training dataset.
            base_val_dataset: Base validation dataset.
            current_train_labels: Current hierarchical labels for training.
            current_val_labels: Current hierarchical labels for validation.
            X_test: Optional test data for per-iteration evaluation.
            group_labels: Optional group labels for test evaluation.
            dataset_name: Name of dataset for evaluation.
            test_subgroups: Optional subgroup masks for AUROC metrics.
        Returns:
            The trained HierarchicalClassifier model.
        """

        current_base_lr = self.config.initial_base_lr
        model = None

        self.logger.training_start(self.config.to_dict())

        trainer = NCTTrainer(
            None, self.device, self.config.to_dict(), logger=self.logger
        )

        for i in range(self.config.num_iterations):
            num_nodes = get_num_nodes(i + 1, self.config.num_classes)
            epochs_for_iter = self.config.epochs_per_iteration[i]

            # Update trainer's iteration number for logging
            trainer.iteration_num = i + 1

            # Compute class weights for imbalanced data
            # Skip class weights when:
            # - using node-balanced sampling in iteration 2+ (redundant)
            # - skip_class_weights_iter1 is True and this is iteration 1
            skip_weights = (i > 0 and self.config.node_balanced_sampling) or (
                i == 0 and self.config.skip_class_weights_iter1
            )
            if self.config.use_class_weights and not skip_weights:
                class_weights, class_counts = compute_class_weights(
                    current_train_labels,
                    num_nodes,
                    return_counts=True,
                    max_weight=self.config.max_class_weight,
                )
            else:
                class_weights, class_counts = None, None

            # Store mask of which nodes have training samples (for inference masking)
            # This is updated each iteration; the final iteration's mask is used at inference
            node_counts = torch.bincount(current_train_labels, minlength=num_nodes)
            self._trained_nodes_mask = node_counts > 0

            self.logger.iteration_start(
                i + 1,
                self.config.num_iterations,
                num_nodes,
                epochs_for_iter,
                class_weights,
                node_counts,  # Use raw counts for logging to show zeros
            )

            if i > 0:
                self.logger.weight_transfer(self.config.hierarchical_architecture)

            model = initialize_or_update_model(
                num_nodes,
                model,
                self.config.backbone,
                self.config.pretrained,
                self.device,
                verbose=self.config.verbose,
                iteration=i + 1,
                hierarchical=self.config.hierarchical_architecture,
                head_hidden_dim=self.config.head_hidden_dim,
                head_dropout=self.config.head_dropout,
                num_channels=self.config.num_channels,
            )

            train_iter_dataset = IterativeDataset(
                base_train_dataset, current_train_labels
            )
            val_iter_dataset = IterativeDataset(base_val_dataset, current_val_labels)

            use_node_balanced = (
                i > 0  # Only from iteration 2+
                and self.config.node_balanced_sampling
            )
            train_iter_loader = self._create_iterative_loader(
                train_iter_dataset,
                is_training=True,
                use_node_balanced=use_node_balanced,
            )

            # train_iter_loader = self._create_iterative_loader(
            #     train_iter_dataset, is_training=True
            # )
            val_iter_loader = self._create_iterative_loader(
                val_iter_dataset, is_training=False
            )

            trainer.model = model
            training_results = trainer.train_iteration(
                train_iter_loader,
                val_iter_loader,
                epochs_for_iter,
                current_base_lr,
                self.config.initial_head_lr,
                class_weights.to(self.device) if class_weights is not None else None,
            )

            self.logger.iteration_end()

            weight_stats = self._compute_weight_changes(model)
            if weight_stats:
                self.logger.log_weight_statistics(i + 1, weight_stats)

            # Save checkpoint after training
            checkpoint_path = None
            if self.config.save_iteration_checkpoints:
                checkpoint_path = CheckpointManager.save_iteration_checkpoint(
                    model, i + 1, self.config, self.config.checkpoint_dir
                )

            # Evaluate test set if provided
            if X_test is not None:
                self._evaluate_test_at_iteration(
                    model,
                    X_test,
                    i + 1,
                    group_labels,
                    dataset_name,
                    checkpoint_path,
                    training_results.get("total_epochs", 0),
                    training_results.get("early_stopped", False),
                    test_subgroups,
                )

            # Update labels for next iteration (returns validation predictions)
            if i < self.config.num_iterations - 1:
                current_train_labels, current_val_labels, val_preds = (
                    self._update_labels_for_next_iteration(
                        model,
                        base_train_dataset,
                        base_val_dataset,
                        current_train_labels,
                        current_val_labels,
                    )
                )

                # Store validation metrics
                self._evaluate_validation_at_iteration(
                    val_preds,
                    current_val_labels,
                    i + 1,
                )

                current_base_lr /= self.config.lr_decay_factor
            else:
                # For final iteration, still evaluate validation to compute pseudo-label metrics
                # This is needed for Optuna optimization which uses final iteration metrics
                val_preds, _ = self._evaluate_for_routing(
                    base_val_dataset, current_val_labels, model
                )
                self._evaluate_validation_at_iteration(
                    val_preds,
                    None,  # No pseudo labels needed (no next iteration)
                    i + 1,
                )

        return model

    def _update_labels_for_next_iteration(
        self,
        model,
        base_train_dataset: Any,
        base_val_dataset: Any,
        train_labels: torch.Tensor,
        val_labels: torch.Tensor,
    ):
        """
        Update hierarchical labels for the next iteration.

        Evaluates current model and routes samples to "easy" or "hard" child nodes
        based on prediction correctness and optional confidence thresholding.

        Args:
            model: Current trained model.
            base_train_dataset: Base training dataset.
            base_val_dataset: Base validation dataset.
            train_labels: Current training labels.
            val_labels: Current validation labels.

        Returns:
            Tuple of (new_train_labels, new_val_labels, val_hierarchical_preds) for next iteration.
        """
        # Evaluate model on current iteration
        train_preds, is_correct_train = self._evaluate_for_routing(
            base_train_dataset, train_labels, model
        )
        val_preds, is_correct_val = self._evaluate_for_routing(
            base_val_dataset, val_labels, model
        )

        # Determine which samples are "difficult" (routed to odd child nodes)
        is_difficult_train = self._determine_difficult_samples(
            model, base_train_dataset, train_labels, is_correct_train
        )
        is_difficult_val = self._determine_difficult_samples(
            model, base_val_dataset, val_labels, is_correct_val
        )

        # Log routing statistics for correctness_certainty routing
        if self.config.routing_strategy == "correctness_certainty":
            routing_stats = {
                "num_wrong": (~is_correct_train).sum().item(),
                "num_uncertain": (is_correct_train & is_difficult_train).sum().item(),
                "num_difficult": is_difficult_train.sum().item(),
                "total": len(train_labels),
            }
            self.logger.log_routing_statistics(routing_stats)

        # Log section header for pseudo-label generation (node merges logged below)
        if self.config.min_samples_per_node > 0:
            next_iteration = len(self._iteration_results) + 1
            self.logger.log(
                f"\n--- Preparing Pseudo-Labels for Iteration {next_iteration} ---",
                level=2,
            )

        new_train_labels = self._compute_pseudo_labels_for_dataset(
            train_preds, train_labels, is_difficult_train, dataset_name="train"
        )
        new_val_labels = self._compute_pseudo_labels_for_dataset(
            val_preds, val_labels, is_difficult_val, dataset_name="val"
        )

        # Return validation predictions for pseudo-label metric computation
        return new_train_labels, new_val_labels, val_preds

    def _evaluate_for_routing(
        self,
        base_dataset: Any,
        labels: torch.Tensor,
        model: nn.Module,
    ):
        """
        Get predictions and correctness for any single dataset.

        Args:
            base_dataset: Base dataset (regular PyTorch Dataset)
            labels: Labels for the dataset
            model: Current trained model

        Returns:
            Tuple of (predictions, is_correct)
        """
        eval_dataset = IterativeDataset(base_dataset, labels)
        loader = self._create_iterative_loader(eval_dataset, is_training=False)
        predictions = self._get_hierarchical_predictions(loader, model=model)
        is_correct = predictions == labels
        return predictions, is_correct

    def _compute_pseudo_labels_for_dataset(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        is_difficult: torch.Tensor,
        dataset_name: str = "data",
    ) -> torch.Tensor:
        """
        Compute pseudo labels for routing to next iteration.

        Uses tree expansion formula: node_id * 2 + is_difficult
        This routes samples from N nodes to 2N child nodes.

        If min_samples_per_node > 0, sparse difficult nodes (odd indices) are
        merged into their easy siblings to prevent training instability.

        Args:
            predictions: Hierarchical predictions (node indices)
            labels: True binary labels
            is_difficult: Boolean tensor indicating difficult samples
            dataset_name: Name of the dataset (train/val/test) for logging

        Returns:
            Pseudo labels for next iteration (range depends on current iteration)
        """
        # Route by label: all samples of a class stay within that class's hierarchy
        # Each class has its own easy/hard split
        proposed_labels = labels * 2 + is_difficult.long()

        # Merge sparse difficult nodes into easy siblings if threshold is set
        min_samples = self.config.min_samples_per_node
        if min_samples > 0:
            num_nodes = proposed_labels.max().item() + 1
            counts = torch.bincount(proposed_labels, minlength=num_nodes)

            # Difficult nodes are odd indices (1, 3, 5, 7, ...)
            # Merge them into easy sibling (node - 1) if below threshold
            for difficult_node in range(1, num_nodes, 2):
                if counts[difficult_node] < min_samples:
                    mask = proposed_labels == difficult_node
                    num_merged = mask.sum().item()
                    if num_merged > 0:
                        proposed_labels[mask] = difficult_node - 1
                        # Log the merge if verbose
                        if self.config.verbose >= 2:
                            self.logger.log(
                                f"    Merged node {difficult_node} ({num_merged} samples) "
                                f"into node {difficult_node - 1} [{dataset_name}]",
                                level=2,
                            )

        return proposed_labels

    def _determine_difficult_samples(
        self,
        model: nn.Module,
        base_dataset: Any,
        labels: torch.Tensor,
        is_correct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Determine which samples should be routed to "difficult" child nodes.
        Strategies:
        - correctness: Simply ~is_correct
        - correctness_certainty: ~is_correct OR (is_correct & uncertain)
        """
        strategy = self.config.routing_strategy

        if strategy == "correctness_certainty":
            return self._apply_entropy_routing(model, base_dataset, labels, is_correct)

        # Default / Fallback to 'correctness'
        return ~is_correct

    def _apply_entropy_routing(
        self,
        model: nn.Module,
        base_dataset: Any,
        labels: torch.Tensor,
        is_correct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply 'correctness_certainty' routing logic:
        - Wrong samples → Always Hard (Baseline)
        - Correct samples → Hard if in high-entropy GMM cluster (Uncertain)
        """
        eval_dataset = IterativeDataset(base_dataset, labels)
        eval_loader = self._create_iterative_loader(eval_dataset, is_training=False)

        # Get model outputs
        model.eval()
        all_outputs = []
        with torch.no_grad():
            for images, _ in eval_loader:
                images = images.to(self.device)
                outputs = model(images)
                all_outputs.append(outputs.cpu())
        outputs = torch.cat(all_outputs)

        # Compute entropy using sigmoid (matches BCE training)
        probs = torch.sigmoid(outputs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # 1. Baseline: Wrong samples are always hard
        is_incorrect = ~is_correct

        # 2. Analyze Correct samples for uncertainty
        correctly_predicted_mask = is_correct
        is_correct_but_uncertain = torch.zeros_like(is_correct, dtype=torch.bool)

        # Perform GMM clustering if enough correct samples
        num_correct = correctly_predicted_mask.sum().item()
        if num_correct > 10:
            entropy_of_correct = (
                entropy[correctly_predicted_mask].numpy().reshape(-1, 1)
            )

            # Fit 1 and 2 cluster models
            single_cluster = GaussianMixture(n_components=1, random_state=42).fit(
                entropy_of_correct
            )
            two_clusters = GaussianMixture(n_components=2, random_state=42).fit(
                entropy_of_correct
            )

            # Check if 2 clusters fit better (lower BIC)
            two_clusters_exist = two_clusters.bic(
                entropy_of_correct
            ) < single_cluster.bic(entropy_of_correct)

            if two_clusters_exist:
                cluster_assignments = two_clusters.predict(entropy_of_correct)

                # Higher mean = higher entropy = uncertain cluster
                uncertain_cluster_id = (
                    1 if two_clusters.means_[1] > two_clusters.means_[0] else 0
                )

                # Mark correct samples in uncertain cluster as hard
                correct_base_indices = torch.where(correctly_predicted_mask)[0]
                for i, idx in enumerate(correct_base_indices):
                    if cluster_assignments[i] == uncertain_cluster_id:
                        is_correct_but_uncertain[idx] = True

        # Final: Hard = Incorrect OR (Correct but Uncertain)
        is_difficult = is_incorrect | is_correct_but_uncertain

        return is_difficult

    # ============================================================================
    # INFERENCE - Model Predictions
    # ============================================================================

    def _get_hierarchical_predictions(
        self, dataloader: DataLoader, model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Get hierarchical node predictions from model.

        If min_samples_per_node > 0 and _trained_nodes_mask is set, masks out
        untrained nodes (sets their logits to -inf) before taking argmax.

        Args:
            dataloader: DataLoader to generate predictions for.
            model: Model to use for predictions. If None, uses self.model.
        Returns:
            Tensor of hierarchical node predictions.
        """
        if model is None:
            model = self.model

        model.eval()
        all_predictions = []

        # Check if we should apply masking for untrained nodes
        apply_mask = (
            self.config.min_samples_per_node > 0
            and self._trained_nodes_mask is not None
        )

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = model(images)

                # Mask out untrained nodes (set their logits to -inf so argmax ignores them)
                if apply_mask:
                    mask = self._trained_nodes_mask.to(outputs.device)
                    # Only mask if we have fewer trained nodes than total outputs
                    if len(mask) == outputs.shape[1]:
                        outputs = outputs.clone()
                        outputs[:, ~mask] = float("-inf")

                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)

    def _get_hierarchical_probabilities(
        self, dataloader: DataLoader, model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Get hierarchical node probabilities from model.

        If min_samples_per_node > 0 and _trained_nodes_mask is set, sets
        probabilities of untrained nodes to 0.

        Args:
            dataloader: DataLoader to generate probabilities for.
            model: Model to use. If None, uses self.model.
        Returns:
            Tensor of hierarchical node probabilities.
        """
        if model is None:
            model = self.model

        model.eval()
        all_probabilities = []

        # Check if we should apply masking for untrained nodes
        apply_mask = (
            self.config.min_samples_per_node > 0
            and self._trained_nodes_mask is not None
        )

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = model(images)
                probabilities = torch.sigmoid(outputs)

                # Zero out probabilities for untrained nodes
                if apply_mask:
                    mask = self._trained_nodes_mask.to(probabilities.device)
                    if len(mask) == probabilities.shape[1]:
                        probabilities = probabilities.clone()
                        probabilities[:, ~mask] = 0.0

                all_probabilities.append(probabilities.cpu())

        return torch.cat(all_probabilities)

    def _get_class_probabilities(
        self,
        dataloader: DataLoader,
        iteration: int,
        model: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """
        Get class probabilities from hierarchical model.

        Args:
            dataloader: DataLoader to generate probabilities for.
            iteration: Current iteration number (determines number of nodes).
            model: Model to use. If None, uses self.model.

        Returns:
            Class probabilities as numpy array of shape (n_samples, num_classes).
        """
        hierarchical_probs = self._get_hierarchical_probabilities(dataloader, model)

        # Sum probabilities for nodes belonging to each class
        class_probs = []
        for class_id in range(self.config.num_classes):
            start_node, end_node = get_class_node_range(class_id, iteration)
            class_raw = hierarchical_probs[:, start_node:end_node].sum(dim=1)
            class_probs.append(class_raw)

        # Normalize
        total = sum(class_probs)
        class_probs = [p / total for p in class_probs]
        return torch.stack(class_probs, dim=1).numpy()

    # ============================================================================
    # EVALUATION - Per-Iteration Evaluation
    # ============================================================================

    def _evaluate_test_at_iteration(
        self,
        model: Any,
        X_test: Any,
        iteration: int,
        group_labels: Optional[Any] = None,
        dataset_name: str = "generic",
        checkpoint_path: Optional[str] = None,
        num_epochs_trained: int = 0,
        early_stopped: bool = False,
        test_subgroups: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Evaluate model on test set after a specific iteration.

        Args:
            model: The trained model at current iteration.
            X_test: Test dataset.
            iteration: Current iteration number (1-indexed).
            group_labels: Optional group labels for fairness analysis.
            dataset_name: Name of dataset for special handling.
            checkpoint_path: Path to saved checkpoint for this iteration.
            num_epochs_trained: Number of epochs trained in this iteration.
            early_stopped: Whether early stopping was triggered.
            test_subgroups: Optional subgroup masks for AUROC metrics.
        """
        if self.config.verbose == 0:
            return

        # Get test predictions and probabilities
        test_loader = self._create_iterative_loader(X_test, is_training=False)
        hierarchical_predictions = self._get_hierarchical_predictions(
            test_loader, model=model
        )
        class_probabilities = self._get_class_probabilities(
            test_loader, iteration, model=model
        )
        y_test = extract_labels_from_dataset(X_test)

        # Convert hierarchical to class predictions
        class_predictions = node_to_class(
            hierarchical_predictions, iteration, self.config.num_classes
        )

        # Convert to numpy
        y_test_np = y_test.cpu().numpy()
        class_predictions_np = class_predictions.cpu().numpy()
        # For binary, use prob of class 1; for multi-class, pass full probability matrix
        class_prob_for_auroc = (
            class_probabilities[:, 1]
            if self.config.num_classes == 2
            else class_probabilities
        )
        hierarchical_predictions_np = hierarchical_predictions.cpu().numpy()
        group_labels_np = np.array(group_labels) if group_labels is not None else None

        # Use stored values from fit()
        group_names = self._group_names if hasattr(self, "_group_names") else None

        self.logger.log(
            f"\n--- Evaluation on Test Set after Iteration {iteration} ---", level=1
        )

        # Generate pseudo labels for next iteration
        test_pseudo_labels = None
        if iteration < self.config.num_iterations:
            # For iteration > 1, use previous iteration's pseudo labels as base
            # This ensures correct label range (e.g., 0-7 for iter 3, not 0-3)
            if iteration > 1:
                current_test_labels = torch.from_numpy(
                    self._iteration_results[-1]["test_pseudo_labels"]
                )
            else:
                current_test_labels = y_test

            is_correct = hierarchical_predictions == current_test_labels
            is_difficult = self._determine_difficult_samples(
                model, X_test, current_test_labels, is_correct
            )
            test_pseudo_labels = self._compute_pseudo_labels_for_dataset(
                hierarchical_predictions,
                current_test_labels,
                is_difficult,
                dataset_name="test",
            )

        # Store test results
        self._iteration_results.append(
            {
                "predictions": class_predictions_np,
                "hierarchical_predictions": hierarchical_predictions_np,
                "test_pseudo_labels": test_pseudo_labels.cpu().numpy()
                if test_pseudo_labels is not None
                else None,
                "checkpoint_path": checkpoint_path,
            }
        )

        # Compute routing quality metrics (test)
        pseudo_metrics = None
        if (
            iteration > 1
            and self._iteration_results[-2]["test_pseudo_labels"] is not None
        ):
            prev_pseudo_labels = self._iteration_results[-2]["test_pseudo_labels"]
            curr_routing = self._iteration_results[-1]["hierarchical_predictions"]
            num_expected_nodes = get_num_nodes(iteration, self.config.num_classes)
            pseudo_metrics = compute_pseudo_label_metrics(
                prev_pseudo_labels, curr_routing, num_expected_nodes
            )

        # Compute test metrics (with AUROC using probability of positive class)
        basic_metrics = calculate_basic_metrics(
            y_test_np, class_predictions_np, y_prob=class_prob_for_auroc
        )
        group_results = None
        if group_labels_np is not None:
            group_results = calculate_group_metrics(
                y_test_np,
                class_predictions_np,
                group_labels_np,
                class_prob_for_auroc,
                group_names,
            )
        node_metrics = calculate_hierarchical_metrics(
            hierarchical_predictions_np, iteration, y_test_np, class_predictions_np
        )

        # Compute subgroup AUROC if test_subgroups provided (e.g., ISIC dataset)
        subgroup_auroc = calculate_subgroup_auroc(
            y_test_np, class_prob_for_auroc, test_subgroups
        )
        if subgroup_auroc:
            basic_metrics.update(subgroup_auroc)

        # Log results (summary metrics first)
        self.logger.log_evaluation_results(basic_metrics, pseudo_metrics, group_results)

        # Track transitions between iterations
        if iteration > 1 and len(self._iteration_results) > 1:
            prev_result = self._iteration_results[-2]
            transitions = compute_iteration_transitions(
                prev_result["predictions"],
                class_predictions_np,
                y_test_np,
                group_labels_np,
            )
            self.logger.log_transitions(
                iteration,
                iteration - 1,
                transitions,
                group_labels,
                dataset_name,
                group_names,
            )

        # Detailed node analysis (verbose, last)
        if group_labels_np is not None and self.config.verbose >= 2:
            analyze_hierarchical_nodes(
                hierarchical_predictions,
                y_test,
                torch.from_numpy(group_labels_np),
                iteration,
                group_names,
                verbose=True,
                hierarchical=self.config.hierarchical_architecture,
                class_names=self._class_names
                if hasattr(self, "_class_names")
                else None,
            )

        # Prepare summary
        training_time = (
            time.time() - self.logger.iteration_start_time
            if self.logger.iteration_start_time
            else 0.0
        )
        if pseudo_metrics:
            basic_metrics["pseudo_label_accuracy"] = pseudo_metrics["accuracy"]
            basic_metrics["pseudo_label_f1"] = pseudo_metrics["f1_score"]
            basic_metrics["pseudo_worst_group_acc"] = pseudo_metrics[
                "worst_group_accuracy"
            ]
            basic_metrics["pseudo_difficult_node_acc"] = pseudo_metrics.get(
                "difficult_node_accuracy", 0.0
            )

        self.logger.log_iteration_summary(
            iteration=iteration,
            test_metrics=basic_metrics,
            group_metrics=group_results,
            node_metrics=node_metrics,
            training_time=training_time,
            num_epochs_trained=num_epochs_trained,
            early_stopped=early_stopped,
        )

    def _evaluate_validation_at_iteration(
        self,
        val_hierarchical_preds: torch.Tensor,
        val_pseudo_labels: Optional[torch.Tensor],
        iteration: int,
    ) -> None:
        """
        Store validation pseudo labels and compute validation metrics.

        Args:
            val_hierarchical_preds: Validation hierarchical predictions from routing.
            val_pseudo_labels: Validation pseudo labels for next iteration.
            iteration: Current iteration number (1-indexed).
        """
        # Ensure results entry exists (may not if no test set)
        if len(self._iteration_results) < iteration:
            self._iteration_results.append(
                {
                    "predictions": None,
                    "hierarchical_predictions": None,
                    "test_pseudo_labels": None,
                    "checkpoint_path": None,
                }
            )

        # Store validation pseudo labels
        self._iteration_results[-1]["val_pseudo_labels"] = (
            val_pseudo_labels.cpu().numpy() if val_pseudo_labels is not None else None
        )

        # Compute validation pseudo-label metrics
        if iteration > 1 and len(self._iteration_results) >= 2:
            prev_val_pseudo_labels = self._iteration_results[-2].get(
                "val_pseudo_labels"
            )
            if prev_val_pseudo_labels is not None:
                num_expected_nodes = get_num_nodes(iteration, self.config.num_classes)
                val_pseudo_metrics = compute_pseudo_label_metrics(
                    prev_val_pseudo_labels,
                    val_hierarchical_preds.cpu().numpy(),
                    num_expected_nodes,
                )
                self._iteration_results[-1]["val_pseudo_metrics"] = val_pseudo_metrics

    def _compute_weight_changes(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute weight change statistics between iterations.
        Args:
            model: Current model after training
        Returns:
            Dictionary with weight change statistics
        """
        curr_weights = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if "weight" in k
        }

        if self._prev_weights is None:
            self._prev_weights = curr_weights
            return {}

        backbone_changes = []
        head_changes = []

        for key in curr_weights.keys():
            # Skip if key not in previous or shape mismatch (happens when heads expand)
            if (
                key not in self._prev_weights
                or self._prev_weights[key].shape != curr_weights[key].shape
            ):
                continue

            prev_w = self._prev_weights[key]
            curr_w = curr_weights[key]
            diff = curr_w - prev_w

            mean_abs_change = torch.mean(torch.abs(diff)).item()
            relative_change = (torch.norm(diff) / torch.norm(prev_w)).item()

            if "base." in key:
                backbone_changes.append((mean_abs_change, relative_change))
            elif "output_heads." in key:
                head_changes.append((mean_abs_change, relative_change))

        stats = {}
        if backbone_changes:
            stats["backbone_mean_abs_change"] = np.mean(
                [x[0] for x in backbone_changes]
            )
            stats["backbone_relative_change"] = np.mean(
                [x[1] for x in backbone_changes]
            )
            stats["backbone_relative_change"] = np.mean(
                [x[1] for x in backbone_changes]
            )
        if head_changes:
            stats["heads_mean_abs_change"] = np.mean([x[0] for x in head_changes])
            stats["heads_relative_change"] = np.mean([x[1] for x in head_changes])

        self._prev_weights = curr_weights

        return stats
