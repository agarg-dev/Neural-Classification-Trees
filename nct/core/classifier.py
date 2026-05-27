"""
Neural Classification Tree (NCT) Classifier.

Main interface for hierarchical image classification using iterative training.
"""

import copy
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ..data.datasets import IterativeDataset, extract_labels_from_dataset
from ..data.loaders import (
    create_data_loader,
    create_node_upsample_sampler,
    create_node_downsample_sampler,
    create_node_geomean_sampler,
)
from ..evaluation.analysis import analyze_hierarchical_nodes, compute_head_calibration_diagnostics
from ..evaluation.metrics import (
    calculate_basic_metrics,
    calculate_group_metrics,
    calculate_hierarchical_metrics,
    calculate_subgroup_auroc,
    compute_consolidated_group_metrics,
    compute_iteration_transitions,
    compute_pseudo_label_metrics,
    get_worst_pseudo_node_stats,
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

        # Store original paths before any modifications (for safe re-fitting)
        self._base_checkpoint_dir = self.config.checkpoint_dir
        self._base_attribution_dir = self.config.attribution_dir

        # Initialize logger
        self.logger = NCTLogger(verbose=self.config.verbose, config=self.config)
        self._iteration_results = []
        self._prev_weights = None  # For weight change tracking
        self._trained_nodes_mask = (
            None  # Mask for nodes with training samples (used at inference)
        )
        self._depth_selection_state = self._new_depth_selection_state()

    def fit(
        self,
        X_train: Any,
        X_val: Any,
        X_test: Optional[Any] = None,
        group_labels: Optional[Any] = None,
        val_group_labels: Optional[Any] = None,
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
            test_subgroups: Optional subgroup masks for AUROC metrics.

        Returns:
            Dict with 'classifier', 'val_pseudo_metrics', 'val_pseudo_metrics_per_iter',
            and 'depth_selection'.
        """

        if not isinstance(X_train, TorchDataset):
            raise NotImplementedError(
                "Direct tensor/array inputs not supported. Please pass PyTorch Dataset objects."
            )

        # Store class and group names for later use
        self._class_names = class_names
        self._group_names = group_names
        self._test_subgroups = test_subgroups

        # Namespace outputs by {base_dir}/{dataset_name}/seed_{seed}/ so re-fitting
        # doesn't nest paths.
        checkpoint_dir = self._base_checkpoint_dir
        attribution_dir = self._base_attribution_dir
        if dataset_name and dataset_name != "generic":
            checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
            attribution_dir = os.path.join(attribution_dir, dataset_name)
        if self.config.seed is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, f"seed_{self.config.seed}")
            attribution_dir = os.path.join(attribution_dir, f"seed_{self.config.seed}")
        self.config.checkpoint_dir = checkpoint_dir
        self.config.attribution_dir = attribution_dir
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.attribution_dir, exist_ok=True)

        self._depth_selection_state = self._new_depth_selection_state()
        self._depth_selection_state["metric"] = self._effective_depth_selection_metric()

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
            val_group_labels=val_group_labels,
        )
        self._is_fitted = True

        # Defer wandb finish if attributions will be logged after fit.
        finish_wandb_now = not self.config.log_attributions
        self.logger.training_end(finish_wandb=finish_wandb_now)

        val_pseudo_metrics_per_iter = [
            r.get("val_pseudo_metrics") for r in self._iteration_results
        ]
        val_pseudo_metrics = val_pseudo_metrics_per_iter[-1] if val_pseudo_metrics_per_iter else None

        return {
            "classifier": self,
            "val_pseudo_metrics": val_pseudo_metrics,
            "val_pseudo_metrics_per_iter": val_pseudo_metrics_per_iter,
            "depth_selection": copy.deepcopy(self._depth_selection_state),
        }

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict class labels. X must be a PyTorch Dataset.

        Returns predictions as a numpy array of shape (n_samples,).
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
        Predict class probabilities. Returns shape (n_samples, num_classes).
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        test_loader = self._create_iterative_loader(X, is_training=False)
        hierarchical_probs = self._get_hierarchical_probabilities(test_loader)

        # Sum probabilities for nodes belonging to each class.
        class_probs = []
        for class_id in range(self.config.num_classes):
            start_node, end_node = get_class_node_range(
                class_id, self.config.num_iterations
            )
            class_raw = hierarchical_probs[:, start_node:end_node].sum(dim=1)
            class_probs.append(class_raw)

        # Normalize across classes
        total = sum(class_probs)
        class_probs = [p / total for p in class_probs]

        return torch.stack(class_probs, dim=1).numpy()

    def _create_iterative_loader(
        self,
        dataset: Any,
        is_training: bool,
        sampling_strategy: str = "none",
    ) -> DataLoader:
        """
        Create a DataLoader for iterative dataset.
        Args:
            dataset: Input dataset (IterativeDataset).
            is_training: Whether this is for training or evaluation.
            sampling_strategy: Node sampling strategy ("none", "upsample",
                "downsample", "geomean"). Only applied when is_training=True.
        Returns:
            DataLoader configured for iterative training/evaluation.
        """
        if is_training:
            if sampling_strategy == "downsample":
                sampler = create_node_downsample_sampler(
                    dataset.hier_labels, generator=self.generator
                )
            elif sampling_strategy == "upsample":
                sampler = create_node_upsample_sampler(
                    dataset.hier_labels, generator=self.generator
                )
            elif sampling_strategy == "geomean":
                sampler = create_node_geomean_sampler(
                    dataset.hier_labels, generator=self.generator
                )
            else:  # "none"
                return create_data_loader(
                    dataset,
                    is_training=True,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    generator=self.generator,
                )
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=True,
                persistent_workers=self.config.num_workers > 0,
                prefetch_factor=4 if self.config.num_workers > 0 else None,
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
            Dict with 'basic_metrics', and (if group_labels given) 'group_metrics',
            'consolidated_group_metrics', and 'node_analysis'.
        """

        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        # Fall back to values stored from fit() if caller didn't pass them.
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

        y_test = extract_labels_from_dataset(X_test)
        y_test_np = y_test.numpy()
        # AUROC takes prob of class 1 for binary; full probability matrix for multi-class.
        y_prob_for_auroc = (
            binary_probabilities[:, 1]
            if self.config.num_classes == 2
            else binary_probabilities
        )
        basic_metrics = calculate_basic_metrics(
            y_test_np, binary_predictions, y_prob=y_prob_for_auroc
        )

        subgroup_auroc = calculate_subgroup_auroc(
            y_test_np, y_prob_for_auroc, test_subgroups
        )
        if subgroup_auroc:
            basic_metrics.update(subgroup_auroc)

        results = {"basic_metrics": basic_metrics}

        if group_labels is not None:
            group_labels_np = np.array(group_labels)
            group_metrics = calculate_group_metrics(
                y_test_np,
                binary_predictions,
                group_labels_np,
                y_prob_for_auroc,
                group_names,
            )
            results["group_metrics"] = group_metrics

            consolidated_metrics = compute_consolidated_group_metrics(
                group_labels_np,
                binary_predictions,
                num_classes=self.config.num_classes,
                num_groups=len(np.unique(group_labels_np)),
            )
            if consolidated_metrics is not None:
                results["consolidated_group_metrics"] = consolidated_metrics

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
                num_classes=self.config.num_classes,
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

        if len(self._iteration_results) == 0 and self.model is None:
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
            attribution_method="layer_gradcam",  # overridden by `methods` arg below
        )

        visualizer.generate_all_visualizations(
            dataset=X_test,
            iteration_results=self._iteration_results,
            save_dir=save_dir,
            num_per_node=num_per_node,
            num_transitions=num_transitions,
            seed=self.config.seed,
            methods=methods,
        )

        if self.config.log_attributions:
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
                methods_to_log = ["layer_gradcam"]

            for method_name in methods_to_log:
                method_save_dir = os.path.join(save_dir, method_name)

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

            self.logger.finish_wandb()

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
        val_group_labels: Optional[Any] = None,
    ) -> nn.Module:
        """Run the iterative training loop that builds the hierarchical tree."""

        current_base_lr = self.config.initial_base_lr
        model = None

        self.logger.training_start(self.config.to_dict())

        trainer = NCTTrainer(
            None, self.device, self.config.to_dict(), logger=self.logger
        )
        # val_group_labels enables per-epoch true-WGA tracking for Spearman correlation
        if val_group_labels is not None:
            trainer.val_group_labels = np.asarray(val_group_labels)

        last_trained_iter = 0

        for i in range(self.config.num_iterations):
            num_nodes = get_num_nodes(i + 1, self.config.num_classes)
            epochs_for_iter = self.config.epochs_per_iteration[i]

            trainer.iteration_num = i + 1

            # upsample/downsample/geomean already balance nodes via sampling, so skip class
            # weights. skip_class_weights_iter1 skips them in iter 1 to keep failures systematic.
            skip_weights = (
                (i > 0 and self.config.sampling_strategy in ("upsample", "downsample", "geomean"))
                or (i == 0 and self.config.skip_class_weights_iter1)
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

            # Updated each iteration; final iteration's mask is used at inference.
            node_counts = torch.bincount(current_train_labels, minlength=num_nodes)
            self._trained_nodes_mask = node_counts > 0

            self.logger.iteration_start(
                i + 1,
                self.config.num_iterations,
                num_nodes,
                epochs_for_iter,
                class_weights,
                node_counts,  # raw counts (including zeros) for the log table
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
                head_type=self.config.head_type,
                iter1_head_type=self.config.iter1_head_type,
                split_node_head_type=self.config.split_node_head_type,
                head_bn_affine=self.config.head_bn_affine,
                num_channels=self.config.num_channels,
                image_size=self.config.image_size,
            )

            train_iter_dataset = IterativeDataset(
                base_train_dataset, current_train_labels
            )
            val_iter_dataset = IterativeDataset(base_val_dataset, current_val_labels)

            # Iteration 1 (i=0) always uses standard sampling regardless of setting.
            effective_strategy = self.config.sampling_strategy if i > 0 else "none"

            train_iter_loader = self._create_iterative_loader(
                train_iter_dataset,
                is_training=True,
                sampling_strategy=effective_strategy,
            )

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

            # Update labels before test eval (sets merged_nodes needed for test pseudo labels)
            if i < self.config.num_iterations - 1:
                current_train_labels, current_val_labels, val_preds = (
                    self._update_labels_for_next_iteration(
                        model,
                        base_train_dataset,
                        base_val_dataset,
                        current_train_labels,
                        current_val_labels,
                        current_iteration=i + 1,
                    )
                )

                # Store validation metrics (with probs for AUROC computation)
                _, _, val_probs = self._evaluate_for_routing(
                    base_val_dataset, current_val_labels, model, return_probs=True
                )
                self._evaluate_validation_at_iteration(
                    val_preds,
                    current_val_labels,
                    i + 1,
                    routing_probs=val_probs,
                )

                current_base_lr /= self.config.lr_decay_factor

            # Evaluate test set
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

            if i == self.config.num_iterations - 1:
                # For final iteration, still evaluate validation to compute pseudo-label metrics
                # This is needed for Optuna optimization which uses final iteration metrics
                val_preds, _, val_probs = self._evaluate_for_routing(
                    base_val_dataset, current_val_labels, model, return_probs=True
                )
                self._evaluate_validation_at_iteration(
                    val_preds,
                    None,  # No pseudo labels needed (no next iteration)
                    i + 1,
                    routing_probs=val_probs,
                )

            last_trained_iter = i + 1
            should_stop = self._apply_depth_selection_check(iteration=i + 1)
            if should_stop and self.config.depth_selection_enabled:
                self._depth_selection_state["early_stopped"] = True
                break

        # Spearman of pseudo-WGA vs true WGA; consolidated variant uses bias-collapsed true (CMNIST only)
        try:
            from scipy.stats import spearmanr as _spearmanr
        except Exception:
            _spearmanr = None
        metric_label = "AUROC" if self._effective_depth_selection_metric() == "auroc" else "WGA"
        for corr_data, variant in (
            (trainer._wga_correlation_data, "full"),
            (trainer._wga_correlation_data_consolidated, "consolidated"),
        ):
            if _spearmanr is None or len(corr_data) < 3:
                continue
            try:
                pseudo_vals = [x[0] for x in corr_data]
                true_vals = [x[1] for x in corr_data]
                spearman_r, _ = _spearmanr(pseudo_vals, true_vals)
                if not (spearman_r != spearman_r):
                    self.logger.log_wga_spearman(
                        float(spearman_r), len(corr_data), variant=variant,
                        pseudo_metric=metric_label, true_metric=metric_label,
                    )
            except Exception:
                pass

        self._finalize_depth_selection(last_trained_iter=last_trained_iter)

        return model

    def _update_labels_for_next_iteration(
        self,
        model,
        base_train_dataset: Any,
        base_val_dataset: Any,
        train_labels: torch.Tensor,
        val_labels: torch.Tensor,
        current_iteration: int,
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
            current_iteration: The iteration that just finished training (1-indexed).

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
        is_difficult_train = self._determine_difficult_samples(is_correct_train)
        is_difficult_val = self._determine_difficult_samples(is_correct_val)

        # Log section header for pseudo-label generation (node merges logged below)
        if self.config.min_samples_per_node > 0:
            self.logger.log(
                f"\n--- Preparing Pseudo-Labels for Iteration {current_iteration + 1} ---",
                level=2,
            )

        # Compute proposed labels (before merging)
        train_proposed = self._compute_proposed_labels(train_labels, is_difficult_train)
        val_proposed = self._compute_proposed_labels(val_labels, is_difficult_val)

        # Determine which nodes to merge based on TRAINING counts only
        merged_nodes = self._determine_nodes_to_merge(train_proposed)
        self._merged_nodes_for_iteration = merged_nodes  # Store for test consistency

        # Apply same merges to both train and val
        new_train_labels = self._apply_node_merges(
            train_proposed, merged_nodes, "train"
        )
        new_val_labels = self._apply_node_merges(val_proposed, merged_nodes, "val")

        # Return validation predictions for pseudo-label metric computation
        return new_train_labels, new_val_labels, val_preds

    def _evaluate_for_routing(
        self,
        base_dataset: Any,
        labels: torch.Tensor,
        model: nn.Module,
        return_probs: bool = False,
    ):
        """
        Get predictions and correctness for any single dataset.

        Args:
            base_dataset: Base dataset (regular PyTorch Dataset)
            labels: Labels for the dataset
            model: Current trained model
            return_probs: If True, also return softmax routing probabilities.

        Returns:
            Tuple of (predictions, is_correct) or (predictions, is_correct, probs) if return_probs=True.
        """
        eval_dataset = IterativeDataset(base_dataset, labels)
        loader = self._create_iterative_loader(eval_dataset, is_training=False)
        if return_probs:
            predictions, probs = self._get_hierarchical_predictions(loader, model=model, return_probs=True)
            is_correct = predictions == labels
            return predictions, is_correct, probs
        predictions = self._get_hierarchical_predictions(loader, model=model)
        is_correct = predictions == labels
        return predictions, is_correct

    def _compute_proposed_labels(
        self,
        labels: torch.Tensor,
        is_difficult: torch.Tensor,
    ) -> torch.Tensor:
        """Pseudo labels for next iteration: labels * 2 + is_difficult."""
        return labels * 2 + is_difficult.long()

    def _determine_nodes_to_merge(
        self,
        proposed_labels: torch.Tensor,
    ) -> list:
        """Difficult nodes (odd indices) below min_samples_per_node, computed from training counts only.

        The returned list is then applied consistently to val and test so all
        three splits share the same node structure.
        """
        merged_nodes = []
        min_samples = self.config.min_samples_per_node

        if min_samples > 0:
            num_nodes = proposed_labels.max().item() + 1
            counts = torch.bincount(proposed_labels, minlength=num_nodes)

            for difficult_node in range(1, num_nodes, 2):
                if counts[difficult_node] < min_samples:
                    merged_nodes.append(difficult_node)

        return merged_nodes

    def _apply_node_merges(
        self,
        proposed_labels: torch.Tensor,
        merged_nodes: list,
        dataset_name: str = "data",
    ) -> torch.Tensor:
        """Merge each difficult node (odd index) into its easy sibling (node - 1)."""
        for difficult_node in merged_nodes:
            mask = proposed_labels == difficult_node
            num_merged = mask.sum().item()
            if num_merged > 0:
                proposed_labels[mask] = difficult_node - 1
                if self.config.verbose >= 2:
                    self.logger.log(
                        f"    Merged node {difficult_node} ({num_merged} samples) "
                        f"into node {difficult_node - 1} [{dataset_name}]",
                        level=2,
                    )

        return proposed_labels

    def _determine_difficult_samples(self, is_correct: torch.Tensor) -> torch.Tensor:
        """Determine which samples should be routed to 'difficult' child nodes."""
        return ~is_correct

    def _get_hierarchical_predictions(
        self, dataloader: DataLoader, model: Optional[nn.Module] = None,
        return_probs: bool = False,
    ):
        """Argmax over hierarchical nodes. Untrained nodes (per _trained_nodes_mask) are masked to -inf."""
        if model is None:
            model = self.model

        model.eval()
        all_predictions = []
        all_probs = [] if return_probs else None

        apply_mask = (
            self.config.min_samples_per_node > 0
            and self._trained_nodes_mask is not None
        )

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = model(images)

                if apply_mask:
                    mask = self._trained_nodes_mask.to(outputs.device)
                    if len(mask) == outputs.shape[1]:
                        outputs = outputs.clone()
                        outputs[:, ~mask] = float("-inf")

                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())

                if return_probs:
                    all_probs.append(torch.softmax(outputs, dim=1).cpu())

        predictions = torch.cat(all_predictions)
        if return_probs:
            return predictions, torch.cat(all_probs)
        return predictions

    def _get_raw_logits(
        self, dataloader: DataLoader, model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Collect raw model logits (before any activation) for all samples.

        Used for calibration diagnostics — logits reflect true head scales.
        """
        if model is None:
            model = self.model
        model.eval()
        all_logits = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = model(images)
                all_logits.append(outputs.cpu())
        return torch.cat(all_logits)

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

        use_softmax = self.config.num_classes > 2

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = model(images)

                if use_softmax:
                    # Multi-class: mask logits BEFORE softmax (set to -inf)
                    if apply_mask:
                        mask = self._trained_nodes_mask.to(outputs.device)
                        if len(mask) == outputs.shape[1]:
                            outputs = outputs.clone()
                            outputs[:, ~mask] = float("-inf")
                    probabilities = torch.softmax(outputs, dim=1)
                else:
                    # Binary: independent sigmoid per head (original behavior)
                    probabilities = torch.sigmoid(outputs)
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
        """Evaluate test set after the given iteration; store metrics into self._iteration_results."""
        if self.config.verbose == 0:
            return

        entry_idx = iteration - 1

        # Get test predictions and probabilities
        test_loader = self._create_iterative_loader(X_test, is_training=False)
        hierarchical_predictions = self._get_hierarchical_predictions(
            test_loader, model=model
        )
        class_probabilities = self._get_class_probabilities(
            test_loader, iteration, model=model
        )
        y_test = extract_labels_from_dataset(X_test)

        class_predictions = node_to_class(
            hierarchical_predictions, iteration, self.config.num_classes
        )

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
                    self._iteration_results[entry_idx - 1]["test_pseudo_labels"]
                )
            else:
                current_test_labels = y_test

            is_correct = hierarchical_predictions == current_test_labels
            is_difficult = self._determine_difficult_samples(is_correct)
            # Compute proposed labels and apply same merges as training
            test_proposed = self._compute_proposed_labels(
                current_test_labels, is_difficult
            )
            merged_nodes = getattr(self, "_merged_nodes_for_iteration", [])
            test_pseudo_labels = self._apply_node_merges(
                test_proposed, merged_nodes, dataset_name="test"
            )

        # Pad results list to the right index then update (preserves any existing validation entry).
        while len(self._iteration_results) <= entry_idx:
            self._iteration_results.append({})

        self._iteration_results[entry_idx].update(
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
            and self._iteration_results[entry_idx - 1].get("test_pseudo_labels") is not None
        ):
            prev_pseudo_labels = self._iteration_results[entry_idx - 1]["test_pseudo_labels"]
            curr_routing = self._iteration_results[entry_idx]["hierarchical_predictions"]
            num_expected_nodes = get_num_nodes(iteration, self.config.num_classes)

            # Routing probabilities for pseudo AUROC (only needed under iter2_auroc criterion).
            routing_probs = None
            if self.config.model_selection_criterion == "iter2_auroc":
                raw_logits = self._get_raw_logits(test_loader, model=model)
                routing_probs = torch.softmax(raw_logits, dim=1).numpy()

            pseudo_metrics = compute_pseudo_label_metrics(
                prev_pseudo_labels,
                curr_routing,
                num_expected_nodes,
                min_samples_threshold=self.config.min_samples_per_node,
                base_num_classes=self.config.num_classes,
                routing_probs=routing_probs,
            )

        # Compute test metrics (AUROC uses probability of positive class)
        basic_metrics = calculate_basic_metrics(
            y_test_np, class_predictions_np, y_prob=class_prob_for_auroc
        )
        group_results = None
        consolidated_group_results = None
        if group_labels_np is not None:
            group_results = calculate_group_metrics(
                y_test_np,
                class_predictions_np,
                group_labels_np,
                class_prob_for_auroc,
                group_names,
            )
            consolidated_group_results = compute_consolidated_group_metrics(
                group_labels_np,
                class_predictions_np,
                num_classes=self.config.num_classes,
                num_groups=len(np.unique(group_labels_np)),
            )
        node_metrics = calculate_hierarchical_metrics(
            hierarchical_predictions_np, iteration, y_test_np, class_predictions_np
        )

        # Subgroup AUROC if test_subgroups provided (e.g., ISIC patch/histopathology splits)
        subgroup_auroc = calculate_subgroup_auroc(
            y_test_np, class_prob_for_auroc, test_subgroups
        )
        if subgroup_auroc:
            basic_metrics.update(subgroup_auroc)

        self.logger.log_evaluation_results(basic_metrics, pseudo_metrics, group_results, consolidated_group_results, class_names=self._class_names)

        # Track transitions between iterations
        if iteration > 1:
            prev_result = self._iteration_results[entry_idx - 1]
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

        # Detailed node analysis (verbose only, runs last)
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
                num_classes=self.config.num_classes,
            )

        # Head calibration diagnostics (verbose >= 2 only)
        if self.config.verbose >= 2:
            raw_logits = self._get_raw_logits(test_loader, model=model).numpy()
            # Apply trained-nodes mask so margin/stats match the effective prediction
            # space (same mask that _get_hierarchical_predictions uses for argmax).
            if (
                self.config.min_samples_per_node > 0
                and self._trained_nodes_mask is not None
                and len(self._trained_nodes_mask) == raw_logits.shape[1]
            ):
                mask_np = self._trained_nodes_mask.cpu().numpy()
                effective_logits = raw_logits.copy()
                effective_logits[:, ~mask_np] = -np.inf
            else:
                effective_logits = raw_logits
            # For iter 1: true node = class label. For iter 2+: use prev pseudo-labels.
            if iteration == 1:
                true_nodes = y_test_np
            elif entry_idx > 0 and self._iteration_results[entry_idx - 1].get("test_pseudo_labels") is not None:
                true_nodes = self._iteration_results[entry_idx - 1]["test_pseudo_labels"]
            else:
                true_nodes = None
            compute_head_calibration_diagnostics(
                logits=effective_logits,
                hierarchical_predictions=hierarchical_predictions_np,
                num_iterations=iteration,
                class_names=self._class_names if hasattr(self, "_class_names") else None,
                num_classes=self.config.num_classes,
                true_node_labels=true_nodes,
                verbose=True,
            )

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
            if "iter2_level_worst_group_accuracy" in pseudo_metrics:
                basic_metrics["pseudo_iter2_level_worst_group_acc"] = pseudo_metrics[
                    "iter2_level_worst_group_accuracy"
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
            consolidated_group_metrics=consolidated_group_results,
        )

    @staticmethod
    def _new_depth_selection_state() -> Dict[str, Any]:
        return {
            "best_iter": None,
            "best_value": None,
            "best_n_worst": None,
            "per_iter": {},
            "stop_reason": None,
            "selected_iter": None,
            "headline_iter": None,
            "early_stopped": False,
            "metric": "wga",
        }

    def _effective_depth_selection_metric(self) -> str:
        return "auroc" if self.config.model_selection_criterion == "iter2_auroc" else "wga"

    def _apply_depth_selection_check(self, iteration: int) -> bool:
        state = self._depth_selection_state
        metric = state["metric"]
        entry_idx = iteration - 1

        if entry_idx >= len(self._iteration_results):
            return False
        val_pseudo_metrics = self._iteration_results[entry_idx].get("val_pseudo_metrics")

        if iteration == 1 or val_pseudo_metrics is None:
            state["per_iter"][iteration] = {
                "value": None, "n_worst": None, "epsilon": None, "decision": "bootstrap",
            }
            self.logger.log_depth_decision(
                iteration=iteration, value=None, n_worst=None,
                best_value=None, best_iter=None, epsilon=None,
                decision="bootstrap", metric=metric,
            )
            return False

        value, n_worst = get_worst_pseudo_node_stats(val_pseudo_metrics, metric=metric)
        if value is None or n_worst is None or n_worst <= 0:
            state["per_iter"][iteration] = {
                "value": value, "n_worst": n_worst, "epsilon": None, "decision": "skip",
            }
            self.logger.log_depth_decision(
                iteration=iteration, value=value, n_worst=n_worst,
                best_value=state["best_value"], best_iter=state["best_iter"],
                epsilon=None, decision="skip", metric=metric,
            )
            return False

        if state["best_iter"] is None:
            state["best_iter"] = iteration
            state["best_value"] = value
            state["best_n_worst"] = n_worst
            state["per_iter"][iteration] = {
                "value": value, "n_worst": n_worst, "epsilon": None, "decision": "initial_best",
            }
            self.logger.log_depth_decision(
                iteration=iteration, value=value, n_worst=n_worst,
                best_value=None, best_iter=None, epsilon=None,
                decision="initial_best", metric=metric,
            )
            return False

        best_value = state["best_value"]
        best_n_worst = state["best_n_worst"]
        z = self.config.depth_selection_confidence
        variance = max(best_value * (1.0 - best_value), 0.0)
        epsilon = z * math.sqrt(variance / max(best_n_worst, 1))

        if value > best_value:
            state["best_iter"] = iteration
            state["best_value"] = value
            state["best_n_worst"] = n_worst
            decision = "new_best"
            should_stop = False
        elif value >= best_value - epsilon:
            decision = "within_epsilon"
            should_stop = False
        else:
            decision = "exceeded_epsilon"
            should_stop = True
            state["stop_reason"] = (
                f"iter {iteration} pseudo_{metric}={value:.4f} dropped "
                f"below best (iter {state['best_iter']}, pseudo_{metric}={best_value:.4f}) "
                f"by more than epsilon={epsilon:.4f}"
            )

        state["per_iter"][iteration] = {
            "value": value, "n_worst": n_worst, "epsilon": epsilon, "decision": decision,
        }
        self.logger.log_depth_decision(
            iteration=iteration, value=value, n_worst=n_worst,
            best_value=best_value, best_iter=state["best_iter"],
            epsilon=epsilon, decision=decision, metric=metric,
        )
        return should_stop

    def _finalize_depth_selection(self, last_trained_iter: int) -> None:
        state = self._depth_selection_state
        enabled = self.config.depth_selection_enabled

        state["selected_iter"] = (
            state["best_iter"] if state["best_iter"] is not None else last_trained_iter
        )
        state["headline_iter"] = (
            state["selected_iter"] if enabled else last_trained_iter
        )

        if state["stop_reason"] is None:
            state["stop_reason"] = (
                f"reached num_iterations={self.config.num_iterations}"
                if last_trained_iter == self.config.num_iterations
                else f"training ended at iter {last_trained_iter}"
            )

        self.logger.log_depth_final_summary(
            enabled=enabled,
            selected_iter=state["selected_iter"],
            headline_iter=state["headline_iter"],
            stop_reason=state["stop_reason"],
            per_iter_stats=state["per_iter"],
            last_trained_iter=last_trained_iter,
            num_iterations=self.config.num_iterations,
            early_stopped=state["early_stopped"],
            metric=state["metric"],
        )

    def _evaluate_validation_at_iteration(
        self,
        val_hierarchical_preds: torch.Tensor,
        val_pseudo_labels: Optional[torch.Tensor],
        iteration: int,
        routing_probs: Optional[torch.Tensor] = None,
    ) -> None:
        """Store validation pseudo labels and compute val pseudo-label metrics for this iteration."""
        entry_idx = iteration - 1

        while len(self._iteration_results) <= entry_idx:
            self._iteration_results.append({})

        self._iteration_results[entry_idx]["val_pseudo_labels"] = (
            val_pseudo_labels.cpu().numpy() if val_pseudo_labels is not None else None
        )

        # Compute validation pseudo-label metrics (need previous iter's labels to compare against)
        if iteration > 1:
            prev_val_pseudo_labels = self._iteration_results[entry_idx - 1].get(
                "val_pseudo_labels"
            )
            if prev_val_pseudo_labels is not None:
                num_expected_nodes = get_num_nodes(iteration, self.config.num_classes)
                val_pseudo_metrics = compute_pseudo_label_metrics(
                    prev_val_pseudo_labels,
                    val_hierarchical_preds.cpu().numpy(),
                    num_expected_nodes,
                    min_samples_threshold=self.config.min_samples_per_node,
                    base_num_classes=self.config.num_classes,
                    routing_probs=routing_probs.numpy() if routing_probs is not None else None,
                )
                self._iteration_results[entry_idx]["val_pseudo_metrics"] = val_pseudo_metrics

    def _compute_weight_changes(self, model: nn.Module) -> Dict[str, float]:
        """Mean-abs and relative weight change vs the previous iteration, split into backbone/heads."""
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
            # Skip if key missing or shape changed (happens when heads expand each iteration).
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
