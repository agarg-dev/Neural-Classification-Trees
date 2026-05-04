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
    NodeHybridSampler,
    create_node_hybrid_sampler,
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

        self.generator = torch.Generator()
        if self.config.seed is not None:
            self.generator.manual_seed(self.config.seed)

        self.model = None
        self._is_fitted = False

        self._base_checkpoint_dir = self.config.checkpoint_dir
        self._base_attribution_dir = self.config.attribution_dir

        self.logger = NCTLogger(verbose=self.config.verbose, config=self.config)
        self._iteration_results = []
        self._prev_weights = None
        self._trained_nodes_mask = None
        self._depth_selection_state = self._new_depth_selection_state()

    # ============================================================================
    # PUBLIC API
    # ============================================================================

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

        self._class_names = class_names
        self._group_names = group_names
        self._test_subgroups = test_subgroups

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

        class_probs = []
        for class_id in range(self.config.num_classes):
            start_node, end_node = get_class_node_range(
                class_id, self.config.num_iterations
            )
            class_raw = hierarchical_probs[:, start_node:end_node].sum(dim=1)
            class_probs.append(class_raw)

        total = sum(class_probs)
        class_probs = [p / total for p in class_probs]

        return torch.stack(class_probs, dim=1).numpy()

    # ============================================================================
    # DATA UTILITIES - DataLoader Creation
    # ============================================================================

    def _create_iterative_loader(
        self,
        dataset: Any,
        is_training: bool,
        sampling_strategy: str = "none",
        hybrid_easy_target: Optional[int] = None,
        hybrid_diff_target: Optional[int] = None,
    ) -> DataLoader:
        """Create a DataLoader for iterative dataset."""
        if is_training:
            if sampling_strategy == "downsample":
                sampler = create_node_downsample_sampler(
                    dataset.hier_labels, generator=self.generator
                )
            elif sampling_strategy == "upsample":
                sampler = create_node_upsample_sampler(
                    dataset.hier_labels, generator=self.generator
                )
            elif sampling_strategy == "hybrid":
                sampler = create_node_hybrid_sampler(
                    dataset.hier_labels,
                    generator=self.generator,
                    easy_target=hybrid_easy_target,
                    diff_target=hybrid_diff_target,
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

    # ============================================================================
    # HEAD ENSEMBLE UTILITIES
    # ============================================================================

    def _reinitialize_output_heads(self, model: nn.Module) -> None:
        """Reinitialize output_heads weights in-place using default Linear init."""
        for head in model.output_heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def _average_output_head_states(
        self, state_dicts: list, model: nn.Module
    ) -> None:
        """Average output_heads weights across state dicts and load into model."""
        head_keys = [k for k in state_dicts[0] if k.startswith("output_heads.")]
        avg_state = copy.deepcopy(model.state_dict())
        for key in head_keys:
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            avg_state[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
        model.load_state_dict(avg_state)

    def _run_head_ensemble(
        self,
        trainer: "NCTTrainer",
        model: nn.Module,
        train_iter_dataset: Any,
        val_iter_loader: DataLoader,
        phase1_epochs: int,
        head_lr: float,
        class_weights: Optional[torch.Tensor],
        val_class_weights: Optional[torch.Tensor],
        sampling_strategy: str,
        hybrid_easy_target: Optional[int] = None,
        hybrid_diff_target: Optional[int] = None,
    ) -> None:
        """Run Phase 1 head_ensemble_size times, then average output_heads weights in-place."""
        ensemble_size = self.config.head_ensemble_size
        collected_states = []

        if self.config.verbose >= 1:
            print(f"  [Ensemble] Running {ensemble_size} Phase 1 runs for head averaging...")

        for run_idx in range(ensemble_size):
            self._reinitialize_output_heads(model)

            run_loader = self._create_iterative_loader(
                train_iter_dataset,
                is_training=True,
                sampling_strategy=sampling_strategy,
                hybrid_easy_target=hybrid_easy_target,
                hybrid_diff_target=hybrid_diff_target,
            )

            trainer.train_phase1_only(
                run_loader,
                val_iter_loader,
                phase1_epochs,
                head_lr,
                class_weights,
                val_class_weights,
            )

            collected_states.append(copy.deepcopy(model.state_dict()))

            if self.config.verbose >= 1:
                print(f"  [Ensemble] Run {run_idx + 1}/{ensemble_size} complete")

        self._average_output_head_states(collected_states, model)

        if self.config.verbose >= 1:
            print(f"  [Ensemble] Averaged {ensemble_size} head ensemble runs")

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

        if len(self._iteration_results) == 0:
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
            attribution_method="layer_gradcam",
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
        val_group_labels: Optional[Any] = None,
    ) -> nn.Module:
        """Execute the main iterative training loop."""

        current_base_lr = self.config.initial_base_lr
        model = None

        self.logger.training_start(self.config.to_dict())

        trainer = NCTTrainer(
            None, self.device, self.config.to_dict(), logger=self.logger
        )
        if val_group_labels is not None:
            trainer.val_group_labels = np.asarray(val_group_labels)

        last_trained_iter = 0

        for i in range(self.config.num_iterations):
            num_nodes = get_num_nodes(i + 1, self.config.num_classes)
            epochs_for_iter = self.config.epochs_per_iteration[i]

            trainer.iteration_num = i + 1

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

            node_counts = torch.bincount(current_train_labels, minlength=num_nodes)
            self._trained_nodes_mask = node_counts > 0

            self.logger.iteration_start(
                i + 1,
                self.config.num_iterations,
                num_nodes,
                epochs_for_iter,
                class_weights,
                node_counts,
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

            effective_strategy = self.config.sampling_strategy if i > 0 else "none"

            hybrid_easy_target = None
            hybrid_diff_target = None
            if effective_strategy == "hybrid":
                if i == 1:
                    _probe = NodeHybridSampler(train_iter_dataset.hier_labels)
                    self._hybrid_easy_target: int = _probe._easy_target
                    self._hybrid_diff_target: int = _probe._diff_target
                hybrid_easy_target = getattr(self, "_hybrid_easy_target", None)
                hybrid_diff_target = getattr(self, "_hybrid_diff_target", None)

                if class_weights is not None:
                    _cw_probe = NodeHybridSampler(
                        train_iter_dataset.hier_labels,
                        easy_target=hybrid_easy_target,
                        diff_target=hybrid_diff_target,
                    )
                    eff = torch.zeros(num_nodes)
                    for n in _cw_probe._easy_node_ids:
                        eff[n] = min(int(node_counts[n].item()), _cw_probe._easy_target)
                    for n in _cw_probe._diff_node_ids:
                        eff[n] = _cw_probe._diff_target
                    valid = eff > 0
                    if valid.sum() > 0:
                        total_eff = eff[valid].sum()
                        n_valid = int(valid.sum().item())
                        cw = torch.zeros(num_nodes)
                        cw[valid] = total_eff / (n_valid * eff[valid])
                        if self.config.max_class_weight > 0:
                            cw[valid] = cw[valid].clamp(max=float(self.config.max_class_weight))
                        class_weights = cw
                    else:
                        class_weights = None

            train_iter_loader = self._create_iterative_loader(
                train_iter_dataset,
                is_training=True,
                sampling_strategy=effective_strategy,
                hybrid_easy_target=hybrid_easy_target,
                hybrid_diff_target=hybrid_diff_target,
            )

            val_iter_loader = self._create_iterative_loader(
                val_iter_dataset, is_training=False
            )

            trainer.model = model

            if self.config.head_ensemble_size > 1 and i > 0:
                cw_device = class_weights.to(self.device) if class_weights is not None else None
                val_cw = torch.ones_like(cw_device) if cw_device is not None else None
                phase1_epochs = int(epochs_for_iter * self.config.phase1_epochs_ratio[i])
                self._run_head_ensemble(
                    trainer,
                    model,
                    train_iter_dataset,
                    val_iter_loader,
                    phase1_epochs,
                    self.config.initial_head_lr,
                    cw_device,
                    val_cw,
                    sampling_strategy=effective_strategy,
                    hybrid_easy_target=hybrid_easy_target,
                    hybrid_diff_target=hybrid_diff_target,
                )

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
                val_preds, _, val_probs = self._evaluate_for_routing(
                    base_val_dataset, current_val_labels, model, return_probs=True
                )
                self._evaluate_validation_at_iteration(
                    val_preds,
                    None,
                    i + 1,
                    routing_probs=val_probs,
                )

            last_trained_iter = i + 1
            should_stop = self._apply_depth_selection_check(iteration=i + 1)
            if should_stop and self.config.depth_selection_enabled:
                self._depth_selection_state["early_stopped"] = True
                break

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
        """Update hierarchical labels for the next iteration."""
        train_preds, is_correct_train = self._evaluate_for_routing(
            base_train_dataset, train_labels, model
        )
        val_preds, is_correct_val = self._evaluate_for_routing(
            base_val_dataset, val_labels, model
        )

        is_difficult_train = self._determine_difficult_samples(is_correct_train)
        is_difficult_val = self._determine_difficult_samples(is_correct_val)

        if self.config.min_samples_per_node > 0:
            self.logger.log(
                f"\n--- Preparing Pseudo-Labels for Iteration {current_iteration + 1} ---",
                level=2,
            )

        train_proposed = self._compute_proposed_labels(train_labels, is_difficult_train)
        val_proposed = self._compute_proposed_labels(val_labels, is_difficult_val)

        # Merges decided from TRAINING counts only, then applied to val/test for consistency
        merged_nodes = self._determine_nodes_to_merge(train_proposed)
        self._merged_nodes_for_iteration = merged_nodes

        new_train_labels = self._apply_node_merges(
            train_proposed, merged_nodes, "train"
        )
        new_val_labels = self._apply_node_merges(val_proposed, merged_nodes, "val")

        return new_train_labels, new_val_labels, val_preds

    def _evaluate_for_routing(
        self,
        base_dataset: Any,
        labels: torch.Tensor,
        model: nn.Module,
        return_probs: bool = False,
    ):
        """Get predictions and correctness for a dataset."""
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
        """Compute proposed pseudo labels: labels * 2 + is_difficult."""
        return labels * 2 + is_difficult.long()

    def _determine_nodes_to_merge(
        self,
        proposed_labels: torch.Tensor,
    ) -> list:
        """Return odd-index difficult nodes with fewer than min_samples_per_node."""
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
        """Merge difficult nodes (odd indices) into their easy siblings (node - 1)."""
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

    # ============================================================================
    # INFERENCE - Model Predictions
    # ============================================================================

    def _get_hierarchical_predictions(
        self, dataloader: DataLoader, model: Optional[nn.Module] = None,
        return_probs: bool = False,
    ):
        """Get hierarchical node predictions from model."""
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
        """Collect raw model logits (before any activation) for all samples."""
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
        """Get hierarchical node probabilities from model."""
        if model is None:
            model = self.model

        model.eval()
        all_probabilities = []

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
                    if apply_mask:
                        mask = self._trained_nodes_mask.to(outputs.device)
                        if len(mask) == outputs.shape[1]:
                            outputs = outputs.clone()
                            outputs[:, ~mask] = float("-inf")
                    probabilities = torch.softmax(outputs, dim=1)
                else:
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
        """Get class probabilities from hierarchical model."""
        hierarchical_probs = self._get_hierarchical_probabilities(dataloader, model)

        class_probs = []
        for class_id in range(self.config.num_classes):
            start_node, end_node = get_class_node_range(class_id, iteration)
            class_raw = hierarchical_probs[:, start_node:end_node].sum(dim=1)
            class_probs.append(class_raw)

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
        """Evaluate model on test set after a specific iteration."""
        if self.config.verbose == 0:
            return

        entry_idx = iteration - 1

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

        test_pseudo_labels = None
        if iteration < self.config.num_iterations:
            if iteration > 1:
                current_test_labels = torch.from_numpy(
                    self._iteration_results[entry_idx - 1]["test_pseudo_labels"]
                )
            else:
                current_test_labels = y_test

            is_correct = hierarchical_predictions == current_test_labels
            is_difficult = self._determine_difficult_samples(is_correct)
            test_proposed = self._compute_proposed_labels(
                current_test_labels, is_difficult
            )
            merged_nodes = getattr(self, "_merged_nodes_for_iteration", [])
            test_pseudo_labels = self._apply_node_merges(
                test_proposed, merged_nodes, dataset_name="test"
            )

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

        pseudo_metrics = None
        if (
            iteration > 1
            and self._iteration_results[entry_idx - 1].get("test_pseudo_labels") is not None
        ):
            prev_pseudo_labels = self._iteration_results[entry_idx - 1]["test_pseudo_labels"]
            curr_routing = self._iteration_results[entry_idx]["hierarchical_predictions"]
            num_expected_nodes = get_num_nodes(iteration, self.config.num_classes)

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

        subgroup_auroc = calculate_subgroup_auroc(
            y_test_np, class_prob_for_auroc, test_subgroups
        )
        if subgroup_auroc:
            basic_metrics.update(subgroup_auroc)

        self.logger.log_evaluation_results(basic_metrics, pseudo_metrics, group_results, consolidated_group_results, class_names=self._class_names)

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

        if self.config.verbose >= 2:
            raw_logits = self._get_raw_logits(test_loader, model=model).numpy()
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
        """Store validation pseudo labels and compute validation metrics."""
        entry_idx = iteration - 1

        while len(self._iteration_results) <= entry_idx:
            self._iteration_results.append({})

        self._iteration_results[entry_idx]["val_pseudo_labels"] = (
            val_pseudo_labels.cpu().numpy() if val_pseudo_labels is not None else None
        )

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
        """Compute weight change statistics between iterations."""
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
            # Skip key not in previous or shape mismatch (heads expand between iterations)
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
