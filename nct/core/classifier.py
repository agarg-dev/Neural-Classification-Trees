"""
Neural Classification Tree (NCT) Classifier.

Main interface for hierarchical image classification using iterative training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader, Dataset as TorchDataset

from .config import NCTConfig
from ..utils.seed import set_seed
from ..utils.device import get_device
from ..utils.logger import NCTLogger
from ..data.loaders import create_data_loader
from ..data.datasets import IterativeDataset, extract_labels_from_dataset
from ..models.hierarchical import HierarchicalClassifier
from ..models.utils import initialize_or_update_model
from ..training.trainer import NCTTrainer
from ..training.loss import compute_class_weights
from ..evaluation.metrics import (
    calculate_basic_metrics, calculate_group_metrics, print_evaluation_report
)
from ..evaluation.analysis import analyze_hierarchical_nodes, get_waterbird_group_names


class NCTClassifier:
    """
    Neural Classification Tree for hierarchical binary classification.

    Builds a tree hierarchy of classifiers through iterative training,
    routing difficult samples to specialized nodes.

    Attributes:
        config: Configuration with all hyperparameters.
        device: Torch device for training and inference.
        model: The HierarchicalClassifier model.
        generator: RNG for reproducible shuffling.
    """

    def __init__(self, num_iterations: int = 3, epochs_per_iteration: Optional[List[int]] = None,
                 batch_size: int = 32, initial_base_lr: float = 1e-5, initial_head_lr: float = 1e-5,
                 lr_decay_factor: float = 2.0, backbone: str = 'resnet50', **kwargs):
        """
        Initialize the NCT classifier.

        Args:
            num_iterations: Number of hierarchical levels.
            epochs_per_iteration: List of epoch counts for each iteration.
            batch_size: Training and inference batch size.
            initial_base_lr: Initial learning rate for backbone.
            initial_head_lr: Initial learning rate for classification heads.
            lr_decay_factor: LR decay factor per iteration.
            backbone: Backbone architecture ('resnet50', 'resnet18', etc.).
            **kwargs: Additional config (optimizer_type, scheduler_type, use_class_weights,
                      pretrained, phase1_epochs_ratio, patience, early_stopping, image_size,
                      num_workers, pin_memory, seed, device, verbose).
                      See NCTConfig for full details and defaults.
        """

        self.config = NCTConfig(
            num_iterations=num_iterations,
            epochs_per_iteration=epochs_per_iteration,
            batch_size=batch_size,
            initial_base_lr=initial_base_lr,
            initial_head_lr=initial_head_lr,
            lr_decay_factor=lr_decay_factor,
            backbone=backbone,
            **kwargs
        )

        self.device = get_device(self.config.device)
        if self.config.seed is not None:
            set_seed(self.config.seed)

        # Persistent generator for reproducible data shuffling
        self.generator = torch.Generator()
        if self.config.seed is not None:
            self.generator.manual_seed(self.config.seed)

        self.model = None
        self._is_fitted = False
        self.logger = NCTLogger(verbose=self.config.verbose)


    # ============================================================================
    # PUBLIC API
    # ============================================================================

    def fit(self, X_train: Any, X_val: Any, X_test: Optional[Any] = None,
            group_labels: Optional[Any] = None, dataset_name: str = "generic") -> 'NCTClassifier':
        """
        Train the hierarchical classifier on the provided data.
        Args:
            X_train: Training data. Must be a PyTorch Dataset object.
            X_val: Validation data. Must be a PyTorch Dataset object.
            X_test: Optional test data for per-iteration evaluation.
            group_labels: Optional group labels for test data evaluation.
            dataset_name: Name of dataset for evaluation (e.g., "waterbird").
        Returns:
            self: The fitted classifier instance.
        """

        if not isinstance(X_train, TorchDataset):
            raise NotImplementedError(
                "Direct tensor/array inputs not supported. Please pass PyTorch Dataset objects."
            )

        y_train = extract_labels_from_dataset(X_train)
        y_val = extract_labels_from_dataset(X_val)
        self.model = self._run_iterative_training(X_train, X_val, y_train, y_val, X_test, group_labels, dataset_name)
        self._is_fitted = True
        self.logger.training_end()
        return self


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

        final_num_nodes = 2 ** self.config.num_iterations
        binary_preds = (hierarchical_preds >= final_num_nodes / 2).long()
        return binary_preds.numpy()


    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for samples.
        Args:
            X: Input data. Must be a PyTorch Dataset object.
        Returns:
            Class probabilities as a numpy array of shape (n_samples, 2).
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        test_loader = self._create_iterative_loader(X, is_training=False)
        hierarchical_probs = self._get_hierarchical_probabilities(test_loader)

        final_num_nodes = 2 ** self.config.num_iterations
        mid_point = final_num_nodes // 2
        class_0_prob = hierarchical_probs[:, :mid_point].sum(dim=1)
        class_1_prob = hierarchical_probs[:, mid_point:].sum(dim=1)
        return torch.stack([class_0_prob, class_1_prob], dim=1).numpy()


    # ============================================================================
    # DATA UTILITIES - DataLoader Creation
    # ============================================================================

    def _create_iterative_loader(self, dataset: Any, is_training: bool) -> DataLoader:
        """
        Create a DataLoader for iterative dataset.
        Args:
            dataset: Input dataset (IterativeDataset).
            is_training: Whether this is for training or evaluation.
        Returns:
            DataLoader configured for iterative training/evaluation.
        """
        if is_training:
            return create_data_loader(
                dataset, is_training=True,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                generator=self.generator
            )
        else:
            return create_data_loader(
                dataset, is_training=False,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                seed=self.config.seed
            )

    def evaluate(self, X_test: Any, group_labels: Optional[Any] = None,
                group_names: Optional[Dict[int, str]] = None, dataset_name: str = "generic") -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        Args:
            X_test: Test data. Must be a PyTorch Dataset object.
            group_labels: Optional group/subpopulation labels for fairness analysis.
            group_names: Optional mapping from group IDs to readable names.
            dataset_name: Name of the dataset (used for special handling).
        Returns:
            Dictionary containing evaluation metrics:
                - basic_metrics: Accuracy, precision, recall, F1.
                - group_metrics: Per-group performance (if group_labels provided).
                - node_analysis: Hierarchical node utilization.
        """

        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        is_waterbird = dataset_name.lower() == "waterbird"
        if is_waterbird and group_names is None:
            group_names = get_waterbird_group_names()
        binary_predictions = self.predict(X_test)
        test_loader = self._create_iterative_loader(X_test, is_training=False)
        hierarchical_predictions = self._get_hierarchical_predictions(test_loader)

        y_test = extract_labels_from_dataset(X_test)
        y_test_np = y_test.numpy()
        basic_metrics = calculate_basic_metrics(y_test_np, binary_predictions)
        results = {'basic_metrics': basic_metrics}

        if group_labels is not None:
            group_labels_np = np.array(group_labels) if not isinstance(group_labels, np.ndarray) else group_labels
            group_metrics = calculate_group_metrics(y_test_np, binary_predictions, group_labels_np, group_names)
            results['group_metrics'] = group_metrics

            node_analysis = analyze_hierarchical_nodes(
                hierarchical_predictions, torch.from_numpy(y_test_np),
                torch.from_numpy(group_labels_np), self.config.num_iterations,
                group_names, verbose=self.config.verbose, is_waterbird=is_waterbird
            )
            results['node_analysis'] = node_analysis

        if self.config.verbose:
            print_evaluation_report(results, "NCT Evaluation Results")

        return results


    # ============================================================================
    # CHECKPOINT MANAGEMENT - Save/Load
    # ============================================================================

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the trained model and configuration to disk.
        Args:
            filepath: Path where the checkpoint will be saved.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'num_heads': len(self.model.output_heads)
        }
        torch.save(save_dict, filepath)

        if self.config.verbose:
            print(f"Checkpoint saved to {filepath}")


    @classmethod
    def from_checkpoint(
        cls,
        filepath: str,
        device: Optional[Union[str, torch.device]] = None
    ) -> 'NCTClassifier':
        """
        Load a trained classifier from a checkpoint file.
        Args:
            filepath: Path to the checkpoint file.
            device: Device to load the model on. If None, uses saved device.
        Returns:
            NCTClassifier: Loaded classifier instance.
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        if 'config' not in checkpoint:
            raise RuntimeError("Invalid checkpoint: missing 'config'")

        config_dict = checkpoint['config']

        if device is not None:
            config_dict['device'] = device

        classifier = cls(**config_dict)

        if 'model_state_dict' not in checkpoint:
            raise RuntimeError("Invalid checkpoint: missing 'model_state_dict'")

        if 'num_heads' in checkpoint:
            num_heads = checkpoint['num_heads']
        else:
            state_dict = checkpoint['model_state_dict']
            num_heads = 0
            for key in state_dict.keys():
                if key.startswith('output_heads.'):
                    head_idx = int(key.split('.')[1])
                    num_heads = max(num_heads, head_idx + 1)
            if num_heads == 0:
                raise RuntimeError("Could not determine model architecture from checkpoint")

        classifier.model = HierarchicalClassifier(
            num_heads=num_heads,
            backbone=classifier.config.backbone,
            pretrained=False,
            head_hidden_dim=classifier.config.head_hidden_dim,
            head_dropout=classifier.config.head_dropout
        ).to(classifier.device)

        try:
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        classifier._is_fitted = True

        if classifier.config.verbose:
            print(f"Checkpoint loaded from {filepath}")

        return classifier


    # ============================================================================
    # TRAINING ORCHESTRATION - Iterative Training Loop
    # ============================================================================

    def _run_iterative_training(self, base_train_dataset: Any, base_val_dataset: Any,
                               current_train_labels: torch.Tensor, current_val_labels: torch.Tensor,
                               X_test: Optional[Any] = None, group_labels: Optional[Any] = None,
                               dataset_name: str = "generic") -> torch.Tensor:
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
        Returns:
            The trained HierarchicalClassifier model.
        """

        current_base_lr = self.config.initial_base_lr
        model = None

        self.logger.training_start(self.config.to_dict())

        trainer = NCTTrainer(None, self.device, self.config.to_dict(), logger=self.logger)

        for i in range(self.config.num_iterations):
            num_nodes = 2 ** (i + 1)
            epochs_for_iter = self.config.epochs_per_iteration[i]

            class_weights = compute_class_weights(current_train_labels, num_nodes) if self.config.use_class_weights else None
            self.logger.iteration_start(i + 1, self.config.num_iterations, num_nodes, epochs_for_iter, class_weights)

            if i > 0:
                self.logger.weight_transfer()

            model = initialize_or_update_model(
                num_nodes, model, self.config.backbone, self.config.pretrained, self.device,
                verbose=self.config.verbose,
                head_hidden_dim=self.config.head_hidden_dim,
                head_dropout=self.config.head_dropout
            )
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            except Exception:
                pass

            train_iter_dataset = IterativeDataset(base_train_dataset, current_train_labels)
            val_iter_dataset = IterativeDataset(base_val_dataset, current_val_labels)

            train_iter_loader = self._create_iterative_loader(train_iter_dataset, is_training=True)
            val_iter_loader = self._create_iterative_loader(val_iter_dataset, is_training=False)

            trainer.model = model
            trainer.train_iteration(
                train_iter_loader, val_iter_loader, epochs_for_iter,
                current_base_lr, self.config.initial_head_lr,
                class_weights.to(self.device) if class_weights is not None else None
            )

            self.logger.iteration_end()

            if X_test is not None:
                self._evaluate_at_iteration(model, X_test, i + 1, group_labels, dataset_name)

            if i < self.config.num_iterations - 1:
                current_train_labels, current_val_labels = self._update_labels_for_next_iteration(
                    model, base_train_dataset, base_val_dataset, current_train_labels, current_val_labels
                )
                current_base_lr /= self.config.lr_decay_factor

        return model
    

    def _update_labels_for_next_iteration(self, model, base_train_dataset: Any, base_val_dataset: Any,
                                        train_labels: torch.Tensor, val_labels: torch.Tensor):
        """
        Update hierarchical labels for the next iteration.
        Evaluates current model and routes samples to "easy" or "hard" child nodes
        based on prediction correctness.
        Args:
            model: Current trained model.
            base_train_dataset: Base training dataset.
            base_val_dataset: Base validation dataset.
            train_labels: Current training labels.
            val_labels: Current validation labels.
        Returns:
            Tuple of (new_train_labels, new_val_labels) for next iteration.
        """
        train_eval_dataset = IterativeDataset(base_train_dataset, train_labels)
        val_eval_dataset = IterativeDataset(base_val_dataset, val_labels)

        train_eval_loader = self._create_iterative_loader(train_eval_dataset, is_training=False)
        val_eval_loader = self._create_iterative_loader(val_eval_dataset, is_training=False)
        train_preds = self._get_hierarchical_predictions(train_eval_loader, model=model)
        is_correct_train = (train_preds == train_labels)
        new_train_labels = train_labels * 2 + (1 - is_correct_train.long())

        val_preds = self._get_hierarchical_predictions(val_eval_loader, model=model)
        is_correct_val = (val_preds == val_labels)
        new_val_labels = val_labels * 2 + (1 - is_correct_val.long())

        return new_train_labels, new_val_labels


    # ============================================================================
    # INFERENCE - Model Predictions
    # ============================================================================

    def _get_hierarchical_predictions(self, dataloader: DataLoader, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Get hierarchical node predictions from model.
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

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)

    def _get_hierarchical_probabilities(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Get hierarchical node probabilities from model.
        Args:
            dataloader: DataLoader to generate probabilities for.
        Returns:
            Tensor of hierarchical node probabilities.
        """
        self.model.eval()
        all_probabilities = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu())

        return torch.cat(all_probabilities)


    # ============================================================================
    # EVALUATION - Per-Iteration Evaluation
    # ============================================================================

    def _evaluate_at_iteration(self, model: Any, X_test: Any, iteration: int,
                               group_labels: Optional[Any] = None, dataset_name: str = "generic") -> None:
        """
        Evaluate model performance after a specific iteration.
        Args:
            model: The trained model at current iteration.
            X_test: Test dataset.
            iteration: Current iteration number (1-indexed).
            group_labels: Optional group labels for fairness analysis.
            dataset_name: Name of dataset for special handling.
        """
        if self.config.verbose == 0:
            return

        print(f"\n--- Evaluation on Test Set after Iteration {iteration} ---")

        test_loader = self._create_iterative_loader(X_test, is_training=False)
        hierarchical_predictions = self._get_hierarchical_predictions(test_loader, model=model)

        final_num_nodes = 2 ** iteration
        binary_predictions = (hierarchical_predictions >= final_num_nodes / 2).long().numpy()

        y_test = extract_labels_from_dataset(X_test)
        y_test_np = y_test.numpy()

        basic_metrics = calculate_basic_metrics(y_test_np, binary_predictions)
        print(f"Test Accuracy: {basic_metrics['accuracy']:.2%}")
        print(f"Test F1 Score: {basic_metrics['f1_score']:.4f}")

        if group_labels is not None:
            is_waterbird = dataset_name.lower() == "waterbird"
            if is_waterbird:
                group_names = get_waterbird_group_names()
            else:
                group_names = None

            group_labels_np = np.array(group_labels) if not isinstance(group_labels, np.ndarray) else group_labels
            group_results = calculate_group_metrics(y_test_np, binary_predictions, group_labels_np, group_names)

            print("\n--- Group-Specific Performance ---")
            for group_name, metrics in group_results['group_metrics'].items():
                print(f"  Accuracy for '{group_name}': {metrics['accuracy']:.2%}")

            print(f"\nWorst-Group Accuracy: {group_results['summary']['worst_group_accuracy']:.2%}")

            if is_waterbird and self.config.verbose >= 2:
                analyze_hierarchical_nodes(
                    hierarchical_predictions, torch.from_numpy(y_test_np),
                    torch.from_numpy(group_labels_np), iteration,
                    group_names, verbose=True, is_waterbird=True
                )
        print()