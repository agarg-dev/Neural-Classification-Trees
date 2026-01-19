"""
Core Training Functionality for NCT.

This module implements the NCTTrainer class which handles the two-phase
training strategy (frozen backbone, then full fine-tuning) for hierarchical models.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ..core.config import OptimizerConfig, SchedulerConfig
from ..utils.hierarchical import get_num_nodes
from .loss import (
    compute_auxiliary_loss,
    compute_l1_penalty,
    compute_l1_penalty_heads,
    weighted_bce_loss,
)
from .optimizer import create_optimizer, create_scheduler, create_two_stage_optimizer


class NCTTrainer:
    """
    Trainer for hierarchical models with two-phase training strategy.

    Phase 1: Train classification heads with frozen backbone.
    Phase 2: Fine-tune the entire model (backbone + heads).

    Supports early stopping, learning rate scheduling, and mixed precision training.
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        device: torch.device,
        config: Dict[str, Any],
        logger,
        iteration_num: int = 1,
    ):
        """
        Initialize the trainer.
        Args:
            model: The hierarchical model to train (can be None initially)
            device: Device for training
            config: Training configuration
            logger: NCTLogger instance for output
            iteration_num: Current iteration number (for logging)
        """
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger
        self.iteration_num = iteration_num

        # Use appropriate device for mixed precision training
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = GradScaler(self.device_type)

        # Early stopping tracking
        self.best_val_loss = float("inf")
        self.best_worst_group_acc = 0.0  # Track best pseudo worst-group accuracy
        self.best_difficult_node_acc = 0.0  # Track best difficult node accuracy (DNA)
        self.patience_counter = 0
        self.best_model_weights = None
        self.total_epochs_trained = 0

        # Auxiliary loss tracking
        self._last_train_aux_loss = 0.0
        self._last_val_aux_loss = 0.0

    def train_iteration(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        base_lr: float,
        head_lr: float,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Train the model for one iteration using two-phase approach.
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            base_lr: Learning rate for backbone.
            head_lr: Learning rate for classification heads.
            class_weights: Class weights for loss computation.
        Returns:
            Dictionary containing training results:
                - phase1_results: Results from phase 1 training.
                - phase2_results: Results from phase 2 training.
                - best_val_loss: Best validation loss achieved.
                - total_epochs: Total number of epochs trained.
                - early_stopped: Whether early stopping was triggered.
        """
        val_class_weights = (
            torch.ones_like(class_weights) if class_weights is not None else None
        )

        # Phase 1 training with its own early stopping
        self._reset_early_stopping()

        # Get phase1 ratio for current iteration
        ratio = self.config["phase1_epochs_ratio"][self.iteration_num - 1]
        phase1_epochs = int(epochs * ratio)
        phase1_results = self._train_phase1(
            train_loader,
            val_loader,
            phase1_epochs,
            head_lr,
            class_weights,
            val_class_weights,
            total_epochs=epochs,
        )
        phase1_best_loss = self.best_val_loss
        phase1_best_wga = self.best_worst_group_acc
        phase1_best_dna = self.best_difficult_node_acc
        phase1_best_weights = copy.deepcopy(self.best_model_weights)

        # Load best weights from phase 1 before starting phase 2
        self.model.load_state_dict(self.best_model_weights)

        # Phase 2 training with separate early stopping (reset counters)
        self._reset_early_stopping()

        phase2_results = self._train_phase2(
            train_loader,
            val_loader,
            epochs - phase1_epochs,
            base_lr,
            head_lr,
            class_weights,
            val_class_weights,
            total_epochs=epochs,
            epoch_offset=phase1_epochs,
        )
        phase2_best_loss = self.best_val_loss
        phase2_best_wga = self.best_worst_group_acc
        phase2_best_dna = self.best_difficult_node_acc
        phase2_best_weights = copy.deepcopy(self.best_model_weights)

        # Load best model from either phase based on the early stopping criterion
        criterion = self.config.get("model_selection_criterion", "auto")
        use_wga_for_selection = (criterion == "wga") or (
            criterion == "auto" and self.iteration_num > 1
        )
        use_dna_for_selection = (criterion == "dna") and (self.iteration_num > 1)

        if use_dna_for_selection:
            # Higher is better for difficult node accuracy
            if phase1_best_dna > phase2_best_dna:
                self.model.load_state_dict(phase1_best_weights)
                self.best_model_weights = phase1_best_weights
            else:
                self.model.load_state_dict(phase2_best_weights)
                self.best_model_weights = phase2_best_weights
        elif use_wga_for_selection:
            # Higher is better for worst-group accuracy
            if phase1_best_wga > phase2_best_wga:
                self.model.load_state_dict(phase1_best_weights)
                self.best_model_weights = phase1_best_weights
            else:
                self.model.load_state_dict(phase2_best_weights)
                self.best_model_weights = phase2_best_weights
        else:
            # Lower is better for validation loss
            if phase1_best_loss < phase2_best_loss:
                self.model.load_state_dict(phase1_best_weights)
                self.best_model_weights = phase1_best_weights
            else:
                self.model.load_state_dict(phase2_best_weights)
                self.best_model_weights = phase2_best_weights

        return {
            "phase1_results": phase1_results,
            "phase2_results": phase2_results,
            "best_val_loss": min(phase1_best_loss, phase2_best_loss),
            "total_epochs": self.total_epochs_trained,
            "early_stopped": phase1_results.get("early_stopped", False)
            or phase2_results.get("early_stopped", False),
        }

    def _reset_early_stopping(self):
        """Reset early stopping tracking for new phase."""
        self.best_val_loss = float("inf")
        self.best_worst_group_acc = -1.0  # Initialize to -1 so first epoch always saves
        self.best_difficult_node_acc = -1.0  # Initialize to -1 so first epoch always saves
        self.patience_counter = 0
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

    def _train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        head_lr: float,
        class_weights: Optional[torch.Tensor],
        val_class_weights: Optional[torch.Tensor],
        total_epochs: int = None,
    ) -> Dict[str, Any]:
        """
        Execute phase 1 training with frozen backbone.
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs for this phase.
            head_lr: Learning rate for classification heads.
            class_weights: Training class weights.
            val_class_weights: Validation class weights.
            total_epochs: Total epochs across both phases (for logging).
        Returns:
            Dictionary with phase 1 training results.
        """
        if epochs <= 0:
            return {"epochs": 0, "train_losses": [], "val_losses": []}

        self.logger.phase_start(
            "Phase 1: Training output heads (frozen base)", 1, epochs
        )

        for param in self.model.base.parameters():
            param.requires_grad = False

        # Get head parameters (includes parent layers in hierarchical mode)
        model_params = self.model.get_trainable_parameters()

        optimizer_config = OptimizerConfig(**self.config.get("optimizer"))
        optimizer = create_optimizer(
            model_params["heads"],
            config=optimizer_config,
            learning_rate=head_lr,
            regularization_type=self.config["regularization_type"],
        )

        scheduler_config = SchedulerConfig(**self.config.get("scheduler"))
        scheduler = create_scheduler(
            optimizer,
            config=scheduler_config,
            num_epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        return self._train_epochs(
            train_loader,
            val_loader,
            epochs,
            optimizer,
            scheduler,
            class_weights,
            val_class_weights,
            phase_name="Phase 1",
            total_epochs=total_epochs,
            epoch_offset=0,
            patience=self.config["phase1_patience"],
        )

    def _train_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        base_lr: float,
        head_lr: float,
        class_weights: Optional[torch.Tensor],
        val_class_weights: Optional[torch.Tensor],
        total_epochs: int = None,
        epoch_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute phase 2 training with full model fine-tuning.
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs for this phase.
            base_lr: Learning rate for backbone.
            head_lr: Learning rate for classification heads.
            class_weights: Training class weights.
            val_class_weights: Validation class weights.
            total_epochs: Total epochs across both phases (for logging).
            epoch_offset: Epoch offset for logging (phase 1 epochs).
        Returns:
            Dictionary with phase 2 training results.
        """
        if epochs <= 0:
            return {"epochs": 0, "train_losses": [], "val_losses": []}

        self.logger.phase_start("Phase 2: Fine-tuning the full model", 2, epochs)

        for param in self.model.base.parameters():
            param.requires_grad = True

        model_params = self.model.get_trainable_parameters()
        optimizer_config = OptimizerConfig(**self.config.get("optimizer"))
        optimizer = create_two_stage_optimizer(
            model_params,
            base_lr=base_lr,
            head_lr=head_lr,
            config=optimizer_config,
            regularization_type=self.config["regularization_type"],
        )

        scheduler_config = SchedulerConfig(**self.config.get("scheduler"))
        scheduler = create_scheduler(
            optimizer,
            config=scheduler_config,
            num_epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        return self._train_epochs(
            train_loader,
            val_loader,
            epochs,
            optimizer,
            scheduler,
            class_weights,
            val_class_weights,
            phase_name="Phase 2",
            total_epochs=total_epochs,
            epoch_offset=epoch_offset,
            patience=self.config["phase2_patience"],
        )

    def _train_epochs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        class_weights: Optional[torch.Tensor],
        val_class_weights: Optional[torch.Tensor],
        phase_name: str = "",
        total_epochs: int = None,
        epoch_offset: int = 0,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute training for a specified number of epochs.
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler.
            class_weights: Training class weights.
            val_class_weights: Validation class weights.
            phase_name: Name of current phase (for logging).
            total_epochs: Total epochs across both phases.
            epoch_offset: Epoch offset for logging.
            patience: Early stopping patience for this phase.
        Returns:
            Dictionary with training results.
        """
        train_losses = []
        val_losses = []

        if total_epochs is None:
            total_epochs = epochs

        phase_num = 1 if "Phase 1" in phase_name else 2
        log_metrics = self.config["log_epoch_metrics"]

        # Log early stopping criterion for this iteration
        criterion = self.config.get("model_selection_criterion", "auto")
        if criterion == "val_loss":
            self.logger.log("    Early stopping criterion: val_loss", level=2)
        elif criterion == "wga":
            self.logger.log(
                "    Early stopping criterion: pseudo WGA (filled groups)", level=2
            )
        elif criterion == "dna":
            if self.iteration_num == 1:
                self.logger.log(
                    "    Early stopping criterion: val_loss (dna: iteration 1)",
                    level=2,
                )
            else:
                self.logger.log(
                    "    Early stopping criterion: Difficult Node Accuracy (DNA)", level=2
                )
        else:  # "auto"
            if self.iteration_num == 1:
                self.logger.log(
                    "    Early stopping criterion: val_loss (auto: iteration 1)",
                    level=2,
                )
            else:
                self.logger.log(
                    "    Early stopping criterion: pseudo WGA (auto: iteration 2+)",
                    level=2,
                )

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, class_weights)
            train_losses.append(train_loss)

            val_result = self._validate_epoch(
                val_loader, val_class_weights, compute_metrics=log_metrics
            )
            val_loss = val_result["loss"]
            val_losses.append(val_loss)

            # Log epoch metrics if enabled
            if log_metrics:
                base_lr = optimizer.param_groups[0]["lr"]
                head_lr = (
                    optimizer.param_groups[1]["lr"]
                    if len(optimizer.param_groups) > 1
                    else base_lr
                )

                self.logger.log_epoch_metrics(
                    iteration=self.iteration_num,
                    phase=phase_num,
                    epoch=epoch_offset + epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_acc=val_result.get("accuracy", 0.0),
                    val_f1=val_result.get("f1", 0.0),
                    base_lr=base_lr,
                    head_lr=head_lr,
                    train_aux_loss=self._last_train_aux_loss,
                    val_aux_loss=self._last_val_aux_loss,
                    worst_group_acc=val_result.get("worst_group_accuracy", 0.0),
                    difficult_node_acc=val_result.get("difficult_node_accuracy", 0.0),
                    difficult_node_f1=val_result.get("difficult_node_f1", 0.0),
                )

            if scheduler is not None:
                # ReduceLROnPlateau needs a metric, others don't
                scheduler_type = self.config["scheduler"]["type"]
                if scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping criterion selection:
            # - "val_loss": Always use validation loss
            # - "wga": Always use pseudo worst-group accuracy (filled groups only)
            # - "dna": val_loss for iteration 1, difficult node accuracy for iteration 2+
            # - "auto": val_loss for iteration 1, wga for iteration 2+
            criterion = self.config.get("model_selection_criterion", "auto")

            use_wga = False
            use_dna = False

            if criterion == "val_loss":
                use_wga = False
                use_dna = False
            elif criterion == "wga":
                use_wga = "worst_group_accuracy_filled" in val_result
            elif criterion == "dna":
                # DNA uses val_loss for iteration 1, DNA for iteration 2+
                use_dna = (
                    self.iteration_num > 1
                ) and "difficult_node_accuracy" in val_result
            else:  # "auto"
                use_wga = (
                    self.iteration_num > 1
                ) and "worst_group_accuracy_filled" in val_result

            if use_dna:
                # Higher is better for difficult node accuracy
                dna = val_result["difficult_node_accuracy"]
                if dna > self.best_difficult_node_acc:
                    self.best_difficult_node_acc = dna
                    self.best_val_loss = val_loss  # Also update for logging
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
            elif use_wga:
                # Higher is better for worst-group accuracy (use filled groups only)
                worst_group_acc = val_result["worst_group_accuracy_filled"]
                if worst_group_acc > self.best_worst_group_acc:
                    self.best_worst_group_acc = worst_group_acc
                    self.best_val_loss = val_loss  # Also update for logging
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
            else:
                # Use validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1

            current_epoch = epoch_offset + epoch + 1
            self.logger.epoch_update(
                current_epoch,
                total_epochs,
                train_loss,
                val_loss,
                self._last_train_aux_loss,
                self._last_val_aux_loss,
                worst_group_acc=val_result.get("worst_group_accuracy", 0.0),
                difficult_node_acc=val_result.get("difficult_node_accuracy", 0.0),
                difficult_node_f1=val_result.get("difficult_node_f1", 0.0),
            )

            if self.config["early_stopping"] and self.patience_counter >= patience:
                self.logger.early_stop(phase_name, phase_num)
                break

        self.total_epochs_trained += len(train_losses)

        completed_all = len(train_losses) == epochs and self.patience_counter < patience
        self.logger.phase_end(phase_num, completed_all)

        return {
            "epochs": len(train_losses),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "early_stopped": len(train_losses) < epochs,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        class_weights: Optional[torch.Tensor],
    ) -> float:
        """
        Execute a single training epoch.
        Args:
            loader: Training data loader.
            optimizer: Optimizer instance.
            class_weights: Class weights for loss computation.
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0

        # Check if auxiliary loss should be computed
        aux_weight = self.config.get("aux_loss_weight", 0.0)
        use_aux_loss = (
            aux_weight > 0
            and self.iteration_num > 1
            and self.config.get("hierarchical_architecture") != "none"
        )

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            with autocast(self.device_type):
                # Get outputs and optionally auxiliary outputs
                if use_aux_loss:
                    outputs, aux_outputs = self.model(images, return_aux_outputs=True)
                else:
                    outputs = self.model(images)

                # Main loss
                loss = weighted_bce_loss(outputs, labels, class_weights)

                # Auxiliary loss
                if use_aux_loss:
                    aux_loss = compute_auxiliary_loss(
                        aux_outputs,
                        labels,
                        self.iteration_num,
                        class_weights,
                        max_weight=self.config.get("max_class_weight"),
                        num_classes=self.config.get("num_classes", 2),
                    )
                    weighted_aux = aux_weight * aux_loss
                    loss = loss + weighted_aux
                    total_aux_loss += aux_loss.item()

                # Add L1 regularization if configured
                if self.config["regularization_type"] in ["l1", "elastic_net"]:
                    lambda_l1 = self.config["lambda_l1"]
                    if lambda_l1 > 0:
                        loss += lambda_l1 * compute_l1_penalty_heads(self.model)
                        # loss += lambda_l1 * compute_l1_penalty(self.model)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        # Store auxiliary loss for logging
        self._last_train_aux_loss = (
            total_aux_loss / num_batches if use_aux_loss and num_batches > 0 else 0.0
        )

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(
        self,
        loader: DataLoader,
        class_weights: Optional[torch.Tensor],
        compute_metrics: bool = False,
    ) -> Dict[str, float]:
        """
        Execute a single validation epoch.
        Args:
            loader: Validation data loader.
            class_weights: Class weights for loss computation.
            compute_metrics: Whether to compute accuracy/F1 (slower)
        Returns:
            Dictionary with loss, pseudo worst-group accuracy, and optionally other metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        # Check if auxiliary loss should be computed
        aux_weight = self.config.get("aux_loss_weight", 0.0)
        use_aux_loss = (
            aux_weight > 0
            and self.iteration_num > 1
            and self.config.get("hierarchical_architecture") != "none"
        )

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with autocast(self.device_type):
                    # Get outputs and optionally auxiliary outputs
                    if use_aux_loss:
                        outputs, aux_outputs = self.model(
                            images, return_aux_outputs=True
                        )
                    else:
                        outputs = self.model(images)

                    # Main loss
                    loss = weighted_bce_loss(outputs, labels, class_weights)

                    # Auxiliary loss (computed separately, not added to loss for validation)
                    if use_aux_loss:
                        aux_loss = compute_auxiliary_loss(
                            aux_outputs,
                            labels,
                            self.iteration_num,
                            class_weights,
                            max_weight=self.config.get("max_class_weight"),
                            num_classes=self.config.get("num_classes", 2),
                        )
                        total_aux_loss += aux_loss.item()

                total_loss += loss.item()
                num_batches += 1

                # Always collect predictions and labels for pseudo worst-group accuracy
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Store auxiliary loss for logging
        self._last_val_aux_loss = (
            total_aux_loss / num_batches if use_aux_loss and num_batches > 0 else 0.0
        )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        result = {"loss": avg_loss}

        # Compute pseudo worst-group accuracy (per hierarchical node)
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            # Number of pseudo-groups = number of hierarchical nodes at this iteration
            num_classes = self.config.get('num_classes', 2)
            num_groups = get_num_nodes(self.iteration_num, num_classes)

            # Compute per-group accuracy and sizes
            group_accuracies = []
            node_sizes = {}
            per_node_accuracy = {}
            for group_id in range(num_groups):
                group_mask = all_labels == group_id
                group_count = group_mask.sum().item()
                node_sizes[group_id] = group_count

                if group_count > 0:
                    group_correct = (
                        (all_preds[group_mask] == all_labels[group_mask]).sum().item()
                    )
                    group_acc = group_correct / group_count
                    group_accuracies.append(group_acc)
                    per_node_accuracy[group_id] = group_acc
                else:
                    per_node_accuracy[group_id] = 0.0

            # Pseudo worst-group accuracy (minimum across filled hierarchical nodes)
            if len(group_accuracies) > 0:
                worst_group_acc = min(group_accuracies)
                avg_acc = sum(group_accuracies) / len(group_accuracies)
                result["worst_group_accuracy"] = worst_group_acc
                result["worst_group_accuracy_filled"] = (
                    worst_group_acc  # Same as above (only filled groups counted)
                )
                result["accuracy"] = avg_acc
                result["num_groups_present"] = len(group_accuracies)
                result["num_groups_expected"] = num_groups
                result["node_sizes"] = node_sizes
                result["per_node_accuracy"] = per_node_accuracy

            # Compute difficult node accuracy (DNA) and F1
            # Difficult nodes have odd indices (1, 3, 5, 7, ...)
            all_labels_np = all_labels.numpy()
            all_preds_np = all_preds.numpy()
            difficult_mask = (all_labels_np % 2 == 1)

            if difficult_mask.sum() > 0:
                difficult_correct = (all_preds_np[difficult_mask] == all_labels_np[difficult_mask]).sum()
                difficult_total = difficult_mask.sum()
                dna = float(difficult_correct / difficult_total)
                # Compute F1 on difficult nodes
                from sklearn.metrics import f1_score as sklearn_f1
                dnf1 = float(sklearn_f1(
                    all_labels_np[difficult_mask],
                    all_preds_np[difficult_mask],
                    average="macro",
                    zero_division=1.0
                ))
            else:
                dna = 1.0  # No difficult samples = perfect routing
                dnf1 = 1.0

            result["difficult_node_accuracy"] = dna
            result["difficult_node_f1"] = dnf1

        return result
