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
from ..evaluation.metrics import calculate_basic_metrics
from .loss import compute_auxiliary_loss, compute_l1_penalty, weighted_bce_loss
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
        phase2_best_weights = copy.deepcopy(self.best_model_weights)

        # Load best model from either phase
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
                )

            if scheduler is not None:
                # ReduceLROnPlateau needs a metric, others don't
                scheduler_type = self.config["scheduler"]["type"]
                if scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1

            current_epoch = epoch_offset + epoch + 1
            self.logger.epoch_update(current_epoch, total_epochs, train_loss, val_loss,
                                    self._last_train_aux_loss, self._last_val_aux_loss)

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
                        aux_outputs, labels, self.iteration_num, class_weights
                    )
                    weighted_aux = aux_weight * aux_loss
                    loss = loss + weighted_aux
                    total_aux_loss += aux_loss.item()

                # Add L1 regularization if configured
                if self.config["regularization_type"] in ["l1", "elastic_net"]:
                    lambda_l1 = self.config["lambda_l1"]
                    if lambda_l1 > 0:
                        loss += lambda_l1 * compute_l1_penalty(self.model)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        # Store auxiliary loss for logging
        self._last_train_aux_loss = total_aux_loss / num_batches if use_aux_loss and num_batches > 0 else 0.0

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
            Dictionary with loss and optionally metrics (accuracy, f1).
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
                        outputs, aux_outputs = self.model(images, return_aux_outputs=True)
                    else:
                        outputs = self.model(images)

                    # Main loss
                    loss = weighted_bce_loss(outputs, labels, class_weights)

                    # Auxiliary loss (computed separately, not added to loss for validation)
                    if use_aux_loss:
                        aux_loss = compute_auxiliary_loss(
                            aux_outputs, labels, self.iteration_num, class_weights
                        )
                        total_aux_loss += aux_loss.item()

                total_loss += loss.item()
                num_batches += 1

                if compute_metrics:
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

        # Store auxiliary loss for logging
        self._last_val_aux_loss = total_aux_loss / num_batches if use_aux_loss and num_batches > 0 else 0.0

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        result = {"loss": avg_loss}

        if compute_metrics and len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = calculate_basic_metrics(
                all_labels.cpu().numpy(), all_preds.cpu().numpy()
            )
            result["accuracy"] = metrics["accuracy"]
            result["f1"] = metrics["f1_score"]

        return result
