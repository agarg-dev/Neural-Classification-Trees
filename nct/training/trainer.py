"""
Core Training Functionality for NCT.

This module implements the NCTTrainer class which handles the two-phase
training strategy (frozen backbone, then full fine-tuning) for hierarchical models.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

from .loss import weighted_bce_loss, compute_class_weights
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
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        logger=None
    ):
        """
        Initialize the trainer.
        Args:
            model: The hierarchical model to train
            device: Device for training
            config: Training configuration
            logger: NCTLogger instance for output
        """
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger

        self.scaler = GradScaler()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_weights = None
        self.total_epochs_trained = 0


    def train_iteration(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        base_lr: float,
        head_lr: float,
        class_weights: Optional[torch.Tensor] = None
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
        val_class_weights = torch.ones_like(class_weights) if class_weights is not None else None

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

        phase1_epochs = max(1, int(epochs * self.config.get('phase1_epochs_ratio', 0.5)))
        phase1_results = self._train_phase1(
            train_loader, val_loader, phase1_epochs, head_lr,
            class_weights, val_class_weights, total_epochs=epochs
        )

        if phase1_results.get('early_stopped', False) and self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        phase2_results = self._train_phase2(
            train_loader, val_loader, epochs - phase1_epochs, base_lr, head_lr,
            class_weights, val_class_weights, total_epochs=epochs, epoch_offset=phase1_epochs
        )

        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        return {
            'phase1_results': phase1_results,
            'phase2_results': phase2_results,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.total_epochs_trained,
            'early_stopped': self.patience_counter >= self.config.get('patience', 10)
        }


    def _train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        head_lr: float,
        class_weights: Optional[torch.Tensor],
        val_class_weights: Optional[torch.Tensor],
        total_epochs: int = None
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
        if self.logger:
            self.logger.phase_start("Phase 1: Training output heads (frozen base)", 1, epochs)

        for param in self.model.base.parameters():
            param.requires_grad = False
        optimizer = create_optimizer(
            self.model.output_heads.parameters(),
            optimizer_type=self.config.get('optimizer_type', 'adamw'),
            learning_rate=head_lr,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        scheduler = create_scheduler(
            optimizer,
            scheduler_type='plateau',
            factor=0.5,
            patience=3
        )

        return self._train_epochs(
            train_loader, val_loader, epochs, optimizer, scheduler,
            class_weights, val_class_weights, phase_name="Phase 1",
            total_epochs=total_epochs, epoch_offset=0
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
        epoch_offset: int = 0
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
            return {'epochs': 0, 'train_losses': [], 'val_losses': []}

        if self.logger:
            self.logger.phase_start("Phase 2: Fine-tuning the full model", 2, epochs)

        for param in self.model.base.parameters():
            param.requires_grad = True

        model_params = self.model.get_trainable_parameters()
        optimizer = create_two_stage_optimizer(
            model_params,
            base_lr=base_lr,
            head_lr=head_lr,
            optimizer_type=self.config.get('optimizer_type', 'adamw'),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        scheduler = create_scheduler(
            optimizer,
            scheduler_type='plateau',
            factor=0.5,
            patience=3
        )

        return self._train_epochs(
            train_loader, val_loader, epochs, optimizer, scheduler,
            class_weights, val_class_weights, phase_name="Phase 2",
            total_epochs=total_epochs, epoch_offset=epoch_offset
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
        epoch_offset: int = 0
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
        Returns:
            Dictionary with training results.
        """
        train_losses = []
        val_losses = []
        patience = self.config.get('patience', 10)
        verbose = self.config.get('verbose', True)

        if total_epochs is None:
            total_epochs = epochs

        for epoch in range(epochs):
            train_loss = self._run_epoch(
                train_loader, optimizer, class_weights, is_training=True
            )
            train_losses.append(train_loss)

            val_loss = self._run_epoch(
                val_loader, None, val_class_weights, is_training=False
            )
            val_losses.append(val_loss)

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1

            if self.logger:
                current_epoch = epoch_offset + epoch + 1
                self.logger.epoch_update(current_epoch, total_epochs, train_loss, val_loss)

            if self.config.get('early_stopping', True) and self.patience_counter >= patience:
                if self.logger:
                    phase_num = 1 if "Phase 1" in phase_name else 2
                    self.logger.early_stop(phase_name, phase_num)
                break

        self.total_epochs_trained += len(train_losses)

        if self.logger:
            phase_num = 1 if "Phase 1" in phase_name else 2
            completed_all = len(train_losses) == epochs and self.patience_counter < patience
            self.logger.phase_end(phase_num, completed_all)

        return {
            'epochs': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'early_stopped': len(train_losses) < epochs
        }


    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        class_weights: Optional[torch.Tensor],
        is_training: bool
    ) -> float:
        """
        Execute a single training or validation epoch.
        Args:
            loader: Data loader.
            optimizer: Optimizer instance (None for validation).
            class_weights: Class weights for loss computation.
            is_training: Whether this is a training epoch.
        Returns:
            Average loss for the epoch.
        """
        self.model.train(is_training)
        total_loss = 0.0
        num_batches = 0

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if is_training:
                    optimizer.zero_grad()

                with autocast():
                    outputs = self.model(images)
                    loss = weighted_bce_loss(outputs, labels, class_weights)

                if is_training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Get predictions from the model.
        Args:
            data_loader: Data loader for prediction
        Returns:
            torch.Tensor: Predicted class indices
        """
        self.model.eval()
        all_predictions = []

        for images, _ in data_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)


    @torch.no_grad()
    def predict_proba(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Get prediction probabilities from the model.
        Args:
            data_loader: Data loader for prediction
        Returns:
            torch.Tensor: Prediction probabilities
        """
        self.model.eval()
        all_probabilities = []

        for images, _ in data_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu())

        return torch.cat(all_probabilities)
