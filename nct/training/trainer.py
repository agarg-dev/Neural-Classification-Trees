"""
Core Training Functionality for NCT.

This module implements the NCTTrainer class which handles the two-phase
training strategy (frozen backbone, then full fine-tuning) for hierarchical models.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Optional, Dict, Any
import numpy as np

from .loss import weighted_bce_loss, compute_class_weights
from .optimizer import create_optimizer, create_scheduler, create_two_stage_optimizer
from ..core.config import OptimizerConfig, SchedulerConfig


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
        logger
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

        # Use appropriate device for mixed precision training
        self.device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(self.device_type)

        # Early stopping tracking
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

        # Phase 1 training with its own early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

        phase1_epochs = max(1, int(epochs * self.config['phase1_epochs_ratio']))
        phase1_results = self._train_phase1(
            train_loader, val_loader, phase1_epochs, head_lr,
            class_weights, val_class_weights, total_epochs=epochs
        )
        phase1_best_loss = self.best_val_loss

        # Load best weights from phase 1 before starting phase 2
        self.model.load_state_dict(self.best_model_weights)

        # Phase 2 training with separate early stopping (reset counters)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

        phase2_results = self._train_phase2(
            train_loader, val_loader, epochs - phase1_epochs, base_lr, head_lr,
            class_weights, val_class_weights, total_epochs=epochs, epoch_offset=phase1_epochs
        )
        phase2_best_loss = self.best_val_loss

        # Load best weights from phase 2
        self.model.load_state_dict(self.best_model_weights)

        return {
            'phase1_results': phase1_results,
            'phase2_results': phase2_results,
            'best_val_loss': min(phase1_best_loss, phase2_best_loss),
            'total_epochs': self.total_epochs_trained,
            'early_stopped': phase1_results.get('early_stopped', False) or phase2_results.get('early_stopped', False)
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
        self.logger.phase_start("Phase 1: Training output heads (frozen base)", 1, epochs)

        for param in self.model.base.parameters():
            param.requires_grad = False

        optimizer_config = OptimizerConfig(**self.config.get('optimizer'))
        optimizer = create_optimizer(
            self.model.output_heads.parameters(),
            config=optimizer_config,
            learning_rate=head_lr
        )

        scheduler_config = SchedulerConfig(**self.config.get('scheduler'))
        scheduler = create_scheduler(
            optimizer,
            config=scheduler_config,
            num_epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        return self._train_epochs(
            train_loader, val_loader, epochs, optimizer, scheduler,
            class_weights, val_class_weights, phase_name="Phase 1",
            total_epochs=total_epochs, epoch_offset=0,
            patience=self.config['phase1_patience']
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

        self.logger.phase_start("Phase 2: Fine-tuning the full model", 2, epochs)

        for param in self.model.base.parameters():
            param.requires_grad = True

        model_params = self.model.get_trainable_parameters()
        optimizer_config = OptimizerConfig(**self.config.get('optimizer'))
        optimizer = create_two_stage_optimizer(
            model_params,
            base_lr=base_lr,
            head_lr=head_lr,
            config=optimizer_config
        )

        scheduler_config = SchedulerConfig(**self.config.get('scheduler'))
        scheduler = create_scheduler(
            optimizer,
            config=scheduler_config,
            num_epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        return self._train_epochs(
            train_loader, val_loader, epochs, optimizer, scheduler,
            class_weights, val_class_weights, phase_name="Phase 2",
            total_epochs=total_epochs, epoch_offset=epoch_offset,
            patience=self.config['phase2_patience']
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
        patience: int = 10
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

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, class_weights)
            train_losses.append(train_loss)

            val_loss = self._validate_epoch(val_loader, val_class_weights)
            val_losses.append(val_loss)

            if scheduler is not None:
                # ReduceLROnPlateau needs a metric, others don't
                scheduler_type = self.config['scheduler']['type']
                if scheduler_type == 'plateau':
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
            self.logger.epoch_update(current_epoch, total_epochs, train_loss, val_loss)

            if self.config['early_stopping'] and self.patience_counter >= patience:
                self.logger.early_stop(phase_name, phase_num)
                break

        self.total_epochs_trained += len(train_losses)

        completed_all = len(train_losses) == epochs and self.patience_counter < patience
        self.logger.phase_end(phase_num, completed_all)

        return {
            'epochs': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'early_stopped': len(train_losses) < epochs
        }


    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        class_weights: Optional[torch.Tensor]
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
        num_batches = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            with autocast(self.device_type):
                outputs = self.model(images)
                loss = weighted_bce_loss(outputs, labels, class_weights)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0
    

    def _validate_epoch(
        self,
        loader: DataLoader,
        class_weights: Optional[torch.Tensor]
    ) -> float:
        """
        Execute a single validation epoch.
        Args:
            loader: Validation data loader.
            class_weights: Class weights for loss computation.
        Returns:
            Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with autocast(self.device_type):
                    outputs = self.model(images)
                    loss = weighted_bce_loss(outputs, labels, class_weights)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0
