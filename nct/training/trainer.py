"""NCTTrainer: two-phase training (frozen backbone, then full fine-tuning) for hierarchical models."""

import copy
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ..core.config import OptimizerConfig, SchedulerConfig
from ..evaluation.metrics import compute_consolidated_group_metrics
from ..utils.hierarchical import get_num_nodes
from .loss import (
    compute_auxiliary_loss,
    compute_l1_penalty,
    compute_l1_penalty_heads,
    weighted_loss,
)
from .optimizer import create_optimizer, create_scheduler, create_two_stage_optimizer


class NCTTrainer:
    """Trainer for hierarchical models with two-phase training strategy.

    Phase 1: Train output heads with frozen base (backbone + parent layers).
    Phase 2: Fine-tune the entire model (base with base_lr, output heads with head_lr).
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        device: torch.device,
        config: Dict[str, Any],
        logger,
        iteration_num: int = 1,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger
        self.iteration_num = iteration_num

        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = GradScaler(self.device_type)

        # Early stopping tracking
        self.best_val_loss = float("inf")
        self.best_worst_group_acc = 0.0
        self.best_iter2_level_wga = 0.0
        self.best_iter2_level_auroc = 0.0
        self.best_difficult_node_acc = 0.0
        self.patience_counter = 0
        self.best_model_weights = None
        self.total_epochs_trained = 0

        # Auxiliary loss tracking
        self._last_train_aux_loss = 0.0
        self._last_val_aux_loss = 0.0

        # (pseudo_wga, true_wga) pairs per epoch for Spearman correlation
        self._wga_correlation_data: list = []
        # Same, against bias-collapsed true WGA (CMNIST-style only)
        self._wga_correlation_data_consolidated: list = []
        self.val_group_labels = None  # set externally by NCTClassifier

    def train_iteration(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        base_lr: float,
        head_lr: float,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Train one iteration via the two-phase approach.

        Returns a dict with 'phase1_results', 'phase2_results', 'best_val_loss',
        'total_epochs', 'early_stopped'.
        """
        val_class_weights = (
            torch.ones_like(class_weights) if class_weights is not None else None
        )

        # Phase 1: frozen backbone, train heads only (early stopping is per-phase)
        self._reset_early_stopping()

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
        phase1_best_iter2_wga = self.best_iter2_level_wga
        phase1_best_iter2_auroc = self.best_iter2_level_auroc
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
        phase2_best_iter2_wga = self.best_iter2_level_wga
        phase2_best_iter2_auroc = self.best_iter2_level_auroc
        phase2_best_dna = self.best_difficult_node_acc
        phase2_best_weights = copy.deepcopy(self.best_model_weights)

        # Load best model from either phase based on the early stopping criterion
        criterion = self.config.get("model_selection_criterion", "auto")
        use_wga_for_selection = (criterion == "wga") or (
            criterion == "auto" and self.iteration_num > 1
        )
        use_iter2_wga_for_selection = (criterion == "iter2_wga") and (
            self.iteration_num > 1
        )
        use_iter2_auroc_for_selection = (criterion == "iter2_auroc") and (
            self.iteration_num > 1
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
        elif use_iter2_auroc_for_selection:
            # Higher is better for iter2-level AUROC
            if phase1_best_iter2_auroc > phase2_best_iter2_auroc:
                self.model.load_state_dict(phase1_best_weights)
                self.best_model_weights = phase1_best_weights
            else:
                self.model.load_state_dict(phase2_best_weights)
                self.best_model_weights = phase2_best_weights
        elif use_iter2_wga_for_selection:
            # Higher is better for iter2-level worst-group accuracy
            if phase1_best_iter2_wga > phase2_best_iter2_wga:
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
        """Reset early-stopping trackers for a new phase."""
        self.best_val_loss = float("inf")
        # Initialise to -1 so the first epoch always wins and saves weights.
        self.best_worst_group_acc = -1.0
        self.best_iter2_level_wga = -1.0
        self.best_iter2_level_auroc = -1.0
        self.best_difficult_node_acc = -1.0
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
        """Phase 1: train output heads only (backbone + parent layers frozen)."""
        if epochs <= 0:
            return {"epochs": 0, "train_losses": [], "val_losses": []}

        self.logger.phase_start(
            "Phase 1: Training output heads (backbone + parent layers frozen)", 1, epochs
        )

        for param in self.model.base.parameters():
            param.requires_grad = False

        # Parent layers exist only in hierarchical mode (empty ModuleList otherwise).
        for parent_layer in self.model.parent_layers:
            for param in parent_layer.parameters():
                param.requires_grad = False

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
        """Phase 2: full-model fine-tuning with separate base/head learning rates."""
        if epochs <= 0:
            return {"epochs": 0, "train_losses": [], "val_losses": []}

        self.logger.phase_start("Phase 2: Fine-tuning the full model", 2, epochs)

        for param in self.model.base.parameters():
            param.requires_grad = True

        for parent_layer in self.model.parent_layers:
            for param in parent_layer.parameters():
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
        """Train `epochs` epochs with the given optimizer/scheduler, with early stopping."""
        train_losses = []
        val_losses = []

        if total_epochs is None:
            total_epochs = epochs

        phase_num = 1 if "Phase 1" in phase_name else 2
        log_metrics = self.config["log_epoch_metrics"]

        criterion = self.config.get("model_selection_criterion", "auto")
        if criterion == "val_loss":
            self.logger.log("    Early stopping criterion: val_loss", level=2)
        elif criterion == "wga":
            self.logger.log(
                "    Early stopping criterion: pseudo WGA (filled groups)", level=2
            )
        elif criterion == "iter2_wga":
            if self.iteration_num == 1:
                self.logger.log(
                    "    Early stopping criterion: val_loss (iter2_wga: iteration 1)",
                    level=2,
                )
            else:
                self.logger.log(
                    "    Early stopping criterion: iteration-2-level pseudo WGA (filled groups)",
                    level=2,
                )
        elif criterion == "iter2_auroc":
            if self.iteration_num == 1:
                self.logger.log(
                    "    Early stopping criterion: val_loss (iter2_auroc: iteration 1)",
                    level=2,
                )
            else:
                self.logger.log(
                    "    Early stopping criterion: iteration-2-level pseudo AUROC",
                    level=2,
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
                    iter2_level_wga=(
                        val_result.get("iter2_level_worst_group_accuracy_filled")
                        if self.iteration_num >= 3
                        else None
                    ),
                    iter2_level_auroc=(
                        val_result.get("iter2_level_worst_group_auroc")
                        if self.iteration_num >= 2
                        else None
                    ),
                    iter2_level_dna=(
                        val_result.get("iter2_level_difficult_node_accuracy")
                        if self.iteration_num >= 3
                        else None
                    ),
                    iter2_level_dnf1=(
                        val_result.get("iter2_level_difficult_node_f1")
                        if self.iteration_num >= 3
                        else None
                    ),
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
            # - "iter2_wga": val_loss for iteration 1, iter2-level pseudo WGA for iteration 2+
            # - "iter2_auroc": val_loss for iteration 1, iter2-level pseudo AUROC for iteration 2+
            # - "dna": val_loss for iteration 1, difficult node accuracy for iteration 2+
            # - "auto": val_loss for iteration 1, wga for iteration 2+
            criterion = self.config.get("model_selection_criterion", "auto")

            use_wga = False
            use_iter2_wga = False
            use_iter2_auroc = False
            use_dna = False

            if criterion == "val_loss":
                pass
            elif criterion == "wga":
                use_wga = "worst_group_accuracy_filled" in val_result
            elif criterion == "iter2_wga":
                use_iter2_wga = (
                    self.iteration_num > 1
                ) and "iter2_level_worst_group_accuracy_filled" in val_result
            elif criterion == "iter2_auroc":
                use_iter2_auroc = (
                    self.iteration_num > 1
                ) and "iter2_level_worst_group_auroc" in val_result
            elif criterion == "dna":
                # DNA uses val_loss for iteration 1, DNA for iteration 2+
                use_dna = (
                    self.iteration_num > 1
                ) and "difficult_node_accuracy" in val_result
            else:  # "auto"
                use_wga = (
                    self.iteration_num > 1
                ) and "worst_group_accuracy_filled" in val_result

            # For non-loss criteria: higher metric is better; also stash val_loss for logging.
            if use_dna:
                dna = val_result["difficult_node_accuracy"]
                if dna > self.best_difficult_node_acc:
                    self.best_difficult_node_acc = dna
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
            elif use_iter2_auroc:
                iter2_auroc = val_result["iter2_level_worst_group_auroc"]
                if math.isnan(iter2_auroc):
                    iter2_auroc = 0.0
                if iter2_auroc > self.best_iter2_level_auroc:
                    self.best_iter2_level_auroc = iter2_auroc
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
            elif use_iter2_wga:
                iter2_wga = val_result["iter2_level_worst_group_accuracy_filled"]
                if iter2_wga > self.best_iter2_level_wga:
                    self.best_iter2_level_wga = iter2_wga
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
            elif use_wga:
                worst_group_acc = val_result["worst_group_accuracy_filled"]
                if worst_group_acc > self.best_worst_group_acc:
                    self.best_worst_group_acc = worst_group_acc
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
            else:
                # val_loss: lower is better
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
                iter2_level_wga=(
                    val_result.get("iter2_level_worst_group_accuracy_filled")
                    if self.iteration_num >= 3
                    else None
                ),
                iter2_level_auroc=(
                    val_result.get("iter2_level_worst_group_auroc")
                    if self.iteration_num >= 2
                    else None
                ),
                iter2_level_dna=(
                    val_result.get("iter2_level_difficult_node_accuracy")
                    if self.iteration_num >= 3
                    else None
                ),
                iter2_level_dnf1=(
                    val_result.get("iter2_level_difficult_node_f1")
                    if self.iteration_num >= 3
                    else None
                ),
                iteration_num=self.iteration_num,
            )

            # Skip iter 1: pseudo-WGA is class-imbalance, not the Easy/Difficult proxy
            if "val_true_wga" in val_result and self.iteration_num >= 2:
                criterion = self.config.get("model_selection_criterion", "auto")
                use_auroc = (criterion == "iter2_auroc")
                if use_auroc:
                    pseudo_corr = val_result.get("iter2_level_worst_group_auroc")
                    if pseudo_corr is not None and math.isnan(pseudo_corr):
                        pseudo_corr = None
                    true_corr = val_result.get("val_true_auroc")
                    true_corr_consolidated = None
                elif self.iteration_num >= 3 and "iter2_level_worst_group_accuracy" in val_result:
                    pseudo_corr = val_result["iter2_level_worst_group_accuracy"]
                    true_corr = val_result["val_true_wga"]
                    true_corr_consolidated = val_result.get("val_true_wga_consolidated")
                elif "worst_group_accuracy" in val_result:
                    pseudo_corr = val_result["worst_group_accuracy"]
                    true_corr = val_result["val_true_wga"]
                    true_corr_consolidated = val_result.get("val_true_wga_consolidated")
                else:
                    pseudo_corr = None
                    true_corr = None
                    true_corr_consolidated = None
                if pseudo_corr is not None and true_corr is not None:
                    self._wga_correlation_data.append((pseudo_corr, true_corr))
                    if true_corr_consolidated is not None:
                        self._wga_correlation_data_consolidated.append(
                            (pseudo_corr, true_corr_consolidated)
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
        """Run one training epoch; returns mean loss."""
        self.model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0

        # Auxiliary loss is only meaningful in tree mode and after the first iteration.
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
                if use_aux_loss:
                    outputs, aux_outputs = self.model(images, return_aux_outputs=True)
                else:
                    outputs = self.model(images)

                num_base_classes = self.config.get("num_classes", 2)
                loss = weighted_loss(outputs, labels, class_weights, num_base_classes)

                if use_aux_loss:
                    aux_loss = compute_auxiliary_loss(
                        aux_outputs,
                        labels,
                        self.iteration_num,
                        class_weights,
                        max_weight=self.config.get("max_class_weight"),
                        num_classes=num_base_classes,
                    )
                    loss = loss + aux_weight * aux_loss
                    total_aux_loss += aux_loss.item()

                # L1 penalty applies for both pure-L1 and elastic-net regularisation.
                if self.config["regularization_type"] in ["l1", "elastic_net"]:
                    lambda_l1 = self.config["lambda_l1"]
                    if lambda_l1 > 0:
                        loss += lambda_l1 * compute_l1_penalty_heads(self.model)


            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

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
        """Run one validation epoch; returns a dict of loss + pseudo-group metrics."""
        self.model.eval()
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        all_probs = []  # only populated when compute_pseudo_auroc is True

        compute_pseudo_auroc = (
            self.config.get("model_selection_criterion", "auto") == "iter2_auroc"
        )

        # Aux loss only applies in tree mode after the first iteration.
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
                    if use_aux_loss:
                        outputs, aux_outputs = self.model(
                            images, return_aux_outputs=True
                        )
                    else:
                        outputs = self.model(images)

                    num_base_classes = self.config.get("num_classes", 2)
                    loss = weighted_loss(outputs, labels, class_weights, num_base_classes)

                    # Aux loss is tracked separately on val so it doesn't perturb the loss metric.
                    if use_aux_loss:
                        aux_loss = compute_auxiliary_loss(
                            aux_outputs,
                            labels,
                            self.iteration_num,
                            class_weights,
                            max_weight=self.config.get("max_class_weight"),
                            num_classes=num_base_classes,
                        )
                        total_aux_loss += aux_loss.item()

                total_loss += loss.item()
                num_batches += 1

                # Always collect predictions for pseudo worst-group accuracy.
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                if compute_pseudo_auroc:
                    all_probs.append(torch.softmax(outputs, dim=1).cpu())

        self._last_val_aux_loss = (
            total_aux_loss / num_batches if use_aux_loss and num_batches > 0 else 0.0
        )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        result = {"loss": avg_loss}

        # Pseudo worst-group accuracy: each hierarchical node is treated as a pseudo group.
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            num_classes = self.config.get('num_classes', 2)
            num_groups = get_num_nodes(self.iteration_num, num_classes)

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

            # Iteration-2-level collapsed pseudo WGA (e.g., 8->4, 16->4 for binary).
            base_num_classes = self.config.get("num_classes", 2)
            iter2_expected_groups = int(base_num_classes) * 2
            if (
                num_groups >= iter2_expected_groups
                and num_groups % iter2_expected_groups == 0
            ):
                all_labels_np = all_labels.numpy()
                all_preds_np = all_preds.numpy()
                collapse_factor = num_groups // iter2_expected_groups
                collapsed_labels = all_labels_np // collapse_factor
                collapsed_preds = all_preds_np // collapse_factor

                iter2_group_accuracies = []
                iter2_group_accuracies_filled = []
                iter2_node_sizes = {}
                iter2_per_node_accuracy = {}
                min_required = max(1, self.config.get("min_samples_per_node", 0))

                for group_id in range(iter2_expected_groups):
                    group_mask = collapsed_labels == group_id
                    group_count = int(group_mask.sum())
                    iter2_node_sizes[group_id] = group_count

                    if group_count > 0:
                        group_correct = (collapsed_preds[group_mask] == collapsed_labels[group_mask]).sum()
                        group_acc = float(group_correct / group_count)
                        iter2_group_accuracies.append(group_acc)
                        iter2_per_node_accuracy[group_id] = group_acc
                        if group_count >= min_required:
                            iter2_group_accuracies_filled.append(group_acc)
                    else:
                        iter2_group_accuracies.append(0.0)
                        iter2_per_node_accuracy[group_id] = 0.0

                result["iter2_level_worst_group_accuracy"] = min(iter2_group_accuracies)
                result["iter2_level_worst_group_accuracy_filled"] = (
                    min(iter2_group_accuracies_filled)
                    if len(iter2_group_accuracies_filled) > 0
                    else 0.0
                )
                result["iter2_level_num_groups_present"] = len(
                    [x for x in iter2_node_sizes.values() if x > 0]
                )
                result["iter2_level_num_groups_expected"] = iter2_expected_groups
                result["iter2_level_node_sizes"] = iter2_node_sizes
                result["iter2_level_per_node_accuracy"] = iter2_per_node_accuracy

                # Iter-2-level DNA/DNF1
                iter2_difficult_mask = (collapsed_labels % 2 == 1)
                if iter2_difficult_mask.sum() > 0:
                    iter2_diff_correct = (
                        collapsed_preds[iter2_difficult_mask]
                        == collapsed_labels[iter2_difficult_mask]
                    ).sum()
                    iter2_dna = float(iter2_diff_correct / iter2_difficult_mask.sum())
                    iter2_diff_labels = np.unique(collapsed_labels[iter2_difficult_mask])
                    iter2_dnf1 = float(f1_score(
                        collapsed_labels[iter2_difficult_mask],
                        collapsed_preds[iter2_difficult_mask],
                        labels=iter2_diff_labels,
                        average="macro",
                        zero_division=1.0,
                    ))
                else:
                    iter2_dna = 1.0
                    iter2_dnf1 = 1.0
                result["iter2_level_difficult_node_accuracy"] = iter2_dna
                result["iter2_level_difficult_node_f1"] = iter2_dnf1

                # Iter-2-level pseudo AUROC (one-vs-rest per collapsed group).
                if compute_pseudo_auroc and len(all_probs) > 0:
                    all_probs_np = torch.cat(all_probs).numpy()
                    collapsed_probs = np.zeros(
                        (len(collapsed_labels), iter2_expected_groups), dtype=np.float64
                    )
                    for j in range(iter2_expected_groups):
                        s = j * collapse_factor
                        e = (j + 1) * collapse_factor
                        collapsed_probs[:, j] = all_probs_np[:, s:e].sum(axis=1)

                    iter2_auroc_scores = []
                    for j in range(iter2_expected_groups):
                        binary_labels = (collapsed_labels == j).astype(int)
                        if 0 < binary_labels.sum() < len(binary_labels):
                            try:
                                iter2_auroc_scores.append(
                                    roc_auc_score(binary_labels, collapsed_probs[:, j])
                                )
                            except ValueError:
                                pass
                    result["iter2_level_worst_group_auroc"] = (
                        float(min(iter2_auroc_scores))
                        if iter2_auroc_scores
                        else float("nan")
                    )
            else:
                all_labels_np = all_labels.numpy()
                all_preds_np = all_preds.numpy()

            # Compute difficult node accuracy (DNA) and F1
            # Difficult nodes have odd indices (1, 3, 5, 7, ...)
            difficult_mask = (all_labels_np % 2 == 1)

            if difficult_mask.sum() > 0:
                difficult_correct = (all_preds_np[difficult_mask] == all_labels_np[difficult_mask]).sum()
                difficult_total = difficult_mask.sum()
                dna = float(difficult_correct / difficult_total)
                # Restrict labels to odd indices so easy classes don't drag macro F1 to 0
                difficult_labels = np.unique(all_labels_np[difficult_mask])
                dnf1 = float(f1_score(
                    all_labels_np[difficult_mask],
                    all_preds_np[difficult_mask],
                    labels=difficult_labels,
                    average="macro",
                    zero_division=1.0
                ))
            else:
                dna = 1.0  # No difficult samples = perfect routing
                dnf1 = 1.0

            result["difficult_node_accuracy"] = dna
            result["difficult_node_f1"] = dnf1

            # Compute true WGA using real group labels (if provided by NCTClassifier)
            if self.val_group_labels is not None:
                nodes_per_class = max(1, num_groups // num_classes)
                class_pred_true_wga = all_preds_np // nodes_per_class
                class_true_true_wga = all_labels_np // nodes_per_class
                vg = self.val_group_labels
                unique_groups = np.unique(vg)
                grp_accs = []
                for g in unique_groups:
                    mask = vg == g
                    if mask.sum() > 0:
                        grp_accs.append(
                            float((class_pred_true_wga[mask] == class_true_true_wga[mask]).mean())
                        )
                if grp_accs:
                    result["val_true_wga"] = min(grp_accs)

                # Bias-collapsed true WGA (2*num_classes cells); None unless colors_per_class > 2
                consolidated = compute_consolidated_group_metrics(
                    vg,
                    class_pred_true_wga,
                    num_classes=num_classes,
                    num_groups=int(len(unique_groups)),
                )
                if consolidated is not None:
                    result["val_true_wga_consolidated"] = float(consolidated["worst_group_accuracy"])

                if (
                    compute_pseudo_auroc
                    and num_classes == 2
                    and len(all_probs) > 0
                    and len(np.unique(class_true_true_wga)) == 2
                ):
                    score_pos = torch.cat(all_probs).numpy()[:, nodes_per_class:].sum(axis=1)
                    try:
                        result["val_true_auroc"] = float(
                            roc_auc_score(class_true_true_wga, score_pos)
                        )
                    except ValueError:
                        pass

        return result
