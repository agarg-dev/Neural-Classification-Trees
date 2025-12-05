"""Optimizer utilities for training."""

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, List, Any, Optional, Union
from ..core.config import OptimizerConfig, SchedulerConfig

try:
    from transformers import get_cosine_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def create_optimizer(
    parameters: Union[List[Dict], List[Any]],
    config: OptimizerConfig,
    learning_rate: float = 1e-3,
    regularization_type: str = 'l2'
) -> optim.Optimizer:
    """
    Create an optimizer with specified parameters.

    Args:
        parameters: Model parameters or parameter groups
        config: OptimizerConfig object with all hyperparameters
        learning_rate: Learning rate (overrides config if provided)
        regularization_type: Type of regularization ('l1', 'l2', 'elastic_net', 'none')

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_type = config.type.lower()

    # Determine weight_decay based on regularization type
    # L2 and elastic_net use optimizer's weight_decay, L1 and none don't
    if regularization_type in ['l2', 'elastic_net']:
        weight_decay = config.weight_decay
    else:  # 'l1' or 'none'
        weight_decay = 0.0

    if optimizer_type == 'adamw':
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov
        )
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_two_stage_optimizer(
    model_parameters: Dict[str, List],
    base_lr: float,
    head_lr: float,
    config: OptimizerConfig,
    regularization_type: str = 'l2'
) -> optim.Optimizer:
    """
    Create optimizer with different learning rates for backbone and heads.

    Args:
        model_parameters: Dict with 'base' and 'heads' parameter lists
        base_lr: Learning rate for backbone
        head_lr: Learning rate for heads
        config: OptimizerConfig object with all hyperparameters
        regularization_type: Type of regularization ('l1', 'l2', 'elastic_net', 'none')

    Returns:
        torch.optim.Optimizer: Configured optimizer with parameter groups
    """
    param_groups = [
        {'params': model_parameters['base'], 'lr': base_lr},
        {'params': model_parameters['heads'], 'lr': head_lr}
    ]
    return create_optimizer(
        param_groups,
        config=config,
        learning_rate=head_lr,  # Default LR (will be overridden by param groups)
        regularization_type=regularization_type
    )



def create_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
) -> Optional[Any]:
    """
    Create a learning rate scheduler with dynamic hyperparameter calculation.

    Args:
        optimizer: PyTorch optimizer
        config: SchedulerConfig object with all hyperparameters
        num_epochs: Number of epochs for this training phase (used for auto-calculation)
        steps_per_epoch: Number of steps per epoch (used for step-based schedulers)

    Returns:
        Optional scheduler object

    Notes:
        - If num_epochs is provided, hyperparameters are auto-calculated based on training duration
        - Config values are used as fallbacks if auto-calculation isn't possible
        - For 'cosine_warmup', steps_per_epoch is required for auto-calculation
    """
    scheduler_type = config.type.lower()

    if scheduler_type == 'none':
        return None

    if scheduler_type == 'plateau':
        patience = config.patience if config.patience != -1 else int(config.plateau_patience_ratio * num_epochs)
        return ReduceLROnPlateau(optimizer, mode='min', factor=config.factor, patience=patience)

    if scheduler_type == 'step':
        step_size = config.step_size if config.step_size != -1 else int(config.step_size_ratio * num_epochs)
        return StepLR(optimizer, step_size=step_size, gamma=config.gamma)

    if scheduler_type == 'cosine':
        T_max = config.T_max if config.T_max != -1 else num_epochs
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.eta_min)

    if scheduler_type == 'cosine_warmup':
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for cosine_warmup scheduler. "
                "Install with: pip install transformers"
            )

        training_steps = config.num_training_steps if config.num_training_steps != -1 else num_epochs * steps_per_epoch
        warmup_steps = config.num_warmup_steps if config.num_warmup_steps != -1 else int(config.warmup_ratio * training_steps)

        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def adjust_learning_rate(
    optimizer: optim.Optimizer,
    decay_factor: float = 2.0,
    param_group_idx: Optional[int] = None
) -> None:
    """
    Manually adjust learning rate by dividing by decay factor.
    Useful for NCT's between-iteration LR decay: current_base_lr /= lr_decay_factor
    Args:
        optimizer: PyTorch optimizer
        decay_factor: Factor to divide learning rate by (e.g., 2.0 halves the LR)
        param_group_idx: Specific parameter group to adjust (None for all groups)
    Example:
        # Decay only backbone LR between NCT iterations
        adjust_learning_rate(optimizer, decay_factor=2.0, param_group_idx=0)
    """
    if param_group_idx is not None:
        # Adjust specific parameter group
        old_lr = optimizer.param_groups[param_group_idx]['lr']
        new_lr = old_lr / decay_factor
        optimizer.param_groups[param_group_idx]['lr'] = new_lr
        print(f"  Adjusted LR for group {param_group_idx}: {old_lr:.2e} -> {new_lr:.2e}")
    else:
        # Adjust all parameter groups
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = old_lr / decay_factor
            param_group['lr'] = new_lr
            print(f"  Adjusted LR for group {i}: {old_lr:.2e} -> {new_lr:.2e}")


def get_current_learning_rates(optimizer: optim.Optimizer) -> List[float]:
    """
    Get current learning rates from optimizer.
    Args:
        optimizer: PyTorch optimizer
    Returns:
        List[float]: Learning rates for each parameter group
    """
    return [param_group['lr'] for param_group in optimizer.param_groups]

