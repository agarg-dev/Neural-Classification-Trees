"""Optimizer + scheduler utilities."""

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
    """Build an optimizer from OptimizerConfig.

    L2 and elastic-net push weight decay into the optimizer; L1/none rely on
    the explicit L1 penalty in the loss and pass weight_decay=0 here.
    """
    optimizer_type = config.type.lower()

    if regularization_type in ['l2', 'elastic_net']:
        weight_decay = config.weight_decay
    else:
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
    """Optimizer with separate param groups: 'base' (backbone + parent_layers) @ base_lr, 'heads' @ head_lr."""
    param_groups = [
        {'params': model_parameters['base'], 'lr': base_lr},
        {'params': model_parameters['heads'], 'lr': head_lr}
    ]
    return create_optimizer(
        param_groups,
        config=config,
        learning_rate=head_lr,  # placeholder; overridden by per-group lr above
        regularization_type=regularization_type
    )



def create_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
) -> Optional[Any]:
    """Build an LR scheduler. Fields left at -1 in SchedulerConfig auto-calc from num_epochs."""
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
    """Divide LR by decay_factor (one param group or all). Used for NCT's per-iteration LR decay."""
    if param_group_idx is not None:
        old_lr = optimizer.param_groups[param_group_idx]['lr']
        new_lr = old_lr / decay_factor
        optimizer.param_groups[param_group_idx]['lr'] = new_lr
        print(f"  Adjusted LR for group {param_group_idx}: {old_lr:.2e} -> {new_lr:.2e}")
    else:
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = old_lr / decay_factor
            param_group['lr'] = new_lr
            print(f"  Adjusted LR for group {i}: {old_lr:.2e} -> {new_lr:.2e}")


def get_current_learning_rates(optimizer: optim.Optimizer) -> List[float]:
    """Return current LR for each param group."""
    return [param_group['lr'] for param_group in optimizer.param_groups]
