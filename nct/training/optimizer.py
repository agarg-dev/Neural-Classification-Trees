"""Optimizer utilities for training."""

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, List, Any, Optional, Union

try:
    from transformers import get_cosine_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def create_optimizer(
    parameters: Union[List[Dict], List[Any]],
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **optimizer_kwargs
) -> optim.Optimizer:
    """
    Create an optimizer with specified parameters.
    Args:
        parameters: Model parameters or parameter groups
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **optimizer_kwargs: Additional optimizer arguments
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adamw':
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **optimizer_kwargs
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **optimizer_kwargs
        )
    elif optimizer_type == 'sgd':
        momentum = optimizer_kwargs.pop('momentum', 0.9)
        return optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            **optimizer_kwargs
        )
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **optimizer_kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_two_stage_optimizer(
    model_parameters: Dict[str, List],
    base_lr: float,
    head_lr: float,
    optimizer_type: str = 'adamw',
    weight_decay: float = 1e-4,
    **optimizer_kwargs
) -> optim.Optimizer:
    """
    Create optimizer with different learning rates for backbone and heads.
    Args:
        model_parameters: Dict with 'base' and 'heads' parameter lists
        base_lr: Learning rate for backbone
        head_lr: Learning rate for heads
        optimizer_type: Optimizer type
        weight_decay: Weight decay
        **optimizer_kwargs: Additional optimizer arguments
    Returns:
        torch.optim.Optimizer: Configured optimizer with parameter groups
    """
    param_groups = [
        {'params': model_parameters['base'], 'lr': base_lr},
        {'params': model_parameters['heads'], 'lr': head_lr}
    ]

    return create_optimizer(
        param_groups,
        optimizer_type=optimizer_type,
        learning_rate=head_lr,  # Default LR (will be overridden by param groups)
        weight_decay=weight_decay,
        **optimizer_kwargs
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'plateau',
    **scheduler_kwargs
) -> Optional[Any]:
    """
    Create a learning rate scheduler.
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('plateau', 'step', 'cosine', 'none')
        **scheduler_kwargs: Additional scheduler arguments
    Returns:
        Optional scheduler object
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_kwargs.get('factor', 0.5),
            patience=scheduler_kwargs.get('patience', 3)
        )
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=scheduler_kwargs.get('step_size', 10),
            gamma=scheduler_kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_kwargs.get('T_max', 50),
            eta_min=scheduler_kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'cosine_warmup':
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for cosine_warmup scheduler. "
                "Install with: pip install transformers"
            )
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_kwargs.get('num_warmup_steps', 100),
            num_training_steps=scheduler_kwargs.get('num_training_steps', 1000)
        )
    elif scheduler_type == 'none':
        return None
    else:
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


def warmup_learning_rate(
    optimizer: optim.Optimizer,
    current_step: int,
    warmup_steps: int,
    base_lr: float
) -> None:
    """
    Apply learning rate warmup.
    Args:
        optimizer: PyTorch optimizer
        current_step: Current training step
        warmup_steps: Number of warmup steps
        base_lr: Target learning rate after warmup
    """
    if current_step < warmup_steps:
        warmup_factor = current_step / warmup_steps
        current_lr = base_lr * warmup_factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

            