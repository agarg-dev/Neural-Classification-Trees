"""Configuration classes for NCT."""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import torch


VALID_OPTIMIZERS = ["adamw", "adam", "sgd", "rmsprop"]
VALID_SCHEDULERS = ["plateau", "step", "cosine", "cosine_warmup", "none"]


# =============================================================================
# Optimizer Configuration
# =============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration with all hyperparameters."""

    type: str = "adamw"
    weight_decay: float = 1e-4

    # SGD-specific
    momentum: float = 0.9
    nesterov: bool = True

    # Adam/AdamW-specific
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self):
        """Validate optimizer configuration."""
        if self.type not in VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer.type must be one of {VALID_OPTIMIZERS}, got '{self.type}'"
            )


# =============================================================================
# Scheduler Configuration
# =============================================================================

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration with all hyperparameters.

    Note: Use -1 as a sentinel value to enable auto-calculation based on training duration.
          Set explicit positive values to override auto-calculation.
    """

    type: str = "plateau"

    # ReduceLROnPlateau-specific
    patience: int = -1
    factor: float = 0.5

    # StepLR-specific
    step_size: int = -1
    gamma: float = 0.1

    # CosineAnnealingLR-specific
    T_max: int = -1
    eta_min: float = 0.0

    # CosineWarmup-specific
    num_warmup_steps: int = -1
    num_training_steps: int = -1

    # Auto-calculation ratios (used when field values are -1)
    plateau_patience_ratio: float = 0.10
    step_size_ratio: float = 0.30
    warmup_ratio: float = 0.10

    def __post_init__(self):
        """Validate scheduler configuration."""
        if self.type not in VALID_SCHEDULERS:
            raise ValueError(
                f"scheduler.type must be one of {VALID_SCHEDULERS}, got '{self.type}'"
            )


# =============================================================================
# Main NCT Configuration
# =============================================================================

@dataclass
class NCTConfig:
    """Configuration class for NCT classifier."""

    # Training parameters
    num_iterations: int = 3
    epochs_per_iteration: List[int] = field(default_factory=lambda: [50, 50, 50])
    batch_size: int = 32

    # Model parameters
    backbone: str = "resnet50"
    pretrained: bool = True
    head_hidden_dim: int = 64
    head_dropout: float = 0.5

    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initial_base_lr: float = 1e-5
    initial_head_lr: float = 1e-5
    lr_decay_factor: float = 2.0

    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss parameters
    use_class_weights: bool = True

    # Training strategy
    phase1_epochs_ratio: float = 0.5
    phase1_patience: int = 10
    phase2_patience: int = 10
    early_stopping: bool = True

    # Data parameters
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True

    # System parameters
    seed: Optional[int] = 42
    device: Optional[Union[str, torch.device]] = 'auto'
    verbose: int = 1  # 0=silent, 1=progress, 2=detailed

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.epochs_per_iteration) != self.num_iterations:
            raise ValueError(
                f"Length of epochs_per_iteration ({len(self.epochs_per_iteration)}) "
                f"must match num_iterations ({self.num_iterations})"
            )

        if self.phase1_epochs_ratio <= 0 or self.phase1_epochs_ratio >= 1:
            raise ValueError("phase1_epochs_ratio must be between 0 and 1")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'NCTConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, 'to_dict'):
                result[field_name] = value.to_dict()
            elif hasattr(value, '__dataclass_fields__'):
                result[field_name] = {
                    f.name: getattr(value, f.name)
                    for f in value.__dataclass_fields__.values()
                }
            else:
                result[field_name] = value
        return result
