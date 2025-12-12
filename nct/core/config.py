"""Configuration classes for NCT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import yaml

VALID_OPTIMIZERS = ["adamw", "adam", "sgd", "rmsprop"]
VALID_SCHEDULERS = ["plateau", "step", "cosine", "cosine_warmup", "none"]


# =============================================================================
# Optimizer Configuration
# =============================================================================


@dataclass
class OptimizerConfig:
    """Optimizer configuration with all hyperparameters."""

    type: str = "adamw"
    weight_decay: float = 1e-4  # L2 regularization coefficient

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
    epochs_per_iteration: Optional[List[int]] = None
    batch_size: int = 32

    # Model parameters
    backbone: str = "resnet50"
    pretrained: bool = True
    head_hidden_dim: int = 256
    head_dropout: float = 0.3

    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initial_base_lr: float = 1e-5
    initial_head_lr: float = 1e-5
    lr_decay_factor: float = 2.0

    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss parameters
    use_class_weights: bool = True
    regularization_type: str = "l2"  # 'l1', 'l2', 'elastic_net', 'none'
    lambda_l1: float = 1e-4  # L1 regularization coefficient
    aux_loss_weight: float = 0.3  # Weight for auxiliary parent loss (0.0 = disabled)

    # Training strategy
    phase1_epochs_ratio: Optional[List[float]] = None
    phase1_patience: int = 10
    phase2_patience: int = 10
    early_stopping: bool = True
    save_iteration_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    attribution_dir: str = "./attributions"

    # Routing strategy
    routing_strategy: str = "argmax"  # 'argmax' or 'confidence'
    confidence_threshold: float = 0.7

    # Hierarchical architecture mode
    hierarchical_architecture: str = (
        "none"  # 'none', 'prediction_routing', 'correct_label_routing'
    )

    # Data parameters
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True

    # System parameters
    seed: Optional[int] = 42
    device: Optional[Union[str, torch.device]] = "auto"
    verbose: int = 1  # 0=silent, 1=progress, 2=detailed

    # Logging parameters
    log_attributions: bool = True
    log_epoch_metrics: bool = (
        True  # Compute accuracy/F1 per epoch (slower but detailed)
    )

    # Wandb integration (optional)
    wandb_enabled: bool = True
    wandb_project: str = "nct-waterbirds"
    wandb_entity: Optional[str] = "ankurgarg-university-of-calgary-in-alberta"
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Generate defaults based on num_iterations if not provided
        if self.epochs_per_iteration is None:
            self.epochs_per_iteration = [50] * self.num_iterations
        if self.phase1_epochs_ratio is None:
            self.phase1_epochs_ratio = [0.5] * self.num_iterations

        assert self.phase1_epochs_ratio is not None
        assert self.epochs_per_iteration is not None
        # Validate lengths
        if len(self.epochs_per_iteration) != self.num_iterations:
            raise ValueError(
                f"Length of epochs_per_iteration ({len(self.epochs_per_iteration)}) "
                f"must match num_iterations ({self.num_iterations})"
            )

        # Validate phase1_epochs_ratio length
        if len(self.phase1_epochs_ratio) != self.num_iterations:
            raise ValueError(
                f"phase1_epochs_ratio length ({len(self.phase1_epochs_ratio)}) must match num_iterations ({self.num_iterations})"
            )

        valid_reg_types = ["l1", "l2", "elastic_net", "none"]
        if self.regularization_type not in valid_reg_types:
            raise ValueError(
                f"regularization_type must be one of {valid_reg_types}, got '{self.regularization_type}'"
            )

        valid_routing_strategies = ["argmax", "confidence"]
        if self.routing_strategy not in valid_routing_strategies:
            raise ValueError(
                f"routing_strategy must be one of {valid_routing_strategies}, got '{self.routing_strategy}'"
            )

        valid_hierarchical_modes = [
            "none",
            "prediction_routing",
            "correct_label_routing",
        ]
        if self.hierarchical_architecture not in valid_hierarchical_modes:
            raise ValueError(
                f"hierarchical_architecture must be one of {valid_hierarchical_modes}, got '{self.hierarchical_architecture}'"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NCTConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "NCTConfig":
        """Load configuration from YAML file with optional overrides."""

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Handle nested configs
        if "optimizer" in config_dict and isinstance(config_dict["optimizer"], dict):
            config_dict["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
        if "scheduler" in config_dict and isinstance(config_dict["scheduler"], dict):
            config_dict["scheduler"] = SchedulerConfig(**config_dict["scheduler"])

        # Apply overrides
        config_dict.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, "to_dict"):
                result[field_name] = value.to_dict()
            elif hasattr(value, "__dataclass_fields__"):
                result[field_name] = {
                    f.name: getattr(value, f.name)
                    for f in value.__dataclass_fields__.values()
                }
            else:
                result[field_name] = value
        return result
