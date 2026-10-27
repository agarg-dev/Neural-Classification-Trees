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
    weight_decay: float = 1e-4

    momentum: float = 0.9
    nesterov: bool = True

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
    """Learning rate scheduler configuration. Use -1 to auto-calculate from training duration."""

    type: str = "plateau"

    patience: int = -1
    factor: float = 0.5

    step_size: int = -1
    gamma: float = 0.1

    T_max: int = -1
    eta_min: float = 0.0

    num_warmup_steps: int = -1
    num_training_steps: int = -1

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

    dfr_style: bool = False

    num_iterations: int = 3
    epochs_per_iteration: Optional[List[int]] = None
    batch_size: int = 32

    backbone: str = "resnet50"
    pretrained: bool = True
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    head_type: str = "mlp"
    iter1_head_type: Optional[str] = None
    split_node_head_type: Union[bool, str] = False
    head_batch_norm: bool = True
    head_bn_affine: bool = False
    num_channels: int = 3
    num_classes: int = 2

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initial_base_lr: float = 1e-5
    initial_head_lr: float = 1e-5
    lr_decay_factor: float = 2.0

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    use_class_weights: bool = True
    skip_class_weights_iter1: bool = False
    max_class_weight: Optional[float] = None
    regularization_type: str = "l2"
    lambda_l1: float = 1e-4
    aux_loss_weight: float = 0.3
    loss_type: str = "bce"

    head_ensemble_size: int = 1

    phase1_epochs_ratio: Optional[List[float]] = None
    phase1_patience: int = 10
    phase2_patience: int = 10
    early_stopping: bool = True
    model_selection_criterion: str = "auto"
    save_iteration_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    attribution_dir: str = "./attributions"

    routing_strategy: str = "correctness"

    depth_selection_enabled: bool = False
    depth_selection_confidence: float = 1.96

    sampling_strategy: str = "none"

    min_samples_per_node: int = 0

    hierarchical_architecture: str = "correct_label_routing"

    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True
    cache_images_in_memory: bool = False

    seed: Optional[int] = 42
    device: Optional[Union[str, torch.device]] = "auto"
    verbose: int = 1

    log_attributions: bool = True
    log_epoch_metrics: bool = True

    wandb_enabled: bool = False
    wandb_project: str = "nct"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs_per_iteration is None:
            self.epochs_per_iteration = [50] * self.num_iterations
        if self.phase1_epochs_ratio is None:
            self.phase1_epochs_ratio = [0.5] * self.num_iterations

        assert self.phase1_epochs_ratio is not None
        assert self.epochs_per_iteration is not None

        if len(self.epochs_per_iteration) != self.num_iterations:
            raise ValueError(
                f"Length of epochs_per_iteration ({len(self.epochs_per_iteration)}) "
                f"must match num_iterations ({self.num_iterations})"
            )

        if len(self.phase1_epochs_ratio) != self.num_iterations:
            raise ValueError(
                f"phase1_epochs_ratio length ({len(self.phase1_epochs_ratio)}) must match num_iterations ({self.num_iterations})"
            )

        valid_loss_types = ["bce", "ce"]
        if self.loss_type not in valid_loss_types:
            raise ValueError(
                f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'"
            )

        valid_reg_types = ["l1", "l2", "elastic_net", "none"]
        if self.regularization_type not in valid_reg_types:
            raise ValueError(
                f"regularization_type must be one of {valid_reg_types}, got '{self.regularization_type}'"
            )

        if self.routing_strategy != "correctness":
            raise ValueError(
                f"routing_strategy must be 'correctness', got '{self.routing_strategy}'"
            )

        valid_sampling_strategies = ["none", "upsample", "downsample", "hybrid", "geomean"]
        if self.sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"sampling_strategy must be one of {valid_sampling_strategies}, got '{self.sampling_strategy}'"
            )

        valid_model_selection_criteria = [
            "val_loss",
            "wga",
            "iter2_wga",
            "iter2_auroc",
            "dna",
            "auto",
        ]
        if self.model_selection_criterion not in valid_model_selection_criteria:
            raise ValueError(
                f"model_selection_criterion must be one of {valid_model_selection_criteria}, got '{self.model_selection_criterion}'"
            )

        if self.depth_selection_confidence <= 0:
            raise ValueError(
                f"depth_selection_confidence must be positive, got {self.depth_selection_confidence}"
            )

        valid_hierarchical_modes = [
            "none",
            "correct_label_routing",
        ]
        if self.hierarchical_architecture not in valid_hierarchical_modes:
            raise ValueError(
                f"hierarchical_architecture must be one of {valid_hierarchical_modes}, got '{self.hierarchical_architecture}'"
            )

        valid_head_types = ["mlp", "linear"]
        if self.head_type not in valid_head_types:
            raise ValueError(
                f"head_type must be one of {valid_head_types}, got '{self.head_type}'"
            )
        if self.iter1_head_type is not None and self.iter1_head_type not in valid_head_types:
            raise ValueError(
                f"iter1_head_type must be one of {valid_head_types}, got '{self.iter1_head_type}'"
            )

        if self.cache_images_in_memory and self.num_workers > 0:
            print(
                f"Warning: num_workers={self.num_workers} with cache_images_in_memory=True "
                f"can cause OOM. Setting num_workers=0."
            )
            self.num_workers = 0

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

        if "optimizer" in config_dict and isinstance(config_dict["optimizer"], dict):
            config_dict["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
        if "scheduler" in config_dict and isinstance(config_dict["scheduler"], dict):
            config_dict["scheduler"] = SchedulerConfig(**config_dict["scheduler"])

        if config_dict.get("dfr_style", False):
            dfr_presets = {
                "head_ensemble_size": 20,
                "head_type": "linear",
            }
            for key, val in dfr_presets.items():
                if key not in config_dict:
                    config_dict[key] = val

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
