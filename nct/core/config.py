"""Configuration dataclasses for NCT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import yaml

VALID_OPTIMIZERS = ["adamw", "adam", "sgd", "rmsprop"]
VALID_SCHEDULERS = ["plateau", "step", "cosine", "cosine_warmup", "none"]


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    weight_decay: float = 1e-4

    # SGD
    momentum: float = 0.9
    nesterov: bool = True

    # Adam / AdamW
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self):
        if self.type not in VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer.type must be one of {VALID_OPTIMIZERS}, got '{self.type}'"
            )


@dataclass
class SchedulerConfig:
    """LR scheduler config. Fields set to -1 are auto-calculated from training duration."""

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
        if self.type not in VALID_SCHEDULERS:
            raise ValueError(
                f"scheduler.type must be one of {VALID_SCHEDULERS}, got '{self.type}'"
            )


@dataclass
class NCTConfig:
    """All hyperparameters for an NCT run. Load from YAML with `NCTConfig.from_yaml(path)`."""

    # Training
    num_iterations: int = 3
    epochs_per_iteration: Optional[List[int]] = None
    batch_size: int = 32

    # Model
    backbone: str = "resnet50"
    pretrained: bool = True
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    head_type: str = "mlp"  # "mlp" | "linear"
    iter1_head_type: Optional[str] = None  # if set, overrides head_type for iter 1 only
    split_node_head_type: Union[bool, str] = False  # True: easy=linear, hard=mlp. "reversed": swap.
    head_bn_affine: bool = False
    num_channels: int = 3
    num_classes: int = 2

    # Optimizer / LR
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initial_base_lr: float = 1e-5
    initial_head_lr: float = 1e-5
    lr_decay_factor: float = 2.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss
    use_class_weights: bool = True
    skip_class_weights_iter1: bool = False  # iter1 unweighted -> failures are more systematic
    max_class_weight: Optional[float] = None
    regularization_type: str = "l2"  # "l1" | "l2" | "elastic_net" | "none"
    lambda_l1: float = 1e-4
    aux_loss_weight: float = 0.3  # parent-layer auxiliary loss; 0 disables

    # Training strategy
    phase1_epochs_ratio: Optional[List[float]] = None
    phase1_patience: int = 10
    phase2_patience: int = 10
    early_stopping: bool = True
    # "auto" -> val_loss for iter1, wga for iter2+. Others: "val_loss", "wga", "iter2_wga", "iter2_auroc", "dna".
    model_selection_criterion: str = "auto"
    save_iteration_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    attribution_dir: str = "./attributions"

    routing_strategy: str = "correctness"  # only supported value

    depth_selection_enabled: bool = False
    depth_selection_confidence: float = 1.96

    # Iter2+ node sampling: "none" | "upsample" | "downsample" | "geomean". Iter1 is always unbalanced.
    sampling_strategy: str = "none"

    min_samples_per_node: int = 0  # difficult nodes below this merge into the easy sibling; 0 disables

    # "correct_label_routing": tree with parent-layer weight sharing. "none": parallel heads.
    hierarchical_architecture: str = "correct_label_routing"

    # Data
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True
    cache_images_in_memory: bool = False  # holds PIL images in RAM; OOMs on big datasets

    # System
    seed: Optional[int] = 42
    device: Optional[Union[str, torch.device]] = "auto"
    verbose: int = 1  # 0 silent, 1 progress, 2 detailed

    log_attributions: bool = True
    log_epoch_metrics: bool = True

    # WandB
    wandb_enabled: bool = True
    wandb_project: str = "nct-waterbirds"
    wandb_entity: Optional[str] = "ankurgarg-university-of-calgary-in-alberta"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.epochs_per_iteration is None:
            self.epochs_per_iteration = [50] * self.num_iterations
        if self.phase1_epochs_ratio is None:
            self.phase1_epochs_ratio = [0.5] * self.num_iterations

        if len(self.epochs_per_iteration) != self.num_iterations:
            raise ValueError(
                f"epochs_per_iteration has length {len(self.epochs_per_iteration)}, "
                f"expected {self.num_iterations}"
            )
        if len(self.phase1_epochs_ratio) != self.num_iterations:
            raise ValueError(
                f"phase1_epochs_ratio has length {len(self.phase1_epochs_ratio)}, "
                f"expected {self.num_iterations}"
            )

        valid_reg = ["l1", "l2", "elastic_net", "none"]
        if self.regularization_type not in valid_reg:
            raise ValueError(f"regularization_type must be one of {valid_reg}, got '{self.regularization_type}'")

        if self.routing_strategy != "correctness":
            raise ValueError(f"routing_strategy must be 'correctness', got '{self.routing_strategy}'")

        valid_sampling = ["none", "upsample", "downsample", "geomean"]
        if self.sampling_strategy not in valid_sampling:
            raise ValueError(f"sampling_strategy must be one of {valid_sampling}, got '{self.sampling_strategy}'")

        valid_msc = ["val_loss", "wga", "iter2_wga", "iter2_auroc", "dna", "auto"]
        if self.model_selection_criterion not in valid_msc:
            raise ValueError(f"model_selection_criterion must be one of {valid_msc}, got '{self.model_selection_criterion}'")

        if self.depth_selection_confidence <= 0:
            raise ValueError(f"depth_selection_confidence must be positive, got {self.depth_selection_confidence}")

        valid_hier = ["none", "correct_label_routing"]
        if self.hierarchical_architecture not in valid_hier:
            raise ValueError(f"hierarchical_architecture must be one of {valid_hier}, got '{self.hierarchical_architecture}'")

        valid_head = ["mlp", "linear"]
        if self.head_type not in valid_head:
            raise ValueError(f"head_type must be one of {valid_head}, got '{self.head_type}'")
        if self.iter1_head_type is not None and self.iter1_head_type not in valid_head:
            raise ValueError(f"iter1_head_type must be one of {valid_head}, got '{self.iter1_head_type}'")

        if self.cache_images_in_memory and self.num_workers > 0:
            print(
                f"Warning: num_workers={self.num_workers} with cache_images_in_memory=True can OOM; "
                f"forcing num_workers=0."
            )
            self.num_workers = 0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NCTConfig":
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "NCTConfig":
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        if isinstance(config_dict.get("optimizer"), dict):
            config_dict["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
        if isinstance(config_dict.get("scheduler"), dict):
            config_dict["scheduler"] = SchedulerConfig(**config_dict["scheduler"])

        config_dict.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**config_dict)

    def to_dict(self) -> dict:
        result = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if hasattr(value, "__dataclass_fields__"):
                result[name] = {f.name: getattr(value, f.name) for f in value.__dataclass_fields__.values()}
            else:
                result[name] = value
        return result
