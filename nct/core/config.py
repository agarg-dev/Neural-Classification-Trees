"""Configuration classes for NCT."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import torch


@dataclass
class NCTConfig:
    """Configuration class for NCT classifier."""

    # Training parameters
    num_iterations: int = 3
    epochs_per_iteration: List[int] = field(default_factory=lambda: [50, 50, 50])
    batch_size: int = 32

    # Learning rates
    initial_base_lr: float = 1e-5
    initial_head_lr: float = 1e-5
    lr_decay_factor: float = 2.0

    # Optimizer parameters
    weight_decay: float = 1e-4
    optimizer_type: str = "adamw"

    # Model parameters
    backbone: str = "resnet50"
    pretrained: bool = True

    # Training strategy
    phase1_epochs_ratio: float = 0.5  # Fraction of epochs for frozen base training (default 50:50 split)
    patience: int = 10  
    early_stopping: bool = True

    # Data parameters
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True

    # Reproducibility
    seed: Optional[int] = 42

    # Device
    device: Optional[Union[str, torch.device]] = 'auto'

    # Logging
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
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }