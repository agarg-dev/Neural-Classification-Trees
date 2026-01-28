"""
NCT: Neural Classification Tree Library

A hierarchical classification library for image datasets using iterative training.
"""

from .core.classifier import NCTClassifier
from .core.config import NCTConfig, OptimizerConfig, SchedulerConfig

__version__ = "0.1.0"

__all__ = [
    "NCTClassifier",
    "NCTConfig",
    "OptimizerConfig",
    "SchedulerConfig",
]