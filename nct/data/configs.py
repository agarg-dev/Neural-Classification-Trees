"""Dataset configurations for spurious correlation benchmarks."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    data_subdir: str
    class_names: Tuple[str, str]
    target_resolution: Tuple[int, int] = (224, 224)

    # Training augmentation parameters
    train_crop_size: int = 224
    train_scale: Tuple[float, float] = (0.7, 1.0)
    train_ratio: Tuple[float, float] = (0.75, 1.333)

    # Evaluation preprocessing parameters
    eval_resize: Optional[int] = None
    eval_centercrop_first: Optional[int] = None
    eval_centercrop: int = 224

    # Group labels for evaluation
    group_names: Optional[Dict[int, str]] = None


WATERBIRD_CONFIG = DatasetConfig(
    name="waterbird",
    data_subdir="waterbird",
    class_names=("Landbird", "Waterbird"),
    train_ratio=(0.75, 4.0 / 3.0),
    eval_resize=256,
    eval_centercrop=224,
    group_names={
        0: "Land Bird on Land (Easy)",
        1: "Land Bird on Water (Difficult)",
        2: "Water Bird on Land (Difficult)",
        3: "Water Bird on Water (Easy)",
    },
)


CELEBA_CONFIG = DatasetConfig(
    name="celeba",
    data_subdir="celeba",
    class_names=("Not Blond", "Blond"),
    train_ratio=(1.0, 4.0 / 3.0),
    eval_centercrop_first=178,
    eval_centercrop=224,
    group_names={
        0: "Not Blond Female (Difficult)",
        1: "Not Blond Male (Easy)",
        2: "Blond Female (Easy)",
        3: "Blond Male (Difficult)",
    },
)


DATASET_REGISTRY = {
    "waterbird": WATERBIRD_CONFIG,
    "celeba": CELEBA_CONFIG,
}
