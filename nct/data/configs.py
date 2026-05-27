"""Dataset configurations for spurious correlation benchmarks."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ISIC_MEAN = (0.71826, 0.56291, 0.52548)
ISIC_STD = (0.16318, 0.14502, 0.17271)

UMNIST_MEAN = (0.1307,)
UMNIST_STD = (0.3081,)

CMNIST_MEAN = (0.5, 0.5, 0.5)
CMNIST_STD = (0.5, 0.5, 0.5)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    data_subdir: str
    class_names: Tuple[str, ...]
    target_resolution: Tuple[int, int] = (224, 224)
    num_classes: int = 2

    train_crop_size: int = 224
    train_scale: Tuple[float, float] = (0.7, 1.0)
    train_ratio: Tuple[float, float] = (0.75, 1.333)
    train_padding: Optional[int] = None
    train_resize: Optional[Tuple[int, int]] = None
    use_vertical_flip: bool = False
    use_horizontal_flip: bool = True

    eval_resize: Optional[int] = None
    eval_centercrop_first: Optional[int] = None
    eval_centercrop: int = 224

    group_names: Optional[Dict[int, str]] = None

    normalize_mean: Tuple[float, ...] = IMAGENET_MEAN
    normalize_std: Tuple[float, ...] = IMAGENET_STD

    num_channels: int = 3


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
    train_ratio=(0.75, 4.0 / 3.0),
    eval_resize=256,
    eval_centercrop=224,
    group_names={
        0: "Not Blond Female (Difficult)",
        1: "Not Blond Male (Easy)",
        2: "Blond Female (Easy)",
        3: "Blond Male (Difficult)",
    },
)


ISIC_CONFIG = DatasetConfig(
    name="isic",
    data_subdir="isic",
    class_names=("Benign", "Malignant"),
    train_ratio=(0.75, 1.333),
    use_vertical_flip=True,
    eval_resize=256,
    eval_centercrop=224,
    group_names={
        0: "Benign No Patch (Difficult)",
        1: "Benign With Patch (Easy)",
        2: "Malignant No Patch (Easy)",
        3: "Malignant With Patch (N/A)",
    },
    normalize_mean=ISIC_MEAN,
    normalize_std=ISIC_STD,
)


UMNIST_CONFIG = DatasetConfig(
    name="umnist",
    data_subdir="umnist",
    class_names=("Digit 0-4", "Digit 5-9"),
    target_resolution=(32, 32),
    train_crop_size=28,
    train_padding=4,
    eval_resize=None,
    use_horizontal_flip=False,
    group_names={
        0: "Digit 0-4 Non-8 (Easy)",
        1: "Digit 0-4 Digit-8 (N/A)",
        2: "Digit 5-9 Non-8 (Easy)",
        3: "Digit 5-9 Digit-8 (Difficult)",
    },
    normalize_mean=UMNIST_MEAN,
    normalize_std=UMNIST_STD,
    num_channels=1,
)


CMNIST_CONFIG = DatasetConfig(
    name="cmnist",
    data_subdir="cmnist",
    class_names=("Digit 0-1", "Digit 2-3", "Digit 4-5", "Digit 6-7", "Digit 8-9"),
    num_classes=5,
    target_resolution=(32, 32),
    train_crop_size=32,
    train_resize=(40, 40),
    train_padding=0,
    use_horizontal_flip=False,
    eval_resize=40,
    eval_centercrop=32,
    group_names={
        0:  "Digit 0-1 | Color 0 (Easy)", 1:  "Digit 0-1 | Color 1", 2:  "Digit 0-1 | Color 2", 3:  "Digit 0-1 | Color 3", 4:  "Digit 0-1 | Color 4",
        5:  "Digit 2-3 | Color 0", 6:  "Digit 2-3 | Color 1 (Easy)", 7:  "Digit 2-3 | Color 2", 8:  "Digit 2-3 | Color 3", 9:  "Digit 2-3 | Color 4",
        10: "Digit 4-5 | Color 0", 11: "Digit 4-5 | Color 1", 12: "Digit 4-5 | Color 2 (Easy)", 13: "Digit 4-5 | Color 3", 14: "Digit 4-5 | Color 4",
        15: "Digit 6-7 | Color 0", 16: "Digit 6-7 | Color 1", 17: "Digit 6-7 | Color 2", 18: "Digit 6-7 | Color 3 (Easy)", 19: "Digit 6-7 | Color 4",
        20: "Digit 8-9 | Color 0", 21: "Digit 8-9 | Color 1", 22: "Digit 8-9 | Color 2", 23: "Digit 8-9 | Color 3", 24: "Digit 8-9 | Color 4 (Easy)",
    },
    normalize_mean=CMNIST_MEAN,
    normalize_std=CMNIST_STD,
    num_channels=3,
)


DATASET_REGISTRY = {
    "waterbird": WATERBIRD_CONFIG,
    "celeba": CELEBA_CONFIG,
    "isic": ISIC_CONFIG,
    "umnist": UMNIST_CONFIG,
    "cmnist": CMNIST_CONFIG,
}
