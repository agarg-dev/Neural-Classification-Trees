"""Dataset configurations for spurious correlation benchmarks."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Default normalization (ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ISIC-specific normalization (from George paper)
ISIC_MEAN = (0.71826, 0.56291, 0.52548)
ISIC_STD = (0.16318, 0.14502, 0.17271)

# UMNIST normalization (standard MNIST - single channel grayscale)
UMNIST_MEAN = (0.1307,)
UMNIST_STD = (0.3081,)

# CMNIST normalization (colored MNIST - RGB)
# Using standard values since colors are synthetic
CMNIST_MEAN = (0.1307, 0.1307, 0.1307)
CMNIST_STD = (0.3081, 0.3081, 0.3081)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    data_subdir: str
    class_names: Tuple[str, ...]  # Tuple of class names (supports 2+ classes)
    target_resolution: Tuple[int, int] = (224, 224)
    num_classes: int = 2  # Number of classes (default 2 for binary classification)

    # Training augmentation parameters
    train_crop_size: int = 224
    train_scale: Tuple[float, float] = (0.7, 1.0)
    train_ratio: Tuple[float, float] = (0.75, 1.333)
    use_vertical_flip: bool = False

    # Evaluation preprocessing parameters
    eval_resize: Optional[int] = None
    eval_centercrop_first: Optional[int] = None
    eval_centercrop: int = 224

    # Group labels for evaluation
    group_names: Optional[Dict[int, str]] = None

    # Normalization parameters
    normalize_mean: Tuple[float, ...] = IMAGENET_MEAN
    normalize_std: Tuple[float, ...] = IMAGENET_STD

    # Number of input channels (3 for RGB, 1 for grayscale)
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
        3: "Malignant With Patch (N/A)",  # No malignant samples have patches
    },
    normalize_mean=ISIC_MEAN,
    normalize_std=ISIC_STD,
)


UMNIST_CONFIG = DatasetConfig(
    name="umnist",
    data_subdir="umnist",
    class_names=("Digit 0-4", "Digit 5-9"),
    target_resolution=(28, 28),
    train_crop_size=28,
    train_scale=(0.9, 1.0),
    train_ratio=(0.9, 1.1),
    eval_resize=None,
    eval_centercrop=28,
    group_names={
        0: "Digit 0-4 Non-8 (Easy)",
        1: "Digit 0-4 Digit-8 (N/A)",  # Digit 8 is in class 1
        2: "Digit 5-9 Non-8 (Easy)",
        3: "Digit 5-9 Digit-8 (Difficult)",  # Undersampled in training
    },
    normalize_mean=UMNIST_MEAN,
    normalize_std=UMNIST_STD,
    num_channels=1,  # Grayscale
)


# CMNIST: Colored MNIST with spurious color correlation
# Based on Correct-N-Contrast paper (https://arxiv.org/abs/2203.01517)
# 5 classes: digit pairs (0,1)->0, (2,3)->1, (4,5)->2, (6,7)->3, (8,9)->4
# 5 colors as spurious attribute, 99.5% correlation in training
CMNIST_CONFIG = DatasetConfig(
    name="cmnist",
    data_subdir="cmnist",
    class_names=("Digit 0-1", "Digit 2-3", "Digit 4-5", "Digit 6-7", "Digit 8-9"),
    num_classes=5,  # 5-class classification
    target_resolution=(28, 28),
    train_crop_size=28,
    train_scale=(0.9, 1.0),
    train_ratio=(0.9, 1.1),
    eval_resize=None,
    eval_centercrop=28,
    group_names={
        # For 5 classes with 5 colors, groups = class * num_colors + color_id
        # Simplified: just track class (spurious color is random in test)
        0: "Class 0 (Digits 0-1)",
        1: "Class 1 (Digits 2-3)",
        2: "Class 2 (Digits 4-5)",
        3: "Class 3 (Digits 6-7)",
        4: "Class 4 (Digits 8-9)",
    },
    normalize_mean=CMNIST_MEAN,
    normalize_std=CMNIST_STD,
    num_channels=3,  # RGB (colored images)
)


DATASET_REGISTRY = {
    "waterbird": WATERBIRD_CONFIG,
    "celeba": CELEBA_CONFIG,
    "isic": ISIC_CONFIG,
    "umnist": UMNIST_CONFIG,
    "cmnist": CMNIST_CONFIG,
}
