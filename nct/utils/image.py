"""Image preprocessing utilities for visualization."""

from typing import Optional, Tuple

import numpy as np
import torch

# Default normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """Tensor (C,H,W) → uint8 (H,W,3). Grayscale is repeated 3× for display; mean/std auto from channels."""
    num_channels = tensor.shape[0]

    if mean is None or std is None:
        if num_channels == 1:
            mean = MNIST_MEAN
            std = MNIST_STD
        else:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD

    mean = np.array(mean)
    std = np.array(std)

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert grayscale to RGB for visualization
    if num_channels == 1:
        img = np.repeat(img, 3, axis=2)

    img = (img * 255).astype(np.uint8)

    return img


def tensor_to_rgb(
    tensor: torch.Tensor,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """Tensor (C,H,W) → float (H,W,3) in [0,1]. Grayscale is repeated 3× for display."""
    num_channels = tensor.shape[0]

    if mean is None or std is None:
        if num_channels == 1:
            mean = MNIST_MEAN
            std = MNIST_STD
        else:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD

    mean = np.array(mean)
    std = np.array(std)

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert grayscale to RGB for visualization
    if num_channels == 1:
        img = np.repeat(img, 3, axis=2)

    return img
