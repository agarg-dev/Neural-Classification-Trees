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
    """
    Denormalize image tensor to uint8 numpy array.

    Args:
        tensor: Image tensor (C, H, W) - supports both RGB (3 channels) and grayscale (1 channel)
        mean: Normalization mean (auto-detected from channels if None)
        std: Normalization std (auto-detected from channels if None)

    Returns:
        Image as uint8 array (H, W, C) in [0, 255] range
        For grayscale, returns (H, W, 3) by repeating the channel for visualization
    """
    num_channels = tensor.shape[0]

    # Auto-detect normalization stats if not provided
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
    """
    Convert normalized tensor to RGB float array.

    Args:
        tensor: Image tensor (C, H, W) - supports both RGB (3 channels) and grayscale (1 channel)
        mean: Normalization mean (auto-detected from channels if None)
        std: Normalization std (auto-detected from channels if None)

    Returns:
        RGB image as float array (H, W, C) in [0, 1] range
        For grayscale, returns (H, W, 3) by repeating the channel for visualization
    """
    num_channels = tensor.shape[0]

    # Auto-detect normalization stats if not provided
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
