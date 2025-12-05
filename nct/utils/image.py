"""Image preprocessing utilities for visualization."""

import numpy as np
import torch


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor to RGB uint8 numpy array.
    
    Args:
        tensor: Image tensor (C, H, W) normalized with ImageNet stats
        
    Returns:
        RGB image as uint8 array (H, W, C) in [0, 255] range
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor to RGB float array.
    
    Args:
        tensor: Image tensor (C, H, W) normalized with ImageNet stats
        
    Returns:
        RGB image as float array (H, W, C) in [0, 1] range
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img
