"""Device management utilities."""

import torch
from typing import Optional, Union


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Args:
        device: Specific device to use. If None or 'auto', will auto-detect
                the best available device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to use for computations
    """
    if device is None or device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        return torch.device(device)

    return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get information about the current device.

    Args:
        device: Device to get info for. If None, uses the default device.

    Returns:
        dict: Device information including name, memory, etc.
    """
    if device is None:
        device = get_device()

    info = {
        'device': str(device),
        'type': device.type,
    }

    if device.type == 'cuda':
        info.update({
            'name': torch.cuda.get_device_name(device),
            'memory_total': torch.cuda.get_device_properties(device).total_memory,
            'memory_allocated': torch.cuda.memory_allocated(device),
            'memory_cached': torch.cuda.memory_reserved(device),
        })

    return info