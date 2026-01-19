"""Visualization utilities for NCT."""

# from .attribution import AttributionVisualizer
# from .attribution_xplique import XpliqueAttribution
from .attribution_captum import CaptumAttribution

__all__ = [
    # 'AttributionVisualizer',  # Original PyTorch GradCAM implementation (legacy)
    # 'XpliqueAttribution',     # Xplique-based multi-method implementation
    "CaptumAttribution",  # Captum-based implementation (recommended)
]
