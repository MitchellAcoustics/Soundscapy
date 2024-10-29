"""
Utility functions and constants for the soundscapy plotting module.
"""

from enum import Enum
from typing import Any, TypedDict


class PlotType(Enum):
    """Enum for supported plot types."""

    SCATTER = "scatter"
    DENSITY = "density"
    SIMPLE_DENSITY = "simple_density"
    JOINT = "joint"


class Backend(Enum):
    """Enum for supported plotting backends."""

    SEABORN = "seaborn"
    PLOTLY = "plotly"


class ExtraParams(TypedDict, total=False):
    """TypedDict for extra parameters passed to plotting functions."""

    color: Any
    marker: str
    linewidth: float
    # Add more potential parameters here


DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)
DEFAULT_FIGSIZE = (5, 5)
DEFAULT_COLORBLIND_PALETTE = "colorblind"
