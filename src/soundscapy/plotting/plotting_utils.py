"""
Core utilities and constants for plotting functionality.
"""

from enum import Enum, auto
from typing import Dict, Union

DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)
DEFAULT_FIGSIZE = (7, 7)

ExtraParams = Dict[str, Union[str, int, float, bool]]


class Backend(Enum):
    """Available plotting backends."""

    SEABORN = auto()
    PLOTLY = auto()


class PlotType(Enum):
    """Types of plots that can be created."""

    SCATTER = auto()
    DENSITY = auto()
    SIMPLE_DENSITY = auto()
    SCATTER_DENSITY = auto()
    JOINT = auto()


class LayerType(Enum):
    """Types of plot layers that can be added to a plot."""

    SCATTER = auto()
    DENSITY = auto()
    SIMPLE_DENSITY = auto()

    @property
    def default_zorder(self) -> int:
        """Get the default z-order for this layer type."""
        if self == LayerType.SCATTER:
            return 1  # Scatter points on top
        elif self == LayerType.DENSITY:
            return 2  # Density in middle
        elif self == LayerType.SIMPLE_DENSITY:
            return 3  # Simple density on bottom
        return 0
