"""
Constants for soundscape plotting functions.

This module provides common constants used for various plot types,
including default column names, limits, and other configuration values.
These constants are used by the parameter models to provide default values.
"""

# Basic defaults
DEFAULT_XCOL = "ISOPleasant"
DEFAULT_YCOL = "ISOEventful"
DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)

DEFAULT_FIGSIZE = (5, 5)
DEFAULT_POINT_SIZE = 20
DEFAULT_BW_ADJUST = 1.2

DEFAULT_COLOR = "#0173B2"  # First color from colorblind palette

RECOMMENDED_MIN_SAMPLES = 30

# Default font settings for axis labels
DEFAULT_FONTDICT = {
    "family": "sans-serif",
    "fontstyle": "normal",
    "fontsize": "large",
    "fontweight": "medium",
    "parse_math": True,
    "c": "black",
    "alpha": 1,
}

# Default SPI text settings
DEFAULT_SPI_TEXT_KWARGS = {
    "x": 0,
    "y": -0.85,
    "fontsize": 10,
    "bbox": {
        "facecolor": "white",
        "edgecolor": "black",
        "boxstyle": "round,pad=0.3",
    },
    "ha": "center",
    "va": "center",
}
