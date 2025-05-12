"""
Default parameters and constants for soundscape plotting functions.

This module provides common default values used for various plot types,
including density plots, scatter plots, and styling parameters. It contains
predefined color schemes, font settings, and layout configurations to ensure
consistent visualization across the package.
"""

import copy
from typing import Any

# Basic defaults
DEFAULT_XCOL = "ISOPleasant"
DEFAULT_YCOL = "ISOEventful"
DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)

DEFAULT_FIGSIZE = (5, 5)
DEFAULT_POINT_SIZE = 20
DEFAULT_BW_ADJUST = 1.2

DEFAULT_COLOR: str = "#0173B2"  # First color from colorblind palette

RECOMMENDED_MIN_SAMPLES: int = 30

# NOTE: The following are kept in only for the plot_functions currently.
# Should be replaced with ParamModels instead.

DEFAULT_XY_LABEL_FONTDICT: dict[str, Any] = {
    "family": "sans-serif",
    "fontstyle": "normal",
    "fontsize": "large",
    "fontweight": "medium",
    "parse_math": True,
    "c": "black",
    "alpha": 1,
}

# Default seaborn parameters
DEFAULT_SEABORN_PARAMS: dict[str, Any] = {
    "x": DEFAULT_XCOL,
    "y": DEFAULT_YCOL,
    "alpha": 0.8,
    "palette": "colorblind",
    "color": DEFAULT_COLOR,
    "zorder": 3,
}

# Default scatter parameters
DEFAULT_SCATTER_PARAMS: dict[str, Any] = {
    **DEFAULT_SEABORN_PARAMS,
    "s": DEFAULT_POINT_SIZE,
}

# Default density parameters
DEFAULT_DENSITY_PARAMS: dict[str, Any] = {
    **DEFAULT_SEABORN_PARAMS,
    "fill": True,
    "common_norm": False,
    "common_grid": False,
    "bw_adjust": DEFAULT_BW_ADJUST,
    "levels": 10,
    "clip": (DEFAULT_XLIM, DEFAULT_YLIM),
}

# Default simple density parameters
DEFAULT_SIMPLE_DENSITY_PARAMS: dict[str, Any] = copy.deepcopy(DEFAULT_DENSITY_PARAMS)
DEFAULT_SIMPLE_DENSITY_PARAMS.update({"thresh": 0.5, "levels": 2, "alpha": 0.5})

# Default style parameters
DEFAULT_STYLE_PARAMS: dict[str, Any] = {
    "xlim": DEFAULT_XLIM,
    "ylim": DEFAULT_YLIM,
    "xlabel": r"$P_{ISO}$",
    "ylabel": r"$E_{ISO}$",
    "diag_lines_zorder": 1,
    "diag_labels_zorder": 4,
    "prim_lines_zorder": 2,
    "data_zorder": 3,
    "title_fontsize": 16,
    "legend_loc": "best",
    "linewidth": 1.5,
    "primary_lines": True,
    "diagonal_lines": False,
    "prim_ax_fontdict": DEFAULT_XY_LABEL_FONTDICT,
}


# Default SPI text kwargs
DEFAULT_SPI_TEXT_KWARGS: dict[str, Any] = {
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
