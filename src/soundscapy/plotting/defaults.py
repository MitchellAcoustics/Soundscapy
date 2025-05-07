from typing import Any

import seaborn as sns

from soundscapy.plotting.plotting_types import (
    DensityParamTypes,
    ScatterParamTypes,
    SeabornPaletteType,
    StyleParamsTypes,
    SubplotsParamsTypes,
)

DEFAULT_TITLE = "Soundscape Density Plot"
DEFAULT_TITLE_FONTSIZE = 14
DEFAULT_XCOL = "ISOPleasant"
DEFAULT_YCOL = "ISOEventful"
DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)
DEFAULT_XLABEL = "$P_{ISO}$"
DEFAULT_YLABEL = "$E_{ISO}$"
DEFAULT_FIGSIZE = (5, 5)
DEFAULT_LEGEND_LOC = "best"
DEFAULT_POINT_SIZE = 20
DEFAULT_ALPHA = 0.8

DATA_ZORDER = 3
DIAG_LINES_ZORDER = 1
DIAG_LABELS_ZORDER = 4
PRIM_LINES_ZORDER = 2
DEFAULT_BW_ADJUST = 1.2

DEFAULT_PALETTE: SeabornPaletteType = "colorblind"
DEFAULT_COLOR: str = "#0173B2"  # First color from colorblind palette

COLORBLIND_CMAP: list[str] = sns.color_palette("colorblind", as_cmap=True)

RECOMMENDED_MIN_SAMPLES: int = 30

DEFAULT_SCATTER_PARAMS: ScatterParamTypes = ScatterParamTypes(
    x=DEFAULT_XCOL,
    y=DEFAULT_YCOL,
    alpha=DEFAULT_ALPHA,
    palette=DEFAULT_PALETTE,
    color=DEFAULT_COLOR,
    legend="auto",
    zorder=DATA_ZORDER,
    s=DEFAULT_POINT_SIZE,
)

DEFAULT_DENSITY_PARAMS: DensityParamTypes = DensityParamTypes(
    x=DEFAULT_XCOL,
    y=DEFAULT_YCOL,
    alpha=DEFAULT_ALPHA,
    fill=True,
    common_norm=False,
    common_grid=False,
    palette=DEFAULT_PALETTE,
    color=DEFAULT_COLOR,
    bw_adjust=DEFAULT_BW_ADJUST,
    zorder=DATA_ZORDER,
)

DEFAULT_STYLE_PARAMS: StyleParamsTypes = StyleParamsTypes(
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
    xlabel=None,
    ylabel=None,
    diag_lines_zorder=DIAG_LINES_ZORDER,
    diag_labels_zorder=DIAG_LABELS_ZORDER,
    prim_lines_zorder=PRIM_LINES_ZORDER,
    data_zorder=DATA_ZORDER,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
    legend_loc=DEFAULT_LEGEND_LOC,
    linewidth=1.5,
    primary_lines=True,
    diagonal_lines=False,
    prim_ax_fontdict={
        "fontstyle": "italic",
        "fontsize": "medium",
        "fontweight": "bold",
        "c": "grey",
        "alpha": 1,
    },
)

DEFAULT_SUBPLOTS_PARAMS: SubplotsParamsTypes = SubplotsParamsTypes(
    sharex=True,
    sharey=True,
)

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

DEFAULT_XY_LABEL_FONTDICT: dict[str, Any] = {
    "family": "sans-serif",
    "fontstyle": "normal",
    "fontsize": "large",
    "fontweight": "medium",
    "parse_math": True,
    "c": "black",
    "alpha": 1,
}
