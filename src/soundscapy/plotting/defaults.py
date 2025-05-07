import copy
from typing import Any

import seaborn as sns

from soundscapy.plotting.plotting_types import (
    DensityParamTypes,
    ScatterParamTypes,
    SeabornPaletteType,
    SeabornParamTypes,
    StyleParamsTypes,
    SubplotsParamsTypes,
)

DEFAULT_TITLE = "Soundscape Density Plot"
DEFAULT_XCOL = "ISOPleasant"
DEFAULT_YCOL = "ISOEventful"

DEFAULT_FIGSIZE = (5, 5)
DEFAULT_POINT_SIZE = 20
DEFAULT_BW_ADJUST = 1.2

DEFAULT_COLOR: str = "#0173B2"  # First color from colorblind palette

COLORBLIND_CMAP: list[str] = sns.color_palette("colorblind", as_cmap=True)

RECOMMENDED_MIN_SAMPLES: int = 30

DEFAULT_XY_LABEL_FONTDICT: dict[str, Any] = {
    "family": "sans-serif",
    "fontstyle": "normal",
    "fontsize": "large",
    "fontweight": "medium",
    "parse_math": True,
    "c": "black",
    "alpha": 1,
}

DEFAULT_SEABORN_PARAMS: SeabornParamTypes = SeabornParamTypes(
    x=DEFAULT_XCOL,
    y=DEFAULT_YCOL,
    alpha=0.8,
    palette="colorblind",
    color=DEFAULT_COLOR,
    zorder=3,
)

DEFAULT_SCATTER_PARAMS: ScatterParamTypes = ScatterParamTypes(
    **DEFAULT_SEABORN_PARAMS,
    s=DEFAULT_POINT_SIZE,
)

DEFAULT_DENSITY_PARAMS: DensityParamTypes = DensityParamTypes(
    **DEFAULT_SEABORN_PARAMS,
    fill=True,
    common_norm=False,
    common_grid=False,
    bw_adjust=DEFAULT_BW_ADJUST,
    levels=10,
)

DEFAULT_SIMPLE_DENSITY_PARAMS: DensityParamTypes = copy.deepcopy(DEFAULT_DENSITY_PARAMS)
DEFAULT_SIMPLE_DENSITY_PARAMS.update({"thresh": 0.5, "levels": 2, "alpha": 0.5})

DEFAULT_STYLE_PARAMS: StyleParamsTypes = StyleParamsTypes(
    xlim=(-1, 1),
    ylim=(-1, 1),
    xlabel=r"$P_{ISO}$",
    ylabel=r"$E_{ISO}$",
    diag_lines_zorder=1,
    diag_labels_zorder=4,
    prim_lines_zorder=2,
    data_zorder=3,
    title_fontsize=14,
    legend_loc="best",
    linewidth=1.5,
    primary_lines=True,
    diagonal_lines=False,
    prim_ax_fontdict=DEFAULT_XY_LABEL_FONTDICT,
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
