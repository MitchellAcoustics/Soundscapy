"""Utility functions and constants for the soundscapy plotting module."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

SeabornPaletteType: TypeAlias = str | list | dict | Colormap

MplLegendLocType: TypeAlias = (
    Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ]
    | tuple[float, float]
)


class SubplotsParamsTypes(TypedDict, total=False):
    """TypedDict for pyplot.subplots() parameters."""

    sharex: bool
    sharey: bool
    squeeze: bool
    width_ratios: Sequence[float] | None
    height_ratios: Sequence[float] | None
    subplot_kw: dict[str, Any] | None
    gridspec_kw: dict[str, Any] | None


TYPED_SUBPLOT_KWS: list[str] = [
    "sharex",
    "sharey",
    "squeeze",
    "width_ratios",
    "height_ratios",
    "subplot_kw",
    "gridspec_kw",
    "fig_kw",
]


class _SeabornParamTypes(TypedDict, total=False):
    """Base typing for seaborn plotting parameters."""

    data: pd.DataFrame | None
    x: str | np.ndarray | pd.Series | None
    y: str | np.ndarray | pd.Series | None
    hue: str | np.ndarray | pd.Series | None
    size: str | np.ndarray | pd.Series | None
    style: str | np.ndarray | pd.Series | None
    palette: SeabornPaletteType | None
    hue_order: Iterable[str] | None
    hue_norm: tuple | Normalize | None
    alpha: float
    legend: Literal["auto", "brief", "full", False]  # NOTE: Might move to Styler?
    # matplotlib kwargs
    color: ColorType | None
    label: str | None
    zorder: float


TYPED_SEABORN_KWS: list[str] = [
    "data",
    "x",
    "y",
    "hue",
    "size",
    "style",
    "palette",
    "hue_order",
    "hue_norm",
    "alpha",
    "legend",
    "color",
    "label",
    "zorder",
]


class ScatterParamTypes(_SeabornParamTypes, total=False):
    """TypedDict for scatter plot parameters."""

    sizes: list | dict | tuple | None
    size_order: list | None
    size_norm: tuple | Normalize | None
    markers: bool | list | dict
    style_order: list
    # ax: Axes | np.ndarray | None  # noqa: ERA001
    marker: str
    linewidth: float
    s: float | None
    # matplotlib kwargs


TYPED_SCATTER_KWS: list[str] = [
    "sizes",
    "size_order",
    "size_norm",
    "markers",
    "style_order",
    "marker",
    "linewidth",
    "s",
]


class DensityParamTypes(_SeabornParamTypes, total=False):
    """TypedDict for density plot parameters."""

    weights: str | np.ndarray | pd.Series | None
    fill: bool | None
    multiple: Literal["layer", "stack", "fill"]
    common_norm: bool
    common_grid: bool
    cumulative: bool
    bw_method: Literal["scott", "silverman"] | float | Callable | None
    bw_adjust: float
    warn_singular: bool
    log_scale: bool | tuple[bool, bool] | float | tuple[float, float] | None
    levels: int | Iterable[float]
    thresh: float
    gridsize: int
    cut: float
    clip: tuple[tuple[float, float], tuple[float, float]] | None
    cbar: bool
    cbar_ax: Axes | None
    cbar_kws: dict[str, Any] | None
    # matplotlib kwargs


TYPED_DENSITY_KWS: list[str] = [
    "weights",
    "fill",
    "multiple",
    "common_norm",
    "common_grid",
    "cumulative",
    "bw_method",
    "bw_adjust",
    "warn_singular",
    "log_scale",
    "levels",
    "thresh",
    "gridsize",
    "cut",
    "clip",
    "cbar",
    "cbar_ax",
    "cbar_kws",
]


class JointPlotParamTypes(TypedDict, total=False):
    """TypedDict for jointplot parameters."""

    data: pd.DataFrame | None
    x: str | np.ndarray | pd.Series | None
    y: str | np.ndarray | pd.Series | None
    height: float
    ratio: float
    space: float
    dropna: bool
    xlim: tuple[float, float] | None
    ylim: tuple[float, float] | None
    marginal_ticks: bool
    hue: str | np.ndarray | pd.Series | None
    palette: SeabornPaletteType | None
    hue_order: Iterable[str] | None
    hue_norm: tuple | Normalize | None


TYPED_JOINTPLOT_KWS: list[str] = [
    "data",
    "x",
    "y",
    "height",
    "ratio",
    "space",
    "dropna",
    "xlim",
    "ylim",
    "marginal_ticks",
    "hue",
    "palette",
    "hue_order",
    "hue_norm",
]


class StyleParamsTypes(TypedDict, total=False):
    """
    Configuration options for styling circumplex plots.

    Attributes:
        diag_lines_zorder (int): Z-order for diagonal lines.
        diag_labels_zorder (int): Z-order for diagonal labels.
        prim_lines_zorder (int): Z-order for primary lines.
        data_zorder (int): Z-order for plotted data.
        bw_adjust (float): Bandwidth adjustment for kernel density estimation.
        figsize (Tuple[int, int]): Figure size (width, height) in inches.
        simple_density (Dict[str, Any]): Configuration for simple density plots.

    """

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    diag_lines_zorder: int
    diag_labels_zorder: int
    prim_lines_zorder: int
    data_zorder: int
    show_labels: bool
    legend_loc: MplLegendLocType
    linewidth: float
    primary_lines: bool
    diagonal_lines: bool
    title_fontsize: int


TYPED_STYLE_KWS: list[str] = [
    "xlim",
    "ylim",
    "diag_lines_zorder",
    "diag_labels_zorder",
    "prim_lines_zorder",
    "data_zorder",
    "show_labels",
    "legend_loc",
    "linewidth",
    "primary_lines",
    "diagonal_lines",
    "title_fontsize",
]
