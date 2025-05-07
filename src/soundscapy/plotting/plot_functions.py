"""Plotting functions for visualising circumplex data."""

# ruff: noqa: ANN003
from collections.abc import Iterable
import functools
import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from matplotlib.typing import ColorType

from soundscapy.plotting.backends import Backend
from soundscapy.plotting.defaults import (
    COLORBLIND_CMAP,
    # DATA_ZORDER,
    # DEFAULT_ALPHA,
    DEFAULT_BW_ADJUST,
    DEFAULT_COLOR,
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_FIGSIZE,
    # DEFAULT_LEGEND_LOC,
    # DEFAULT_PALETTE,
    DEFAULT_POINT_SIZE,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SEABORN_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_TITLE,
    # DEFAULT_TITLE_FONTSIZE,
    DEFAULT_XCOL,
    # DEFAULT_XLABEL,
    # DEFAULT_XLIM,
    DEFAULT_XY_LABEL_FONTDICT,
    DEFAULT_YCOL,
    # DEFAULT_YLABEL,
    # DEFAULT_YLIM,
    # DIAG_LABELS_ZORDER,
    # DIAG_LINES_ZORDER,
    # PRIM_LINES_ZORDER,
    RECOMMENDED_MIN_SAMPLES,
)
from soundscapy.plotting.plotting_types import MplLegendLocType, SeabornPaletteType
from soundscapy.sspylogging import get_logger

logger = get_logger()


def scatter(
    data: pd.DataFrame,
    *,
    x: str = DEFAULT_XCOL,
    y: str = DEFAULT_YCOL,
    title: str | None = "Soundscape Scatter Plot",
    ax: Axes | None = None,
    hue: str | np.ndarray | pd.Series | None = None,
    palette: SeabornPaletteType | None = "colorblind",
    color: ColorType | None = DEFAULT_COLOR,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    s: float = DEFAULT_POINT_SIZE,
    legend: Literal["auto", "brief", "full", False] = "auto",
    prim_labels: bool | None = None,  # Alias for primary_labels, deprecated
    **kwargs,
) -> Axes:
    """
    Plot ISOcoordinates as scatter points on a soundscape circumplex grid.

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure. Either a long-form collection of vectors that can be
        assigned to named variables or a wide-form dataset that will be internally
        reshaped.
    x : str, optional
        column name for x variable, by default "ISOPleasant"
    y : str, optional
        column name for y variable, by default "ISOEventful"
    title : str, optional
        Title to add to circumplex plot, by default "Soundscape Scatter Plot"
    diagonal_lines : bool, optional
        whether to include diagonal dimension labels (e.g. calm, etc.), by default False
    xlim, ylim : tuple, optional
        Limits of the circumplex plot, by default (-1, 1)
        It's recommended to set these such that the x and y axes have the same aspect
    figsize : tuple, optional
        Size of the figure to return if `ax` is None, by default (5, 5)
    legend_loc : str, optional
        relative location of legend, by default "lower left"
    hue : str, optional
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric,
        although color mapping will behave differently in latter case, by default None
    s : int, optional
        size of scatter points, by default 20
    palette : string, list, dict or matplotlib.colors.Colormap, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping.
        by default colorblind
    legend : {"auto", "brief", "full" or False}, optional
        How to draw the legend. If “brief”, numeric hue and size variables will be
        represented with a sample of evenly spaced values. If “full”, every group will
        get an entry in the legend. If “auto”, choose between brief or full
        representation based on number of levels.
        If False, no legend data is added and no legend is drawn. By default, "auto"
    ax : matplotlib.axes.Axes, optional
        Pre-existing matplotlib axes for the plot, by default None
        If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.

    Returns
    -------
    matplotlib.axes.Axes
    Axes object containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Removes the palette if no hue is specified
    palette = palette if hue is not None else None

    # Get style params / kwargs
    xlabel, ylabel, xlim, ylim, legend_loc, diagonal_lines, prim_ax_fontdict = (
        _pop_style_kwargs(kwargs)
    )

    p = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        s=s,
        palette=palette,
        color=color,
        legend=legend,
        ax=ax,
        zorder=DEFAULT_STYLE_PARAMS["data_zorder"],
        **kwargs,
    )

    xlabel, ylabel = _deal_w_default_labels(
        x=x, y=y, xlabel=xlabel, ylabel=ylabel, prim_labels=prim_labels
    )
    _set_style()
    _circumplex_grid(
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        diagonal_lines=diagonal_lines,
        prim_ax_fontdict=prim_ax_fontdict,
    )
    if title is not None:
        _set_circum_title(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if legend is not None and hue is not None:
        _move_legend(ax=ax, new_loc=legend_loc)
    return p


def density(
    data: pd.DataFrame,
    *,
    x: str = DEFAULT_XCOL,
    y: str = DEFAULT_YCOL,
    title: str | None = "Soundscape Density Plot",
    ax: Axes | None = None,
    hue: str | np.ndarray | pd.Series | None = None,
    incl_scatter: bool = True,
    density_type: str = "full",
    palette: SeabornPaletteType | None = "colorblind",
    color: ColorType | None = DEFAULT_COLOR,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    legend: Literal["auto", "brief", "full", False] = "auto",
    prim_labels: bool | None = None,  # Alias for primary_labels, deprecated
    scatter_kws: dict | None = None,
    incl_outline: bool = False,
    alpha: float = DEFAULT_SEABORN_PARAMS["alpha"],
    fill: bool = True,
    levels: int | Iterable[float] = 10,
    thresh: float = 0.05,
    bw_adjust: float = DEFAULT_BW_ADJUST,
    **kwargs,
) -> Axes:
    """
    Plot a density plot of ISOCoordinates.

    Creates a wrapper around `seaborn.kdeplot` and adds functionality and styling to
    customise it for circumplex plots.
    The density plot is a combination of a kernel density estimate and a scatter plot.

    Parameters
    ----------
    color
    data : pd.DataFrame, np.ndarray, mapping or sequence
        Input data structure. Either a long-form collection of vectors that can be assigned to
        named variables or a wide-form dataset that will be internally reshaped.
    x : vector or key in `data`, optional
        Column name for x variable, by default "ISOPleasant"
    y : vector or key in `data`, optional
        Column name for y variable, by default "ISOEventful"
    incl_scatter : bool, optional
        Whether to include a scatter plot of the data, by default True
    density_type : {"full", "simple"}, optional
        Type of density plot to draw, by default "full"
    title : str, optional
        Title to add to circumplex plot, by default "Soundscapy Density Plot"
    diagonal_lines : bool, optional
        Whether to include diagonal dimension labels (e.g. calm, etc.), by default False
    xlim, ylim : tuple, optional
        Limits of the circumplex plot, by default (-1, 1)
        It's recommended to set these such that the x and y axes have the same aspect
    scatter_kws : dict, optional
        Keyword arguments to pass to `seaborn.scatterplot`, by default dict(s=25, linewidth=0)
    incl_outline : bool, optional
    figsize : tuple, optional
        Size of the figure to return if `ax` is None, by default (5, 5)
    legend_loc : str, optional
        Relative location of legend, by default "lower left"
    alpha : float, optional
        Proportional opacity of the heatmap fill, by default 0.75
    legend : bool, optional
        If False, suppress the legend for semantic variables, by default True
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes object to use for the plot, by default None
    hue : vector or key in `data`, optional
        Semantic variable that is mapped to determine the color of plot elements, by default None
    palette : Union[str, list, dict, matplotlib.colors.Colormap], optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to
        seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping.
        by default colorblind
    fill : bool, optional
        If True, fill in the area under univariate density curves or between bivariate contours. If None, the default
        depends on `multiple`. by default True.
    levels : int or vector, optional
        Number of contour levels or values to draw contours at. A vector argument must have increasing values in [0, 1].
        Levels correspond to iso-proportionas of the density: e.g. 20% of the probability mass will lie below the
        contour drawn for 0.2. Only relevant with bivariate data.
        by default 10
    thresh : number in [0, 1], optional
        Lowest iso-proportional level at which to draw a contour line. Ignored when `levels` is a vector. Only relevant
        with bivariate plots.
        by default 0.05
    bw_adjust : number, optional
        Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
    **kwargs : dict, optional#
        Other keyword arguments are passed to one of the following matplotlib functions:
        - `matplotlib.axes.Axes.plot()` (univariate, `fill=False`),
        - `matplotlib.axes.fill_between()` (univariate, `fill=True`),
        - `matplotlib.axes.Axes.contour()` (bivariate, `fill=True`),
        - `matplotlib.axes.Axes.contourf()` (bivariate, `fill=True`).

    Returns
    -------
    matplotlib.axes.Axes
        Axes object containing the plot.

    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    scatter_kws = {"s": 25, "linewidth": 0} if scatter_kws is None else scatter_kws

    # Removes the palette if no hue is specified
    palette = palette if hue is not None else None

    # Get style params / kwargs
    xlabel, ylabel, xlim, ylim, legend_loc, diagonal_lines, prim_ax_fontdict = (
        _pop_style_kwargs(kwargs)
    )

    if density_type == "simple":
        thresh = DEFAULT_SIMPLE_DENSITY_PARAMS["thresh"]
        levels = DEFAULT_SIMPLE_DENSITY_PARAMS["levels"]
        alpha = DEFAULT_SIMPLE_DENSITY_PARAMS["alpha"]
        incl_outline = True

    if incl_scatter:
        d = sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            palette=palette,
            zorder=DEFAULT_SCATTER_PARAMS["zorder"],
            **scatter_kws,
        )

    if incl_outline:
        d = sns.kdeplot(
            data=data,
            x=x,
            y=y,
            alpha=1,
            ax=ax,
            hue=hue,
            palette=palette,
            levels=levels,
            thresh=thresh,
            bw_adjust=bw_adjust,
            fill=False,
            zorder=DEFAULT_DENSITY_PARAMS["zorder"],
            color=color,
            **kwargs,
        )

    d = sns.kdeplot(
        data=data,
        x=x,
        y=y,
        alpha=alpha,
        legend=legend,  # type: ignore[reportArgumentType d]
        ax=ax,
        hue=hue,
        palette=palette,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        fill=fill,
        zorder=DEFAULT_DENSITY_PARAMS["zorder"],
        color=color,
        **kwargs,
    )

    xlabel, ylabel = _deal_w_default_labels(
        x=x, y=y, xlabel=xlabel, ylabel=ylabel, prim_labels=prim_labels
    )
    _set_style()
    _circumplex_grid(
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        diagonal_lines=diagonal_lines,
        prim_ax_fontdict=prim_ax_fontdict,
    )
    if title is not None:
        _set_circum_title(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if legend is not None and hue is not None:
        _move_legend(ax=ax, new_loc=legend_loc)

    return d


# TODO: Consider changing to displot
def jointplot(
    data=None,
    x="ISOPleasant",
    y="ISOEventful",
    incl_scatter=True,
    density_type="full",
    title="Soundscape Joint Plot",
    diagonal_lines=False,
    xlim=(-1, 1),
    ylim=(-1, 1),
    scatter_kws=dict(s=25, linewidth=0),
    incl_outline=False,
    legend_loc="lower left",
    alpha=0.75,
    joint_kws={},
    marginal_kws={"fill": True, "common_norm": False},
    hue=None,
    color=None,
    palette="colorblind",
    fill=True,
    bw_adjust=None,
    thresh=0.1,
    levels=10,
    legend=False,
    marginal_kind="kde",
):
    """
    Create a jointplot with distribution or scatter in the center and distributions on the margins.

    This method works by calling sns.jointplot() and creating a circumplex grid in the joint position, then
    overlaying a density or circumplex_scatter plot. The options for both the joint and marginal plots can be
    passed through the sns.jointplot() separately to customise them separately. The marginal distribution plots
    can be either a density or histogram.

    Parameters
    ----------
    color
    data : pd.DataFrame, np.ndarray, mapping, or sequence
        Input data structure. Either a long-form collection of vectors that can be assigned to named variables or a
        wide-form dataset that will be internally reshaped.
    x : vector or key in `data`, optional
        column name for x variable, by default "ISOPleasant"
    y : vector or key in `data`, optional
        column name for y variable, by default "ISOEventful"
    incl_scatter : bool, optional
        Whether to include a scatter plot of the data, by default True
    density_type : str, optional
        Type of density plot to draw, by default "full"
    diagonal_lines : bool, optional
        whether to include diagonal dimension axis labels in the joint plot, by default False
    palette : str, optional
        [description], by default "colorblind"
    incl_scatter : bool, optional
        plot coordinate scatter underneath density plot, by default False
    fill : bool, optional
        whether to fill the density plot, by default True
    bw_adjust : [type], optional
        [description], by default default_bw_adjust
    alpha : float, optional
        [description], by default 0.95
    legend : bool, optional
        whether to include the hue labels legend, by default False
    legend_loc : str, optional
        relative location of the legend, by default "lower left"
    marginal_kind : str, optional
        density or histogram plot in the margins, by default "kde"
    hue : vector or key in data, optional
        Grouping variable that will produce points with different colors. Can be either categorical or numeric,
        although color mapping will behave differently in latter case, by default None
    joint_kws : dict, optional
        Arguments to pass to density or scatter joint plot, by default {}
    marginal_kws : dict, optional
        Arguments to pass to marginal distribution plots, by default {"fill": True}
    hue : vector or key in `data`, optional
        Semantic variable that is mapped to determine the color of plot elements.
    palette : string, list, dict, or `matplotlib.colors.Colormap`, optional
        Method for choosing the colors to use when mapping the `hue` semantic. String values are passed to
        `color_palette()`. List or dict values imply categorical mapping, while a colormap object implies numeric
        mapping.
        by default, `"colorblind"`
    fill : bool, optional
        If True, fill in the area under univariate density curves or between bivariate contours. If None, the default
        depends on `multiple`. by default True
    bw_adjust : number, optional
        Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
        See Notes. by default default_bw_adjust (1.2)
    thresh : number in [0, 1], optional
        Lowest iso-proportional level at which to draw a contour line. Ignored when `levels` is a vector. Only relevant
        with bivariate plots. by default 0.1
    levels : int or vector, optional
        Number of contour levels or values to draw contours at. A vector argument must have increasing values in [0, 1].
        Levels correspond to iso-proportionas of the density: e.g. 20% of the probability mass will lie below the
        contour drawn for 0.2. Only relevant with bivariate data.
        by default 10
    legend : bool, optional
        If False, suppress the legend for semantic variables, by default False
    legend_loc : str, optional
        Relative location of the legend, by default "lower left"
    marginal_kind : str, optional
        density or histogram plot in the margins, by default "kde"

    Returns
    -------
    plt.Axes

    """
    if bw_adjust is None:
        bw_adjust = default_bw_adjust

    if density_type == "simple":
        thresh = simple_density["thresh"]
        levels = simple_density["levels"]
        alpha = simple_density["alpha"]
        incl_outline = simple_density["incl_outline"]

    if hue is None:
        # Removes the palette if no hue is specified
        palette = None
        color = sns.color_palette("colorblind", 1)[0] if color is None else color

    g = sns.JointGrid()
    density(
        data,
        x=x,
        y=y,
        incl_scatter=incl_scatter,
        density_type=density_type,
        title=None,
        diagonal_lines=diagonal_lines,
        xlim=xlim,
        ylim=ylim,
        scatter_kws=scatter_kws,
        incl_outline=incl_outline,
        legend_loc=legend_loc,
        alpha=alpha,
        legend=legend,
        ax=g.ax_joint,
        hue=hue,
        palette=palette,
        fill=fill,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        **joint_kws,
    )
    # if legend and hue:
    #     _move_legend(g.ax_joint, legend_loc)

    if marginal_kind == "hist":
        sns.histplot(
            data=data,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            binrange=xlim,
            legend=False,
            **marginal_kws,
        )
        sns.histplot(
            data=data,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            binrange=ylim,
            legend=False,
            **marginal_kws,
        )
    elif marginal_kind == "kde":
        sns.kdeplot(
            data=data,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kws,
        )
        sns.kdeplot(
            data=data,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kws,
        )
    g.ax_marg_x.set_title(title, pad=6.0)

    return g


def _pop_style_kwargs(kwargs: dict[str, Any]) -> tuple:
    # Get style params / kwargs
    xlabel: str | None | Literal[False] = kwargs.pop(
        "xlabel", DEFAULT_STYLE_PARAMS["xlabel"]
    )
    ylabel: str | None | Literal[False] = kwargs.pop(
        "ylabel", DEFAULT_STYLE_PARAMS["ylabel"]
    )
    xlim: tuple[float, float] = kwargs.pop("xlim", DEFAULT_STYLE_PARAMS["xlim"])
    ylim: tuple[float, float] = kwargs.pop("ylim", DEFAULT_STYLE_PARAMS["ylim"])
    legend_loc: MplLegendLocType = kwargs.pop(
        "legend_loc", DEFAULT_STYLE_PARAMS["legend_loc"]
    )
    diagonal_lines: bool = kwargs.pop(
        "diagonal_lines", DEFAULT_STYLE_PARAMS["diagonal_lines"]
    )

    # Pull out any fontdict options which might be loose in the kwargs
    prim_ax_fontdict = kwargs.pop("prim_ax_fontdict", DEFAULT_XY_LABEL_FONTDICT.copy())
    for key in DEFAULT_XY_LABEL_FONTDICT:
        if key in kwargs:
            prim_ax_fontdict[key] = kwargs.pop(key)
    return xlabel, ylabel, xlim, ylim, legend_loc, diagonal_lines, prim_ax_fontdict


def _move_legend(
    ax: Axes,
    new_loc: MplLegendLocType,
    **kwargs,
) -> None:
    """
    Move legend to desired relative location.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    new_loc : MplLegendLocType
        The location of the legend

    """
    old_legend = ax.get_legend()
    if old_legend is None:
        logger.debug("_move_legend: No legend found for axis.")
        return
    handles = [h for h in old_legend.legend_handles if isinstance(h, Artist | tuple)]
    if not handles:
        logger.debug("_move_legend: No legend handles found.")
        return

    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    if len(handles) != len(labels):
        labels = labels[: len(handles)]
    ax.legend(handles, labels, loc=new_loc, title=title, **kwargs)


def _set_style() -> None:
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})


def _circumplex_grid(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str | Literal[False],
    ylabel: str | Literal[False],
    *,
    line_weights: float = 1.5,
    diagonal_lines: bool = False,
    prim_ax_fontdict: dict[str, str] | None = DEFAULT_XY_LABEL_FONTDICT,
) -> None:
    """
    Create the base layer grids and label lines for the soundscape circumplex.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        plt subplot Axes to add the circumplex grids to
    xlabel: str | Literal[False]
        Label for the x-axis, can be set to False to omit.
    ylabel: str | Literal[False]
        Label for the y-axis, can be set to False to omit.
    prim_labels: bool, optional
        flag for whether to include the custom primary labels ISOPleasant and ISOEventful
            by default True
        If using your own x and y names, you should set this to False.
    diagonal_lines : bool, optional
        flag for whether the include the diagonal dimensions (calm, etc)
            by default False

    """  # noqa: E501
    # Setting up the grids
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # grids and ticks
    ax.get_xaxis().set_minor_locator(AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(AutoMinorLocator())

    ax.grid(visible=True, which="major", color="grey", alpha=0.5)
    ax.grid(
        visible=True,
        which="minor",
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.4,
        zorder=DEFAULT_STYLE_PARAMS["prim_lines_zorder"],
    )

    _primary_lines_and_labels(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        line_weights=line_weights,
        prim_ax_fontdict=prim_ax_fontdict,
    )
    if diagonal_lines:
        _diagonal_lines_and_labels(ax, line_weights=line_weights)


def _set_circum_title(
    ax: Axes, title: str, xlabel: str | Literal[False], ylabel: str | Literal[False]
) -> None:
    """
    Set the title for the circumplex plot.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    prim_labels: bool, optional
        Whether to include the custom primary labels ISOPleasant and ISOEventful
        by default True
        If using your own x and y names, you should set this to False.
    title : str | None
        Title to set

    """
    title_pad = 30.0 if (xlabel is not False or ylabel is not False) else 6.0
    ax.set_title(title, pad=title_pad, fontsize=DEFAULT_STYLE_PARAMS["title_fontsize"])


def _deal_w_default_labels(
    x: str,
    y: str,
    xlabel: str | None | Literal[False],
    ylabel: str | None | Literal[False],
    prim_labels: bool | None,
) -> tuple[str | Literal[False], str | Literal[False]]:
    """
    Deal with the default labels for the circumplex plot.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    prim_labels: bool, optional
        flag for whether to include the custom primary labels ISOPleasant and ISOEventful
          by default True
        If using your own x and y names, you should set this to False.

    """
    xlabel = x if xlabel is None else xlabel  # xlabel = x col name if not provided
    ylabel = y if ylabel is None else ylabel

    if prim_labels is not None:
        warnings.warn(
            "The `prim_labels` argument is deprecated and will be removed. "
            "Use `xlabel` and `ylabel` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if prim_labels is False:
            xlabel = False
            ylabel = False

    return xlabel, ylabel


def _primary_lines_and_labels(
    ax: Axes,
    xlabel: str | Literal[False],
    ylabel: str | Literal[False],
    line_weights: float,
    prim_ax_fontdict: dict[str, str] | None = DEFAULT_XY_LABEL_FONTDICT,
) -> None:
    import re

    # Add lines and labels for circumplex model
    ## Primary Axes
    ax.axhline(  # Horizontal line
        y=0,
        color="grey",
        linestyle="dashed",
        alpha=1,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["prim_lines_zorder"],
    )
    ax.axvline(  # vertical line
        x=0,
        color="grey",
        linestyle="dashed",
        alpha=1,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["prim_lines_zorder"],
    )

    # Add labels for circumplex model
    # Check for math mode in labels
    if (re.search(r"\$.*\$", xlabel) if isinstance(xlabel, str) else False) or (
        re.search(r"\$.*\$", ylabel) if isinstance(ylabel, str) else False
    ):
        logger.warning(
            "parse_math is set to True, but $ $ indicates a math label. "
            "This may cause issues with the circumplex plot."
        )

    ax.set_xlabel(
        xlabel, fontdict=prim_ax_fontdict
    ) if xlabel is not False else ax.xaxis.label.set_visible(False)

    ax.set_ylabel(
        ylabel, fontdict=prim_ax_fontdict
    ) if ylabel is not False else ax.yaxis.label.set_visible(False)


def _diagonal_lines_and_labels(ax: Axes, line_weights: float) -> None:
    diag_ax_font = {
        "fontstyle": "italic",
        "fontsize": "small",
        "fontweight": "bold",
        "c": "black",
        "alpha": 0.5,
    }
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.plot(  # uppward diagonal
        [x_lim[0], x_lim[1]],
        [y_lim[0], y_lim[1]],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["diag_lines_zorder"],
    )
    ax.plot(  # downward diagonal
        [x_lim[0], x_lim[1]],
        [y_lim[1], y_lim[0]],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["diag_lines_zorder"],
    )

    ### Labels
    ax.text(  # Vibrant label
        x=x_lim[1] / 2,
        y=y_lim[1] / 2,
        s="(vibrant)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )
    ax.text(  # Chaotic label
        x=x_lim[0] / 2,
        y=y_lim[1] / 2,
        s="(chaotic)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )
    ax.text(  # monotonous label
        x=x_lim[0] / 2,
        y=y_lim[0] / 2,
        s="(monotonous)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )
    ax.text(  # calm label
        x=x_lim[1] / 2,
        y=y_lim[0] / 2,
        s="(calm)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )


def iso_annotation(
    ax: Axes,
    data: pd.DataFrame,
    location: str,
    *,
    x_adj: int = 0,
    y_adj: int = 0,
    x_key: str = DEFAULT_XCOL,
    y_key: str = DEFAULT_YCOL,
    ha: str = "center",
    va: str = "center",
    fontsize: str = "small",
    arrowprops: dict | None = None,
    **text_kwargs,
) -> None:
    """
    Add text annotations to circumplex plot based on coordinate values.

    Directly uses plt.annotate

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        existing plt axes to add to
    data : pd.Dataframe
        dataframe of coordinate points
    location : str
        name of the coordinate to plot
    x_adj : int, optional
        value to adjust x location by, by default 0
    y_adj : int, optional
        value to adjust y location by, by default 0
    x_key : str, optional
        name of x column, by default "ISOPleasant"
    y_key : str, optional
        name of y column, by default "ISOEventful"
    ha : str, optional
        horizontal alignment, by default "center"
    va : str, optional
        vertical alignment, by default "center"
    fontsize : str, optional
        by default "small"
    arrowprops : dict, optional
        dict of properties to send to plt.annotate,
        by default dict(arrowstyle="-", ec="black")

    """
    if arrowprops is None:
        arrowprops = {"arrowstyle": "-", "ec": "black"}

    ax.annotate(
        text=data["LocationID"][location],
        xy=(
            data[x_key][location],
            data[y_key][location],
        ),
        xytext=(
            data[x_key][location] + x_adj,
            data[y_key][location] + y_adj,
        ),
        ha=ha,
        va=va,
        arrowprops=arrowprops,
        annotation_clip=True,
        fontsize=fontsize,
        **text_kwargs,
    )


@functools.wraps(scatter)
def scatter_plot(*args, **kwargs) -> Axes:  # noqa: ANN002
    """
    Wrapper for the scatter function to maintain backwards compatibility.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the scatter function.
    **kwargs : dict
        Keyword arguments to pass to the scatter function.

    Returns
    -------
    Axes
        The Axes object containing the plot.

    """  # noqa: D401
    warnings.warn(
        "The `scatter_plot` function is deprecated and will be removed in a "
        "future version. Use `scatter` instead."
        "\nAs of v0.8, `scatter_plot` is an alias for `scatter` and does not maintain "
        "full backwards compatibility with v0.7. It may work, or some arguments may "
        "fail.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = [a for a in args if not isinstance(a, Backend)]
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("backend", "show_labels", "apply_styling")
    }

    return scatter(*args, **kwargs)


# @functools.wraps(density)
# def density_plot(*args, **kwargs) -> Axes:  # noqa: ANN002
#     """
#     Wrapper for the density function to maintain backwards compatibility.

#     Parameters
#     ----------
#     *args : tuple
#         Positional arguments to pass to the density function.
#     **kwargs : dict
#         Keyword arguments to pass to the density function.

#     Returns
#     -------
#     Axes
#         The Axes object containing the plot.

#     """  # noqa: D401
