"""Plotting functions for visualising circumplex data."""

# ruff: noqa: ANN003
import functools
import warnings
from collections.abc import Iterable
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
    DEFAULT_BW_ADJUST,
    DEFAULT_COLOR,
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_FIGSIZE,
    DEFAULT_POINT_SIZE,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SEABORN_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_XCOL,
    DEFAULT_XY_LABEL_FONTDICT,
    DEFAULT_YCOL,
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

    Creates a scatter plot of data on a standardized circumplex grid with the custom
    Soundscapy styling for soundscape circumplex visualisations.

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure containing coordinate data, typically with ISOPleasant
        and ISOEventful columns.
    x : str, optional
        Column name for x variable, by default "ISOPleasant"
    y : str, optional
        Column name for y variable, by default "ISOEventful"
    title : str | None, optional
        Title to add to circumplex plot, by default "Soundscape Scatter Plot"
    ax : matplotlib.axes.Axes, optional
        Pre-existing matplotlib axes for the plot, by default None
        If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.
    hue : str | np.ndarray | pd.Series | None, optional
        Grouping variable that will produce points with different colors.

        Can be either categorical or numeric,
        although color mapping will behave differently in latter case, by default None
    palette : SeabornPaletteType, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping, by default "colorblind"
    color : ColorType | None, optional
        Color to use for the plot elements when not using hue mapping,
        by default "#0173B2" (first color from colorblind palette)
    figsize : tuple[int, int], optional
        Size of the figure to return if `ax` is None, by default (5, 5)
    s : float, optional
        Size of scatter points, by default 20
    legend : {"auto", "brief", "full", False}, optional
        How to draw the legend. If "brief", numeric hue and size variables will be
        represented with a sample of evenly spaced values. If "full", every group will
        get an entry in the legend. If "auto", choose between brief or full
        representation based on number of levels.

        If False, no legend data is added and no legend is drawn, by default "auto"
    prim_labels : bool | None, optional
        Deprecated. Use xlabel and ylabel parameters instead.

    **kwargs : dict, optional
        Additional styling parameters:

        - xlabel, ylabel : str | Literal[False], optional
            Custom axis labels. By default "$P_{ISO}$" and "$E_{ISO}$"
            with math rendering.

            If None is passed, the column names (x and y) will be used as labels.

            If a string is provided, it will be used as the label.

            If False is passed, axis labels will be hidden.
        - xlim, ylim : tuple[float, float], optional
            Limits for x and y axes, by default (-1, 1) for both
        - legend_loc : MplLegendLocType, optional
            Location of legend, by default "best"
        - diagonal_lines : bool, optional
            Whether to include diagonal dimension labels (e.g. calm, etc.),
            by default False
        - prim_ax_fontdict : dict, optional
            Font dictionary for axis labels with these defaults:

            {
                "family": "sans-serif",
                "fontstyle": "normal",
                "fontsize": "large",
                "fontweight": "medium",
                "parse_math": True,
                "c": "black",
                "alpha": 1,
            }
        - fontsize, fontweight, fontstyle, family, c, alpha, parse_math:
            Direct parameters for font styling in axis labels

    Returns
    -------
    matplotlib.axes.Axes
        Axes object containing the plot.

    Notes
    -----
    This function applies special styling appropriate for circumplex plots including
    gridlines, axis labels, and proportional axes.

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

    Creates a kernel density estimate visualization of data distribution on a
    circumplex grid with the custom Soundscapy styling for soundscape circumplex
    visualisations. Can optionally include a scatter plot of the underlying data points.

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure containing coordinate data, typically with ISOPleasant
        and ISOEventful columns.
    x : str, optional
        Column name for x variable, by default "ISOPleasant"
    y : str, optional
        Column name for y variable, by default "ISOEventful"
    title : str | None, optional
        Title to add to circumplex plot, by default "Soundscape Density Plot"
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes object to use for the plot, by default None

        If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.
    hue : str | np.ndarray | pd.Series | None, optional
        Grouping variable that will produce density contours with different colors.
        Can be either categorical or numeric, although color mapping will behave
        differently in latter case, by default None
    incl_scatter : bool, optional
        Whether to include a scatter plot of the data points, by default True
    density_type : {"full", "simple"}, optional
        Type of density plot to draw. "full" uses default parameters, "simple"
        uses a lower number of levels (2), higher threshold (0.5), and lower alpha (0.5)
        for a cleaner visualization, by default "full"
    palette : SeabornPaletteType | None, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping, by default "colorblind"
    color : ColorType | None, optional
        Color to use for the plot elements when not using hue mapping,
        by default "#0173B2" (first color from colorblind palette)
    figsize : tuple[int, int], optional
        Size of the figure to return if `ax` is None, by default (5, 5)
    legend : {"auto", "brief", "full", False}, optional
        How to draw the legend. If "brief", numeric hue variables will be
        represented with a sample of evenly spaced values. If "full", every group will
        get an entry in the legend. If "auto", choose between brief or full
        representation based on number of levels.

        If False, no legend data is added and no legend is drawn, by default "auto"
    prim_labels : bool | None, optional
        Deprecated. Use xlabel and ylabel parameters instead.
    scatter_kws : dict | None, optional
        Keyword arguments to pass to `seaborn.scatterplot` if incl_scatter is True,
        by default {"s": 25, "linewidth": 0}
    incl_outline : bool, optional
        Whether to include an outline for the density contours, by default False
    alpha : float, optional
        Proportional opacity of the density fill, by default 0.8
    fill : bool, optional
        If True, fill in the area between bivariate contours, by default True
    levels : int | Iterable[float], optional
        Number of contour levels or values to draw contours at. A vector argument
        must have increasing values in [0, 1]. Levels correspond to iso-proportions
        of the density: e.g. 20% of the probability mass will lie below the
        contour drawn for 0.2, by default 10
    thresh : float, optional
        Lowest iso-proportional level at which to draw a contour line. Ignored when
        `levels` is a vector, by default 0.05
    bw_adjust : float, optional
        Factor that multiplicatively scales the bandwidth. Increasing will make
        the density estimate smoother, by default 1.2

    **kwargs : dict, optional
        Additional styling parameters:

        - xlabel, ylabel : str | Literal[False], optional
            Custom axis labels. By default "$P_{ISO}$" and "$E_{ISO}$" with math
            rendering.

            If None is passed, the column names (x and y) will be used as labels.

            If a string is provided, it will be used as the label.

            If False is passed, axis labels will be hidden.
        - xlim, ylim : tuple[float, float], optional
            Limits for x and y axes, by default (-1, 1) for both
        - legend_loc : MplLegendLocType, optional
            Location of legend, by default "best"
        - diagonal_lines : bool, optional
            Whether to include diagonal dimension labels (e.g. calm, etc.),
            by default False
        - prim_ax_fontdict : dict, optional
            Font dictionary for axis labels with these defaults:

            {
                "family": "sans-serif",
                "fontstyle": "normal",
                "fontsize": "large",
                "fontweight": "medium",
                "parse_math": True,
                "c": "black",
                "alpha": 1,
            }
        - fontsize, fontweight, fontstyle, family, c, alpha, parse_math:
            Direct parameters for font styling in axis labels

        Also accepts additional keyword arguments for matplotlib's contour and contourf
        functions.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object containing the plot.

    Notes
    -----
    This function will raise a warning if the dataset has fewer than
    RECOMMENDED_MIN_SAMPLES (30) data points, as density plots are not reliable
    with small sample sizes.

    """
    # Check if dataset is large enough for density plots
    _valid_density(data)

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


def jointplot(
    data: pd.DataFrame,
    *,
    x: str = DEFAULT_XCOL,
    y: str = DEFAULT_YCOL,
    title: str | None = "Soundscape Joint Plot",
    hue: str | np.ndarray | pd.Series | None = None,
    incl_scatter: bool = True,
    density_type: str = "full",
    palette: SeabornPaletteType | None = "colorblind",
    color: ColorType | None = DEFAULT_COLOR,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    scatter_kws: dict[str, Any] | None = None,
    incl_outline: bool = False,
    alpha: float = DEFAULT_SEABORN_PARAMS["alpha"],
    fill: bool = True,
    levels: int | Iterable[float] = 10,
    thresh: float = 0.05,
    bw_adjust: float = DEFAULT_BW_ADJUST,
    legend: Literal["auto", "brief", "full", False] = "auto",
    prim_labels: bool | None = None,  # Alias for primary_labels, deprecated
    joint_kws: dict[str, Any] | None = None,
    marginal_kws: dict[str, Any] | None = None,
    marginal_kind: str = "kde",
    **kwargs,
) -> sns.JointGrid:
    """
    Create a jointplot with a central distribution and marginal plots.

    Creates a visualization with a main plot (density or scatter) in the center and
    marginal distribution plots along the x and y axes. The main plot uses the custom
    Soundscapy styling for soundscape circumplex visualisations, and the marginals show
    the individual distributions of each variable.

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure containing coordinate data, typically with ISOPleasant
        and ISOEventful columns.
    x : str, optional
        Column name for x variable, by default "ISOPleasant"
    y : str, optional
        Column name for y variable, by default "ISOEventful"
    title : str | None, optional
        Title to add to the jointplot, by default "Soundscape Joint Plot"
    hue : str | np.ndarray | pd.Series | None, optional
        Grouping variable that will produce plots with different colors.
        Can be either categorical or numeric, although color mapping will behave
        differently in latter case, by default None
    incl_scatter : bool, optional
        Whether to include a scatter plot of the data points in the joint plot,
        by default True
    density_type : {"full", "simple"}, optional
        Type of density plot to draw. "full" uses default parameters, "simple"
        uses a lower number of levels (2), higher threshold (0.5), and lower alpha (0.5)
        for a cleaner visualization, by default "full"
    palette : SeabornPaletteType | None, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping, by default "colorblind"
    color : ColorType | None, optional
        Color to use for the plot elements when not using hue mapping,
        by default "#0173B2" (first color from colorblind palette)
    figsize : tuple[int, int], optional
        Size of the figure to create (determines height, width is proportional),
        by default (5, 5)
    scatter_kws : dict[str, Any] | None, optional
        Additional keyword arguments to pass to scatter plot if incl_scatter is True,
        by default None
    incl_outline : bool, optional
        Whether to include an outline for the density contours, by default False
    alpha : float, optional
        Opacity level for the density fill, by default 0.8
    fill : bool, optional
        Whether to fill the density contours, by default True
    levels : int | Iterable[float], optional
        Number of contour levels or specific levels to draw. A vector argument
        must have increasing values in [0, 1], by default 10
    thresh : float, optional
        Lowest iso-proportion level at which to draw contours, by default 0.05
    bw_adjust : float, optional
        Factor that multiplicatively scales the bandwidth. Increasing will make
        the density estimate smoother, by default 1.2
    legend : {"auto", "brief", "full", False}, optional
        How to draw the legend for hue mapping, by default "auto"
    prim_labels : bool | None, optional
        Deprecated. Use xlabel and ylabel parameters instead.
    joint_kws : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the joint plot, by default None
    marginal_kws : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the marginal plots,
        by default {"fill": True, "common_norm": False}
    marginal_kind : {"kde", "hist"}, optional
        Type of plot to draw in the marginal axes, either "kde" for kernel
        density estimation or "hist" for histogram, by default "kde"

    **kwargs : dict, optional
        Additional styling parameters:

        - xlabel, ylabel : str | Literal[False], optional
            Custom axis labels. By default "$P_{ISO}$" and "$E_{ISO}$" with
            math rendering.

            If None is passed, the column names (x and y) will be used as labels.

            If a string is provided, it will be used as the label.

            If False is passed, axis labels will be hidden.
        - xlim, ylim : tuple[float, float], optional
            Limits for x and y axes, by default (-1, 1) for both
        - legend_loc : MplLegendLocType, optional
            Location of legend, by default "best"
        - diagonal_lines : bool, optional
            Whether to include diagonal dimension labels (e.g. calm, etc.),
            by default False
        - prim_ax_fontdict : dict, optional
            Font dictionary for axis labels with these defaults:

            {
                "family": "sans-serif",
                "fontstyle": "normal",
                "fontsize": "large",
                "fontweight": "medium",
                "parse_math": True,
                "c": "black",
                "alpha": 1,
            }

    Returns
    -------
    sns.JointGrid
        The seaborn JointGrid object containing the plot

    Notes
    -----
    This function will raise a warning if the dataset has fewer than
    RECOMMENDED_MIN_SAMPLES (30) data points, as density plots are not reliable
    with small sample sizes.

    """
    # Check if dataset is large enough for density plots
    _valid_density(data)

    # Initialize default dicts if None
    scatter_kws = {} if scatter_kws is None else scatter_kws
    joint_kws = {} if joint_kws is None else joint_kws
    marginal_kws = (
        {"fill": True, "common_norm": False} if marginal_kws is None else marginal_kws
    )

    # Get style parameters
    xlabel, ylabel, xlim, ylim, legend_loc, diagonal_lines, prim_ax_fontdict = (
        _pop_style_kwargs(kwargs)
    )

    if density_type == "simple":
        thresh = DEFAULT_SIMPLE_DENSITY_PARAMS["thresh"]
        levels = DEFAULT_SIMPLE_DENSITY_PARAMS["levels"]
        alpha = DEFAULT_SIMPLE_DENSITY_PARAMS["alpha"]
        incl_outline = True

    # Handle hue and color
    if hue is None:
        # Removes the palette if no hue is specified
        palette = None
        color = sns.color_palette("colorblind", 1)[0] if color is None else color

    # Create the joint grid
    g = sns.JointGrid(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        height=figsize[0],  # Use figsize for height
        ratio=3,  # Default ratio of joint to marginal plots
        space=0.2,  # Space between joint and marginal plots
        xlim=xlim,
        ylim=ylim,
    )

    # Add the density plot to the joint plot area
    density(
        data,
        x=x,
        y=y,
        incl_scatter=incl_scatter,
        density_type=density_type,
        title=None,  # We'll set the title separately
        ax=g.ax_joint,
        hue=hue,
        palette=palette,
        color=color,
        scatter_kws=scatter_kws,
        incl_outline=incl_outline,
        legend_loc=legend_loc,
        alpha=alpha,
        legend=legend,
        fill=fill,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        diagonal_lines=diagonal_lines,
        xlim=xlim,
        ylim=ylim,
        **joint_kws,
    )

    # Add the marginal plots
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

    # Set title
    if title is not None:
        g.ax_marg_x.set_title(title, pad=6.0)

    # Process primary labels (x and y)
    xlabel, ylabel = _deal_w_default_labels(
        x=x, y=y, xlabel=xlabel, ylabel=ylabel, prim_labels=prim_labels
    )

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
        whether to include the custom primary labels ISOPleasant and ISOEventful
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


@functools.wraps(density)
def density_plot(*args, **kwargs) -> Axes:  # noqa: ANN002
    """
    Wrapper for the density function to maintain backwards compatibility.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the density function.
    **kwargs : dict
        Keyword arguments to pass to the density function.

    Returns
    -------
    Axes
        The Axes object containing the plot.

    """  # noqa: D401
    warnings.warn(
        "The `density_plot` function is deprecated and will be removed in a "
        "future version. Use `density` instead."
        "\nAs of v0.8, `density_plot` is an alias for `density` and does not maintain "
        "full backwards compatibility with v0.7. It may work, or some arguments may "
        "fail.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = [a for a in args if not isinstance(a, Backend)]
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in (
            "backend",
            "show_labels",
            "apply_styling",
            "simple_density",
            "simple_density_thresh",
            "simple_density_levels",
            "simple_density_alpha",
        )
    }

    # Convert simple_density parameters to the new API if they exist
    if "density_type" not in kwargs and kwargs.pop("simple_density", False):
        kwargs["density_type"] = "simple"

    return density(*args, **kwargs)


def _valid_density(data: pd.DataFrame) -> None:
    """
    Check if the data is valid for density plots.

    Raises a warning if the dataset is too small for reliable density estimation.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be checked

    Raises
    ------
    UserWarning
        If the data is too small for density plots (< RECOMMENDED_MIN_SAMPLES).

    """
    if len(data) < RECOMMENDED_MIN_SAMPLES:
        warnings.warn(
            "Density plots are not recommended for "
            f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
            UserWarning,
            stacklevel=2,
        )
