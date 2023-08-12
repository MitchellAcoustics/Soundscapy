"""Plotting functions for visualising circumplex data."""

from typing import Union, Tuple, List

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

diag_lines_zorder = 1
diag_labels_zorder = 4
prim_lines_zorder = 2
data_zorder = 3
default_bw_adjust = 1.2
default_figsize = (5, 5)

simple_density = dict(thresh=0.5, levels=2, incl_outline=True, alpha=0.5)


def scatter(
    data=None,
    x="ISOPleasant",
    y="ISOEventful",
    title="Soundscape Scatter Plot",
    diagonal_lines=False,
    xlim=(-1, 1),
    ylim=(-1, 1),
    figsize=(5, 5),
    legend_loc="lower left",
    hue=None,
    style=None,
    s=20,
    palette="colorblind",
    hue_order=None,
    hue_norm=None,
    sizes=None,
    size_order=None,
    size_norm=None,
    markers=True,
    style_order=None,
    alpha=None,
    legend="auto",
    ax=None,
    **scatter_kws,
):
    """Plot ISOcoordinates as scatter points on a soundscape circumplex grid

    Makes use of seaborn.scatterplot. We have made all of the `seaborn.scatterplot` arguments available, but have also added or changed some specific
    options for circumplex plotting.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, mapping or sequence
        Input data structure. Either a long-form collection of vectors that can be assigned to
        named variables or a wide-form dataset that will be internally reshaped.
    x : vector or key in `data`, optional
        column name for x variable, by default "ISOPleasant"
    y : vector or key in `data`, optional
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
    palette : string, list, dict or matplotlib.colors.Colormap, optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to
        seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping.
        by default colorblind
    s : int, optional
        size of scatter points, by default 10
    hue : vector or key in data, optional
        Grouping variable that will produce points with different colors. Can be either categorical or numeric,
        although color mapping will behave differently in latter case, by default None
    style : vector or key in data
        Grouping variable that will produce points with different markers. Can have a numeric dtype but will always
        be treated as categorical.
    hue_order : vector of strings
        Specify the order of processing and plotting for categorical levels of the `hue` semantic
    hue_norm : tuple or matplotlib.colors.Normalize
        Either a pair of values that set the normalization range in data units or an object that will map from data
        units into a [0, 1] interval. Usage implies numeric mapping.
    sizes : list, dict, or tuple
        An object that determines how sizes are chosen when `size` is used. It can always be a list of size values or
        a dict mapping levels of the `size` variable to sizes. When `size` is numeric, it can also be a tuple
        specifying the minimum and maximum size to use such that other values are normalized within this range.
    size_order : list
        Specified order for appearance of the `size` variable levels, otherwise they are determined from the data. Not
        relevant when the `size` variable is numeric.
    size_norm : tuple or Normalization object
        Normalization in data units for scaling plot objects when the `size` variable is numeric.
    markers : boolean, list, or dictionary
        Object determining how to draw the markers for different levels of the `style` variable. Setting to `True` will
        use default markers, or you can pass a list of markers or a dictionary mapping levels of the `style` variable
        to markers. Setting to `False` will draw marker-less lines. Markers are specified as in matplotlib.
    style_order : list
        Specified order for appearance of the `style` variable levels otherwise they are determined from the data. Not
        relevant when the `style` variable is numeric.
    alpha : float
        Proportional opacity of the points.
    legend : {"auto", "brief", "full" or False}, optional
        How to draw the legend. If “brief”, numeric hue and size variables will be represented with a sample of evenly
        spaced values. If “full”, every group will get an entry in the legend. If “auto”, choose between brief or full
        representation based on number of levels. If False, no legend data is added and no legend is drawn.
        by default, "auto"
    ax : matplotlib.axes.Axes, optional
        Pre-existing matplotlib axes for the plot, by default None
        If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if hue is None:
        # Removes the palette if no hue is specified
        palette = None

    s = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        style=style,
        s=s,
        palette=palette,
        hue_order=hue_order,
        hue_norm=hue_norm,
        sizes=sizes,
        size_order=size_order,
        size_norm=size_norm,
        markers=markers,
        style_order=style_order,
        alpha=alpha,
        legend=legend,
        ax=ax,
        zorder=data_zorder,
        **scatter_kws,
    )
    ax = _deal_w_default_labels(ax, False)
    _circumplex_grid(ax, False, diagonal_lines, xlim, ylim)
    _set_circum_title(ax, False, title)
    if legend and hue:
        _move_legend(ax, legend_loc)
    return s


# TODO: Consider changing to displot
def density(
    data: Union[pd.DataFrame, np.ndarray] = None,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    incl_scatter: bool = True,
    density_type: str = "full",
    title: Union[str, None] = "Soundscapy Density Plot",
    diagonal_lines: bool = False,
    xlim: tuple = (-1, 1),
    ylim: tuple = (-1, 1),
    scatter_kws: dict = dict(s=25, linewidth=0),
    incl_outline: bool = False,
    figsize: tuple = (5, 5),
    legend_loc: str = "lower left",
    alpha: float = 0.75,
    gridsize: int = 200,
    kernel: str = None,
    cut: Union[float, int] = 3,
    clip: Tuple[int] = None,
    legend: bool = False,
    cumulative: bool = False,
    cbar: bool = False,
    cbar_ax: matplotlib.axes.Axes = None,
    cbar_kws: dict = None,
    ax: matplotlib.axes.Axes = None,
    weights: str = None,
    hue: str = None,
    palette="colorblind",
    hue_order: List[str] = None,
    hue_norm=None,
    multiple: str = "layer",
    common_norm: bool = False,
    common_grid: bool = False,
    levels: int = 10,
    thresh: float = 0.05,
    bw_method="scott",
    bw_adjust: Union[float, int] = None,
    log_scale: Union[bool, int, float] = None,
    color: str = "blue",
    fill: bool = True,
    warn_singular: bool = True,
    **kwargs,
):
    """Plot a density plot of ISOCoordinates.

    Creates a wrapper around `seaborn.kdeplot` and adds functionality and styling to customise it for circumplex plots.
    The density plot is a combination of a kernel density estimate and a scatter plot.

    Parameters
    ----------
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
    gridsize : int, optional
        Nuber of points on each dimension of the evaluation grid, by default 200
    kernel : str, optional
        Function that defines the kernel, by default None
    cut : Union[float, int], optional
        Factor, multiplied by the smoothing bandwidth, that determines how far the evaluation grid extends past the
        extreme datapoints. When set to 0, truncate the curve at the data limits, by default 3
    clip : tuple[int], optional
        Do not evaluate the density outside of these limits, by default None
    legend : bool, optional
        If False, suppress the legend for semantic variables, by default True
    cumulative : bool, optional
        If True, estimate a cumulative distribution function, by default False
    cbar : bool, optional
        If True, add a colorbar to annotate the color mapping in a bivariate plot. Note: does not currently support
        plots with a `hue` variable well, by default False
    cbar_ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the colorbar, by default None
    cbar_kws : dict, optional
        Keyword arguments for the colorbar, by default None
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes object to use for the plot, by default None
    weights : vector or key in `data`, optional
        If provided, weight the kernel density estimation using these values, by default None
    hue : vector or key in `data`, optional
        Semantic variable that is mapped to determine the color of plot elements, by default None
    palette : Union[str, list, dict, matplotlib.colors.Colormap], optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to
        seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping.
        by default colorblind
    hue_order : list[str], optional
        Specify the order of processing and plotting for categorical levels of the `hue` semantic, by default None
    hue_norm : Union[tuple, matplotlib.colors.Normalize], optional
        Either a pair of values that set the normalization range in data units or an object , by default None
    multiple : {"layer", "stack", "fill"}, optional
        Whether to plot multiple elements when semantic mapping creates subsets. Only relevant with univariate data,
        by default 'layer'
    common_norm : bool, optional
        If True, scale each conditional density by the number of observations such that the total area under all
        densities sums to 1. Otherwise, normalize each density independently, by default False
    common_grid : bool, optional
        If True, use the same evaluation grid for each kernel density estimate. Only relevant with univariate data.
        by default, False
    levels : int or vector, optional
        Number of contour levels or values to draw contours at. A vector argument must have increasing values in [0, 1].
         Levels correspond to iso-proportions of the density: e.g., 20% of the probability mass will lie below the
         contour drawn for 0.2. Only relevant with bivariate data.
         by default, 10
    thresh : number in [0, 1]
        Lowest iso-proportion level at which to draw a contour line. Ignored with `levels` is a vector. Only relevant
        with bivariate data.
    bw_method : string, scalar, or callable, optional
        Method for determining the smoothing bandwidth to use; passed to `scipy.stats.gaussian_kde`.
    bw_adjust : number, optional
        Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
        See Notes.
    log_scale : bool or number, or pair of bools or numbers, optional
        Set axis scale(s) to log. A single value sets the data axis for univariate distributions and both axes for
        bivariate distributions. A pair of values sets each axis independently. Numeric values are interpreted as the
        desired base (default 10). If False, defer to the existing Axes scale.
        by default None
    color : matplotlib color
        Single color specification for when hue mapping is not used. Otherwise the plot will try to hook into the
        matplotlib property cycle, by default "blue"
    fill : bool, optional
        If True, fill in the area under univariate density curves or between bivariate contours. If None, the default
        depends on `multiple`. by default True.
    warn_singular : bool, optional
        If True, issue a warning when trying to estimate the density of data with zero variance, by default True
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
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if bw_adjust is None:
        bw_adjust = default_bw_adjust

    if hue is None:
        # Removes the palette if no hue is specified
        palette = None

    if density_type == "simple":
        thresh = simple_density["thresh"]
        levels = simple_density["levels"]
        alpha = simple_density["alpha"]
        incl_outline = simple_density["incl_outline"]

    if incl_scatter:
        d = sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            palette=palette,
            zorder=data_zorder,
            **scatter_kws,
        )

    if incl_outline:
        d = sns.kdeplot(
            data=data,
            x=x,
            y=y,
            alpha=1,
            gridsize=gridsize,
            kernel=kernel,
            cut=cut,
            clip=clip,
            cumulative=cumulative,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            ax=ax,
            weights=weights,
            hue=hue,
            palette=palette,
            hue_order=hue_order,
            hue_norm=hue_norm,
            multiple=multiple,
            common_norm=common_norm,
            common_grid=common_grid,
            levels=levels,
            thresh=thresh,
            bw_method=bw_method,
            bw_adjust=bw_adjust,
            log_scale=log_scale,
            color=color,
            fill=False,
            warn_singular=warn_singular,
            zorder=data_zorder,
            **kwargs,
        )

    d = sns.kdeplot(
        data=data,
        x=x,
        y=y,
        alpha=alpha,
        gridsize=gridsize,
        kernel=kernel,
        cut=cut,
        clip=clip,
        legend=legend,
        cumulative=cumulative,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        ax=ax,
        weights=weights,
        hue=hue,
        palette=palette,
        hue_order=hue_order,
        hue_norm=hue_norm,
        multiple=multiple,
        common_norm=common_norm,
        common_grid=common_grid,
        levels=levels,
        thresh=thresh,
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        log_scale=log_scale,
        color=color,
        fill=fill,
        warn_singular=warn_singular,
        zorder=data_zorder,
        **kwargs,
    )

    _circumplex_grid(
        ax, prim_labels=False, diagonal_lines=diagonal_lines, xlim=xlim, ylim=ylim
    )
    _set_circum_title(ax, prim_labels=False, title=title)
    _deal_w_default_labels(ax, prim_labels=False)
    if legend:
        _move_legend(ax, legend_loc)

    return d


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
    color=None,
    joint_kws={},
    marginal_kws={"fill": True, "common_norm": False},
    hue=None,
    palette="colorblind",
    hue_order=None,
    hue_norm=None,
    common_norm=False,
    fill=True,
    bw_adjust=None,
    thresh=0.1,
    levels=10,
    legend=False,
    marginal_kind="kde",
):
    """Create a jointplot with distribution or scatter in the center and distributions on the margins.

    This method works by calling sns.jointplot() and creating a circumplex grid in the joint position, then
    overlaying a density or circumplex_scatter plot. The options for both the joint and marginal plots can be
    passed through the sns.jointplot() separately to customise them separately. The marginal distribution plots
    can be either a density or histogram.

    Parameters
    ----------
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
    hue_order : vector of strings, optional.
        Specify the order of processing and plotting for categorical levels of the `hue` semantic.
    hue_norm : tuple or matplotlib.colors.Normalize, optional
        Either a pair of values that set the normalization range in data units or an object that will map from data
        units into a [0, 1] interval. Usage implies numeric mapping.
    common_norm : bool, optional
        If True, scale each conditional density by the number of observations such that the total area under all
        densities sums to 1. Otherwise, normalize each density independently, by default False.
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
        hue_order=hue_order,
        hue_norm=hue_norm,
        common_norm=common_norm,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        color=color,
        fill=fill,
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


def _move_legend(ax, new_loc, **kws):
    """Moves legend to desired relative location.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    new_loc : str or pair of floats
        The location of the legend
    """
    old_legend = ax.get_legend()
    handles = old_legend.legend_handles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


def _circumplex_grid(
    ax, prim_labels=True, diagonal_lines=False, xlim=(-1, 1), ylim=(-1, 1)
):
    """Create the base layer grids and label lines for the soundscape circumplex

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        plt subplot Axes to add the circumplex grids to
    prim_labsl: bool, optional
        flag for whether to include the custom primary labels ISOPleasant and ISOEventful, by default True
        If using your own x and y names, you should set this to False.
    diagonal_lines : bool, optional
        flag for whether the include the diagonal dimensions (calm, etc), by default False
    """
    # Setting up the grids
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    line_weights = 1.5
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # grids and ticks
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(visible=True, which="major", color="grey", alpha=0.5)
    ax.grid(
        visible=True,
        which="minor",
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.4,
        zorder=prim_lines_zorder,
    )

    ax = _primary_lines_and_labels(ax, prim_labels, line_weights=line_weights)
    if diagonal_lines:
        ax = _diagonal_lines_and_labels(ax, line_weights=line_weights)

    return ax


def _set_circum_title(ax, prim_labels, title):
    """Set the title for the circumplex plot

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    prim_labels: bool, optional
        flag for whether to include the custom primary labels ISOPleasant and ISOEventful, by default True
        If using your own x and y names, you should set this to False.
    title : str
        Title to set
    """
    title_pad = 30.0 if prim_labels is True else 6.0
    ax.set_title(title, pad=title_pad)
    return ax


def _deal_w_default_labels(ax, prim_labels):
    """Deal with the default labels for the circumplex plot

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    prim_labels: bool, optional
        flag for whether to include the custom primary labels ISOPleasant and ISOEventful, by default True
        If using your own x and y names, you should set this to False.
    """
    if prim_labels is True or prim_labels == "none":
        # hide axis labels
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

    return ax


def _primary_lines_and_labels(ax, prim_labels, line_weights):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # Add lines and labels for circumplex model
    ## Primary Axes
    ax.plot(  # Horizontal line
        [x_lim[0], x_lim[1]],
        [0, 0],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
        zorder=prim_lines_zorder,
    )
    ax.plot(  # vertical line
        [0, 0],
        [y_lim[0], y_lim[1]],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
        zorder=prim_lines_zorder,
    )

    if prim_labels is True:
        prim_ax_font = {
            "fontstyle": "italic",
            "fontsize": "medium",
            "fontweight": "bold",
            "c": "grey",
            "alpha": 1,
        }
        ### Labels
        ax.text(  # ISOPleasant Label
            x=x_lim[1] + 0.01,
            y=0,
            s="ISO\nPleasant",
            ha="left",
            va="center",
            fontdict=prim_ax_font,
        )
        ax.text(  # ISOEventful Label
            x=0,
            y=y_lim[1] + 0.01,
            s="ISO\nEventful",
            ha="center",
            va="bottom",
            fontdict=prim_ax_font,
        )

    return ax


def _diagonal_lines_and_labels(ax, line_weights):
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
        zorder=diag_lines_zorder,
    )
    ax.plot(  # downward diagonal
        [x_lim[0], x_lim[1]],
        [y_lim[1], y_lim[0]],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=diag_lines_zorder,
    )

    ### Labels
    ax.text(  # Vibrant label
        x=x_lim[1] / 2,
        y=y_lim[1] / 2,
        s="(vibrant)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    ax.text(  # Chaotic label
        x=x_lim[0] / 2,
        y=y_lim[1] / 2,
        s="(chaotic)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    ax.text(  # monotonous label
        x=x_lim[0] / 2,
        y=y_lim[0] / 2,
        s="(monotonous)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    ax.text(  # calm label
        x=x_lim[1] / 2,
        y=y_lim[0] / 2,
        s="(calm)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    return ax


# Unsupported right now.


def single_lmplot(
    sf,
    x,
    y,
    groups=None,
    group_order=None,
    ylim=None,
    xlim=None,
    order=1,
    fit_reg=True,
    corr="pearson",
    **kwargs,
):
    """Unfinished and unsupported"""
    if y in ["ISOPleasant", "ISOEventful"] and ylim is None:
        ylim = (-1, 1)
    if corr:
        df = sf[[y, x]].dropna()
        if corr == "pearson":
            r, p = pearsonr(df[y], df[x])
        elif corr == "spearman":
            r, p == spearmanr(df[y], df[x])

        if p < 0.01:
            symb = "**"
        elif p < 0.05:
            symb = "*"
        else:
            symb = ""
        text = f"{round(r, 2)}{symb}"
    else:
        text = None

    if groups is None:
        g = sns.lmplot(
            x=x,
            y=y,
            data=sf,
            height=8,
            aspect=1,
            order=order,
            fit_reg=fit_reg,
            **kwargs,
        )
    else:
        g = sns.lmplot(
            x=x,
            y=y,
            data=sf,
            height=8,
            aspect=1,
            order=order,
            hue=groups,
            hue_order=group_order,
            fit_reg=fit_reg,
            **kwargs,
        )

    g.set(ylim=ylim)
    g.set(xlim=xlim)

    ax = g.axes.ravel()[0]
    ax.text(
        np.mean(ax.get_xlim()),
        min(ax.get_ylim()) * 0.75,
        text,
        ha="center",
        fontweight=750,
    )

    return g


def grouped_lmplot(
    sf,
    x,
    y,
    groups,
    group_order=None,
    ylim=None,
    xlim=None,
    order=1,
    fit_reg=True,
    corr="pearson",
    col_wrap=4,
    scatter_kws=None,
    line_kws=None,
    facet_kws={"sharex": False},
    **kwargs,
):
    """Unfinished and unsupported"""
    if y in ["ISOPleasant", "ISOEventful"] and ylim is None:
        ylim = (-1, 1)
    grid = sns.lmplot(
        x=x,
        y=y,
        col=groups,
        order=order,
        col_wrap=col_wrap,
        data=sf,
        fit_reg=fit_reg,
        height=4,
        facet_kws=facet_kws,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        **kwargs,
    )
    grid.set(ylim=ylim)
    grid.set(xlim=xlim)
    if group_order is None:
        group_order = sf[groups].unique()

    if corr:
        for location, ax in zip(group_order, grid.axes.ravel()):
            df = sf.loc[sf[groups] == location, [y, x]].dropna()
            if corr == "pearson":
                r, p = pearsonr(df[y], df[x])
            elif corr == "spearman":
                r, p = spearmanr(df[y], df[x])
            if p < 0.01:
                symb = "**"
            elif p < 0.05:
                symb = "*"
            else:
                symb = ""
            fontweight = 750 if p < 0.05 else None

            text = f"{round(r, 2)}{symb}"
            ax.text(
                np.mean(ax.get_xlim()),
                min(ax.get_ylim()) * 0.75,
                text,
                ha="center",
                fontweight=fontweight,
            )
    grid.set_titles("{col_name}")
    return grid


def iso_annotation(
    ax,
    data,
    location,
    x_adj=0,
    y_adj=0,
    x_key="ISOPleasant",
    y_key="ISOEventful",
    ha="center",
    va="center",
    fontsize="small",
    arrowprops=dict(arrowstyle="-", ec="black"),
    **text_kwargs,
):
    """add text annotations to circumplex plot based on coordinate values

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
        dict of properties to send to plt.annotate, by default dict(arrowstyle="-", ec="black")

    Example
    -------
    >>> fig, axes = plt.subplots(1,1, figsize=(5,5))
    >>> df_mean.isd.scatter(xlim=(-.5, .5),ylim=(-.5, .5),ax=axes)
    >>> for location in df_mean.LocationID:
    >>>     plotting.iso_annotation(axes, df_mean, location)
    """
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
