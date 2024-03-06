"""Plotting functions for visualising circumplex data."""

from typing import Tuple, Union

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

diag_lines_zorder = 1
diag_labels_zorder = 4
prim_lines_zorder = 2
data_zorder = 3
default_bw_adjust = 1.2
default_figsize = (5, 5)

simple_density = dict(thresh=0.5, levels=2, incl_outline=True, alpha=0.5)


def scatter(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    title: str = "Soundscape Scatter Plot",
    diagonal_lines: bool = False,
    xlim: Tuple[int] = (-1, 1),
    ylim: Tuple[int] = (-1, 1),
    figsize: Tuple[int] = (5, 5),
    legend_loc: str = "lower left",
    hue: str = None,
    s: int = 20,
    palette: str = "colorblind",
    legend: str = "auto",
    ax: mpl.axes.Axes = None,
    **kwargs,
):
    """Plot ISOcoordinates as scatter points on a soundscape circumplex grid

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure. Either a long-form collection of vectors that can be assigned to
        named variables or a wide-form dataset that will be internally reshaped.
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
        Grouping variable that will produce points with different colors. Can be either categorical or numeric,
        although color mapping will behave differently in latter case, by default None
    s : int, optional
        size of scatter points, by default 20
    palette : string, list, dict or matplotlib.colors.Colormap, optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to
        seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping.
        by default colorblind
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
    Axes object containing the plot.
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
        palette=palette,
        ax=ax,
        zorder=data_zorder,
        s=s,
        legend=legend,
        **kwargs,
    )
    ax = _deal_w_default_labels(ax, False)
    _circumplex_grid(ax, False, diagonal_lines, xlim, ylim)
    _set_circum_title(ax, False, title)
    if legend and hue:
        _move_legend(ax, legend_loc)
    return s


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
    legend: bool = False,
    ax: matplotlib.axes.Axes = None,
    hue: str = None,
    palette="colorblind",
    color=None,
    fill: bool = True,
    levels: int = 10,
    thresh: float = 0.05,
    bw_adjust=None,
    **kwargs,
):
    """Plot a density plot of ISOCoordinates.

    Creates a wrapper around `seaborn.kdeplot` and adds functionality and styling to customise it for circumplex plots.
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
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if bw_adjust is None:
        bw_adjust = default_bw_adjust

    if hue is None:
        # Removes the palette if no hue is specified
        palette = None
        color = sns.color_palette("colorblind", 1)[0] if color is None else color

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
            ax=ax,
            hue=hue,
            palette=palette,
            levels=levels,
            thresh=thresh,
            bw_adjust=bw_adjust,
            fill=False,
            zorder=data_zorder,
            color=color,
            **kwargs,
        )

    d = sns.kdeplot(
        data=data,
        x=x,
        y=y,
        alpha=alpha,
        legend=legend,
        ax=ax,
        hue=hue,
        palette=palette,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        fill=fill,
        zorder=data_zorder,
        color=color,
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
    """Create a jointplot with distribution or scatter in the center and distributions on the margins.

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
