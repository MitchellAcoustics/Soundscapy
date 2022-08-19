"""Plotting functions for visualising circumplex data."""

#%%
import matplotlib.axes
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Union

diag_lines_zorder = 1
diag_labels_zorder = 4
prim_lines_zorder = 2
data_zorder = 3
default_bw_adjust = 1.2
default_figsize = (5, 5)

#%%

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
    x_bins=None,
    y_bins=None,
    units=None,
    estimator=None,
    ci=95,
    n_boot=1000,
    alpha=None,
    x_jitter=None,
    y_jitter=None,
    legend="auto",
    ax=None,
    **scatter_kws,
):
    """Plot ISOcoordinates as scatter points on a soundscape circumplex grid

    Makes use of seaborn.scatterplot

    Parameters
    ----------
    style
    hue_order
    hue_norm
    sizes
    size_order
    size_norm
    markers
    style_order
    x_bins
    y_bins
    units
    estimator
    ci
    n_boot
    alpha
    x_jitter
    y_jitter
    legend
    ax : plt.Axes, optional
        existing matplotlib axes, by default None
    title : str, optional
        , by default "Soundscape Scatter Plot"
    hue : vector or key in data, optional
        Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case, by default None
    x : str, optional
        column name for x variable, by default "ISOPleasant"
    y : str, optional
        column name for y variable, by default "ISOEventful"
    diagonal_lines : bool, optional
        whether to include diagonal dimension labels (e.g. calm, etc.), by default False
    figsize : tuple, optional
        by default (5, 5)
    palette : string, list, dict or matplotlib.colors.Colormap, optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object implies numeric mapping.
        by default colorblind
    legend : bool, optional
        whether to include legend with the hue values, by default False
    legend_loc : str, optional
        relative location of legend, by default "lower left"
    s : int, optional
        size of scatter points, by default 10

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

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
        x_bins=x_bins,
        y_bins=y_bins,
        units=units,
        estimator=estimator,
        ci=ci,
        n_boot=n_boot,
        alpha=alpha,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
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
    title: str = "Soundscapy Density Plot",
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
    clip: tuple[int] = None,
    legend: bool = False,
    cumulative: bool = False,
    cbar: bool = False,
    cbar_ax: matplotlib.axes.Axes = None,
    cbar_kws: dict = None,
    ax: matplotlib.axes.Axes = None,
    weights: str = None,
    hue: str = None,
    palette="colorblind",
    hue_order: list[str] = None,
    hue_norm=None,
    multiple: str = "layer",
    common_norm: bool = False,
    common_grid: bool = False,
    levels: int = 10,
    thresh: float = 0.05,
    bw_method="scott",
    bw_adjust: Union[float, int] = 1,
    log_scale: Union[bool, int, float] = None,
    color: str = "blue",
    fill: bool = True,
    data2: Union[pd.DataFrame, np.ndarray] = None,
    warn_singular: bool = True,
    **kwargs,
):
    """Plot a density plot of ISOCoordinates.

    Parameters
    ----------
    x : str, optional
        Column name for x variable, by default None
    y : str, optional
        Column name for y variable, by default None
    gridsize : int, optional
        Grid size for the plot, by default 200
    kernel : str, optional
        Kernel for density estimation, by default None
    cut : Union[float, int], optional
        Cutoff for the kernel density estimation, by default 3
    clip : tuple[int], optional
        Clip limits for the density estimation, by default None
    legend : bool, optional
        Whether to include a legend, by default True
    cumulative : bool, optional
        Whether to plot the cumulative density, by default False
    cbar : bool, optional
        Whether to include a colorbar, by default False
    cbar_ax : matplotlib.axes.Axes, optional
        Existing axes object to use for the colorbar, by default None
    cbar_kws : dict, optional
        Keyword arguments for the colorbar, by default None
    ax : matplotlib.axes.Axes, optional
        Existing axes object to use for the plot, by default None
    weights : str, optional
        Column name for weights, by default None
    hue : str, optional
        Column name for hue, by default None
    palette : Union[str, list, dict, matplotlib.colors.Colormap], optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object implies numeric mapping.
        by default colorblind
    hue_order : list[str], optional
        Order to use for the hue variable, by default None
    hue_norm : Union[tuple, matplotlib.colors.Normalize], optional
        Normalization to use for the hue variable, by default None
    multiple : str, optional
        Whether to plot multiple densities on the same axes, by default 'layer'
    common_norm : bool, optional
        Whether to use the same normalization for all dens
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if bw_adjust is None:
        bw_adjust = default_bw_adjust

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
            data2=data2,
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
        data2=data2,
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


#
# def density(
#     data=None,
#     *,
#     ax=None,
#     title="Soundscape Density Plot",
#     x="ISOPleasant",
#     y="ISOEventful",
#     prim_labels=False,
#     diagonal_lines=False,
#     xlim=(-1, 1),
#     ylim=(-1, 1),
#     incl_scatter=False,
#     incl_outline=False,
#     figsize=(5, 5),
#     palette="colorblind",
#     scatter_color="black",
#     outline_color="black",
#     fill_color="blue",
#     fill=True,
#     hue=None,
#     common_norm=False,
#     bw_adjust=default_bw_adjust,
#     alpha=0.95,
#     legend=False,
#     legend_loc="lower left",
#     s=10,
#     scatter_kwargs={},
#     **density_kwargs,
# ):
#     """Create a bivariate distribution plot of ISOCoordinates
#
#     This method works by creating a circumplex_grid, then overlaying a sns.kdeplot() using the ISOCoordinate data.
#     If a scatter is also included, it overlays a sns.scatterplot() using the given options underneath the density plot.
#
#     If using a hue grouping, it is recommended to only plot the 50th percentile contour so as to not create a cluttered
#     figure. This can be done with the options thresh = 0.5, levels = 2.
#
#     Parameters
#     ----------
#     *
#     density_type
#     ax : plt.Axes, optional
#         existing subplot axes, by default None
#     title : str, optional
#         by default "Soundscape Density Plot"
#     x : str, optional
#         column name for x variable, by default "ISOPleasant"
#     y : str, optional
#         column name for y variable, by default "ISOEventful"
#     prim_labels : bool, optional
#         whether to include ISOPleasant and ISOEventful axis labels, by default True
#     diagonal_lines : bool, optional
#         whether to include diagonal dimension axis labels (i.e. calm, etc.), by default False
#     incl_scatter : bool, optional
#         plot coordinate scatter underneath density plot, by default False
#     incl_outline : bool, optional
#         include a thicker outline around the density plot, by default False
#     figsize : tuple, optional
#         by default (5, 5)
#     palette : str, optional
#         Method for choosing the colors to use when mapping the hue semantic. String values are passed to seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object implies numeric mapping.
#         by default "colorblind"
#     scatter_color : str, optional
#         define a color for the scatter points. Does not work with a hue grouping variable, by default "black"
#     outline_color : str, optional
#         define a color for the add'l density outline, by default "black"
#     fill_color : str, optional
#         define a color for the density fill, does not work with a hue grouping variable, by default "blue"
#     fill : bool, optional
#         whether to fill the density plot, by default True
#     hue : vector or key in data, optional
#         Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case, by default None
#     common_norm : bool, optional
#         [description], by default False
#     bw_adjust : [type], optional
#         [description], by default default_bw_adjust
#     alpha : float, optional
#         [description], by default 0.95
#     legend : bool, optional
#         whether to include the hue labels legend, by default False
#     legend_loc : str, optional
#         relative location of the legend, by default "lower left"
#     s : int, optional
#         size of the scatter points, by default 10
#     scatter_kwargs : dict, optional
#         additional arguments for sns.scatterplot(), by default {}
#
#     Returns
#     -------
#     plt.Axes
#     """
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#
#     if incl_scatter:
#         d = sns.scatterplot(
#             data=data,
#             x=x,
#             y=y,
#             hue=hue,
#             s=s,
#             ax=ax,
#             legend=legend,
#             color=scatter_color,
#             palette=palette,
#             zorder=data_zorder,
#             **scatter_kwargs,
#         )
#
#     if incl_outline:
#         d = sns.kdeplot(
#             data=data,
#             x=x,
#             y=y,
#             fill=False,
#             ax=ax,
#             alpha=1,
#             color=outline_color,
#             palette=palette,
#             hue=hue,
#             common_norm=common_norm,
#             legend=legend,
#             zorder=data_zorder,
#             bw_adjust=bw_adjust,
#             **density_kwargs,
#         )
#
#     d = sns.kdeplot(
#         data=data,
#         x=x,
#         y=y,
#         fill=fill,
#         ax=ax,
#         alpha=alpha,
#         palette=palette,
#         color=fill_color,
#         hue=hue,
#         common_norm=common_norm,
#         legend=legend,
#         zorder=data_zorder,
#         bw_adjust=bw_adjust,
#         **density_kwargs,
#     )
#
#     _circumplex_grid(ax, prim_labels, diagonal_lines, xlim, ylim)
#     _set_circum_title(ax, prim_labels, title)
#     _deal_w_default_labels(ax, prim_labels)
#     if legend:
#         _move_legend(ax, legend_loc)
#     return d


def circumplex_jointplot_density(
    data,
    title="Soundscape Joint Plot",
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=False,
    diagonal_lines=False,
    xlim=(-1, 1),
    ylim=(-1, 1),
    palette="colorblind",
    incl_scatter=False,
    scatter_color="black",
    fill=True,
    bw_adjust=default_bw_adjust,
    alpha=0.95,
    legend=False,
    legend_loc="lower left",
    marginal_kind="kde",
    hue=None,
    joint_kwargs={},
    marginal_kwargs={"fill": True},
):
    """Create a jointplot with distribution or scatter in the center and distributions on the margins.

    This method works by calling sns.jointplot() and creating a circumplex grid in the joint position, then overlaying a density or circumplex_scatter plot. The options for both the joint and marginal plots can be passed through the sns.jointplot() separately to customise them separately. The marginal distribution plots can be either a density or histogram.

    Parameters
    ----------
    title : str, optional
        by default "Soundscape Joint Plot"
    x : str, optional
        column name for x variable, by default "ISOPleasant"
    y : str, optional
        column name for y variable, by default "ISOEventful"
    prim_labels : bool, optional
        whether to include ISOPleasant and ISOEventful axis labels in the joint plot, by default False
    diagonal_lines : bool, optional
        whether to include diagonal dimension axis labels in the joint plot, by default False
    palette : str, optional
        [description], by default "colorblind"
    incl_scatter : bool, optional
        plot coordinate scatter underneath density plot, by default False
    scatter_color : str, optional
        define a color for the scatter points, by default "black"
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
        Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case, by default None
    joint_kwargs : dict, optional
        Arguments to pass to density or scatter joint plot, by default {}
    marginal_kwargs : dict, optional
        Arguments to pass to marginal distribution plots, by default {"fill": True}

    Returns
    -------
    plt.Axes
    """
    g = sns.JointGrid()
    density(
        data,
        ax=g.ax_joint,
        title=None,
        x=x,
        y=y,
        prim_labels=prim_labels,
        diagonal_lines=diagonal_lines,
        xlim=xlim,
        ylim=ylim,
        incl_scatter=incl_scatter,
        palette=palette,
        scatter_color=scatter_color,
        fill=fill,
        hue=hue,
        bw_adjust=bw_adjust,
        alpha=alpha,
        legend=legend,
        **joint_kwargs,
    )
    if legend:
        _move_legend(g.ax_joint, legend_loc)

    if marginal_kind == "hist":
        sns.histplot(
            data=data,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            binrange=(-1, 1),
            legend=False,
            **marginal_kwargs,
        )
        sns.histplot(
            data=data,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            binrange=(-1, 1),
            legend=False,
            **marginal_kwargs,
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
            **marginal_kwargs,
        )
        sns.kdeplot(
            data=data,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kwargs,
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
    old_legend = ax.legend_
    handles = old_legend.legendHandles
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

    ax.grid(b=True, which="major", color="grey", alpha=0.5)
    ax.grid(
        b=True,
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
    title_pad = 30.0 if prim_labels is True else 6.0
    ax.set_title(title, pad=title_pad)
    return ax


def _deal_w_default_labels(ax, prim_labels):
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
        xy=(data[x_key][location], data[y_key][location],),
        xytext=(data[x_key][location] + x_adj, data[y_key][location] + y_adj,),
        ha=ha,
        va=va,
        arrowprops=arrowprops,
        annotation_clip=True,
        fontsize=fontsize,
        **text_kwargs,
    )
