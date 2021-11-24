#%%

from math import pi
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .parameters import PAQ_COLS

diag_lines_zorder = 1
diag_labels_zorder = 4
prim_lines_zorder = 2
data_zorder = 3
default_bw_adjust = 1.2
default_figsize = (8, 8)

#%%


def paq_radar(sf, ax=None, index=None, figsize=default_figsize):
    """Plots a radar (spider) plot showing the 8 PAQs in their appropriate relationships
    This plot is very useful for visualising the original PAQs, before they undergo the ISO transformation.
    Radar plots are not typically recommended for most purposes, but they are particularly appropriate here where
    the spatial relationship between the PAQs is actually meaningful.
    :param sf: SurveyFrame
    :param ax: matplotlib Axes if adding to a multiplot, default None
    :param index: column to convert to index for labelling, default None if index is already correctly set
    :param figsize:
    :return: matplotlib Axes containing the radar plot
    """
    if index:
        sf = sf.convert_column_to_index(col=index)
    data = sf[PAQ_COLS]
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(polar=True)
    # ---------- Part 1: create background
    # Number of variables
    categories = [
        "          pleasant",
        "    vibrant",
        "eventful",
        "chaotic    ",
        "annoying          ",
        "monotonous            ",
        "uneventful",
        "calm",
    ]
    N = len(categories)

    # What will be the angle of each axis in the plot (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(1, 5)

    # -------- Part 2: Add plots

    # Plot each individual = each line of the data
    fill_col = ["b", "r", "g"]
    for i in range(len(data.index)):
        # Ind1
        values = data.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=data.index[i])
        ax.fill(angles, values, fill_col[i], alpha=0.1)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    return ax


def circumplex_scatter(
    sf,
    ax=None,
    title="Soundscape Scatter Plot",
    group=None,
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=True,
    diagonal_lines=False,
    palette=None,
    legend=False,
    legend_loc="lower left",
    s=100,
    figsize=default_figsize,
    **scatter_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if palette is None:
        n_colors = len(sf[group].unique()) if group else len(sf)
        palette = sns.color_palette("husl", as_cmap=True)

    if group is None:
        group = sf.index.values

    sns.scatterplot(
        data=sf,
        x=x,
        y=y,
        hue=group,
        s=s,
        ax=ax,
        legend=legend,
        palette=palette,
        zorder=data_zorder,
        **scatter_kwargs,
    )
    ax = _deal_w_default_labels(ax, prim_labels)
    _circumplex_grid(ax, prim_labels, diagonal_lines)
    _set_circum_title(ax, prim_labels, title)
    if legend:
        _move_legend(ax, legend_loc)
    return ax


def circumplex_density(
    sf,
    ax=None,
    title="Soundscape Density Plot",
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=True,
    diagonal_lines=False,
    palette="Blues",
    fill=True,
    group=None,
    bw_adjust=default_bw_adjust,
    alpha=0.95,
    legend=False,
    legend_loc="lower left",
    figsize=default_figsize,
    **density_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.kdeplot(
        data=sf,
        x=x,
        y=y,
        fill=fill,
        ax=ax,
        alpha=alpha,
        palette=palette,
        legend=legend,
        zorder=data_zorder,
        bw_adjust=bw_adjust,
        hue=group,
        **density_kwargs,
    )
    _circumplex_grid(ax, prim_labels, diagonal_lines)
    _set_circum_title(ax, prim_labels, title)
    _deal_w_default_labels(ax, prim_labels)
    if legend:
        _move_legend(ax, legend_loc)
    return ax


def circumplex_jointplot(
    sf,
    title="Soundscape Joint Plot",
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=False,
    diagonal_lines=False,
    palette="Blues",
    fill=True,
    bw_adjust=default_bw_adjust,
    alpha=0.95,
    legend=False,
    legend_loc="lower left",
    s=100,
    marginal_kind="density",
    joint_kind="density",
    group=None,
    joint_kwargs={},
    marginal_kwargs={"fill": True},
):
    g = sns.JointGrid()
    if joint_kind == "density":
        circumplex_density(
            sf,
            g.ax_joint,
            title=None,
            x=x,
            y=y,
            prim_labels=prim_labels,
            diagonal_lines=diagonal_lines,
            palette=palette,
            group=group,
            fill=fill,
            bw_adjust=bw_adjust,
            alpha=alpha,
            legend=legend,
            **joint_kwargs,
        )
    elif joint_kind == "scatter":
        circumplex_scatter(
            sf,
            g.ax_joint,
            title=None,
            group=group,
            x=x,
            y=y,
            prim_labels=prim_labels,
            diagonal_lines=diagonal_lines,
            palette=palette,
            legend=legend,
            legend_loc=legend_loc,
            s=s,
            **joint_kwargs,
        )
    else:
        raise AttributeError("joint_kind not recognised")
    if legend:
        _move_legend(g.ax_joint, legend_loc)

    if marginal_kind == "hist":
        sns.histplot(
            data=sf,
            x=x,
            hue=group,
            ax=g.ax_marg_x,
            binrange=(-1, 1),
            legend=False,
            **marginal_kwargs,
        )
        sns.histplot(
            data=sf,
            y=y,
            hue=group,
            ax=g.ax_marg_y,
            binrange=(-1, 1),
            legend=False,
            **marginal_kwargs,
        )
    elif marginal_kind == "density":
        sns.kdeplot(
            data=sf,
            x=x,
            hue=group,
            ax=g.ax_marg_x,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kwargs,
        )
        sns.kdeplot(
            data=sf,
            y=y,
            hue=group,
            ax=g.ax_marg_y,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kwargs,
        )
    g.ax_marg_x.set_title(title, pad=6.0)

    return g


def single_lmplot(
    sf,
    x,
    y,
    groups=None,
    group_order=None,
    ylim=None,
    xlim=None,
    order=1,
    reg=True,
    **kwargs,
):
    if y in ["ISOPleasant", "ISOEventful"] and ylim is None:
        ylim = (-1, 1)
    if reg:
        df = sf[[y, x]].dropna()
        r, p = pearsonr(df[y], df[x])
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
            x=x, y=y, data=sf, height=8, aspect=1, order=order, fit_reg=reg, **kwargs
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
            fit_reg=reg,
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
    col_wrap=4,
    scatter_kws=None,
    line_kws=None,
    facet_kws={"sharex": False},
    **kwargs,
):
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

    if fit_reg:
        for location, ax in zip(group_order, grid.axes.ravel()):
            df = sf.loc[sf[groups] == location, [y, x]].dropna()
            r, p = pearsonr(df[y], df[x])
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


def _move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


def _circumplex_grid(ax, prim_labels=True, diagonal_lines=False):

    # Setting up the grids
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    line_weights = 1.5
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

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
    prim_ax_font = {
        "fontstyle": "italic",
        "fontsize": "medium",
        "fontweight": "bold",
        "c": "grey",
        "alpha": 1,
    }
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # Add lines and labels for circumplex model
    ## Primary Axes
    ax.plot(  # Horizontal line
        [-1, 1],
        [0, 0],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
        zorder=prim_lines_zorder,
    )
    ax.plot(  # vertical line
        [0, 0],
        [1, -1],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
        zorder=prim_lines_zorder,
    )

    if prim_labels is True:
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

    ax.plot(  # upward diagonal
        [-1, 1],
        [-1, 1],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=diag_lines_zorder,
    )
    ax.plot(  # downward diagonal
        [-1, 1],
        [1, -1],
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
