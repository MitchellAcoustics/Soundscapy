"""
High level functions for creating various types of circumplex plots.

These functions provide a high-level interface for creating common plot types
using the CircumplexPlot class with the Seaborn Objects API.
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from soundscapy.plotting.circumplex_plot import CircumplexPlot
from soundscapy.plotting.plotting_utils import (
    DEFAULT_XLIM,
    DEFAULT_YLIM,
)


def scatter_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: str | None = None,
    title: str = "Soundscape Scatter Plot",
    xlim: tuple[float, float] = DEFAULT_XLIM,
    ylim: tuple[float, float] = DEFAULT_YLIM,
    palette: str = "colorblind",
    diagonal_lines: bool = False,
    show_labels: bool = True,
    pointsize: int = 30,
    alpha: float = 0.7,
    ax: plt.Axes | None = None,
    as_objects: bool = False,
    **kwargs: Any,
) -> so.Plot | plt.Axes:
    """
    Create a scatter plot using the circumplex model.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x, y : str
        Column names for coordinates
    hue : str, optional
        Column name for color grouping
    title : str
        Title for the plot
    xlim, ylim : tuple
        Axis limits
    palette : str or list or dict
        Color palette to use for hue
    diagonal_lines : bool
        Whether to show diagonal lines and quadrant labels
    show_labels : bool
        Whether to show axis labels
    pointsize : int
        Size of scatter points
    alpha : float
        Opacity of scatter points
    ax : plt.Axes, optional
        Axes to plot on (for matplotlib compatibility)
    as_objects : bool
        If True, return Seaborn Objects plot; if False, return Matplotlib axes
    **kwargs
        Additional keyword arguments for scatter plot

    Returns
    -------
    so.Plot | plt.Axes
        The completed plot object or axes

    """
    plot = (
        CircumplexPlot(data, x, y, hue, xlim, ylim, palette)
        .add_scatter(pointsize=pointsize, alpha=alpha, **kwargs)
        .add_grid(diagonal_lines=diagonal_lines, show_labels=show_labels)
        .add_title(title)
    )

    if as_objects:
        return plot.build(as_objects=True)
    if ax is not None:
        # If an axes is provided, draw directly on it
        plot.build(as_objects=True)
        # Clear previous contents
        ax.clear()
        # Use the ax limits and title from our plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # Draw points and style - only use palette if hue is provided
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=palette if hue else None,
            s=pointsize,
            alpha=alpha,
            ax=ax,
            **kwargs,
        )
        # Add grid lines
        ax.grid(True, which="major", color="grey", alpha=0.5)
        ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
        ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)

        # Add diagonal lines if requested
        if diagonal_lines:
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[0], ylim[1]],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                linewidth=1.5,
            )
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[1], ylim[0]],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                linewidth=1.5,
            )

        return ax
    return plot.get_axes()


def density_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: str | None = None,
    title: str = "Soundscape Density Plot",
    xlim: tuple[float, float] = DEFAULT_XLIM,
    ylim: tuple[float, float] = DEFAULT_YLIM,
    palette: str = "colorblind",
    fill: bool = True,
    alpha: float = 0.5,
    levels: int = 8,
    bw_adjust: float = 1.2,
    simple_density: bool = False,
    incl_scatter: bool = False,
    scatter_size: int = 15,
    scatter_alpha: float = 0.5,
    diagonal_lines: bool = False,
    show_labels: bool = True,
    ax: plt.Axes | None = None,
    as_objects: bool = False,
    **kwargs: Any,
) -> so.Plot | plt.Axes:
    """
    Create a density plot using the circumplex model.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x, y : str
        Column names for coordinates
    hue : str, optional
        Column name for color grouping
    title : str
        Title for the plot
    xlim, ylim : tuple
        Axis limits
    palette : str or list or dict
        Color palette to use for hue
    fill : bool
        Whether to fill the contours
    alpha : float
        Opacity of the fill
    levels : int
        Number of contour levels
    bw_adjust : float
        Bandwidth adjustment factor
    simple_density : bool
        If True, use simplified density with fewer levels and an outline
    incl_scatter : bool
        Whether to include scatter points with the density
    scatter_size : int
        Size of scatter points (if included)
    scatter_alpha : float
        Opacity of scatter points (if included)
    diagonal_lines : bool
        Whether to show diagonal lines and quadrant labels
    show_labels : bool
        Whether to show axis labels
    ax : plt.Axes, optional
        Axes to plot on (for matplotlib compatibility)
    as_objects : bool
        If True, return Seaborn Objects plot; if False, return Matplotlib axes
    **kwargs
        Additional keyword arguments for density plot

    Returns
    -------
    so.Plot | plt.Axes
        The completed plot object or axes

    """
    cp = CircumplexPlot(data, x, y, hue, xlim, ylim, palette)

    # Add density layer
    cp.add_density(
        alpha=alpha,
        fill=fill,
        levels=levels,
        bw_adjust=bw_adjust,
        simple=simple_density,
    )

    # Add scatter if requested
    if incl_scatter:
        cp.add_scatter(pointsize=scatter_size, alpha=scatter_alpha)

    # Complete the plot
    cp.add_grid(diagonal_lines=diagonal_lines, show_labels=show_labels)
    cp.add_title(title)

    if as_objects:
        return cp.build(as_objects=True)
    if ax is not None:
        # If an axes is provided, draw directly on it
        # Clear previous contents
        ax.clear()
        # Use the ax limits and title
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        # Draw the KDE
        if simple_density:
            # Simple density with fewer levels
            sns.kdeplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                fill=fill,
                alpha=alpha,
                levels=2,
                bw_adjust=bw_adjust,
                ax=ax,
                **kwargs,
            )
            # Add outline
            sns.kdeplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                fill=False,
                alpha=1.0,
                levels=2,
                bw_adjust=bw_adjust,
                ax=ax,
            )
        else:
            # Regular density
            sns.kdeplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                fill=fill,
                alpha=alpha,
                levels=levels,
                bw_adjust=bw_adjust,
                ax=ax,
                **kwargs,
            )

        # Add scatter if requested
        if incl_scatter:
            sns.scatterplot(
                data=data, x=x, y=y, hue=hue, s=scatter_size, alpha=scatter_alpha, ax=ax
            )

        # Add grid lines
        ax.grid(True, which="major", color="grey", alpha=0.5)
        ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
        ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)

        # Add diagonal lines if requested
        if diagonal_lines:
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[0], ylim[1]],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                linewidth=1.5,
            )
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[1], ylim[0]],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                linewidth=1.5,
            )

        return ax
    return cp.get_axes()


def joint_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: str | None = None,
    title: str = "Soundscape Joint Plot",
    plot_type: str = "scatter",
    **kwargs: Any,
) -> sns.JointGrid:
    """
    Create a joint plot with marginals using matplotlib.

    This function falls back to matplotlib/seaborn because
    Seaborn Objects does not yet fully support joint plots with marginals.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x, y : str
        Column names for coordinates
    hue : str, optional
        Column name for color grouping
    title : str
        Title for the plot
    plot_type : str
        Type of plot: "scatter", "density", or "simple_density"
    **kwargs
        Additional parameters for sns.jointplot

    Returns
    -------
    sns.JointGrid
        The joint plot grid

    """
    # Fall back to traditional seaborn for jointplot
    kind = "scatter" if plot_type == "scatter" else "kde"

    g = sns.jointplot(data=data, x=x, y=y, hue=hue, kind=kind, **kwargs)

    # Add grid elements to the central plot
    ax = g.ax_joint
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))

    # Add zero lines
    ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
    ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)

    # Add grid
    ax.grid(True, which="major", color="grey", alpha=0.5)

    # Add title
    g.fig.suptitle(title, y=1.05)

    # Add scatter if requested
    if plot_type in ["density", "simple_density"] and kwargs.get("incl_scatter", False):
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            s=kwargs.get("scatter_size", 15),
            alpha=kwargs.get("scatter_alpha", 0.5),
        )

    return g


def create_circumplex_subplots(
    data_list: list[pd.DataFrame],
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: str | None = None,
    subtitles: list[str] | None = None,
    title: str = "Circumplex Subplots",
    plot_type: str = "density",
    incl_scatter: bool = False,
    cols: int = 2,
    as_objects: bool = False,
    **kwargs: Any,
) -> so.Plot | plt.Figure:
    """
    Create a figure with multiple circumplex plots.

    Parameters
    ----------
    data_list : list of DataFrames
        List of data sources to plot
    x, y : str
        Column names for coordinates
    hue : str, optional
        Column name for color grouping
    subtitles : list of str, optional
        Titles for individual subplots
    title : str
        Main title for the plot
    plot_type : str
        Type of plot: "scatter", "density", or "simple_density"
    incl_scatter : bool
        Whether to include scatter points on density plots
    cols : int
        Number of columns for subplots
    as_objects : bool
        If True, return Seaborn Objects plot; if False, return Matplotlib figure
    **kwargs
        Additional arguments for plot functions

    Returns
    -------
    so.Plot | plt.Figure
        The plot object or figure

    """
    # Generate subplot titles if not provided
    if subtitles is None:
        subtitles = [f"Plot {i + 1}" for i in range(len(data_list))]

    # Remove any layout parameters that don't belong in plot functions
    plotting_kwargs = kwargs.copy()
    if "nrows" in plotting_kwargs:
        plotting_kwargs.pop("nrows")
    if "ncols" in plotting_kwargs:
        plotting_kwargs.pop("ncols")

    # For the refactored version, we'll create a matplotlib figure directly
    # instead of using the faceting in Seaborn Objects
    nrows = (len(data_list) - 1) // cols + 1

    # Create a new figure
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 6, nrows * 6), squeeze=False)
    axes = axes.flatten()

    # Create individual plots
    for i, (data, subtitle) in enumerate(zip(data_list, subtitles, strict=False)):
        if i < len(axes):
            # Create a plot for this axis
            if plot_type == "scatter":
                scatter_plot(
                    data,
                    x=x,
                    y=y,
                    hue=hue,
                    title=subtitle,
                    ax=axes[i],
                    **plotting_kwargs,
                )
            elif plot_type == "simple_density":
                density_plot(
                    data,
                    x=x,
                    y=y,
                    hue=hue,
                    title=subtitle,
                    simple_density=True,
                    incl_scatter=incl_scatter,
                    ax=axes[i],
                    **plotting_kwargs,
                )
            else:
                density_plot(
                    data,
                    x=x,
                    y=y,
                    hue=hue,
                    title=subtitle,
                    incl_scatter=incl_scatter,
                    ax=axes[i],
                    **plotting_kwargs,
                )

    # Hide any unused axes
    for i in range(len(data_list), len(axes)):
        axes[i].set_visible(False)

    # Add a title to the figure
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # We'll return the figure directly for legacy compatibility
    return fig
