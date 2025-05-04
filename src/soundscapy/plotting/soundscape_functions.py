"""
Function-based API for creating soundscape plots with Seaborn Objects.

This module provides high-level functions for creating various types of
soundscape plots using seaborn.objects API and custom Mark/Stat components.
These functions offer a more direct, functional approach compared to the
CircumplexPlot builder class.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from soundscapy.plotting.marks import (
    SoundscapeCircumplex,
    SoundscapeQuadrantLabels,
)
from soundscapy.plotting.plotting_utils import DEFAULT_XLIM, DEFAULT_YLIM
from soundscapy.plotting.stats import SoundscapeCoordinates

# Path to soundscapy.mplstyle
STYLE_PATH = Path(__file__).parent / "soundscapy.mplstyle"


def use_soundscapy_style():
    """
    Apply the soundscapy matplotlib style.

    This function activates the built-in style sheet for consistent
    soundscape plot appearance.
    """
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        # Fall back to built-in style if custom style file isn't found
        plt.style.use("seaborn-v0_8-colorblind")


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
    point_size: float = 30,
    alpha: float = 0.7,
    marker: str = "o",
    ax: plt.Axes | None = None,
    as_objects: bool = False,
    **kwargs: Any,
) -> so.Plot | plt.Axes:
    """
    Create a scatter plot using the soundscape circumplex model.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x : str, default="ISOPleasant"
        Column name for x-axis
    y : str, default="ISOEventful"
        Column name for y-axis
    hue : str, optional
        Column name for color grouping
    title : str, default="Soundscape Scatter Plot"
        Title for the plot
    xlim : tuple, default=(-1, 1)
        X-axis limits
    ylim : tuple, default=(-1, 1)
        Y-axis limits
    palette : str, default="colorblind"
        Color palette to use
    diagonal_lines : bool, default=False
        Whether to show diagonal lines and quadrant labels
    show_labels : bool, default=True
        Whether to show axis labels
    point_size : float, default=30
        Size of scatter points
    alpha : float, default=0.7
        Opacity of scatter points
    marker : str, default="o"
        Marker style for scatter points
    ax : plt.Axes, optional
        Axes to plot on (for matplotlib compatibility)
    as_objects : bool, default=False
        If True, return seaborn.objects.Plot; if False, return Matplotlib axes
    **kwargs : Any
        Additional keyword arguments for scatter plot

    Returns
    -------
    so.Plot | plt.Axes
        The completed plot object or axes

    """
    use_soundscapy_style()

    # Create base plot
    plot = so.Plot(data, x=x, y=y)

    # Add scatter points
    plot = plot.add(
        so.Dots(pointsize=point_size, alpha=alpha, marker=marker), color=hue
    )

    # Apply color palette if needed
    if hue:
        plot = plot.scale(color=so.Nominal(palette))

    # Add circumplex grid
    plot = plot.add(SoundscapeCircumplex(xlim=xlim, ylim=ylim))

    # Add quadrant labels if requested
    if diagonal_lines:
        plot = plot.add(SoundscapeQuadrantLabels(xlim=xlim, ylim=ylim))

    # Add title and labels
    plot = plot.label(title=title)

    # Set layout
    plot = plot.layout(size=(6, 6))

    # Hide labels if requested
    if not show_labels:
        plot = plot.label(x=None, y=None)

    if as_objects:
        return plot

    if ax is not None:
        # If an axes is provided, clear and draw on it
        ax.clear()
        plot.plot(ax)
        return ax

    # Create a new figure and axes, draw on it, and return the axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plot.plot(ax)
    return ax


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
    bw_adjust: float = 1.2,
    levels: int = 8,
    simple_density: bool = False,
    incl_scatter: bool = False,
    scatter_size: float = 15,
    scatter_alpha: float = 0.5,
    diagonal_lines: bool = False,
    show_labels: bool = True,
    ax: plt.Axes | None = None,
    as_objects: bool = False,
    **kwargs: Any,
) -> so.Plot | plt.Axes:
    """
    Create a density plot using the soundscape circumplex model.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x : str, default="ISOPleasant"
        Column name for x-axis
    y : str, default="ISOEventful"
        Column name for y-axis
    hue : str, optional
        Column name for color grouping
    title : str, default="Soundscape Density Plot"
        Title for the plot
    xlim : tuple, default=(-1, 1)
        X-axis limits
    ylim : tuple, default=(-1, 1)
        Y-axis limits
    palette : str, default="colorblind"
        Color palette to use
    fill : bool, default=True
        Whether to fill the contours
    alpha : float, default=0.5
        Opacity of the fill
    bw_adjust : float, default=1.2
        Bandwidth adjustment factor
    levels : int, default=8
        Number of contour levels
    simple_density : bool, default=False
        If True, use simplified density with fewer levels and an outline
    incl_scatter : bool, default=False
        Whether to include scatter points
    scatter_size : float, default=15
        Size of scatter points (if included)
    scatter_alpha : float, default=0.5
        Opacity of scatter points (if included)
    diagonal_lines : bool, default=False
        Whether to show diagonal lines and quadrant labels
    show_labels : bool, default=True
        Whether to show axis labels
    ax : plt.Axes, optional
        Axes to plot on (for matplotlib compatibility)
    as_objects : bool, default=False
        If True, return seaborn.objects.Plot; if False, return Matplotlib axes
    **kwargs : Any
        Additional keyword arguments for density plot

    Returns
    -------
    so.Plot | plt.Axes
        The completed plot object or axes

    """
    use_soundscapy_style()

    # Create base plot
    plot = so.Plot(data, x=x, y=y)

    # Add density layer
    if simple_density:
        # Simple density with fewer levels and outline
        plot = plot.add(
            so.Area(fill=fill, alpha=alpha),
            so.KDE(bw_adjust=bw_adjust, levels=2),
            color=hue,
        )
        # Add outline
        plot = plot.add(
            so.Line(alpha=1.0), so.KDE(bw_adjust=bw_adjust, levels=2), color=hue
        )
    else:
        # Regular density
        plot = plot.add(
            so.Area(fill=fill, alpha=alpha),
            so.KDE(bw_adjust=bw_adjust, levels=levels),
            color=hue,
        )

    # Apply color palette if needed
    if hue:
        plot = plot.scale(color=so.Nominal(palette))

    # Add scatter if requested
    if incl_scatter:
        plot = plot.add(so.Dots(pointsize=scatter_size, alpha=scatter_alpha), color=hue)
        # Apply color palette again for scatter
        if hue:
            plot = plot.scale(color=so.Nominal(palette))

    # Add circumplex grid
    plot = plot.add(SoundscapeCircumplex(xlim=xlim, ylim=ylim))

    # Add quadrant labels if requested
    if diagonal_lines:
        plot = plot.add(SoundscapeQuadrantLabels(xlim=xlim, ylim=ylim))

    # Add title and labels
    plot = plot.label(title=title)

    # Set layout
    plot = plot.layout(size=(6, 6))

    # Hide labels if requested
    if not show_labels:
        plot = plot.label(x=None, y=None)

    if as_objects:
        return plot

    if ax is not None:
        # If an axes is provided, clear and draw on it
        ax.clear()
        plot.plot(ax)
        return ax

    # Create a new figure and axes, draw on it, and return the axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plot.plot(ax)
    return ax


def joint_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: str | None = None,
    title: str = "Soundscape Joint Plot",
    kind: str = "scatter",
    xlim: tuple[float, float] = DEFAULT_XLIM,
    ylim: tuple[float, float] = DEFAULT_YLIM,
    palette: str = "colorblind",
    diagonal_lines: bool = False,
    **kwargs: Any,
) -> sns.JointGrid:
    """
    Create a joint plot with marginals using traditional seaborn.

    This function falls back to traditional seaborn because
    seaborn.objects does not yet fully support joint plots with marginals.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x : str, default="ISOPleasant"
        Column name for x-axis
    y : str, default="ISOEventful"
        Column name for y-axis
    hue : str, optional
        Column name for color grouping
    title : str, default="Soundscape Joint Plot"
        Title for the plot
    kind : str, default="scatter"
        Type of plot: "scatter", "kde", or "hex"
    xlim : tuple, default=(-1, 1)
        X-axis limits
    ylim : tuple, default=(-1, 1)
        Y-axis limits
    palette : str, default="colorblind"
        Color palette to use
    diagonal_lines : bool, default=False
        Whether to show diagonal lines
    **kwargs : Any
        Additional parameters for sns.jointplot

    Returns
    -------
    sns.JointGrid
        The joint plot grid

    """
    use_soundscapy_style()

    # Create joint plot using traditional seaborn
    g = sns.jointplot(
        data=data, x=x, y=y, hue=hue, kind=kind, palette=palette, **kwargs
    )

    # Apply limits
    g.ax_joint.set_xlim(xlim)
    g.ax_joint.set_ylim(ylim)

    # Add zero lines
    g.ax_joint.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
    g.ax_joint.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)

    # Add grid
    g.ax_joint.grid(True, which="major", color="grey", alpha=0.5)

    # Add title
    g.fig.suptitle(title, y=1.05)

    # Add diagonal lines if requested
    if diagonal_lines:
        g.ax_joint.plot(
            [xlim[0], xlim[1]],
            [ylim[0], ylim[1]],
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            linewidth=1.5,
        )
        g.ax_joint.plot(
            [xlim[0], xlim[1]],
            [ylim[1], ylim[0]],
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            linewidth=1.5,
        )

    return g


def create_circumplex_subplots(
    data_list: Sequence[pd.DataFrame],
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: str | None = None,
    subtitles: Sequence[str] | None = None,
    title: str = "Circumplex Subplots",
    plot_type: str = "density",
    incl_scatter: bool = False,
    cols: int = 2,
    **kwargs: Any,
) -> plt.Figure:
    """
    Create a figure with multiple circumplex plots.

    Parameters
    ----------
    data_list : Sequence[pd.DataFrame]
        List of data sources to plot
    x : str, default="ISOPleasant"
        Column name for x-axis
    y : str, default="ISOEventful"
        Column name for y-axis
    hue : str, optional
        Column name for color grouping
    subtitles : Sequence[str], optional
        Titles for individual subplots
    title : str, default="Circumplex Subplots"
        Main title for the figure
    plot_type : str, default="density"
        Type of plot: "scatter", "density", or "simple_density"
    incl_scatter : bool, default=False
        Whether to include scatter points on density plots
    cols : int, default=2
        Number of columns for subplots
    **kwargs : Any
        Additional arguments for plot functions

    Returns
    -------
    plt.Figure
        The matplotlib Figure with subplots

    """
    use_soundscapy_style()

    # Generate subplot titles if not provided
    if subtitles is None:
        subtitles = [f"Plot {i + 1}" for i in range(len(data_list))]

    # Remove any layout parameters that don't belong in plot functions
    plotting_kwargs = kwargs.copy()
    if "nrows" in plotting_kwargs:
        plotting_kwargs.pop("nrows")
    if "ncols" in plotting_kwargs:
        plotting_kwargs.pop("ncols")

    # Calculate rows needed
    nrows = (len(data_list) - 1) // cols + 1

    # Create figure and axes
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 6, nrows * 6), squeeze=False)
    axes = axes.flatten()

    # Create individual plots
    for i, (data, subtitle) in enumerate(zip(data_list, subtitles, strict=False)):
        if i < len(axes):
            # Create plot for this axis
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
            else:  # "density" or default
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
    fig.tight_layout()

    return fig


def add_calculated_coords(
    data: pd.DataFrame,
    paq_cols=("PAQ1", "PAQ2", "PAQ3", "PAQ4", "PAQ5", "PAQ6", "PAQ7", "PAQ8"),
    angles=(0, 45, 90, 135, 180, 225, 270, 315),
    val_range=(5, 1),
    output_cols=("ISOPleasant", "ISOEventful"),
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Calculate and add ISO coordinates to a dataframe.

    This is a convenience function to use the SoundscapeCoordinates
    stat outside of a plotting context.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with PAQ columns
    paq_cols : tuple, default=("PAQ1", "PAQ2", "PAQ3", "PAQ4", "PAQ5", "PAQ6", "PAQ7", "PAQ8")
        Column names for PAQ data
    angles : tuple, default=(0, 45, 90, 135, 180, 225, 270, 315)
        Angles for each PAQ in degrees
    val_range : tuple, default=(5, 1)
        (max, min) range of original PAQ responses
    output_cols : tuple, default=("ISOPleasant", "ISOEventful")
        Column names for output coordinates
    overwrite : bool, default=False
        Whether to overwrite existing coordinate columns

    Returns
    -------
    pd.DataFrame
        Data with added ISO coordinate columns

    Raises
    ------
    ValueError
        If coordinate columns already exist and overwrite=False

    """
    # Check if columns already exist
    for col in output_cols:
        if col in data.columns and not overwrite:
            raise ValueError(
                f"{col} already exists in dataframe. Use overwrite=True to replace."
            )

    # Create the stat and apply it to the data
    stat = SoundscapeCoordinates(
        paq_cols=paq_cols,
        angles=angles,
        val_range=val_range,
        output_cols=output_cols,
    )

    # Apply the stat
    result = stat._apply(data)

    return result
