"""
Utility functions for creating various types of circumplex plots.
"""

from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

from .backends import SeabornBackend
from .circumplex_plot import CircumplexPlot, CircumplexPlotParams
from .plotting_utils import (
    Backend,
    DEFAULT_FIGSIZE,
    DEFAULT_XLIM,
    DEFAULT_YLIM,
    ExtraParams,
    PlotType,
)
from .stylers import StyleOptions


def scatter_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: Optional[str] = None,
    title: str = "Soundscape Scatter Plot",
    xlim: Tuple[float, float] = DEFAULT_XLIM,
    ylim: Tuple[float, float] = DEFAULT_YLIM,
    palette: str = "colorblind",
    diagonal_lines: bool = False,
    show_labels: bool = True,
    legend=True,
    legend_location: str = "best",
    backend: Backend = Backend.SEABORN,
    apply_styling: bool = True,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    ax: Optional[plt.Axes] = None,
    extra_params: ExtraParams = {},
    **kwargs: Any,
) -> plt.Axes | go.Figure:
    """Create a scatter plot using the CircumplexPlot class."""
    params = CircumplexPlotParams(
        x=x,
        y=y,
        hue=hue,
        title=title,
        xlim=xlim,
        ylim=ylim,
        palette=palette if hue else None,
        diagonal_lines=diagonal_lines,
        show_labels=show_labels,
        legend=legend,
        legend_location=legend_location,
        extra_params={**extra_params, **kwargs},
    )

    style_options = StyleOptions(figsize=figsize)

    plot = CircumplexPlot(data, params, backend, style_options)
    plot.scatter(apply_styling=apply_styling, ax=ax)

    if isinstance(plot._backend, SeabornBackend):
        return plot.get_axes()
    else:
        return plot.get_figure()


def density_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: Optional[str] = None,
    title: str = "Soundscape Density Plot",
    xlim: Tuple[float, float] = DEFAULT_XLIM,
    ylim: Tuple[float, float] = DEFAULT_YLIM,
    palette: str = "colorblind",
    fill: bool = True,
    incl_outline: bool = False,
    incl_scatter: bool = False,
    diagonal_lines: bool = False,
    show_labels: bool = True,
    legend=True,
    legend_location: str = "best",
    backend: Backend = Backend.SEABORN,
    apply_styling: bool = True,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    simple_density: bool = False,
    simple_density_thresh: float = 0.5,
    simple_density_levels: int = 2,
    simple_density_alpha: float = 0.5,
    ax: Optional[plt.Axes] = None,
    extra_params: ExtraParams = {},
    **kwargs: Any,
) -> plt.Axes | go.Figure:
    """
    Create a density plot using the CircumplexPlot class.

    Parameters remain the same as before for backward compatibility.
    """
    params = CircumplexPlotParams(
        x=x,
        y=y,
        hue=hue,
        title=title,
        xlim=xlim,
        ylim=ylim,
        palette=palette if hue else None,
        fill=fill,
        incl_outline=incl_outline,
        diagonal_lines=diagonal_lines,
        show_labels=show_labels,
        legend=legend,
        legend_location=legend_location,
        extra_params={**extra_params, **kwargs},
    )

    style_options = StyleOptions(
        figsize=figsize,
        simple_density=dict(
            thresh=simple_density_thresh,
            levels=simple_density_levels,
            alpha=simple_density_alpha,
        )
        if simple_density
        else None,
    )

    # Create plot and add layers
    plot = CircumplexPlot(data, params, backend, style_options)

    # Add layers in the correct order
    if simple_density:
        plot.simple_density(apply_styling=False, ax=ax)
    else:
        plot.density(apply_styling=False, ax=ax)

    if incl_scatter:
        # Use a lower alpha for the scatter when combined with density
        scatter_alpha = params.alpha * 0.5 if params.alpha is not None else 0.4
        plot.scatter(alpha=scatter_alpha, apply_styling=False, ax=ax)

    if apply_styling:
        plot._update_plot(ax)

    if isinstance(plot._backend, SeabornBackend):
        return plot.get_axes()
    else:
        return plot.get_figure()


def create_circumplex_subplots(
    data_list: List[pd.DataFrame],
    plot_type: PlotType | str = PlotType.DENSITY,
    incl_scatter: bool = True,
    subtitles: Optional[List[str]] = None,
    title: str = "Circumplex Subplots",
    nrows: int = None,
    ncols: int = None,
    figsize: Tuple[int, int] = (10, 10),
    **kwargs: Any,
) -> plt.Figure:
    """Create a figure with multiple circumplex plot subplots."""
    if isinstance(plot_type, str):
        plot_type = PlotType[plot_type.upper()]

    if nrows is None and ncols is None:
        nrows = 2
        ncols = len(data_list) // nrows + (len(data_list) % nrows > 0)
    elif nrows is None:
        nrows = len(data_list) // ncols + (len(data_list) % ncols > 0)
    elif ncols is None:
        ncols = len(data_list) // nrows + (len(data_list) % nrows > 0)

    if subtitles is None:
        subtitles = [f"({i + 1})" for i in range(len(data_list))]
    elif len(subtitles) != len(data_list):
        raise ValueError("Number of subtitles must match number of dataframes")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    color = kwargs.get("color", sns.color_palette("colorblind", 1)[0])

    # Create plots using the new layer system
    for data, ax, subtitle in zip(data_list, axes, subtitles):
        params = CircumplexPlotParams(title=subtitle, color=color, **kwargs)
        plot = CircumplexPlot(data, params=params)

        if plot_type in [PlotType.DENSITY, PlotType.SCATTER_DENSITY]:
            plot.density(apply_styling=False)
            if incl_scatter:
                plot.scatter(alpha=0.5, apply_styling=False)
        elif plot_type == PlotType.SIMPLE_DENSITY:
            plot.simple_density(apply_styling=False)
            if incl_scatter:
                plot.scatter(alpha=0.5, apply_styling=False)
        elif plot_type == PlotType.SCATTER:
            plot.scatter(apply_styling=False)

        plot._update_plot(ax)

    plt.suptitle(title)
    plt.tight_layout()
    return fig
