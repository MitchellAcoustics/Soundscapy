"""
Utility functions for creating various types of circumplex plots.
"""

from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

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
    """
    Create a scatter plot using the CircumplexPlot class.

    Parameters
    ----------
        data (pd.DataFrame): The data to plot.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        hue (Optional[str]): Column name for color-coding data points.
        title (str): Title of the plot.
        xlim (Tuple[float, float]): x-axis limits.
        ylim (Tuple[float, float]): y-axis limits.
        palette (str): Color palette to use.
        diagonal_lines (bool): Whether to draw diagonal lines.
        show_labels (bool): Whether to show axis labels.
        legend (bool): Whether to show the legend.
        legend_location (str): Location of the legend.
        backend (Backend): The plotting backend to use.
        apply_styling (bool): Whether to apply circumplex-specific styling.
        figsize (Tuple[int, int]): Size of the figure.
        ax (Optional[plt.Axes]): A matplotlib Axes object to plot on.
        extra_params (ExtraParams): Additional parameters for backend-specific functions.
        **kwargs: Additional keyword arguments to pass to the backend.

    Returns
    -------
        plt.Axes | go.Figure: The resulting plot object.

    """
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
    incl_scatter: bool = True,
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

    Parameters
    ----------
        data (pd.DataFrame): The data to plot.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        hue (Optional[str]): Column name for color-coding data points.
        title (str): Title of the plot.
        xlim (Tuple[float, float]): x-axis limits.
        ylim (Tuple[float, float]): y-axis limits.
        palette (str): Color palette to use.
        fill (bool): Whether to fill the density contours.
        incl_outline (bool): Whether to include an outline for the density contours.
        diagonal_lines (bool): Whether to draw diagonal lines.
        show_labels (bool): Whether to show axis labels.
        legend (bool): Whether to show the legend.
        legend_location (str): Location of the legend.
        backend (Backend): The plotting backend to use.
        apply_styling (bool): Whether to apply circumplex-specific styling.
        figsize (Tuple[int, int]): Size of the figure.
        simple_density (bool): Whether to use simple density plot (Seaborn only).
        simple_density_thresh (float): Threshold for simple density plot.
        simple_density_levels (int): Number of levels for simple density plot.
        simple_density_alpha (float): Alpha value for simple density plot.
        ax (Optional[plt.Axes]): A matplotlib Axes object to plot on.
        extra_params (ExtraParams): Additional parameters for backend-specific functions.
        **kwargs: Additional keyword arguments to pass to the backend.

    Returns
    -------
        plt.Axes | go.Figure: The resulting plot object.

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

    plot = CircumplexPlot(data, params, backend, style_options)

    if incl_scatter and backend == Backend.SEABORN:
        plot.scatter(apply_styling=True, ax=ax)
        ax = plot.get_axes()
    elif incl_scatter and backend == Backend.PLOTLY:
        # TODO: Implement overlaying scatter on density plot for Plotly backend
        raise NotImplementedError(
            "Overlaying a scatter on a density plot is not yet supported for Plotly backend. "
            "Please change to Seaborn backend or use `incl_scatter=False`."
        )

    if simple_density:
        plot.simple_density(apply_styling=apply_styling, ax=ax)
    else:
        plot.density(apply_styling=apply_styling, ax=ax)

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
    """
    Create a figure with subplots containing circumplex plots.

    Parameters
    ----------
        data_list (List[pd.DataFrame]): List of DataFrames to plot.
        plot_type (PlotType): Type of plot to create.
        incl_scatter (bool): Whether to include scatter points on density plots.
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        figsize (tuple): Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to pass to scatter_plot or density_plot.

    Returns
    -------
        matplotlib.figure.Figure: A figure containing the subplots.

    Example
    -------
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data1 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
        ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
        >>> data2 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
        ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
        >>> fig = create_circumplex_subplots([data1, data2], plot_type=PlotType.SCATTER, nrows=1, ncols=2)
        >>> isinstance(fig, plt.Figure)
        True
    """
    if isinstance(plot_type, str):
        plot_type = PlotType[plot_type.upper()]

    if nrows is None and ncols is None:
        nrows = 2
        ncols = len(data_list) // nrows
    elif nrows is None:
        nrows = len(data_list) // ncols
    elif ncols is None:
        ncols = len(data_list) // nrows

    if subtitles is None:
        subtitles = [f"({i + 1})" for i in range(len(data_list))]
    elif len(subtitles) != len(data_list):
        raise ValueError("Number of subtitles must match number of dataframes")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    color = kwargs.get("color", sns.color_palette("colorblind", 1)[0])

    for data, ax, subtitle in zip(data_list, axes, subtitles):
        if plot_type == PlotType.SCATTER or incl_scatter:
            scatter_plot(data, title=subtitle, ax=ax, color=color, **kwargs)
        if plot_type == PlotType.DENSITY:
            density_plot(data, title=subtitle, ax=ax, color=color, **kwargs)
        elif plot_type == PlotType.SIMPLE_DENSITY:
            density_plot(
                data, title=subtitle, simple_density=True, ax=ax, color=color, **kwargs
            )

    plt.suptitle(title)

    plt.tight_layout()
    return fig
