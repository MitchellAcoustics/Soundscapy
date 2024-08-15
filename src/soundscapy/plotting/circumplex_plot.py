import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .backends import PlotlyBackend, SeabornBackend
from .stylers import StyleOptions


@dataclass
class CircumplexPlotParams:
    """
    Parameters for customizing CircumplexPlot.

    Attributes:
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        hue (str | None): Column name for color-coding data points.
        title (str): Title of the plot.
        xlim (tuple[float, float]): x-axis limits.
        ylim (tuple[float, float]): y-axis limits.
        diagonal_lines (bool): Whether to draw diagonal lines.
        show_labels (bool): Whether to show axis labels.
        legend_location (str): Location of the legend.
        extra_params (dict[str, Any]): Additional parameters for backend-specific functions.
    """

    x: str = "ISOPleasant"
    y: str = "ISOEventful"
    hue: str | None = None
    title: str = "Soundscape Plot"
    xlim: Tuple[float, float] = (-1, 1)
    ylim: Tuple[float, float] = (-1, 1)
    alpha: float = 0.8
    fill: bool = True
    palette: str = "colorblind" if hue else None
    incl_outline: bool = (False,)
    diagonal_lines: bool = False
    show_labels: bool = True
    legend_location: str = "best"
    extra_params: dict[str, Any] = field(default_factory=dict)


class Backend(Enum):
    """
    Enum for supported plotting backends.
    """

    SEABORN = "seaborn"
    PLOTLY = "plotly"


class CircumplexPlot:
    """
    A class for creating circumplex plots using different backends.

    This class provides methods for creating scatter plots and density plots
    based on the circumplex model of soundscape perception. It supports multiple
    backends (currently Seaborn and Plotly) and offers various customization options.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        params: CircumplexPlotParams = CircumplexPlotParams(),
        backend: Backend = Backend.SEABORN,
        style_options: StyleOptions = StyleOptions(),
    ):
        self.data = data
        self.params = params
        self.style_options = style_options
        self._backend = self._create_backend(backend)
        self._plot = None

    def _create_backend(self, backend: Backend):
        """
        Create the appropriate backend based on the backend enum.

        Args:
            backend (Backend): The backend to create.

        Returns:
            PlotBackend: The created backend instance.

        Raises:
            ValueError: If an unsupported backend is specified.
        """
        if backend == Backend.SEABORN:
            return SeabornBackend(style_options=self.style_options)
        elif backend == Backend.PLOTLY:
            return PlotlyBackend(style_options=self.style_options)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def scatter(
        self, apply_styling: bool = True, ax: Optional[plt.Axes] = None
    ) -> "CircumplexPlot":
        """
        Create a scatter plot.

        Args:
            apply_styling (bool): Whether to apply circumplex-specific styling.
            ax (plt.Axes, optional): A matplotlib Axes object to plot on. If None, a new figure is created.

        Returns:
            CircumplexPlot: The current instance for method chaining.
        """
        self._plot = self._backend.create_scatter(self.data, self.params, ax)
        if apply_styling:
            self._plot = self._backend.apply_styling(self._plot, self.params)
        return self

    def density(
        self, apply_styling: bool = True, ax: Optional[plt.Axes] = None
    ) -> "CircumplexPlot":
        """
        Create a density plot.

        Args:
            apply_styling (bool): Whether to apply circumplex-specific styling.
            ax (plt.Axes, optional): A matplotlib Axes object to plot on. If None, a new figure is created.

        Returns:
            CircumplexPlot: The current instance for method chaining.
        """
        self._plot = self._backend.create_density(self.data, self.params, ax)
        if apply_styling:
            self._plot = self._backend.apply_styling(self._plot, self.params)
        return self

    def simple_density(
        self, apply_styling: bool = True, ax: Optional[plt.Axes] = None
    ) -> "CircumplexPlot":
        """
        Create a simple density plot.

        Args:
            apply_styling (bool): Whether to apply circumplex-specific styling.
            ax (plt.Axes, optional): A matplotlib Axes object to plot on. If None, a new figure is created.

        Returns:
            CircumplexPlot: The current instance for method chaining.
        """
        if isinstance(self._backend, SeabornBackend):
            self._plot = self._backend.create_simple_density(self.data, self.params, ax)
            if apply_styling:
                self._plot = self._backend.apply_styling(self._plot, self.params)
        else:
            raise NotImplementedError(
                "Simple density plots are only available for the Seaborn backend."
            )
        return self

    def show(self):
        """
        Display the plot.

        For Seaborn backend, this uses plt.show(). For Plotly backend, it calls the backend's show method.
        """
        self._backend.show(self._plot)

    def get_figure(self):
        """
        Get the figure object of the plot.

        Returns:
            The figure object (matplotlib.figure.Figure for Seaborn, go.Figure for Plotly).

        Raises:
            ValueError: If no plot has been created yet.
        """
        if self._plot is None:
            raise ValueError(
                "No plot has been created yet. Call scatter() or density() first."
            )
        return self._plot

    def get_axes(self):
        """
        Get the axes object of the plot (only for Seaborn backend).

        Returns:
            matplotlib.axes.Axes: The axes object.

        Raises:
            ValueError: If no plot has been created yet.
            AttributeError: If using Plotly backend.
        """
        if self._plot is None:
            raise ValueError(
                "No plot has been created yet. Call scatter() or density() first."
            )
        if isinstance(self._backend, SeabornBackend):
            return self._plot[1]  # Return the axes object
        else:
            raise AttributeError("Axes object is not available for Plotly backend")

    def get_style_options(self) -> StyleOptions:
        """
        Get the current StyleOptions.

        Returns:
            StyleOptions: The current style options.
        """
        return copy.deepcopy(self.style_options)

    def update_style_options(self, **kwargs) -> "CircumplexPlot":
        """
        Update the StyleOptions with new values.

        Args:
            **kwargs: Keyword arguments corresponding to StyleOptions attributes.

        Returns:
            CircumplexPlot: The current instance for method chaining.
        """
        new_style_options = copy.deepcopy(self.style_options)
        for key, value in kwargs.items():
            if hasattr(new_style_options, key):
                setattr(new_style_options, key, value)
            else:
                raise ValueError(f"Invalid StyleOptions attribute: {key}")

        self.style_options = new_style_options
        self._backend.style_options = new_style_options
        return self

    def iso_annotation(self, location, x_adj=0, y_adj=0, **kwargs):
        """
        Add an annotation to the plot (only for Seaborn backend).

        Args:
            location: The index of the point to annotate.
            x_adj (float): Adjustment to the x-coordinate of the annotation.
            y_adj (float): Adjustment to the y-coordinate of the annotation.
            **kwargs: Additional keyword arguments to pass to ax.annotate().

        Returns:
            CircumplexPlot: The current instance for method chaining.

        Raises:
            AttributeError: If using Plotly backend.
        """
        if isinstance(self._backend, SeabornBackend):
            ax = self.get_axes()
            x = self.data[self.params.x].iloc[location]
            y = self.data[self.params.y].iloc[location]
            ax.annotate(
                text=self.data.index[location],
                xy=(x, y),
                xytext=(x + x_adj, y + y_adj),
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", ec="black"),
                **kwargs,
            )
        else:
            raise AttributeError("iso_annotation is not available for Plotly backend")
        return self


def scatter_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    hue: Optional[str] = None,
    title: str = "Soundscape Scatter Plot",
    xlim: Tuple[float, float] = (-1, 1),
    ylim: Tuple[float, float] = (-1, 1),
    palette: str = "colorblind",
    diagonal_lines: bool = False,
    show_labels: bool = True,
    legend_location: str = "best",
    backend: Backend = Backend.SEABORN,
    apply_styling: bool = True,
    figsize: Tuple[int, int] = (5, 5),
    ax: Optional[plt.Axes] = None,
    extra_params: Dict[str, Any] = {},
    **kwargs: Any,
) -> plt.Axes | go.Figure:
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
    xlim: Tuple[float, float] = (-1, 1),
    ylim: Tuple[float, float] = (-1, 1),
    palette: str = "colorblind",
    fill: bool = True,
    incl_outline: bool = False,
    diagonal_lines: bool = False,
    show_labels: bool = True,
    legend_location: str = "best",
    backend: Backend = Backend.SEABORN,
    apply_styling: bool = True,
    figsize: Tuple[int, int] = (5, 5),
    simple_density: bool = False,
    simple_density_thresh: float = 0.5,
    simple_density_levels: int = 2,
    simple_density_alpha: float = 0.5,
    ax: Optional[plt.Axes] = None,
    extra_params: Dict[str, Any] = {},
    **kwargs: Any,
) -> plt.Axes | go.Figure:
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
    plot_type: str = "density",
    incl_scatter: bool = True,
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[int, int] = (10, 10),
    **kwargs: Any,
) -> plt.Figure:
    """
    Create a figure with subplots containing circumplex plots.

    Parameters:
        data_list (List[pd.DataFrame]): List of DataFrames to plot.
        plot_type (str): Type of plot to create ("scatter" or "density"). Default is "scatter".
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        figsize (tuple): Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to pass to scatter_plot or density_plot.

    Returns:
        matplotlib.figure.Figure: A figure containing the subplots.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data1 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
        ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
        >>> data2 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
        ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
        >>> fig = create_circumplex_subplots([data1, data2], plot_type="scatter", nrows=1, ncols=2)
        >>> isinstance(fig, plt.Figure)
        True
    """
    import seaborn as sns

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    color = kwargs.get("color", sns.color_palette("colorblind", 1)[0])

    for data, ax in zip(data_list, axes):
        if plot_type == "scatter" or incl_scatter:
            scatter_plot(data, ax=ax, color=color, **kwargs)
        if plot_type == "density":
            density_plot(data, ax=ax, color=color, **kwargs)
        elif plot_type not in ["scatter", "density", "simple_density"]:
            raise ValueError(
                "plot_type must be either 'scatter', 'density', or 'simple_density'"
            )

    plt.tight_layout()
    return fig
