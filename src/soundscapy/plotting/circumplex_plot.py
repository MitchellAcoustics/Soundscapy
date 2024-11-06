"""
Main module for creating circumplex plots using different backends.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from soundscapy.plotting.backends import PlotlyBackend, SeabornBackend
from soundscapy.plotting.plotting_utils import (
    Backend,
    DEFAULT_XLIM,
    DEFAULT_YLIM,
    ExtraParams,
    PlotType,
)
from soundscapy.plotting.stylers import StyleOptions


@dataclass
class CircumplexPlotParams:
    """Parameters for customizing CircumplexPlot."""

    x: str = "ISOPleasant"
    y: str = "ISOEventful"
    hue: Optional[str] = None
    title: str = "Soundscape Plot"
    xlim: Tuple[float, float] = DEFAULT_XLIM
    ylim: Tuple[float, float] = DEFAULT_YLIM
    alpha: float = 0.8
    fill: bool = True
    palette: Optional[str] = None
    incl_outline: bool = False  # Fixed from (False,)
    diagonal_lines: bool = False
    show_labels: bool = True
    legend: bool = "auto"
    legend_location: str = "best"
    extra_params: ExtraParams = field(default_factory=dict)

    def __post_init__(self):
        if self.palette is None:
            self.palette = "colorblind" if self.hue else None


class CircumplexPlot:
    """
    A class for creating circumplex plots using different backends.

    This class provides methods for creating scatter plots and density plots
    based on the circumplex model of soundscape perception. It supports multiple
    backends (currently Seaborn and Plotly) and offers various customization options.

    """

    # TODO: Implement jointplot method for Seaborn backend.
    # TODO: Implement density plots for Plotly backend.
    # TODO: Improve Plotly backend to support more customization options.

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
        """Create the appropriate backend based on the backend enum."""
        if backend == Backend.SEABORN:
            return SeabornBackend(style_options=self.style_options)
        elif backend == Backend.PLOTLY:
            return PlotlyBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_plot(
        self,
        plot_type: PlotType,
        apply_styling: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> "CircumplexPlot":
        """Create a plot based on the specified plot type."""
        if plot_type == PlotType.SCATTER:
            if isinstance(self._backend, SeabornBackend):
                self._plot = self._backend.create_scatter(self.data, self.params, ax)
            else:
                self._plot = self._backend.create_scatter(self.data, self.params)
        elif plot_type == PlotType.DENSITY:
            if isinstance(self._backend, SeabornBackend):
                self._plot = self._backend.create_density(self.data, self.params, ax)
            else:
                raise NotImplementedError(
                    "Density plots are only available for the Seaborn backend."
                )
        elif plot_type == PlotType.SIMPLE_DENSITY:
            if isinstance(self._backend, SeabornBackend):
                self._plot = self._backend.create_simple_density(
                    self.data, self.params, ax
                )
            else:
                raise NotImplementedError(
                    "Simple density plots are only available for the Seaborn backend."
                )
        elif plot_type == PlotType.JOINT:
            if isinstance(self._backend, SeabornBackend):
                self._plot = self._backend.create_jointplot(self.data, self.params)
            else:
                raise NotImplementedError(
                    "Joint plots are only available for the Seaborn backend."
                )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        if apply_styling:
            self._plot = self._backend.apply_styling(self._plot, self.params)
        return self

    def scatter(
        self, apply_styling: bool = True, ax: Optional[plt.Axes] = None
    ) -> "CircumplexPlot":
        """Create a scatter plot."""
        return self._create_plot(PlotType.SCATTER, apply_styling, ax)

    def density(
        self, apply_styling: bool = True, ax: Optional[plt.Axes] = None
    ) -> "CircumplexPlot":
        """Create a density plot."""
        return self._create_plot(PlotType.DENSITY, apply_styling, ax)

    def jointplot(self, apply_styling: bool = True) -> "CircumplexPlot":
        """Create a joint plot."""
        return self._create_plot(PlotType.JOINT, apply_styling)

    def simple_density(
        self, apply_styling: bool = True, ax: Optional[plt.Axes] = None
    ) -> "CircumplexPlot":
        """Create a simple density plot."""
        return self._create_plot(PlotType.SIMPLE_DENSITY, apply_styling, ax)

    def show(self):
        """Display the plot."""
        if self._plot is None:
            raise ValueError(
                "No plot has been created yet. Call scatter(), density(), or simple_density() first."
            )
        self._backend.show(self._plot)

    def get_figure(self):
        """Get the figure object of the plot."""
        if self._plot is None:
            raise ValueError(
                "No plot has been created yet. Call scatter(), density(), or simple_density() first."
            )
        return self._plot

    def get_axes(self):
        """Get the axes object of the plot (only for Seaborn backend)."""
        if self._plot is None:
            raise ValueError(
                "No plot has been created yet. Call scatter(), density(), or simple_density() first."
            )
        if isinstance(self._backend, SeabornBackend):
            return self._plot[1]  # Return the axes object
        else:
            raise AttributeError("Axes object is not available for Plotly backend")

    def get_style_options(self) -> StyleOptions:
        """Get the current StyleOptions."""
        return copy.deepcopy(self.style_options)

    def update_style_options(self, **kwargs) -> "CircumplexPlot":
        """Update the StyleOptions with new values."""
        new_style_options = copy.deepcopy(self.style_options)
        for key, value in kwargs.items():
            if hasattr(new_style_options, key):
                setattr(new_style_options, key, value)
            else:
                raise ValueError(f"Invalid StyleOptions attribute: {key}")

        self.style_options = new_style_options
        self._backend.style_options = new_style_options
        return self

    def iso_annotation(self, location, x_adj: float = 0, y_adj: float = 0, **kwargs):
        """Add an annotation to the plot (only for Seaborn backend)."""
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
