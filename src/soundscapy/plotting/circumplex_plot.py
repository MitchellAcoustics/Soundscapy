"""
Main module for creating circumplex plots using different backends.
"""

import copy
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from soundscapy.plotting.backends import PlotlyBackend, SeabornBackend
from soundscapy.plotting.plotting_utils import (
    Backend,
    DEFAULT_XLIM,
    DEFAULT_YLIM,
    ExtraParams,
    LayerType,
)
from soundscapy.plotting.stylers import StyleOptions


@dataclass
class Layer:
    """Represents a single plot layer with its data and styling."""

    type: LayerType
    data: pd.DataFrame
    color: Optional[str] = None
    alpha: Optional[float] = None
    label: Optional[str] = None
    hue: Optional[str] = None
    zorder: Optional[int] = None
    params: Dict = field(default_factory=dict)


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
    color: Optional[str] = None
    incl_outline: bool = False
    diagonal_lines: bool = False
    show_labels: bool = True
    legend: bool = "auto"
    legend_location: str = "best"
    extra_params: ExtraParams = field(default_factory=dict)

    def __post_init__(self):
        if self.palette is None and self.hue:
            self.palette = "colorblind"
        if self.color is None and self.palette and not self.hue:
            self.color = sns.color_palette(self.palette)[0]


class CircumplexPlot:
    """
    A class for creating and managing circumplex plots using different backends.

    Supports incremental building of plots through layer addition and maintains
    consistency across plot components.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        params: CircumplexPlotParams = CircumplexPlotParams(),
        style_options: StyleOptions = StyleOptions(),
        backend: Backend = Backend.SEABORN,
        **kwargs: Any,
    ):
        self.data = data
        self.params = params or CircumplexPlotParams()
        self.style_options = style_options or StyleOptions()

        # If we have kwargs, route them to the appropriate parameter object
        if kwargs:
            self._update_from_kwargs(kwargs)

        self._backend = self._create_backend(backend)
        self._layers: List[Layer] = []
        self._color_registry: Dict[str, str] = {}
        self._current_plot = None

    def _update_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Update parameters from kwargs, routing them to the appropriate object.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Keyword arguments to route to either params or style_options
        """
        # Get the field names for each parameter class
        param_fields = {field.name for field in fields(CircumplexPlotParams)}
        style_fields = {field.name for field in fields(StyleOptions)}

        # Sort kwargs into appropriate dictionaries
        param_updates = {}
        style_updates = {}
        extra_params = {}

        for key, value in kwargs.items():
            in_params = key in param_fields
            in_style = key in style_fields

            if in_params and in_style:
                warnings.warn(
                    f"Parameter '{key}' exists in both CircumplexPlotParams and StyleOptions. "
                    "Using as plot parameter."
                )
                param_updates[key] = value
            elif in_params:
                param_updates[key] = value
            elif in_style:
                style_updates[key] = value
            else:
                extra_params[key] = value
                warnings.warn(
                    f"Unknown parameter '{key}' - adding to extra_params.", UserWarning
                )

        # Update the objects
        if param_updates:
            # Create a new params object with updates
            new_params = copy.deepcopy(self.params)
            for key, value in param_updates.items():
                setattr(new_params, key, value)
            self.params = new_params

        if style_updates:
            # Create a new style_options object with updates
            new_style = copy.deepcopy(self.style_options)
            for key, value in style_updates.items():
                setattr(new_style, key, value)
            self.style_options = new_style

        if extra_params:
            # Add any remaining kwargs to extra_params
            self.params.extra_params.update(extra_params)

    def _create_backend(self, backend: Backend):
        """Create the appropriate backend based on the backend enum."""
        if backend == Backend.SEABORN:
            return SeabornBackend(style_options=self.style_options)
        elif backend == Backend.PLOTLY:
            return PlotlyBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _get_color(self, group_key: Optional[str] = None) -> str:
        """Get or generate a color for a group."""
        if group_key is None:
            return (
                self.params.color
                or sns.color_palette(self.params.palette or "colorblind")[0]
            )

        if group_key not in self._color_registry:
            palette = self.params.palette or "colorblind"
            used_colors = set(self._color_registry.values())
            available_colors = [
                c for c in sns.color_palette(palette) if c not in used_colors
            ]
            self._color_registry[group_key] = (
                available_colors[0]
                if available_colors
                else sns.color_palette(palette)[0]
            )

        return self._color_registry[group_key]

    def _add_layer(
        self,
        layer_type: LayerType,
        data: Optional[pd.DataFrame] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        label: Optional[str] = None,
        hue: Optional[str] = None,
        zorder: Optional[int] = None,
        **kwargs,
    ) -> "CircumplexPlot":
        """Add a new layer to the plot."""
        data = data if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for layer")

        # Handle color and alpha defaults
        if color is None and not hue:
            color = self._get_color(label)
        alpha = alpha if alpha is not None else self.params.alpha

        # Create and add the layer
        layer = Layer(
            type=layer_type,
            data=data,
            color=color,
            alpha=alpha,
            label=label,
            hue=hue,
            zorder=zorder,
            params=kwargs,
        )
        self._layers.append(layer)
        return self

    def _update_plot(self, ax: Optional[plt.Axes] = None) -> None:
        """Update the plot with current layers."""
        if isinstance(self._backend, SeabornBackend):
            if ax is None and self._current_plot is None:
                fig, ax = plt.subplots(figsize=self.style_options.figsize)
                self._current_plot = (fig, ax)
            elif ax is not None:
                self._current_plot = (ax.figure, ax)

            fig, ax = self._current_plot
            ax.clear()

            # Plot layers in order
            for layer in sorted(self._layers, key=lambda x: x.zorder or 0):
                self._backend.plot_layer(layer, self.params, ax=ax)

            # Apply styling
            self._backend.apply_styling(self._current_plot, self.params)
        else:
            raise NotImplementedError(
                "Only Seaborn backend is currently supported for layer-based plotting"
            )

    # Public methods that maintain backward compatibility
    def scatter(
        self,
        data: Optional[pd.DataFrame] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        label: Optional[str] = None,
        apply_styling: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> "CircumplexPlot":
        """Create or add a scatter plot."""
        self._add_layer(
            LayerType.SCATTER,
            data=data,
            color=color,
            alpha=alpha,
            label=label,
            zorder=self.style_options.data_zorder + 1,
        )
        if apply_styling:
            self._update_plot(ax)
        return self

    def density(
        self,
        data: Optional[pd.DataFrame] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        label: Optional[str] = None,
        apply_styling: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> "CircumplexPlot":
        """Create or add a density plot."""
        self._add_layer(
            LayerType.DENSITY,
            data=data,
            color=color,
            alpha=alpha,
            label=label,
            zorder=self.style_options.data_zorder,
        )
        if apply_styling:
            self._update_plot(ax)
        return self

    def simple_density(
        self,
        data: Optional[pd.DataFrame] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        label: Optional[str] = None,
        apply_styling: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> "CircumplexPlot":
        """
        Create or add a simple density plot with outline.

        A simple density plot shows fewer contour levels with a cleaner appearance.
        By default, it includes an outline around the contours.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to plot. If None, uses data provided at initialization.
        color : str, optional
            Color for the density contours.
        alpha : float, optional
            Transparency for the filled contours.
        label : str, optional
            Label for the legend.
        apply_styling : bool, optional
            Whether to apply styling after adding the layer.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        CircumplexPlot
            The current CircumplexPlot instance for method chaining.

        Notes
        -----
        The appearance of the simple density plot can be customized through
        the style_options.simple_density settings:
            - thresh: Threshold for density contours
            - levels: Number of contour levels
            - alpha: Transparency for filled contours
            - incl_outline: Whether to include outline
            - outline_alpha: Transparency for outline
            - outline_linewidth: Width of outline lines
        """
        self._add_layer(
            LayerType.SIMPLE_DENSITY,
            data=data,
            color=color,
            alpha=alpha,
            label=label,
            zorder=self.style_options.data_zorder,
        )
        if apply_styling:
            self._update_plot(ax)
        return self

    def get_figure(self):
        """Get the figure object of the plot."""
        if self._current_plot is None:
            raise ValueError("No plot has been created yet. Add some layers first.")
        return self._current_plot[0]

    def get_axes(self):
        """Get the axes object of the plot (only for Seaborn backend)."""
        if self._current_plot is None:
            raise ValueError("No plot has been created yet. Add some layers first.")
        if isinstance(self._backend, SeabornBackend):
            return self._current_plot[1]
        else:
            raise AttributeError("Axes object is not available for Plotly backend")

    def show(self):
        """Display the plot."""
        if self._current_plot is None:
            raise ValueError("No plot has been created yet. Add some layers first.")
        plt.show()
