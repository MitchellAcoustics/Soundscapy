"""
Layer-based visualization components for plotting.

This module provides a system of layer classes that implement different visualization
techniques for ISO plots. Each layer encapsulates a specific visualization method
and knows how to render itself on a given context.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet

from soundscapy.plotting.defaults import (
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
)
from soundscapy.plotting.plot_context import PlotContext
from soundscapy.plotting.plotting_types import DensityParamTypes, ScatterParamTypes


class Layer:
    """
    Base class for all visualization layers.

    A Layer encapsulates a specific visualization technique and its associated
    parameters. Layers know how to render themselves onto a PlotContext's axes.

    Attributes
    ----------
    custom_data : pd.DataFrame | None
        Optional custom data for this specific layer, overriding context data
    params : dict
        Parameters for the layer
    """

    def __init__(
        self, custom_data: Optional[pd.DataFrame] = None, **params: Any
    ) -> None:
        """
        Initialize a Layer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer, overriding context data
        **params : dict
            Parameters for the layer
        """
        self.custom_data = custom_data
        self.params = params

    def render(self, context: PlotContext) -> None:
        """
        Render this layer on the given context.

        Parameters
        ----------
        context : PlotContext
            The context containing data and axes for rendering
        """
        if context.ax is None:
            raise ValueError("Cannot render layer: context has no associated axes")

        # Use custom data if provided, otherwise context data
        data = self.custom_data if self.custom_data is not None else context.data

        if data is None:
            raise ValueError("No data available for rendering layer")

        self._render_implementation(data, context, context.ax)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
    ) -> None:
        """
        Implement actual rendering (to be overridden by subclasses).

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        """
        raise NotImplementedError("Subclasses must implement _render_implementation")


class ScatterLayer(Layer):
    """Layer for rendering scatter plots."""

    def __init__(
        self, custom_data: Optional[pd.DataFrame] = None, **params: Any
    ) -> None:
        """
        Initialize a ScatterLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        **params : dict
            Parameters for the scatter plot
        """
        default_params = DEFAULT_SCATTER_PARAMS.copy()
        merged_params = {**default_params, **params}
        super().__init__(custom_data=custom_data, **merged_params)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
    ) -> None:
        """
        Render a scatter plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        """
        # Get data-specific properties or fall back to context defaults
        x = self.params.get("x", context.x)
        y = self.params.get("y", context.y)
        hue = self.params.get("hue", context.hue)

        # Filter out x, y, hue and data parameters to avoid duplicate kwargs
        plot_params = {
            k: v for k, v in self.params.items() if k not in ("x", "y", "hue", "data")
        }

        # Render scatter plot
        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, **plot_params)


class DensityLayer(Layer):
    """Layer for rendering kernel density plots."""

    def __init__(
        self,
        custom_data: Optional[pd.DataFrame] = None,
        include_outline: bool = False,
        **params: Any,
    ) -> None:
        """
        Initialize a DensityLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        include_outline : bool
            Whether to include an outline around the density plot
        **params : dict
            Parameters for the density plot
        """
        default_params = DEFAULT_DENSITY_PARAMS.copy()
        merged_params = {**default_params, **params}
        self.include_outline = include_outline
        super().__init__(custom_data=custom_data, **merged_params)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
    ) -> None:
        """
        Render a density plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        """
        # Check if there's enough data for a meaningful density plot
        if len(data) < 30:
            import warnings

            warnings.warn(
                "Density plots are not recommended for small datasets (<30 samples).",
                UserWarning,
                stacklevel=2,
            )

        # Get data-specific properties or fall back to context defaults
        x = self.params.get("x", context.x)
        y = self.params.get("y", context.y)
        hue = self.params.get("hue", context.hue)

        # Filter out x, y, hue and data parameters to avoid duplicate kwargs
        plot_params = {
            k: v for k, v in self.params.items() if k not in ("x", "y", "hue", "data")
        }

        # Render density plot
        sns.kdeplot(data=data, x=x, y=y, hue=hue, ax=ax, **plot_params)

        # If requested, add an outline around the density plot
        if self.include_outline:
            outline_params = plot_params.copy()
            outline_params.update({"fill": False, "alpha": 1, "legend": False})
            sns.kdeplot(data=data, x=x, y=y, hue=hue, ax=ax, **outline_params)


class SimpleDensityLayer(DensityLayer):
    """Layer for rendering simplified density plots with fewer contour levels."""

    def __init__(
        self,
        custom_data: Optional[pd.DataFrame] = None,
        include_outline: bool = True,
        **params: Any,
    ) -> None:
        """
        Initialize a SimpleDensityLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        include_outline : bool
            Whether to include an outline around the density plot
        **params : dict
            Parameters for the density plot
        """
        default_params = DEFAULT_SIMPLE_DENSITY_PARAMS.copy()
        merged_params = {**default_params, **params}
        super().__init__(
            custom_data=custom_data, include_outline=include_outline, **merged_params
        )
