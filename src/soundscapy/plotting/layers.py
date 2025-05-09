"""
Layer-based visualization components for plotting.

This module provides a system of layer classes that implement different visualization
techniques for ISO plots. Each layer encapsulates a specific visualization method
and knows how to render itself on a given context.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import seaborn as sns

from soundscapy.plotting.defaults import RECOMMENDED_MIN_SAMPLES
from soundscapy.plotting.plotting_types import (
    DensityParams,
    ScatterParams,
    SeabornParams,
    SimpleDensityParams,
)
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from soundscapy.plotting.plot_context import PlotContext

logger = get_logger()


class Layer:
    """
    Base class for all visualization layers.

    A Layer encapsulates a specific visualization technique and its associated
    parameters. Layers know how to render themselves onto a PlotContext's axes.

    Attributes
    ----------
    custom_data : pd.DataFrame | None
        Optional custom data for this specific layer, overriding context data
    params : ParamModel
        Parameter model instance for this layer

    """

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        param_model: type[SeabornParams] = SeabornParams,
        **params: Any,
    ) -> None:
        """
        Initialize a Layer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer, overriding context data
        param_model : type[ParamModel] | None
            The parameter model class to use, if None uses a generic ParamModel
        **params : dict
            Parameters for the layer

        """
        self.custom_data = custom_data
        # Create parameter model instance
        self.params = param_model(**params)

    def render(self, context: PlotContext) -> None:
        """
        Render this layer on the given context.

        Parameters
        ----------
        context : PlotContext
            The context containing data and axes for rendering

        """
        if context.ax is None:
            msg = "Cannot render layer: context has no associated axes"
            raise ValueError(msg)

        # Use custom data if provided, otherwise context data
        data = self.custom_data if self.custom_data is not None else context.data

        if data is None:
            msg = "No data available for rendering layer"
            raise ValueError(msg)

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
        msg = "Subclasses must implement _render_implementation"
        raise NotImplementedError(msg)


class ScatterLayer(Layer):
    """Layer for rendering scatter plots."""

    def __init__(self, custom_data: pd.DataFrame | None = None, **params: Any) -> None:
        """
        Initialize a ScatterLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        **params : dict
            Parameters for the scatter plot

        """
        self.params: ScatterParams
        super().__init__(custom_data=custom_data, param_model=ScatterParams, **params)

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

        # Filter out x, y, hue and data parameters to avoid duplicate kwargs
        plot_params = self.params.model_copy()
        plot_params = plot_params.drop(["x", "y", "data"])

        # Apply palette only if hue is used
        plot_params.crosscheck_palette_hue()

        # Render scatter plot
        sns.scatterplot(data=data, x=x, y=y, ax=ax, **plot_params.as_dict())


class DensityLayer(Layer):
    """Layer for rendering kernel density plots."""

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        *,
        param_model: type[DensityParams] = DensityParams,
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
        self.include_outline = include_outline
        self.params: DensityParams
        super().__init__(custom_data=custom_data, param_model=param_model, **params)

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
        if len(data) < RECOMMENDED_MIN_SAMPLES:
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
        plot_params = self.params.model_copy()
        plot_params = plot_params.drop(["x", "y", "data"])

        # Apply palette only if hue is used
        plot_params.crosscheck_palette_hue()

        # Render density plot
        sns.kdeplot(data=data, x=x, y=y, ax=ax, **plot_params.as_dict())

        # If requested, add an outline around the density plot
        if self.include_outline:
            sns.kdeplot(
                data=data,
                x=x,
                y=y,
                ax=ax,
                **plot_params.get_outline_dict(),
            )


class SimpleDensityLayer(DensityLayer):
    """Layer for rendering simplified density plots with fewer contour levels."""

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        *,
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
        super().__init__(
            custom_data=custom_data,
            include_outline=include_outline,
            param_model=SimpleDensityParams,
            **params,
        )
