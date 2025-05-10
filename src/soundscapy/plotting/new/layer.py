"""
Layer classes for visualization.

This module provides the base Layer class and specialized layer implementations
for different visualization techniques. Layers know how to render themselves onto
a PlotContext's axes using parameters provided by the context.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar, cast

import seaborn as sns

from soundscapy.plotting.new.constants import RECOMMENDED_MIN_SAMPLES
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from soundscapy.plotting.new.parameter_models import (
        BaseParams,
        DensityParams,
        ScatterParams,
        SimpleDensityParams,
        SPISimpleDensityParams,
    )
    from soundscapy.plotting.new.protocols import PlotContext

logger = get_logger()


class Layer:
    """
    Base class for all visualization layers.

    A Layer encapsulates a specific visualization technique. Layers know how to
    render themselves onto a PlotContext's axes using parameters provided by the context.

    Attributes
    ----------
    custom_data : pd.DataFrame | None
        Optional custom data for this specific layer, overriding context data
    param_overrides : dict[str, Any]
        Parameter overrides for this layer

    """

    # Class registry for layer types
    _layer_registry: ClassVar[dict[str, type[Layer]]] = {}

    # Parameter type this layer uses (for getting params from context)
    param_type: ClassVar[str] = "base"

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        **param_overrides: Any,
    ) -> None:
        """
        Initialize a Layer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer, overriding context data
        **param_overrides : dict[str, Any]
            Parameter overrides for this layer

        """
        self.custom_data = custom_data
        self.param_overrides = param_overrides

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses in the registry."""
        super().__init_subclass__(**kwargs)
        # Skip registration for the base class
        if cls is not Layer:
            cls._layer_registry[cls.__name__.lower()] = cls

    def render(self, context: PlotContext) -> None:
        """
        Render this layer on the given context.

        Parameters
        ----------
        context : PlotContext
            The context containing data and axes for rendering

        Raises
        ------
        ValueError
            If the context has no associated axes or data

        """
        if context.ax is None:
            msg = "Cannot render layer: context has no associated axes"
            raise ValueError(msg)

        # Use custom data if provided, otherwise context data
        data = self.custom_data if self.custom_data is not None else context.data

        if data is None:
            msg = "No data available for rendering layer"
            raise ValueError(msg)

        # Get parameters from context and apply overrides
        params = self._get_params_from_context(context)

        # Render the layer
        self._render_implementation(data, context, context.ax, params)

    def _get_params_from_context(self, context: PlotContext) -> BaseParams:
        """
        Get parameters from context and apply overrides.

        Parameters
        ----------
        context : PlotContext
            The context to get parameters from

        Returns
        -------
        BaseParams
            The parameters for this layer

        """
        # Get parameters from context based on layer type
        params = context.get_params_for_layer(type(self))

        # Apply overrides
        if self.param_overrides:
            params.update(**self.param_overrides)

        return cast("BaseParams", params)

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
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
        params : BaseParams
            The parameters for this layer

        Raises
        ------
        NotImplementedError
            If not implemented by subclass

        """
        msg = "Subclasses must implement _render_implementation"
        raise NotImplementedError(msg)

    @classmethod
    def create(
        cls, context: PlotContext, layer_type: str | None = None, **kwargs: Any
    ) -> Layer:
        """
        Factory method to create a layer of the specified type.

        Parameters
        ----------
        context : PlotContext
            The context to associate with the layer
        layer_type : str | None
            The type of layer to create (e.g., 'scatter', 'density')
            If None, uses the class name
        **kwargs : Any
            Additional parameters for the layer

        Returns
        -------
        Layer
            The created layer instance

        Raises
        ------
        ValueError
            If the layer type is unknown

        """  # noqa: D401
        if layer_type is None:
            # Use the current class if no type specified
            return cls(context=context, **kwargs)

        # Get the layer class from the registry
        layer_type = layer_type.lower()
        if layer_type not in cls._layer_registry:
            msg = f"Unknown layer type: {layer_type}"
            raise ValueError(msg)

        # Create and return the layer
        layer_class = cls._layer_registry[layer_type]
        return layer_class(**kwargs)


class ScatterLayer(Layer):
    """Layer for rendering scatter plots."""

    param_type = "scatter"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
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
        params : BaseParams
            The parameters for this layer

        """
        # Cast params to the correct type
        scatter_params = cast("ScatterParams", params)

        # Create a copy of the parameters with data
        kwargs = scatter_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Render the scatter plot
        sns.scatterplot(ax=ax, **kwargs)


class DensityLayer(Layer):
    """Layer for rendering density plots."""

    param_type = "density"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
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
        params : BaseParams
            The parameters for this layer

        """
        # Check if we have enough data for a density plot
        if len(data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for "
                f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

        # Cast params to the correct type
        density_params = cast("DensityParams", params)

        # Create a copy of the parameters with data
        kwargs = density_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Render the density plot
        sns.kdeplot(ax=ax, **kwargs)


class SimpleDensityLayer(DensityLayer):
    """Layer for rendering simple density plots (filled contours)."""

    param_type = "simple_density"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render a simple density plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        # Check if we have enough data for a density plot
        if len(data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for "
                f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

        # Cast params to the correct type
        simple_density_params = cast("SimpleDensityParams", params)

        # Create a copy of the parameters with data
        kwargs = simple_density_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Set specific parameters for simple density
        kwargs["levels"] = simple_density_params.levels
        kwargs["thresh"] = getattr(simple_density_params, "thresh", 0.05)

        # Render the simple density plot
        sns.kdeplot(ax=ax, **kwargs)


class SPISimpleLayer(SimpleDensityLayer):
    """Layer for rendering SPI simple density plots."""

    param_type = "spi_simple_density"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render an SPI simple density plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        # Cast params to the correct type
        spi_params = cast("SPISimpleDensityParams", params)

        # Create a copy of the parameters with data
        kwargs = spi_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Set specific parameters for SPI simple density
        kwargs["color"] = spi_params.color
        kwargs["label"] = spi_params.label

        # Render the SPI simple density plot
        sns.kdeplot(ax=ax, **kwargs)

        # Add SPI score text if needed
        if hasattr(spi_params, "show_score") and spi_params.show_score:
            self._add_spi_score_text(context, ax, spi_params)

    def _add_spi_score_text(
        self, context: PlotContext, ax: Axes, params: SPISimpleDensityParams
    ) -> None:
        """
        Add SPI score text to the plot.

        Parameters
        ----------
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : SPISimpleDensityParams
            The parameters for this layer

        """
        # This is a simplified version - in a real implementation,
        # we would calculate and display the actual SPI score
        if params.show_score == "under title":
            # Add text under the title
            if context.title:
                ax.set_title(f"{context.title}\nSPI Score: 0.75")
        elif params.show_score == "on axis" and params.axis_text_kw:
            # Add text on the axis
            text_kwargs = params.axis_text_kw.copy()
            ax.text(s="SPI Score: 0.75", transform=ax.transAxes, **text_kwargs)
