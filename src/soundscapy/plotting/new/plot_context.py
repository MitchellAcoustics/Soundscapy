"""
Data and state management for plotting layer_mgr.

This module provides the PlotContext class that manages custom_data, state, and parameters
for ISOPlot visualizations. The PlotContext is the central component in the plotting
architecture, owning both custom_data and parameter models for different layer types.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from soundscapy.plotting.new.constants import DEFAULT_XCOL, DEFAULT_YCOL
from soundscapy.plotting.new.parameter_models import (
    DensityParams,
    ScatterParams,
    SimpleDensityParams,
    SPISimpleDensityParams,
    StyleParams,
    _ParamModels,
)
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from soundscapy.plotting.new.iso_plot import ISOPlot
    from soundscapy.plotting.new.layer import Layer
    from soundscapy.plotting.new.protocols import RenderableLayer


logger = get_logger()


@dataclass
class ParamModels:
    """
    Container for parameter models with methods for retrieval and transformation.

    Attributes
    ----------
    scatter : ScatterParams
    density : DensityParams
    simple_density : SimpleDensityParams
    spi_simple_density : SPISimpleDensityParams
    style : StyleParams

    """

    scatter: ScatterParams = field(default_factory=ScatterParams)
    density: DensityParams = field(default_factory=DensityParams)
    simple_density: SimpleDensityParams = field(default_factory=SimpleDensityParams)
    spi_simple_density: SPISimpleDensityParams = field(
        default_factory=SPISimpleDensityParams
    )
    style: StyleParams = field(default_factory=StyleParams)

    def get(
        self, model_name: str, default: _ParamModels | None = None
    ) -> _ParamModels | None:
        return getattr(self, model_name, default)

    @property
    def names(self) -> list[str]:
        return list(self.__dict__.keys())

    def as_dict(self) -> dict[str, _ParamModels]:
        return self.__dict__


class PlotContext:
    """
    Manages custom_data, state, and parameters for a plot or subplot.

    This class centralizes the management of custom_data, coordinates, parameters, and other
    state needed for rendering plot layer_mgr. It owns parameter models for different
    layer types and provides them to layer_mgr when needed.

    Attributes
    ----------
    data : pd.DataFrame | None
        The custom_data associated with this context
    x : str
        The column name for x-axis custom_data
    y : str
        The column name for y-axis custom_data
    hue : str | None
        The column name for color encoding, if any
    ax : Axes | None
        The matplotlib Axes object this context is associated with
    title : str | None
        The title for this context's plot
    layers : list[RenderableLayer]
        The visualization layer_mgr to be rendered on this context
    parent : PlotContext | None
        The parent context, if this is a child context

    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        x: str = DEFAULT_XCOL,
        y: str = DEFAULT_YCOL,
        hue: str | None = None,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> None:
        """
        Initialize a PlotContext.

        Parameters
        ----------
        data : pd.DataFrame | None
            Data to be visualized
        x : str
            Column name for x-axis custom_data
        y : str
            Column name for y-axis custom_data
        hue : str | None
            Column name for color encoding
        ax : Axes | None
            Matplotlib axis to render on
        title : str | None
            Title for this plot context

        """
        # Basic properties
        self.data = data
        self.x = x
        self.y = y
        self.hue = hue
        self.ax = ax
        self.title = title
        self.layers: list[RenderableLayer] = []
        self.parent: PlotContext | None = None

        # Parameter models for different layer types
        self._param_models: ParamModels = field(default_factory=ParamModels)

        # Initialize default parameter models
        self._init_param_models()

    def _init_param_models(self) -> None:
        """Initialize parameter models with context values."""
        # Common parameters for all models
        data = self.data
        x = self.x
        y = self.y
        hue = self.hue

        # Create parameter models for different layer types
        self._param_models = ParamModels(
            scatter=ScatterParams(data=data, x=x, y=y, hue=hue),
            density=DensityParams(data=data, x=x, y=y, hue=hue),
            simple_density=SimpleDensityParams(data=data, x=x, y=y, hue=hue),
            spi_simple_density=SPISimpleDensityParams(data=data, x=x, y=y, hue=hue),
            style=StyleParams(),
        )

    def get_params(self, param_type: str) -> _ParamModels:
        """
        Get parameters for a specific type.

        Parameters
        ----------
        param_type : str
            The type of parameters to get (e.g., 'scatter', 'density')

        Returns
        -------
        BaseParams
            The parameter model instance

        Raises
        ------
        ValueError
            If the parameter type is unknown

        """
        model = self._param_models.get(param_type)
        if model is not None:
            return model
        msg = f"Unknown parameter type: {param_type}"
        raise ValueError(msg)

    def get_params_for_layer(self, layer_type: type[Layer]) -> _ParamModels:
        """
        Get parameters appropriate for a specific layer type.

        This method maps layer types to their corresponding parameter models.

        Parameters
        ----------
        layer_type : type[RenderableLayer]
            The type of layer to get parameters for

        Returns
        -------
        ParamModel
            The parameter model instance

        """
        # Map layer class names to parameter types
        # This could be improved with a more formal registry
        layer_name = layer_type.__name__.lower()

        if "scatter" in layer_name:
            return self.get_params("scatter")
        if "simple" in layer_name:
            if "spi" in layer_name:
                return self.get_params("spi_simple_density")
            return self.get_params("simple_density")
        if "density" in layer_name:
            return self.get_params("density")

        # Default to scatter parameters if no match
        logger.warning(
            f"No specific parameters for layer type {layer_type.__name__}, "  # noqa: G004
            "using scatter parameters"
        )
        return self.get_params("scatter")

    def update_params(self, param_type: str, **kwargs: Any) -> _ParamModels:
        """
        Update parameters for a specific type.

        Parameters
        ----------
        param_type : str
            The type of parameters to update
        **kwargs : Any
            Parameter values to update

        Returns
        -------
        BaseParams
            The updated parameter model instance

        """
        params = self.get_params(param_type)
        params.update(**kwargs)
        return params

    def create_child(
        self,
        data: pd.DataFrame | None = None,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> PlotContext:
        """
        Create a child context that inherits properties from this context.

        Parameters
        ----------
        data : pd.DataFrame | None
            Data for the child context. If None, inherits from parent.
        title : str | None
            Title for the child context
        ax : Axes | None
            Matplotlib axis for the child context

        Returns
        -------
        PlotContext
            A new child context with inherited properties

        """
        child = PlotContext(
            data=data if data is not None else self.data,
            x=self.x,
            y=self.y,
            hue=self.hue,
            ax=ax,
            title=title,
        )
        # Copy parameter models from parent to child
        child._param_models = dataclasses.replace(self._param_models)

        # Set parent reference
        child.parent = self

        return child

    @staticmethod
    def ensure_axes_exist(plot: ISOPlot) -> None:
        """
        Check if we have axes to render on, create if needed.

        This method ensures that the plot has axes to render on,
        creating them if necessary.

        Parameters
        ----------
        plot : Any
            The parent plot instance

        """
        if plot.figure is None:
            # Create a new figure and axes
            logger.info("Creating new figure and axes")
            plot.figure, plot.axes = plt.subplots(figsize=(5, 5))

    def get_axes_by_spec(
        self, plot: ISOPlot, spec: int | tuple[int, int] | list[int] | None
    ) -> list[Axes]:
        """
        Get axes based on specification.

        Parameters
        ----------
        plot : ISOPlot
            The parent plot instance
        spec : int | tuple[int, int] | list[int] | None
            The axis specification:
            - None: All subplot axes
            - int: Single subplot at flattened index
            - tuple[int, int]: Subplot at (row, col)
            - list[int]: Multiple subplot_mgr at specified indices

        Returns
        -------
        list[Axes]
            List of matplotlib Axes objects

        """
        # Get the contexts based on specification
        contexts = self.get_contexts_by_spec(plot, spec)

        # Extract the axes from each context
        return [context.ax for context in contexts if context.ax is not None]

    @classmethod
    def get_contexts_by_spec(
        cls, plot: ISOPlot, spec: int | tuple[int, int] | list[int] | None
    ) -> list[PlotContext]:
        """
        Resolve which subplot contexts to target based on axis specification.

        Parameters
        ----------
        plot : ISOPlot
            The parent plot instance
        spec : int | tuple[int, int] | list[int] | None
            The axis specification:
            - None: All subplot contexts
            - int: Single subplot at flattened index
            - tuple[int, int]: Subplot at (row, col)
            - list[int]: Multiple subplot_mgr at specified indices

        Returns
        -------
        list[PlotContext]
            List of target subplot contexts

        """
        # If no specific axis, target all subplot contexts
        if spec is None:
            return plot.subplot_contexts

        # Convert axis specification to list of indices
        indices = cls.resolve_axis_indices(plot, spec)

        # Get the contexts for each valid index
        target_contexts = []
        for idx in indices:
            if 0 <= idx < len(plot.subplot_contexts):
                target_contexts.append(plot.subplot_contexts[idx])
            else:
                msg = f"Subplot index {idx} out of range"
                raise IndexError(msg)

        return target_contexts

    @staticmethod
    def resolve_axis_indices(
        plot: ISOPlot, spec: int | tuple[int, int] | list[int]
    ) -> list[int]:
        """
        Convert axis specification to list of indices.

        Parameters
        ----------
        plot : Any
            The parent plot instance
        spec : int | tuple[int, int] | list[int]
            The axis specification to resolve

        Returns
        -------
        list[int]
            List of flattened indices

        Raises
        ------
        ValueError
            If an invalid axis specification is provided

        """
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, tuple) and len(spec) == 2:
            # Convert (row, col) to flattened index
            row, col = spec
            return [row * plot.subplots_params.ncols + col]
        if isinstance(spec, list):
            return spec
        msg = f"Invalid axis specification: {spec}"
        raise ValueError(msg)
