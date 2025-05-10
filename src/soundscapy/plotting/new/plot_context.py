"""
Data and state management for plotting layers.

This module provides the PlotContext class that manages data, state, and parameters
for ISOPlot visualizations. The PlotContext is the central component in the plotting
architecture, owning both data and parameter models for different layer types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from soundscapy.plotting.new.constants import DEFAULT_XCOL, DEFAULT_YCOL
from soundscapy.plotting.new.parameter_models import (
    BaseParams,
    DensityParams,
    ScatterParams,
    SimpleDensityParams,
    SPISimpleDensityParams,
    StyleParams,
)
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from soundscapy.plotting.new.protocols import ParamModel, RenderableLayer

logger = get_logger()


class PlotContext:
    """
    Manages data, state, and parameters for a plot or subplot.

    This class centralizes the management of data, coordinates, parameters, and other
    state needed for rendering plot layers. It owns parameter models for different
    layer types and provides them to layers when needed.

    Attributes
    ----------
    data : pd.DataFrame | None
        The data associated with this context
    x : str
        The column name for x-axis data
    y : str
        The column name for y-axis data
    hue : str | None
        The column name for color encoding, if any
    ax : Axes | None
        The matplotlib Axes object this context is associated with
    title : str | None
        The title for this context's plot
    layers : list[RenderableLayer]
        The visualization layers to be rendered on this context
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
            Column name for x-axis data
        y : str
            Column name for y-axis data
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
        self._param_models: dict[str, BaseParams] = {}

        # Initialize default parameter models
        self._init_param_models()

    def _init_param_models(self) -> None:
        """Initialize parameter models with context values."""
        # Common parameters for all models
        common_params = {
            "data": self.data,
            "x": self.x,
            "y": self.y,
            "hue": self.hue,
        }

        # Create parameter models for different layer types
        self._param_models["scatter"] = ScatterParams(**common_params)
        self._param_models["density"] = DensityParams(**common_params)
        self._param_models["simple_density"] = SimpleDensityParams(**common_params)
        self._param_models["spi_simple_density"] = SPISimpleDensityParams(
            **common_params
        )
        self._param_models["style"] = StyleParams(
            xlim=(-1, 1),
            ylim=(-1, 1),
            xlabel=r"$P_{ISO}$",
            ylabel=r"$E_{ISO}$",
        )

    def get_params(self, param_type: str) -> BaseParams:
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
        if param_type not in self._param_models:
            msg = f"Unknown parameter type: {param_type}"
            raise ValueError(msg)

        return self._param_models[param_type]

    def get_params_for_layer(self, layer_type: type[RenderableLayer]) -> ParamModel:
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
            return cast("ParamModel", self.get_params("scatter"))
        if "simpledensity" in layer_name:
            if "spi" in layer_name:
                return cast("ParamModel", self.get_params("spi_simple_density"))
            return cast("ParamModel", self.get_params("simple_density"))
        if "density" in layer_name:
            return cast("ParamModel", self.get_params("density"))

        # Default to scatter parameters if no match
        logger.warning(
            f"No specific parameters for layer type {layer_type.__name__}, "  # noqa: G004
            "using scatter parameters"
        )
        return cast("ParamModel", self.get_params("scatter"))

    def update_params(self, param_type: str, **kwargs: Any) -> BaseParams:
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
        for param_type, model in self._param_models.items():
            child._param_models[param_type] = model.model_copy()

        # Set parent reference
        child.parent = self

        return child
