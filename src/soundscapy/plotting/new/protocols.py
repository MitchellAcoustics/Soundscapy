"""
Protocol classes for the plotting module.

This module defines Protocol classes that specify interfaces for various
components in the plotting system. These protocols enable structural typing
rather than nominal typing, making it easier to compose functionality without
complex inheritance hierarchies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

# Type variable for generic parameter models
P = TypeVar("P", bound="ParamModel")


class RenderableLayer(Protocol):
    """Protocol defining what a renderable layer must implement."""

    def render(self, context: PlotContext) -> None:
        """
        Render the layer on the given context.

        Parameters
        ----------
        context : PlotContext
            The context containing custom_data and axes for rendering

        """
        ...


class ParameterProvider(Protocol):
    """Protocol defining how parameters are provided."""

    def get_params(self, param_type: str) -> ParamModel:
        """
        Get parameters for a specific type.

        Parameters
        ----------
        param_type : str
            The type of parameters to get

        Returns
        -------
        ParamModel
            The parameter model instance

        """
        ...


class ParamModel(Protocol):
    """Protocol defining the interface for parameter models."""

    def update(self, **kwargs: Any) -> Any:
        """
        Update parameters with new values.

        Parameters
        ----------
        **kwargs : Any
            New parameter values

        Returns
        -------
        Self
            The updated parameter instance

        """
        ...

    def as_dict(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter values

        """
        ...


class PlotContext(Protocol):
    """Protocol defining the interface for plot contexts."""

    data: pd.DataFrame | None
    x: str
    y: str
    hue: str | None
    ax: Axes | None
    title: str | None
    layers: list[RenderableLayer]

    def get_params_for_layer(self, layer_type: type[RenderableLayer]) -> ParamModel:
        """
        Get parameters appropriate for a specific layer type.

        Parameters
        ----------
        layer_type : type[RenderableLayer]
            The type of layer to get parameters for

        Returns
        -------
        ParamModel
            The parameter model instance

        """
        ...

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
        ...
