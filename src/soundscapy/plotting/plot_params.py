"""
Parameter management for plotting functions.

This module provides the PlotParams class to handle parameter validation, merging,
and access for different types of plot configurations. This ensures consistent
parameter management and improves type safety.
"""

from __future__ import annotations

import copy
from typing import Any, TypeVar, cast

from soundscapy.plotting.defaults import (
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_SUBPLOTS_PARAMS,
)
from soundscapy.plotting.plotting_types import (
    DensityParamTypes,
    ScatterParamTypes,
    StyleParamsTypes,
    SubplotsParamsTypes,
)

# Type variable for generic parameter typing
T = TypeVar("T", bound=dict[str, Any])


class PlotParams:
    """
    Class to manage parameter sets for different plot types.

    This class provides a consistent interface for working with different parameter
    types, ensuring type safety and providing utilities for parameter validation,
    merging, and access.

    Attributes
    ----------
    params : dict
        The current parameter values
    default_params : dict
        The default parameter values for this parameter type

    """

    def __init__(self, params_type: str, **initial_params: Any) -> None:
        """
        Initialize a PlotParams instance.

        Parameters
        ----------
        params_type : str
            The type of parameters to manage ('scatter', 'density', 'style', etc.)
        **initial_params :
            Initial parameter values to set

        """
        self.params_type = params_type
        self.default_params = self._get_default_params(params_type)
        self.params = copy.deepcopy(self.default_params)

        # Update with initial parameters
        if initial_params:
            self.update(**initial_params)

    def _get_default_params(self, params_type: str) -> dict[str, Any]:
        """
        Get the appropriate default parameters based on type.

        Parameters
        ----------
        params_type : str
            The type of parameters to get defaults for

        Returns
        -------
        dict
            The default parameters for the specified type

        Raises
        ------
        ValueError
            If an unknown parameter type is specified

        """
        if params_type == "scatter":
            return copy.deepcopy(DEFAULT_SCATTER_PARAMS)
        if params_type == "density":
            return copy.deepcopy(DEFAULT_DENSITY_PARAMS)
        if params_type == "simple_density":
            return copy.deepcopy(DEFAULT_SIMPLE_DENSITY_PARAMS)
        if params_type == "style":
            return copy.deepcopy(DEFAULT_STYLE_PARAMS)
        if params_type == "subplots":
            return copy.deepcopy(DEFAULT_SUBPLOTS_PARAMS)
        msg = f"Unknown parameter type: {params_type}"
        raise ValueError(msg)

    def update(self, **kwargs: Any) -> None:
        """
        Update parameters with new values.

        Parameters
        ----------
        **kwargs : Any
            New parameter values to set

        """
        # Filter out None values to avoid overriding defaults
        self.params.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value.

        Parameters
        ----------
        key : str
            The parameter name
        default : Any, optional
            Default value if parameter doesn't exist

        Returns
        -------
        Any
            The parameter value

        """
        return self.params.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Get a parameter value using dictionary-like access.

        Parameters
        ----------
        key : str
            The parameter name

        Returns
        -------
        Any
            The parameter value

        Raises
        ------
        KeyError
            If the parameter doesn't exist

        """
        return self.params[key]

    def items(self):
        """Return parameter items for iteration."""
        return self.params.items()

    def as_dict(self) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns
        -------
        dict
            The parameters as a dictionary

        """
        return self.params.copy()

    def as_typed_dict(
        self,
    ) -> ScatterParamTypes | DensityParamTypes | StyleParamsTypes | SubplotsParamsTypes:
        """
        Get parameters as a typed dictionary based on params_type.

        Returns
        -------
        TypedDict
            The parameters as the appropriate TypedDict

        """
        if self.params_type == "scatter":
            return cast(ScatterParamTypes, self.params.copy())
        if self.params_type in ("density", "simple_density"):
            return cast(DensityParamTypes, self.params.copy())
        if self.params_type == "style":
            return cast(StyleParamsTypes, self.params.copy())
        if self.params_type == "subplots":
            return cast(SubplotsParamsTypes, self.params.copy())
        msg = f"Unknown parameter type: {self.params_type}"
        raise ValueError(msg)

    def reset(self) -> None:
        """Reset parameters to default values."""
        self.params = copy.deepcopy(self.default_params)
