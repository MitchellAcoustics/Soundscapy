"""
Parameter management for plotting functions.

This module provides parameter validation, merging, and access for different types
of plot configurations. This ensures consistent parameter management and
improves type safety.

This module will be deprecated in a future version as the functionality has been
moved directly to the ParamModel class in plotting_types.py.
"""

from __future__ import annotations

from typing import Any

from soundscapy.plotting.plotting_types import ParamModel
from soundscapy.sspylogging import get_logger

logger = get_logger()


class PlotParams:
    """
    Compatibility wrapper for parameter management.

    This class provides a consistent interface for working with different parameter
    types, ensuring type safety and providing utilities for parameter validation,
    merging, and access.

    This class is now a simple wrapper around ParamModel and will be deprecated
    in a future version.

    Attributes
    ----------
    params_type : str
        The type of parameters being managed
    model : ParamModel
        The validated parameter model instance
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
        self.model = ParamModel.create(params_type, **initial_params)

    def update(self, **kwargs: Any) -> None:
        """
        Update parameters with new values.

        Parameters
        ----------
        **kwargs : Any
            New parameter values to set
        """
        self.model.update(**kwargs)

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
        return self.model.get(key, default)

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
        return self.model[key]

    def items(self):
        """Return parameter items for iteration."""
        return self.model.as_dict().items()

    def as_dict(self) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns
        -------
        dict
            The parameters as a dictionary
        """
        return self.model.as_dict()

    def reset(self) -> None:
        """Reset parameters to default values."""
        self.model = ParamModel.create(self.params_type)
