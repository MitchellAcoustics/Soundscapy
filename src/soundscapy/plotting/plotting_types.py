"""Utility functions and constants for the soundscapy plotting module."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    TypeAlias,
)

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.typing import ColorType
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_snake

from soundscapy.plotting.defaults import (
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_SUBPLOTS_PARAMS,
)
from soundscapy.sspylogging import get_logger

logger = get_logger()

# Type aliases
SeabornPaletteType: TypeAlias = str | list | dict | Colormap

MplLegendLocType: TypeAlias = (
    Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ]
    | tuple[float, float]
)


class ParamModel(BaseModel):
    """
    Base model for parameter validation.

    This class provides the foundation for all parameter models with
    common configuration settings and utility methods. It also maintains
    the default parameter registry for creating parameter instances.
    """

    # Registry of default parameters
    _default_params: ClassVar[dict[str, dict[str, Any]]] = {
        "scatter": DEFAULT_SCATTER_PARAMS,
        "density": DEFAULT_DENSITY_PARAMS,
        "simple_density": DEFAULT_SIMPLE_DENSITY_PARAMS,
        "style": DEFAULT_STYLE_PARAMS,
        "subplots": DEFAULT_SUBPLOTS_PARAMS,
    }

    # Registry for parameter model classes
    _param_registry: ClassVar[dict[str, type[ParamModel]]] = {}

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        arbitrary_types_allowed=True,  # Allow complex matplotlib types
        validate_assignment=True,  # Validate when attributes are set
        alias_generator=to_snake,  # Use snake_case for aliases
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses in the registry."""
        super().__init_subclass__(**kwargs)
        cls.__register_class__()

    @classmethod
    def __register_class__(cls) -> None:
        """Register this class in the parameter registry."""
        # Skip registration for the base class
        if cls is ParamModel:
            return

        # Extract class name and register (e.g., 'ScatterParams' -> 'scatter')
        class_name = cls.__name__
        if class_name.endswith("Params"):
            param_type = to_snake(class_name[:-6])
            cls._param_registry[param_type] = cls

    @classmethod
    def create(cls, param_type: str, **kwargs: Any) -> ParamModel:
        """
        Create a parameter instance of the specified type.

        Parameters
        ----------
        param_type : str
            The type of parameters to create ('scatter', 'density', etc.)
        **kwargs : Any
            Initial parameter values

        Returns
        -------
        ParamModel
            Instance of the appropriate parameter class

        Raises
        ------
        ValueError
            If the parameter type is unknown

        """
        # Get the parameter model class
        model_class = cls.get_param_class(param_type)

        # Get default parameter values
        default_params = cls._get_default_params(param_type)

        # Create instance with defaults and then update with kwargs
        return model_class(**default_params).update(**kwargs)

    @classmethod
    def get_param_class(cls, param_type: str) -> type[ParamModel]:
        """
        Get the parameter model class for a parameter type.

        Parameters
        ----------
        param_type : str
            The type of parameters to get

        Returns
        -------
        Type[ParamModel]
            The parameter model class

        Raises
        ------
        ValueError
            If the parameter type is unknown

        """
        if param_type not in cls._param_registry:
            msg = f"Unknown parameter type: {param_type}"
            raise ValueError(msg)

        return cls._param_registry[param_type]

    @classmethod
    def _get_default_params(cls, param_type: str) -> dict[str, Any]:
        """
        Get the default parameters for a parameter type.

        Parameters
        ----------
        param_type : str
            The type of parameters to get defaults for

        Returns
        -------
        Dict[str, Any]
            Default parameters for the specified type

        Raises
        ------
        ValueError
            If the parameter type is unknown

        """
        if param_type in cls._default_params:
            return copy.deepcopy(cls._default_params[param_type])

        msg = f"Unknown parameter type: {param_type}"
        raise ValueError(msg)

    def update(self, **kwargs: Any) -> Self:
        """
        Update parameters with new values.

        Parameters
        ----------
        **kwargs : Any
            New parameter values

        Returns
        -------
        Self
            The updated parameter instance (for chaining)

        """
        # Filter out None values to avoid overriding defaults with None
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        for key, value in filtered_kwargs.items():
            try:
                setattr(self, key, value)
            except ValueError as e:
                logger.warning(f"Invalid value for {key}: {e}")

        return self

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value with a default fallback.

        Parameters
        ----------
        key : str
            Name of the parameter
        default : Any, optional
            Default value if parameter doesn't exist

        Returns
        -------
        Any
            Parameter value or default

        """
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Get a parameter value using dictionary-style access.

        Parameters
        ----------
        key : str
            Name of the parameter

        Returns
        -------
        Any
            Parameter value

        Raises
        ------
        KeyError
            If the parameter doesn't exist

        """
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e)) from e

    def as_dict(self) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter values

        """
        return self.model_dump(exclude_none=True)


class SeabornParams(ParamModel):
    """Base parameters for seaborn plotting functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = None
    y: str | np.ndarray | pd.Series | None = None
    hue: str | np.ndarray | pd.Series | None = None
    size: str | np.ndarray | pd.Series | None = None
    style: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = None
    hue_order: Iterable[str] | None = None
    hue_norm: tuple | Normalize | None = None
    alpha: float | None = None
    legend: Literal["auto", "brief", "full", False] | None = None
    # matplotlib kwargs
    color: ColorType | None = None
    label: str | None = None
    zorder: float | None = None


class ScatterParams(SeabornParams):
    """Parameters for scatter plot functions."""

    sizes: list | dict | tuple | None = None
    size_order: list | None = None
    size_norm: tuple | Normalize | None = None
    markers: bool | list | dict | None = None
    style_order: list | None = None
    marker: str | None = None
    linewidth: float | None = None
    s: float | None = None


class DensityParams(SeabornParams):
    """Parameters for density plot functions."""

    weights: str | np.ndarray | pd.Series | None = None
    fill: bool | None = None
    multiple: Literal["layer", "stack", "fill"] | None = None
    common_norm: bool | None = None
    common_grid: bool | None = None
    cumulative: bool | None = None
    bw_method: Literal["scott", "silverman"] | float | Callable | None = None
    bw_adjust: float | None = None
    warn_singular: bool | None = None
    log_scale: bool | tuple[bool, bool] | float | tuple[float, float] | None = None
    levels: int | Iterable[float] | None = None
    thresh: float | None = None
    gridsize: int | None = None
    cut: float | None = None
    clip: tuple[tuple[float, float], tuple[float, float]] | None = None
    cbar: bool | None = None
    cbar_ax: Axes | None = None
    cbar_kws: dict[str, Any] | None = None
    include_outline: bool | None = None


class SimpleDensityParams(DensityParams):
    """Parameters for simple density plots."""

    # Override default levels for simple density plots
    levels: int | Iterable[float] | None = 10
    alpha: float | None = 0.5


class JointPlotParams(ParamModel):
    """Parameters for jointplot functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = None
    y: str | np.ndarray | pd.Series | None = None
    height: float | None = None
    ratio: float | None = None
    space: float | None = None
    dropna: bool | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    marginal_ticks: bool | None = None
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = None
    hue_order: Iterable[str] | None = None
    hue_norm: tuple | Normalize | None = None


class StyleParams(ParamModel):
    """
    Configuration options for styling circumplex plots.

    Attributes
    ----------
    xlim : tuple[float, float] | None
        X-axis limits.
    ylim : tuple[float, float] | None
        Y-axis limits.
    xlabel : str | None
        X-axis label. If None, use column name.
    ylabel : str | None
        Y-axis label. If None, use column name.
    diag_lines_zorder : int | None
        Z-order for diagonal lines.
    diag_labels_zorder : int | None
        Z-order for diagonal labels.
    prim_lines_zorder : int | None
        Z-order for primary lines.
    data_zorder : int | None
        Z-order for plotted data.
    legend_loc : MplLegendLocType | None
        Legend location.
    linewidth : float | None
        Line width for plot elements.
    primary_lines : bool | None
        Whether to show primary axes lines.
    diagonal_lines : bool | None
        Whether to show diagonal lines.
    title_fontsize : int | None
        Font size for the title.
    prim_ax_fontdict : dict[str, Any] | None
        Font settings for primary axes labels.

    """

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    # if None: use col name (default), if False: no label
    xlabel: str | Literal[False] | None = None
    ylabel: str | Literal[False] | None = None
    diag_lines_zorder: int | None = None
    diag_labels_zorder: int | None = None
    prim_lines_zorder: int | None = None
    data_zorder: int | None = None
    legend_loc: MplLegendLocType | Literal[False] | None = None
    linewidth: float | None = None
    primary_lines: bool | None = None
    diagonal_lines: bool | None = None
    title_fontsize: int | None = None
    prim_ax_fontdict: dict[str, Any] | None = None


class SubplotsParams(ParamModel):
    """Parameters for subplot configuration."""

    sharex: bool | None = None
    sharey: bool | None = None
    squeeze: bool | None = None
    width_ratios: Sequence[float] | None = None
    height_ratios: Sequence[float] | None = None
    subplot_kw: dict[str, Any] | None = None
    gridspec_kw: dict[str, Any] | None = None
