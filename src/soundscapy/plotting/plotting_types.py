"""Utility functions and constants for the soundscapy plotting module."""

# ruff: noqa: ANN401, TC002, TC003
from __future__ import annotations

from collections.abc import Iterable
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    TypeAlias,
)

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.typing import ColorType
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_snake

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
    common configuration settings and utility methods.
    """

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

        # Create instance and update with kwargs
        return model_class().update(**kwargs)

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

    # @validate_call
    def update(
        self,
        *,
        extra: Literal["allow", "forbid", "ignore"] = "allow",
        na_rm: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Update parameters with new values.

        Parameters
        ----------
        extra : Literal["allow", "forbid", "ignore"], optional
            Controls how extra fields are handled. Default is "allow".
        **kwargs : Any
            New parameter values

        Returns
        -------
        Self
            The updated parameter instance (for chaining)

        """
        if extra == "forbid":
            # Forbid extra fields
            unknown_keys = set(kwargs) - set(self.model_fields)
            if unknown_keys:
                msg = f"Unknown parameters: {unknown_keys}"
                raise ValueError(msg)
        elif extra == "ignore":
            # Ignore extra fields
            kwargs = {k: v for k, v in kwargs.items() if k in self.model_fields}
        elif extra != "allow":
            msg = f"Invalid value for 'extra': {extra}"
            raise ValueError(msg)
        # Remove None values if na_rm is True
        if na_rm:
            # Filter out None values to avoid overriding defaults with None
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except ValueError as e:  # noqa: PERF203
                logger.warning("Invalid value for %s: %s", key, e)

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

    def as_dict(self, **kwargs) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter values

        """
        return self.model_dump(**kwargs)

    def get_changed_params(self) -> dict[str, Any]:
        """
        Get parameters that have been changed from their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary of changed parameters

        """
        return self.model_dump(exclude_unset=True)

    def get_multiple(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple parameters as a dictionary.

        Parameters
        ----------
        keys : list[str]
            List of parameter names

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter values

        """
        return {key: getattr(self, key) for key in keys if hasattr(self, key)}

    def pop(self, key: str) -> Any:
        """
        Remove a parameter and return its value.

        Parameters
        ----------
        key : str
            Name of the parameter

        Returns
        -------
        Any
            Value of the removed parameter

        Raises
        ------
        KeyError
            If the parameter doesn't exist

        """
        if not hasattr(self, key):
            msg = f"Parameter '{key}' does not exist."
            raise KeyError(msg)
        value = getattr(self, key)
        delattr(self, key)
        return value

    def drop(self, keys: str | Iterable[str], *, ignore_missing: bool = True) -> Self:
        """
        Remove a parameter without returning its value.

        Parameters
        ----------
        keys : str | Iterable[str]
            Name of the parameter or list of parameters

        Raises
        ------
        KeyError
            If the parameter doesn't exist

        """
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if not hasattr(self, k):
                if ignore_missing:
                    continue
                msg = f"Parameter '{k}' does not exist."
                raise KeyError(msg)
            delattr(self, k)
        return self

    @property
    def field_names(self) -> list[str]:
        """
        Get the names of all fields in the model.

        Returns
        -------
        List[str]
            List of field names

        """
        return list(self.model_fields.keys())


class SeabornParams(ParamModel):
    """Base parameters for seaborn plotting functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = "ISOPleasant"
    y: str | np.ndarray | pd.Series | None = "ISOEventful"
    palette: SeabornPaletteType | None = "colorblind"
    alpha: float = 0.8
    color: ColorType | None = "#0173B2"  # First color from colorblind palette
    zorder: float = 3
    hue: str | np.ndarray | pd.Series | None = None

    def crosscheck_palette_hue(self) -> None:
        """
        Check if the palette is valid for the given hue.

        Parameters
        ----------
        palette : SeabornPaletteType
            The color palette to use.
        hue : str | np.ndarray | pd.Series | None
            The column name for color encoding.

        Returns
        -------
        SeabornPaletteType
            The validated color palette.

        Raises
        ------
        ValueError
            If the palette is not valid for the given hue.

        """
        self.palette = self.palette if self.hue is not None else None

    def as_seaborn_kwargs(self) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        return self.as_dict()


class ScatterParams(SeabornParams):
    """Parameters for scatter plot functions."""

    s: float | None = 20  # DEFAULT_POINT_SIZE


class DensityParams(SeabornParams):
    """Parameters for density plot functions."""

    fill: bool = True
    common_norm: bool = False
    common_grid: bool = False
    bw_adjust: float = 1.2  # DEFAULT_BW_ADJUST
    levels: int | tuple[float, ...] = 10
    clip: tuple[tuple[float, float], tuple[float, float]] | None = (
        (-1, 1),
        (-1, 1),
    )  # DEFAULT_XLIM, DEFAULT_YLIM

    def as_seaborn_kwargs(self) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        # None to drop yet
        return super().as_seaborn_kwargs()

    def to_outline(
        self,
        *,
        alpha: float = 1,
        fill: bool = False,
    ) -> Self:
        """
        Get parameters for the outline of density plots.

        Parameters
        ----------
        levels : int | tuple[float, float], optional
            The levels for the outline. Default is (0, 0.5).
        alpha : float, optional
            The alpha value for the outline. Default is 1.

        Returns
        -------
        dict[str, Any]
            The parameters for the outline of density plots.

        """
        # Set levels and alpha for simple density plots
        return self.model_copy(update={"alpha": alpha, "fill": fill, "legend": False})


class SimpleDensityParams(DensityParams):
    """Parameters for simple density plots."""

    # Override default levels for simple density plots
    thresh: float = 0.5
    levels: int | tuple[float, ...] = 2
    alpha: float = 0.5


class SPISeabornParams(SeabornParams):
    """Base parameters for seaborn plotting functions for SPI data."""

    color: ColorType | None = "red"
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = None
    label: str = "SPI"
    n: int = 1000
    show_score: Literal["on axis", "under title"] = "under title"
    axis_text_kw: dict[str, Any] | None = None

    def as_seaborn_kwargs(self) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        self.drop(["n", "show_score", "axis_text_kw"])
        return self.as_dict()


class SPISimpleDensityParams(SPISeabornParams, SimpleDensityParams):
    """Parameters for simple density plotting of SPI data."""

    # include_outline: bool = True
    #
    # def as_seaborn_kwargs(self) -> dict[str, Any]:
    #     self.drop(["include_outline"])
    #     return super().as_seaborn_kwargs()


class JointPlotParams(ParamModel):
    """Parameters for jointplot functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = "ISOPleasant"
    y: str | np.ndarray | pd.Series | None = "ISOEventful"
    xlim: tuple[float, float] | None = (-1, 1)  # DEFAULT_XLIM
    ylim: tuple[float, float] | None = (-1, 1)
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = "colorblind"
    marginal_ticks: bool | None = None


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

    xlim: tuple[float, float] = (-1, 1)  # DEFAULT_XLIM
    ylim: tuple[float, float] = (-1, 1)  # DEFAULT_YLIM
    # if None: use col name (default), if False: no label
    xlabel: str | Literal[False] | None = r"$P_{ISO}$"
    ylabel: str | Literal[False] | None = r"$E_{ISO}$"
    diag_lines_zorder: int = 1
    diag_labels_zorder: int = 4
    prim_lines_zorder: int = 2
    data_zorder: int = 3
    legend_loc: MplLegendLocType | Literal[False] = "best"
    linewidth: float = 1.5
    primary_lines: bool = True
    diagonal_lines: bool = False
    title_fontsize: int = 14
    # This should be properly defined with its own Pydantic model in a future iteration
    prim_ax_fontdict: dict[str, Any] = {
        "family": "sans-serif",
        "fontstyle": "normal",
        "fontsize": "large",
        "fontweight": "medium",
        "parse_math": True,
        "c": "black",
        "alpha": 1,
    }


class SubplotsParams(ParamModel):
    """Parameters for subplot configuration."""

    nrows: int = 1
    ncols: int = 1
    figsize: tuple[float, float] = (5, 5)
    sharex: bool | Literal["none", "all", "row", "col"] = True
    sharey: bool | Literal["none", "all", "row", "col"] = True

    @property
    def n_subplots(self) -> int:
        """
        Calculate the total number of subplots.

        Returns
        -------
        int
            Total number of subplots.

        """
        return self.nrows * self.ncols

    def get_plt_subplot_args(self) -> dict[str, Any]:
        """
        Pass matplotlib subplot arguments to a plt.subplots call.

        Parameters
        ----------
        ax : Any
            Matplotlib Axes object.

        Returns
        -------
        dict[str, Any]
            Dictionary of subplot parameters.

        """
        return {
            "nrows": self.nrows,
            "ncols": self.ncols,
            "figsize": self.figsize,
            "sharex": self.sharex,
            "sharey": self.sharey,
        }
