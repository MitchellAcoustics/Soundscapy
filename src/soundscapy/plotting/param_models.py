from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable
from dataclasses import field, fields
from typing import Any, ClassVar, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.typing import ColorType
from pydantic import validate_call
from pydantic.dataclasses import dataclass as pydantic_dataclass

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

T = TypeVar("T", bound="ParamModel")


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class ParamModel:
    """
    Base model for parameter validation using dataclasses.

    This class provides the foundation for all parameter models with
    common configuration settings and utility methods.
    """

    # Registry for parameter model classes
    _param_registry: ClassVar[dict[str, type[ParamModel]]] = {}

    # Dictionary to store extra fields

    def __post_init__(self) -> None:
        """Process extra fields after initialization."""
        # Move any extra fields to _extra_fields
        for key, value in list(self.__dict__.items()):
            if key not in self.defined_field_names:
                setattr(self, key, value)

    @validate_call
    def update(
        self,
        *,
        extra: Literal["allow", "forbid", "ignore"] = "allow",
        ignore_null: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Update the attributes of the instance based on the provided parameters.

        Parameters
        ----------
        extra : {'allow', 'forbid', 'ignore'}, default='allow'
            Determines how to handle extra fields in `kwargs`.
        ignore_null : bool, default=True
            If True, removes `None` values from `kwargs`.
        **kwargs : Any
            Field names and values to be updated.

        """
        # Create a copy of kwargs to avoid modifying it during iteration
        update_kwargs = kwargs.copy()

        if extra == "forbid":
            # Forbid extra fields
            unknown_keys = set(update_kwargs) - set(self.defined_field_names)
            if unknown_keys:
                msg = f"Unknown parameters: {unknown_keys}"
                raise ValueError(msg)
        elif extra == "ignore":
            # Ignore extra fields
            update_kwargs = {
                k: v for k, v in update_kwargs.items() if k in self.defined_field_names
            }

        # Remove None values if ignore_null is True
        if ignore_null:
            update_kwargs = {k: v for k, v in update_kwargs.items() if v is not None}

        # Update fields
        for key, value in update_kwargs.items():
            if key in self.defined_field_names or extra == "allow":
                setattr(self, key, value)

    def get_defaults(self) -> dict[str, Any]:
        return {
            fld.name: fld.default for fld in fields(self) if fld.name != "_extra_fields"
        }

    @property
    def defaults(self):
        return self.get_defaults()

    @property
    def _extra_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in self.get_defaults()}

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
        if key in self.current_field_names:
            return getattr(self, key)
        return default

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
        if key in self.current_field_names:
            return getattr(self, key)
        msg = f"Parameter '{key}' does not exist."
        raise KeyError(msg)

    def as_dict(self, drop: list[str] | None = None) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter values

        """
        dictionary = self.__dict__.copy()
        if drop is not None:
            for key in drop:
                dictionary.pop(key, None)
        return dictionary

    def copy(self):
        return copy.deepcopy(self)

    def model_copy(self):
        warnings.warn(
            "model_copy is deprecated. Use copy instead."
            "Kept only for backwards compatibility with Pydantic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.copy()

    def get_changed_params(self) -> dict[str, Any]:
        """
        Get parameters that have been changed from their defaults.

        This method compares the current parameter values against the default values
        and returns a dictionary containing only the parameters that differ from
        their defaults.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameters that differ from defaults, with keys as parameter names
            and values as their current values (not default values).

        """
        default_values = self.get_defaults()
        current_values = self.as_dict()

        # Use dictionary comprehension to identify and collect changed parameters
        # Return the current values (not default values) for the changed parameters
        return {
            param_name: current_values[param_name]
            for param_name, default_value in default_values.items()
            if current_values[param_name] != default_value
        }

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
        return {key: self[key] for key in keys if key in self.current_field_names}

    def pop(self, key: str) -> Any:
        """
        Remove a parameter and return its value.

        For fields defined in the model, the value is reset to its default.

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
        if key in self.defaults:
            value = getattr(self, key)
            setattr(self, key, self.defaults[key])  # Reset to default/None
            return value
        if key in self.current_field_names:
            value = self.get(key)
            delattr(self, key)
            return value
        msg = f"Parameter '{key}' does not exist."
        raise KeyError(msg)

    def drop(self, keys: str | Iterable[str], *, ignore_missing: bool = True) -> None:
        """
        Remove a parameter without returning its value.

        Parameters
        ----------
        keys : str | Iterable[str]
            Name of the parameter or list of parameters
        ignore_missing : bool, default=True
            If True, ignore missing keys. If False, raise KeyError for missing keys.

        """
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            try:
                delattr(self, key)
            except (KeyError, AttributeError) as e:  # noqa: PERF203
                if not ignore_missing:
                    if isinstance(e, AttributeError):
                        msg = f"Parameter '{key}' does not exist."
                        raise KeyError(msg) from e
                    raise

    @property
    def defined_field_names(self) -> list[str]:
        """
        Get the names of all fields defined for the model.

        Returns
        -------
        List[str]
            List of field names

        """
        return list(self.defaults.keys())

    @property
    def current_field_names(self) -> list[str]:
        """
        Get the names of all current fields.

        Returns
        -------
        List[str]
            List of field names

        """
        return list(self.as_dict().keys())


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
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

        This method ensures that palette is only used when hue is provided.
        """
        self.palette = self.palette if self.hue is not None else None

    def as_seaborn_kwargs(self, drop: list[str] | None = None) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        return self.as_dict(drop=drop)


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class ScatterParams(SeabornParams):
    """Parameters for scatter plot functions."""

    s: float | None = 20  # DEFAULT_POINT_SIZE


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
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
    # NOTE: Would like to add include_outline here, but would need to refactor
    #       throughout the code.

    def as_seaborn_kwargs(self, drop: list[str] | None = None) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        # None to drop yet
        if drop is None:
            drop = []
        # Add common parameters to drop list
        drop.extend(["incl_outline"])
        return super().as_seaborn_kwargs(drop=drop)

    def to_outline(self, *, alpha: float = 1, fill: bool = False) -> DensityParams:
        """
        Convert to outline parameters.

        Parameters
        ----------
        alpha : float, default=1
            Alpha value for the outline.
        fill : bool, default=False
            Whether to fill the outline.

        Returns
        -------
        DensityParams
            New instance with outline parameters.

        """
        # Create a copy of the parameters
        params_dict = self.as_dict()

        # Update parameters for outline
        params_dict.update(alpha=alpha, fill=fill, legend=False)

        # Create a new instance with the updated parameters
        return DensityParams(**params_dict)


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class SimpleDensityParams(DensityParams):
    """Parameters for simple density plot functions."""

    # Override default levels for simple density plots
    thresh: float = 0.5
    levels: int | tuple[float, ...] = 2
    alpha: float = 0.5


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class SPISeabornParams(SeabornParams):
    """Base parameters for SPI seaborn plotting functions."""

    color: ColorType | None = "red"
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = None
    label: str = "SPI"
    n: int = 1000
    show_score: Literal["on axis", "under title"] = "under title"
    axis_text_kw: dict[str, Any] | None = field(
        default_factory=lambda: (
            {
                "x": 0,
                "y": -0.85,
                "fontsize": 10,
                "bbox": {
                    "facecolor": "white",
                    "edgecolor": "black",
                    "boxstyle": "round,pad=0.3",
                },
                "ha": "center",
                "va": "center",
            }
        )
    )

    def as_seaborn_kwargs(self, drop: list[str] | None = None) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        droplist = ["n", "show_score", "axis_text_kw"]
        if isinstance(drop, list):
            droplist.extend(drop)
        return self.as_dict(drop=droplist)


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class SPISimpleDensityParams(SPISeabornParams, SimpleDensityParams):
    """Parameters for SPI simple density plot functions."""


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class JointPlotParams(ParamModel):
    """Parameters for joint plot functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = "ISOPleasant"
    y: str | np.ndarray | pd.Series | None = "ISOEventful"
    xlim: tuple[float, float] | None = (-1, 1)  # DEFAULT_XLIM
    ylim: tuple[float, float] | None = (-1, 1)
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = "colorblind"
    marginal_ticks: bool | None = None


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class StyleParams(ParamModel):
    """Parameters for plot styling."""

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
    prim_ax_fontdict: dict[str, Any] = field(
        default_factory=lambda: (
            {
                "family": "sans-serif",
                "fontstyle": "normal",
                "fontsize": "large",
                "fontweight": "medium",
                "parse_math": True,
                "c": "black",
                "alpha": 1,
            }
        )
    )


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class SubplotsParams(ParamModel):
    """Parameters for subplots."""

    nrows: int = 1
    ncols: int = 1
    figsize: tuple[float, float] = (5, 5)
    sharex: bool | Literal["none", "all", "row", "col"] = True
    sharey: bool | Literal["none", "all", "row", "col"] = True
    subplot_by: str | None = None
    n_subplots_by: int = -1
    """"The number of subplots allocated for each subplot_by category."""

    auto_allocate_axes: bool = False
    adjust_figsize: bool = True

    @property
    def n_subplots(self) -> int:
        """
        Get the number of subplots.

        Returns
        -------
        int
            Number of subplots

        """
        return self.nrows * self.ncols

    def as_plt_subplots_args(self) -> dict[str, Any]:
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
        kwargs = self.as_dict()
        for key in [
            "subplot_by",
            "n_subplots_by",
            "auto_allocate_axes",
            "adjust_figsize",
        ]:
            kwargs.pop(key, None)
        return kwargs
