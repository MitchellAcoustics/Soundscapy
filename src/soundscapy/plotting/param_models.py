"""Utility functions and constants for the soundscapy plotting module."""

# ruff: noqa: ANN401, TC002, TC003
from __future__ import annotations

from collections.abc import Iterable
from typing import (
    Any,
    ClassVar,
    Literal,
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

param_model_config = ConfigDict(
    extra="allow",  # Allow extra fields for flexibility
    arbitrary_types_allowed=True,  # Allow complex matplotlib types
    validate_assignment=True,  # Validate when attributes are set
    alias_generator=to_snake,  # Use snake_case for aliases
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

    def update(
        self,
        *,
        extra: Literal["allow", "forbid", "ignore"] = "allow",
        ignore_null: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Update the attributes of the instance based on the provided parameters.

        The method allows for managing extra fields, removing `None` values, and
        setting instance attributes dynamically based on the input `kwargs`.
        Behavior can be adjusted via the `extra` and `na_rm` parameters.

        Parameters
        ----------
        extra : {'allow', 'forbid', 'ignore'}, default='allow'
            Determines how to handle extra fields in `kwargs`:
            - 'allow': All fields in `kwargs` are allowed and processed.
            - 'forbid': Raises a `ValueError` if any key in `kwargs` is not a valid
              field of the model.
            - 'ignore': Ignores fields in `kwargs` that are not valid fields of the
              model.
        ignore_null : bool, default=True
            If True, removes `None` values from `kwargs` to avoid overwriting default
            instance attributes with `None`.
        **kwargs : Any
            Arbitrary keyword arguments representing field names and values to be
            updated for the instance.

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
        # Remove None values if ignore_null is True
        if ignore_null:
            # Filter out None values to avoid overriding defaults with None
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Use proper Pydantic model update method
        self.model_validate(kwargs)

        # Update using model_copy to ensure proper field validation
        updated_model = self.model_copy(update=kwargs)
        for field_name in kwargs:
            if field_name in self.model_fields or extra == "allow":
                try:
                    setattr(self, field_name, getattr(updated_model, field_name))
                except ValueError as e:
                    logger.warning("Invalid value for %s: %s", field_name, e)

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
        model_dict = self.model_dump()
        return model_dict.get(key, default)

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
        if key in self.model_fields:
            return self.model_dump().get(key)
        msg = f"Parameter '{key}' does not exist."
        raise KeyError(msg)

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
        return self.model_dump(exclude_defaults=True)

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
        model_dict = self.model_dump()
        return {key: model_dict.get(key) for key in keys if key in model_dict}

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
        model_dict = self.model_dump()
        if key not in model_dict:
            msg = f"Parameter '{key}' does not exist."
            raise KeyError(msg)
        value = model_dict[key]

        # Create a new dict without the popped key
        new_data = {k: v for k, v in model_dict.items() if k != key}

        # Clear all current fields and update with new data
        for k in list(model_dict.keys()):
            if hasattr(self, k):
                object.__delattr__(self, k)

        # Update with new data (excluding the popped key)
        updated_model = self.model_copy(update=new_data)
        for field_name in new_data:
            if field_name in self.model_fields or field_name in new_data:
                setattr(self, field_name, getattr(updated_model, field_name))

        return value

    def drop(self, keys: str | Iterable[str], *, ignore_missing: bool = True) -> None:
        """
        Remove a parameter without returning its value.

        Parameters
        ----------
        keys : str | Iterable[str]
            Name of the parameter or list of parameters
        ignore_missing : bool, default=True
            If True, ignore missing keys. If False, raise KeyError for missing keys.

        Raises
        ------
        KeyError
            If the parameter doesn't exist and ignore_missing is False

        """
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            _ = self.pop(key)

    @property
    def defined_field_names(self) -> list[str]:
        """
        Get the names of all fields defined for the model.

        Returns
        -------
        List[str]
            List of field names

        """
        return list(self.model_fields.keys())

    @property
    def current_field_names(self) -> list[str]:
        """
        Retrieves the current field names.

        This property method fetches and returns the current set of field names
        associated with the instance. It provides a read-only interface to
        access field names as computed or stored in the object.

        Returns
        -------
        list[str]
            A list of strings where each string represents a field name.

        """
        return list(self.model_dump().keys())


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
        new = self.model_copy()
        return new.as_dict()


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
    # NOTE: Would like to add include_outline here, but would need to refactor
    #       throughout the code.

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
    ) -> DensityParams:
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
    axis_text_kw: dict[str, Any] | None = {
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

    def as_seaborn_kwargs(self) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        new = self.model_copy()
        new.drop(["n", "show_score", "axis_text_kw"])
        return new.as_dict()


class SPISimpleDensityParams(SPISeabornParams, SimpleDensityParams):
    """Parameters for simple density plotting of SPI data."""


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
    subplot_by: str | None = None
    n_subplots_by: int = -1
    """"The number of subplots allocated for each subplot_by category."""

    auto_allocate_axes: bool = False
    adjust_figsize: bool = True

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
        new = self.model_copy()
        new.drop(
            ["subplot_by", "n_subplots_by", "auto_allocate_axes", "adjust_figsize"]
        )
        return new.as_dict()
