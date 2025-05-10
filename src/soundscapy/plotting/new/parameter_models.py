"""
Parameter models for the plotting module.

This module provides Pydantic models for parameter validation and management.
These models replace the dictionary-based defaults in the original implementation
and provide a single source of truth for parameter values with proper type validation.
"""

from __future__ import annotations

from typing import Any, Literal, Self, TypeAlias

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.typing import ColorType
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_snake

from soundscapy.plotting.new.constants import (
    DEFAULT_BW_ADJUST,
    DEFAULT_COLOR,
    DEFAULT_FONTDICT,
    DEFAULT_POINT_SIZE,
    DEFAULT_SPI_TEXT_KWARGS,
    DEFAULT_XCOL,
    DEFAULT_XLIM,
    DEFAULT_YCOL,
    DEFAULT_YLIM,
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


class BaseParams(BaseModel):
    """
    Base model for all parameter types.

    This class provides common configuration settings and utility methods
    for all parameter models.
    """

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
        na_rm: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Update parameters with new values.

        Parameters
        ----------
        extra : Literal["allow", "forbid", "ignore"], optional
            Controls how extra fields are handled. Default is "allow".
        na_rm : bool, optional
            Whether to remove None values. Default is True.
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


class AxisParams(BaseParams):
    """Parameters for axis configuration."""

    x: str = DEFAULT_XCOL
    y: str = DEFAULT_YCOL
    xlim: tuple[float, float] = DEFAULT_XLIM
    ylim: tuple[float, float] = DEFAULT_YLIM
    xlabel: str | Literal[False] | None = r"$P_{ISO}$"
    ylabel: str | Literal[False] | None = r"$E_{ISO}$"


class SeabornParams(BaseParams):
    """Base parameters for seaborn plotting functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = DEFAULT_XCOL
    y: str | np.ndarray | pd.Series | None = DEFAULT_YCOL
    palette: SeabornPaletteType | None = "colorblind"
    alpha: float = 0.8
    color: ColorType | None = DEFAULT_COLOR
    zorder: float = 3
    hue: str | np.ndarray | pd.Series | None = None

    def crosscheck_palette_hue(self) -> None:
        """
        Check if the palette is valid for the given hue.

        Sets palette to None if hue is None.
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

    s: float | None = DEFAULT_POINT_SIZE


class DensityParams(SeabornParams):
    """Parameters for density plot functions."""

    fill: bool = True
    common_norm: bool = False
    common_grid: bool = False
    bw_adjust: float = DEFAULT_BW_ADJUST
    levels: int | tuple[float, ...] = 10
    clip: tuple[tuple[float, float], tuple[float, float]] | None = (
        DEFAULT_XLIM,
        DEFAULT_YLIM,
    )

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
        alpha : float, optional
            The alpha value for the outline. Default is 1.
        fill : bool, optional
            Whether to fill the outline. Default is False.

        Returns
        -------
        Self
            The parameters for the outline of density plots.

        """
        return self.model_copy(update={"alpha": alpha, "fill": fill, "legend": False})


class SimpleDensityParams(DensityParams):
    """Parameters for simple density plots."""

    # Override default levels for simple density plots
    thresh: float = 0.5
    levels: int | tuple[float, ...] = 2
    alpha: float = 0.5


class SPISeabornParams(SeabornParams):
    """Base parameters for seaborn plotting functions for SPI custom_data."""

    color: ColorType | None = "red"
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = None
    label: str = "SPI"
    n: int = 1000
    show_score: Literal["on axis", "under title"] = "under title"
    axis_text_kw: dict[str, Any] | None = Field(
        default_factory=lambda: DEFAULT_SPI_TEXT_KWARGS.copy()
    )
    # msn_params is not directly stored in the parameter model
    # but is used by SPILayer to generate SPI custom_data
    # It should be passed to the layer constructor, not to the parameter model

    def as_seaborn_kwargs(self) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        new = self.model_copy()
        # Drop SPI-specific parameters
        return new.as_dict(exclude={"n", "show_score", "axis_text_kw"})


class SPISimpleDensityParams(SimpleDensityParams):
    """Parameters for simple density plotting of SPI custom_data."""

    color: ColorType | None = "red"
    label: str = "SPI"
    n: int = 1000
    show_score: Literal["on axis", "under title"] = "under title"
    axis_text_kw: dict[str, Any] | None = Field(
        default_factory=lambda: DEFAULT_SPI_TEXT_KWARGS.copy()
    )

    def as_seaborn_kwargs(self) -> dict[str, Any]:
        """
        Convert parameters to kwargs compatible with seaborn functions.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter values suitable for seaborn plotting functions.

        """
        return self.as_dict(exclude={"n", "show_score", "axis_text_kw"})


class JointPlotParams(BaseParams):
    """Parameters for jointplot functions."""

    data: pd.DataFrame | None = None
    x: str | np.ndarray | pd.Series | None = DEFAULT_XCOL
    y: str | np.ndarray | pd.Series | None = DEFAULT_YCOL
    xlim: tuple[float, float] | None = DEFAULT_XLIM
    ylim: tuple[float, float] | None = DEFAULT_YLIM
    hue: str | np.ndarray | pd.Series | None = None
    palette: SeabornPaletteType | None = "colorblind"
    marginal_ticks: bool | None = None


class StyleParams(BaseParams):
    """
    Configuration options for style_mgr circumplex plots.
    """

    xlim: tuple[float, float] = DEFAULT_XLIM
    ylim: tuple[float, float] = DEFAULT_YLIM
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
    prim_ax_fontdict: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_FONTDICT.copy()
    )


class SubplotsParams(BaseParams):
    """Parameters for subplot configuration."""

    nrows: int = 1
    ncols: int = 1
    figsize: tuple[float, float] = (5, 5)
    sharex: bool | Literal["none", "all", "row", "col"] = True
    sharey: bool | Literal["none", "all", "row", "col"] = True
    subplot_by: str | None = None
    n_subplots_by: int = -1
    auto_allocate_axes: bool = False
    adjust_figsize: bool = True

    @property
    def n_subplots(self) -> int:
        """
        Calculate the total number of subplot_mgr.

        Returns
        -------
        int
            Total number of subplot_mgr.

        """
        return self.nrows * self.ncols

    def as_plt_subplots_args(self) -> dict[str, Any]:
        """
        Pass matplotlib subplot arguments to a plt.subplot_mgr call.

        Returns
        -------
        dict[str, Any]
            Dictionary of subplot parameters.

        """
        return self.as_dict(
            exclude={
                "subplot_by",
                "n_subplots_by",
                "auto_allocate_axes",
                "adjust_figsize",
            }
        )
