"""
Main module for creating circumplex plots using different backends.

Example:
-------
>>> from soundscapy import isd, surveys
>>> from soundscapy.plotting.circumplex_plot import CircumplexPlot
>>> df = isd.load()
>>> df = surveys.add_iso_coords(df)
>>> sub_df = isd.select_location_ids(df, ['CamdenTown', 'RegentsParkJapan'])
>>> cp = (
>>>     CircumplexPlot(data=sub_df, hue="SessionID")
>>>     .create_subplots(
>>>         subplot_by="LocationID",
>>>         auto_allocate_axes=True,
>>>         adjust_figsize=True
>>>     )
>>>     .add_scatter()
>>>     .add_simple_density(fill=False)
>>>     .apply_styling()
>>> )
>>> cp.show() # doctest: +SKIP

"""

import copy
import warnings
from collections.abc import Generator, Iterable
from typing import Any, Literal, Unpack

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

from soundscapy.plotting.plotting_types import (
    DensityParamTypes,
    JointPlotParamTypes,
    ScatterParamTypes,
    SeabornPaletteType,
    StyleParamsTypes,
    SubplotsParamsTypes,
)
from soundscapy.spi.msn import CentredParams, DirectParams, MultiSkewNorm, spi_score
from soundscapy.sspylogging import get_logger

logger = get_logger()

DEFAULT_TITLE = "Soundscape Density Plot"
DEFAULT_TITLE_FONTSIZE = 14
DEFAULT_XCOL = "ISOPleasant"
DEFAULT_YCOL = "ISOEventful"
DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)
DEFAULT_FIGSIZE = (5, 5)

DATA_ZORDER = 3
DIAG_LINES_ZORDER = 1
DIAG_LABELS_ZORDER = 4
PRIM_LINES_ZORDER = 2
DEFAULT_BW_ADJUST = 1.2

DEFAULT_PALETTE: SeabornPaletteType = "colorblind"
DEFAULT_COLOR: str = "#0173B2"  # First color from colorblind palette

COLORBLIND_CMAP: list[str] = sns.color_palette("colorblind", as_cmap=True)

RECOMMENDED_MIN_SAMPLES: int = 30

DEFAULT_SCATTER_PARAMS: ScatterParamTypes = ScatterParamTypes(
    x=DEFAULT_XCOL,
    y=DEFAULT_YCOL,
    alpha=0.8,
    palette=DEFAULT_PALETTE,
    color=DEFAULT_COLOR,
    legend="auto",
    zorder=DATA_ZORDER,
)

DEFAULT_DENSITY_PARAMS: DensityParamTypes = DensityParamTypes(
    x=DEFAULT_XCOL,
    y=DEFAULT_YCOL,
    alpha=0.8,
    fill=True,
    common_norm=False,
    common_grid=False,
    palette=DEFAULT_PALETTE,
    color=DEFAULT_COLOR,
    bw_adjust=DEFAULT_BW_ADJUST,
    zorder=DATA_ZORDER,
)

DEFAULT_STYLE_PARAMS: StyleParamsTypes = StyleParamsTypes(
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
    diag_lines_zorder=DIAG_LINES_ZORDER,
    diag_labels_zorder=DIAG_LABELS_ZORDER,
    prim_lines_zorder=PRIM_LINES_ZORDER,
    data_zorder=DATA_ZORDER,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
    show_labels=True,
    legend_location="best",
    legend=False,
    linewidth=1.5,
    primary_lines=True,
    diagonal_lines=False,
)

DEFAULT_SUBPLOTS_PARAMS: SubplotsParamsTypes = SubplotsParamsTypes(
    sharex=True,
    sharey=True,
)

DEFAULT_SPI_TEXT_KWARGS: dict[str, Any] = {
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


class CircumplexPlot:
    """
    A class for creating circumplex plots using different backends.

    This class provides methods for creating scatter plots and density plots
    based on the circumplex model of soundscape perception.

    Example:
    -------
    >>> from soundscapy import isd, surveys
    >>> df = isd.load()
    >>> df = surveys.add_iso_coords(df)
    >>> ct = isd.select_location_ids(df, ["CamdenTown", "RegentsParkJapan"])
    >>> cp = (
            CircumplexPlot(ct, hue="LocationID")
            .create_subplots()
            .add_scatter()
            .add_density()
            .apply_styling()
        )
    >>> cp.show() # doctest: +SKIP

    """

    # TODO(MitchellAcoustics): Implement jointplot method for Seaborn backend.  # noqa: E501, TD003

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        x: str = "ISOPleasant",
        y: str = "ISOEventful",
        title: str | None = "Soundscape Density Plot",
        hue: str | None = None,
        palette: str | None = "colorblind",
        figure: Figure | SubFigure | None = None,
        axes: Axes | np.ndarray | None = None,
    ) -> None:
        """
        Initialize a CircumplexPlot instance.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            The data to be plotted, by default None
        xcol : str, optional
            Name of the column to use for x-axis, by default "ISOPleasant"
        ycol : str, optional
            Name of the column to use for y-axis, by default "ISOEventful"
        title : str, optional
            Title of the plot, by default "Soundscape Density Plot"
        hue : str | None, optional
            Column name for color encoding, by default None
        palette : str | None, optional
            Color palette to use, by default "colorblind"
        figsize : tuple[int, int], optional
            Size of the figure (width, height), by default (5, 5)
        figure : Figure | SubFigure | None, optional
            Existing figure to plot on, by default None
        axes : Axes | np.ndarray | None, optional
            Existing axes to plot on, by default None

        """
        data, x, y = self._check_data_x_y(data, x, y)
        self._data = data
        self.x = x
        self.y = y

        self._check_data_hue(data, hue)
        self.hue = hue

        self.title = title
        self.figure = figure
        self.axes = axes
        self.labels = []

        self.palette = self._crosscheck_palette_hue(palette, hue)

        self._has_subplots = False

        self._scatter_params: ScatterParamTypes = copy.deepcopy(DEFAULT_SCATTER_PARAMS)
        self._scatter_params.update(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=self.palette,
        )

        self._density_params: DensityParamTypes = copy.deepcopy(DEFAULT_DENSITY_PARAMS)
        self._density_params.update(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=self.palette,
        )

        self._simple_density_params = copy.deepcopy(DEFAULT_DENSITY_PARAMS)
        # Override default params with user-provided params
        self._simple_density_params.update(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=self.palette,
            thresh=0.5,
            levels=2,
            alpha=0.5,
        )

        self._style_params: StyleParamsTypes = copy.deepcopy(DEFAULT_STYLE_PARAMS)

    @staticmethod
    def _check_data_x_y(
        data: pd.DataFrame | None,
        x: str | pd.Series | np.ndarray | None,
        y: str | pd.Series | np.ndarray | None,
    ) -> tuple[pd.DataFrame, str, str]:
        """
        Allocate data to the class attributes.

        Parameters
        ----------
            data : pd.DataFrame | None
                The data to be plotted.
            x : str | pd.Series | np.ndarray | None
                The x-axis data.
            y : str | pd.Series | np.ndarray | None
                The y-axis data.

        """
        if data is None:
            if x is not None and y is not None:
                # If data is not provided, and x and y are provided as arrays or Series:
                if isinstance(x, np.ndarray | pd.Series) and isinstance(
                    y, np.ndarray | pd.Series
                ):
                    logger.info("Combine x and y data into DataFrame.")

                    # Can get the name from the Series
                    xcol = (
                        x.get("name", DEFAULT_XCOL)
                        if isinstance(x, pd.Series)
                        else DEFAULT_XCOL
                    )
                    ycol = (
                        y.get("name", DEFAULT_YCOL)
                        if isinstance(y, pd.Series)
                        else DEFAULT_YCOL
                    )

                    data = pd.DataFrame({xcol: x, ycol: y})
                    x = xcol
                    y = ycol

                    return data, x, y

                # If data is not provided, and x and y are provided as strings:
                if isinstance(x, str) or isinstance(y, str):
                    msg = "x and y cannot be strings when data is not provided."
                    raise TypeError(msg)
                raise TypeError

            # If data is not provided, and x and y are not provided:
            msg = (
                "No data provided. "
                "Please provide data to CircumplexPlot to make it available "
                "to the whole Figure."
            )
            raise ValueError(msg)

        # If data is provided as DataFrame, and x and y are provided as strings:
        if isinstance(data, pd.DataFrame) and isinstance(x, str) and isinstance(y, str):
            # If data is provided, and x and y are provided as arrays or Series:
            if not isinstance(x, str) and not isinstance(y, str):
                msg = (
                    "x and y cannot be arrays or Series when data is provided."
                    "Please provide data as a DataFrame, and x and y as column names."
                )
                raise TypeError(msg)
            if x not in data.columns or y not in data.columns:
                msg = (
                    f"Invalid x or y column names. "
                    f"Available columns are: {data.columns.tolist()}"
                )
                raise ValueError(msg)

            logger.info("Data and columns are valid.")
            return data, x, y

        msg = "Invalid data provided. Please provide a DataFrame."
        raise ValueError(msg)

    @staticmethod
    def _check_data_hue(
        data: pd.DataFrame | None,
        hue: str | np.ndarray | pd.Series | None,
    ) -> None:
        """
        Check if the hue is valid for the given data.

        Parameters
        ----------
            data : pd.DataFrame | None
                The data to be plotted.
            hue : str | np.ndarray | pd.Series | None
                The column name for color encoding.

        """
        if data is None:
            msg = (
                "No data provided. "
                "Please provide data to CircumplexPlot to make it available "
                "to the whole Figure."
            )
            raise ValueError(msg)
        if isinstance(hue, str) and hue not in data.columns:
            msg = (
                f"Invalid hue column '{hue}'. "
                f"Available columns are: {data.columns.tolist()}"
            )
            raise ValueError(msg)

        # NOTE: Not implementing every possible check, can't be bothered.
        # Let Seaborn handle the other cases.
        if hue is not None and not isinstance(hue, str | np.ndarray | pd.Series):
            msg = "hue must be a string, numpy array, or pandas Series."
            raise TypeError(msg)

    def create_subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (5, 5),
        subplot_by: str | None = None,
        subplot_datas: list[pd.DataFrame] | None = None,
        subplot_titles: list[str] | None = None,
        *,
        adjust_figsize: bool = True,
        auto_allocate_axes: bool = False,  # Considered experimental
        **kwargs: Unpack[SubplotsParamsTypes],
    ) -> "CircumplexPlot":
        """
        Create subplots for the circumplex plot.

        Parameters
        ----------
            nrows (int): Number of rows in the subplot grid.
            ncols (int): Number of columns in the subplot grid.
            adjust_figsize (bool): Whether to adjust the figure size
                based on the number of subplots.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self.figsize = figsize
        subplot_kwargs = copy.deepcopy(DEFAULT_SUBPLOTS_PARAMS)
        fig_kw = kwargs.pop("fig_kw", {})
        subplot_kwargs.update(**kwargs)

        # Create a list of dataframes and titles for each subplot
        # based on the unique values in the specified column
        if subplot_by:
            subplot_datas, subplot_titles = self._setup_subplot_by(
                subplot_by, subplot_datas, subplot_titles
            )

        if subplot_titles and auto_allocate_axes:
            # Attempt to allocate axes based on the number of subplots
            nrows, ncols = self._allocate_subplot_axes(subplot_titles)

        if adjust_figsize:
            self.figsize = (ncols * self.figsize[0], nrows * self.figsize[1])

        self.figure, self.axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=self.figsize, **subplot_kwargs, **fig_kw
        )  # type: ignore[reportCallIssue]
        self._nrows = nrows
        self._ncols = ncols
        self._naxes = nrows * ncols
        self._subplots_params = kwargs
        self._has_subplots = self._naxes > 1

        # If subplot_datas or subplot_titles are provided, validate them
        if subplot_datas is not None or subplot_titles is not None:
            self._validate_subplots_datas(subplot_datas, subplot_titles)

        # Assign subplot data and titles to the class attributes
        # (incl. if None were provided)
        self._subplot_datas = subplot_datas
        self._subplot_titles = subplot_titles
        self._subplot_labels: list[list[str] | None] = [None] * self._naxes

        return self

    def show(self) -> None:
        """
        Show the figure.

        This method is a wrapper around plt.show() to display the figure.

        """
        if self.figure is None:
            msg = (
                "No figure object provided. "
                "Please create a figure using create_subplots() first."
            )
            raise ValueError(msg)
        if self._has_subplots:
            plt.tight_layout()
        plt.show()

    def _setup_subplot_by(
        self,
        subplot_by: str,
        subplot_datas: list[pd.DataFrame] | None,
        subplot_titles: list[str] | None,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """If subplot_by is provided, create subplots based on the unique values."""
        if subplot_datas:
            msg = "Provide either subplot_by or subplot_datas, but not both."
            raise ValueError(msg)
        if self._data is None:
            msg = (
                "No data provided for subplots. "
                "Please provide data to CircumplexPlot to make it available "
                "to the whole Figure."
            )
            raise ValueError(msg)
        if subplot_by not in self._data.columns:
            msg = (
                f"Invalid subplot_by column '{subplot_by}'. "
                f"Available columns are: {self._data.columns.tolist()}"
            )
            raise ValueError(msg)

        # Create subplot_datas based on the unique values in the specified column
        full_data = self._data.copy()
        unique_values = full_data[subplot_by].unique()

        # Create a list of dataframes for each unique value
        subplot_datas = [
            full_data[full_data[subplot_by] == value] for value in unique_values
        ]

        # Create subplot titles based on the unique values
        if subplot_titles is None:
            subplot_titles = [str(value) for value in unique_values]
        elif len(subplot_titles) != len(unique_values):
            msg = (
                "Number of subplot titles must match the number of unique values "
                f"for '{subplot_by}'. Got {len(subplot_titles)} titles and "
                f"{len(unique_values)} unique values."
            )
            raise ValueError(msg)
        else:
            # Keep the provided subplot titles
            msg = (
                "Not recommended to provide separate subplot titles when using subplot_by. "  # noqa: E501
                "Consider using the default titles based on unique values. "
                "Manual subplot_titles may not be in the same order as the data."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        return subplot_datas, subplot_titles

    def _validate_subplots_datas(
        self,
        subplot_datas: list[pd.DataFrame] | None,
        subplot_titles: list[str] | None,
    ) -> None:
        """
        Validate the subplots parameters.

        This checks the number of rows, columns, and axes
        against the provided subplot data and titles.
        """
        naxes = self._naxes
        has_subplots = self._has_subplots

        if (subplot_datas or subplot_titles) and not has_subplots:
            msg = (
                "Multiple subplot data or titles provided, but only one plot created. "
                "Please create subplots first."
            )
            raise ValueError(msg)
        if (subplot_datas and subplot_titles) and len(subplot_datas) != len(
            subplot_titles
        ):
            msg = (
                "Number of subplot data and titles must match. "
                f"Got {len(subplot_datas)} data and {len(subplot_titles)} titles."
            )
            raise ValueError(msg)
        if subplot_datas and len(subplot_datas) != naxes:
            msg = (
                "Number of subplot data must match number of axes. "
                f"Got {len(subplot_datas)} data and {naxes} axes."
            )
            raise ValueError(msg)
        if subplot_titles and len(subplot_titles) != naxes:
            msg = (
                "Number of subplot titles must match number of axes. "
                f"Got {len(subplot_titles)} titles and {naxes} axes."
            )
            raise ValueError(msg)

    def _allocate_subplot_axes(self, subplot_titles: list[str]) -> tuple[int, int]:
        """Allocate the subplot axes based on the number of data subsets."""
        msg = (
            "This is an experimental feature. "
            "The number of rows and columns may not be optimal."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

        n_datasets = len(subplot_titles)
        ncols = int(np.ceil(np.sqrt(n_datasets)))
        nrows = int(np.ceil(n_datasets / ncols))
        return nrows, ncols

    def _check_for_axes(self) -> None:
        """
        Check if the axes object is provided.

        Returns
        -------
            Axes: The axes object to be used for plotting.

        Raises
            ValueError: If the axes object does not exist.

        """
        if self.axes is None:
            msg = (
                "No axes object provided. "
                "Please create a figure and axes using create_subplots() first."
            )
            raise ValueError(msg)

    @staticmethod
    def _crosscheck_palette_hue(
        palette: SeabornPaletteType | None, hue: str | np.ndarray | pd.Series | None
    ) -> SeabornPaletteType | None:
        """
        Check if the palette is valid for the given hue.

        Parameters
        ----------
            palette : SeabornPaletteType
                The color palette to use.
            hue : str | np.ndarray | pd.Series | None
                The column name for color encoding.

        Raises
        ------
            ValueError: If the palette is not valid for the given hue.

        """
        return palette if hue is not None else None

    def get_figure(self) -> Figure | SubFigure:
        """
        Get the figure object.

        Returns
        -------
            Figure | SubFigure: The figure object to be used for plotting.

        Raises
        ------
            ValueError: If the figure object does not exist.
            TypeError: If the figure object is not a valid Figure or SubFigure.

        """
        if self.figure is None:
            msg = (
                "No figure object provided. "
                "Please create a figure using create_subplots() first."
            )
            raise ValueError(msg)
        if isinstance(self.figure, Figure | SubFigure):
            return self.figure
        msg = "Invalid figure object. Please provide a valid Figure or SubFigure."
        raise TypeError(msg)

    def get_axes(self) -> Axes | np.ndarray:
        """
        Get the axes object.

        Returns
        -------
            Axes | np.ndarray: The axes object to be used for plotting.

        Raises
        ------
            ValueError: If the axes object does not exist.
            TypeError: If the axes object is not a valid Axes or ndarray of Axes.

        """
        self._check_for_axes()
        if isinstance(self.axes, Axes | np.ndarray):
            return self.axes
        msg = "Invalid axes object. Please provide a valid Axes or ndarray of Axes."
        raise TypeError(msg)

    def get_single_axes(self, ax_idx: int | tuple[int, int] | None = None) -> Axes:
        """
        Get the axes object.

        Returns
        -------
            Axes | np.ndarray: The axes object to be used for plotting.

        Raises
        ------
            ValueError: If the axes object does not exist.
            TypeError: If the axes object is not a valid Axes or ndarray of Axes.
            ValueError: If the axes index is invalid.

        """
        self._check_for_axes()

        def validate_tuple_axes_index(
            nrows: int, ncols: int, naxes: int, ax_idx: tuple[int, int]
        ) -> None:
            """
            Validate the tuple axes index.

            This checks the ax_idx types and compares the implied number of axes
            with the actual number of axes in the figure.
            """
            if (
                len(ax_idx) != 2  # noqa: PLR2004
                or not isinstance(ax_idx[0], int)
                or not isinstance(ax_idx[1], int)
                or ax_idx[0] < 0
                or ax_idx[1] < 0
            ):
                msg = (
                    "Invalid axes index provided. "
                    "Expected a tuple of 2 positive integers."
                )
                raise ValueError(msg)

            if ax_idx[0] >= (nrows - 1) or ax_idx[1] >= (ncols - 1):
                msg = (
                    "Invalid axes index provided."
                    f" The figure contains {nrows} rows and {ncols} columns. "
                    f"ax_idx implied {ax_idx[0] + 1} rows and {ax_idx[1] + 1} columns."
                )
                raise ValueError(msg)

            idx_implied_n_axes = (ax_idx[0] + 1) * (ax_idx[1] + 1)
            if naxes < idx_implied_n_axes:
                msg = (
                    "Invalid axes index provided."
                    f" The figure contains {naxes} axes. "
                    f"ax_idx implied {idx_implied_n_axes} axes."
                )
                raise ValueError(msg)

        def validate_int_axes_index(naxes: int, ax_idx: int) -> None:
            """
            Validate the integer axes index.

            This checks the ax_idx type and compares the implied number of axes
            with the actual number of axes in the figure.
            """
            if not isinstance(ax_idx, int) or ax_idx < 0:
                msg = "Invalid axes index provided. Expected a positive integer."
                raise ValueError(msg)

            if (ax_idx + 1) > naxes:
                msg = (
                    "Invalid axes index provided."
                    f" The figure contains {naxes} axes. "
                    f"ax_idx implied {ax_idx + 1} axes."
                )
                raise ValueError(msg)

        if isinstance(self.axes, np.ndarray) and ax_idx is not None:
            if isinstance(ax_idx, tuple):
                validate_tuple_axes_index(self._nrows, self._ncols, self._naxes, ax_idx)
                return self.axes[ax_idx[0], ax_idx[1]]

            validate_int_axes_index(self._naxes, ax_idx)
            return self.axes.flatten()[ax_idx]

        if isinstance(self.axes, Axes) and (ax_idx == 0 or ax_idx is None):
            return self.axes

        msg = "Invalid axes index provided."
        raise ValueError(msg)

    def yield_axes_objects(self) -> Generator[Axes, None, None]:
        """
        Generate a sequence of axes objects to iterate over.

        This method is a helper to iterate over all axes in the figure,
        whether the figure contains a single Axes object or an array of Axes objects.

        Yields
        ------
        Axes
            Individual matplotlib Axes objects from the current figure.

        """
        if isinstance(self.axes, np.ndarray):
            yield from self.axes.flatten()
        elif isinstance(self.axes, Axes):
            yield self.axes

    @staticmethod
    def _valid_density(data: pd.DataFrame) -> None:
        """
        Check if the data is valid for density plots.

        Raises
        ------
            UserWarning: If the data is too small for density plots.

        """
        if len(data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for "
                f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

    def add_scatter(
        self,
        on_axis: int | tuple[int, int] | None = None,
        **kwargs: Unpack[ScatterParamTypes],
    ) -> "CircumplexPlot":
        """
        Add a scatter plot to the existing axes.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self._check_for_axes()
        scatter_params = copy.deepcopy(self._scatter_params)
        # Update with any additional parameters provided
        scatter_params.update(**kwargs)
        scatter_params["palette"] = self._crosscheck_palette_hue(
            scatter_params.get("palette"), scatter_params.get("hue")
        )

        if scatter_params.get("data") is None and self._subplot_datas is None:
            msg = (
                "No data provided for scatter plot. "
                "You should either pass data to CircumplexPlot to make it available "
                "to the whole Figure, or to this method to plot on a single axis."
            )
            raise ValueError(msg)

        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                if self._subplot_datas is not None:
                    scatter_params["data"] = self._subplot_datas[i]
                sns.scatterplot(ax=axis, **scatter_params)
        else:
            axis = self.get_single_axes(on_axis)
            if self._subplot_datas is not None:
                # Get the corresponding subplot idx to match up to the data
                ax_idx = (
                    on_axis
                    if isinstance(on_axis, int)
                    else ((on_axis[0] + 1) * (on_axis[1] + 1) - 1)
                )
                scatter_params["data"] = self._subplot_datas[ax_idx]
            sns.scatterplot(ax=axis, **scatter_params)
        return self

    def _record_density_label(
        self, idx: int, **density_params: Unpack[DensityParamTypes]
    ) -> None:
        """See: https://github.com/mwaskom/seaborn/issues/3523 ."""
        label = density_params.get("label")
        if (
            self.hue is not None or density_params.get("hue") is not None
        ) and label is not None:
            warnings.warn(
                "Cannot set custom labels for density plots with hue.",
                UserWarning,
                stacklevel=2,
            )
        elif self._subplot_labels[idx] is None:
            self._subplot_labels[idx] = [label]  # type: ignore[reportCallIssue]
        else:
            self._subplot_labels[idx].append(label)  # type: ignore[reportOptionalMemberAccess]

    # TODO(MitchellAcoustcs): Should refactor to have a similar ._sns_scatter() method.  # noqa: E501, TD003
    #       to avoid code duplication.
    #       Then have a ._sns_plot() method that calls the appropriate one?
    #       This would simplify the label setting...
    #       Right now I manually call _record_density_label() in every add_*_density() method.  # noqa: E501

    def _sns_density(
        self,
        axis: Axes,
        *,
        include_outline: bool = False,
        **density_params: Unpack[DensityParamTypes],  # type: ignore[reportGeneralTypeIssues]
    ) -> None:
        """Add a density plot to the selected axes."""
        # Check if the data is provided either in the class or in the method call  # noqa: E501
        # If provided in the class call, it would have been added to
        # self._density_params during object initialization and copied to density_params  # noqa: E501
        data = density_params.get("data")
        if data is not None:
            self._valid_density(data)
        else:
            msg = "No data provided for density plot."
            raise ValueError(msg)

        # Plot the density plot on the current axis
        sns.kdeplot(
            ax=axis,
            **density_params,
        )
        if include_outline:
            outline_params = density_params.copy()
            outline_params.update({"fill": False, "alpha": 1, "legend": False})
            sns.kdeplot(
                ax=axis,
                **outline_params,
            )

    def add_density(
        self,
        on_axis: int | tuple[int, int] | None = None,
        *,
        include_outline: bool = False,
        **kwargs: Unpack[DensityParamTypes],
    ) -> "CircumplexPlot":
        """
        Add a density plot to the existing axes.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self._check_for_axes()
        # Start with the default density parameters
        density_params = copy.deepcopy(self._density_params)
        # Update with any additional parameters provided
        # in this method call
        density_params.update(**kwargs)
        density_params["palette"] = self._crosscheck_palette_hue(
            density_params.get("palette"), density_params.get("hue")
        )

        # If no axis is specified, plot on all axes
        # in the figure (if multiple subplots exist)
        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                # If subplot data is provided, use it for the density plot
                if self._subplot_datas is not None:
                    density_params["data"] = self._subplot_datas[i]
                self._sns_density(
                    axis=axis, include_outline=include_outline, **density_params
                )
                self._record_density_label(i, **density_params)

        # If an axis is specified, plot on that axis
        else:
            axis = self.get_single_axes(on_axis)
            # If subplot data is provided, use it for the density plot
            if self._subplot_datas is not None:
                # Get the corresponding subplot idx to match up to the data
                ax_idx = (
                    on_axis
                    if isinstance(on_axis, int)
                    else ((on_axis[0] + 1) * (on_axis[1] + 1) - 1)
                )
                density_params["data"] = self._subplot_datas[ax_idx]
                self._record_density_label(ax_idx, **density_params)
            self._sns_density(
                axis=axis, include_outline=include_outline, **density_params
            )
            self._record_density_label(0, **density_params)
        return self

    def add_simple_density(
        self,
        on_axis: int | tuple[int, int] | None = None,
        thresh: float = 0.5,
        levels: int | Iterable[float] = 2,
        alpha: float = 0.5,
        *,
        include_outline: bool = True,
        **kwargs: Unpack[DensityParamTypes],  # type: ignore[reportGeneralTypeIssues]
    ) -> "CircumplexPlot":
        """
        Add a simple density plot to the existing axes.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self._check_for_axes()
        # Start with the default density parameters
        simple_density_params = copy.deepcopy(self._simple_density_params)
        # Update with any additional parameters provided
        # in this method call
        simple_density_params.update(
            thresh=thresh, levels=levels, alpha=alpha, **kwargs
        )  # type: ignore[reportCallIssue]
        simple_density_params["palette"] = self._crosscheck_palette_hue(
            simple_density_params.get("palette"), simple_density_params.get("hue")
        )

        # If no axis is specified, plot on all axes
        # in the figure (if multiple subplots exist)
        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                # If subplot data is provided, use it for the density plot
                if self._subplot_datas is not None:
                    simple_density_params["data"] = self._subplot_datas[i]
                self._sns_density(
                    axis=axis, include_outline=include_outline, **simple_density_params
                )
                # Record the label for the subplot
                self._record_density_label(i, **simple_density_params)

        # If an axis is specified, plot on that axis
        else:
            axis = self.get_single_axes(on_axis)
            # If subplot data is provided, use it for the density plot
            if self._subplot_datas is not None:
                # Get the corresponding subplot idx to match up to the data
                ax_idx = (
                    on_axis
                    if isinstance(on_axis, int)
                    else ((on_axis[0] + 1) * (on_axis[1] + 1) - 1)
                )
                simple_density_params["data"] = self._subplot_datas[ax_idx]

                # Record the label for the subplot
                self._record_density_label(ax_idx, **simple_density_params)
            self._sns_density(
                axis=axis, include_outline=include_outline, **simple_density_params
            )
            # Record the label for the main plot
            # (if no subplot data is provided)
            self._record_density_label(0, **simple_density_params)

        return self

    def add_spi_simple_density(
        self,
        spi_data: pd.DataFrame | np.ndarray | None = None,
        spi_params: DirectParams | CentredParams | None = None,
        n: int = 1000,
        on_axis: int | tuple[int, int] | None = None,
        label: str = "SPI",
        *,
        show_score: Literal["on axis", "under title", False] = "under title",
        axis_text_kw: dict[str, Any] | None = None,
        **kwargs: Unpack[DensityParamTypes],  # type: ignore[reportGeneralTypeIssues]
    ) -> "CircumplexPlot":
        """
        Add a SPI plot to the existing axes.

        Parameters
        ----------
            spi_data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        Examples
        --------
        >>> import soundscapy as sspy
        >>> from soundscapy.plotting import CircumplexPlot
        >>> from soundscapy.spi import MultiSkewNorm, DirectParams
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> df = sspy.isd.load()
        >>> df = sspy.surveys.add_iso_coords(df)
        >>> sub_df = sspy.isd.select_location_ids(df,['CamdenTown', 'RegentsParkJapan'])
        >>> spi = DirectParams(
                xi=np.array([0.5, 0.7]),
                omega=np.array([[0.1, 0.05], [0.05, 0.1]]),
                alpha=np.array([0, -5]),
                )
        >>> spi_p = (
                CircumplexPlot(sub_df)
                .create_subplots(subplot_by="LocationID", auto_allocate_axes=True)
                .add_scatter()
                .add_simple_density(label="Test")
                .add_spi_simple_density(
                    spi_params=spi, label="Target", show_score="on axis"
                )
                .apply_styling(
                    legend=True, diagonal_lines=True, legend_location="lower left"
                )
            )
        >>> spi_p.show() # doctest: +SKIP

        """
        if spi_data is not None and spi_params is not None:
            msg = "Please provide either spi_data or spi_params, not both."
            raise ValueError(msg)
        if spi_data is None and spi_params is None:
            msg = (
                "No data provided for SPI plot. "
                "Please provide either spi_data or spi_params."
            )
            raise ValueError(msg)

        if spi_params is not None:
            spi_msn = MultiSkewNorm.from_params(spi_params)
            sample_data = spi_msn.sample(n=n, return_sample=True)
            self._spi_data = pd.DataFrame(sample_data, columns=[self.x, self.y])

        elif spi_data is not None:
            xcol = kwargs.get("x", self.x)
            ycol = kwargs.get("y", self.y)
            if not (isinstance(xcol, str) and isinstance(ycol, str)):
                msg = "Sorry, at the moment in this method, x and y must be strings."
                raise ValueError(msg)

            # Check if the data is a DataFrame
            if isinstance(spi_data, pd.DataFrame) and (
                xcol not in spi_data.columns or ycol not in spi_data.columns
            ):
                spi_data = spi_data.rename(columns={xcol: self.x, self.y: self.y})

            if isinstance(spi_data, np.ndarray):
                if len(spi_data.shape) != 2 or spi_data.shape[1] != 2:  # noqa: PLR2004
                    msg = (
                        "Invalid shape for SPI data. "
                        "Expected a 2D array with 2 columns."
                    )
                    raise ValueError(msg)
                # Convert the numpy array to a DataFrame
                spi_data = pd.DataFrame(spi_data, columns=[self.x, self.y])

            self._valid_density(spi_data)
            self._spi_data = spi_data

        self._check_for_axes()

        # Start with the default density parameters
        spi_density_params = copy.deepcopy(self._simple_density_params)
        spi_density_params.update(color="r")
        spi_density_params.update(**kwargs)
        spi_density_params.update(
            data=self._spi_data, x=self.x, y=self.y, label=label, hue=None, palette=None
        )

        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                self._sns_density(axis=axis, include_outline=True, **spi_density_params)
                # Record the label for the subplot
                self._record_density_label(i, **spi_density_params)

                if show_score:
                    # If subplot data is provided, use it for the spi_score
                    if self._subplot_datas is not None:
                        test_data = self._subplot_datas[i][[self.x, self.y]]
                    else:
                        test_data = self._data[[self.x, self.y]]
                    spi_val = spi_score(target=self._spi_data, test=test_data)

                    if show_score == "under title":
                        # Add the SPI score to the plot title
                        if self._subplot_titles is not None:
                            self._subplot_titles[i] = (
                                f"{self._subplot_titles[i]}\nSPI: {spi_val}"
                            )
                        else:
                            axis.set_title(f"{axis.get_title()}\nSPI: {spi_val}")
                    if show_score == "on axis":
                        # Add the SPI score to the plot
                        self._add_spi_score_to_axis(
                            axis=axis,
                            spi_val=spi_val,
                            text_kw=axis_text_kw,
                        )

        # If no axis is specified, plot on all axes
        # in the figure (if multiple subplots exist)
        else:
            axis = self.get_single_axes(on_axis)
            self._sns_density(axis=axis, **spi_density_params)
            # Record the label for the main plot
            self._record_density_label(0, **spi_density_params)

            if show_score:
                # If subplot data is provided, use it for the spi_score
                if self._subplot_datas is not None:
                    # Get the corresponding subplot idx to match up to the data
                    ax_idx = (
                        on_axis
                        if isinstance(on_axis, int)
                        else ((on_axis[0] + 1) * (on_axis[1] + 1) - 1)
                    )
                    test_data = self._subplot_datas[ax_idx][[self.x, self.y]]
                else:
                    test_data = self._data[[self.x, self.y]]
                spi_val = spi_score(target=self._spi_data, test=test_data)

                if show_score == "under title":
                    # Add the SPI score to the plot title
                    axis.set_title(f"{axis.get_title()}\nSPI: {spi_val}")
                if show_score == "on axis":
                    # Add the SPI score to the plot
                    self._add_spi_score_to_axis(
                        axis=axis,
                        spi_val=spi_val,
                        text_kw=axis_text_kw,
                    )

        return self

    @staticmethod
    def _add_spi_score_to_axis(
        axis: Axes,
        spi_val: float,
        text_kw: dict[str, Any] | None = None,
    ) -> None:
        """Add the SPI score to the specified axis."""
        text_kwargs = DEFAULT_SPI_TEXT_KWARGS.copy()
        text_kwargs["s"] = f"SPI: {spi_val}"
        if text_kw is not None:
            text_kwargs.update(text_kw)
        # Add the SPI score to the plot
        axis.text(**text_kwargs)

    # def create_jointplot(
    #     self, **kwargs: Unpack[JointPlotParamTypes]
    # ) -> tuple[Figure | SubFigure | None, Axes]:
    #     """
    #     Create a joint plot using Seaborn.

    #     Examples
    #     --------
    #     >>> import soundscapy as sspy
    #     >>> from soundscapy.plotting import Backend, CircumplexPlot, StyleOptions, CircumplexPlotParams
    #     >>> data = sspy.isd.load()
    #     >>> data = sspy.surveys.add_iso_coords(data, overwrite=True)
    #     >>> sample_data = sspy.isd.select_location_ids(data, ['CamdenTown'])
    #     >>> plot = CircumplexPlot(data=sample_data, backend=Backend.SEABORN)
    #     >>> g = plot.jointplot()
    #     >>> g.show() # doctest: +SKIP

    #     """

    #     self._jointgrid = sns.JointGrid(xlim=params.xlim, ylim=params.ylim)
    #     joint_params = params
    #     joint_params.title = ""
    #     SeabornBackend.create_density(self, data, joint_params, ax=g.ax_joint)

    #     margin_params = params
    #     margin_params.title = ""
    #     sns.kdeplot(data, x=params.x, ax=g.ax_marg_x, fill=True, alpha=params.alpha)
    #     sns.kdeplot(data, y=params.y, ax=g.ax_marg_y, fill=True, alpha=params.alpha)

    #     return (
    #         g.fig,
    #         g.ax_joint,
    #     )  # TODO: Should return the whole JointGrid object - repeat throughout plotting methods

    def apply_styling(
        self,
        **kwargs: Unpack[StyleParamsTypes],
    ) -> "CircumplexPlot":
        """
        Apply styling to the Seaborn plot.

        Returns
        -------
            tuple: The styled figure and axes objects.

        """
        self._style_params.update(**kwargs)
        self._check_for_axes()

        self._set_style()
        self._circumplex_grid()
        self._set_title()
        self._set_axes_titles()
        self._deal_w_default_labels()
        if self._style_params["primary_lines"]:
            self._primary_lines_and_labels()
        if self._style_params["diagonal_lines"]:
            self._diagonal_lines_and_labels()

        try:
            self._add_density_labels()
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Could not add density labels. Error: {e}",
                UserWarning,
                stacklevel=2,
            )

        if (
            self._style_params["legend"]
            and self._style_params["legend_location"] is not False
        ):
            # NOTE: Should really check for the presence of a legend.
            # If hue is added in the .add_* methods,
            # it doesn't show up in the class attributes.
            self._move_legend()

        return self

    def _set_style(self) -> None:
        """Set the overall style for the plot."""
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    def _circumplex_grid(self) -> "CircumplexPlot":
        """Add the circumplex grid to the plot."""
        for _, axis in enumerate(self.yield_axes_objects()):
            axis.set_xlim(self._style_params.get("xlim"))
            axis.set_ylim(self._style_params.get("ylim"))
            axis.set_aspect("equal")

            axis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
            axis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

            axis.grid(visible=True, which="major", color="grey", alpha=0.5)
            axis.grid(
                visible=True,
                which="minor",
                color="grey",
                linestyle="dashed",
                linewidth=0.5,
                alpha=0.4,
                zorder=self._style_params.get("prim_lines_zorder"),
            )

        return self

    def _set_title(self) -> "CircumplexPlot":
        """Set the title of the plot."""
        if self.title and self._has_subplots:
            figure = self.get_figure()
            figure.suptitle(self.title, fontsize=self._style_params["title_fontsize"])
        elif self.title and not self._has_subplots:
            axis = self.get_single_axes()
            if axis.get_title() == "":
                axis.set_title(
                    self.title, fontsize=self._style_params["title_fontsize"]
                )
            else:
                figure = self.get_figure()
                figure.suptitle(
                    self.title, fontsize=self._style_params["title_fontsize"]
                )
        return self

    def _set_axes_titles(self) -> "CircumplexPlot":
        """Set the titles of the subplots plot."""
        if self._has_subplots and self._subplot_titles is not None:
            for i, axis in enumerate(self.yield_axes_objects()):
                axis.set_title(self._subplot_titles[i])
        return self

    def _deal_w_default_labels(self) -> "CircumplexPlot":
        """Handle the default labels for the axes."""
        if not self._style_params.get("show_labels"):
            for _, axis in enumerate(self.yield_axes_objects()):
                axis.set_xlabel("")
                axis.set_ylabel("")
        return self

    def _add_density_labels(self) -> "CircumplexPlot":
        """
        See https://github.com/mwaskom/seaborn/issues/3523 .

        This is a workaround to add labels to the density plots.
        """
        for i, axis in enumerate(self.yield_axes_objects()):
            labels = self._subplot_labels[i]
            if labels is not None:
                contours = [
                    child
                    for child in axis.get_children()
                    if isinstance(child, QuadContourSet)
                ]  # Get the contour artists in the axis
                colormaps = [np.array(c.get_cmap().colors) for c in contours]  # type: ignore[reportAttributeAccessIssue]
                # outline cmap contains 2 colors
                outlines = [a for a in colormaps if len(a) == 2]  # noqa: PLR2004
                # filled cmaps contain many colors
                filled = [a for a in colormaps if len(a) != 2]  # noqa: PLR2004
                if len(outlines) == len(labels):
                    # If there are as many outlines as plots, use that
                    colors = [a[0] for a in outlines][::-1]
                elif len(filled) == len(labels):
                    # Otherwise, get the 'average' color from the filled cmaps
                    colors = [np.median(a, axis=0) for a in filled]
                else:
                    warnings.warn(
                        "Could not match up plot colors to subplot labels."
                        f"Unable to add legend for plot: {i}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                new_handles = [Line2D([0], [0], color=c) for c in colors]
                axis.legend(
                    new_handles,
                    labels,
                )
        return self

    def _move_legend(self) -> "CircumplexPlot":
        """Move the legend to the specified location."""
        for _, axis in enumerate(self.yield_axes_objects()):
            old_legend = axis.get_legend()
            if old_legend is None:
                axis.legend()
                old_legend = axis.get_legend()
            handles = old_legend.legend_handles
            labels = [t.get_text() for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()
            axis.legend(
                handles,
                labels,
                loc=self._style_params.get("legend_location"),
                title=title,
            )
        return self

    def _primary_lines_and_labels(self) -> "CircumplexPlot":
        """Add primary lines to the plot."""
        for _, axis in enumerate(self.yield_axes_objects()):
            axis.axhline(
                y=0,
                color="grey",
                linestyle="dashed",
                alpha=1,
                lw=self._style_params.get("linewidth"),
                zorder=self._style_params.get("prim_lines_zorder"),
            )
            axis.axvline(
                x=0,
                color="grey",
                linestyle="dashed",
                alpha=1,
                lw=self._style_params.get("linewidth"),
                zorder=self._style_params.get("prim_lines_zorder"),
            )
        return self

    def _diagonal_lines_and_labels(self) -> "CircumplexPlot":
        """Add diagonal lines and labels to the plot."""
        for _, axis in enumerate(self.yield_axes_objects()):
            xlim = self._style_params.get("xlim")
            ylim = self._style_params.get("ylim")
            axis.plot(
                xlim,
                ylim,
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                lw=self._style_params["linewidth"],
                zorder=self._style_params["diag_lines_zorder"],
            )
            axis.plot(
                xlim,
                ylim[::-1],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                lw=self._style_params["linewidth"],
                zorder=self._style_params["diag_lines_zorder"],
            )

            diag_ax_font = {
                "fontstyle": "italic",
                "fontsize": "small",
                "fontweight": "bold",
                "color": "black",
                "alpha": 0.5,
            }
            axis.text(
                xlim[1] / 2,
                ylim[1] / 2,
                "(vibrant)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params["diag_labels_zorder"],
            )
            axis.text(
                xlim[0] / 2,
                ylim[1] / 2,
                "(chaotic)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params["diag_labels_zorder"],
            )
            axis.text(
                xlim[0] / 2,
                ylim[0] / 2,
                "(monotonous)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params["diag_labels_zorder"],
            )
            axis.text(
                xlim[1] / 2,
                ylim[0] / 2,
                "(calm)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params["diag_labels_zorder"],
            )
        return self

    def iso_annotation(
        self, location_idx: int, x_adj: float = 0, y_adj: float = 0, **kwargs
    ) -> "CircumplexPlot":
        """Add an annotation to the plot (only for Seaborn backend)."""
        x = self._data[self.x].iloc[location_idx]
        y = self._data[self.y].iloc[location_idx]
        if isinstance(self.axes, np.ndarray):
            for i, _ in enumerate(self.axes.flatten()):
                self.axes[i].annotate(
                    text=self._data.index[location_idx],
                    xy=(x, y),
                    xytext=(x + x_adj, y + y_adj),
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "-", "ec": "black"},
                    **kwargs,
                )
        elif isinstance(self.axes, Axes):
            self.axes.annotate(
                text=self._data.index[location_idx],
                xy=(x, y),
                xytext=(x + x_adj, y + y_adj),
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "-", "ec": "black"},
                **kwargs,
            )
        else:
            msg = "Invalid axes object. Please provide a valid Axes or ndarray of Axes."
            raise TypeError(msg)
        return self
