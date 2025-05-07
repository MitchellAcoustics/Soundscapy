"""
Main module for creating circumplex plots using different backends.

Example:
-------
>>> from soundscapy import isd, surveys
>>> from soundscapy.plotting.iso_plot import ISOPlot
>>> df = isd.load()
>>> df = surveys.add_iso_coords(df)
>>> sub_df = isd.select_location_ids(df, ['CamdenTown', 'RegentsParkJapan'])
>>> cp = (
>>>     ISOPlot(data=sub_df, hue="SessionID")
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
# ruff: noqa: SLF001

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Literal

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

from soundscapy.plotting.defaults import (
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SPI_TEXT_KWARGS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_SUBPLOTS_PARAMS,
    DEFAULT_XCOL,
    DEFAULT_XLIM,
    DEFAULT_YCOL,
    DEFAULT_YLIM,
    RECOMMENDED_MIN_SAMPLES,
)
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from soundscapy.plotting.plotting_types import (
        DensityParamTypes,
        JointPlotParamTypes,  # noqa: F401
        ScatterParamTypes,
        SeabornPaletteType,
        StyleParamsTypes,
        SubplotsParamsTypes,
    )

    try:
        from soundscapy.spi.msn import (
            CentredParams,
            DirectParams,
        )
    except ImportError as e:
        msg = (
            "SPI functionality requires additional dependencies. "
            "Install with: pip install soundscapy[spi]"
        )
        raise ImportError(msg) from e

logger = get_logger()


class ISOPlot:
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
            ISOPlot(ct, hue="LocationID")
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
        x: str | np.ndarray | pd.Series | None = "ISOPleasant",
        y: str | np.ndarray | pd.Series | None = "ISOEventful",
        title: str | None = "Soundscape Density Plot",
        hue: str | None = None,
        palette: SeabornPaletteType | None = "colorblind",
        figure: Figure | SubFigure | None = None,
        axes: Axes | np.ndarray | None = None,
    ) -> None:
        """
        Initialize a ISOPlot instance.

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

        self._subplot_datas = None
        self._subplot_titles = None
        self._subplot_labels: list[list[str] | None] = [None]

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

        self._spi_data = None

    def _clone(self) -> ISOPlot:
        new = ISOPlot()

        new._data = self._data
        new.x = self.x
        new.y = self.y

        new.hue = self.hue

        new.title = self.title
        new.figure = self.figure
        new.axes = self.axes
        new.labels = self.labels

        new.palette = self.palette

        new._nrows = self._nrows
        new._ncols = self._ncols
        new._naxes = self._naxes
        new._subplots_params = self._subplots_params
        new._has_subplots = self._has_subplots

        new._subplot_datas = self._subplot_datas
        new._subplot_titles = self._subplot_titles
        new._subplot_labels = self._subplot_labels

        new._scatter_params = copy.deepcopy(self._scatter_params)
        new._density_params = copy.deepcopy(self._density_params)
        new._simple_density_params = copy.deepcopy(self._simple_density_params)
        new._style_params = copy.deepcopy(self._style_params)

        new._spi_data = self._spi_data

        return new

    @staticmethod
    def _check_data_x_y(
        data: pd.DataFrame | None,
        x: str | pd.Series | np.ndarray | None,
        y: str | pd.Series | np.ndarray | None,
    ) -> tuple[pd.DataFrame | None, str, str]:
        """
        Allocate data to the class attributes.

        Parameters
        ----------
            data : pd.DataFrame | None
                The data to be plotted.
            x : str | pd.Series | np.ndarray
                The x-axis data.
            y : str | pd.Series | np.ndarray
                The y-axis data.

        """
        if not isinstance(data, pd.DataFrame) and data is not None:
            msg = (
                "data must be a pandas DataFrame or None. "
                "Please provide data as a DataFrame."
            )
            raise TypeError(msg)

        # If data is provided as DataFrame
        if isinstance(data, pd.DataFrame):
            # and x and y are not provided as strings:
            if not isinstance(x, str) or not isinstance(y, str):
                msg = (
                    "x and y cannot be arrays or Series when data is provided."
                    "Please provide data as a DataFrame, "
                    "and x and y as column names (str)."
                )
                raise TypeError(msg)

            # and x and y are provided as strings:
            if isinstance(x, str) and isinstance(y, str):
                if x not in data.columns or y not in data.columns:
                    msg = (
                        f"Invalid x or y column names. "
                        f"Available columns are: {data.columns.tolist()}"
                    )
                    raise ValueError(msg)

                logger.info("Data and columns are valid.")
                return data, x, y

        # By this point, we know data is None
        if isinstance(x, pd.Series | np.ndarray) and isinstance(
            y, pd.Series | np.ndarray
        ):
            # If data is not provided, and x and y are provided as arrays or Series:
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

        # If data is None, and x and y are strings:
        if isinstance(x, str) and isinstance(y, str):
            logger.info(
                "No data provided to ISOPlot. "
                "Assuming will be passed later in the chain."
            )
            return data, x, y
        msg = (
            "Invalid data provided. "
            "Please provide data as a DataFrame, "
            "or x and y as arrays or Series."
        )
        raise TypeError(msg)

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
            return

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
    ) -> ISOPlot:
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
            nrows=nrows, ncols=ncols, figsize=self.figsize, **subplot_kwargs
        )
        self._nrows = nrows
        self._ncols = ncols
        self._naxes = nrows * ncols
        self._subplots_params = kwargs
        self._has_subplots = self._naxes > 1

        # If subplot_datas or subplot_titles are provided, validate them
        if subplot_datas is not None or subplot_titles is not None:
            self._validate_subplots_datas(subplot_datas, subplot_titles)

        # new = self._clone() # TODO(MitchellAcoustcs): Refactor to use this  # noqa: E501, ERA001, TD003

        # Assign subplot data and titles to the class attributes
        # (incl. if None were provided)
        self._subplot_datas = subplot_datas
        self._subplot_titles = subplot_titles
        self._subplot_labels = [None] * self._naxes

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
                "Please provide data to ISOPlot to make it available "
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
        if subplot_datas and len(subplot_datas) > naxes:
            msg = (
                "Number of subplot data must not exceed number of axes. "
                f"Got {len(subplot_datas)} data and {naxes} axes."
            )
            raise ValueError(msg)
        if subplot_titles and len(subplot_titles) > naxes:
            msg = (
                "Number of subplot titles must not exceed number of axes. "
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
        on_axis: Axes | int | tuple[int, int] | None = None,
        **kwargs: Unpack[ScatterParamTypes],
    ) -> ISOPlot:
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
                "You should either pass data to ISOPlot to make it available "
                "to the whole Figure, or to this method to plot on a single axis."
            )
            raise ValueError(msg)

        if isinstance(on_axis, Axes):
            self.axes = on_axis
            sns.scatterplot(ax=on_axis, **scatter_params)

        elif on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                if self._subplot_datas is not None:
                    if i >= len(self._subplot_datas):
                        logger.info("Axis index exceeds available subplot data.")
                        break
                    scatter_params["data"] = self._subplot_datas[i]
                sns.scatterplot(ax=axis, **scatter_params)
        else:
            axis = self.get_single_axes(on_axis)
            if self._subplot_datas is not None:
                # Get the corresponding subplot idx to match up to the data
                ax_idx = self._get_ax_idx_from_on_axis(on_axis)
                if ax_idx < len(self._subplot_datas):
                    scatter_params["data"] = self._subplot_datas[ax_idx]
                else:
                    logger.info("Axis index exceeds available subplot data.")
                    return self
            sns.scatterplot(ax=axis, **scatter_params)
        return self

    def _record_density_label(
        self, idx: int, density_params: DensityParamTypes
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
        density_params: DensityParamTypes,
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
            **density_params,  # type: ignore[reportArgumentType]
        )
        if include_outline:
            outline_params = density_params.copy()
            outline_params.update({"fill": False, "alpha": 1, "legend": False})
            sns.kdeplot(
                ax=axis,
                **outline_params,  # type: ignore[reportArgumentType]
            )

    def add_density(
        self,
        on_axis: int | tuple[int, int] | None = None,
        *,
        include_outline: bool = False,
        **kwargs: Unpack[DensityParamTypes],
    ) -> ISOPlot:
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
                    if i >= len(self._subplot_datas):
                        logger.info("Axis index exceeds available subplot data.")
                        break
                    density_params["data"] = self._subplot_datas[i]
                self._sns_density(
                    axis=axis,
                    include_outline=include_outline,
                    density_params=density_params,
                )
                self._record_density_label(i, density_params)

        # If an axis is specified, plot on that axis
        else:
            axis = self.get_single_axes(on_axis)
            # If subplot data is provided, use it for the density plot
            if self._subplot_datas is not None:
                # Get the corresponding subplot idx to match up to the data
                ax_idx = self._get_ax_idx_from_on_axis(on_axis)
                if ax_idx < len(self._subplot_datas):
                    density_params["data"] = self._subplot_datas[ax_idx]
                    self._record_density_label(ax_idx, density_params)
                else:
                    logger.info("Axis index exceeds available subplot data.")
                    return self
            self._sns_density(
                axis=axis,
                include_outline=include_outline,
                density_params=density_params,
            )
            self._record_density_label(0, density_params)
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
    ) -> ISOPlot:
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
                    if i >= len(self._subplot_datas):
                        logger.info("Axis index exceeds available subplot data.")
                        break
                    simple_density_params["data"] = self._subplot_datas[i]
                self._sns_density(
                    axis=axis,
                    include_outline=include_outline,
                    density_params=simple_density_params,
                )
                # Record the label for the subplot
                self._record_density_label(i, simple_density_params)

        # If an axis is specified, plot on that axis
        else:
            axis = self.get_single_axes(on_axis)
            # If subplot data is provided, use it for the density plot
            if self._subplot_datas is not None:
                # Get the corresponding subplot idx to match up to the data
                ax_idx = self._get_ax_idx_from_on_axis(on_axis)
                if ax_idx < len(self._subplot_datas):
                    simple_density_params["data"] = self._subplot_datas[ax_idx]

                    # Record the label for the subplot
                    self._record_density_label(ax_idx, simple_density_params)
                else:
                    logger.info("Axis index exceeds available subplot data.")
                    return self
            self._sns_density(
                axis=axis,
                include_outline=include_outline,
                density_params=simple_density_params,
            )
            # Record the label for the main plot
            # (if no subplot data is provided)
            self._record_density_label(0, simple_density_params)

        return self

    def _prepare_spi_data(
        self,
        spi_data: pd.DataFrame | np.ndarray | None,
        spi_params: DirectParams | CentredParams | None,
        n: int,
        kwargs: dict,
    ) -> pd.DataFrame:
        """
        Validate and prepare SPI data from either direct data or parameters.

        Parameters
        ----------
        spi_data : pd.DataFrame | np.ndarray | None
            Data to use for SPI plotting
        spi_params : DirectParams | CentredParams | None
            Parameters to generate SPI data
        n : int
            Number of samples to generate if using spi_params
        kwargs : dict
            Additional parameters

        Returns
        -------
        pd.DataFrame
            Prepared data for SPI plotting

        """
        from soundscapy.spi.msn import MultiSkewNorm

        # Input validation
        if spi_data is not None and spi_params is not None:
            msg = "Please provide either spi_data or spi_params, not both."
            raise ValueError(msg)

        # Generate data from parameters if provided
        if spi_params is not None:
            spi_msn = MultiSkewNorm.from_params(spi_params)
            sample_data = spi_msn.sample(n=n, return_sample=True)
            return pd.DataFrame(sample_data, columns=[self.x, self.y])

        if spi_data is not None:
            # Process provided data
            return self._process_spi_data(spi_data, kwargs)
        msg = (
            "No data provided for SPI plot. "
            "Please provide either spi_data or spi_params."
        )
        raise ValueError(msg)

    def _process_spi_data(
        self, spi_data: pd.DataFrame | np.ndarray, kwargs: dict
    ) -> pd.DataFrame:
        """
        Process SPI data into standard format.

        Parameters
        ----------
        spi_data : pd.DataFrame | np.ndarray
            Data to process
        kwargs : dict
            Additional parameters with x and y column names

        Returns
        -------
        pd.DataFrame
            Processed data in standard format

        """
        xcol = kwargs.get("x", self.x)
        ycol = kwargs.get("y", self.y)

        if not (isinstance(xcol, str) and isinstance(ycol, str)):
            msg = "Sorry, at the moment in this method, x and y must be strings."
            raise TypeError(msg)

        # DataFrame handling
        if isinstance(spi_data, pd.DataFrame):
            if xcol not in spi_data.columns or ycol not in spi_data.columns:
                spi_data = spi_data.rename(columns={xcol: self.x, ycol: self.y})
            self._valid_density(spi_data)
            return spi_data

        # Numpy array handling
        if isinstance(spi_data, np.ndarray):
            if len(spi_data.shape) != 2 or spi_data.shape[1] != 2:  # noqa: PLR2004
                msg = "Invalid shape for SPI data. Expected a 2D array with 2 columns."
                raise ValueError(msg)
            spi_data = pd.DataFrame(spi_data, columns=[self.x, self.y])
            self._valid_density(spi_data)
            return spi_data

        msg = "Invalid SPI data type. Expected DataFrame or numpy array."
        raise TypeError(msg)

    def _prepare_spi_simple_density_params(
        self,
        label: str,
        spi_simple_dens_args: DensityParamTypes,
    ) -> DensityParamTypes:
        """
        Prepare parameters for SPI density plotting.

        Parameters
        ----------
        label : str
            Label for the plot
        kwargs : dict
            Additional parameters

        Returns
        -------
        dict
            Parameters for density plotting

        """
        spi_density_params = copy.deepcopy(self._simple_density_params)
        spi_density_params.update(color="r")
        spi_density_params.update(**spi_simple_dens_args)
        spi_density_params.update(
            data=self._spi_data, x=self.x, y=self.y, label=label, hue=None, palette=None
        )
        return spi_density_params

    def _show_score_on_axis(
        self,
        axis: Axes,
        ax_idx: int,
        show_score: Literal["on axis", "under title"],
        axis_text_kw: dict[str, Any] | None = None,
    ) -> None:
        """Add the SPI score to the specified axis."""
        from soundscapy.spi import spi_score

        # If subplot data is provided, use it for the spi_score
        if self._subplot_datas is not None:
            if ax_idx >= len(self._subplot_datas):
                logger.info("Axis index exceeds available subplot data.")
                return
            test_data = self._subplot_datas[ax_idx][[self.x, self.y]]
        else:
            if self._data is None:
                msg = "Data is not available for SPI to score against the target. "
                raise ValueError(msg)
            test_data = self._data[[self.x, self.y]]
        spi_val = spi_score(target=self._spi_data, test=test_data)  # type: ignore[reportArgumentType]

        if show_score == "under title":
            self._add_spi_score_under_title(axis, ax_idx, spi_val)
        if show_score == "on axis":
            # Add the SPI score to the plot
            self._add_spi_score_as_text(
                axis=axis,
                spi_val=spi_val,
                text_kw=axis_text_kw,
            )

    @staticmethod
    def _add_spi_score_as_text(
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

    def _add_spi_score_under_title(self, axis: Axes, ax_idx: int, spi_val: int) -> None:
        # Add the SPI score to the plot title
        if self._subplot_titles is not None:
            self._subplot_titles[ax_idx] = (
                f"{self._subplot_titles[ax_idx]}\nSPI: {spi_val}"
            )
        else:
            axis.set_title(f"{axis.get_title()}\nSPI: {spi_val}")

    def _get_ax_idx_from_on_axis(self, on_axis: int | tuple[int, int]) -> int:
        """Convert on_axis parameter to a flat index."""
        return (
            on_axis
            if isinstance(on_axis, int)
            else ((on_axis[0] + 1) * (on_axis[1] + 1) - 1)
        )

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
    ) -> ISOPlot:
        """
        Add a SPI plot to the existing axes.

        Parameters
        ----------
            spi_data : The data to plot.
            spi_params : Parameters to generate SPI data.
            n : Number of samples to generate if using parameters.
            on_axis : Axis to plot on (None for all axes).
            label : Label for the SPI plot.
            show_score : How to display the SPI score.
            axis_text_kw : Text parameters for on-axis display.
            **kwargs: Additional parameters for density plotting.

        Returns
        -------
            ISOPlot: The current plot instance for chaining.

        Examples
        --------
        >>> import soundscapy as sspy
        >>> from soundscapy.plotting import ISOPlot
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
                ISOPlot(sub_df)
                .create_subplots(subplot_by="LocationID", auto_allocate_axes=True)
                .add_scatter()
                .add_simple_density(label="Test")
                .add_spi_simple_density(
                    spi_params=spi, label="Target", show_score="on axis"
                )
                .apply_styling(
                    diagonal_lines=True, legend_loc="lower left"
                )
            )
        >>> spi_p.show() # doctest: +SKIP

        """
        try:
            from soundscapy.spi.msn import (
                CentredParams,  # noqa: F401
                DirectParams,  # noqa: F401
                MultiSkewNorm,  # noqa: F401
                spi_score,  # noqa: F401
            )
        except ImportError as e:
            msg = (
                "SPI functionality requires additional dependencies. "
                "Install with: pip install soundscapy[spi]"
            )
            raise ImportError(msg) from e

        # Prepare the SPI data
        self._spi_data = self._prepare_spi_data(spi_data, spi_params, n, kwargs)  # type: ignore[reportArgumentType]
        self._check_for_axes()

        # Prepare density parameters
        spi_density_params = self._prepare_spi_simple_density_params(label, kwargs)  # type: ignore[reportCallIssue]

        # TODO(MitchellAcoustics): Feature - collect SPI scores into a table  # noqa: E501, TD003
        # Thinking this should not be returned from this method, but just attached
        # to the ISOPlot object and retrieved with a get_spi_scores() method.
        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                if self._subplot_datas is not None:  # noqa: SIM102
                    if i >= len(self._subplot_datas):
                        logger.info("Axis index exceeds available subplot data.")
                        break

                # Create the plot
                self._sns_density(
                    axis=axis, include_outline=True, density_params=spi_density_params
                )
                # Record the label for the subplot
                self._record_density_label(i, spi_density_params)

                if show_score:
                    self._show_score_on_axis(
                        axis, ax_idx=i, show_score=show_score, axis_text_kw=axis_text_kw
                    )

        # If no axis is specified, plot on all axes
        # in the figure (if multiple subplots exist)
        else:
            axis = self.get_single_axes(on_axis)
            # Create the plot
            self._sns_density(axis=axis, density_params=spi_density_params)
            # Record the label for the main plot
            self._record_density_label(0, spi_density_params)

            if show_score:
                self._show_score_on_axis(
                    axis=axis,
                    ax_idx=0,
                    show_score=show_score,
                    axis_text_kw=axis_text_kw,
                )

        return self

    def apply_styling(
        self,
        **kwargs: Unpack[StyleParamsTypes],
    ) -> ISOPlot:
        """
        Apply styling to the Seaborn plot.

        Returns
        -------
            ISOPlot: The current plot instance for chaining.

        """
        self._style_params.update(**kwargs)
        self._check_for_axes()

        self._set_style()
        self._circumplex_grid()
        self._set_title()
        self._set_axes_titles()
        self._primary_labels()
        if self._style_params.get("primary_lines"):
            self._primary_lines()
        if self._style_params.get("diagonal_lines"):
            self._diagonal_lines_and_labels()

        try:
            self._add_density_labels()
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Could not add density labels. Error: {e}",
                UserWarning,
                stacklevel=2,
            )

        if self._style_params.get("legend_loc") is not False:
            # NOTE: Should really check for the presence of a legend.
            # If hue is added in the .add_* methods,
            # it doesn't show up in the class attributes.
            self._move_legend()

        return self

    def _set_style(self) -> None:
        """Set the overall style for the plot."""
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    def _circumplex_grid(self) -> ISOPlot:
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

    def _set_title(self) -> ISOPlot:
        """Set the title of the plot."""
        if self.title and self._has_subplots:
            figure = self.get_figure()
            figure.suptitle(
                self.title, fontsize=self._style_params.get("title_fontsize")
            )
        elif self.title and not self._has_subplots:
            axis = self.get_single_axes()
            if axis.get_title() == "":
                axis.set_title(
                    self.title, fontsize=self._style_params.get("title_fontsize")
                )
            else:
                figure = self.get_figure()
                figure.suptitle(
                    self.title, fontsize=self._style_params.get("title_fontsize")
                )
        return self

    def _set_axes_titles(self) -> ISOPlot:
        """Set the titles of the subplots plot."""
        if self._has_subplots and self._subplot_titles is not None:
            for i, axis in enumerate(self.yield_axes_objects()):
                if i >= len(self._subplot_titles):
                    logger.info("Axis index exceeds available subplot titles.")
                    break
                axis.set_title(self._subplot_titles[i])
        return self

    def _add_density_labels(self) -> ISOPlot:
        """
        See https://github.com/mwaskom/seaborn/issues/3523 .

        This is a workaround to add labels to the density plots.
        """
        for i, axis in enumerate(self.yield_axes_objects()):
            labels = self._subplot_labels[i]
            if (
                labels is not None
                and len([lab for lab in labels if lab is not None]) > 0
            ):
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

    def _move_legend(self) -> ISOPlot:
        """Move the legend to the specified location."""
        for i, axis in enumerate(self.yield_axes_objects()):
            old_legend = axis.get_legend()
            if old_legend is None:
                logger.debug("_move_legend: No legend found for axis %s", i)
                continue

            # Get handles and filter out None values
            handles = [
                h for h in old_legend.legend_handles if isinstance(h, Artist | tuple)
            ]
            # Skip if no valid handles remain
            if not handles:
                logger.warning(
                    "_move_legend: No valid handles found in legend for axis %s", i
                )
                continue

            labels = [t.get_text() for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()
            # Ensure labels and handles match in length
            if len(handles) != len(labels):
                labels = labels[: len(handles)]

            axis.legend(
                handles,
                labels,
                loc=self._style_params.get("legend_loc"),
                title=title,
            )
        return self

    def _primary_lines(self) -> ISOPlot:
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

    def _primary_labels(self) -> ISOPlot:
        """Handle the default labels for the x and y axes."""
        xlabel = self._style_params.get("xlabel")
        ylabel = self._style_params.get("ylabel")
        xlabel = self.x if xlabel is None else xlabel
        ylabel = self.y if ylabel is None else ylabel

        for _, axis in enumerate(self.yield_axes_objects()):
            axis.set_xlabel(
                xlabel, fontdict=self._style_params.get("label_fontdict")
            ) if xlabel is not False else axis.xaxis.label.set_visible(False)

            axis.set_ylabel(
                ylabel, fontdict=self._style_params.get("label_fontdict")
            ) if ylabel is not False else axis.yaxis.label.set_visible(False)

        return self

    def _diagonal_lines_and_labels(self) -> ISOPlot:
        """Add diagonal lines and labels to the plot."""
        for _, axis in enumerate(self.yield_axes_objects()):
            xlim = self._style_params.get("xlim", DEFAULT_XLIM)
            ylim = self._style_params.get("ylim", DEFAULT_YLIM)
            axis.plot(
                xlim,
                ylim,
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                lw=self._style_params.get("linewidth"),
                zorder=self._style_params.get("diag_lines_zorder"),
            )
            axis.plot(
                xlim,
                ylim[::-1],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                lw=self._style_params.get("linewidth"),
                zorder=self._style_params.get("diag_lines_zorder"),
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
                zorder=self._style_params.get("diag_labels_zorder"),
            )
            axis.text(
                xlim[0] / 2,
                ylim[1] / 2,
                "(chaotic)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params.get("diag_labels_zorder"),
            )
            axis.text(
                xlim[0] / 2,
                ylim[0] / 2,
                "(monotonous)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params.get("diag_labels_zorder"),
            )
            axis.text(
                xlim[1] / 2,
                ylim[0] / 2,
                "(calm)",
                ha="center",
                va="center",
                fontdict=diag_ax_font,
                zorder=self._style_params.get("diag_labels_zorder"),
            )
        return self

    def iso_annotation(
        self,
        location_idx: int,
        x_adj: float = 0,
        y_adj: float = 0,
        **kwargs,  # noqa: ANN003
    ) -> ISOPlot:
        """Add an annotation to the plot (only for Seaborn backend)."""
        warnings.warn(
            "This method is deprecated / not implemented. "
            "Most likely it will not be added and will break things.",
            UserWarning,
            stacklevel=2,
        )
        x = self._data[self.x].iloc[location_idx]  # type: ignore[reportOptionalSubscript]
        y = self._data[self.y].iloc[location_idx]  # type: ignore[reportOptionalSubscript]
        if isinstance(self.axes, np.ndarray):
            for i, _ in enumerate(self.axes.flatten()):
                self.axes[i].annotate(
                    text=self._data.index[location_idx],  # type: ignore[reportOptionalMemberAccess]
                    xy=(x, y),
                    xytext=(x + x_adj, y + y_adj),
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "-", "ec": "black"},
                    **kwargs,
                )
        elif isinstance(self.axes, Axes):
            self.axes.annotate(
                text=self._data.index[location_idx],  # type: ignore[reportOptionalMemberAccess]
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
