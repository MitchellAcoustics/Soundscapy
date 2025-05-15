"""
Main module for creating circumplex plots using different backends.

Examples
--------
>>> from soundscapy import isd, surveys
>>> from soundscapy.plotting.iso_plot import ISOPlot
>>> df = isd.load()
>>> df = surveys.add_iso_coords(df)
>>> sub_df = isd.select_location_ids(df, ['CamdenTown', 'RegentsParkJapan'])
>>> isoplot = (
...    ISOPlot(data=sub_df, hue="SessionID")
...    .create_subplots(
...        subplot_by="LocationID",
...        auto_allocate_axes=True,
...        adjust_figsize=True
...    )
...    .add_scatter()
...    .add_simple_density(fill=False)
...    .style()
... )
>>> isoplot.show() # xdoctest: +SKIP

"""
# ruff: noqa: SLF001, G004

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from soundscapy.plotting.defaults import (
    DEFAULT_STYLE_PARAMS,
    DEFAULT_XCOL,
    DEFAULT_YCOL,
    RECOMMENDED_MIN_SAMPLES,
)
from soundscapy.plotting.layers import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPIDensityLayer,
    SPIScatterLayer,
    SPISimpleLayer,
)
from soundscapy.plotting.param_models import (
    DensityParams,
    ScatterParams,
    SimpleDensityParams,
    SPISimpleDensityParams,
    StyleParams,
    SubplotsParams,
)
from soundscapy.plotting.plot_context import PlotContext
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from soundscapy.plotting.param_models import SeabornPaletteType
    from soundscapy.spi.msn import (
        CentredParams,
        DirectParams,
    )

logger = get_logger()


class ExperimentalWarning(Warning):
    """A warning class to signify experimental features."""


class ISOPlot:
    """
    A class for creating circumplex plots using different backends.

    This class provides methods for creating scatter plots and density plots
    based on the circumplex model of soundscape perception.

    Examples
    --------
    >>> from soundscapy import isd, surveys
    >>> df = isd.load()
    >>> df = surveys.add_iso_coords(df)
    >>> ct = isd.select_location_ids(df, ["CamdenTown", "RegentsParkJapan"])
    >>> cp = (ISOPlot(ct, hue="LocationID")
    ...         .create_subplots()
    ...         .add_scatter()
    ...         .add_density()
    ...         .style())
    >>> cp.show() # xdoctest: +SKIP

    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        x: str | np.ndarray | pd.Series | None = "ISOPleasant",
        y: str | np.ndarray | pd.Series | None = "ISOEventful",
        title: str | None = "Soundscape Density Plot",
        hue: str | None = None,
        palette: SeabornPaletteType | None = "colorblind",
        figure: Figure | None = None,  # Removed SubFigure type, don't think we need it
        axes: Axes | np.ndarray | None = None,
    ) -> None:
        """
        Initialize a ISOPlot instance.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            The data to be plotted, by default None
        x : str | np.ndarray | pd.Series | None, optional
            Column name or data for x-axis, by default "ISOPleasant"
        y : str | np.ndarray | pd.Series | None, optional
            Column name or data for y-axis, by default "ISOEventful"
        title : str | None, optional
            Title of the plot, by default "Soundscape Density Plot"
        hue : str | None, optional
            Column name for color encoding, by default None
        palette : SeabornPaletteType | None, optional
            Color palette to use, by default "colorblind"
        figure : Figure | SubFigure | None, optional
            Existing figure to plot on, by default None
        axes : Axes | np.ndarray | None, optional
            Existing axes to plot on, by default None

        Examples
        --------
        Create a plot with default parameters:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...    columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot()
        >>> isinstance(plot, ISOPlot)
        True

        Create a plot with a DataFrame:

        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> plot = ISOPlot(data=data, hue='Group')
        >>> plot.hue
        'Group'


        Create a plot directly with arrays:

        >>> x, y = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100).T
        >>> plot = ISOPlot(x=x, y=y)
        >>> isinstance(plot, ISOPlot)
        True

        """
        warnings.warn(
            "`ISOPlot` is currently under development and should be considered "
            "experimental. `ISOPlot` implements an experimental API for creating "
            "layered soundscape circumplex plots. Use with caution.",
            ExperimentalWarning,
            stacklevel=2,
        )

        # Process and validate input data and coordinates
        data, x, y = self._check_data_x_y(data, x, y)
        self._check_data_hue(data, hue)

        # Initialize the main plot context
        self.main_context = PlotContext(
            data=data,
            x=x if isinstance(x, str) else DEFAULT_XCOL,
            y=y if isinstance(y, str) else DEFAULT_YCOL,
            hue=hue,
            title=title,
        )

        # Store additional plot attributes
        self.figure = figure
        self.axes = axes
        self.palette = palette

        # Initialize subplot management
        self.subplot_contexts: list[PlotContext] = []
        self.subplots_params = SubplotsParams()

        # Initialize parameter managers
        self._scatter_params = ScatterParams(
            data=data,
            x=self.main_context.x,
            y=self.main_context.y,
            hue=hue,
            palette=self.palette,
        )

        self._density_params = DensityParams(
            data=data,
            x=self.main_context.x,
            y=self.main_context.y,
            hue=hue,
            palette=self.palette,
        )

        self._simple_density_params = SimpleDensityParams(
            data=data,
            x=self.main_context.x,
            y=self.main_context.y,
            hue=hue,
        )

        self._spi_scatter_params = NotImplementedError
        self._spi_density_params = NotImplementedError
        self._spi_simple_density_params = SPISimpleDensityParams(
            x=self.main_context.x,
            y=self.main_context.y,
        )

        self._style_params = StyleParams()

        # SPI-related attributes
        self._spi_data = None

    @property
    def x(self) -> str:
        """Get the x-axis column name."""
        return self.main_context.x

    @property
    def y(self) -> str:
        """Get the y-axis column name."""
        return self.main_context.y

    @property
    def hue(self) -> str | None:
        """Get the hue column name."""
        return self.main_context.hue

    @property
    def title(self) -> str | None:
        """Get the plot title."""
        return self.main_context.title

    @property
    def _nrows(self) -> int:
        """Get the number of rows in the subplot grid."""
        return self.subplots_params.nrows

    @property
    def _ncols(self) -> int:
        """Get the number of columns in the subplot grid."""
        return self.subplots_params.ncols

    @property
    def _naxes(self) -> int:
        """Get the number of axes."""
        return self.subplots_params.n_subplots

    @property
    def _has_subplots(self) -> bool:
        """Check if the plot has subplots."""
        return self._naxes > 1

    @property
    def _data(self) -> pd.DataFrame | None:
        """Get the main data."""
        return self.main_context.data

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
        # NOTE: Move to PlotContext class?
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
            return data, xcol, ycol

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

        if isinstance(hue, str) and not (hue in data.columns or hue in data.index.name):
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
        auto_allocate_axes: bool = False,
        **kwargs,
    ) -> ISOPlot:
        """
        Create subplots for the circumplex plot.

        Parameters
        ----------
        nrows : int, optional
            Number of rows in the subplot grid, by default 1
        ncols : int, optional
            Number of columns in the subplot grid, by default 1
        figsize : tuple[int, int], optional
            Size of the figure (width, height), by default (5, 5)
        subplot_by : str | None, optional
            Column name to create subplots by unique values, by default None
        subplot_datas : list[pd.DataFrame] | None, optional
            List of dataframes for each subplot, by default None
        subplot_titles : list[str] | None, optional
            List of titles for each subplot, by default None
        adjust_figsize : bool, optional
            Whether to adjust the figure size based on nrows/ncols, by default True
        auto_allocate_axes : bool, optional
            Whether to automatically determine nrows/ncols based on data,
            by default False
        **kwargs :
            Additional parameters for plt.subplots

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Create a basic subplot grid:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> plot = ISOPlot(data=data).create_subplots(nrows=2, ncols=2)
        >>> len(plot.subplot_contexts) == 4
        True
        >>> plot.close()  # Clean up

        Create subplots by a column in the data:

        >>> plot = (ISOPlot(data=data)
        ...         .create_subplots(nrows=1, ncols=2, subplot_by='Group'))
        >>> len(plot.subplot_contexts) == 2
        True
        >>> plot.close()  # Clean up

        Create subplots with auto-allocation of axes:

        >>> plot = (ISOPlot(data=data)
        ...        .create_subplots(subplot_by='Group', auto_allocate_axes=True))
        >>> len(plot.subplot_contexts) == 2
        True
        >>> plot.close()  # Clean up

        """
        # Set up subplot params
        self.subplots_params.update(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            subplot_by=subplot_by,
            adjust_figsize=adjust_figsize,
            auto_allocate_axes=auto_allocate_axes,
            **kwargs,
        )
        # Create a list of dataframes and titles for each subplot
        # based on the unique values in the specified column
        if self.subplots_params.subplot_by:
            logger.debug(
                f"Creating subplots by unique values in {self.subplots_params.subplot_by}."
            )
            subplot_datas, subplot_titles, n_subplots_by = self._setup_subplot_by(
                self.subplots_params.subplot_by, subplot_datas, subplot_titles
            )
        else:
            n_subplots_by = -1

        if subplot_titles and self.subplots_params.auto_allocate_axes:
            # Attempt to allocate axes based on the number of subplots
            self.subplots_params.nrows, self.subplots_params.ncols = (
                self._allocate_subplot_axes(subplot_titles)
            )

        if adjust_figsize:
            self.subplots_params.figsize = (
                self.subplots_params.ncols * self.subplots_params.figsize[0],
                self.subplots_params.nrows * self.subplots_params.figsize[1],
            )

        logger.debug(f"Subplot parameters: {self.subplots_params}")

        # Create the figure and axes
        self.figure, self.axes = plt.subplots(
            **self.subplots_params.as_plt_subplots_args()
        )

        # If subplot_datas or subplot_titles are provided, validate them
        if subplot_datas is not None or subplot_titles is not None:
            self._validate_subplots_datas(subplot_datas, subplot_titles)

        # Create PlotContext objects for each subplot
        self.subplot_contexts = []

        for i, ax in enumerate(self.yield_axes_objects()):
            if i >= self._naxes:
                break
            if subplot_by and i >= n_subplots_by:
                logger.debug(f"Created {i + 1} subplots for {subplot_by}.")
                break
            # Get data and title for this subplot if available
            data = (
                subplot_datas[i] if subplot_datas and i < len(subplot_datas) else None
            )
            title = (
                subplot_titles[i]
                if subplot_titles and i < len(subplot_titles)
                else None
            )

            context = self.main_context.create_child(data=data, title=title, ax=ax)
            self.subplot_contexts.append(context)

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

    def close(self, fig: int | str | Figure | None = None) -> None:
        """
        Close the figure.

        This method is a wrapper around plt.close() to close the figure.

        """
        if fig is None:
            fig = self.figure
            if fig is None:
                msg = (
                    "No figure object provided. "
                    "Please create a figure using create_subplots() first."
                )
                raise ValueError(msg)
        plt.close(fig)

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        """
        Save the figure.

        This method is a wrapper around plt.savefig() to save the figure.

        """
        if self.figure is None:
            msg = (
                "No figure object provided. "
                "Please create a figure using create_subplots() first."
            )
            raise ValueError(msg)
        self.figure.savefig(*args, **kwargs)

    def _setup_subplot_by(
        self,
        subplot_by: str,
        subplot_datas: list[pd.DataFrame] | None,
        subplot_titles: list[str] | None,
    ) -> tuple[list[pd.DataFrame], list[str], int]:
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
        n_subplots_by = len(unique_values)
        logger.debug(f"Found {n_subplots_by + 1} unique values in '{subplot_by}'.")
        if n_subplots_by < 2:  # noqa: PLR2004
            warnings.warn(
                f"Only {n_subplots_by} unique values found in '{subplot_by}'. "
                "Subplots may not be meaningful.",
                UserWarning,
                stacklevel=2,
            )

        # Create a list of dataframes for each unique value
        subplot_datas = [
            full_data[full_data[subplot_by] == value] for value in unique_values
        ]

        # Create subplot titles based on the unique values
        if subplot_titles is None:
            subplot_titles = [str(value) for value in unique_values]
        elif len(subplot_titles) != n_subplots_by:
            msg = (
                "Number of subplot titles must match the number of unique values "
                f"for '{subplot_by}'. Got {len(subplot_titles)} titles and "
                f"{n_subplots_by} unique values."
            )
            raise ValueError(msg)
        else:
            # Keep the provided subplot titles
            msg = (
                "Not recommended to provide separate subplot titles when using "
                "subplot_by. Consider using the default titles based on unique values. "
                "Manual subplot_titles may not be in the same order as the data."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        return subplot_datas, subplot_titles, n_subplots_by

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

        Raises
        ------
        ValueError
            If the axes object does not exist.

        """
        if self.axes is None:
            msg = (
                "No axes object provided. "
                "Please create a figure and axes using create_subplots() first."
            )
            raise ValueError(msg)

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
        Get a specific axes object.

        Parameters
        ----------
        ax_idx : int | tuple[int, int] | None, optional
            The index of the axes to get. If None, returns the first axes.
            Can be an integer for flattened access or a tuple of (row, col).

        Returns
        -------
        Axes
            The requested matplotlib Axes object

        Raises
        ------
        ValueError
            If the axes object does not exist or the index is invalid.
        TypeError
            If the axes object is not a valid Axes or ndarray of Axes.

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

    def add_layer(
        self,
        layer_class: type[Layer],
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> ISOPlot:
        """
        Add a visualization layer, optionally targeting specific subplot(s).

        Parameters
        ----------
        layer_class : Layer subclass
            The type of layer to add
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes:
            - int: Index of subplot (flattened)
            - tuple: (row, col) coordinates
            - list: Multiple indices to apply the layer to
            - None: Apply to all subplots (default)
        data : pd.DataFrame, optional
            Custom data for this specific layer, overriding context data
        **params : dict
            Parameters for the layer

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a scatter layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting.layers import ScatterLayer
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> # Will create 2x2 subplots all with the same data
        >>> plot = (ISOPlot(data=data)
        ...         .create_subplots(nrows=2, ncols=2)
        ...         .add_layer(ScatterLayer)
        ...         .style())
        >>> plot.show() # xdoctest: +SKIP
        >>> all(len(ctx.layers) == 1 for ctx in plot.subplot_contexts)
            True
        >>> plot.close()  # Clean up

        Add a layer to a specific subplot:

        >>> plot = (ISOPlot(data=data)
        ...         .create_subplots(nrows=2, ncols=2)
        ...         .add_layer(ScatterLayer, on_axis=0)
        ...         .style())
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 1
        True
        >>> all(len(ctx.layers) == 0 for ctx in plot.subplot_contexts[1:])
        True
        >>> plot.close()

        Add a layer to multiple subplots:

        >>> plot = (ISOPlot(data=data)
        ...            .create_subplots(nrows=2, ncols=2)
        ...            .add_layer(ScatterLayer, on_axis=[0, 2])
        ...            .style())
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 1
        True
        >>> len(plot.subplot_contexts[2].layers) == 1
        True
        >>> len(plot.subplot_contexts[1].layers) == 0
        True
        >>> plot.close()

        Add a layer with custom data to a specific subplot:
        >>> custom_data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.1, 50),
        ...     'ISOEventful': rng.normal(0.15, 0.2, 50),
        ... })
        >>> plot = (ISOPlot(data=data)
        ...        .create_subplots(nrows=2, ncols=2)
        ...        .add_layer(ScatterLayer) # Add to all subplots
        ...        # Add a layer with custom data to the first subplot
        ...        .add_layer(ScatterLayer, data=data.iloc[:50], on_axis=0, color='red')
        ...        # Add a layer with custom data to the second subplot
        ...        .add_layer(ScatterLayer, data=custom_data, on_axis=1)
        ...        .style())
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.close()

        """
        # TODO(MitchellAcoustics): Need to handle legend/label creation   # noqa: TD003
        #                          for new data added to a specific subplot
        # Create the layer instance
        layer = layer_class(custom_data=data, **params)

        # Check if we have axes to render on
        self._check_for_axes()

        # If no subplots created yet, add to main context
        if not self.subplot_contexts:
            if self.main_context.ax is None:
                # Get the single axis and assign it to main context
                if isinstance(self.axes, Axes):
                    self.main_context.ax = self.axes
                elif isinstance(self.axes, np.ndarray) and self.axes.size > 0:
                    self.main_context.ax = self.axes.flatten()[0]

            # Add layer to main context
            self.main_context.layers.append(layer)
            # Render the layer immediately
            layer.render(self.main_context)
            return self

        # Handle various axis targeting options
        target_contexts = self._resolve_target_contexts(on_axis)
        logger.debug(f"N target contexts: {len(target_contexts)}")

        # Add the layer to each target context and render it
        for i, context in enumerate(target_contexts):
            if data is not None and i >= self.subplots_params.n_subplots_by > 0:
                # If custom data is provided, use it for the specific subplot
                break
            context.layers.append(layer)
            layer.render(context)

        return self

    def _resolve_target_contexts(
        self, on_axis: int | tuple[int, int] | list[int] | None
    ) -> list[PlotContext]:
        """
        Resolve which subplot contexts to target based on axis specification.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None
            The axis specification:
            - None: All subplot contexts
            - int: Single subplot at flattened index
            - tuple[int, int]: Subplot at (row, col)
            - list[int]: Multiple subplots at specified indices

        Returns
        -------
        list[PlotContext]
            List of target subplot contexts

        """
        # If no specific axis, target all subplot contexts
        if on_axis is None:
            return self.subplot_contexts

        # Convert axis specification to list of indices
        indices = self._resolve_axis_indices(on_axis)

        # Get the contexts for each valid index
        target_contexts = []
        for idx in indices:
            if 0 <= idx < len(self.subplot_contexts):
                target_contexts.append(self.subplot_contexts[idx])
            else:
                msg = f"Subplot index {idx} out of range"
                raise IndexError(msg)

        return target_contexts

    def _resolve_axis_indices(
        self, on_axis: int | tuple[int, int] | list[int]
    ) -> list[int]:
        """
        Convert axis specification to list of indices.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int]
            The axis specification to resolve

        Returns
        -------
        list[int]
            List of flattened indices

        Raises
        ------
        ValueError
            If an invalid axis specification is provided

        """
        if isinstance(on_axis, int):
            return [on_axis]
        if isinstance(on_axis, tuple) and len(on_axis) == 2:  # noqa: PLR2004
            # Convert (row, col) to flattened index
            row, col = on_axis
            return [row * self._ncols + col]
        if isinstance(on_axis, list):
            return on_axis
        msg = f"Invalid axis specification: {on_axis}"
        raise ValueError(msg)

    def add_scatter(
        self,
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> ISOPlot:
        """
        Add a scatter layer to specific subplot(s).

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes
        data : pd.DataFrame, optional
            Custom data for this specific scatter plot
        **params : dict
            Parameters for the scatter plot

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a scatter layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> plot = (ISOPlot(data=data)
        ...           .create_subplots(nrows=2, ncols=1)
        ...           .add_scatter(s=50, alpha=0.7, hue='Group')
        ...           .style())
        >>> plot.show() # xdoctest: +SKIP
        >>> all(len(ctx.layers) == 1 for ctx in plot.subplot_contexts)
        True
        >>> plot.close()  # Clean up

        Add a scatter layer with custom data to a specific subplot:

        >>> custom_data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.1, 50),
        ...     'ISOEventful': rng.normal(0.15, 0.2, 50),
        ... })
        >>> plot = (ISOPlot(data=data)
        ...            .create_subplots(nrows=2, ncols=1)
        ...            .add_scatter(hue='Group')
        ...            .add_scatter(on_axis=0, data=custom_data, color='red')
        ...            .style())
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.subplot_contexts[0].layers[1].custom_data is custom_data
        True
        >>> plot.close()  # Clean up

        """
        # Merge default scatter parameters with provided ones
        # Remove data from scatter_params to avoid conflict
        scatter_params = self._scatter_params.copy()
        scatter_params.drop("data")
        scatter_params.update(**params)

        return self.add_layer(
            ScatterLayer,
            data=data,
            on_axis=on_axis,
            **scatter_params.as_dict(drop=["data"]),
        )

    def add_spi(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        spi_target_data: pd.DataFrame | np.ndarray | None = None,
        msn_params: DirectParams | CentredParams | None = None,
        *,
        layer_class: type[Layer] = SPISimpleLayer,
        **params: Any,
    ) -> ISOPlot:
        """
        Add a SPI layer to specific subplot(s).

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes
        spi_target_data : pd.DataFrame | np.ndarray | None, optional
            Custom data for this specific SPI plot
        msn_params : DirectParams | CentredParams | None, optional
            Parameters for the SPI plot

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a SPI layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.spi import DirectParams
        >>> rng = np.random.default_rng(42)
        >>>    # Create a DataFrame with random data
        >>> data = pd.DataFrame(
        ...    rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...    columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>>    # Define MSN parameters for the SPI target
        >>> msn_params = DirectParams(
        ...     xi=np.array([0.5, 0.7]),
        ...     omega=np.array([[0.1, 0.05], [0.05, 0.1]]),
        ...     alpha=np.array([0, -5]),
        ...     )
        >>>    # Create the plot with only an SPI layer
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_spi(msn_params=msn_params)
        ...     .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 2
        True
        >>> plot.close()  # Clean up

        Add an SPI layer over top of 'real' data:
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_density()
        ...     .add_spi(msn_params=msn_params, show_score="on axis")
        ...     .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 3
        True

        Add a SPI layer from spi data:
        >>> # Create a custom distribution
        >>> from soundscapy.spi import MultiSkewNorm
        >>> import soundscapy as sspy
        >>> spi_msn = MultiSkewNorm.from_params(msn_params)
        >>> # Generate random samples
        >>> spi_msn.sample(1000)
        >>> data = sspy.add_iso_coords(sspy.isd.load())
        >>> data = sspy.isd.select_location_ids(
        ...     data,
        ...     ['CamdenTown', 'PancrasLock', 'RussellSq', 'RegentsParkJapan']
        ... )

        >>> mp3 = (
        ...     ISOPlot(
        ...         data=data,
        ...         title="Soundscape Density Plots with corrected ISO coordinates",
        ...         hue="SessionID",
        ...     )
        ...     .create_subplots(
        ...         subplot_by="LocationID",
        ...         figsize=(4, 4),
        ...         auto_allocate_axes=True,
        ...     )
        ...     .add_scatter()
        ...     .add_simple_density(fill=False)
        ...     .add_spi(spi_target_data=spi_msn.sample_data, show_score="under title")
        ...     .style()
        ... )
        >>> mp3.show() # xdoctest: +SKIP
        >>> plot.close()  # Clean up

        # BUG: This last doctest doesn't show the spi score under the title

        """
        if layer_class == SPISimpleLayer:
            spi_simple_params = self._spi_simple_density_params.copy()
            spi_simple_params.drop("data")
            spi_simple_params.update(**params)

            return self.add_layer(
                layer_class,
                on_axis=on_axis,
                msn_params=msn_params,
                spi_target_data=spi_target_data,
                **spi_simple_params.as_dict(drop=["data"]),
            )
        if layer_class in (SPIDensityLayer, SPIScatterLayer):
            msg = (
                "Only the simple density layer type is currently supported for "
                "SPI plots. Please use SPISimpleLayer"
            )
            raise NotImplementedError(msg)

        msg = "Invalid layer class provided. Expected SPISimpleLayer. "
        raise ValueError(msg)

    def add_density(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        *,
        include_outline: bool = False,
        **params: Any,
    ) -> ISOPlot:
        """
        Add a density layer to specific subplot(s).

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes
        data : pd.DataFrame, optional
            Custom data for this specific density plot
        include_outline : bool, optional
            Whether to include an outline around the density plot, by default False
        **params : dict
            Parameters for the density plot

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a density layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.25, 50),
        ...     'ISOEventful': rng.normal(0.15, 0.4, 50),
        ... })
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_density()
        ...     .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 1
        True
        >>> plot.close()  # Clean up

        Add a density layer with custom settings:

        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_density(levels=5, alpha=0.7)
        ...     .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 1
        True
        >>> plot.close()  # Clean up

        """
        # Merge default density parameters with provided ones
        density_params = self._density_params.copy()
        density_params.drop("data")
        density_params.update(**params)

        return self.add_layer(
            DensityLayer,
            data=data,
            on_axis=on_axis,
            include_outline=include_outline,
            **density_params.as_dict(drop=["data"]),
        )

    def add_simple_density(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        *,
        include_outline: bool = True,
        **params: Any,
    ) -> ISOPlot:
        """
        Add a simple density layer to specific subplot(s).

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes
        data : pd.DataFrame, optional
            Custom data for this specific density plot
        thresh : float, optional
            Threshold for density contours, by default 0.5
        levels : int | Iterable[float], optional
            Contour levels, by default 2
        alpha : float, optional
            Transparency level, by default 0.5
        include_outline : bool, optional
            Whether to include an outline around the density plot, by default True
        **params : dict
            Additional parameters for the density plot

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a simple density layer:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.25, 30),
        ...     'ISOEventful': rng.normal(0.15, 0.4, 30),
        ... })
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_simple_density()
        ...     .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 2
        True
        >>> plot.close()  # Clean up

        Add a simple density with splitting by group:
        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> plot = (
        ...     ISOPlot(data=data, hue='Group')
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_simple_density()
        ...     .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 2
        True
        >>> plot.close()
        ...

        """
        # Merge default simple density parameters with provided ones
        simple_density_params = self._simple_density_params.copy()
        simple_density_params.drop("data")
        simple_density_params.update(**params)

        return self.add_layer(
            SimpleDensityLayer,
            on_axis=on_axis,
            data=data,
            include_outline=include_outline,
            **simple_density_params.as_dict(drop=["data"]),
        )

    def add_annotation(
        self,
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float],
        arrowprops: dict[str, Any] | None = None,
    ) -> ISOPlot:
        """
        Add an annotation to the plot.

        Parameters
        ----------
        text : str
            The text to display in the annotation.
        xy : tuple[float, float]
            The point to annotate.
        xytext : tuple[float, float]
            The point at which to place the text.
        arrowprops : dict[str, Any] | None, optional
            Properties for the arrow connecting the annotation text to the point.

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        """
        msg = "AnnotationLayer is not yet implemented. "
        raise NotImplementedError(msg)
        # TODO(MitchellAcoustics): Implement AnnotationLayer  # noqa: TD003
        return self.add_layer(
            "AnnotationLayer",
            text=text,
            xy=xy,
            xytext=xytext,
            arrowprops=arrowprops,
        )

    def style(
        self,
        **kwargs: Any,
    ) -> ISOPlot:
        """
        Apply styling to the plot.

        Parameters
        ----------
        **kwargs: Styling parameters to override defaults

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Apply styling with default parameters:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # Create simple data for styling example
        >>> data = pd.DataFrame(
        ...     np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...             rng.integers(1, 3, 100)],
        ...     columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> # Create plot with default styling
        >>> plot = (
        ...    ISOPlot(data=data)
        ...       .create_subplots()
        ...       .add_scatter()
        ...       .style()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.get_figure() is not None
        True
        >>> plot.close()  # Clean up

        Apply styling with custom parameters:

        >>> plot = (
        ...         ISOPlot(data=data)
        ...         .create_subplots()
        ...         .add_scatter()
        ...         .style(xlim=(-2, 2), ylim=(-2, 2), primary_lines=False)
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.get_figure() is not None
        True
        >>> plot.close()  # Clean up

        Demonstrate the fluent interface (method chaining):

        >>> # Create plot with method chaining
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots(nrows=1, ncols=1)
        ...     .add_scatter(alpha=0.7)
        ...     .add_density(levels=5)
        ...     .style(title_fontsize=14)
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> # Verify results
        >>> isinstance(plot, ISOPlot)
        True
        >>> plot.close()  # Clean up

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

        if self._style_params.get("legend_loc") is not False:
            self._move_legend()

        return self

    @staticmethod
    def _set_style() -> None:
        """Set the overall style for the plot."""
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    def _circumplex_grid(self) -> None:
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

    def _set_title(self) -> None:
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

    def _set_axes_titles(self) -> None:
        """Set the titles of the subplots."""
        for context in self.subplot_contexts:
            if context.ax and context.title:
                context.ax.set_title(context.title)

    def _primary_lines(self) -> None:
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

    def _primary_labels(self) -> None:
        """Handle the default labels for the x and y axes."""
        xlabel = self._style_params.get("xlabel")
        ylabel = self._style_params.get("ylabel")

        xlabel = self.x if xlabel is None else xlabel
        ylabel = self.y if ylabel is None else ylabel
        fontdict = self._style_params.get("prim_ax_fontdict")

        # BUG: For some reason, this ruins the sharex and sharey
        #       functionality, but only when a layer is applied
        #       a specific subplot.
        for _, axis in enumerate(self.yield_axes_objects()):
            axis.set_xlabel(
                xlabel, fontdict=fontdict
            ) if xlabel is not False else axis.xaxis.label.set_visible(False)

            axis.set_ylabel(
                ylabel, fontdict=fontdict
            ) if ylabel is not False else axis.yaxis.label.set_visible(False)

    def _diagonal_lines_and_labels(self) -> None:
        """
        Add diagonal lines and labels to the plot.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...    columns=['ISOPleasant', 'ISOEventful'])
        >>> # Create a plot with diagonal lines and labels
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .style(diagonal_lines=True)
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.close('all')

        """
        for _, axis in enumerate(self.yield_axes_objects()):
            xlim = self._style_params.get("xlim", DEFAULT_STYLE_PARAMS["xlim"])
            ylim = self._style_params.get("ylim", DEFAULT_STYLE_PARAMS["ylim"])
            axis.plot(
                xlim,
                ylim,
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                lw=self._style_params.get("linewidth"),
                zorder=self._style_params.get("diag_lines_zorder"),
            )
            logger.debug("Plotting diagonal line for axis.")
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

    def _move_legend(self) -> None:
        """Move the legend to the specified location."""
        for i, axis in enumerate(self.yield_axes_objects()):
            old_legend = axis.get_legend()
            if old_legend is None:
                # logger.debug("_move_legend: No legend found for axis %s", i)
                continue

            # Get handles and filter out None values
            handles = [
                h for h in old_legend.legend_handles if isinstance(h, Artist | tuple)
            ]
            # Skip if no valid handles remain
            if not handles:
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
