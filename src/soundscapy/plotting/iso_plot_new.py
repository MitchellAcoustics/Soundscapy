"""
Main module for creating circumplex plots using different backends.

Example:
-------
>>> from soundscapy import isd, surveys
>>> from soundscapy.plotting.iso_plot_new import ISOPlot
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
# ruff: noqa: SLF001, G004

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

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
from matplotlib.figure import Figure, SubFigure

from soundscapy.plotting.defaults import (
    DEFAULT_DENSITY_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_SUBPLOTS_PARAMS,
    DEFAULT_XCOL,
    DEFAULT_YCOL,
    RECOMMENDED_MIN_SAMPLES,
)
from soundscapy.plotting.layers import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
)
from soundscapy.plotting.plot_context import PlotContext
from soundscapy.plotting.plot_params import PlotParams
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
        """
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
        self.palette = self._crosscheck_palette_hue(palette, hue)

        # Initialize subplot management
        self._has_subplots = False
        self._nrows = 1
        self._ncols = 1
        self._naxes = 1
        self.subplot_contexts: list[PlotContext] = []

        # Initialize parameter managers
        self._scatter_params = PlotParams(
            "scatter",
            data=data,
            x=self.main_context.x,
            y=self.main_context.y,
            hue=hue,
            palette=self.palette,
        )

        self._density_params = PlotParams(
            "density",
            data=data,
            x=self.main_context.x,
            y=self.main_context.y,
            hue=hue,
            palette=self.palette,
        )

        self._simple_density_params = PlotParams("simple_density")
        self._style_params = PlotParams("style")

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
    def hue(self) -> Optional[str]:
        """Get the hue column name."""
        return self.main_context.hue

    @property
    def title(self) -> Optional[str]:
        """Get the plot title."""
        return self.main_context.title

    @property
    def _data(self) -> Optional[pd.DataFrame]:
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
        auto_allocate_axes: bool = False,
        **kwargs: Unpack[SubplotsParamsTypes],
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
            Whether to automatically determine nrows/ncols based on data, by default False
        **kwargs :
            Additional parameters for plt.subplots

        Returns
        -------
        ISOPlot
            The current plot instance for chaining
        """
        self.figsize = figsize

        # Set up subplot parameters
        subplot_params = PlotParams("subplots", **kwargs)

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

        # Create the figure and axes
        self.figure, self.axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=self.figsize, **subplot_params.as_dict()
        )

        # Store subplot configuration
        self._nrows = nrows
        self._ncols = ncols
        self._naxes = nrows * ncols
        self._has_subplots = self._naxes > 1

        # If subplot_datas or subplot_titles are provided, validate them
        if subplot_datas is not None or subplot_titles is not None:
            self._validate_subplots_datas(subplot_datas, subplot_titles)

        # Create PlotContext objects for each subplot
        self.subplot_contexts = []

        # Helper function to get ax from flattened or 2D array
        def get_axes_at_index(idx: int) -> Optional[Axes]:
            if isinstance(self.axes, Axes):
                return self.axes if idx == 0 else None
            elif isinstance(self.axes, np.ndarray):
                if self.axes.ndim == 1:
                    return self.axes[idx] if idx < len(self.axes) else None
                else:
                    # 2D array of axes
                    flat_axes = self.axes.flatten()
                    return flat_axes[idx] if idx < len(flat_axes) else None
            return None

        # Create context for each subplot
        for i in range(self._naxes):
            # Get data and title for this subplot if available
            data = (
                subplot_datas[i]
                if subplot_datas and i < len(subplot_datas)
                else self._data
            )
            title = (
                subplot_titles[i]
                if subplot_titles and i < len(subplot_titles)
                else None
            )

            # Get the axis for this subplot
            ax = get_axes_at_index(i)

            # Create a child context for this subplot
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
                "Not recommended to provide separate subplot titles when using subplot_by. "
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
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
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
        """
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

        # Add the layer to each target context and render it
        for context in target_contexts:
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
                raise IndexError(f"Subplot index {idx} out of range")

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
        elif isinstance(on_axis, tuple) and len(on_axis) == 2:
            # Convert (row, col) to flattened index
            row, col = on_axis
            return [row * self._ncols + col]
        elif isinstance(on_axis, list):
            return on_axis
        else:
            raise ValueError(f"Invalid axis specification: {on_axis}")

    def add_scatter(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        **params: Unpack[ScatterParamTypes],
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
        """
        # Merge default scatter parameters with provided ones
        scatter_params = self._scatter_params.as_dict()
        scatter_params.update(params)

        return self.add_layer(
            ScatterLayer, on_axis=on_axis, data=data, **scatter_params
        )

    def add_density(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        *,
        include_outline: bool = False,
        **params: Unpack[DensityParamTypes],
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
        """
        # Merge default density parameters with provided ones
        density_params = self._density_params.as_dict()
        density_params.update(params)

        return self.add_layer(
            DensityLayer,
            on_axis=on_axis,
            data=data,
            include_outline=include_outline,
            **density_params,
        )

    def add_simple_density(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        thresh: float = 0.5,
        levels: int | Iterable[float] = 2,
        alpha: float = 0.5,
        *,
        include_outline: bool = True,
        **params: Unpack[DensityParamTypes],
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
        """
        # Merge default simple density parameters with provided ones
        simple_density_params = self._simple_density_params.as_dict()
        simple_density_params.update(thresh=thresh, levels=levels, alpha=alpha)
        simple_density_params.update(params)

        return self.add_layer(
            SimpleDensityLayer,
            on_axis=on_axis,
            data=data,
            include_outline=include_outline,
            **simple_density_params,
        )

    def apply_styling(
        self,
        **kwargs: Unpack[StyleParamsTypes],
    ) -> ISOPlot:
        """
        Apply styling to the plot.

        Parameters
        ----------
        **kwargs: Styling parameters to override defaults

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

        if self._style_params.get("legend_loc") is not False:
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
        """Set the titles of the subplots."""
        for context in self.subplot_contexts:
            if context.ax and context.title:
                context.ax.set_title(context.title)
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
        fontdict = self._style_params.get("prim_ax_fontdict")

        for _, axis in enumerate(self.yield_axes_objects()):
            axis.set_xlabel(
                xlabel, fontdict=fontdict
            ) if xlabel is not False else axis.xaxis.label.set_visible(False)

            axis.set_ylabel(
                ylabel, fontdict=fontdict
            ) if ylabel is not False else axis.yaxis.label.set_visible(False)

        return self

    def _diagonal_lines_and_labels(self) -> ISOPlot:
        """Add diagonal lines and labels to the plot."""
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
