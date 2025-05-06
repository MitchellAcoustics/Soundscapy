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

"""

import copy
import warnings
from collections.abc import Generator, Iterable
from typing import Unpack

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from soundscapy.plotting.plotting_types import (
    DensityParamTypes,
    JointPlotParamTypes,
    MplLegendLocType,
    ScatterParamTypes,
    SeabornPaletteType,
    StyleParamsTypes,
    SubplotsParamsTypes,
)

DEFAULT_TITLE = "Soundscape Density Plot"
DEFAULT_XCOL = "ISOPleasant"
DEFAULT_YCOL = "ISOEventful"
DEFAULT_XLIM = (-1, 1)
DEFAULT_YLIM = (-1, 1)
DEFAULT_FIGSIZE = (5, 5)

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
    bw_adjust=1.2,
)

DEFAULT_STYLE_PARAMS: StyleParamsTypes = StyleParamsTypes(
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
    diag_lines_zorder=1,
    diag_labels_zorder=4,
    prim_lines_zorder=2,
    data_zorder=3,
    show_labels=True,
    legend_location="best",
    linewidth=1.5,
    primary_lines=True,
    diagonal_lines=False,
)

DEFAULT_SUBPLOTS_PARAMS: SubplotsParamsTypes = SubplotsParamsTypes(
    sharex=True,
    sharey=True,
)


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
    >>> cp

    """

    # TODO: Implement jointplot method for Seaborn backend.

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        xcol: str = "ISOPleasant",
        ycol: str = "ISOEventful",
        title: str | None = "Soundscape Density Plot",
        title_fontsize: int = 16,
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
        self._data = data
        self.x = xcol
        self.y = ycol
        self.title = title
        self.title_fontsize = title_fontsize
        self.hue = hue
        self.figure = figure
        self.axes = axes

        self.palette: SeabornPaletteType | None = palette if hue is not None else None

        self._has_subplots = False

        self._scatter_params: ScatterParamTypes = copy.deepcopy(DEFAULT_SCATTER_PARAMS)
        self._scatter_params.update(
            data=data,
            x=xcol,
            y=ycol,
            hue=hue,
            palette=self.palette,
        )

        self._density_params: DensityParamTypes = copy.deepcopy(DEFAULT_DENSITY_PARAMS)
        self._density_params.update(
            data=data,
            x=xcol,
            y=ycol,
            hue=hue,
            palette=self.palette,
        )

        self._simple_density_params = copy.deepcopy(DEFAULT_DENSITY_PARAMS)
        # Override default params with user-provided params
        self._simple_density_params.update(
            data=data,
            x=xcol,
            y=ycol,
            hue=hue,
            palette=self.palette,
            thresh=0.5,
            levels=2,
            alpha=0.5,
        )

        self._style_params: StyleParamsTypes = copy.deepcopy(DEFAULT_STYLE_PARAMS)

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

        return self

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
                len(ax_idx) != 2
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

        # If no axis is specified, plot on all axes
        # in the figure (if multiple subplots exist)
        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                # If subplot data is provided, use it for the density plot
                if self._subplot_datas is not None:
                    density_params["data"] = self._subplot_datas[i]
                # Check if the data is provided either in the class or in the method call  # noqa: E501
                # If provided in the class call, it would have been added to
                # self._density_params during object initialization and copied to density_params  # noqa: E501
                d = density_params.get("data")
                if d is not None:
                    self._valid_density(d)
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
            # Check if the data is provided either in the class or in the method call
            # If provided in the class call, it would have been added to
            # self._density_params during object initialization and copied to density_params  # noqa: E501
            d = density_params.get("data")
            if d is not None:
                self._valid_density(d)
            else:
                msg = "No data provided for density plot."
                raise ValueError(msg)

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

        # If no axis is specified, plot on all axes
        # in the figure (if multiple subplots exist)
        if on_axis is None:
            for i, axis in enumerate(self.yield_axes_objects()):
                # If subplot data is provided, use it for the density plot
                if self._subplot_datas is not None:
                    simple_density_params["data"] = self._subplot_datas[i]
                # Check if the data is provided either in the class or in the method call  # noqa: E501
                # If provided in the class call, it would have been added to
                # self._density_params during object initialization and copied to simple_density_params  # noqa: E501
                d = simple_density_params.get("data")
                if d is not None:
                    self._valid_density(d)
                else:
                    msg = "No data provided for density plot."
                    raise ValueError(msg)

                # Plot the density plot on the current axis
                sns.kdeplot(
                    ax=axis,
                    **simple_density_params,
                )
                if include_outline:
                    outline_params = simple_density_params.copy()
                    outline_params.update({"fill": False, "alpha": 1, "legend": False})
                    sns.kdeplot(
                        ax=axis,
                        **outline_params,
                    )

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
            # Check if the data is provided either in the class or in the method call
            # If provided in the class call, it would have been added to
            # self._density_params during object initialization
            # and copied to density_params
            d = simple_density_params.get("data")
            if d is not None:
                self._valid_density(d)
            else:
                msg = "No data provided for density plot."
                raise ValueError(msg)

            sns.kdeplot(
                ax=axis,
                **simple_density_params,
            )
            if include_outline:
                outline_params = simple_density_params.copy()
                outline_params.update({"fill": False, "alpha": 1, "legend": False})
                sns.kdeplot(
                    ax=axis,
                    **outline_params,
                )

        return self

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
        xlim: tuple[float, float] = (-1, 1),
        ylim: tuple[float, float] = (-1, 1),
        diag_lines_zorder: int = 1,
        diag_labels_zorder: int = 4,
        prim_lines_zorder: int = 2,
        data_zorder: int = 3,
        legend_location: MplLegendLocType = "best",
        linewidth: float = 1.5,
        *,
        primary_lines: bool = True,
        diagonal_lines: bool = False,
        show_labels: bool = True,
    ) -> "CircumplexPlot":
        """
        Apply styling to the Seaborn plot.

        Returns
        -------
            tuple: The styled figure and axes objects.

        """
        self._style_params.update(
            xlim=xlim,
            ylim=ylim,
            diag_lines_zorder=diag_lines_zorder,
            diag_labels_zorder=diag_labels_zorder,
            prim_lines_zorder=prim_lines_zorder,
            data_zorder=data_zorder,
            legend_location=legend_location,
            linewidth=linewidth,
            show_labels=show_labels,
            primary_lines=primary_lines,
            diagonal_lines=diagonal_lines,
        )
        self._check_for_axes()

        self._set_style()
        self._circumplex_grid()
        self._set_title()
        self._set_axes_titles()
        self._deal_w_default_labels()
        if primary_lines:
            self._primary_lines_and_labels()
        if diagonal_lines:
            self._diagonal_lines_and_labels()
        if self.hue and self._style_params.get("legend_location") is not False:
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
            axis.set_xticks(np.arange(-1, 1.1, 0.2))
            axis.set_yticks(np.arange(-1, 1.1, 0.2))
            axis.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
            axis.axhline(0, color="black", linewidth=0.5)
            axis.axvline(0, color="black", linewidth=0.5)
        return self

    def _set_title(self) -> "CircumplexPlot":
        """Set the title of the plot."""
        if self.title:
            figure = self.get_figure()
            figure.suptitle(self.title, fontsize=self.title_fontsize)
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

    def _move_legend(self) -> "CircumplexPlot":
        """Move the legend to the specified location."""
        for _, axis in enumerate(self.yield_axes_objects()):
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
