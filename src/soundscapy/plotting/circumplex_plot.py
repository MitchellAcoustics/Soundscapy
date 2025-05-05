"""Main module for creating circumplex plots using different backends."""

import copy
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from soundscapy.plotting.plotting_utils import (
    DEFAULT_CIRCUMPLEX_PLOT_PARAMS,
    DEFAULT_STYLE_OPTIONS,
    CircumplexPlotParams,
    SeabornStyler,
    StyleOptions,
)

RECOMMENDED_MIN_SAMPLES: int = 30


class CircumplexPlot:
    """
    A class for creating circumplex plots using different backends.

    This class provides methods for creating scatter plots and density plots
    based on the circumplex model of soundscape perception. It supports multiple
    backends (currently Seaborn and Plotly) and offers various customization options.

    Example:
    -------
    >>> from soundscapy import isd, surveys
    >>> df = isd.load()
    >>> df = surveys.add_iso_coords(df)
    >>> ct = isd.select_location_ids(df, ["CamdenTown", "RegentsParkJapan"])
    >>> cp = (
            CircumplexPlot(ct, params=params)
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
        data: pd.DataFrame,
        xcol: str = "ISOPleasant",
        ycol: str = "ISOEventful",
        params: CircumplexPlotParams = DEFAULT_CIRCUMPLEX_PLOT_PARAMS,
        style_options: StyleOptions = DEFAULT_STYLE_OPTIONS,
        fig: Figure | SubFigure | None = None,
        ax: Axes | np.ndarray | None = None,
    ) -> None:
        self._data = data
        self.x = xcol
        self.y = ycol
        self._params = params
        self._style_options = style_options
        self._styler = SeabornStyler(self._params, self._style_options)
        self.fig = fig
        self.ax = ax

    def create_subplots(
        self, nrows: int = 1, ncols: int = 1, **kwargs
    ) -> "CircumplexPlot":
        """
        Create subplots for the circumplex plot.

        Parameters
        ----------
            nrows (int): Number of rows in the subplot grid.
            ncols (int): Number of columns in the subplot grid.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self.fig, self.ax = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=self._style_options.figsize, **kwargs
        )
        return self

    def _check_for_ax(self) -> "CircumplexPlot":
        """
        Check if the axes object is provided.

        Returns
        -------
            Axes: The axes object to be used for plotting.

        Raises
            ValueError: If the axes object does not exist.

        """
        if self.ax is None:
            msg = (
                "No axes object provided. "
                "Please create a figure and axes using create_subplots() first."
            )
            raise ValueError(msg)
        return self

    def _valid_density(self) -> None:
        """
        Check if the data is valid for density plots.

        Raises
        ------
            UserWarning: If the data is too small for density plots.

        """
        if len(self._data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                f"Density plots are not recommended for small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

    def add_scatter(self) -> "CircumplexPlot":
        """
        Add a scatter plot to the existing axes.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self._check_for_ax()

        if isinstance(self.ax, np.ndarray):
            for i, axis in enumerate(self.ax.flatten()):
                self.ax[i] = sns.scatterplot(
                    data=self._data,
                    x=self._params.x,
                    y=self._params.y,
                    hue=self._params.hue,
                    palette=self._params.palette if self._params.hue else None,
                    alpha=self._params.alpha,
                    ax=axis,
                    zorder=self._style_options.data_zorder,
                    legend=self._params.legend,
                    **self._params.extra_params,
                )
        elif isinstance(self.ax, Axes):
            self.ax = sns.scatterplot(
                data=self._data,
                x=self._params.x,
                y=self._params.y,
                hue=self._params.hue,
                palette=self._params.palette if self._params.hue else None,
                alpha=self._params.alpha,
                ax=self.ax,
                zorder=self._style_options.data_zorder,
                legend=self._params.legend,
                **self._params.extra_params,
            )
        else:
            msg = "Invalid axes object. Please provide a valid Axes or ndarray of Axes."
            raise TypeError(msg)
        return self

    def add_density(self) -> "CircumplexPlot":
        """
        Add a density plot to the existing axes.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self._valid_density()
        self._check_for_ax()

        if isinstance(self.ax, np.ndarray):
            for i, axis in enumerate(self.ax.flatten()):
                self.ax[i] = sns.kdeplot(
                    data=self._data,
                    x=self._params.x,
                    y=self._params.y,
                    hue=self._params.hue,
                    palette=self._params.palette if self._params.hue else None,
                    fill=self._params.fill,
                    alpha=self._params.alpha,
                    ax=axis,
                    bw_adjust=self.style_options.bw_adjust,
                    zorder=self._style_options.data_zorder,
                    common_norm=False,
                    legend=self._params.legend,
                    **self._params.extra_params,
                )

                if self._params.incl_outline:
                    self.ax[i] = sns.kdeplot(
                        data=self._data,
                        x=self._params.x,
                        y=self._params.y,
                        hue=self._params.hue,
                        alpha=1,
                        palette=self._params.palette,
                        fill=False,
                        ax=axis,
                        bw_adjust=self._style_options.bw_adjust,
                        zorder=self._style_options.data_zorder,
                        common_norm=False,
                        legend=False,
                        **self._params.extra_params,
                    )

        elif isinstance(self.ax, Axes):
            self.ax = sns.kdeplot(
                data=self._data,
                x=self._params.x,
                y=self._params.y,
                hue=self._params.hue,
                palette=self._params.palette if self._params.hue else None,
                fill=self._params.fill,
                alpha=self._params.alpha,
                ax=self.ax,
                bw_adjust=self._style_options.bw_adjust,
                zorder=self._style_options.data_zorder,
                common_norm=False,
                legend=self._params.legend,
                **self._params.extra_params,
            )
            if self._params.incl_outline:
                self.ax = sns.kdeplot(
                    data=self._data,
                    x=self._params.x,
                    y=self._params.y,
                    hue=self._params.hue,
                    alpha=1,
                    palette=self._params.palette,
                    fill=False,
                    ax=self.ax,
                    bw_adjust=self._style_options.bw_adjust,
                    zorder=self._style_options.data_zorder,
                    common_norm=False,
                    legend=False,
                    **self._params.extra_params,
                )
        else:
            msg = "Invalid axes object. Please provide a valid Axes or ndarray of Axes."
            raise TypeError(msg)
        return self

    def add_simple_density(self) -> "CircumplexPlot":
        """
        Add a simple density plot to the existing axes.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        self._valid_density()
        self._check_for_ax()

        if isinstance(self.ax, np.ndarray):
            for i, axis in enumerate(self.ax.flatten()):
                self.ax[i] = sns.kdeplot(
                    data=self._data,
                    x=self._params.x,
                    y=self._params.y,
                    hue=self._params.hue,
                    palette=self._params.palette if self._params.hue else None,
                    alpha=self._params.alpha,
                    ax=axis,
                    zorder=self._style_options.data_zorder,
                    legend=self._params.legend,
                    **self._params.extra_params,
                )
        elif isinstance(self.ax, Axes):
            self.ax = sns.kdeplot(
                data=self._data,
                x=self._params.x,
                y=self._params.y,
                hue=self._params.hue,
                palette=self._params.palette if self._params.hue else None,
                alpha=self._params.alpha,
                ax=self.ax,
                zorder=self._style_options.data_zorder,
                legend=self._params.legend,
                **self._params.extra_params,
            )
        else:
            msg = "Invalid axes object. Please provide a valid Axes or ndarray of Axes."
            raise TypeError(msg)
        return self

    def get_style_options(self) -> StyleOptions:
        """Get the current StyleOptions."""
        return copy.deepcopy(self.style_options)

    def update_style_options(self, **kwargs) -> "CircumplexPlot":
        """Update the StyleOptions with new values."""
        new_style_options = copy.deepcopy(self.style_options)
        for key, value in kwargs.items():
            if hasattr(new_style_options, key):
                setattr(new_style_options, key, value)
            else:
                msg = f"Invalid StyleOptions attribute: {key}"
                raise ValueError(msg)

        self.style_options = new_style_options
        return self

    def apply_styling(
        self,
    ) -> "CircumplexPlot":
        """
        Apply styling to the Seaborn plot.

        Returns
        -------
            tuple: The styled figure and axes objects.

        """
        self._check_for_ax()
        self.fig, self.ax = self._styler.apply_styling(self.fig, self.ax)
        return self

    def iso_annotation(
        self, location_idx: int, x_adj: float = 0, y_adj: float = 0, **kwargs
    ) -> "CircumplexPlot":
        """Add an annotation to the plot (only for Seaborn backend)."""
        x = self._data[self._params.x].iloc[location_idx]
        y = self._data[self._params.y].iloc[location_idx]
        if isinstance(self.ax, np.ndarray):
            for i, _ in enumerate(self.ax.flatten()):
                self.ax[i].annotate(
                    text=self._data.index[location_idx],
                    xy=(x, y),
                    xytext=(x + x_adj, y + y_adj),
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "-", "ec": "black"},
                    **kwargs,
                )
        elif isinstance(self.ax, Axes):
            self.ax.annotate(
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
