import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from soundscapy.plotting.plotting_utils import (
    DEFAULT_STYLE_OPTIONS,
    CircumplexPlotParams,
)
from soundscapy.plotting.stylers import SeabornStyler, StyleOptions


class SeabornBackend:
    """Backend for creating plots using Seaborn and Matplotlib."""

    def __init__(self, style_options: StyleOptions = DEFAULT_STYLE_OPTIONS) -> None:
        """
        Initialize the SeabornBackend with style options.

        Parameters
        ----------
        style_options : StyleOptions, optional
            The styling options for plots, by default DEFAULT_STYLE_OPTIONS

        """
        self.style_options = style_options

    def create_scatter(
        self, data: pd.DataFrame, params: CircumplexPlotParams, ax: Axes | None = None
    ) -> tuple[Figure | SubFigure | None, Axes]:
        """
        Create a scatter plot using Seaborn.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style_options.figsize)
        else:
            fig = ax.get_figure()
        if "incl_scatter" in params.extra_params:
            params.extra_params.pop("incl_scatter")
        if "density_type" in params.extra_params:
            params.extra_params.pop("density_type")
        if "fill" in params.extra_params:
            params.extra_params.pop("fill")
        sns.scatterplot(
            data=data,
            x=params.x,
            y=params.y,
            hue=params.hue,
            palette=params.palette if params.hue else None,
            alpha=params.alpha,
            ax=ax,
            zorder=self.style_options.data_zorder,
            legend=params.legend,
            **params.extra_params,
        )
        return fig, ax

    def create_density(
        self, data: pd.DataFrame, params: CircumplexPlotParams, ax: Axes | None = None
    ) -> tuple[Figure | SubFigure | None, Axes]:
        """
        Create a density plot using Seaborn.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        if len(data) < 30:
            warnings.warn(
                "Density plots are not recommended for small datasets (<30 samples).",
                UserWarning,
                stacklevel=2,
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=self.style_options.figsize)
        else:
            fig = ax.get_figure()

        sns.kdeplot(
            data=data,
            x=params.x,
            y=params.y,
            hue=params.hue,
            palette=params.palette,
            fill=params.fill,
            alpha=params.alpha,
            ax=ax,
            bw_adjust=self.style_options.bw_adjust,
            zorder=self.style_options.data_zorder,
            common_norm=False,
            legend=params.legend,
            **params.extra_params,
        )
        if params.incl_outline:
            sns.kdeplot(
                data=data,
                x=params.x,
                y=params.y,
                hue=params.hue,
                alpha=1,
                palette=params.palette,
                fill=False,
                ax=ax,
                bw_adjust=self.style_options.bw_adjust,
                zorder=self.style_options.data_zorder,
                common_norm=False,
                legend=False,
                **params.extra_params,
            )
        return fig, ax

    def create_jointplot(
        self, data: pd.DataFrame, params: CircumplexPlotParams
    ) -> tuple[Figure | SubFigure | None, Axes]:
        """
        Create a joint plot using Seaborn.

        Examples
        --------
        >>> import soundscapy as sspy
        >>> from soundscapy.plotting import Backend, CircumplexPlot, StyleOptions, CircumplexPlotParams
        >>> data = sspy.isd.load()
        >>> data = sspy.surveys.add_iso_coords(data, overwrite=True)
        >>> sample_data = sspy.isd.select_location_ids(data, ['CamdenTown'])
        >>> plot = CircumplexPlot(data=sample_data, backend=Backend.SEABORN)
        >>> g = plot.jointplot()
        >>> g.show() # doctest: +SKIP

        """
        g = sns.JointGrid(xlim=params.xlim, ylim=params.ylim)
        joint_params = params
        joint_params.title = ""
        SeabornBackend.create_density(self, data, joint_params, ax=g.ax_joint)

        margin_params = params
        margin_params.title = ""
        sns.kdeplot(data, x=params.x, ax=g.ax_marg_x, fill=True, alpha=params.alpha)
        sns.kdeplot(data, y=params.y, ax=g.ax_marg_y, fill=True, alpha=params.alpha)

        return (
            g.fig,
            g.ax_joint,
        )  # TODO: Should return the whole JointGrid object - repeat throughout plotting methods

    def create_simple_density(
        self, data: pd.DataFrame, params: CircumplexPlotParams, ax: Axes | None = None
    ) -> tuple[Figure | SubFigure | None, Axes]:
        """
        Create a simple density plot using Seaborn with simplified styling.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.
            ax (Axes | None, optional): Matplotlib axes object to plot on.
                If None, a new figure is created.

        Returns
        -------
            tuple: A tuple containing the figure and axes objects.

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style_options.figsize)
        else:
            fig = ax.get_figure()

        sns.kdeplot(
            data=data,
            x=params.x,
            y=params.y,
            hue=params.hue,
            palette=params.palette,
            fill=params.fill,
            ax=ax,
            thresh=self.style_options.simple_density["thresh"],
            levels=self.style_options.simple_density["levels"],
            alpha=self.style_options.simple_density["alpha"],
            zorder=self.style_options.data_zorder,
            common_norm=False,
            legend=params.legend,
            **params.extra_params,
        )
        if params.incl_outline:
            sns.kdeplot(
                data=data,
                x=params.x,
                y=params.y,
                hue=params.hue,
                palette=params.palette,
                fill=False,
                ax=ax,
                thresh=self.style_options.simple_density["thresh"],
                levels=self.style_options.simple_density["levels"],
                zorder=self.style_options.data_zorder,
                alpha=1,
                common_norm=False,
                legend=False,
                **params.extra_params,
            )
        return fig, ax

    def apply_styling(
        self,
        plot_obj: tuple[Figure | SubFigure | None, Axes],
        params: CircumplexPlotParams,
    ) -> tuple[Figure | SubFigure | None, Axes]:
        """
        Apply styling to the Seaborn plot.

        Parameters
        ----------
            plot_obj (tuple): A tuple containing the figure and axes objects.
            params (CircumplexPlotParams): The parameters for styling.

        Returns
        -------
            tuple: The styled figure and axes objects.

        """
        fig, ax = plot_obj
        styler = SeabornStyler(params, self.style_options)
        return styler.apply_styling(fig, ax)

    def show(self, plot_obj: tuple[Figure | SubFigure | None, Axes]) -> None:
        """
        Display the Matplotlib figure.

        Parameters
        ----------
            plot_obj: A tuple containing the figure and axes objects.

        """
        fig, _ = plot_obj
        plt.show()
