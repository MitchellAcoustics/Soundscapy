import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from soundscapy.plotting.stylers import SeabornStyler, StyleOptions


class PlotBackend(ABC):
    """
    Abstract base class for plot backends.

    This class defines the interface for creating scatter and density plots,
    as well as applying styling to the plots.
    """

    @abstractmethod
    def create_scatter(self, data, params):
        """
        Create a scatter plot.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.

        Returns
        -------
            The created plot object.
        """
        pass

    @abstractmethod
    def create_density(self, data, params):
        """
        Create a density plot.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.

        Returns
        -------
            The created plot object.
        """
        pass

    @abstractmethod
    def apply_styling(self, plot_obj, params):
        """
        Apply styling to the plot.

        Parameters
        ----------
            plot_obj: The plot object to style.
            params (CircumplexPlotParams): The parameters for styling.

        Returns
        -------
            The styled plot object.
        """
        pass


class SeabornBackend(PlotBackend):
    """
    Backend for creating plots using Seaborn and Matplotlib.
    """

    def __init__(self, style_options: StyleOptions = StyleOptions()):
        self.style_options = style_options

    def create_scatter(self, data, params, ax=None):
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

    def create_density(self, data, params, ax=None):
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

    def create_jointplot(self, data, params):
        """

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

    def create_simple_density(self, data, params, ax=None):
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

    def apply_styling(self, plot_obj, params):
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

    def show(self, plot_obj):
        """
        Display the Matplotlib figure.

        Parameters
        ----------
            fig: The figure to display.
        """
        fig, _ = plot_obj
        plt.show()


class PlotlyBackend(PlotBackend):
    """
    Backend for creating plots using Plotly.
    """

    def __init__(self):
        warnings.warn(
            "PlotlyBackend is very experimental and not fully implemented.", UserWarning
        )
        pass

    def create_scatter(self, data, params):
        """
        Create a scatter plot using Plotly.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.

        Returns
        -------
            go.Figure: A Plotly figure object.
        """
        fig = px.scatter(
            data,
            x=params.x,
            y=params.y,
            color=params.hue,
            title=params.title,
            range_x=params.xlim,
            range_y=params.ylim,
            **params.extra_params,
        )
        fig.update_layout(
            width=600,
            height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        return fig

    def create_density(self, data, params):
        """
        Create a density plot using Plotly.

        Parameters
        ----------
            data (pd.DataFrame): The data to plot.
            params (CircumplexPlotParams): The parameters for the plot.

        Returns
        -------
            go.Figure: A Plotly figure object.
        """
        if len(data) < 30:
            warnings.warn(
                "Density plots are not recommended for small datasets (<30 samples). Consider using a scatter plot instead.",
                UserWarning,
            )

        fig = px.density_heatmap(
            data,
            x=params.x,
            y=params.y,
            title=params.title,
            range_x=params.xlim,
            range_y=params.ylim,
            **params.extra_params,
        )
        fig.update_layout(
            width=600,
            height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        scatter_trace = px.scatter(
            data, x=params.x, y=params.y, color=params.hue, opacity=0.5
        ).data[0]
        fig.add_trace(scatter_trace)
        return fig

    def apply_styling(self, plot_obj, params):
        """
        Apply styling to the Plotly plot.

        Parameters
        ----------
            plot_obj (go.Figure): A Plotly figure object.
            params (CircumplexPlotParams): The parameters for styling.

        Returns
        -------
            go.Figure: The styled Plotly figure object.
        """
        fig = plot_obj
        if params.diagonal_lines:
            fig.add_trace(
                go.Scatter(
                    x=[params.xlim[0], params.xlim[1]],
                    y=[params.ylim[0], params.ylim[1]],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[params.xlim[0], params.xlim[1]],
                    y=[params.ylim[1], params.ylim[0]],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    showlegend=False,
                )
            )
        return fig

    def show(self, fig):
        """
        Display the Plotly figure.

        Parameters
        ----------
            fig (go.Figure): The Plotly figure to display.
        """
        fig.show()
