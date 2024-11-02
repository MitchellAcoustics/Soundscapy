"""
Backend implementations for CircumplexPlot.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from soundscapy.plotting.stylers import SeabornStyler, StyleOptions
from soundscapy.plotting.plotting_utils import LayerType


class PlotBackend(ABC):
    """
    Abstract base class for plot backends.

    This class defines the interface for creating scatter and density plots,
    as well as applying styling to the plots.
    """

    @abstractmethod
    def plot_layer(self, layer: Any, params: Any, ax: Optional[plt.Axes] = None):
        """Plot a single layer."""
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
    """Backend for creating plots using Seaborn and Matplotlib."""

    def __init__(self, style_options: StyleOptions = StyleOptions()):
        self.style_options = style_options

    def plot_layer(self, layer, params, ax=None):
        """
        Plot a single layer using the appropriate plotting function.

        Parameters
        ----------
        layer : Layer
            The layer to plot
        params : CircumplexPlotParams
            The parameters for the plot
        ax : matplotlib.axes.Axes, optional
            The axes to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style_options.figsize)
        else:
            fig = ax.get_figure()

        if layer.type == LayerType.SCATTER:
            self._plot_scatter_layer(layer, params, ax)
        elif layer.type == LayerType.DENSITY:
            self._plot_density_layer(layer, params, ax)
        elif layer.type == LayerType.SIMPLE_DENSITY:
            self._plot_simple_density_layer(layer, params, ax)

        return fig, ax

    def _plot_scatter_layer(self, layer, params, ax):
        """Plot a scatter layer."""
        sns.scatterplot(
            data=layer.data,
            x=params.x,
            y=params.y,
            hue=layer.hue or params.hue,
            color=layer.color,
            alpha=layer.alpha,
            ax=ax,
            label=layer.label,
            legend=params.legend if layer.hue else False,
            zorder=layer.zorder,
            **layer.params,
        )

    def _plot_density_layer(self, layer, params, ax):
        """Plot a density layer."""
        sns.kdeplot(
            data=layer.data,
            x=params.x,
            y=params.y,
            hue=layer.hue or params.hue,
            color=layer.color,
            fill=params.fill,
            alpha=layer.alpha,
            ax=ax,
            label=layer.label,
            bw_adjust=self.style_options.bw_adjust,
            zorder=layer.zorder,
            legend=params.legend if layer.hue else False,
            **layer.params,
        )

    def _plot_simple_density_layer(self, layer, params, ax):
        """Plot a simple density layer with optional outline."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style_options.figsize)
        else:
            fig = ax.get_figure()

        # Get simple density settings
        density_settings = self.style_options.simple_density

        # Handle color for both fill and outline
        if layer.hue or params.hue:
            color = None
            palette = params.palette if layer.hue or params.hue else None
        else:
            color = layer.color
            palette = None

        # First plot filled contours
        sns.kdeplot(
            data=layer.data,
            x=params.x,
            y=params.y,
            hue=layer.hue or params.hue,
            color=color,
            palette=palette,
            fill=params.fill,
            alpha=layer.alpha or density_settings["alpha"],
            thresh=density_settings["thresh"],
            levels=density_settings["levels"],
            legend=params.legend if layer.hue else False,
            bw_adjust=self.style_options.bw_adjust,
            zorder=layer.zorder or self.style_options.data_zorder,
            ax=ax,
        )

        # Add outline if included in settings
        if density_settings["incl_outline"]:
            sns.kdeplot(
                data=layer.data,
                x=params.x,
                y=params.y,
                hue=layer.hue or params.hue,
                color=color,
                palette=palette,
                fill=False,
                thresh=density_settings["thresh"],
                levels=density_settings["levels"],
                alpha=density_settings["outline_alpha"],
                linewidths=density_settings["outline_linewidth"],
                legend=False,  # Don't add duplicate legend entries
                bw_adjust=self.style_options.bw_adjust,
                zorder=(layer.zorder or self.style_options.data_zorder)
                + 0.1,  # Slightly higher
                ax=ax,
            )

        return fig, ax

    def apply_styling(
        self, plot_obj: Tuple[plt.Figure, plt.Axes], params: Any
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Apply styling to the plot."""
        fig, ax = plot_obj
        styler = SeabornStyler(params, self.style_options)
        return styler.apply_styling(fig, ax)


class PlotlyBackend(PlotBackend):
    """
    Backend for creating plots using Plotly.

    This backend supports the layer system by maintaining a Plotly figure
    and adding traces for each layer.
    """

    def __init__(self, style_options: StyleOptions = StyleOptions()):
        self.style_options = style_options
        self.fig = None
        self._traces: List[Dict] = []

    def plot_layer(self, layer, params, ax=None):
        """
        Add a new layer to the Plotly figure.

        Parameters
        ----------
        layer : Layer
            Layer to add to the plot
        params : CircumplexPlotParams
            Plot parameters
        ax : Any, optional
            Ignored for Plotly backend

        Returns
        -------
        go.Figure
            Updated Plotly figure
        """
        # Initialize figure if needed
        if self.fig is None:
            self.fig = go.Figure()

        # Get base color for non-hue plots
        if layer.hue is None and layer.color is None:
            layer.color = params.color or px.colors.qualitative.Plotly[0]

        if layer.type == LayerType.SCATTER:
            self._add_scatter_layer(layer, params)
        elif layer.type == LayerType.DENSITY:
            self._add_density_layer(layer, params)
        elif layer.type == LayerType.SIMPLE_DENSITY:
            self._add_simple_density_layer(layer, params)

        return self.fig

    def _add_scatter_layer(self, layer, params):
        """Add a scatter plot layer."""
        data = layer.data

        if layer.hue or params.hue:
            hue_col = layer.hue or params.hue
            groups = data[hue_col].unique()
            palette = params.palette or "colorblind"
            colors = (
                px.colors.qualitative.Set2
                if palette == "colorblind"
                else getattr(px.colors.qualitative, palette, px.colors.qualitative.Set2)
            )

            for idx, group in enumerate(groups):
                group_data = data[data[hue_col] == group]
                self.fig.add_trace(
                    go.Scatter(
                        x=group_data[params.x],
                        y=group_data[params.y],
                        mode="markers",
                        name=str(group),
                        marker=dict(
                            color=colors[idx % len(colors)],
                            opacity=layer.alpha or 0.8,
                        ),
                        showlegend=params.legend,
                    )
                )
        else:
            self.fig.add_trace(
                go.Scatter(
                    x=data[params.x],
                    y=data[params.y],
                    mode="markers",
                    marker=dict(
                        color=layer.color,
                        opacity=layer.alpha or 0.8,
                    ),
                    name=layer.label or "Scatter",
                    showlegend=params.legend and layer.label is not None,
                )
            )

    def _add_density_layer(self, layer, params):
        """Add a density contour layer."""
        data = layer.data

        if layer.hue or params.hue:
            hue_col = layer.hue or params.hue
            groups = data[hue_col].unique()
            palette = params.palette or "colorblind"
            colors = (
                px.colors.qualitative.Set2
                if palette == "colorblind"
                else getattr(px.colors.qualitative, palette, px.colors.qualitative.Set2)
            )

            for idx, group in enumerate(groups):
                group_data = data[data[hue_col] == group]
                self._add_single_density(
                    group_data[params.x],
                    group_data[params.y],
                    color=colors[idx % len(colors)],
                    name=str(group),
                    alpha=layer.alpha,
                    params=params,
                    show_legend=params.legend,
                )
        else:
            self._add_single_density(
                data[params.x],
                data[params.y],
                color=layer.color,
                name=layer.label or "Density",
                alpha=layer.alpha,
                params=params,
                show_legend=params.legend and layer.label is not None,
            )

    def _add_simple_density_layer(self, layer, params):
        """Add a simple density layer with fewer contour levels."""
        data = layer.data

        if layer.hue or params.hue:
            hue_col = layer.hue or params.hue
            groups = data[hue_col].unique()
            palette = params.palette or "colorblind"
            colors = (
                px.colors.qualitative.Set2
                if palette == "colorblind"
                else getattr(px.colors.qualitative, palette, px.colors.qualitative.Set2)
            )

            for idx, group in enumerate(groups):
                group_data = data[data[hue_col] == group]
                self._add_single_density(
                    group_data[params.x],
                    group_data[params.y],
                    color=colors[idx % len(colors)],
                    name=str(group),
                    alpha=layer.alpha,
                    params=params,
                    simple=True,
                    show_legend=params.legend,
                )
        else:
            self._add_single_density(
                data[params.x],
                data[params.y],
                color=layer.color,
                name=layer.label or "Simple Density",
                alpha=layer.alpha,
                params=params,
                simple=True,
                show_legend=params.legend and layer.label is not None,
            )

    def _add_single_density(
        self, x, y, color, name, alpha, params, simple=False, show_legend=True
    ):
        """Add a single density contour trace."""
        nbins = 20 if not simple else 10
        colorscale = [
            [
                0,
                f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0)",
            ],
            [1, color],
        ]

        self.fig.add_trace(
            go.Histogram2dContour(
                x=x,
                y=y,
                colorscale=colorscale,
                contours=dict(
                    coloring="fill",
                    showlines=params.incl_outline,
                ),
                nbinsx=nbins,
                nbinsy=nbins,
                opacity=alpha or 0.8,
                name=name,
                showscale=False,
                showlegend=show_legend,
                ncontours=20
                if not simple
                else self.style_options.simple_density["levels"],
            )
        )

    def apply_styling(self, fig, params):
        """Apply styling to the Plotly figure."""
        # Update layout for axis scaling and size
        fig.update_layout(
            width=self.style_options.figsize[0] * 100,
            height=self.style_options.figsize[1] * 100,
            title=params.title,
            xaxis=dict(
                title=params.x if params.show_labels else None,
                range=params.xlim,
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.5)",
            ),
            yaxis=dict(
                title=params.y if params.show_labels else None,
                range=params.ylim,
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.5)",
            ),
            plot_bgcolor="white",
            showlegend=params.legend,
        )

        # Add diagonal lines if requested
        if params.diagonal_lines:
            # Add diagonal lines
            fig.add_trace(
                go.Scatter(
                    x=[params.xlim[0], params.xlim[1]],
                    y=[params.ylim[0], params.ylim[1]],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1),
                    showlegend=False,
                    name="Diagonal",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[params.xlim[0], params.xlim[1]],
                    y=[params.ylim[1], params.ylim[0]],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1),
                    showlegend=False,
                    name="Diagonal",
                )
            )

            # Add diagonal labels
            for label, pos in [
                ("(vibrant)", (params.xlim[1] / 2, params.ylim[1] / 2)),
                ("(chaotic)", (params.xlim[0] / 2, params.ylim[1] / 2)),
                ("(monotonous)", (params.xlim[0] / 2, params.ylim[0] / 2)),
                ("(calm)", (params.xlim[1] / 2, params.ylim[0] / 2)),
            ]:
                fig.add_annotation(
                    x=pos[0],
                    y=pos[1],
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color="gray", italic=True),
                )

        # Update legend position if specified
        if params.legend_location != "best":
            fig.update_layout(
                legend=dict(
                    yanchor="top" if "upper" in params.legend_location else "bottom",
                    y=0.99 if "upper" in params.legend_location else 0.01,
                    xanchor="left" if "left" in params.legend_location else "right",
                    x=0.01 if "left" in params.legend_location else 0.99,
                )
            )

        return fig

    def show(self, fig):
        """Display the Plotly figure."""
        fig.show()
