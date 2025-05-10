"""
Styling methods for ISOPlot.

This module provides a mixin class with methods for styling ISOPlot instances,
including grid lines, labels, and other visual elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from soundscapy.plotting.defaults import DEFAULT_STYLE_PARAMS
from soundscapy.plotting.plotting_types import ParamModel

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class ISOPlotStylingMixin:
    """Mixin providing styling methods for ISOPlot."""

    def apply_styling(self, **kwargs: Any) -> Any:
        """
        Apply styling to the plot.

        This method applies various styling elements to the plot, including:
        - Setting axis limits and labels
        - Adding grid lines
        - Adding diagonal lines (if enabled)
        - Setting titles
        - Configuring legends

        Parameters
        ----------
        **kwargs : Any
            Additional styling parameters to override defaults

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0, 0.5, 100),
        ...     'ISOEventful': rng.normal(0, 0.5, 100),
        ... })
        >>> plot = (ISOPlot(data=data)
        ...         .add_scatter()
        ...         .apply_styling(
        ...             xlim=(-1.5, 1.5),
        ...             ylim=(-1.5, 1.5),
        ...             diagonal_lines=True
        ...         ))
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.close()  # Clean up

        """
        # Update style parameters with provided kwargs
        self._style_params = ParamModel.create("style", **{**DEFAULT_STYLE_PARAMS, **kwargs})

        # Check if we have axes to style
        self._check_for_axes()

        # Apply styling to each axes
        for ax in self.yield_axes_objects():
            # Set axis limits
            ax.set_xlim(self._style_params.xlim)
            ax.set_ylim(self._style_params.ylim)

            # Set up grid
            self._circumplex_grid(ax)

            # Add primary lines if enabled
            if self._style_params.primary_lines:
                self._primary_lines(ax)
                self._primary_labels(ax)

            # Add diagonal lines if enabled
            if self._style_params.diagonal_lines:
                self._diagonal_lines_and_labels(ax)

        # Set titles
        self._set_title()
        self._set_axes_titles()

        # Move legend if needed
        if self._style_params.legend_loc and self._style_params.legend_loc is not False:
            self._move_legend()

        return self

    def _set_style(self) -> None:
        """Set the style for the plot."""
        plt.style.use("seaborn-v0_8-whitegrid")

    def _circumplex_grid(self, ax: Axes) -> None:
        """
        Set up the grid for a circumplex plot.

        Parameters
        ----------
        ax : Axes
            The axes to set up the grid for
        """
        # Set up grid
        ax.grid(True, linestyle="--", alpha=0.7, zorder=0)
        
        # Set up minor ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        
        # Set up tick parameters
        ax.tick_params(which="both", direction="in")
        ax.tick_params(which="minor", length=4)
        ax.tick_params(which="major", length=7)

    def _set_title(self) -> None:
        """Set the title for the plot."""
        # If we have a figure title and no subplots, set it on the first axes
        if not self._has_subplots and self.title is not None:
            ax = self.get_single_axes()
            ax.set_title(self.title, fontsize=self._style_params.title_fontsize)
        
        # If we have a figure title and subplots, set it on the figure
        elif self._has_subplots and self.title is not None:
            fig = self.get_figure()
            fig.suptitle(self.title, fontsize=self._style_params.title_fontsize)

    def _set_axes_titles(self) -> None:
        """Set titles for individual axes in subplots."""
        # Set titles for individual subplot contexts if they have titles
        for i, context in enumerate(self.subplot_contexts):
            if context.title is not None:
                ax = self.get_single_axes(i)
                ax.set_title(context.title, fontsize=self._style_params.title_fontsize)

    def _primary_lines(self, ax: Axes) -> None:
        """
        Add primary axis lines to the plot.

        Parameters
        ----------
        ax : Axes
            The axes to add the lines to
        """
        # Add horizontal and vertical lines at x=0 and y=0
        ax.axhline(
            y=0,
            color="black",
            linestyle="-",
            linewidth=self._style_params.linewidth,
            zorder=self._style_params.prim_lines_zorder,
        )
        ax.axvline(
            x=0,
            color="black",
            linestyle="-",
            linewidth=self._style_params.linewidth,
            zorder=self._style_params.prim_lines_zorder,
        )

    def _primary_labels(self, ax: Axes) -> None:
        """
        Add labels for primary axes.

        Parameters
        ----------
        ax : Axes
            The axes to add the labels to
        """
        # Set x and y labels if they are provided
        if self._style_params.xlabel is not False:
            xlabel = self._style_params.xlabel or self.x
            ax.set_xlabel(
                xlabel,
                fontdict=self._style_params.prim_ax_fontdict,
            )
        
        if self._style_params.ylabel is not False:
            ylabel = self._style_params.ylabel or self.y
            ax.set_ylabel(
                ylabel,
                fontdict=self._style_params.prim_ax_fontdict,
            )

    def _diagonal_lines_and_labels(self, ax: Axes) -> None:
        """
        Add diagonal lines and labels to the plot.

        Parameters
        ----------
        ax : Axes
            The axes to add the diagonal lines and labels to
        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate diagonal line endpoints
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Determine the smaller range to keep lines within bounds
        range_min = min(x_range, y_range)
        
        # Calculate line endpoints
        diag1_start = (xlim[0], ylim[0])
        diag1_end = (xlim[0] + range_min, ylim[0] + range_min)
        
        diag2_start = (xlim[1], ylim[0])
        diag2_end = (xlim[1] - range_min, ylim[0] + range_min)
        
        # Draw diagonal lines
        ax.plot(
            [diag1_start[0], diag1_end[0]],
            [diag1_start[1], diag1_end[1]],
            color="black",
            linestyle="--",
            linewidth=self._style_params.linewidth,
            zorder=self._style_params.diag_lines_zorder,
        )
        
        ax.plot(
            [diag2_start[0], diag2_end[0]],
            [diag2_start[1], diag2_end[1]],
            color="black",
            linestyle="--",
            linewidth=self._style_params.linewidth,
            zorder=self._style_params.diag_lines_zorder,
        )
        
        # Add diagonal labels
        # Calculate positions for labels
        label_offset = 0.05  # Offset from the end of the line
        
        # First diagonal (bottom-left to top-right)
        diag1_label_pos = (
            diag1_end[0] - label_offset * x_range,
            diag1_end[1] + label_offset * y_range,
        )
        
        # Second diagonal (bottom-right to top-left)
        diag2_label_pos = (
            diag2_end[0] + label_offset * x_range,
            diag2_end[1] + label_offset * y_range,
        )
        
        # Add the labels
        ax.text(
            diag1_label_pos[0],
            diag1_label_pos[1],
            "Exciting",
            ha="right",
            va="bottom",
            rotation=45,
            zorder=self._style_params.diag_labels_zorder,
        )
        
        ax.text(
            diag2_label_pos[0],
            diag2_label_pos[1],
            "Chaotic",
            ha="left",
            va="bottom",
            rotation=-45,
            zorder=self._style_params.diag_labels_zorder,
        )
        
        # Add labels for the bottom half of the diagonals
        diag3_label_pos = (
            diag1_start[0] + label_offset * x_range,
            diag1_start[1] - label_offset * y_range,
        )
        
        diag4_label_pos = (
            diag2_start[0] - label_offset * x_range,
            diag2_start[1] - label_offset * y_range,
        )
        
        ax.text(
            diag3_label_pos[0],
            diag3_label_pos[1],
            "Boring",
            ha="left",
            va="top",
            rotation=45,
            zorder=self._style_params.diag_labels_zorder,
        )
        
        ax.text(
            diag4_label_pos[0],
            diag4_label_pos[1],
            "Calm",
            ha="right",
            va="top",
            rotation=-45,
            zorder=self._style_params.diag_labels_zorder,
        )

    def _move_legend(self) -> None:
        """Move the legend to the specified location."""
        # Get the figure
        fig = self.get_figure()
        
        # Find all legends in the figure
        legends = []
        for ax in self.yield_axes_objects():
            legend = ax.get_legend()
            if legend is not None:
                legends.append(legend)
        
        # If we have legends, move them to the specified location
        if legends:
            for legend in legends:
                legend.set_zorder(100)  # Ensure legend is on top
                
                # If legend_loc is specified, move the legend
                if self._style_params.legend_loc:
                    # Remove the legend from its current position
                    legend.remove()
                    
                    # Get the axes the legend belongs to
                    ax = legend.axes
                    
                    # Add the legend back at the specified location
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(
                            handles,
                            labels,
                            loc=self._style_params.legend_loc,
                        )