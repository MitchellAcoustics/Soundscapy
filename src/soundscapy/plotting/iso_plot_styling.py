"""
Styling methods for ISOPlot.

This module provides a mixin class with methods for styling ISOPlot instances,
including grid lines, labels, and other visual elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.artist import Artist

from soundscapy.plotting.plotting_types import ParamModel
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = get_logger()


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
        Apply styling with default parameters

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
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
        ...       .apply_styling()
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
        ...         .apply_styling(xlim=(-2, 2), ylim=(-2, 2), primary_lines=False)
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
        ...     .apply_styling(title_fontsize=14)
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> # Verify results
        >>> isinstance(plot, ISOPlot)
        True
        >>> plot.close()  # Clean up

        """
        # Update style parameters with provided kwargs
        # Use the default values from the StyleParams class and override with kwargs
        self._style_params = ParamModel.create("style", **kwargs)
        # Check if we have axes to style
        self._check_for_axes()
        self._set_style()

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

    @staticmethod
    def _set_style() -> None:
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
        ax.set_xlim(self._style_params.get("xlim"))
        ax.set_ylim(self._style_params.get("ylim"))
        ax.set_aspect("equal")

        ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

        ax.grid(visible=True, which="major", color="grey", alpha=0.5)
        ax.grid(
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
            color="grey",
            linestyle="dashed",
            alpha=1,
            lw=self._style_params.get("linewidth"),
            zorder=self._style_params.get("prim_lines_zorder"),
        )
        ax.axvline(
            x=0,
            color="grey",
            linestyle="dashed",
            alpha=1,
            lw=self._style_params.get("linewidth"),
            zorder=self._style_params.get("prim_lines_zorder"),
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

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...    columns=['ISOPleasant', 'ISOEventful'])
        >>> # Create a plot with diagonal lines and labels
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .apply_styling(diagonal_lines=True)
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.close('all')

        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot(
            xlim,
            ylim,
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            lw=self._style_params.get("linewidth"),
            zorder=self._style_params.get("diag_lines_zorder"),
        )
        logger.debug("Plotting diagonal line for axis.")
        ax.plot(
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
        ax.text(
            xlim[1] / 2,
            ylim[1] / 2,
            "(vibrant)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self._style_params.get("diag_labels_zorder"),
        )
        ax.text(
            xlim[0] / 2,
            ylim[1] / 2,
            "(chaotic)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self._style_params.get("diag_labels_zorder"),
        )
        ax.text(
            xlim[0] / 2,
            ylim[0] / 2,
            "(monotonous)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self._style_params.get("diag_labels_zorder"),
        )
        ax.text(
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
