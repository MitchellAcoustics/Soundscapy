"""
Styling utilities for circumplex plots using Seaborn and Matplotlib.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import matplotlib as mpl
import seaborn as sns

from .plotting_utils import DEFAULT_FIGSIZE


@dataclass
class StyleOptions:
    """
    Configuration options for styling circumplex plots.

    Attributes:
        diag_lines_zorder (int): Z-order for diagonal lines.
        diag_labels_zorder (int): Z-order for diagonal labels.
        prim_lines_zorder (int): Z-order for primary lines.
        data_zorder (int): Z-order for plotted data.
        bw_adjust (float): Bandwidth adjustment for kernel density estimation.
        figsize (Tuple[int, int]): Figure size (width, height) in inches.
        simple_density (Dict[str, Any]): Configuration for simple density plots.
    """

    diag_lines_zorder: int = 1
    diag_labels_zorder: int = 4
    prim_lines_zorder: int = 2
    data_zorder: int = 3
    bw_adjust: float = 1.2
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE
    simple_density: Dict[str, Any] = field(
        default_factory=lambda: {
            "thresh": 0.5,
            "levels": 2,
            "incl_outline": True,
            "alpha": 0.5,
        }
    )


class SeabornStyler:
    """
    Class for applying Seaborn styles to circumplex plots.
    """

    def __init__(self, params: Any, style_options: StyleOptions = StyleOptions()):
        self.params = params
        self.style_options = style_options

    def apply_styling(
        self, fig: mpl.figure.Figure, ax: mpl.axes.Axes
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Apply styling to the plot.

        Parameters
        ----------
            fig (mpl.figure.Figure): The figure object.
            ax (mpl.axes.Axes): The axes object.

        Returns
        -------
            Tuple[mpl.figure.Figure, mpl.axes.Axes]: The styled figure and axes.
        """
        self.set_style()
        self.circumplex_grid(ax)
        self.set_circum_title(ax)
        self.deal_w_default_labels(ax)
        if self.params.hue and self.params.legend is not False:
            self.move_legend(ax)
        return fig, ax

    def set_style(self) -> None:
        """Set the overall style for the plot."""
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    def circumplex_grid(self, ax: mpl.axes.Axes) -> None:
        """Add the circumplex grid to the plot."""
        ax.set_xlim(self.params.xlim)
        ax.set_ylim(self.params.ylim)

        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

        ax.grid(visible=True, which="major", color="grey", alpha=0.5)
        ax.grid(
            visible=True,
            which="minor",
            color="grey",
            linestyle="dashed",
            linewidth=0.5,
            alpha=0.4,
            zorder=self.style_options.prim_lines_zorder,
        )

        self.primary_lines_and_labels(ax)
        if self.params.diagonal_lines:
            self.diagonal_lines_and_labels(ax)

    def set_circum_title(self, ax: mpl.axes.Axes) -> None:
        """Set the title for the circumplex plot."""
        title_pad = 6.0
        ax.set_title(self.params.title, pad=title_pad)

    def deal_w_default_labels(self, ax: mpl.axes.Axes) -> None:
        """Handle the default labels for the axes."""
        if not self.params.show_labels:
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel(self.params.x)
            ax.set_ylabel(self.params.y)

    def move_legend(self, ax: mpl.axes.Axes) -> None:
        """Move the legend to the specified location."""
        old_legend = ax.get_legend()
        handles = old_legend.legend_handles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        ax.legend(handles, labels, loc=self.params.legend_location, title=title)

    def primary_lines_and_labels(self, ax: mpl.axes.Axes) -> None:
        """Add primary lines to the plot."""
        line_weights = 1.5

        ax.axhline(
            y=0,
            color="grey",
            linestyle="dashed",
            alpha=1,
            lw=line_weights,
            zorder=self.style_options.prim_lines_zorder,
        )
        ax.axvline(
            x=0,
            color="grey",
            linestyle="dashed",
            alpha=1,
            lw=line_weights,
            zorder=self.style_options.prim_lines_zorder,
        )

    def diagonal_lines_and_labels(self, ax: mpl.axes.Axes) -> None:
        """Add diagonal lines and labels to the plot."""
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        line_weights = 1.5

        ax.plot(
            [x_lim[0], x_lim[1]],
            [y_lim[0], y_lim[1]],
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            lw=line_weights,
            zorder=self.style_options.diag_lines_zorder,
        )
        ax.plot(
            [x_lim[0], x_lim[1]],
            [y_lim[1], y_lim[0]],
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            lw=line_weights,
            zorder=self.style_options.diag_lines_zorder,
        )

        diag_ax_font = {
            "fontstyle": "italic",
            "fontsize": "small",
            "fontweight": "bold",
            "color": "black",
            "alpha": 0.5,
        }
        ax.text(
            x_lim[1] / 2,
            y_lim[1] / 2,
            "(vibrant)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self.style_options.diag_labels_zorder,
        )
        ax.text(
            x_lim[0] / 2,
            y_lim[1] / 2,
            "(chaotic)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self.style_options.diag_labels_zorder,
        )
        ax.text(
            x_lim[0] / 2,
            y_lim[0] / 2,
            "(monotonous)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self.style_options.diag_labels_zorder,
        )
        ax.text(
            x_lim[1] / 2,
            y_lim[0] / 2,
            "(calm)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=self.style_options.diag_labels_zorder,
        )
