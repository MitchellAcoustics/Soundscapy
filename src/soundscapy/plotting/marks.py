"""
Custom Mark components for Soundscape plots using seaborn.objects.

This module contains custom Mark classes that extend the functionality of
seaborn.objects.Plot for creating soundscape plots. These marks handle
specialized plot elements like circumplex grids, quadrant labels, and
point annotations.
"""

import seaborn.objects as so

from soundscapy.plotting.plotting_utils import DEFAULT_XLIM, DEFAULT_YLIM


class SoundscapeCircumplex(so.Mark):
    """Mark for adding circumplex grid elements to a plot."""

    def __init__(
        self,
        xlim=DEFAULT_XLIM,
        ylim=DEFAULT_YLIM,
        primary_color="grey",
        primary_alpha=1.0,
        primary_linewidth=1.5,
        grid_color="grey",
        grid_alpha=0.5,
        **kwargs,
    ):
        """
        Initialize the circumplex grid mark.

        Parameters
        ----------
        xlim : tuple, default=(-1, 1)
            X-axis limits
        ylim : tuple, default=(-1, 1)
            Y-axis limits
        primary_color : str, default="grey"
            Color for primary elements (zero lines)
        primary_alpha : float, default=1.0
            Alpha transparency for primary elements
        primary_linewidth : float, default=1.5
            Line width for primary elements
        grid_color : str, default="grey"
            Color for grid lines
        grid_alpha : float, default=0.5
            Alpha transparency for grid lines
        **kwargs :
            Additional keyword arguments passed to parent class

        """
        self.xlim = xlim
        self.ylim = ylim
        self.primary_color = primary_color
        self.primary_alpha = primary_alpha
        self.primary_linewidth = primary_linewidth
        self.grid_color = grid_color
        self.grid_alpha = grid_alpha
        super().__init__(**kwargs)

    def _plot(self, axes, data, x, y, **kwargs):
        """
        Draw circumplex grid elements on the plot.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes to draw on
        data : pd.DataFrame
            The plot data
        x, y : str
            Column names for coordinates
        **kwargs :
            Additional keyword arguments

        Returns
        -------
        self : Mark
            The mark instance for method chaining

        """
        # Apply limits
        axes.set_xlim(self.xlim)
        axes.set_ylim(self.ylim)

        # Add major and minor grid
        axes.grid(
            visible=True, which="major", color=self.grid_color, alpha=self.grid_alpha
        )
        axes.grid(
            visible=True,
            which="minor",
            color=self.grid_color,
            linestyle="dashed",
            linewidth=0.5,
            alpha=self.grid_alpha * 0.8,
        )
        axes.minorticks_on()

        # Add zero lines
        axes.axhline(
            y=0,
            color=self.primary_color,
            linestyle="dashed",
            alpha=self.primary_alpha,
            linewidth=self.primary_linewidth,
        )
        axes.axvline(
            x=0,
            color=self.primary_color,
            linestyle="dashed",
            alpha=self.primary_alpha,
            linewidth=self.primary_linewidth,
        )

        return self


class SoundscapeQuadrantLabels(so.Mark):
    """Mark for adding diagonal lines and quadrant labels to a soundscape plot."""

    def __init__(
        self,
        xlim=DEFAULT_XLIM,
        ylim=DEFAULT_YLIM,
        line_color="grey",
        line_alpha=0.5,
        line_width=1.5,
        label_color="black",
        label_alpha=0.5,
        label_size="small",
        labels=("(vibrant)", "(chaotic)", "(monotonous)", "(calm)"),
        **kwargs,
    ):
        """
        Initialize the quadrant labels mark.

        Parameters
        ----------
        xlim : tuple, default=(-1, 1)
            X-axis limits
        ylim : tuple, default=(-1, 1)
            Y-axis limits
        line_color : str, default="grey"
            Color for diagonal lines
        line_alpha : float, default=0.5
            Alpha transparency for diagonal lines
        line_width : float, default=1.5
            Line width for diagonal lines
        label_color : str, default="black"
            Color for quadrant labels
        label_alpha : float, default=0.5
            Alpha transparency for quadrant labels
        label_size : str or int, default="small"
            Font size for quadrant labels
        labels : tuple, default=("(vibrant)", "(chaotic)", "(monotonous)", "(calm)")
            Text for each quadrant label
        **kwargs :
            Additional keyword arguments passed to parent class

        """
        self.xlim = xlim
        self.ylim = ylim
        self.line_color = line_color
        self.line_alpha = line_alpha
        self.line_width = line_width
        self.label_color = label_color
        self.label_alpha = label_alpha
        self.label_size = label_size
        self.labels = labels
        super().__init__(**kwargs)

    def _plot(self, axes, data, x, y, **kwargs):
        """
        Draw diagonal lines and quadrant labels.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes to draw on
        data : pd.DataFrame
            The plot data
        x, y : str
            Column names for coordinates
        **kwargs :
            Additional keyword arguments

        Returns
        -------
        self : Mark
            The mark instance for method chaining

        """
        # Draw diagonal lines
        axes.plot(
            [self.xlim[0], self.xlim[1]],
            [self.ylim[0], self.ylim[1]],
            linestyle="dashed",
            color=self.line_color,
            alpha=self.line_alpha,
            linewidth=self.line_width,
        )
        axes.plot(
            [self.xlim[0], self.xlim[1]],
            [self.ylim[1], self.ylim[0]],
            linestyle="dashed",
            color=self.line_color,
            alpha=self.line_alpha,
            linewidth=self.line_width,
        )

        # Add quadrant labels
        label_style = {
            "fontstyle": "italic",
            "fontsize": self.label_size,
            "fontweight": "bold",
            "color": self.label_color,
            "alpha": self.label_alpha,
        }

        # Upper right (vibrant)
        axes.text(
            self.xlim[1] / 2,
            self.ylim[1] / 2,
            self.labels[0],
            ha="center",
            va="center",
            fontdict=label_style,
        )

        # Upper left (chaotic)
        axes.text(
            self.xlim[0] / 2,
            self.ylim[1] / 2,
            self.labels[1],
            ha="center",
            va="center",
            fontdict=label_style,
        )

        # Lower left (monotonous)
        axes.text(
            self.xlim[0] / 2,
            self.ylim[0] / 2,
            self.labels[2],
            ha="center",
            va="center",
            fontdict=label_style,
        )

        # Lower right (calm)
        axes.text(
            self.xlim[1] / 2,
            self.ylim[0] / 2,
            self.labels[3],
            ha="center",
            va="center",
            fontdict=label_style,
        )

        return self


class SoundscapePointAnnotation(so.Mark):
    """Mark for adding annotations to specific points in a soundscape plot."""

    def __init__(
        self,
        points=None,
        texts=None,
        offsets=(0.1, 0.1),
        fontsize=9,
        ha="center",
        va="center",
        arrow_style="-",
        arrow_color="black",
        arrow_alpha=0.7,
        **kwargs,
    ):
        """
        Initialize the point annotation mark.

        Parameters
        ----------
        points : list, default=None
            List of point indices to annotate
        texts : list, default=None
            List of texts to use for annotations
        offsets : tuple, default=(0.1, 0.1)
            (x_offset, y_offset) for annotation position
        fontsize : int, default=9
            Font size for annotation text
        ha : str, default="center"
            Horizontal alignment
        va : str, default="center"
            Vertical alignment
        arrow_style : str, default="-"
            Style of the annotation arrow
        arrow_color : str, default="black"
            Color of the annotation arrow
        arrow_alpha : float, default=0.7
            Alpha transparency of the annotation arrow
        **kwargs :
            Additional keyword arguments passed to parent class

        """
        self.points = points if points is not None else []
        self.texts = texts if texts is not None else []
        self.offsets = offsets
        self.fontsize = fontsize
        self.ha = ha
        self.va = va
        self.arrow_props = {
            "arrowstyle": arrow_style,
            "color": arrow_color,
            "alpha": arrow_alpha,
        }
        super().__init__(**kwargs)

    def _plot(self, axes, data, x, y, **kwargs):
        """
        Add annotations to specific points.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes to draw on
        data : pd.DataFrame
            The plot data
        x, y : str
            Column names for coordinates
        **kwargs :
            Additional keyword arguments

        Returns
        -------
        self : Mark
            The mark instance for method chaining

        """
        if not self.points:
            return self

        # Add annotations for each point
        for i, point_idx in enumerate(self.points):
            try:
                # Get coordinates from data
                if isinstance(point_idx, int):
                    x_val = data[x].iloc[point_idx]
                    y_val = data[y].iloc[point_idx]
                else:
                    x_val = data.loc[point_idx, x]
                    y_val = data.loc[point_idx, y]

                # Get text (default to index if not provided)
                if i < len(self.texts):
                    text = self.texts[i]
                else:
                    text = str(point_idx)

                # Add annotation
                axes.annotate(
                    text=text,
                    xy=(x_val, y_val),
                    xytext=(x_val + self.offsets[0], y_val + self.offsets[1]),
                    fontsize=self.fontsize,
                    ha=self.ha,
                    va=self.va,
                    arrowprops=self.arrow_props,
                )
            except (KeyError, IndexError):
                # Skip invalid indices
                pass

        return self
