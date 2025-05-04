"""
Main module for creating circumplex plots using Seaborn Objects API.

This module provides the CircumplexPlot class, which is a builder for creating
circumplex plots with a grammar of graphics approach. It allows for layering
of different plot elements (scatter, density) and customization of styling.

Note: This class is maintained for backwards compatibility. For new code,
consider using the direct function-based API instead.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from soundscapy.plotting.marks import (
    SoundscapeCircumplex,
    SoundscapePointAnnotation,
    SoundscapeQuadrantLabels,
)
from soundscapy.plotting.plotting_utils import DEFAULT_XLIM, DEFAULT_YLIM
from soundscapy.plotting.soundscape_functions import use_soundscapy_style


def apply_circumplex_grid(
    plot: so.Plot,
    xlim: tuple[float, float] = DEFAULT_XLIM,
    ylim: tuple[float, float] = DEFAULT_YLIM,
    x_label: str | None = None,
    y_label: str | None = None,
    *,
    diagonal_lines: bool = False,
    show_labels: bool = True,
) -> so.Plot:
    """
    Apply circumplex grid styling to a Seaborn Objects plot.

    Parameters
    ----------
    plot : so.Plot
        The plot to style
    xlim, ylim : tuple
        Axis limits
    diagonal_lines : bool
        Whether to draw diagonal lines and quadrant labels
    show_labels : bool
        Whether to keep axis labels
    x_label, y_label : str, optional
        Custom labels for axes

    Returns
    -------
    so.Plot
        The styled plot

    """
    warnings.warn(
        "The apply_circumplex_grid function is deprecated. "
        "Use SoundscapeCircumplex and SoundscapeQuadrantLabels marks instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Apply limits and axes appearance
    plot = plot.limit(x=xlim, y=ylim)

    # Apply square aspect ratio and layout
    plot = plot.layout(size=(6, 6))

    # Add the circumplex grid mark
    plot = plot.add(SoundscapeCircumplex(xlim=xlim, ylim=ylim))

    # Add quadrant labels if requested
    if diagonal_lines:
        plot = plot.add(SoundscapeQuadrantLabels(xlim=xlim, ylim=ylim))

    # Apply axis label changes if requested
    if not show_labels:
        plot = plot.label(x=None, y=None)
    elif x_label is not None or y_label is not None:
        labels = {}
        if x_label is not None:
            labels["x"] = x_label
        if y_label is not None:
            labels["y"] = y_label
        plot = plot.label(**labels)

    return plot


def add_annotation(
    plot: so.Plot,
    data: pd.DataFrame,
    idx: int | str,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    text: str | None = None,
    x_offset: float = 0.1,
    y_offset: float = 0.1,
    **kwargs: Any,
) -> so.Plot:
    """
    Add an annotation to a Seaborn Objects plot.

    Parameters
    ----------
    plot : so.Plot
        The plot to annotate
    data : pd.DataFrame
        Data containing the point to annotate
    idx : int or str
        Index of the point to annotate
    x, y : str
        Column names for coordinates
    text : str, optional
        Text to display (defaults to index value if None)
    x_offset, y_offset : float
        Offsets for annotation position
    **kwargs
        Additional keyword arguments for annotation

    Returns
    -------
    so.Plot
        The plot with annotation added

    """
    warnings.warn(
        "The add_annotation function is deprecated. "
        "Use SoundscapePointAnnotation mark instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Get the text if not provided
    if text is None:
        text = str(data.index[idx]) if isinstance(idx, int) else str(idx)

    # Add the annotation using the mark
    plot = plot.add(
        SoundscapePointAnnotation(
            points=[idx], texts=[text], offsets=(x_offset, y_offset), **kwargs
        )
    )

    return plot


class CircumplexPlot:
    """
    A builder class for creating circumplex plots using Seaborn Objects API.

    This class allows for a layered grammar of graphics approach to building
    circumplex plots including scatter, density, and other elements.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x, y : str
        Column names for coordinates
    hue : str, optional
        Column name for color grouping
    xlim, ylim : tuple
        Axis limits for the plot
    palette : str
        Color palette to use

    """

    def __init__(
        self,
        data: pd.DataFrame,
        x: str = "ISOPleasant",
        y: str = "ISOEventful",
        hue: str | None = None,
        xlim: tuple[float, float] = DEFAULT_XLIM,
        ylim: tuple[float, float] = DEFAULT_YLIM,
        palette: str = "colorblind",
    ) -> None:
        """
        Initialize a CircumplexPlot instance.

        Parameters
        ----------
        data : pd.DataFrame
            Data to plot
        x : str, default="ISOPleasant"
            Column name for x-axis
        y : str, default="ISOEventful"
            Column name for y-axis
        hue : str, optional
            Column name for color grouping
        xlim : tuple[float, float]
            Axis limits for x-axis
        ylim : tuple[float, float]
            Axis limits for y-axis
        palette : str, default="colorblind"
            Color palette to use

        """
        warnings.warn(
            "The CircumplexPlot class is maintained for backwards compatibility. "
            "For new code, consider using the function-based API instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )

        # Apply the soundscapy style
        use_soundscapy_style()

        self.data = data
        self.x = x
        self.y = y
        self.hue = hue
        self.xlim = xlim
        self.ylim = ylim
        self.palette_name = palette

        # Initialize the plot
        self.plot = so.Plot(data, x=x, y=y)

        # Track what's been added
        self.has_scatter = False
        self.has_density = False
        self.has_grid = False

    def add_scatter(
        self,
        pointsize: float = 30,
        alpha: float = 0.7,
        marker: str = "o",
        color: str | None = None,  # Overrides hue if provided
    ) -> "CircumplexPlot":
        """
        Add a scatter layer to the plot.

        Parameters
        ----------
        pointsize : int or float
            Size of the scatter points
        alpha : float
            Opacity of the points
        marker : str
            Marker style
        color : str, optional
            Override for hue variable

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        color_var = color if color is not None else self.hue

        # Add the dots mark
        self.plot = self.plot.add(
            so.Dots(pointsize=pointsize, alpha=alpha, marker=marker), color=color_var
        )

        # If we have a color variable and palette, apply it using scale
        if color_var and hasattr(self, "palette_name"):
            self.plot = self.plot.scale(color=so.Nominal(self.palette_name))

        self.has_scatter = True
        return self

    def add_density(
        self,
        alpha: float = 0.5,
        fill: bool = True,
        levels: int = 8,
        bw_adjust: float = 1.2,
        color: str | None = None,  # Overrides hue if provided
        simple: bool = False,
        **kwargs: Any,  # For backwards compatibility
    ) -> "CircumplexPlot":
        """
        Add a density layer to the plot.

        Parameters
        ----------
        alpha : float
            Opacity of the fill color for the density
        fill : bool
            Whether to fill the contours
        levels : int
            Number of contour levels
        bw_adjust : float
            Bandwidth adjustment factor
        color : str, optional
            Override for hue variable
        simple : bool
            If True, use simplified density with fewer levels

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        color_var = color if color is not None else self.hue

        if simple:
            # For simple density, use just a few levels
            self.plot = self.plot.add(
                so.Area(alpha=alpha, fill=fill),
                so.KDE(bw_adjust=bw_adjust, levels=2),
                color=color_var,
            )

            # Apply palette if needed
            if color_var and hasattr(self, "palette_name"):
                self.plot = self.plot.scale(color=so.Nominal(self.palette_name))

            # Add outline
            self.plot = self.plot.add(
                so.Line(alpha=1.0),
                so.KDE(bw_adjust=bw_adjust, levels=2),
                color=color_var,
            )

            # Apply palette to the outline too if needed
            if color_var and hasattr(self, "palette_name"):
                self.plot = self.plot.scale(color=so.Nominal(self.palette_name))
        else:
            # Use the Area mark with KDE stat for regular density plots
            self.plot = self.plot.add(
                so.Area(alpha=alpha, fill=fill),
                so.KDE(bw_adjust=bw_adjust, levels=levels),
                color=color_var,
            )

            # Apply palette if needed
            if color_var and hasattr(self, "palette_name"):
                self.plot = self.plot.scale(color=so.Nominal(self.palette_name))

        self.has_density = True
        return self

    def add_grid(
        self, *, diagonal_lines: bool = False, show_labels: bool = True
    ) -> "CircumplexPlot":
        """
        Add circumplex grid to the plot.

        Parameters
        ----------
        diagonal_lines : bool
            Whether to show diagonal lines and quadrant labels
        show_labels : bool
            Whether to show axis labels

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        # Add the circumplex grid
        self.plot = self.plot.add(SoundscapeCircumplex(xlim=self.xlim, ylim=self.ylim))

        # Add quadrant labels if requested
        if diagonal_lines:
            self.plot = self.plot.add(
                SoundscapeQuadrantLabels(xlim=self.xlim, ylim=self.ylim)
            )

        # Hide labels if requested
        if not show_labels:
            self.plot = self.plot.label(x=None, y=None)

        self.has_grid = True
        return self

    def add_annotation(
        self,
        idx: int | str,
        text: str | None = None,
        x_offset: float = 0.1,
        y_offset: float = 0.1,
        **kwargs: Any,
    ) -> "CircumplexPlot":
        """
        Add an annotation to the plot.

        Parameters
        ----------
        idx : int or str
            Index of the point to annotate
        text : str, optional
            Text to display (defaults to index value if None)
        x_offset, y_offset : float
            Offsets for annotation position
        **kwargs
            Additional keyword arguments for annotation

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        # Get text if not provided
        if text is None:
            text = str(self.data.index[idx]) if isinstance(idx, int) else str(idx)

        # Add annotation
        self.plot = self.plot.add(
            SoundscapePointAnnotation(
                points=[idx], texts=[text], offsets=(x_offset, y_offset), **kwargs
            )
        )

        return self

    def add_title(self, title: str) -> "CircumplexPlot":
        """
        Add a title to the plot.

        Parameters
        ----------
        title : str
            Title text

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        self.plot = self.plot.label(title=title)
        return self

    def add_legend(
        self, title: str | None = None, loc: str = "best"
    ) -> "CircumplexPlot":
        """
        Customize the legend appearance.

        Parameters
        ----------
        title : str, optional
            Legend title (defaults to hue variable name)
        loc : str
            Legend location

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        # Store the legend parameters for when we create the final plot
        self._legend_title = title if title is not None else self.hue
        self._legend_loc = loc

        return self

    def facet(
        self,
        column: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
    ) -> "CircumplexPlot":
        """
        Add faceting to the plot.

        Parameters
        ----------
        column : str, optional
            Variable for column faceting
        row : str, optional
            Variable for row faceting
        col_wrap : int, optional
            Number of columns to wrap facets

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        self.plot = self.plot.facet(col=column, row=row, wrap=col_wrap)
        return self

    def build(self, as_objects: bool = True) -> so.Plot | tuple[plt.Figure, plt.Axes]:
        """
        Complete the plot with any default elements that haven't been added.

        Parameters
        ----------
        as_objects : bool
            If True, return the Seaborn Objects plot; if False, convert to Matplotlib axes

        Returns
        -------
        so.Plot or (plt.Figure, plt.Axes)
            The completed plot object or (figure, axes) tuple

        """
        # Add grid if not already added
        if not self.has_grid:
            self.add_grid()

        # Ensure correct aspect ratio
        self.plot = self.plot.layout(size=(6, 6))

        # Apply legend customization if requested
        if hasattr(self, "_legend_title") and self.hue is not None:
            # Create a label mapping for the hue variable
            # This sets the legend title to the specified value or the hue variable name
            self.plot = self.plot.label(**{self.hue: self._legend_title})

        if as_objects:
            return self.plot

        # Create a new figure with the right size
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw to the axes
        self.plot.plot(ax)

        # Apply legend location if needed (after plotting)
        if hasattr(self, "_legend_loc") and ax.get_legend() is not None:
            ax.legend(loc=self._legend_loc)

        # Return the figure and axes to be compatible with the legacy API
        return fig, ax

    @property
    def seaborn_plot(self) -> so.Plot:
        """
        Return the underlying Seaborn Objects plot.

        Returns
        -------
        so.Plot
            The Seaborn Objects plot

        """
        return self.plot

    def get_matplotlib_objects(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Return matplotlib figure and axes objects.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes for the plot

        """
        fig, ax = plt.subplots(figsize=(6, 6))
        self.plot.plot(ax)
        return fig, ax

    def show(self) -> None:
        """
        Build and display the plot.

        Uses the proper pyplot=True approach which works in both
        notebook and non-notebook contexts.
        """
        # Ensure any default elements are added
        if not self.has_grid:
            self.add_grid()

        # Draw the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        self.plot.plot(ax)
        plt.show()

    # Legacy API compatibility methods
    def scatter(
        self, apply_styling: bool = True, ax: plt.Axes | None = None
    ) -> "CircumplexPlot":
        """
        Create a scatter plot (legacy API compatibility).

        Parameters
        ----------
        apply_styling : bool
            Whether to apply styling (always True in this implementation)
        ax : plt.Axes, optional
            Axes to plot on (ignored in objects implementation)

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        self.add_scatter()
        self.add_grid()
        return self

    def density(
        self, apply_styling: bool = True, ax: plt.Axes | None = None
    ) -> "CircumplexPlot":
        """
        Create a density plot (legacy API compatibility).

        Parameters
        ----------
        apply_styling : bool
            Whether to apply styling (always True in this implementation)
        ax : plt.Axes, optional
            Axes to plot on (ignored in objects implementation)

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        self.add_density()
        self.add_grid()
        return self

    def simple_density(
        self, apply_styling: bool = True, ax: plt.Axes | None = None
    ) -> "CircumplexPlot":
        """
        Create a simple density plot (legacy API compatibility).

        Parameters
        ----------
        apply_styling : bool
            Whether to apply styling (always True in this implementation)
        ax : plt.Axes, optional
            Axes to plot on (ignored in objects implementation)

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        self.add_density(simple=True)
        self.add_grid()
        return self

    def jointplot(self, apply_styling: bool = True) -> "CircumplexPlot":
        """
        Create a joint plot (legacy API compatibility).

        Parameters
        ----------
        apply_styling : bool
            Whether to apply styling (not used in this implementation)

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        # Fall back to traditional seaborn for jointplot
        g = sns.jointplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.hue,
            kind="kde",
        )

        # Add grid elements to the central plot
        ax = g.ax_joint
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # Add zero lines
        ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
        ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
        # Add grid
        ax.grid(True, which="major", color="grey", alpha=0.5)
        # Store for get_figure and get_axes
        self._joint_grid = g

        return self

    def get_figure(self) -> plt.Figure | tuple[plt.Figure, plt.Axes]:
        """
        Get the figure object (legacy API compatibility).

        Returns
        -------
        plt.Figure or tuple
            Figure object or (figure, axes) tuple depending on plotting method

        """
        if hasattr(self, "_joint_grid"):
            return self._joint_grid.fig
        fig, ax = self.build(as_objects=False)
        return fig

    def get_axes(self) -> plt.Axes:
        """
        Get the axes object (legacy API compatibility).

        Returns
        -------
        plt.Axes
            Axes object

        """
        if hasattr(self, "_joint_grid"):
            return self._joint_grid.ax_joint
        fig, ax = self.build(as_objects=False)
        return ax

    def iso_annotation(
        self, location: int | str, x_adj: float = 0, y_adj: float = 0, **kwargs: Any
    ) -> "CircumplexPlot":
        """
        Add an annotation to the plot (legacy API compatibility).

        Parameters
        ----------
        location : int
            Index of the point to annotate
        x_adj, y_adj : float
            Offsets for annotation position
        **kwargs
            Additional keyword arguments for annotation

        Returns
        -------
        CircumplexPlot
            Self for method chaining

        """
        return self.add_annotation(location, x_offset=x_adj, y_offset=y_adj, **kwargs)
