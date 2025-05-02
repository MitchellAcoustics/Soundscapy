"""
Main module for creating circumplex plots using Seaborn Objects API.

This module provides the CircumplexPlot class, which is a builder for creating
circumplex plots with a grammar of graphics approach. It allows for layering
of different plot elements (scatter, density) and customization of styling.
"""

import matplotlib.pyplot as plt
import seaborn.objects as so
import pandas as pd
import numpy as np
import seaborn as sns

from soundscapy.plotting.plotting_utils import (
    DEFAULT_XLIM,
    DEFAULT_YLIM,
    PlotType
)


# =============== Core building blocks for circumplex plots ===============

def apply_circumplex_grid(
    plot: so.Plot,
    xlim: tuple[float, float] = DEFAULT_XLIM,
    ylim: tuple[float, float] = DEFAULT_YLIM,
    diagonal_lines: bool = False,
    show_labels: bool = True,
    x_label: str | None = None,
    y_label: str | None = None
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
    # Apply limits and axes appearance
    plot = plot.limit(x=xlim, y=ylim)

    # Apply square aspect ratio and layout
    plot = plot.layout(size=(6, 6))

    # Compile the plot to a temporary figure to apply matplotlib styling
    # This creates a temporary figure for styling
    fig, ax = plt.subplots(figsize=(6, 6))

    # Add grid lines
    ax.grid(True, which="major", color='grey', alpha=0.5)
    ax.grid(True, which="minor", color='grey', linestyle='dashed',
           linewidth=0.5, alpha=0.4)
    ax.minorticks_on()

    # Add zero lines
    ax.axhline(y=0, color='grey', linestyle='dashed', alpha=1, linewidth=1.5)
    ax.axvline(x=0, color='grey', linestyle='dashed', alpha=1, linewidth=1.5)

    # Add diagonal elements if requested
    if diagonal_lines:
        # Draw diagonal lines
        ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]],
               linestyle='dashed', color='grey', alpha=0.5, linewidth=1.5)
        ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]],
               linestyle='dashed', color='grey', alpha=0.5, linewidth=1.5)

        # Add diagonal labels
        diag_font = {'fontstyle': 'italic', 'fontsize': 'small',
                     'fontweight': 'bold', 'color': 'black', 'alpha': 0.5}

        ax.text(xlim[1]/2, ylim[1]/2, "(vibrant)",
               ha='center', va='center', fontdict=diag_font)
        ax.text(xlim[0]/2, ylim[1]/2, "(chaotic)",
               ha='center', va='center', fontdict=diag_font)
        ax.text(xlim[0]/2, ylim[0]/2, "(monotonous)",
               ha='center', va='center', fontdict=diag_font)
        ax.text(xlim[1]/2, ylim[0]/2, "(calm)",
               ha='center', va='center', fontdict=diag_font)

    # Apply axis label changes if requested
    if not show_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")
    elif x_label is not None or y_label is not None:
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

    # Now use the fully prepared axes for the plot
    plot_with_styling = plot.on(ax)

    # Clean up the temporary figure - we don't need it as we've transferred the styling
    plt.close(fig)

    return plot_with_styling

def add_annotation(
        plot: so.Plot,
        data: pd.DataFrame,
        idx: int | str,
        x: str = "ISOPleasant",
        y: str = "ISOEventful",
        text: str | None = None,
        x_offset: float = 0.1,
        y_offset: float = 0.1,
        **kwargs
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
    # Get coordinates from data
    x_val = data[x].iloc[idx] if isinstance(idx, int) else data.loc[idx, x]
    y_val = data[y].iloc[idx] if isinstance(idx, int) else data.loc[idx, y]

    # Default text is the index value
    if text is None:
        text = str(data.index[idx]) if isinstance(idx, int) else str(idx)

    # Default annotation styling
    annotation_defaults = {
        "ha": "center",
        "va": "center",
        "fontsize": 9,
        "arrowprops": {"arrowstyle": "-", "color": "black", "alpha": 0.7}
    }
    annotation_defaults.update(kwargs)

    # Create a temporary figure to add the annotation
    fig, ax = plt.subplots(figsize=(6, 6))

    # Add the annotation
    ax.annotate(
        text=text,
        xy=(x_val, y_val),
        xytext=(x_val + x_offset, y_val + y_offset),
        **annotation_defaults
    )

    # Now use the fully prepared axes for the plot
    plot_with_annotation = plot.on(ax)

    # Clean up the temporary figure
    plt.close(fig)

    return plot_with_annotation

# =============== Main Circumplex Plot Class ===============

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
        **kwargs  # For backwards compatibility with tests
    ):
        self.data = data
        self.x = x
        self.y = y
        self.hue = hue
        self.xlim = xlim
        self.ylim = ylim
        self.fill = kwargs.get('fill', True)
        self.palette_name = palette

        # Initialize the plot
        self.plot = so.Plot(data, x=x, y=y)

        # Track what's been added
        self.has_scatter = False
        self.has_density = False
        self.has_grid = False

    def add_scatter(
        self,
        pointsize=30,
        alpha=0.7,
        marker="o",
        color=None  # Overrides hue if provided
    ):
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
            so.Dots(pointsize=pointsize, alpha=alpha, marker=marker),
            color=color_var
        )

        # If we have a color variable and palette, apply it using scale
        if color_var and hasattr(self, 'palette_name'):
            self.plot = self.plot.scale(color=so.Nominal(self.palette_name))

        self.has_scatter = True
        return self

    def add_density(
        self,
        alpha=0.5,
        fill=True,
        levels=8,
        bw_adjust=1.2,
        color=None,  # Overrides hue if provided
        simple=False,
        **kwargs  # For backwards compatibility
    ):
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
            levels = 2

        # Use the Area mark with KDE stat for density plots
        self.plot = self.plot.add(
            so.Area(alpha=alpha, fill=fill),
            so.KDE(bw_adjust=bw_adjust),
            color=color_var
        )

        # Apply palette if needed
        if color_var and hasattr(self, 'palette_name'):
            self.plot = self.plot.scale(color=so.Nominal(self.palette_name))

        # For simple density, add an outline
        if simple:
            self.plot = self.plot.add(
                so.Line(alpha=1.0),
                so.KDE(bw_adjust=bw_adjust),
                color=color_var
            )

            # Apply palette to the outline too if needed
            if color_var and hasattr(self, 'palette_name'):
                self.plot = self.plot.scale(color=so.Nominal(self.palette_name))

        self.has_density = True
        return self

    def add_grid(self, diagonal_lines=False, show_labels=True):
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
        self.plot = apply_circumplex_grid(
            self.plot,
            xlim=self.xlim,
            ylim=self.ylim,
            diagonal_lines=diagonal_lines,
            show_labels=show_labels,
            x_label=self.x if show_labels else None,
            y_label=self.y if show_labels else None
        )

        self.has_grid = True
        return self

    def add_annotation(self, idx, text=None, x_offset=0.1, y_offset=0.1, **kwargs):
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
        self.plot = add_annotation(
            self.plot,
            self.data,
            idx,
            x=self.x,
            y=self.y,
            text=text,
            x_offset=x_offset,
            y_offset=y_offset,
            **kwargs
        )

        return self

    def add_title(self, title):
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

    def add_legend(self, title=None, loc="best"):
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
        # For Seaborn Objects, we can handle this more directly
        # by adding a proper label to the plot

        # Just store the legend parameters for when we create the final plot
        self._legend_title = title if title is not None else self.hue
        self._legend_loc = loc

        return self

    def facet(self, column=None, row=None, col_wrap=None):
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

    def build(self, as_objects=True):
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
        if hasattr(self, '_legend_title') and self.hue is not None:
            # Create a label mapping for the hue variable
            self.plot = self.plot.label(**{self.hue: self._legend_title})

        if as_objects:
            return self.plot
        else:
            # Create a new figure with the right size
            fig, ax = plt.subplots(figsize=(6, 6))

            # Use pyplot mode for rendering
            # This will render the plot directly to the current figure
            self.plot.plot(pyplot=True)

            # Get the current axes
            ax = plt.gca()

            # Apply legend location if needed (after plotting)
            if hasattr(self, '_legend_loc') and ax.get_legend() is not None:
                ax.legend(loc=self._legend_loc)

            # Return the figure and axes to be compatible with the legacy API
            fig = ax.figure

            return fig, ax

    @property
    def seaborn_plot(self):
        """
        Return the underlying Seaborn Objects plot.

        Returns
        -------
        so.Plot
            The Seaborn Objects plot
        """
        return self.plot

    def get_matplotlib_objects(self):
        """
        Return matplotlib figure and axes objects.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes for the plot
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        self.plot.plot(ax=ax)
        return fig, ax

    def show(self):
        """
        Build and display the plot.

        Uses the proper pyplot=True approach which works in both
        notebook and non-notebook contexts.
        """
        # Ensure any default elements are added
        if not self.has_grid:
            self.add_grid()

        # This is the correct way to display a plot in a notebook
        self.plot.plot(pyplot=True)

    # Legacy API compatibility methods
    def scatter(self, apply_styling=True, ax=None):
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

    def density(self, apply_styling=True, ax=None):
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

    def simple_density(self, apply_styling=True, ax=None):
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

    def jointplot(self, apply_styling=True):
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
        ax.axhline(y=0, color='grey', linestyle='dashed', alpha=1, linewidth=1.5)
        ax.axvline(x=0, color='grey', linestyle='dashed', alpha=1, linewidth=1.5)

        # Add grid
        ax.grid(True, which="major", color='grey', alpha=0.5)

        # Store for get_figure and get_axes
        self._joint_grid = g

        return self

    def get_figure(self):
        """
        Get the figure object (legacy API compatibility).

        Returns
        -------
        plt.Figure or tuple
            Figure object or (figure, axes) tuple depending on plotting method
        """
        if hasattr(self, '_joint_grid'):
            return self._joint_grid.fig
        else:
            fig, ax = self.build(as_objects=False)
            return fig

    def get_axes(self):
        """
        Get the axes object (legacy API compatibility).

        Returns
        -------
        plt.Axes
            Axes object
        """
        if hasattr(self, '_joint_grid'):
            return self._joint_grid.ax_joint
        else:
            fig, ax = self.build(as_objects=False)
            return ax

    def iso_annotation(self, location, x_adj=0, y_adj=0, **kwargs):
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
