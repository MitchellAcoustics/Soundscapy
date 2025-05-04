"""
Base module for creating soundscape plots using Seaborn Objects API.

This module provides the SoundscapePlot class, which directly subclasses
seaborn.objects.Plot to create specialized visualizations for soundscape data.
"""

import copy
import functools
import inspect
from typing import Any, Callable, TypeVar, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from seaborn._core.typing import (
    OrderSpec,
    VariableSpec,
    VariableSpecList,
)

from soundscapy.plotting.plotting_utils import DEFAULT_XLIM, DEFAULT_YLIM

# The TypeVar helps with type annotations for method chaining
P = TypeVar("P", bound="SoundscapePlot")


def _fix_return_type(method: Callable) -> Callable:
    """
    Decorator to ensure methods return the correct subclass type.

    This decorator wraps methods that would normally return so.Plot
    and ensures they return the subclass type instead. It transfers
    custom state attributes from the original object to the result,
    which is especially important for methods that return a new plot
    instance rather than modifying the current one.

    Parameters
    ----------
    method : Callable
        The method to wrap

    Returns
    -------
    Callable
        A wrapped method that preserves SoundscapePlot type
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Call the original method
        result = method(self, *args, **kwargs)

        # If the result is a Plot instance but not our subclass (or a subclass of our subclass),
        # we need to transfer our custom state and cast to the correct type
        if isinstance(result, so.Plot) and not isinstance(result, type(self)):
            # Identify custom attributes (starting with underscore but not dunder methods)
            # that exist in self but not in the result
            for attr_name in dir(self):
                if (
                    attr_name.startswith("_")
                    and not attr_name.startswith("__")
                    and hasattr(self, attr_name)
                    and not hasattr(result, attr_name)
                ):
                    try:
                        # Try to deep copy the attribute for proper immutability
                        setattr(
                            result, attr_name, copy.deepcopy(getattr(self, attr_name))
                        )
                    except (TypeError, AttributeError):
                        # Fall back to regular assignment if deep copy fails
                        setattr(result, attr_name, getattr(self, attr_name))

            # Cast the result to our actual class type (preserving exact subclass)
            result = cast(type(self), result)

        return result

    return wrapper


class SoundscapePlot(so.Plot):
    """
    Base class for soundscape visualization that extends seaborn's Plot.

    This class subclasses seaborn.objects.Plot to create a specialized
    plotting API for soundscape data visualization while maintaining
    full compatibility with the Seaborn Objects interface. It includes
    methods like add_grid() that provide soundscape-specific functionality,
    while ensuring all inherited seaborn methods return the correct type.

    Examples
    --------
    >>> import soundscapy as sspy
    >>> from soundscapy.plotting import SoundscapePlot
    >>> data = sspy.isd.load()
    >>> data = sspy.surveys.add_iso_coords(data, overwrite=True)
    >>> plot = (SoundscapePlot(data)
    ...         .add(so.Dots(), color="location_id")
    ...         .add_grid(diagonal_lines=True)
    ...         .label(title="Soundscape Plot"))
    >>> plot.show()

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x, y : str
        Column names for coordinates (default to ISO coordinate names)
    xlim, ylim : tuple[float, float]
        Axis limits for the plot (default to standard -1,1 range)
    **kwargs
        Additional parameters to pass to so.Plot.__init__

    """

    def __init__(
        self,
        data: pd.DataFrame,
        x: str = "ISOPleasant",
        y: str = "ISOEventful",
        xlim: tuple[float, float] = DEFAULT_XLIM,
        ylim: tuple[float, float] = DEFAULT_YLIM,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a SoundscapePlot object.

        Parameters
        ----------
        data : pd.DataFrame
            Data to plot
        x : str, default="ISOPleasant"
            Column name for x-axis, defaults to ISO pleasantness dimension
        y : str, default="ISOEventful"
            Column name for y-axis, defaults to ISO eventfulness dimension
        **kwargs : Any
            Additional parameters to pass to the seaborn Plot constructor

        """
        # Initialize with soundscape defaults
        super().__init__(data, x=x, y=y, **kwargs)

        # Store soundscape-specific state as private attributes
        self._xlim = xlim
        self._ylim = ylim
        self._has_grid = False

    def _clone(self: P) -> P:
        """
        Create a copy of this plot object with deep-copied state.

        This method is crucial for the immutable builder pattern.

        Returns
        -------
        SoundscapePlot
            A new plot with copied state

        """
        new = super()._clone()
        # Copy our custom attributes
        new._xlim = copy.deepcopy(self._xlim)
        new._ylim = copy.deepcopy(self._ylim)
        new._has_grid = self._has_grid
        return cast(P, new)

    @_fix_return_type
    def add_grid(
        self: P,
        xlim: tuple[float, float] = DEFAULT_XLIM,
        ylim: tuple[float, float] = DEFAULT_YLIM,
        x_label: str | None = None,
        y_label: str | None = None,
        *,
        diagonal_lines: bool = False,
        show_labels: bool = True,
    ) -> P:
        """
        Add a standard soundscape analysis grid with zero lines and optionally diagonal lines.

        Parameters
        ----------
        xlim, ylim : tuple
            Axis limits
        x_label, y_label : str, optional
            Custom labels for axes
        diagonal_lines : bool
            Whether to draw diagonal lines and quadrant labels
        show_labels : bool
            Whether to keep axis labels

        Returns
        -------
        SoundscapePlot
            The plot with grid styling applied

        """
        # Clone first to maintain immutability
        new_plot = self._clone()

        # Apply limits and square aspect ratio
        new_plot = cast(P, new_plot.limit(x=xlim, y=ylim).layout(size=(6, 6)))

        # Create a temporary matplotlib figure for styling
        fig, ax = plt.subplots(figsize=(6, 6))

        # Add zero lines
        ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
        ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)

        # Add grid
        ax.grid(visible=True, which="major", color="grey", alpha=0.5)
        ax.grid(
            visible=True,
            which="minor",
            color="grey",
            linestyle="dashed",
            linewidth=0.5,
            alpha=0.4,
        )
        ax.minorticks_on()

        # Handle diagonal lines if requested
        if diagonal_lines:
            # Add diagonal lines
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[0], ylim[1]],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                linewidth=1.5,
            )
            ax.plot(
                [xlim[0], xlim[1]],
                [ylim[1], ylim[0]],
                linestyle="dashed",
                color="grey",
                alpha=0.5,
                linewidth=1.5,
            )

            # Add quadrant labels if requested
            if show_labels:
                diag_font = {
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
                    fontdict=diag_font,
                )
                ax.text(
                    xlim[0] / 2,
                    ylim[1] / 2,
                    "(chaotic)",
                    ha="center",
                    va="center",
                    fontdict=diag_font,
                )
                ax.text(
                    xlim[0] / 2,
                    ylim[0] / 2,
                    "(monotonous)",
                    ha="center",
                    va="center",
                    fontdict=diag_font,
                )
                ax.text(
                    xlim[1] / 2,
                    ylim[0] / 2,
                    "(calm)",
                    ha="center",
                    va="center",
                    fontdict=diag_font,
                )

        # Apply axis label changes
        if not show_labels:
            ax.set_xlabel("")
            ax.set_ylabel("")
        elif x_label is not None or y_label is not None:
            if x_label is not None:
                ax.set_xlabel(x_label)
            if y_label is not None:
                ax.set_ylabel(y_label)

        # Transfer styling to the plot - cast to ensure type correctness
        plot_with_styling = cast(P, new_plot.on(ax))

        # Clean up the temporary figure
        plt.close(fig)

        # Update state to track that grid has been added
        plot_with_styling._has_grid = True
        plot_with_styling._xlim = xlim
        plot_with_styling._ylim = ylim

        return plot_with_styling

    def show(self: P, **kwargs) -> None:
        """
        Display the plot.

        This is a convenience method that ensures grid is added
        and handles proper rendering in both notebook and script contexts.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib.pyplot.show()

        """
        # Ensure grid is added if not already
        plot_to_show = self
        if not plot_to_show._has_grid:
            plot_to_show = plot_to_show.add_grid()

        # Use the correctly configured pyplot mode and pass through all kwargs
        plot_to_show.plot(pyplot=True).show(**kwargs)

    # We don't need to explicitly override parent methods that don't add custom functionality
    # The __init_subclass__ method will automatically apply the _fix_return_type decorator
    # to all inherited methods from so.Plot

    # Apply the _fix_return_type decorator to all parent methods automatically
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Ensure all inherited so.Plot methods maintain proper return types.

        This special method is called whenever a subclass of SoundscapePlot is created.
        It automatically applies the _fix_return_type decorator to all methods
        inherited from so.Plot that haven't been explicitly overridden, ensuring
        consistent return type behavior throughout the inheritance hierarchy.
        """
        super().__init_subclass__(**kwargs)

        # Find all eligible methods in the so.Plot class
        for name, method in inspect.getmembers(so.Plot, inspect.isfunction):
            # Apply the decorator if:
            # 1. Method isn't already defined in this class
            # 2. Method isn't private (doesn't start with _)
            # 3. Method has 'self' as a parameter (instance method)
            if (
                name not in cls.__dict__
                and not name.startswith("_")
                and "self" in inspect.signature(method).parameters
            ):
                # Add the wrapped method to our class
                setattr(cls, name, _fix_return_type(method))

    # Apply the same fix to the current class after definition
    @classmethod
    def _apply_return_type_fix(cls):
        """
        Apply the return type fix to the current class.

        This is called at the end of the class definition to ensure that
        even the base SoundscapePlot class gets the decorator applied to
        its inherited methods.
        """
        # Similar logic to __init_subclass__, but for the current class
        for name, method in inspect.getmembers(so.Plot, inspect.isfunction):
            if (
                name not in cls.__dict__
                and not name.startswith("_")
                and "self" in inspect.signature(method).parameters
            ):
                setattr(cls, name, _fix_return_type(method))


# Apply the return type fix to SoundscapePlot itself
SoundscapePlot._apply_return_type_fix()
