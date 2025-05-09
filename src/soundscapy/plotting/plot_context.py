"""
Data and state management for plotting layers.

This module provides the PlotContext class that manages data and state for ISOPlot
visualizations, enabling a more flexible architecture for plot generation with support
for layered visualizations and subplot management.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


class PlotContext:
    """
    Manages data and state for a plot or subplot.

    This class centralizes the management of data, coordinates, and other state
    needed for rendering plot layers, allowing for consistent data access patterns
    and simplified layer implementation.

    Attributes
    ----------
    data : pd.DataFrame
        The data associated with this context
    x : str
        The column name for x-axis data
    y : str
        The column name for y-axis data
    hue : str | None
        The column name for color encoding, if any
    ax : Axes | None
        The matplotlib Axes object this context is associated with
    title : str | None
        The title for this context's plot
    layers : list
        The visualization layers to be rendered on this context

    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        x: str = "ISOPleasant",
        y: str = "ISOEventful",
        hue: str | None = None,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> None:
        """
        Initialize a PlotContext.

        Parameters
        ----------
        data : pd.DataFrame | None
            Data to be visualized
        x : str
            Column name for x-axis data
        y : str
            Column name for y-axis data
        hue : str | None
            Column name for color encoding
        ax : Axes | None
            Matplotlib axis to render on
        title : str | None
            Title for this plot context

        """
        self.data = data
        self.x = x
        self.y = y
        self.hue = hue
        self.ax = ax
        self.title = title
        self.layers = []
        self.parent: PlotContext | None = None

    def create_child(
        self,
        data: pd.DataFrame | None = None,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> PlotContext:
        """
        Create a child context that inherits properties from this context.

        Parameters
        ----------
        data : pd.DataFrame | None
            Data for the child context. If None, inherits from parent.
        title : str | None
            Title for the child context
        ax : Axes | None
            Matplotlib axis for the child context

        Returns
        -------
        PlotContext
            A new child context with inherited properties

        """
        child = PlotContext(
            data=data if data is not None else self.data,
            x=self.x,
            y=self.y,
            hue=self.hue,
            ax=ax,
            title=title,
        )
        child.parent = self
        return child

    def traverse_parents(self) -> Generator[PlotContext, None, None]:
        """
        Yield the chain of parent PlotContext objects.

        Works by traversing upwards from the current PlotContext instance.

        Yields
        ------
        Generator[PlotContext, None, None]
            A generator object that yields each parent PlotContext object, starting
            from the immediate parent of the current context and moving upward to the
            root of the hierarchy.

        """
        context = self
        while context.parent is not None:
            yield context.parent
            context = context.parent

    def traverse_children(self) -> Generator[PlotContext, None, None]:
        """
        Yield the chain of child PlotContext objects.

        Works by traversing downwards from the current PlotContext instance.

        Yields
        ------
        Generator[PlotContext, None, None]
            A generator object that yields each child PlotContext object, starting
            from the immediate child of the current context and moving downward to the
            leaf of the hierarchy.

        """
        for layer in self.layers:
            yield from layer.traverse_children()
