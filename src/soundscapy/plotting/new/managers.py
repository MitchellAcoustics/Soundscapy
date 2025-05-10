"""
Manager classes for the plotting module.

This module provides manager classes that encapsulate specific functionality
for the ISOPlot class. These managers replace the mixin-based approach in the
original implementation, using composition instead of inheritance.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from soundscapy.plotting.new.constants import (
    RECOMMENDED_MIN_SAMPLES,
)
from soundscapy.plotting.new.layer import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPISimpleLayer,
)
from soundscapy.plotting.new.protocols import RenderableLayer
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from soundscapy.plotting.new.plot_context import PlotContext

logger = get_logger()


class LayerManager:
    """
    Manages the creation and rendering of visualization layers.

    This class encapsulates the layer-related functionality that was previously
    implemented as a mixin in the ISOPlot class.

    Attributes
    ----------
    plot : Any
        The parent plot instance

    """

    def __init__(self, plot: Any) -> None:
        """
        Initialize a LayerManager.

        Parameters
        ----------
        plot : Any
            The parent plot instance

        """
        self.plot = plot

    def add_scatter(
        self,
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> Any:
        """
        Add a scatter layer to the plot.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the scatter layer

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        return self.add_layer(
            ScatterLayer,
            data=data,
            on_axis=on_axis,
            **params,
        )

    def add_density(
        self,
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> Any:
        """
        Add a density layer to the plot.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the density layer

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        # Check if we have enough data for a density plot
        plot_data = data if data is not None else self.plot.main_context.data
        if plot_data is not None and len(plot_data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for "
                f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

        return self.add_layer(
            DensityLayer,
            data=data,
            on_axis=on_axis,
            **params,
        )

    def add_simple_density(
        self,
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> Any:
        """
        Add a simple density layer to the plot.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the simple density layer

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        return self.add_layer(
            SimpleDensityLayer,
            data=data,
            on_axis=on_axis,
            **params,
        )

    def add_spi_simple(
        self,
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> Any:
        """
        Add an SPI simple layer to the plot.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the SPI simple layer

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        return self.add_layer(
            SPISimpleLayer,
            data=data,
            on_axis=on_axis,
            **params,
        )

    def add_layer(
        self,
        layer_class: type[RenderableLayer],
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> Any:
        """
        Add a visualization layer, optionally targeting specific subplot(s).

        Parameters
        ----------
        layer_class : type[RenderableLayer]
            The type of layer to add
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the layer

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        # Create the layer instance
        layer = cast("Layer", layer_class(custom_data=data, **params))

        # Check if we have axes to render on
        self._check_for_axes()

        # If no subplots created yet, add to main context
        if not self.plot.subplot_contexts:
            if self.plot.main_context.ax is None:
                # Get the single axis and assign it to main context
                if isinstance(self.plot.axes, Axes):
                    self.plot.main_context.ax = self.plot.axes
                elif isinstance(self.plot.axes, np.ndarray) and self.plot.axes.size > 0:
                    self.plot.main_context.ax = self.plot.axes.flatten()[0]

            # Add layer to main context
            self.plot.main_context.layers.append(layer)
            # Render the layer immediately
            layer.render(self.plot.main_context)
            return self.plot

        # Handle various axis targeting options
        target_contexts = self._resolve_target_contexts(on_axis)

        # Add the layer to each target context and render it
        for context in target_contexts:
            context.layers.append(layer)
            layer.render(context)

        return self.plot

    def _check_for_axes(self) -> None:
        """
        Check if we have axes to render on, create if needed.

        This method ensures that the plot has axes to render on,
        creating them if necessary.
        """
        if self.plot.figure is None:
            # Create a new figure and axes
            self.plot.figure, self.plot.axes = plt.subplots(figsize=(5, 5))

    def _resolve_target_contexts(
        self, on_axis: int | tuple[int, int] | list[int] | None
    ) -> list[PlotContext]:
        """
        Resolve which subplot contexts to target based on axis specification.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None
            The axis specification:
            - None: All subplot contexts
            - int: Single subplot at flattened index
            - tuple[int, int]: Subplot at (row, col)
            - list[int]: Multiple subplots at specified indices

        Returns
        -------
        list[PlotContext]
            List of target subplot contexts

        """
        # If no specific axis, target all subplot contexts
        if on_axis is None:
            return self.plot.subplot_contexts

        # Convert axis specification to list of indices
        indices = self._resolve_axis_indices(on_axis)

        # Get the contexts for each valid index
        target_contexts = []
        for idx in indices:
            if 0 <= idx < len(self.plot.subplot_contexts):
                target_contexts.append(self.plot.subplot_contexts[idx])
            else:
                msg = f"Subplot index {idx} out of range"
                raise IndexError(msg)

        return target_contexts

    def _resolve_axis_indices(
        self, on_axis: int | tuple[int, int] | list[int]
    ) -> list[int]:
        """
        Convert axis specification to list of indices.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int]
            The axis specification to resolve

        Returns
        -------
        list[int]
            List of flattened indices

        Raises
        ------
        ValueError
            If an invalid axis specification is provided

        """
        if isinstance(on_axis, int):
            return [on_axis]
        if isinstance(on_axis, tuple) and len(on_axis) == 2:
            # Convert (row, col) to flattened index
            row, col = on_axis
            return [row * self.plot.subplots_params.ncols + col]
        if isinstance(on_axis, list):
            return on_axis
        msg = f"Invalid axis specification: {on_axis}"
        raise ValueError(msg)


class StyleManager:
    """
    Manages the styling of plots.

    This class encapsulates the styling-related functionality that was previously
    implemented as a mixin in the ISOPlot class.

    Attributes
    ----------
    plot : Any
        The parent plot instance

    """

    def __init__(self, plot: Any) -> None:
        """
        Initialize a StyleManager.

        Parameters
        ----------
        plot : Any
            The parent plot instance

        """
        self.plot = plot

    def apply_styling(
        self,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **style_params: Any,
    ) -> Any:
        """
        Apply styling to the plot.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **style_params : Any
            Additional styling parameters

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        # Update style parameters
        self.plot.main_context.update_params("style", **style_params)

        # If no subplots, apply to main context
        if not self.plot.subplot_contexts:
            self._apply_styling_to_context(self.plot.main_context)
            return self.plot

        # Apply to specified subplots
        target_contexts = self._resolve_target_contexts(on_axis)
        for context in target_contexts:
            self._apply_styling_to_context(context)

        return self.plot

    def _apply_styling_to_context(self, context: PlotContext) -> None:
        """
        Apply styling to a specific context.

        Parameters
        ----------
        context : PlotContext
            The context to apply styling to

        """
        if context.ax is None:
            return

        # Get style parameters
        style_params = context.get_params("style")

        # Apply styling to the axes
        ax = context.ax

        # Set limits
        if hasattr(style_params, "xlim"):
            ax.set_xlim(style_params.xlim)
        if hasattr(style_params, "ylim"):
            ax.set_ylim(style_params.ylim)

        # Set labels
        if hasattr(style_params, "xlabel") and style_params.xlabel is not False:
            xlabel = style_params.xlabel or context.x
            ax.set_xlabel(xlabel, fontdict=style_params.prim_ax_fontdict)

        if hasattr(style_params, "ylabel") and style_params.ylabel is not False:
            ylabel = style_params.ylabel or context.y
            ax.set_ylabel(ylabel, fontdict=style_params.prim_ax_fontdict)

        # Set title
        if context.title:
            ax.set_title(context.title, fontsize=style_params.title_fontsize)

        # Add primary axes lines
        if hasattr(style_params, "primary_lines") and style_params.primary_lines:
            self._add_primary_lines(ax, style_params)

        # Add diagonal lines
        if hasattr(style_params, "diagonal_lines") and style_params.diagonal_lines:
            self._add_diagonal_lines(ax, style_params)

        # Add legend if needed
        if hasattr(style_params, "legend_loc") and style_params.legend_loc:
            ax.legend(loc=style_params.legend_loc)

    def _add_primary_lines(self, ax: Axes, style_params: Any) -> None:
        """
        Add primary axes lines to the plot.

        Parameters
        ----------
        ax : Axes
            The axes to add lines to
        style_params : Any
            The style parameters

        """
        # Add horizontal and vertical lines at 0
        ax.axhline(
            y=0,
            color="black",
            linestyle="-",
            linewidth=style_params.linewidth,
            zorder=style_params.prim_lines_zorder,
        )
        ax.axvline(
            x=0,
            color="black",
            linestyle="-",
            linewidth=style_params.linewidth,
            zorder=style_params.prim_lines_zorder,
        )

    def _add_diagonal_lines(self, ax: Axes, style_params: Any) -> None:
        """
        Add diagonal lines to the plot.

        Parameters
        ----------
        ax : Axes
            The axes to add lines to
        style_params : Any
            The style parameters

        """
        # Add diagonal lines
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Diagonal line from bottom-left to top-right
        ax.plot(
            xlim,
            ylim,
            color="black",
            linestyle="--",
            linewidth=style_params.linewidth,
            zorder=style_params.diag_lines_zorder,
        )

        # Diagonal line from bottom-right to top-left
        ax.plot(
            xlim,
            ylim[::-1],
            color="black",
            linestyle="--",
            linewidth=style_params.linewidth,
            zorder=style_params.diag_lines_zorder,
        )

    def _resolve_target_contexts(
        self, on_axis: int | tuple[int, int] | list[int] | None
    ) -> list[PlotContext]:
        """
        Resolve which subplot contexts to target based on axis specification.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None
            The axis specification:
            - None: All subplot contexts
            - int: Single subplot at flattened index
            - tuple[int, int]: Subplot at (row, col)
            - list[int]: Multiple subplots at specified indices

        Returns
        -------
        list[PlotContext]
            List of target subplot contexts

        """
        # If no specific axis, target all subplot contexts
        if on_axis is None:
            return self.plot.subplot_contexts

        # Convert axis specification to list of indices
        indices = self._resolve_axis_indices(on_axis)

        # Get the contexts for each valid index
        target_contexts = []
        for idx in indices:
            if 0 <= idx < len(self.plot.subplot_contexts):
                target_contexts.append(self.plot.subplot_contexts[idx])
            else:
                msg = f"Subplot index {idx} out of range"
                raise IndexError(msg)

        return target_contexts

    def _resolve_axis_indices(
        self, on_axis: int | tuple[int, int] | list[int]
    ) -> list[int]:
        """
        Convert axis specification to list of indices.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int]
            The axis specification to resolve

        Returns
        -------
        list[int]
            List of flattened indices

        Raises
        ------
        ValueError
            If an invalid axis specification is provided

        """
        if isinstance(on_axis, int):
            return [on_axis]
        if isinstance(on_axis, tuple) and len(on_axis) == 2:
            # Convert (row, col) to flattened index
            row, col = on_axis
            return [row * self.plot.subplots_params.ncols + col]
        if isinstance(on_axis, list):
            return on_axis
        msg = f"Invalid axis specification: {on_axis}"
        raise ValueError(msg)


class SubplotManager:
    """
    Manages the creation and configuration of subplots.

    This class encapsulates the subplot-related functionality that was previously
    implemented directly in the ISOPlot class.

    Attributes
    ----------
    plot : Any
        The parent plot instance

    """

    def __init__(self, plot: Any) -> None:
        """
        Initialize a SubplotManager.

        Parameters
        ----------
        plot : Any
            The parent plot instance

        """
        self.plot = plot

    def create_subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = True,
        sharey: bool | Literal["none", "all", "row", "col"] = True,
        subplot_by: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a grid of subplots.

        Parameters
        ----------
        nrows : int, optional
            Number of rows, by default 1
        ncols : int, optional
            Number of columns, by default 1
        figsize : tuple[float, float] | None, optional
            Figure size, by default None
        sharex : bool | Literal["none", "all", "row", "col"], optional
            Whether to share x-axis, by default True
        sharey : bool | Literal["none", "all", "row", "col"], optional
            Whether to share y-axis, by default True
        subplot_by : str | None, optional
            Column to create subplots by, by default None
        **kwargs : Any
            Additional parameters for subplots

        Returns
        -------
        Any
            The parent plot instance for chaining

        """
        # Update subplot parameters
        self.plot.subplots_params.update(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize or (5 * ncols, 5 * nrows),
            sharex=sharex,
            sharey=sharey,
            subplot_by=subplot_by,
            **kwargs,
        )

        # Create figure and axes
        self.plot.figure, self.plot.axes = plt.subplots(
            **self.plot.subplots_params.as_plt_subplots_args()
        )

        # Create subplot contexts
        self._create_subplot_contexts()

        return self.plot

    def _create_subplot_contexts(self) -> None:
        """
        Create subplot contexts based on the current configuration.

        This method creates a PlotContext for each subplot, either with
        the same data or with data split by a grouping variable.
        """
        # Clear existing subplot contexts
        self.plot.subplot_contexts = []

        # Get subplot parameters
        params = self.plot.subplots_params

        # If subplot_by is specified, create subplots by group
        if params.subplot_by and self.plot.main_context.data is not None:
            self._create_subplots_by_group()
            return

        # Otherwise, create a grid of subplots with the same data
        axes = self.plot.axes
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        # Create a context for each axis
        for i in range(params.nrows):
            for j in range(params.ncols):
                # Get the axis for this subplot
                ax = (
                    axes[i, j]
                    if params.nrows > 1 and params.ncols > 1
                    else axes[i]
                    if params.nrows > 1
                    else axes[j]
                    if params.ncols > 1
                    else axes
                )

                # Create a title for this subplot
                title = (
                    f"Subplot {i * params.ncols + j + 1}"
                    if self.plot.main_context.title is None
                    else f"{self.plot.main_context.title} {i * params.ncols + j + 1}"
                )

                # Create a child context for this subplot
                context = self.plot.main_context.create_child(
                    ax=ax,
                    title=title,
                )

                # Add to subplot contexts
                self.plot.subplot_contexts.append(context)

    def _create_subplots_by_group(self) -> None:
        """
        Create subplots by grouping the data.

        This method creates a subplot for each unique value in the
        subplot_by column of the data.
        """
        # Get subplot parameters
        params = self.plot.subplots_params
        subplot_by = params.subplot_by

        if subplot_by is None or self.plot.main_context.data is None:
            return

        # Get unique values in the subplot_by column
        data = self.plot.main_context.data
        groups = data[subplot_by].unique()

        # Limit to the number of subplots if specified
        if params.n_subplots_by > 0:
            groups = groups[: params.n_subplots_by]

        # Check if we have enough subplots
        if len(groups) > params.n_subplots:
            msg = f"Not enough subplots for all groups: {len(groups)} groups, {params.n_subplots} subplots"
            raise ValueError(msg)

        # Get axes array
        axes = self.plot.axes
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        # Create a context for each group
        for i, group in enumerate(groups):
            # Get the row and column for this subplot
            row = i // params.ncols
            col = i % params.ncols

            # Get the axis for this subplot
            ax = (
                axes[row, col]
                if params.nrows > 1 and params.ncols > 1
                else axes[row]
                if params.nrows > 1
                else axes[col]
                if params.ncols > 1
                else axes
            )

            # Filter data for this group
            group_data = data[data[subplot_by] == group]

            # Create a title for this subplot
            title = (
                f"{group}"
                if self.plot.main_context.title is None
                else f"{self.plot.main_context.title}: {group}"
            )

            # Create a child context for this subplot
            context = self.plot.main_context.create_child(
                data=group_data,
                ax=ax,
                title=title,
            )

            # Add to subplot contexts
            self.plot.subplot_contexts.append(context)
