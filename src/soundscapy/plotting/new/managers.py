"""
Manager classes for the plotting module.

This module provides manager classes that encapsulate specific functionality
for the ISOPlot class. These managers replace the mixin-based approach in the
original implementation, using composition instead of inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from soundscapy.plotting.new.layer import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPISimpleLayer,
)
from soundscapy.plotting.new.plot_context import PlotContext
from soundscapy.sspylogging import get_logger

try:
    # Python 3.13 made a @deprecated decorator available
    from warnings import deprecated

except ImportError:
    # Fall back to using specific module
    from deprecated import deprecated

if TYPE_CHECKING:
    from soundscapy.plotting.new import ISOPlot

    _ISOPlotT = TypeVar("_ISOPlotT", bound="ISOPlot")


logger = get_logger()

# Type definitions
AxisSpec = int | tuple[int, int] | list[int]
LayerType = Literal["scatter", "density", "simple_density", "spi_simple"]
LayerClass = type[Layer]
LayerSpec = LayerType | LayerClass | Layer


class LayerManager:
    """
    Manages the creation and rendering of visualization layers.

    Attributes
    ----------
    plot : Any
        The parent plot instance

    """

    _LAYER_CLASSES: ClassVar[dict] = {
        "scatter": ScatterLayer,
        "density": DensityLayer,
        "simple_density": SimpleDensityLayer,
        "spi_simple": SPISimpleLayer,
    }

    def __init__(self, plot: Any) -> None:
        """
        Initialize a LayerManager.

        Parameters
        ----------
        plot : Any
            The parent plot instance

        """
        self.plot = plot

    # Pass a fully instantiated layer
    @overload
    def add_layer(
        self, layer: Layer, *, on_axis: AxisSpec | None = None
    ) -> _ISOPlotT: ...

    # Pass an uninstantiated layer class
    @overload
    def add_layer(
        self,
        layer_class: LayerClass,
        data: pd.DataFrame | None = None,
        *,
        on_axis: AxisSpec | None = None,
        **params: Any,
    ) -> _ISOPlotT: ...

    # Pass a string of the layer type name
    @overload
    def add_layer(
        self,
        layer_type: LayerType,
        data: pd.DataFrame | None = None,
        *,
        on_axis: AxisSpec | None = None,
        **params: Any,
    ) -> _ISOPlotT: ...

    def add_layer(
        self,
        layer_spec: LayerSpec,
        data: pd.DataFrame | None = None,
        *,
        on_axis: AxisSpec | None = None,
        **params: Any,
    ) -> _ISOPlotT:
        """
        Add a visualization layer, optionally targeting specific subplot(s).

        Parameters
        ----------
        layer_spec : LayerSpec
            Either:
            - A string layer type ("scatter", "density", "simple_density", "spi_simple")
            - A layer class (ScatterLayer, DensityLayer, etc.)
            - An already instantiated Layer object
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None.
            Ignored if layer_spec is a Layer instance.
        on_axis : AxisSpec, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the layer. Special parameters:
            - msn_params: Used only for "spi_simple" layer type
            Ignored if layer_spec is a Layer instance.


        Returns
        -------
        ISOPlot
            The parent plot instance for chaining

        Examples
        --------
        >>> import soundscapy as sspy
        >>> from soundscapy.plotting.new import ISOPlot
        >>> data = sspy.add_iso_coords(sspy.isd.load())
        >>> plot = ISOPlot(data).create_subplots(2, 2)

        # Using layer type string
        >>> custom_df = data.query("LocationID == 'CamdenTown'")
        >>> plot.layer_mgr.add_layer("scatter", x="col1", y="col2", alpha=0.5)
        >>> plot.layer_mgr.add_layer("density", data=custom_df, on_axis=1)

        # Using layer type class (works essentially the same as the string version)
        >>> plot.layer_mgr.add_layer(ScatterLayer, x="col1", y="col2", alpha=0.5)
        >>> plot.layer_mgr.add_layer(DensityLayer, data=custom_df, on_axis=1)

        # Using instantiated Layer object
        >>> my_layer = ScatterLayer(x="col1", y="col2", alpha=0.7)
        >>> plot.layer_mgr.add_layer(my_layer, on_axis=0)

        """
        # Handle the case when an instantiated Layer is provided
        if isinstance(layer_spec, Layer):
            # Use the provided layer directly
            return self._render_layer(layer_spec, on_axis=on_axis, **params)

        # Get the layer class from either the class or
        layer_class = self._resolve_layer_class(layer_spec)

        # Create the layer instance
        is_spi = "spi" in layer_class.__name__.lower()
        if is_spi:
            layer = layer_class(spi_target_data=data, **params)
        else:
            layer = layer_class(custom_data=data, **params)

        # Render the layer and return the plot
        return self._render_layer(layer, on_axis=on_axis)

    def _resolve_layer_class(self, layer_spec: str | type[Layer]) -> type[Layer]:
        """
        Resolve a layer specification to a Layer class.

        Parameters
        ----------
        layer_spec : str | type[Layer]
            Either a string layer type ("scatter", "density", "simple_density", "spi_simple")
                or a Layer class (ScatterLayer, DensityLayer, etc.)

        Returns
        -------
        type[Layer]
            The resolved Layer class

        Raises
        ------
        ValueError
            If an unknown layer type string is provided
        TypeError
            If layer_spec is not a string or a Layer subclass

        """
        # Case 1: layer_spec is a Layer class
        if isinstance(layer_spec, type) and issubclass(layer_spec, Layer):
            return layer_spec

        # Case 2: layer_spec is a string
        if isinstance(layer_spec, str):
            layer_class = self._LAYER_CLASSES.get(layer_spec)
            if layer_class is None:
                msg = (
                    f"Unknown layer type: {layer_spec}. "
                    f"Available layer types: {list(self._LAYER_CLASSES.keys())}"
                )
                raise ValueError(msg)
            return layer_class

        # If we get here, layer_spec was an invalid type
        msg = (
            "Expected `layer_spec` to be either: "
            f"  - str (layer type name): {list(self._LAYER_CLASSES.keys())} "
            "  - Uninstantiated Layer class, e.g. ScatterLayer "
            "  - Already instantiated Layer object "
            f"Got: {type(layer_spec).__name__}"
        )
        raise TypeError(msg)

    def _render_layer(
        self, layer: Layer, *, on_axis: AxisSpec | None = None
    ) -> _ISOPlotT:
        """
        Render a layer on the appropriate axes.

        Parameters
        ----------
        layer : Layer
            The layer to render
        on_axis : AxisSpec | None, optional
            Target specific axis/axes, by default None

        Returns
        -------
        ISOPlot
            The parent plot instance for chaining

        """
        # TODO: This should maybe be moved to a different class's responsibility
        #       What about encapsulating this in .get_contexts_by_spec ?
        #       Then LayerManager doesn't need to know anything about the context,
        #       it just asks for a list of contexts to add the layer to.
        # If no subplots created yet, add to main context
        if self.plot.figure is None or self.plot.axes is None:
            msg = "Cannot add layer to main context before creating subplots."
            raise RuntimeError(msg)

        # Handle various axis targeting options
        # TODO: If feels like this should be done via
        #       self.plot.get_contexts_by_spec(on_axis), rather than a classmethod
        target_contexts = PlotContext.get_contexts_by_spec(self.plot, on_axis)

        # Add the layer to each target context and render it
        for context in target_contexts:
            context.layers.append(layer)
            layer.render(context)

        return self.plot

    @deprecated()
    def add_scatter(self, data=None, *, on_axis=None, **params) -> _ISOPlotT:  # noqa: ANN001
        """Legacy method that forwards to add_layer(layer_type="scatter", ...)."""
        return self.add_layer("scatter", data=data, on_axis=on_axis, **params)

    @deprecated()
    def add_density(self, data=None, *, on_axis=None, **params) -> _ISOPlotT:  # noqa: ANN001
        """Legacy method that forwards to add_layer(layer_type="density", ...)."""
        return self.add_layer("density", data=data, on_axis=on_axis, **params)

    @deprecated()
    def add_simple_density(self, data=None, *, on_axis=None, **params) -> _ISOPlotT:  # noqa: ANN001
        """Legacy method that forwards to add_layer(layer_type="simple_density",...)."""
        return self.add_layer("simple_density", data=data, on_axis=on_axis, **params)

    @deprecated()
    def add_spi_simple(
        self,
        data=None,  # noqa: ANN001
        *,
        msn_params=None,  # noqa: ANN001
        on_axis=None,  # noqa: ANN001
        **params,
    ) -> _ISOPlotT:
        """Legacy method that forwards to add_layer(layer_type="spi_simple", ...)."""
        return self.add_layer(
            "spi_simple", data=data, on_axis=on_axis, msn_params=msn_params, **params
        )


class StyleManager:
    """
    Manages the style_mgr of plots.

    This class encapsulates the style_mgr-related functionality that was previously
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
        Apply style_mgr to the plot.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **style_params : Any
            Additional style_mgr parameters

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
        target_contexts = PlotContext.get_contexts_by_spec(self.plot, on_axis)
        for context in target_contexts:
            self._apply_styling_to_context(context)

        return self.plot

    def _apply_styling_to_context(self, context: PlotContext) -> None:
        """
        Apply style_mgr to a specific context.

        Parameters
        ----------
        context : PlotContext
            The context to apply style_mgr to

        """
        if context.ax is None:
            return

        # Get style parameters
        style_params = context.get_params("style")

        # Apply style_mgr to the axes
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

        # Add diagonal lines and labels
        if hasattr(style_params, "diagonal_lines") and style_params.diagonal_lines:
            self._add_diagonal_lines(ax, style_params)
            self._add_diagonal_labels(ax, style_params)

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
            color="grey",
            linestyle="dashed",
            alpha=0.5,
            linewidth=style_params.linewidth,
            zorder=style_params.diag_lines_zorder,
        )

        # Diagonal line from bottom-right to top-left
        ax.plot(
            xlim,
            ylim[::-1],
            color="grey",
            linestyle="dashed",
            alpha=0.5,
            linewidth=style_params.linewidth,
            zorder=style_params.diag_lines_zorder,
        )

    def _add_diagonal_labels(self, ax: Axes, style_params: Any) -> None:
        """
        Add diagonal labels to the plot.

        Parameters
        ----------
        ax : Axes
            The axes to add labels to
        style_params : Any
            The style parameters

        """
        # Add diagonal labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Define font dictionary for diagonal labels
        diag_ax_font = {
            "fontstyle": "italic",
            "fontsize": "small",
            "fontweight": "bold",
            "color": "black",
            "alpha": 0.5,
        }

        # Add the four diagonal labels
        ax.text(
            xlim[1] / 2,
            ylim[1] / 2,
            "(vibrant)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=style_params.diag_labels_zorder,
        )
        ax.text(
            xlim[0] / 2,
            ylim[1] / 2,
            "(chaotic)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=style_params.diag_labels_zorder,
        )
        ax.text(
            xlim[0] / 2,
            ylim[0] / 2,
            "(monotonous)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=style_params.diag_labels_zorder,
        )
        ax.text(
            xlim[1] / 2,
            ylim[0] / 2,
            "(calm)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
            zorder=style_params.diag_labels_zorder,
        )


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
        the same custom_data or with custom_data split by a grouping variable.
        """
        # Clear existing subplot contexts
        self.plot.subplot_contexts = []

        # Get subplot parameters
        params = self.plot.subplots_params

        # If subplot_by is specified, create subplots by group
        if params.subplot_by and self.plot.main_context.data is not None:
            self._create_subplots_by_group()
            return

        # Otherwise, create a grid of subplots with the same custom_data
        axes = self.plot.axes
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        # Create a context for each axis
        for i, ax in enumerate(axes.flatten()):
            # Create a title for this subplot
            title = (
                f"Subplot {i + 1}"
                if self.plot.main_context.title is None
                else f"{self.plot.main_context.title} {i + 1}"
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
        Create subplots by grouping the custom_data.

        This method creates a subplot for each unique value in the
        subplot_by column of the custom_data.
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

            # Filter custom_data for this group
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
