"""
Manager classes for the plotting module.

This module provides manager classes that encapsulate specific functionality
for the ISOPlot class. These managers replace the mixin-based approach in the
original implementation, using composition instead of inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker
from matplotlib.artist import Artist

from soundscapy.plotting.new.layer import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPISimpleLayer,
)
from soundscapy.plotting.new.parameter_models import MplLegendLocType
from soundscapy.plotting.new.plot_context import PlotContext
from soundscapy.sspylogging import get_logger

try:
    # Python 3.13 made a @deprecated decorator available
    from warnings import deprecated  # type: ignore[attr-defined]

except ImportError:
    # Fall back to using specific module
    from deprecated import deprecated

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from soundscapy.plotting.new import ISOPlot, StyleParams

logger = get_logger()

# Type definitions
AxisSpec = int | tuple[int, int] | list[int]
LayerType = Literal["scatter", "density", "simple_density", "spi_simple"]
LayerSpec = LayerType | type[Layer] | Layer


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

    def add_layer(
        self,
        layer_spec: LayerSpec,
        data: pd.DataFrame | None = None,
        *,
        on_axis: AxisSpec | None = None,
        subplot_by: str | None = None,
        **params: Any,
    ) -> ISOPlot:
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
        subplot_by : str | None, optional
            Column to split data across existing subplots, by default None.
            If provided, the data will be split based on unique values in this column
            and rendered on the corresponding subplots.
            Note: This is different from the subplot_by parameter in create_subplots,
            which creates new subplots based on unique values.
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
            return self._render_layer(
                layer_spec, on_axis=on_axis, subplot_by=subplot_by, **params
            )

        # Get the layer class from either the class or
        layer_class = self._resolve_layer_class(layer_spec)

        # If subplot_by is provided, we need to handle it specially
        if subplot_by is not None:
            return self._render_layer_with_subplot_by(
                layer_class, data, subplot_by, on_axis, **params
            )

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

    def _render_layer_with_subplot_by(
        self,
        layer_class: type[Layer],
        data: pd.DataFrame | None,
        subplot_by: str,
        on_axis: AxisSpec | None = None,
        **params: Any,
    ) -> ISOPlot:
        """
        Render a layer with data split across subplots based on a grouping variable.

        Parameters
        ----------
        layer_class : type[Layer]
            The layer class to instantiate
        data : pd.DataFrame | None
            The data to split and render
        subplot_by : str
            Column to split data by
        on_axis : AxisSpec | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the layer

        Returns
        -------
        ISOPlot
            The parent plot instance for chaining

        Raises
        ------
        ValueError
            If the subplot_by column doesn't exist in the data
        RuntimeError
            If no subplots have been created yet

        """
        # If no subplots created yet, raise an error
        if self.plot.figure is None or self.plot.axes is None:
            msg = "Cannot add layer to main context before creating subplots."
            raise RuntimeError(msg)

        # If no data provided, use the main context data
        if data is None:
            data = self.plot.main_context.data

        # If still no data, raise an error
        if data is None:
            msg = "No data provided for layer and no data in main context."
            raise ValueError(msg)

        # Validate that the subplot_by column exists in the data
        if subplot_by not in data.columns:
            msg = (
                f"Invalid subplot_by column '{subplot_by}'. "
                f"Available columns are: {data.columns.tolist()}"
            )
            raise ValueError(msg)

        # Get the unique values in the subplot_by column
        unique_values = data[subplot_by].unique()

        # Get the target contexts
        target_contexts = PlotContext.get_contexts_by_spec(self.plot, on_axis)

        # Check if we have enough subplots
        if len(unique_values) > len(target_contexts):
            msg = (
                f"Not enough subplots for all unique values in '{subplot_by}'. "
                f"Got {len(unique_values)} unique values and {len(target_contexts)} subplots."
            )
            raise ValueError(msg)

        # For each unique value, create a layer with the filtered data and render it
        for i, value in enumerate(unique_values):
            # Filter the data for this value
            filtered_data = data[data[subplot_by] == value]

            # Create the layer instance
            is_spi = "spi" in layer_class.__name__.lower()
            if is_spi:
                layer = layer_class(spi_target_data=filtered_data, **params)
            else:
                layer = layer_class(custom_data=filtered_data, **params)

            # Add the layer to the context and render it
            context = target_contexts[i]
            context.layers.append(layer)
            layer.render(context)

        return self.plot

    def _render_layer(
        self,
        layer: Layer,
        *,
        on_axis: AxisSpec | None = None,
        subplot_by: str | None = None,
        **params: Any,
    ) -> ISOPlot:
        """
        Render a layer on the appropriate axes.

        Parameters
        ----------
        layer : Layer
            The layer to render
        on_axis : AxisSpec | None, optional
            Target specific axis/axes, by default None
        subplot_by : str | None, optional
            Column to split data by, by default None
        **params : Any
            Additional parameters for the layer

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

        # If subplot_by is provided, we need to handle it specially
        if subplot_by is not None:
            # We can't handle subplot_by for an already instantiated layer
            msg = (
                "Cannot use subplot_by with an already instantiated layer. "
                "Please provide the layer class and data instead."
            )
            raise ValueError(msg)

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
    def add_scatter(
        self,
        data=None,  # noqa: ANN001
        *,
        on_axis=None,  # noqa: ANN001
        subplot_by=None,  # noqa: ANN001
        **params,
    ) -> ISOPlot:
        """Legacy method that forwards to add_layer(layer_type="scatter", ...)."""
        return self.add_layer(
            "scatter", data=data, on_axis=on_axis, subplot_by=subplot_by, **params
        )

    @deprecated()
    def add_density(
        self,
        data=None,  # noqa: ANN001
        *,
        on_axis=None,  # noqa: ANN001
        subplot_by=None,  # noqa: ANN001
        **params,
    ) -> ISOPlot:
        """Legacy method that forwards to add_layer(layer_type="density", ...)."""
        return self.add_layer(
            "density", data=data, on_axis=on_axis, subplot_by=subplot_by, **params
        )

    @deprecated()
    def add_simple_density(
        self,
        data=None,  # noqa: ANN001
        *,
        on_axis=None,  # noqa: ANN001
        subplot_by=None,  # noqa: ANN001
        **params,
    ) -> ISOPlot:
        """Legacy method that forwards to add_layer(layer_type="simple_density",...)."""
        return self.add_layer(
            "simple_density",
            data=data,
            on_axis=on_axis,
            subplot_by=subplot_by,
            **params,
        )

    @deprecated()
    def add_spi_simple(
        self,
        data=None,  # noqa: ANN001
        *,
        msn_params=None,  # noqa: ANN001
        on_axis=None,  # noqa: ANN001
        subplot_by=None,  # noqa: ANN001
        **params,
    ) -> ISOPlot:
        """Legacy method that forwards to add_layer(layer_type="spi_simple", ...)."""
        return self.add_layer(
            "spi_simple",
            data=data,
            on_axis=on_axis,
            subplot_by=subplot_by,
            msn_params=msn_params,
            **params,
        )


class StyleManager:
    """
    Manages the styling of plots.

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
        self.param_overrides = {}

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
        self.param_overrides = style_params
        self._set_sns_style()
        # TODO: Need to set suptitle at some point

        # If no subplots, apply to main context
        if not self.plot.subplot_contexts:
            self.plot.main_context.update_params("style", **style_params)
            self._apply_styling_to_context(self.plot.main_context)
            return self.plot

        # Apply to specified subplots
        target_contexts = PlotContext.get_contexts_by_spec(self.plot, on_axis)
        for context in target_contexts:
            context.update_params("style", **style_params)
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
        style_params = cast("StyleParams", context.get_params("style"))
        # Apply style_mgr to the axes
        ax = context.ax

        self._circumplex_grid(ax, style_params)

        self._set_axis_title(ax, context, style_params)
        # Alternative?
        # ax.set_title(
        #   context.title, fontsize=style_params.title_fontsize)

        self._primary_labels(ax, context, style_params)
        # Alternative?
        # ax.set_xlabel(xlabel, fontdict=style_params.prim_ax_fontdict)  # noqa: ERA001
        # ax.set_ylabel(ylabel, fontdict=style_params.prim_ax_fontdict)  # noqa: ERA001

        if style_params.get("primary_lines"):
            self._add_primary_lines(ax, style_params)
        if style_params.get("diagonal_lines"):
            self._add_diagonal_lines(ax, style_params)
            self._add_diagonal_labels(ax, style_params)
        if style_params.get("legend_loc") is not False:
            # ax.legend(loc=style_params.get("legend_loc"))  # noqa: ERA001
            self._move_legend(ax, style_params.get("legend_loc"))

    @staticmethod
    def _set_sns_style() -> None:
        """Set the overall style for the plot."""
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    @staticmethod
    def _circumplex_grid(axis: Axes, style_params: StyleParams) -> None:
        """Add the circumplex grid to the plot."""
        axis.set_xlim(style_params.get("xlim"))
        axis.set_ylim(style_params.get("ylim"))
        axis.set_aspect("equal")

        axis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        axis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

        axis.grid(visible=True, which="major", color="grey", alpha=0.5)
        axis.grid(
            visible=True,
            which="minor",
            color="grey",
            linestyle="dashed",
            linewidth=0.5,
            alpha=0.4,
            zorder=style_params.get("prim_lines_zorder"),
        )

    @staticmethod
    def _set_axis_title(
        ax: Axes, context: PlotContext, style_params: StyleParams
    ) -> None:
        if ax and context.title:
            ax.set_title(context.title, fontsize=style_params.title_fontsize)

    @staticmethod
    def _primary_labels(
        axis: Axes, context: PlotContext, style_params: StyleParams
    ) -> None:
        """Handle the default labels for the x and y axes."""
        xlabel = style_params.get("xlabel")
        ylabel = style_params.get("ylabel")

        xlabel = context.x if xlabel is None else xlabel
        ylabel = context.y if ylabel is None else ylabel
        fontdict = style_params.get("prim_ax_fontdict")

        # BUG: For some reason, this ruins the sharex and sharey
        #       functionality, but only when a layer is applied
        #       a specific subplot.
        axis.set_xlabel(
            xlabel, fontdict=fontdict
        ) if xlabel is not False else axis.xaxis.label.set_visible(False)

        axis.set_ylabel(
            ylabel, fontdict=fontdict
        ) if ylabel is not False else axis.yaxis.label.set_visible(False)

    @staticmethod
    def _add_primary_lines(axis: Axes, style_params: StyleParams) -> None:
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
        axis.axhline(
            y=0,
            color="grey",
            linestyle="dashed",
            alpha=1,
            lw=style_params.get("linewidth"),
            zorder=style_params.get("prim_lines_zorder"),
        )
        axis.axvline(
            x=0,
            color="grey",
            linestyle="dashed",
            alpha=1,
            lw=style_params.get("linewidth"),
            zorder=style_params.get("prim_lines_zorder"),
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

    @staticmethod
    def _move_legend(axis: Axes, legend_loc: MplLegendLocType) -> None:
        """Move the legend to the specified location."""
        old_legend = axis.get_legend()
        if old_legend is None:
            # logger.debug("_move_legend: No legend found for axis %s", i)
            return

        # Get handles and filter out None values
        handles = [
            h for h in old_legend.legend_handles if isinstance(h, Artist | tuple)
        ]
        # Skip if no valid handles remain
        if not handles:
            return

        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        # Ensure labels and handles match in length
        if len(handles) != len(labels):
            labels = labels[: len(handles)]

        axis.legend(
            handles,
            labels,
            loc=legend_loc,
            title=title,
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
        subplot_titles: list[str] | None = None,
        *,
        auto_allocate_axes: bool = False,
        adjust_figsize: bool = True,
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
        subplot_titles : list[str] | None, optional
            Custom titles for subplots, by default None
        auto_allocate_axes : bool, optional
            Whether to automatically determine nrows/ncols based on data,
            by default False
        adjust_figsize : bool, optional
            Whether to adjust the figure size based on nrows/ncols, by default True
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
            subplot_titles=subplot_titles,
            auto_allocate_axes=auto_allocate_axes,
            adjust_figsize=adjust_figsize,
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
            if params.subplot_titles is not None:
                title = params.subplot_titles[i]
            else:
                title = None

            # Create a child context for this subplot
            context = self.plot.main_context.create_child(
                ax=ax,
                title=title,
            )

            # Add to subplot contexts
            self.plot.subplot_contexts.append(context)

    def _allocate_subplot_axes(self, n_groups: int) -> tuple[int, int]:
        """
        Allocate the subplot axes based on the number of data subsets.

        Parameters
        ----------
        n_groups : int
            Number of groups to allocate subplots for

        Returns
        -------
        tuple[int, int]
            Tuple of (nrows, ncols)

        """
        import warnings

        msg = (
            "This is an experimental feature. "
            "The number of rows and columns may not be optimal."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

        ncols = int(np.ceil(np.sqrt(n_groups)))
        nrows = int(np.ceil(n_groups / ncols))
        return nrows, ncols

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

        # Validate that the subplot_by column exists in the data
        data = self.plot.main_context.data
        if subplot_by not in data.columns:
            msg = (
                f"Invalid subplot_by column '{subplot_by}'. "
                f"Available columns are: {data.columns.tolist()}"
            )
            raise ValueError(msg)

        # Get unique values in the subplot_by column
        groups = data[subplot_by].unique()
        n_subplots_by = len(groups)

        # Warn if there are few unique values
        if n_subplots_by < 2:  # noqa: PLR2004
            import warnings

            warnings.warn(
                f"Only {n_subplots_by} unique values found in '{subplot_by}'. "
                "Subplots may not be meaningful.",
                UserWarning,
                stacklevel=2,
            )

        # Limit to the number of subplots if specified
        if params.n_subplots_by > 0:
            groups = groups[: params.n_subplots_by]
            n_subplots_by = len(groups)

        # Auto-allocate axes if requested
        if params.auto_allocate_axes:
            nrows, ncols = self._allocate_subplot_axes(n_subplots_by)
            # Update the subplot parameters
            self.plot.subplots_params.update(nrows=nrows, ncols=ncols)
            # Recreate the figure and axes with the new dimensions
            self.plot.figure, self.plot.axes = plt.subplots(
                **self.plot.subplots_params.as_plt_subplots_args()
            )

        # Check if we have enough subplots
        if n_subplots_by > params.n_subplots:
            msg = f"Not enough subplots for all groups: {n_subplots_by} groups, {params.n_subplots} subplots"
            raise ValueError(msg)

        # Handle subplot titles
        subplot_titles = params.subplot_titles
        if subplot_titles is None:
            # Create subplot titles based on the unique values
            subplot_titles = [str(value) for value in groups]
        elif len(subplot_titles) != n_subplots_by:
            # Validate that the number of titles matches the number of unique values
            msg = (
                "Number of subplot titles must match the number of unique values "
                f"for '{subplot_by}'. Got {len(subplot_titles)} titles and "
                f"{n_subplots_by} unique values."
            )
            raise ValueError(msg)
        else:
            # Warn if custom titles are provided with subplot_by
            import warnings

            warnings.warn(
                "Not recommended to provide separate subplot titles when using "
                "subplot_by. Consider using the default titles based on unique values. "
                "Manual subplot_titles may not be in the same order as the data.",
                UserWarning,
                stacklevel=2,
            )

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

            # Create a title for this subplot using the custom title or group value
            title = (
                subplot_titles[i]
                if self.plot.main_context.title is None
                else f"{self.plot.main_context.title}: {subplot_titles[i]}"
            )

            # Create a child context for this subplot
            context = self.plot.main_context.create_child(
                data=group_data,
                ax=ax,
                title=title,
            )

            # Add to subplot contexts
            self.plot.subplot_contexts.append(context)
