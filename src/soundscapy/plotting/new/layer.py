"""
Layer classes for visualization.

This module provides the base Layer class and specialized layer implementations
for different visualization techniques. Layers know how to render themselves onto
a PlotContext's axes using parameters provided by the context.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns

from soundscapy.plotting.new.constants import RECOMMENDED_MIN_SAMPLES
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from soundscapy.plotting.new.parameter_models import (
        BaseParams,
        DensityParams,
        ScatterParams,
        SimpleDensityParams,
        SPISimpleDensityParams,
    )
    from soundscapy.plotting.new.protocols import PlotContext
    from soundscapy.spi.msn import (
        CentredParams,
        DirectParams,
    )

logger = get_logger()


class Layer:
    """
    Base class for all visualization layers.

    A Layer encapsulates a specific visualization technique. Layers know how to
    render themselves onto a PlotContext's axes using parameters provided by the context.

    Attributes
    ----------
    custom_data : pd.DataFrame | None
        Optional custom data for this specific layer, overriding context data
    param_overrides : dict[str, Any]
        Parameter overrides for this layer

    """

    # Class registry for layer types
    _layer_registry: ClassVar[dict[str, type[Layer]]] = {}

    # Parameter type this layer uses (for getting params from context)
    param_type: ClassVar[str] = "base"

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        **param_overrides: Any,
    ) -> None:
        """
        Initialize a Layer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer, overriding context data
        **param_overrides : dict[str, Any]
            Parameter overrides for this layer

        """
        self.custom_data = custom_data
        self.param_overrides = param_overrides

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses in the registry."""
        super().__init_subclass__(**kwargs)
        # Skip registration for the base class
        if cls is not Layer:
            cls._layer_registry[cls.__name__.lower()] = cls

    def render(self, context: PlotContext) -> None:
        """
        Render this layer on the given context.

        Parameters
        ----------
        context : PlotContext
            The context containing data and axes for rendering

        Raises
        ------
        ValueError
            If the context has no associated axes or data

        """
        if context.ax is None:
            msg = "Cannot render layer: context has no associated axes"
            raise ValueError(msg)

        # Use custom data if provided, otherwise context data
        data = self.custom_data if self.custom_data is not None else context.data

        if data is None:
            msg = "No data available for rendering layer"
            raise ValueError(msg)

        # Get parameters from context and apply overrides
        params = self._get_params_from_context(context)

        # Remove palette if no hue in this layer
        if params.hue is None:
            params.palette = None

        # Render the layer
        self._render_implementation(data, context, context.ax, params)

    def _get_params_from_context(self, context: PlotContext) -> BaseParams:
        """
        Get parameters from context and apply overrides.

        Parameters
        ----------
        context : PlotContext
            The context to get parameters from

        Returns
        -------
        BaseParams
            The parameters for this layer

        """
        # Get parameters from context based on layer type
        params = context.get_params_for_layer(type(self))

        # Apply overrides
        if self.param_overrides:
            params.update(**self.param_overrides)

        return cast("BaseParams", params)

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Implement actual rendering (to be overridden by subclasses).

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        Raises
        ------
        NotImplementedError
            If not implemented by subclass

        """
        msg = "Subclasses must implement _render_implementation"
        raise NotImplementedError(msg)

    @classmethod
    def create(
        cls, context: PlotContext, layer_type: str | None = None, **kwargs: Any
    ) -> Layer:
        """
        Factory method to create a layer of the specified type.

        Parameters
        ----------
        context : PlotContext
            The context to associate with the layer
        layer_type : str | None
            The type of layer to create (e.g., 'scatter', 'density')
            If None, uses the class name
        **kwargs : Any
            Additional parameters for the layer

        Returns
        -------
        Layer
            The created layer instance

        Raises
        ------
        ValueError
            If the layer type is unknown

        """  # noqa: D401
        if layer_type is None:
            # Use the current class if no type specified
            return cls(context=context, **kwargs)

        # Get the layer class from the registry
        layer_type = layer_type.lower()
        if layer_type not in cls._layer_registry:
            msg = f"Unknown layer type: {layer_type}"
            raise ValueError(msg)

        # Create and return the layer
        layer_class = cls._layer_registry[layer_type]
        return layer_class(**kwargs)


class ScatterLayer(Layer):
    """Layer for rendering scatter plots."""

    param_type = "scatter"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render a scatter plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        # Cast params to the correct type
        scatter_params = cast("ScatterParams", params)

        # Create a copy of the parameters with data
        kwargs = scatter_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Render the scatter plot
        sns.scatterplot(ax=ax, **kwargs)


class DensityLayer(Layer):
    """Layer for rendering density plots."""

    param_type = "density"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render a density plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        # Check if we have enough data for a density plot
        if len(data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for "
                f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

        # Cast params to the correct type
        density_params = cast("DensityParams", params)

        # Create a copy of the parameters with data
        kwargs = density_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Render the density plot
        sns.kdeplot(ax=ax, **kwargs)


class SimpleDensityLayer(DensityLayer):
    """Layer for rendering simple density plots (filled contours)."""

    param_type = "simple_density"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render a simple density plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        # Check if we have enough data for a density plot
        if len(data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for "
                f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
                UserWarning,
                stacklevel=2,
            )

        # Cast params to the correct type
        simple_density_params = cast("SimpleDensityParams", params)

        # Create a copy of the parameters with data
        kwargs = simple_density_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Set specific parameters for simple density
        kwargs["levels"] = simple_density_params.levels
        kwargs["thresh"] = getattr(simple_density_params, "thresh", 0.05)

        # Render the simple density plot
        sns.kdeplot(ax=ax, **kwargs)


class SPILayer(Layer):
    """Base layer for rendering SPI plots."""

    param_type = "spi"

    def __init__(
        self,
        spi_target_data: pd.DataFrame | np.ndarray | None = None,
        *,
        msn_params: DirectParams | CentredParams | None = None,
        n: int = 10000,
        custom_data: pd.DataFrame | None = None,
        **params: Any,
    ) -> None:
        """
        Initialize an SPILayer.

        Parameters
        ----------
        spi_target_data : pd.DataFrame | np.ndarray | None, optional
            Pre-sampled data for SPI target distribution.
            When None, msn_params must be provided.
        msn_params : DirectParams | CentredParams | None, optional
            Parameters to generate SPI data if no spi_target_data is provided
        n : int, optional
            Number of samples to generate if using msn_params, by default 10000
        custom_data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        **params : Any
            Additional parameters for the layer

        Notes
        -----
        Either spi_target_data or msn_params must be provided, but not both.
        The test data for SPI calculations will be retrieved from the plot context.

        """
        # If custom_data is provided but spi_target_data is not, use custom_data as spi_target_data
        if custom_data is not None and spi_target_data is None:
            logger.warning(
                "`spi_target_data` not found, but `custom_data` was found. "
                "Using `custom_data` as the SPI target data. "
                "\nNote: Passing the SPI data to `spi_target_data` is preferred."
            )
            spi_target_data = custom_data
            custom_data = None

        # Validate inputs and get SPI parameters
        spi_target_data, self.spi_params = self._validate_spi_inputs(
            spi_target_data, msn_params
        )

        # Generate the SPI target data
        self.spi_data: pd.DataFrame = self._generate_spi_data(
            spi_target_data, self.spi_params, n
        )

        # Add n to params
        params["n"] = n

        # Initialize the base layer with the SPI data
        super().__init__(custom_data=self.spi_data, **params)

    def render(self, context: PlotContext) -> None:
        """
        Render this layer on the given context.

        Parameters
        ----------
        context : PlotContext
            The context containing data and axes for rendering

        """
        if context.ax is None:
            msg = "Cannot render layer: context has no associated axes"
            raise ValueError(msg)

        # Get the SPI target data
        target_data = self.spi_data

        # Process the SPI data to match the context
        target_data = self._process_spi_data(target_data, context)

        if target_data is None:
            msg = "No data available for rendering SPI layer"
            raise ValueError(msg)

        # Get parameters from context
        params = self._get_params_from_context(context)

        # Render the layer
        self._render_implementation(target_data, context, context.ax, params)

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render an SPI plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        target_data = data[[context.x, context.y]]

        # Get test data from context
        test_data = context.data
        if test_data is None:
            warnings.warn(
                "Cannot find data to test SPI against. Skipping this plot.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Calculate SPI score
        spi_score = self._calc_context_spi_score(target_data, test_data)

        # Show the score
        self.show_score(
            spi_score,
            show_score=params.show_score
            if hasattr(params, "show_score")
            else "under title",
            context=context,
            ax=ax,
            axis_text_kwargs=params.axis_text_kw
            if hasattr(params, "axis_text_kw")
            else {},
        )

    def show_score(
        self,
        spi_score: int | None,
        show_score: Literal["on axis", "under title"],
        context: PlotContext,
        ax: Axes,
        axis_text_kwargs: dict[str, Any],
    ) -> None:
        """
        Show the SPI score on the plot.

        Parameters
        ----------
        spi_score : int | None
            The SPI score to show
        show_score : Literal["on axis", "under title"]
            Where to show the score
        context : PlotContext
            The context containing data and axes for rendering
        ax : Axes
            The axes to render the score on
        axis_text_kwargs : dict[str, Any]
            Additional arguments for the axis text

        """
        if spi_score is not None:
            if show_score == "on axis":
                self._add_score_as_text(
                    ax=ax,
                    spi_score=spi_score,
                    **axis_text_kwargs,
                )
            elif show_score == "under title":
                self._add_score_under_title(
                    context=context,
                    ax=ax,
                    spi_score=spi_score,
                )

    @staticmethod
    def _add_score_as_text(ax: Axes, spi_score: int, **text_kwargs: Any) -> None:
        """
        Add the SPI score as text on the axis.

        Parameters
        ----------
        ax : Axes
            The axes to add the text to
        spi_score : int
            The SPI score to show
        **text_kwargs : Any
            Additional arguments for the text

        """
        from soundscapy.plotting.new.constants import DEFAULT_SPI_TEXT_KWARGS

        text_kwargs_copy = DEFAULT_SPI_TEXT_KWARGS.copy()
        text_kwargs_copy.update(**text_kwargs)
        text_kwargs_copy["s"] = f"SPI: {spi_score}"

        ax.text(transform=ax.transAxes, **text_kwargs_copy)

    @staticmethod
    def _add_score_under_title(context: PlotContext, ax: Axes, spi_score: int) -> None:
        """
        Add the SPI score under the title.

        Parameters
        ----------
        context : PlotContext
            The context containing data and axes for rendering
        ax : Axes
            The axes to add the text to
        spi_score : int
            The SPI score to show

        """
        if context.title is not None:
            new_title = f"{context.title}\nSPI: {spi_score}"
        else:
            new_title = f"SPI: {spi_score}"

        ax.set_title(new_title)

    @staticmethod
    def _validate_spi_inputs(
        spi_data: pd.DataFrame | np.ndarray | None,
        spi_params: DirectParams | CentredParams | None,
    ) -> tuple[pd.DataFrame | np.ndarray | None, DirectParams | CentredParams | None]:
        """
        Validate the right combination of inputs for the SPI plot.

        Parameters
        ----------
        spi_data : pd.DataFrame | np.ndarray | None
            Data to use for SPI plotting
        spi_params : DirectParams | CentredParams | None
            Parameters to generate SPI data

        Returns
        -------
        tuple[pd.DataFrame | np.ndarray | None, DirectParams | CentredParams | None]
            Validated data and parameters

        """
        # Input validation
        if spi_data is None and spi_params is None:
            msg = (
                "No data provided for SPI plot. "
                "Please provide either spi_data or msn_params."
            )
            raise ValueError(msg)

        if spi_data is not None and spi_params is not None:
            msg = (
                "Please provide either spi_data or msn_params, not both. "
                "Got: \n"
                f"\n`spi_data`: {type(spi_data)}\n`spi_params`: {type(spi_params)}"
            )
            raise ValueError(msg)

        if spi_data is not None and not isinstance(spi_data, pd.DataFrame | np.ndarray):
            msg = "Invalid data type for SPI plot. Expected DataFrame or ndarray."
            raise TypeError(msg)

        if spi_params is not None:
            # Check if the import is available
            try:
                from soundscapy.spi.msn import CentredParams, DirectParams

                if not isinstance(spi_params, (DirectParams, CentredParams)):
                    msg = (
                        "Invalid parameters for SPI plot. "
                        "Expected DirectParams or CentredParams."
                    )
                    raise TypeError(msg)
            except ImportError:
                msg = (
                    "Could not import DirectParams or CentredParams from soundscapy.spi.msn. "
                    "Please ensure the module is available."
                )
                raise ImportError(msg)

        return spi_data, spi_params

    @staticmethod
    def _generate_spi_data(
        spi_data: pd.DataFrame | np.ndarray | None,
        spi_params: DirectParams | CentredParams | None,
        n: int,
    ) -> pd.DataFrame:
        """
        Validate and prepare SPI data from either direct data or parameters.

        Parameters
        ----------
        spi_data : pd.DataFrame | np.ndarray | None
            Data to use for SPI plotting
        spi_params : DirectParams | CentredParams | None
            Parameters to generate SPI data
        n : int
            Number of samples to generate if using msn_params

        Returns
        -------
        pd.DataFrame
            Prepared data for SPI plotting

        """
        # Generate data from parameters if provided
        if spi_params is not None:
            try:
                from soundscapy.spi.msn import MultiSkewNorm

                spi_msn = MultiSkewNorm.from_params(spi_params)
                sample_data = spi_msn.sample(n=n, return_sample=True)
                spi_data = pd.DataFrame(
                    sample_data,
                    columns=["x", "y"],
                )
            except ImportError:
                msg = (
                    "Could not import MultiSkewNorm from soundscapy.spi.msn. "
                    "Please ensure the module is available."
                )
                raise ImportError(msg)

        if spi_data is not None:
            # Process provided data
            if isinstance(spi_data, np.ndarray):
                if len(spi_data.shape) != 2 or spi_data.shape[1] != 2:  # noqa: PLR2004
                    msg = (
                        "Invalid shape for SPI data. "
                        "Expected a 2D array with 2 columns."
                    )
                    raise ValueError(msg)
                spi_data = pd.DataFrame(spi_data, columns=["x", "y"])
            return spi_data

        msg = "Please provide either spi_data or msn_params."
        raise ValueError(msg)

    def _process_spi_data(
        self, spi_data: pd.DataFrame | np.ndarray, context: PlotContext
    ) -> pd.DataFrame:
        """
        Process SPI data into standard format.

        Parameters
        ----------
        spi_data : pd.DataFrame | np.ndarray
            Data to process
        context : PlotContext
            The context containing state for rendering

        Returns
        -------
        pd.DataFrame
            Processed data in standard format

        """
        params = self._get_params_from_context(context)
        xcol = getattr(params, "x", context.x)
        ycol = getattr(params, "y", context.y)

        # DataFrame handling
        if isinstance(spi_data, pd.DataFrame):
            if xcol not in spi_data.columns or ycol not in spi_data.columns:
                spi_data = spi_data.rename(columns={"x": xcol, "y": ycol})
            return spi_data

        # Numpy array handling
        if isinstance(spi_data, np.ndarray):
            if len(spi_data.shape) != 2 or spi_data.shape[1] != 2:  # noqa: PLR2004
                msg = "Invalid shape for SPI data. Expected a 2D array with 2 columns."
                raise ValueError(msg)
            return pd.DataFrame(spi_data, columns=[xcol, ycol])

        msg = "Invalid SPI data type. Expected DataFrame or numpy array."
        raise TypeError(msg)

    @staticmethod
    def _calc_context_spi_score(
        target_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> int | None:
        """
        Calculate the SPI score between target and test data.

        Parameters
        ----------
        target_data : pd.DataFrame
            The target data
        test_data : pd.DataFrame
            The test data

        Returns
        -------
        int | None
            The SPI score, or None if calculation failed

        """
        try:
            from soundscapy.spi import spi_score

            return spi_score(target=target_data, test=test_data)
        except ImportError:
            warnings.warn(
                "Could not import spi_score from soundscapy.spi. "
                "SPI score calculation will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            return None


class SPISimpleLayer(SPILayer, SimpleDensityLayer):
    """Layer for rendering SPI simple density plots."""

    param_type = "spi_simple_density"

    # Note: SPIDensityLayer and SPIScatterLayer could be implemented similarly
    # by inheriting from SPILayer and DensityLayer or ScatterLayer respectively.
    # For example:
    #
    # class SPIDensityLayer(SPILayer, DensityLayer):
    #     """Layer for rendering SPI density plots."""
    #     param_type = "spi_density"
    #
    # class SPIScatterLayer(SPILayer, ScatterLayer):
    #     """Layer for rendering SPI scatter plots."""
    #     param_type = "spi_scatter"

    def _render_implementation(
        self,
        data: pd.DataFrame,
        context: PlotContext,
        ax: Axes,
        params: BaseParams,
    ) -> None:
        """
        Render an SPI simple density plot.

        Parameters
        ----------
        data : pd.DataFrame
            The data to render
        context : PlotContext
            The context containing state for rendering
        ax : Axes
            The matplotlib axes to render on
        params : BaseParams
            The parameters for this layer

        """
        # Cast params to the correct type
        spi_params = cast("SPISimpleDensityParams", params)

        # Create a copy of the parameters with data
        kwargs = spi_params.as_seaborn_kwargs()
        kwargs["data"] = data

        # Ensure x and y are set correctly
        kwargs["x"] = context.x
        kwargs["y"] = context.y

        # Set specific parameters for SPI simple density
        kwargs["color"] = spi_params.color
        kwargs["label"] = spi_params.label

        # Render the SPI simple density plot
        sns.kdeplot(ax=ax, **kwargs)

        # Calculate SPI score
        target_data = data[[context.x, context.y]]
        test_data = context.data

        if test_data is not None:
            spi_score = self._calc_context_spi_score(target_data, test_data)

            # Show the score
            if hasattr(spi_params, "show_score") and spi_params.show_score:
                self.show_score(
                    spi_score,
                    show_score=spi_params.show_score,
                    context=context,
                    ax=ax,
                    axis_text_kwargs=spi_params.axis_text_kw or {},
                )
