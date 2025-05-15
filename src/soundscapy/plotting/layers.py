"""
Layer-based visualization components for plotting.

This module provides a system of layer classes that implement different visualization
techniques for ISO plots. Each layer encapsulates a specific visualization method
and knows how to render itself on a given context.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns

from soundscapy.plotting.defaults import RECOMMENDED_MIN_SAMPLES
from soundscapy.plotting.param_models import (
    DensityParams,
    ScatterParams,
    SeabornParams,
    SimpleDensityParams,
    SPISeabornParams,
    SPISimpleDensityParams,
)
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from soundscapy import CentredParams, DirectParams
    from soundscapy.plotting.plot_context import PlotContext

logger = get_logger()


class Layer:
    """
    Base class for all visualization layers.

    A Layer encapsulates a specific visualization technique and its associated
    parameters. Layers know how to render themselves onto a PlotContext's axes.

    Attributes
    ----------
    custom_data : pd.DataFrame | None
        Optional custom data for this specific layer, overriding context data
    params : ParamModel
        Parameter model instance for this layer

    """

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        param_model: type[SeabornParams] = SeabornParams,
        **params: Any,
    ) -> None:
        """
        Initialize a Layer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer, overriding context data
        param_model : type[ParamModel] | None
            The parameter model class to use, if None uses a generic ParamModel
        **params : dict
            Parameters for the layer

        """
        self.custom_data = custom_data
        # Create parameter model instance
        self.params = param_model(**params)

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

        # Use custom data if provided, otherwise context data
        data = self.custom_data if self.custom_data is not None else context.data

        if data is None:
            msg = "No data available for rendering layer"
            raise ValueError(msg)

        self._render_implementation(data, context, context.ax)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
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

        """
        msg = "Subclasses must implement _render_implementation"
        raise NotImplementedError(msg)


class ScatterLayer(Layer):
    """Layer for rendering scatter plots."""

    def __init__(self, custom_data: pd.DataFrame | None = None, **params: Any) -> None:
        """
        Initialize a ScatterLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        **params : dict
            Parameters for the scatter plot

        """
        super().__init__(custom_data=custom_data, param_model=ScatterParams, **params)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
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

        """
        # Get data-specific properties or fall back to context defaults
        x = self.params.get("x", context.x)
        y = self.params.get("y", context.y)

        # Filter out x, y, hue and data parameters to avoid duplicate kwargs
        plot_params = self.params.copy()

        # Apply palette only if hue is used
        plot_params.crosscheck_palette_hue()

        # Render scatter plot
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            **plot_params.as_seaborn_kwargs(drop=["x", "y", "data"]),
        )


class DensityLayer(Layer):
    """Layer for rendering kernel density plots."""

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        *,
        param_model: type[DensityParams] = DensityParams,
        include_outline: bool = False,
        **params: Any,
    ) -> None:
        """
        Initialize a DensityLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        include_outline : bool
            Whether to include an outline around the density plot
        **params : dict
            Parameters for the density plot

        """
        self.include_outline = include_outline
        self.params: DensityParams
        super().__init__(custom_data=custom_data, param_model=param_model, **params)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
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

        """
        # Check if there's enough data for a meaningful density plot
        self._valid_density_size(data)

        # Get data-specific properties or fall back to context defaults
        x = self.params.get("x", context.x)
        y = self.params.get("y", context.y)

        # Filter out x, y, hue and data parameters to avoid duplicate kwargs
        plot_params = self.params.copy()

        # Apply palette only if hue is used
        plot_params.crosscheck_palette_hue()

        # Render density plot
        sns.kdeplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            **plot_params.as_seaborn_kwargs(
                # BUG: This should have been handled by SPISeabornParams
                #  but it seems to be ignored in the SeabornParams,
                #  so filtering brute force here.
                drop=["x", "y", "data", "n", "show_score", "axis_text_kw"]
            ),
        )

        # If requested, add an outline around the density plot
        if self.include_outline:
            sns.kdeplot(
                data=data,
                x=x,
                y=y,
                ax=ax,
                **plot_params.to_outline().as_seaborn_kwargs(
                    # BUG: This should have been handled by SPISeabornParams
                    #  but it seems to be ignored in the SeabornParams,
                    #  so filtering brute force here.
                    drop=["x", "y", "data", "n", "show_score", "axis_text_kw"]
                ),
            )

    @staticmethod
    def _valid_density_size(data: pd.DataFrame) -> None:
        # Check if there's enough data for a meaningful density plot
        if len(data) < RECOMMENDED_MIN_SAMPLES:
            warnings.warn(
                "Density plots are not recommended for small datasets (<30 samples).",
                UserWarning,
                stacklevel=2,
            )


class SimpleDensityLayer(DensityLayer):
    """Layer for rendering simplified density plots with fewer contour levels."""

    def __init__(
        self,
        custom_data: pd.DataFrame | None = None,
        *,
        include_outline: bool = True,
        param_model: type[SimpleDensityParams] = SimpleDensityParams,
        **params: Any,
    ) -> None:
        """
        Initialize a SimpleDensityLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        include_outline : bool
            Whether to include an outline around the density plot
        **params : dict
            Parameters for the density plot

        """
        super().__init__(
            custom_data=custom_data,
            include_outline=include_outline,  # Could move into params now
            param_model=param_model,
            **params,
        )


class SPILayer(Layer):
    """Layer for rendering SPI plots."""

    def __init__(
        self,
        spi_target_data: pd.DataFrame | np.ndarray | None = None,
        *,
        # TODO(MitchellAcoustics): Allow passing raw param values,  # noqa: TD003
        #  not just Param objects
        msn_params: DirectParams | CentredParams | None = None,
        n: int = 10000,
        param_model: type[SPISeabornParams] = SPISeabornParams,
        **params: Any,
    ) -> None:
        """
        Initialize an SPILayer.

        Parameters
        ----------
        spi_target_data : pd.DataFrame | np.ndarray | None
            Pre-sampled data for SPI target distribution.
            When None, msn_params must be provided.
        msn_params : DirectParams | CentredParams | None
            Parameters to generate SPI data if no spi_target_data is provided
        n : int
            Number of samples to generate if using msn_params
        param_model : type[SPISeabornParams]
            The parameter model class to use
        **params : dict
            Parameters for the layer. For compatibility with other layers,
            if 'custom_data' is present and spi_target_data is None,
            custom_data will be used as the SPI target data.

        Notes
        -----
        Either spi_target_data or msn_params must be provided, but not both.
        The test data for SPI calculations will be retrieved from the plot context.

        """
        # The custom_data passed when adding this layer should be the spi_data.
        # We will retrieve the test_data from the subplot context, so real data layers
        # need to be passed before this one, or use the data from the
        # main ISOPlot context
        custom_data = params.pop("custom_data", None)
        if custom_data is not None and spi_target_data is None:
            logger.warning(
                "`spi_target_data` not found, but `custom_data` was found. "
                "Using `custom_data` as the SPI target data. "
                "\nNote: Passing the SPI data to `spi_target_data` is preferred."
            )
            spi_target_data = custom_data

        # Check that we have the information needed to generate SPI target data
        # (either the spi_data or msn_params)
        spi_target_data, self.spi_params = self._validate_spi_inputs(
            spi_target_data, msn_params
        )
        # Generate the spi target data
        self.spi_data: pd.DataFrame = self._generate_spi_data(
            spi_target_data, self.spi_params, n
        )
        params["n"] = n
        self.params: SPISeabornParams

        super().__init__(custom_data=self.spi_data, param_model=param_model, **params)

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

        target_data = self.spi_data
        # Mutate spi_data to match the context test_data.
        target_data = self._process_spi_data(target_data, context)

        if target_data is None:
            msg = "No data available for rendering layer"
            raise ValueError(msg)

        # Now we have the SPI target_data and the test data in context.data
        # SPILayer._render_implementation() will handle calculating the SPI score
        # against this particular context / ax test data

        self._render_implementation(target_data, context, context.ax)

    def _render_implementation(
        self, data: pd.DataFrame, context: PlotContext, ax: Axes
    ) -> None:
        target_data = data if data is not None else self.spi_data
        target_data = target_data[[context.x, context.y]]

        test_data = context.data
        if test_data is None:
            warnings.warn(
                "Cannot find data to test SPI against. Skipping this plot.",
                UserWarning,
                stacklevel=2,
            )
            return
        test_data = test_data[[context.x, context.y]]

        # Process spi_data then pass to DensityLayer method
        spi_sc = self._calc_context_spi_score(target_data, test_data)
        self.show_score(
            spi_sc,
            show_score=self.params.show_score,
            context=context,
            ax=ax,
            axis_text_kwargs={"fontsize": 12},
        )

        super()._render_implementation(target_data, context, ax)

    def show_score(
        self,
        spi_sc: int | None,
        show_score: Literal["on axis", "under title"],
        context: PlotContext,
        ax: Axes,
        axis_text_kwargs: dict[str, Any],
    ) -> None:
        """
        Show the SPI score on the plot.

        Parameters
        ----------
        spi_sc : int | None
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
        if spi_sc is not None:
            if show_score == "on axis":
                self._add_score_as_text(
                    ax=ax,
                    spi_sc=spi_sc,
                    **axis_text_kwargs,
                )
            elif show_score == "under title":
                self._add_score_under_title(
                    context=context,
                    ax=ax,
                    spi_sc=spi_sc,
                )

    @staticmethod
    def _add_score_as_text(ax: Axes, spi_sc: int, **text_kwargs: Any) -> None:
        """
        Add the SPI score as text on the axis.

        Parameters
        ----------
        axis : Axes
            The axes to add the text to
        spi_sc : int
            The SPI score to show
        **text_kwargs : dict[str, Any]
            Additional arguments for the text

        """
        from soundscapy.plotting.defaults import DEFAULT_SPI_TEXT_KWARGS

        text_kwargs = DEFAULT_SPI_TEXT_KWARGS.copy()
        text_kwargs.update(**text_kwargs)
        text_kwargs["s"] = f"SPI: {spi_sc}"

        ax.text(**text_kwargs)

    @staticmethod
    def _add_score_under_title(context: PlotContext, ax: Axes, spi_sc: int) -> None:
        """
        Add the SPI score under the title.

        Parameters
        ----------
        axis : Axes
            The axes to add the text to
        spi_sc : int
            The SPI score to show

        """
        if context.title is not None:
            new_title = f"{context.title}\nSPI: {spi_sc}"
        else:
            new_title = f"SPI: {spi_sc}"

        ax.set_title(new_title)

    @staticmethod
    def _validate_spi_inputs(
        spi_data: pd.DataFrame | np.ndarray | None,
        spi_params: DirectParams | CentredParams | None,
    ) -> tuple[pd.DataFrame | np.ndarray | None, DirectParams | CentredParams | None]:
        """Validate the right combination of inputs for the SPI plot."""
        from soundscapy.spi.msn import CentredParams, DirectParams

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

        if spi_params is not None and not isinstance(
            spi_params, DirectParams | CentredParams
        ):
            msg = (
                "Invalid parameters for SPI plot. "
                "Expected DirectParams or CentredParams."
            )
            raise TypeError(msg)

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
        kwargs : dict
            Additional parameters

        Returns
        -------
        pd.DataFrame | np.ndarray
            Prepared data for SPI plotting

        """
        from soundscapy.spi.msn import MultiSkewNorm

        # Generate data from parameters if provided
        if spi_params is not None:
            spi_msn = MultiSkewNorm.from_params(spi_params)
            sample_data = spi_msn.sample(n=n, return_sample=True)
            spi_data = pd.DataFrame(
                sample_data,
                columns=["x", "y"],
            )
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

        msg = "Please provide either spi_data or msn_params, not both."
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
        kwargs : dict
            Additional parameters with x and y column names

        Returns
        -------
        pd.DataFrame
            Processed data in standard format

        """
        xcol = self.params.get("x", context.x)
        ycol = self.params.get("y", context.y)

        if not (isinstance(xcol, str) and isinstance(ycol, str)):
            msg = "Sorry, at the moment in this method, x and y must be strings."
            raise TypeError(msg)

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
        from soundscapy.spi import spi_score

        return spi_score(target=target_data, test=test_data)


class SPISimpleLayer(SPILayer, SimpleDensityLayer):
    """Layer for rendering simplified SPI plots with fewer contour levels."""

    def __init__(
        self,
        spi_target_data: pd.DataFrame | np.ndarray | None = None,
        *,
        msn_params: DirectParams | CentredParams | None = None,
        include_outline: bool = True,
        **params: Any,
    ) -> None:
        """
        Initialize an SPISimpleLayer.

        Parameters
        ----------
        custom_data : pd.DataFrame | None
            Optional custom data for this specific layer
        msn_params : DirectParams | CentredParams | None
            Parameters to generate SPI data if no custom data is provided
        include_outline : bool
            Whether to include an outline around the density plot
        **params : dict
            Parameters for the density plot

        """
        self.params: SPISimpleDensityParams
        super().__init__(
            spi_target_data=spi_target_data,
            include_outline=include_outline,
            param_model=SPISimpleDensityParams,
            msn_params=msn_params,
            **params,
        )


class SPIDensityLayer(SPILayer, DensityLayer):
    """Layer for rendering simplified SPI plots with fewer contour levels."""

    def __init__(self) -> None:
        """
        Initialize SPIDensityLayer.

        This initialization is not supported and will raise NotImplementedError.
        Use SPISimpleLayer instead.
        """
        msg = (
            "Only the simple density layer type is currently supported for SPI plots. "
            "Please use SPISimpleLayer"
        )
        raise NotImplementedError(msg)


class SPIScatterLayer(SPILayer, ScatterLayer):
    """Layer for rendering simplified SPI plots with fewer contour levels."""

    def __init__(self) -> None:
        """
        Initialize SPIScatterLayer.

        This initialization is not supported and will raise NotImplementedError.
        Use SPISimpleLayer instead.
        """
        msg = (
            "Only the simple density layer type is currently supported for SPI plots. "
            "Please use SPISimpleLayer"
        )
        raise NotImplementedError(msg)
