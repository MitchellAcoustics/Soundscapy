"""
Layer-specific methods for ISOPlot.

This module provides a mixin class with methods for adding different types of
visualization layers to ISOPlot instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from soundscapy.plotting.layers import (
    DensityLayer,
    Layer,
    ScatterLayer,
    SimpleDensityLayer,
    SPISimpleLayer,
)

if TYPE_CHECKING:
    from soundscapy.spi.msn import (
        CentredParams,
        DirectParams,
    )


class ISOPlotLayersMixin:
    """Mixin providing layer-specific methods for ISOPlot."""

    def add_layer(
        self,
        layer_class: type[Layer],
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> Any:
        """
        Add a visualization layer, optionally targeting specific subplot(s).

        This is a stub method that should be implemented by classes using this mixin.
        The actual implementation is in the ISOPlot class.

        Parameters
        ----------
        layer_class : Layer subclass
            The type of layer to add
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes
        data : pd.DataFrame, optional
            Custom data for this specific layer
        **params : dict
            Parameters for the layer

        Returns
        -------
        Any
            The current plot instance for chaining

        """
        msg = "Classes using ISOPlotLayersMixin must implement add_layer"
        raise NotImplementedError(msg)

    def get_single_axes(self, ax_idx: int | tuple[int, int] | None = None) -> Any:
        """
        Get a specific axes object.

        This is a stub method that should be implemented by classes using this mixin.
        The actual implementation is in the ISOPlot class.

        Parameters
        ----------
        ax_idx : int | tuple[int, int] | None, optional
            The index of the axes to get. If None, returns the first axes.
            Can be an integer for flattened access or a tuple of (row, col).

        Returns
        -------
        Any
            The requested matplotlib Axes object

        """
        msg = "Classes using ISOPlotLayersMixin must implement get_single_axes"
        raise NotImplementedError(msg)

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
            Custom data for this specific layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes:
            - int: Index of subplot (flattened)
            - tuple: (row, col) coordinates
            - list: Multiple indices to apply the layer to
            - None: Apply to all subplots (default)
        **params : Any
            Additional parameters for the scatter layer

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a scatter layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> plot = (ISOPlot(data=data)
        ...           .create_subplots(nrows=2, ncols=1)
        ...           .add_scatter(s=50, alpha=0.7, hue='Group')
        ...           .apply_styling())
        >>> plot.show() # xdoctest: +SKIP
        >>> all(len(ctx.layers) == 1 for ctx in plot.subplot_contexts)
        True
        >>> plot.close()  # Clean up

        Add a scatter layer with custom data to a specific subplot:

        >>> custom_data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.1, 50),
        ...     'ISOEventful': rng.normal(0.15, 0.2, 50),
        ... })
        >>> plot = (ISOPlot(data=data)
        ...            .create_subplots(nrows=2, ncols=1)
        ...            .add_scatter(hue='Group')
        ...            .add_scatter(on_axis=0, data=custom_data, color='red')
        ...            .apply_styling())
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.subplot_contexts[0].layers[1].custom_data is custom_data
        True
        >>> plot.close()  # Clean up

        """
        return self.add_layer(ScatterLayer, data=data, on_axis=on_axis, **params)

    def add_spi(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        spi_target_data: pd.DataFrame | np.ndarray | None = None,
        msn_params: DirectParams | CentredParams | None = None,
        *,
        layer_class: type[Layer] = SPISimpleLayer,
        **params: Any,
    ) -> Any:
        """
        Add an SPI (Soundscape Perception Index) layer to the plot.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes:
            - int: Index of subplot (flattened)
            - tuple: (row, col) coordinates
            - list: Multiple indices to apply the layer to
            - None: Apply to all subplots (default)
        spi_target_data : pd.DataFrame | np.ndarray | None, optional
            Pre-sampled data for SPI target distribution, by default None
        msn_params : DirectParams | CentredParams | None, optional
            Parameters to generate SPI data if no spi_target_data is provided, by default None
        layer_class : type[Layer], optional
            The type of SPI layer to add, by default SPISimpleLayer
        **params : Any
            Additional parameters for the SPI layer

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Notes
        -----
        Either spi_target_data or msn_params must be provided, but not both.
        The test data for SPI calculations will be retrieved from the plot context.

        Examples
        --------
        Add a SPI layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.spi import DirectParams
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>>    # Create a DataFrame with random data
        >>> data = pd.DataFrame(
        ...    rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...    columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>>    # Define MSN parameters for the SPI target
        >>> msn_params = DirectParams(
        ...     xi=np.array([0.5, 0.7]),
        ...     omega=np.array([[0.1, 0.05], [0.05, 0.1]]),
        ...     alpha=np.array([0, -5]),
        ...     )
        >>>    # Create the plot with only an SPI layer
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_spi(msn_params=msn_params)
        ...     .apply_styling()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 2
        True
        >>> plot.close()  # Clean up

        Add an SPI layer over top of 'real' data:
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_density()
        ...     .add_spi(msn_params=msn_params, show_score="on axis")
        ...     .apply_styling()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 3
        True
        >>> plot.close()  # Clean up

        """
        # Validate that we have either spi_target_data or msn_params
        if spi_target_data is None and msn_params is None:
            msg = (
                "No data provided for SPI plot. "
                "Please provide either spi_target_data or msn_params."
            )
            raise ValueError(msg)

        if spi_target_data is not None and msn_params is not None:
            msg = (
                "Please provide either spi_target_data or msn_params, not both. "
                "Got: \n"
                f"\n`spi_target_data`: {type(spi_target_data)}\n`msn_params`: {type(msn_params)}"
            )
            raise ValueError(msg)

        # Add the SPI layer
        return self.add_layer(
            layer_class,
            on_axis=on_axis,
            spi_target_data=spi_target_data,
            msn_params=msn_params,
            **params,
        )

    def add_density(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        *,
        include_outline: bool = False,
        **params: Any,
    ) -> Any:
        """
        Add a density layer to the plot.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes:
            - int: Index of subplot (flattened)
            - tuple: (row, col) coordinates
            - list: Multiple indices to apply the layer to
            - None: Apply to all subplots (default)
        data : pd.DataFrame | None, optional
            Custom data for this specific layer, by default None
        include_outline : bool, optional
            Whether to include an outline around the density plot, by default False
        **params : Any
            Additional parameters for the density layer

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a density layer to all subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.25, 50),
        ...     'ISOEventful': rng.normal(0.15, 0.4, 50),
        ... })
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_density()
        ...     .apply_styling()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 1
        True
        >>> plot.close()  # Clean up

        Add a density layer with custom settings:

        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_density(levels=5, alpha=0.7)
        ...     .apply_styling()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 1
        True
        >>> plot.close()  # Clean up

        """
        return self.add_layer(
            DensityLayer,
            data=data,
            on_axis=on_axis,
            include_outline=include_outline,
            **params,
        )

    def add_simple_density(
        self,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        data: pd.DataFrame | None = None,
        *,
        include_outline: bool = True,
        **params: Any,
    ) -> Any:
        """
        Add a simplified density layer to the plot.

        This creates a density plot with fewer contour levels, typically used
        to highlight the main density region.

        Parameters
        ----------
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes:
            - int: Index of subplot (flattened)
            - tuple: (row, col) coordinates
            - list: Multiple indices to apply the layer to
            - None: Apply to all subplots (default)
        data : pd.DataFrame | None, optional
            Custom data for this specific layer, by default None
        include_outline : bool, optional
            Whether to include an outline around the density plot, by default True
        **params : Any
            Additional parameters for the simple density layer

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a simple density layer:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.2, 0.25, 30),
        ...     'ISOEventful': rng.normal(0.15, 0.4, 30),
        ... })
        >>> plot = (
        ...     ISOPlot(data=data)
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_simple_density()
        ...     .apply_styling()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 2
        True
        >>> plot.close()  # Clean up

        Add a simple density with splitting by group:
        >>> data = pd.DataFrame(
        ...    np.c_[rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...          rng.integers(1, 3, 100)],
        ...    columns=['ISOPleasant', 'ISOEventful', 'Group'])
        >>> plot = (
        ...     ISOPlot(data=data, hue='Group')
        ...     .create_subplots()
        ...     .add_scatter()
        ...     .add_simple_density()
        ...     .apply_styling()
        ... )
        >>> plot.show() # xdoctest: +SKIP
        >>> len(plot.subplot_contexts[0].layers) == 2
        True
        >>> plot.close()

        """
        return self.add_layer(
            SimpleDensityLayer,
            data=data,
            on_axis=on_axis,
            include_outline=include_outline,
            **params,
        )

    def add_annotation(
        self,
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float],
        arrowprops: dict[str, Any] | None = None,
    ) -> Any:
        """
        Add an annotation to the plot.

        Parameters
        ----------
        text : str
            The text of the annotation
        xy : tuple[float, float]
            The point (x, y) to annotate
        xytext : tuple[float, float]
            The position (x, y) to place the text
        arrowprops : dict[str, Any] | None, optional
            Properties used to draw the arrow, by default None

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting import ISOPlot
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0, 0.5, 100),
        ...     'ISOEventful': rng.normal(0, 0.5, 100),
        ... })
        >>> plot = (ISOPlot(data=data)
        ...         .create_subplots()
        ...         .add_scatter()
        ...         .add_annotation(
        ...             "Interesting point",
        ...             xy=(0.5, 0.5),
        ...             xytext=(0.7, 0.7),
        ...             arrowprops=dict(arrowstyle="->")
        ...         )
        ...         .apply_styling())
        >>> plot.show() # xdoctest: +SKIP
        >>> plot.close()  # Clean up

        """
        # Default arrow properties if none provided
        if arrowprops is None:
            arrowprops = {"arrowstyle": "->"}

        # Get the current axes
        ax = self.get_single_axes()
        ax.annotate(text, xy=xy, xytext=xytext, arrowprops=arrowprops)

        return self
