"""
Main plotting class for the soundscapy package.

This module provides the ISOPlot class, which is the main entry point for creating
circumplex plots. The class uses composition instead of inheritance to delegate
functionality to specialized manager classes.

Examples
--------
Create a simple scatter plot:

>>> import pandas as pd
>>> import numpy as np
>>> from soundscapy.plotting.new import ISOPlot
>>> # Create some sample data
>>> rng = np.random.default_rng(42)
>>> data = pd.DataFrame(
...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
...     columns=['ISOPleasant', 'ISOEventful']
... )
>>> # Create a plot and add a scatter layer
>>> plot = ISOPlot(data=data)
>>> plot.add_scatter()
>>> plot.apply_styling()
>>> isinstance(plot, ISOPlot)
True

Create a plot with subplots and multiple layers:

>>> # Add a group column to the data
>>> data['Group'] = rng.integers(1, 3, 100)
>>> # Create a plot with subplots by group
>>> plot = (ISOPlot(data=data, hue='Group')
...         .add_scatter()
...         .add_simple_density(fill=False)
...         .apply_styling())
>>> isinstance(plot, ISOPlot)
True

"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soundscapy.plotting.new.constants import (
    DEFAULT_XCOL,
    DEFAULT_YCOL,
)
from soundscapy.plotting.new.layer import (
    Layer,
)
from soundscapy.plotting.new.managers import (
    LayerManager,
    StyleManager,
    SubplotManager,
)
from soundscapy.plotting.new.parameter_models import (
    SubplotsParams,
)
from soundscapy.plotting.new.plot_context import PlotContext
from soundscapy.sspylogging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = get_logger()


class ISOPlot:
    """
    A class for creating circumplex plots using different visualization layers.

    This class provides methods for creating scatter plots, density plots, and other
    visualizations based on the circumplex model of soundscape perception. It uses
    composition to delegate functionality to specialized manager classes.

    Attributes
    ----------
    main_context : PlotContext
        The main plot context
    figure : Figure | None
        The matplotlib figure
    axes : Axes | np.ndarray | None
        The matplotlib axes
    subplot_contexts : list[PlotContext]
        List of subplot contexts
    subplots_params : SubplotsParams
        Parameters for subplot configuration
    layers : LayerManager
        Manager for layer-related functionality
    styling : StyleManager
        Manager for styling-related functionality
    subplots : SubplotManager
        Manager for subplot-related functionality

    Examples
    --------
    Create a plot with default parameters:

    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> plot = ISOPlot()
    >>> isinstance(plot, ISOPlot)
    True

    Create a plot with a DataFrame:

    >>> data = pd.DataFrame(
    ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
    ...     columns=['ISOPleasant', 'ISOEventful']
    ... )
    >>> plot = ISOPlot(data=data)
    >>> plot.x
    'ISOPleasant'
    >>> plot.y
    'ISOEventful'

    Create a plot with a DataFrame and hue:

    >>> data['Group'] = rng.integers(1, 3, 100)
    >>> plot = ISOPlot(data=data, hue='Group')
    >>> plot.hue
    'Group'

    Create a plot directly with arrays:

    >>> x, y = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100).T
    >>> plot = ISOPlot(x=x, y=y)
    >>> isinstance(plot, ISOPlot)
    True
    >>> plot.main_context.data is not None
    True

    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        x: str | np.ndarray | pd.Series | None = DEFAULT_XCOL,
        y: str | np.ndarray | pd.Series | None = DEFAULT_YCOL,
        title: str | None = "Soundscape Plot",
        hue: str | None = None,
        palette: str | list | dict | None = "colorblind",
        figure: Figure | None = None,
        axes: Axes | np.ndarray | None = None,
    ) -> None:
        """
        Initialize an ISOPlot instance.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            The data to be plotted, by default None
        x : str | np.ndarray | pd.Series | None, optional
            Column name or data for x-axis, by default DEFAULT_XCOL
        y : str | np.ndarray | pd.Series | None, optional
            Column name or data for y-axis, by default DEFAULT_YCOL
        title : str | None, optional
            Title of the plot, by default "Soundscape Plot"
        hue : str | None, optional
            Column name for color encoding, by default None
        palette : str | list | dict | None, optional
            Color palette to use, by default "colorblind"
        figure : Figure | None, optional
            Existing figure to plot on, by default None
        axes : Axes | np.ndarray | None, optional
            Existing axes to plot on, by default None

        """
        # Process and validate input data and coordinates
        data, x, y = self._check_data_x_y(data, x, y)
        self._check_data_hue(data, hue)

        # Initialize the main plot context
        self.main_context = PlotContext(
            data=data,
            x=x if isinstance(x, str) else DEFAULT_XCOL,
            y=y if isinstance(y, str) else DEFAULT_YCOL,
            hue=hue,
            title=title,
        )

        # Store additional plot attributes
        self.figure = figure
        self.axes = axes
        self.palette = palette

        # Initialize subplot management
        self.subplot_contexts: list[PlotContext] = []
        self.subplots_params = SubplotsParams()

        # Initialize managers using composition
        self.layers = LayerManager(self)
        self.styling = StyleManager(self)
        self.subplots = SubplotManager(self)

    @property
    def x(self) -> str:
        """Get the x-axis column name."""
        return self.main_context.x

    @property
    def y(self) -> str:
        """Get the y-axis column name."""
        return self.main_context.y

    @property
    def hue(self) -> str | None:
        """Get the hue column name."""
        return self.main_context.hue

    @property
    def title(self) -> str | None:
        """Get the plot title."""
        return self.main_context.title

    @staticmethod
    def _check_data_x_y(
        data: pd.DataFrame | None,
        x: str | pd.Series | np.ndarray | None,
        y: str | pd.Series | np.ndarray | None,
    ) -> tuple[
        pd.DataFrame | None, str | np.ndarray | pd.Series, str | np.ndarray | pd.Series
    ]:
        """
        Process and validate input data and coordinates.

        Parameters
        ----------
        data : pd.DataFrame | None
            The data to be plotted
        x : str | pd.Series | np.ndarray | None
            Column name or data for x-axis
        y : str | pd.Series | np.ndarray | None
            Column name or data for y-axis

        Returns
        -------
        tuple[pd.DataFrame | None, str | np.ndarray | pd.Series, str | np.ndarray | pd.Series]
            Processed data, x, and y

        """  # noqa: E501
        # Case 1: x and y are arrays/series, data is None
        if (
            isinstance(x, (np.ndarray, pd.Series))
            and isinstance(y, (np.ndarray, pd.Series))
            and data is None
        ):
            # Create a DataFrame from x and y
            data = pd.DataFrame(
                {
                    DEFAULT_XCOL: x,
                    DEFAULT_YCOL: y,
                }
            )
            x = DEFAULT_XCOL
            y = DEFAULT_YCOL
            return data, x, y

        # Case 2: data is provided, x and y are column names
        if data is not None:
            # Ensure x and y are column names if they're strings
            if isinstance(x, str) and x not in data.columns:
                msg = f"Column '{x}' not found in data"
                raise ValueError(msg)
            if isinstance(y, str) and y not in data.columns:
                msg = f"Column '{y}' not found in data"
                raise ValueError(msg)
            return data, x or DEFAULT_XCOL, y or DEFAULT_YCOL

        # Case 3: No data provided, use default column names
        return None, x or DEFAULT_XCOL, y or DEFAULT_YCOL

    @staticmethod
    def _check_data_hue(data: pd.DataFrame | None, hue: str | None) -> None:
        """
        Check if the hue column exists in the data.

        Parameters
        ----------
        data : pd.DataFrame | None
            The data to be plotted
        hue : str | None
            Column name for color encoding

        Raises
        ------
        ValueError
            If the hue column is not found in the data

        """
        if data is not None and hue is not None and hue not in data.columns:
            msg = f"Hue column '{hue}' not found in data"
            raise ValueError(msg)

    # Convenience methods that delegate to managers

    def create_subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        subplot_by: str | None = None,
        *,
        sharex: bool | Literal["none", "all", "row", "col"] = True,
        sharey: bool | Literal["none", "all", "row", "col"] = True,
        **kwargs: Any,
    ) -> ISOPlot:
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
        subplot_by : str | None, optional
            Column to create subplots by, by default None
        *
        sharex : bool | Literal["none", "all", "row", "col"], optional
            Whether to share x-axis, by default True
        sharey : bool | Literal["none", "all", "row", "col"], optional
            Whether to share y-axis, by default True
        **kwargs : Any
            Additional parameters for subplots

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Create a 2x2 grid of subplots:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(nrows=2,ncols=2)
        >>> len(plot.subplot_contexts)
        4

        Create subplots by a grouping variable:

        >>> data['Group'] = rng.integers(1, 3, 100)
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(subplot_by='Group')
        >>> len(plot.subplot_contexts)
        2
        >>> plot.subplot_contexts[0].title is not None
        True

        """
        return self.subplots.create_subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            subplot_by=subplot_by,
            **kwargs,
        )

    def add_scatter(
        self,
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> ISOPlot:
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
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a scatter layer to a plot:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_scatter()
        >>> len(plot.main_context.layers)
        1

        Add a scatter layer with custom parameters:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_scatter(s=50, alpha=0.5, color='red')
        >>> len(plot.main_context.layers)
        1

        Add a scatter layer to a specific subplot:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(nrows=2,ncols=2)
        >>> plot = plot.add_scatter(on_axis=0)
        >>> len(plot.subplot_contexts[0].layers)
        1
        >>> len(plot.subplot_contexts[1].layers)
        0

        """
        return self.layers.add_scatter(
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
    ) -> ISOPlot:
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
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a density layer to a plot:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_density()
        >>> len(plot.main_context.layers)
        1

        Add a density layer with custom parameters:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_density(fill=False, levels=5, alpha=0.7)
        >>> len(plot.main_context.layers)
        1

        Add a density layer to a specific subplot:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(nrows=2,ncols=2)
        >>> plot = plot.add_density(on_axis=1)
        >>> len(plot.subplot_contexts[0].layers)
        0
        >>> len(plot.subplot_contexts[1].layers)
        1

        """
        return self.layers.add_density(
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
    ) -> ISOPlot:
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
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a simple density layer to a plot:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_simple_density()
        >>> len(plot.main_context.layers)
        1

        Add a simple density layer with custom parameters:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_simple_density(fill=False, thresh=0.3)
        >>> len(plot.main_context.layers)
        1

        Add a simple density layer to multiple subplots:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(nrows=2,ncols=2)
        >>> plot = plot.add_simple_density(on_axis=[0, 2])
        >>> len(plot.subplot_contexts[0].layers)
        1
        >>> len(plot.subplot_contexts[1].layers)
        0
        >>> len(plot.subplot_contexts[2].layers)
        1

        """
        return self.layers.add_simple_density(
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
    ) -> ISOPlot:
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
        ISOPlot
            The current plot instance for chaining

        """
        return self.layers.add_spi_simple(
            data=data,
            on_axis=on_axis,
            **params,
        )

    def add_layer(
        self,
        layer_class: type[Layer],
        data: pd.DataFrame | None = None,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **params: Any,
    ) -> ISOPlot:
        """
        Add a visualization layer, optionally targeting specific subplot(s).

        Parameters
        ----------
        layer_class : Layer subclass
            The type of layer to add
        data : pd.DataFrame | None, optional
            Custom data for this layer, by default None
        on_axis : int | tuple[int, int] | list[int] | None, optional
            Target specific axis/axes, by default None
        **params : Any
            Additional parameters for the layer

        Returns
        -------
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Add a scatter layer using the generic add_layer method:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from soundscapy.plotting.new import ScatterLayer
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_layer(ScatterLayer)
        >>> len(plot.main_context.layers)
        1
        >>> isinstance(plot.main_context.layers[0], ScatterLayer)
        True

        Add a density layer to a specific subplot:

        >>> from soundscapy.plotting.new import DensityLayer
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(nrows=2,ncols=2)
        >>> plot = plot.add_layer(DensityLayer, on_axis=3, fill=False)
        >>> len(plot.subplot_contexts[3].layers)
        1
        >>> isinstance(plot.subplot_contexts[3].layers[0], DensityLayer)
        True

        Add a layer with custom data:

        >>> custom_data = pd.DataFrame({
        ...     'ISOPleasant': rng.normal(0.5, 0.1, 50),
        ...     'ISOEventful': rng.normal(0.5, 0.1, 50),
        ... })
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_layer(ScatterLayer, data=custom_data, color='red')
        >>> len(plot.main_context.layers)
        1

        """
        return self.layers.add_layer(
            layer_class=layer_class,
            data=data,
            on_axis=on_axis,
            **params,
        )

    def apply_styling(
        self,
        *,
        on_axis: int | tuple[int, int] | list[int] | None = None,
        **style_params: Any,
    ) -> ISOPlot:
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
        ISOPlot
            The current plot instance for chaining

        Examples
        --------
        Apply default styling to a plot:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_scatter()
        >>> plot = plot.apply_styling()
        >>> isinstance(plot, ISOPlot)
        True

        Apply custom styling to a plot:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.add_scatter()
        >>> plot = plot.apply_styling(
        ...     xlim=(-2, 2),
        ...     ylim=(-2, 2),
        ...     xlabel="Pleasant",
        ...     ylabel="Eventful",
        ...     primary_lines=True,
        ...     diagonal_lines=True
        ... )
        >>> isinstance(plot, ISOPlot)
        True

        Apply styling to a specific subplot:

        >>> plot = ISOPlot(data=data)
        >>> plot = plot.create_subplots(nrows=2,ncols=2)
        >>> plot = plot.add_scatter()
        >>> plot = plot.apply_styling(on_axis=0, title="Subplot 0")
        >>> isinstance(plot, ISOPlot)
        True

        """
        return self.styling.apply_styling(
            on_axis=on_axis,
            **style_params,
        )

    @functools.wraps(plt.show)
    def show(self) -> None:
        """
        Display the plot.

        Examples
        --------
        Create and show a plot:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot.add_scatter()
        >>> plot.apply_styling()
        >>> # plot.show()  # Uncomment to display the plot

        """
        plt.show()

    @functools.wraps(plt.close)
    def close(self) -> None:
        """
        Close the plot.

        Examples
        --------
        Create, show, and close a plot:

        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.multivariate_normal([0.2, 0.15], [[0.1, 0], [0, 0.2]], 100),
        ...     columns=['ISOPleasant', 'ISOEventful']
        ... )
        >>> plot = ISOPlot(data=data)
        >>> plot.add_scatter()
        >>> plot.apply_styling()
        >>> # plot.show()  # Uncomment to display the plot
        >>> plot.close()  # Close the plot

        """
        if self.figure is not None:
            plt.close(self.figure)
