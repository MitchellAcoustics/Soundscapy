"""Plotting functions for visualizing circumplex data."""

# ruff: noqa: ANN003
import warnings
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from matplotlib.typing import ColorType

from soundscapy.plotting.backends_deprecated import Backend
from soundscapy.plotting.defaults import (
    DEFAULT_BW_ADJUST,
    DEFAULT_COLOR,
    DEFAULT_FIGSIZE,
    DEFAULT_SEABORN_PARAMS,
    DEFAULT_SIMPLE_DENSITY_PARAMS,
    DEFAULT_STYLE_PARAMS,
    DEFAULT_XCOL,
    DEFAULT_XY_LABEL_FONTDICT,
    DEFAULT_YCOL,
    RECOMMENDED_MIN_SAMPLES,
)
from soundscapy.plotting.iso_plot import ISOPlot
from soundscapy.plotting.param_models import (
    DensityParams,
    MplLegendLocType,
    ScatterParams,
    SeabornPaletteType,
    SimpleDensityParams,
    StyleParams,
    SubplotsParams,
)
from soundscapy.sspylogging import get_logger

# Error messages
PLOT_LAYER_TYPE_ERROR = (
    "The `plot_layers` argument must be a string or a sequence of strings. "
    "Got {type} instead."
)
PLOT_LAYER_VALUE_ERROR = (
    "Plot / layer type not understood. "
    "Supported layers are: 'scatter', 'density', 'simple_density'. Got: {layers}"
)
SUBPLOT_DATA_ERROR = (
    "If data is a DataFrame, subplot_by must be a grouping column in the "
    "dataframe to create subplots."
)
XY_DATA_ERROR = (
    "If data is a DataFrame, x and y must be column names in the dataframe. "
    "Got: x: {x}, y: {y}."
)
DATA_LIST_TYPE_ERROR = "`data_list` should contain only pandas DataFrames."
DATA_TYPE_ERROR = (
    "data must be a DataFrame with a provided `subplot_by` column or a "
    "list of DataFrames to create subplots."
)
SUBPLOT_TITLES_ERROR = (
    "Not enough `subplot_titles` provided. "
    "Need to provide at least as many titles as subplots: {n_subplots}"
)
PRIM_LABELS_DEPRECATION_WARNING = (
    "The `prim_labels` parameter is deprecated. Use `xlabel` and `ylabel` instead."
)

logger = get_logger()

STYLE_PARAMS = StyleParams()
SCATTER_PARAMS = ScatterParams()
DENSITY_PARAMS = DensityParams()
SIMPLE_DENSITY_PARAMS = SimpleDensityParams()


def iso_plot(
    data: pd.DataFrame,
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    title: str | None = "Soundscapy Plot",
    plot_layers: Literal["scatter", "density", "simple_density"]
    | Sequence[Literal["scatter", "density", "simple_density"]] = (
        "scatter",
        "density",
    ),
    **kwargs,
) -> Axes:
    """
    Plot a soundscape visualization based on the specified metrics using different
    combinations of layers such as scatter, density, or simple density plots.

    The function generates a 2D plot (via Matplotlib Axes object) based on the `x` and
    `y` metrics provided. Users can choose between individual layers or specify a
    combination of the supported plot layers. It supports automatic handling for
    specific layer combinations, such as "scatter + density". The core plotting
    functionality is delegated to other helper functions (`scatter` and `density`).

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing the metrics to be plotted. Must include the columns
        specified for `x` and `y`.
    x : str, optional
        The column name within `data` to be used for the x-axis. Defaults to
        "ISOPleasant".
    y : str, optional
        The column name within `data` to be used for the y-axis. Defaults to
        "ISOEventful".
    title : str or None, optional
        The title of the plot. If not specified, defaults to "Soundscapy Plot".
    plot_layers : {"scatter", "density", "simple_density"}
        or Sequence[{"scatter", "density", "simple_density"}], optional

        Specifies the type or combination of plot layers to generate. Valid options
        include:
         - "scatter": A scatter plot
         - "density": A density plot (without scatter points, unless combined)
         - "simple_density": A simplified density plot (without scatter points,
           unless combined).

        Can be passed as a string (single layer) or as a sequence of strings
        (combination of layers). Defaults to ("scatter", "density").
    **kwargs : any
        Additional keyword arguments to be passed to the underlying plotting
        functions (`scatter` or `density`). These allow for customization such as
        marker styles, colors, etc.

    Returns
    -------
    Axes : matplotlib.axes._axes.Axes
        A Matplotlib Axes object corresponding to the generated plot.

    Notes
    -----
    This function supports only specific combinations of layers. If an unsupported
    combination is specified, an exception will be raised. Layer compatibility
    rules:
      - Single layers: "scatter", "density", or "simple_density".
      - Dual layers:
        - "scatter" + "density"
        - "scatter" + "simple_density"

    Raises
    ------
    TypeError
        If the `plot_layers` argument is not a string or a sequence of strings.
    ValueError
        If the `plot_layers` argument specifies an unsupported plot type or
        combination of plot layers.

    Examples
    --------
    Basic density and scatter plot with default settings:

    >>> import soundscapy as sspy
    >>> import matplotlib.pyplot as plt
    >>> data = sspy.isd.load()
    >>> data = sspy.add_iso_coords(data)
    >>> ax = sspy.iso_plot(data)
    >>> plt.show()  # xdoctest: +SKIP

    Basic scatter plot:

    >>> ax = sspy.iso_plot(data, plot_layers="scatter")
    >>> plt.show()  # xdoctest: +SKIP

    Simple density plot with fewer contour levels:

    >>> ax = sspy.iso_plot(data, plot_layers="simple_density")
    >>> plt.show() # xdoctest: +SKIP

    Simple density with scatter points

    >>> ax = sspy.iso_plot(data, plot_layers=["scatter", "simple_density"])
    >>> plt.show() # xdoctest: +SKIP

    Density plot with custom styling:

    >>> sub_data = sspy.isd.select_location_ids(
    ...    data, ['CamdenTown', 'PancrasLock', 'RegentsParkJapan', 'RegentsParkFields'])
    >>> ax = sspy.iso_plot(
    ...     sub_data,
    ...     hue="SessionID",
    ...     plot_layers = ["scatter", "simple_density"],
    ...     legend_loc="upper right",
    ...     fill = False,
    ... )
    >>> plt.show() # xdoctest: +SKIP

    Add density to existing plots:

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    >>> axes[0] = sspy.iso_plot(
    ...     sspy.isd.select_location_ids(data, ['CamdenTown', 'PancrasLock']),
    ...     ax=axes.flatten()[0], title="CamdenTown and PancrasLock", hue="LocationID",
    ... )
    >>> axes[1] = sspy.iso_plot(
    ...     sspy.isd.select_location_ids(data, ['RegentsParkJapan']),
    ...     ax=axes.flatten()[1], title="RegentsParkJapan"
    ... )
    >>> plt.tight_layout()
    >>> plt.show() # xdoctest: +SKIP
    >>> plt.close('all')

    """  # noqa: D205
    if isinstance(plot_layers, str):
        plot_layers = [plot_layers]

    if not isinstance(plot_layers, Sequence):
        raise TypeError(PLOT_LAYER_TYPE_ERROR.format(type=type(plot_layers)))

    # Handle single layer case
    if len(plot_layers) == 1:
        layer_type = plot_layers[0]
        if layer_type == "scatter":
            return scatter(data, x=x, y=y, title=title, **kwargs)
        if layer_type == "simple_density":
            return density(
                data,
                x=x,
                y=y,
                title=title,
                density_type="simple",
                incl_scatter=False,
                **kwargs,
            )
        if layer_type == "density":
            return density(data, x=x, y=y, title=title, incl_scatter=False, **kwargs)

        raise ValueError(PLOT_LAYER_VALUE_ERROR.format(layers=plot_layers))

    # Handle two layer case
    if len(plot_layers) == 2:
        layers_set = set(plot_layers)

        if "scatter" in layers_set and "density" in layers_set:
            return density(data, x=x, y=y, title=title, incl_scatter=True, **kwargs)

        if "scatter" in layers_set and "simple_density" in layers_set:
            return density(
                data,
                x=x,
                y=y,
                title=title,
                density_type="simple",
                incl_scatter=True,
                **kwargs,
            )

        # Default case for unrecognized but valid length combinations
        return density(data, x=x, y=y, title=title, incl_scatter=True, **kwargs)

    # More than 2 layers is not supported
    raise ValueError(PLOT_LAYER_VALUE_ERROR.format(layers=plot_layers))


def create_iso_subplots(
    data: pd.DataFrame | list[pd.DataFrame],
    x: str = "ISOPleasant",
    y: str = "ISOEventful",
    subplot_by: str | None = None,
    title: str | None = "Soundscapy Plot",
    plot_layers: Literal["scatter", "density", "simple_density"]
    | Sequence[Literal["scatter", "simple_density", "density"]] = (
        "scatter",
        "density",
    ),
    *,
    subplot_size: tuple[int, int] = (4, 4),
    subplot_titles: Literal["by_group", "numbered"] | list[str] | None = "by_group",
    subplot_title_prefix: str = "Plot",  # Only used if subplot_titles = 'numbered'
    nrows: int | None = None,
    ncols: int | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """
    Create a set of subplots displaying data visualizations for soundscape analysis.

    This function generates a collection of subplots, where each subplot corresponds
    to a subset of the input data. The subplots can display scatter plots, density
    plots, or simplified density plots, and can be organized by specific grouping
    criteria. Users can specify titles, overall size, row and column layout, and
    layering of plot types.

    Parameters
    ----------
    data : pandas.DataFrame or list of pandas.DataFrame
        Input data to be visualized. Can be a single data frame or a list of data
        frames for use in multiple subplots.
    x : str, optional
        The name of the column in the data to be used for the x-axis. Default is
        "ISOPleasant".
    y : str, optional
        The name of the column in the data to be used for the y-axis. Default is
        "ISOEventful".
    subplot_by : str or None, optional
        The column name by which to group data into subplots. If None, data is not
        grouped and plotted in a single set of axes. Default is None.
    title : str or None, optional
        The overarching title of the figure. If None, no overall title is added.
        Default is "Soundscapy Plot".
    plot_layers : Literal["scatter", "density", "simple_density"] or Sequence of
        such Literals, optional
        Type(s) of plot layers to include in each subplot. Can be a single type
        or a sequence of types. Default is ("scatter", "density").
    subplot_size : tuple of int, optional
        Size of each subplot in inches as (width, height). Default is (4, 4).
    subplot_titles : Literal["by_group", "numbered"], list of str, or None,
        optional
        Determines how subplot titles are assigned. Options are "by_group" (titles
        derived from group names), "numbered" (titles as indices), or a list of
        custom titles. If None, no titles are added. Default is "by_group".
    subplot_title_prefix : str, optional
        Prefix for subplot titles if "numbered" is selected as `subplot_titles`.
        Default is "Plot".
    nrows : int or None, optional
        Number of rows for the subplot grid. If None, automatically calculated
        based on the number of subplots. Default is None.
    ncols : int or None, optional
        Number of columns for the subplot grid. If None, automatically calculated
        based on the number of subplots. Default is None.
    **kwargs
        Additional keyword arguments to pass to matplotlib's `plt.subplots` or for
        customizing the figure and subplots.

    Returns
    -------
    tuple
        A tuple containing:
        - fig : matplotlib.figure.Figure
            The created matplotlib figure object containing the subplots.
        - np.ndarray
            An array of matplotlib.axes.Axes objects corresponding to the subplots.

    Examples
    --------
    Basic subplots with default settings:
    >>> import soundscapy as sspy
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> data = sspy.isd.load()
    >>> data = sspy.add_iso_coords(data)
    >>> four_locs = sspy.isd.select_location_ids(data,
    ...     ['CamdenTown', 'PancrasLock', 'RegentsParkJapan', 'RegentsParkFields']
    ... )
    >>> fig, axes = sspy.create_iso_subplots(four_locs, subplot_by="LocationID")
    >>> plt.show() # xdoctest: +SKIP

    Create subplots by specifying a list of data
    >>> data1 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
    ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
    >>> data2 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
    ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
    >>> fig, axes = create_iso_subplots(
    ...     [data1, data2], plot_layers="scatter", nrows=1, ncols=2
    ... )
    >>> plt.show() # xdoctest: +SKIP
    >>> assert len(axes) == 2
    >>> plt.close('all')

    """
    # Process input data and prepare for subplot creation
    data_list, subplot_titles_list, n_subplots = _prepare_subplot_data(
        data=data, x=y, y=y, subplot_by=subplot_by, subplot_titles=subplot_titles
    )

    # Calculate subplot layout
    nrows, ncols, n_subplots = allocate_subplot_axes(nrows, ncols, n_subplots)

    # Set up figure and subplots
    if title:
        vert_adjust = 1.2
    else:
        vert_adjust = 1.0
    figsize = kwargs.pop(
        "figsize", (ncols * subplot_size[0], nrows * (vert_adjust * subplot_size[1]))
    )

    subplots_params = SubplotsParams()
    subplots_params.update(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_by=subplot_by,
        extra="ignore",
        **kwargs,
    )

    fig, axes = plt.subplots(**subplots_params.as_plt_subplots_args())

    # Create each subplot
    _create_subplots(
        data_list,
        axes,
        n_subplots,
        subplot_titles_list,
        x,
        y,
        plot_layers,
        subplot_title_prefix,
        **kwargs,
    )

    # Add overall title and adjust layout
    if title:
        fig.suptitle(title, fontsize=DEFAULT_STYLE_PARAMS["title_fontsize"])

    fig.tight_layout()

    return fig, axes


def _prepare_subplot_data(
    data: pd.DataFrame | list[pd.DataFrame],
    x: str,
    y: str,
    subplot_by: str | None,
    subplot_titles: Literal["by_group", "numbered"] | list[str] | None,
) -> tuple[list[pd.DataFrame], list[str] | Literal["numbered"] | None, int]:
    """
    Prepare data and title information for subplots.

    Parameters
    ----------
    data : pd.DataFrame or list[pd.DataFrame]
        The data to be plotted, either as a single DataFrame or a list of DataFrames
    subplot_by : str or None
        Column name to group data by for subplots (required if data is a DataFrame)
    subplot_titles : "by_group", "numbered", list[str], or None
        How to title the subplots

    Returns
    -------
    tuple
        A tuple containing:
        - data_list: list of DataFrames for each subplot
        - subplot_titles_list: list of titles or "numbered" or None
        - n_subplots: number of subplots

    """
    # Handle list of DataFrames input
    if isinstance(data, list):
        if not all(isinstance(d, pd.DataFrame) for d in data):
            raise TypeError(DATA_LIST_TYPE_ERROR)

        data_list: list[pd.DataFrame] = data
        n_subplots = len(data_list)
        if subplot_titles == "by_group":
            warnings.warn(
                "No group provided for titles. Falling back to numbered subplots."
                "Recommended to manually provide subplot titles or explicitly set "
                "`subplot_titles = 'numbered'` if providing own splits of data.",
                stacklevel=2,
            )
            subplot_titles_list: Literal["numbered"] | list[str] | None = "numbered"
        else:
            subplot_titles_list = subplot_titles  # type: ignore[assignment]

    # Handle DataFrame input
    elif isinstance(data, pd.DataFrame):
        if subplot_by is None:
            raise ValueError(SUBPLOT_DATA_ERROR)
        if subplot_by not in data.columns:
            raise ValueError(SUBPLOT_DATA_ERROR)
        if x not in data.columns and y not in data.columns:
            raise ValueError(SUBPLOT_DATA_ERROR)

        # Get the number of subplots needed
        subplot_groups = data[subplot_by].unique()
        n_subplots = len(subplot_groups)

        # Split the data into groups based on the subplot_by column
        data_list = [
            cast("pd.DataFrame", data[data[subplot_by] == val])
            for val in subplot_groups
        ]

        # Set subplot titles based on groups if requested
        if subplot_titles == "by_group":
            subplot_titles_list = subplot_groups.tolist()
        else:
            warnings.warn(
                "No group provided for titles. Falling back to numbered subplots."
                "Recommended to manually provide subplot titles or explicitly set "
                "`subplot_titles = 'numbered'` if providing own splits of data.",
                stacklevel=2,
            )
            subplot_titles_list = "numbered"

    # Handle invalid input type
    else:
        raise TypeError(DATA_TYPE_ERROR)

    # Validate subplot titles if provided as a list
    if isinstance(subplot_titles_list, list) and len(subplot_titles_list) < n_subplots:
        raise ValueError(SUBPLOT_TITLES_ERROR.format(n_subplots=n_subplots))

    if n_subplots <= 1:
        msg = "Only one subplot provided. Use `iso_plot` for a single plot. "
        raise ValueError(msg)

    return data_list, subplot_titles_list, n_subplots


def _create_subplots(
    data_list: list[pd.DataFrame],
    axes: np.ndarray,
    n_subplots: int,
    subplot_titles: list[str] | Literal["numbered"] | None,
    x: str,
    y: str,
    plot_layers: Literal["scatter", "density", "simple_density"]
    | Sequence[Literal["scatter", "density", "simple_density"]],
    subplot_title_prefix: str,
    **kwargs,
) -> None:
    """Create individual subplots from the data."""
    for i, dataframe in enumerate(data_list):
        ax = cast("Axes", axes.flatten()[i])

        # Skip if we're beyond the number of subplots
        if i >= n_subplots:
            ax.remove()
            continue

        # Determine subplot title
        if subplot_titles == "numbered":
            sub_title = f"{subplot_title_prefix} {i + 1}"
        elif isinstance(subplot_titles, list):
            sub_title = subplot_titles[i]
        else:
            sub_title = None

        # Create the plot
        iso_plot(
            data=dataframe,
            x=x,
            y=y,
            title=None,
            plot_layers=plot_layers,
            ax=ax,
            **kwargs,
        )

        if sub_title:
            ax.set_title(sub_title, fontsize=12)


def allocate_subplot_axes(
    nrows: int | None, ncols: int | None, n_subplots: int
) -> tuple[int, int, int]:
    """
    Allocate the subplot axes based on the number of data subsets.

    Parameters
    ----------
    nrows : int | None
        Number of rows for subplots. Can be None to auto-determine.
    ncols : int | None
        Number of columns for subplots. Can be None to auto-determine.
    n_subplots : int
        Total number of subplots needed.

    Returns
    -------
    tuple[int, int]
        The number of rows and columns for the subplot grid.

    Notes
    -----
    Logic for determining subplot layout:
    - If both nrows and ncols are specified, use those values
    - If both are None, calculate a grid as close to square as possible
    - If only one is specified, calculate the other to fit all subplots

    """
    # If both dimensions are specified, return them as is
    if nrows is not None and ncols is not None:
        return nrows, ncols, n_subplots

    # If both dimensions are None, create a grid as close to square as possible
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n_subplots)))
        nrows = int(np.ceil(n_subplots / ncols))
    # If only one dimension is specified, calculate the other
    elif nrows is not None and ncols is None:
        ncols = int(np.ceil(n_subplots / nrows))
    elif nrows is None and ncols is not None:
        nrows = int(np.ceil(n_subplots / ncols))
    else:
        msg = (
            "We should never reach this point - both nrows and ncols are None, "
            "but were missed in earlier check."
        )
        raise ValueError(msg)

    n_subplots = nrows * ncols

    return nrows, ncols, n_subplots


def scatter(
    data: pd.DataFrame,
    title: str | None = "Soundscape Scatter Plot",
    ax: Axes | None = None,
    *,
    x: str | None = "ISOPleasant",
    y: str | None = "ISOEventful",
    hue: str | None = None,
    palette: SeabornPaletteType | None = "colorblind",
    legend: Literal["auto", "brief", "full", False] = "auto",
    prim_labels: bool | None = None,  # Alias for primary_labels, deprecated
    **kwargs,
) -> Axes:
    """
    Plot ISOcoordinates as scatter points on a soundscape circumplex grid.

    Creates a scatter plot of data on a standardized circumplex grid with the custom
    Soundscapy styling for soundscape circumplex visualisations.

    Parameters
    ----------
    data
        Input data structure containing coordinate data, typically with ISOPleasant
        and ISOEventful columns.
    x : str, optional
        Column name for x variable, by default "ISOPleasant"
    y : str, optional
        Column name for y variable, by default "ISOEventful"
    title : str | None, optional
        Title to add to circumplex plot, by default "Soundscape Scatter Plot"
    ax : matplotlib.axes.Axes, optional
        Pre-existing matplotlib axes for the plot, by default None
        If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.
    hue : str | np.ndarray | pd.Series | None, optional
        Grouping variable that will produce points with different colors.

        Can be either categorical or numeric,
        although color mapping will behave differently in latter case, by default None
    palette : SeabornPaletteType, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping, by default "colorblind"
    color : ColorType | None, optional
        Color to use for the plot elements when not using hue mapping,
        by default "#0173B2" (first color from colorblind palette)
    figsize : tuple[int, int], optional
        Size of the figure to return if `ax` is None, by default (5, 5)
    s : float, optional
        Size of scatter points, by default 20
    legend : {"auto", "brief", "full", False}, optional
        How to draw the legend. If "brief", numeric hue and size variables will be
        represented with a sample of evenly spaced values. If "full", every group will
        get an entry in the legend. If "auto", choose between brief or full
        representation based on number of levels.

        If False, no legend data is added and no legend is drawn, by default "auto"
    prim_labels : bool | None, optional
        Deprecated. Use xlabel and ylabel parameters instead.

    Other Parameters
    ----------------
    xlabel, ylabel
        Custom axis labels. By default "$P_{ISO}$" and "$E_{ISO}$"
        with math rendering.

        If None is passed, the column names (x and y) will be used as labels.

        If a string is provided, it will be used as the label.

        If False is passed, axis labels will be hidden.
    xlim, ylim : tuple[float, float], optional
        Limits for x and y axes, by default (-1, 1) for both
    legend_loc : MplLegendLocType, optional
        Location of legend, by default "best"
    diagonal_lines : bool, optional
        Whether to include diagonal dimension labels (e.g. calm, etc.),
        by default False
    prim_ax_fontdict : dict, optional
        Font dictionary for axis labels with these defaults:

        {
            "family": "sans-serif",
            "fontstyle": "normal",
            "fontsize": "large",
            "fontweight": "medium",
            "parse_math": True,
            "c": "black",
            "alpha": 1,
        }
    fontsize, fontweight, fontstyle, family, c, alpha, parse_math:
        Direct parameters for font styling in axis labels

    Returns
    -------
        Axes object containing the plot.

    Notes
    -----
    This function applies special styling appropriate for circumplex plots including
    gridlines, axis labels, and proportional axes.

    Examples
    --------
    Basic scatter plot with default settings:

    >>> import soundscapy as sspy
    >>> import matplotlib.pyplot as plt
    >>> data = sspy.isd.load()
    >>> data = sspy.add_iso_coords(data)
    >>> ax = sspy.scatter(data)
    >>> plt.show() # xdoctest: +SKIP

    Scatter plot with grouping by location:

    >>> ax = sspy.scatter(data, hue="LocationID", diagonal_lines=True, legend=False)
    >>> plt.show() # xdoctest: +SKIP
    >>> plt.close('all')

    """
    style_args, subplots_args, kwargs = _setup_style_and_subplots_args_from_kwargs(
        x=x, y=y, prim_labels=prim_labels, kwargs=kwargs
    )

    scatter_args = ScatterParams()
    scatter_args.update(
        data=data,
        x=x,
        y=y,
        palette=palette,
        hue=hue,
        legend=legend,
        extra="allow",
        ignore_null=False,
        **kwargs,
    )  # pass all the rest to scatter

    # Removes the palette if no hue is specified
    scatter_args.crosscheck_palette_hue()

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=subplots_args.figsize)

    p = sns.scatterplot(ax=ax, **scatter_args.as_dict())

    _set_style()
    _circumplex_grid(
        ax=ax,
        **style_args.get_multiple(
            ["xlim", "ylim", "xlabel", "ylabel", "diagonal_lines", "prim_ax_fontdict"]
        ),
    )
    if title is not None:
        _set_circum_title(
            ax=ax,
            title=title,
            xlabel=style_args.get("xlabel"),
            ylabel=style_args.get("ylabel"),
        )
    if legend is not None and hue is not None and style_args.legend_loc is not False:
        _move_legend(ax=ax, new_loc=style_args.get("legend_loc"))
    return p


def density(
    data: pd.DataFrame,
    title: str | None = "Soundscape Density Plot",
    ax: Axes | None = None,
    *,
    x: str | None = "ISOPleasant",
    y: str | None = "ISOEventful",
    hue: str | None = None,
    incl_scatter: bool = True,
    density_type: str = "full",
    palette: SeabornPaletteType | None = "colorblind",
    scatter_kws: dict | None = None,
    legend: Literal["auto", "brief", "full", False] = "auto",
    prim_labels: bool | None = None,  # Alias for primary_labels, deprecated
    **kwargs,
) -> Axes:
    """
    Plot a density plot of ISOCoordinates.

    Creates a kernel density estimate visualization of data distribution on a
    circumplex grid with the custom Soundscapy styling for soundscape circumplex
    visualisations. Can optionally include a scatter plot of the underlying data points.

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure containing coordinate data, typically with ISOPleasant
        and ISOEventful columns.
    title : str | None, optional
        Title to add to circumplex plot, by default "Soundscape Density Plot"
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes object to use for the plot, by default None
        If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.
    x : str, optional
        Column name for x variable, by default "ISOPleasant"
    y : str, optional
        Column name for y variable, by default "ISOEventful"
    hue : str | np.ndarray | pd.Series | None, optional
        Grouping variable that will produce density contours with different colors.
        Can be either categorical or numeric, although color mapping will behave
        differently in latter case, by default None
    incl_scatter : bool, optional
        Whether to include a scatter plot of the data points, by default True
    density_type : {"full", "simple"}, optional
        Type of density plot to draw. "full" uses default parameters, "simple"
        uses a lower number of levels (2), higher threshold (0.5), and lower alpha (0.5)
        for a cleaner visualization, by default "full"
    palette : SeabornPaletteType | None, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping, by default "colorblind"
    scatter_kws : dict | None, optional
        Keyword arguments to pass to `seaborn.scatterplot` if incl_scatter is True,
        by default {"s": 25, "linewidth": 0}
    incl_outline : bool, optional
        Whether to include an outline for the density contours, by default False
    legend : {"auto", "brief", "full", False}, optional
        How to draw the legend. If "brief", numeric hue variables will be
        represented with a sample of evenly spaced values. If "full", every group will
        get an entry in the legend. If "auto", choose between brief or full
        representation based on number of levels.
        If False, no legend data is added and no legend is drawn, by default "auto"
    prim_labels : bool | None, optional
        Deprecated. Use xlabel and ylabel parameters instead.

    **kwargs : dict, optional
        Additional styling parameters:

        - alpha : float, optional
            Proportional opacity of the density fill, by default 0.8
        - fill : bool, optional
            If True, fill in the area between bivariate contours, by default True
        - levels : int | Iterable[float], optional
            Number of contour levels or values to draw contours at, by default 10
        - thresh : float, optional
            Lowest iso-proportional level at which to draw a contour line,
            by default 0.05
        - bw_adjust : float, optional
            Factor that multiplicatively scales the bandwidth, by default 1.2
        - xlabel, ylabel : str | Literal[False], optional
            Custom axis labels. By default, "$P_{ISO}$" and "$E_{ISO}$" with math
            rendering.
            If None is passed, the column names (x and y) will be used as labels.
            If a string is provided, it will be used as the label.
            If False is passed, axis labels will be hidden.
        - xlim, ylim : tuple[float, float], optional
            Limits for x and y axes, by default (-1, 1) for both
        - legend_loc : MplLegendLocType, optional
            Location of legend, by default "best"
        - diagonal_lines : bool, optional
            Whether to include diagonal dimension labels (e.g. calm, etc.),
            by default False
        - figsize : tuple[int, int], optional
            Size of the figure to return if `ax` is None, by default (5, 5)
        - prim_ax_fontdict : dict, optional
            Font dictionary for axis labels with these defaults:

            {
                "family": "sans-serif",
                "fontstyle": "normal",
                "fontsize": "large",
                "fontweight": "medium",
                "parse_math": True,
                "c": "black",
                "alpha": 1,
            }
        - fontsize, fontweight, fontstyle, family, c, alpha, parse_math:
            Direct parameters for font styling in axis labels

        Also accepts additional keyword arguments for matplotlib's contour and contourf
        functions.

    Returns
    -------
        Axes object containing the plot.

    Notes
    -----
    This function will raise a warning if the dataset has fewer than
    RECOMMENDED_MIN_SAMPLES (30) data points, as density plots are not reliable
    with small sample sizes.

    Examples
    --------
    Basic density plot with default settings:

    >>> import soundscapy as sspy
    >>> import matplotlib.pyplot as plt
    >>> data = sspy.isd.load()
    >>> data = sspy.add_iso_coords(data)
    >>> ax = sspy.density(data)
    >>> plt.show() # xdoctest: +SKIP

    Simple density plot with fewer contour levels:

    >>> ax = sspy.density(data, density_type="simple")
    >>> plt.show() # xdoctest: +SKIP

    Density plot with custom styling:

    >>> sub_data = sspy.isd.select_location_ids(
    ...    data, ['CamdenTown', 'PancrasLock', 'RegentsParkJapan', 'RegentsParkFields'])
    >>> ax = sspy.density(
    ...     sub_data,
    ...     hue="SessionID",
    ...     incl_scatter=True,
    ...     legend_loc="upper right",
    ...     fill = False,
    ...     density_type = "simple",
    ... )
    >>> plt.show() # xdoctest: +SKIP

    Add density to existing plots:

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    >>> axes[0] = sspy.density(
    ...     sspy.isd.select_location_ids(data, ['CamdenTown', 'PancrasLock']),
    ...     ax=axes[0], title="CamdenTown and PancrasLock", hue="LocationID",
    ...     density_type="simple"
    ... )
    >>> axes[1] = sspy.density(
    ...     sspy.isd.select_location_ids(data, ['RegentsParkJapan']),
    ...     ax=axes[1], title="RegentsParkJapan"
    ... )
    >>> plt.tight_layout()
    >>> plt.show() # xdoctest: +SKIP
    >>> plt.close('all')

    """
    style_args, subplots_args, kwargs = _setup_style_and_subplots_args_from_kwargs(
        x=x, y=y, prim_labels=prim_labels, kwargs=kwargs
    )

    # Set up density parameters
    density_args = _setup_density_params(
        data=data,
        x=x,
        y=y,
        hue=hue,
        density_type=density_type,
        palette=palette,
        legend=legend,
        **kwargs,
    )

    # Check if dataset is large enough for density plots
    _valid_density(data)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=subplots_args.get("figsize"))

    # Removes the palette if no hue is specified
    if density_args.get("hue") is None:
        density_args.update(palette=None)

    # Set up scatter parameters if needed
    scatter_args = ScatterParams()
    scatter_args.update(
        data=data,
        x=x,
        y=y,
        palette=palette,
        hue=density_args.get("hue"),
        color=density_args.get("color"),
        **(scatter_kws or {}),
    )

    scatter_args.crosscheck_palette_hue()
    density_args.crosscheck_palette_hue()

    if incl_scatter:
        d = sns.scatterplot(ax=ax, **scatter_args.as_seaborn_kwargs())

    if density_type == "simple":
        d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
        d = sns.kdeplot(ax=ax, **density_args.to_outline().as_seaborn_kwargs())

    elif density_type == "full":
        d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
    else:
        raise ValueError

    _set_style()
    _circumplex_grid(
        ax=ax,
        xlim=style_args.get("xlim"),
        ylim=style_args.get("ylim"),
        xlabel=style_args.get("xlabel"),
        ylabel=style_args.get("ylabel"),
        diagonal_lines=style_args.get("diagonal_lines"),
        prim_ax_fontdict=style_args.get("prim_ax_fontdict"),
    )
    if title is not None:
        _set_circum_title(
            ax=ax,
            title=title,
            xlabel=style_args.get("xlabel"),
            ylabel=style_args.get("ylabel"),
        )
    if legend is not None and hue is not None:
        _move_legend(ax=ax, new_loc=style_args.get("legend_loc"))

    return d


def _deal_w_default_labels(
    x: str | None,
    y: str | list[str] | None,
    prim_labels: bool | None,
    style_args: StyleParams,
) -> StyleParams:
    """
    Handle the default labels for the circumplex plot.

    Parameters
    ----------
    x : str or None
        Column name for x variable
    y : str, list[str], or None
        Column name for y variable
    prim_labels : bool or None
        Whether to include the custom primary labels (deprecated)
    style_args : StyleParams
        Style parameters object to update

    Returns
    -------
    StyleParams
        Updated style parameters with appropriate label settings

    """
    # Set default labels based on column names if not already specified
    xlabel = style_args.get("xlabel", x if x is not None else "")
    ylabel = style_args.get(
        "ylabel", y if y is not None and not isinstance(y, list) else ""
    )

    # Handle deprecated prim_labels parameter
    if prim_labels is not None:
        warnings.warn(
            PRIM_LABELS_DEPRECATION_WARNING,
            DeprecationWarning,
            stacklevel=2,
        )
        if prim_labels is False:
            xlabel = False
            ylabel = False

    # Update style args with the determined labels
    style_args.update(xlabel=xlabel, ylabel=ylabel, ignore_null=False)

    return style_args


def _setup_style_and_subplots_args_from_kwargs(
    x: str | None, y: str | list[str] | None, prim_labels: bool | None, kwargs: dict
) -> tuple[StyleParams, SubplotsParams, dict]:
    """
    Set up style and subplot parameters from keyword arguments.

    Parameters
    ----------
    x : str or None
        Column name for x variable
    y : str, list[str], or None
        Column name for y variable
    prim_labels : bool or None
        Whether to include the custom primary labels (deprecated)
    kwargs : dict
        Keyword arguments to process

    Returns
    -------
    tuple
        A tuple containing:
        - style_args: StyleParams object with style settings
        - subplots_args: SubplotsParams object with subplot settings
        - kwargs: remaining keyword arguments with style parameters removed

    """
    # Initialize and update style parameters
    style_args = StyleParams()
    style_args.update(**kwargs, extra="ignore", ignore_null=False)

    # Handle default labels
    style_args = _deal_w_default_labels(
        x=x, y=y, prim_labels=prim_labels, style_args=style_args
    )

    # Initialize and update subplot parameters
    subplots_args = SubplotsParams()
    subplots_args.update(**kwargs, extra="ignore", ignore_null=False)

    # Remove style and scatter args from kwargs to avoid duplicates
    for key in subplots_args.defined_field_names + style_args.defined_field_names:
        if key in kwargs:
            kwargs.pop(key)

    return style_args, subplots_args, kwargs


def create_circumplex_subplots(
    data_list: list[pd.DataFrame],
    plot_type: Literal["density", "scatter", "simple_density"] = "density",
    incl_scatter: bool = True,  # noqa: FBT001, FBT002
    subtitles: list[str] | None = None,
    title: str = "Circumplex Subplots",
    nrows: int | None = None,
    ncols: int | None = None,
    figsize: tuple[int, int] = (10, 10),
    **kwargs: Any,
) -> Figure:
    """
    Create a figure with subplots containing circumplex plots.

    .. deprecated:: 0.8.0
       Use :func:`create_iso_subplots` instead.

    Parameters
    ----------
        data_list : List of DataFrames to plot.
        plot_type : Type of plot to create.
        incl_scatter : Whether to include scatter points on density plots.
        subtitles : List of subtitles for each subplot.
        title : Title for the entire figure.
        nrows : Number of rows in the subplot grid.
        ncols : Number of columns in the subplot grid.
        figsize : Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to pass to scatter_plot or density_plot.

    Returns
    -------
        matplotlib.figure.Figure: A figure containing the subplots.

    Example
    -------
        >>> import pandas as pd
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(42)
        >>> data1 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
        ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
        >>> data2 = pd.DataFrame({'ISOPleasant': np.random.uniform(-1, 1, 50),
        ...                       'ISOEventful': np.random.uniform(-1, 1, 50)})
        >>> fig = create_circumplex_subplots(
        ...     [data1, data2], plot_type="scatter", nrows=1, ncols=2
        ... )
        >>> plt.show() # xdoctest: +SKIP
        >>> isinstance(fig, plt.Figure)
        True
        >>> plt.close('all')

    """
    warnings.warn(
        "The `create_circumplex_subplots` function is deprecated and will be removed "
        "in a future version. Use `create_iso_subplots` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Map plot_type to plot_layers
    if plot_type == "scatter":
        plot_layers = ["scatter"]
    elif plot_type == "density":
        plot_layers = ["density"]
        if incl_scatter:
            plot_layers.insert(0, "scatter")
    elif plot_type == "simple_density":
        plot_layers = ["simple_density"]
        if incl_scatter:
            plot_layers.insert(0, "scatter")
    else:
        warnings.warn(
            "Can't recognize plot type. Using default 'density' plot type with scatter.",
            UserWarning,
            stacklevel=2,
        )
        plot_layers = ["scatter", "density"]

    # Map subtitles to subplot_titles
    subplot_titles = subtitles

    # Call create_iso_subplots with translated parameters
    fig, _ = create_iso_subplots(
        data=data_list,
        subplot_titles=subplot_titles,
        title=title,
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        plot_layers=plot_layers,  # type: ignore[arg-type]
        **kwargs,
    )

    return fig


def jointplot(
    data: pd.DataFrame,
    *,
    x: str = DEFAULT_XCOL,
    y: str = DEFAULT_YCOL,
    title: str | None = "Soundscape Joint Plot",
    hue: str | None = None,
    incl_scatter: bool = True,
    density_type: str = "full",
    palette: SeabornPaletteType | None = "colorblind",
    color: ColorType | None = DEFAULT_COLOR,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    scatter_kws: dict[str, Any] | None = None,
    incl_outline: bool = False,
    alpha: float = DEFAULT_SEABORN_PARAMS["alpha"],
    fill: bool = True,
    levels: int | tuple[float, ...] = 10,
    thresh: float = 0.05,
    bw_adjust: float = DEFAULT_BW_ADJUST,
    legend: Literal["auto", "brief", "full", False] = "auto",
    prim_labels: bool | None = None,  # Alias for primary_labels, deprecated
    joint_kws: dict[str, Any] | None = None,
    marginal_kws: dict[str, Any] | None = None,
    marginal_kind: str = "kde",
    **kwargs,
) -> sns.JointGrid:
    """
    Create a jointplot with a central distribution and marginal plots.

    Creates a visualization with a main plot (density or scatter) in the center and
    marginal distribution plots along the x and y axes. The main plot uses the custom
    Soundscapy styling for soundscape circumplex visualisations, and the marginals show
    the individual distributions of each variable.

    Parameters
    ----------
    data : pd.DataFrame
        Input data structure containing coordinate data, typically with ISOPleasant
        and ISOEventful columns.
    x : str, optional
        Column name for x variable, by default "ISOPleasant"
    y : str, optional
        Column name for y variable, by default "ISOEventful"
    title : str | None, optional
        Title to add to the jointplot, by default "Soundscape Joint Plot"
    hue : str | np.ndarray | pd.Series | None, optional
        Grouping variable that will produce plots with different colors.
        Can be either categorical or numeric, although color mapping will behave
        differently in latter case, by default None
    incl_scatter : bool, optional
        Whether to include a scatter plot of the data points in the joint plot,
        by default True
    density_type : {"full", "simple"}, optional
        Type of density plot to draw. "full" uses default parameters, "simple"
        uses a lower number of levels (2), higher threshold (0.5), and lower alpha (0.5)
        for a cleaner visualization, by default "full"
    palette : SeabornPaletteType | None, optional
        Method for choosing the colors to use when mapping the hue semantic.
        String values are passed to seaborn.color_palette().
        List or dict values imply categorical mapping, while a colormap object
        implies numeric mapping, by default "colorblind"
    color : ColorType | None, optional
        Color to use for the plot elements when not using hue mapping,
        by default "#0173B2" (first color from colorblind palette)
    figsize : tuple[int, int], optional
        Size of the figure to create (determines height, width is proportional),
        by default (5, 5)
    scatter_kws : dict[str, Any] | None, optional
        Additional keyword arguments to pass to scatter plot if incl_scatter is True,
        by default None
    incl_outline : bool, optional
        Whether to include an outline for the density contours, by default False
    alpha : float, optional
        Opacity level for the density fill, by default 0.8
    fill : bool, optional
        Whether to fill the density contours, by default True
    levels : int | Iterable[float], optional
        Number of contour levels or specific levels to draw. A vector argument
        must have increasing values in [0, 1], by default 10
    thresh : float, optional
        Lowest iso-proportion level at which to draw contours, by default 0.05
    bw_adjust : float, optional
        Factor that multiplicatively scales the bandwidth. Increasing will make
        the density estimate smoother, by default 1.2
    legend : {"auto", "brief", "full", False}, optional
        How to draw the legend for hue mapping, by default "auto"
    prim_labels : bool | None, optional
        Deprecated. Use xlabel and ylabel parameters instead.
    joint_kws : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the joint plot, by default None
    marginal_kws : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the marginal plots,
        by default {"fill": True, "common_norm": False}
    marginal_kind : {"kde", "hist"}, optional
        Type of plot to draw in the marginal axes, either "kde" for kernel
        density estimation or "hist" for histogram, by default "kde"

    **kwargs : dict, optional
        Additional styling parameters:

        - xlabel, ylabel : str | Literal[False], optional
            Custom axis labels. By default "$P_{ISO}$" and "$E_{ISO}$" with
            math rendering.

            If None is passed, the column names (x and y) will be used as labels.

            If a string is provided, it will be used as the label.

            If False is passed, axis labels will be hidden.
        - xlim, ylim : tuple[float, float], optional
            Limits for x and y axes, by default (-1, 1) for both
        - legend_loc : MplLegendLocType, optional
            Location of legend, by default "best"
        - diagonal_lines : bool, optional
            Whether to include diagonal dimension labels (e.g. calm, etc.),
            by default False
        - prim_ax_fontdict : dict, optional
            Font dictionary for axis labels with these defaults:

            {
                "family": "sans-serif",
                "fontstyle": "normal",
                "fontsize": "large",
                "fontweight": "medium",
                "parse_math": True,
                "c": "black",
                "alpha": 1,
            }

    Returns
    -------
    sns.JointGrid
        The seaborn JointGrid object containing the plot

    Notes
    -----
    This function will raise a warning if the dataset has fewer than
    RECOMMENDED_MIN_SAMPLES (30) data points, as density plots are not reliable
    with small sample sizes.

    Examples
    --------
    Basic jointplot with default settings:

    >>> import soundscapy as sspy
    >>> import matplotlib.pyplot as plt
    >>> data = sspy.isd.load()
    >>> data = sspy.add_iso_coords(data)
    >>> g = sspy.jointplot(data)
    >>> plt.show() # xdoctest: +SKIP

    Jointplot with histogram marginals:

    >>> g = sspy.jointplot(data, marginal_kind="hist")
    >>> plt.show() # xdoctest: +SKIP

    Jointplot with custom styling and grouping:

    >>> g = sspy.jointplot(
    ...     data,
    ...     hue="LocationID",
    ...     incl_scatter=True,
    ...     density_type="simple",
    ...     diagonal_lines=True,
    ...     figsize=(6, 6),
    ...     title="Grouped Soundscape Analysis"
    ... )
    >>> plt.show() # xdoctest: +SKIP
    >>> plt.close('all')

    """
    # Check if dataset is large enough for density plots
    _valid_density(data)

    style_args, subplots_args, kwargs = _setup_style_and_subplots_args_from_kwargs(
        x=x, y=y, prim_labels=prim_labels, kwargs=kwargs
    )

    # Initialize default dicts if None
    scatter_args = ScatterParams()
    scatter_args.update(**scatter_kws) if scatter_kws is not None else None

    joint_kws = {} if joint_kws is None else joint_kws
    marginal_kws = (
        {"fill": True, "common_norm": False} if marginal_kws is None else marginal_kws
    )

    if density_type == "simple":
        thresh = DEFAULT_SIMPLE_DENSITY_PARAMS["thresh"]
        levels = DEFAULT_SIMPLE_DENSITY_PARAMS["levels"]
        alpha = DEFAULT_SIMPLE_DENSITY_PARAMS["alpha"]
        incl_outline = True

    # Handle hue and color
    if hue is None:
        # Removes the palette if no hue is specified
        palette = None
        color = sns.color_palette("colorblind", 1)[0] if color is None else color

    # Create the joint grid
    g = sns.JointGrid(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        # height=figsize[0],  # Use figsize for height
        xlim=style_args.xlim,
        ylim=style_args.ylim,
    )

    # Add the density plot to the joint plot area
    density(
        data,
        x=x,
        y=y,
        incl_scatter=incl_scatter,
        density_type=density_type,
        title=None,  # We'll set the title separately
        ax=g.ax_joint,
        hue=hue,
        palette=palette,
        color=color,
        scatter_kws=scatter_kws,
        incl_outline=incl_outline,
        legend_loc=style_args.legend_loc,
        alpha=alpha,
        legend=legend,
        fill=fill,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        diagonal_lines=style_args.diagonal_lines,
        xlim=style_args.xlim,
        ylim=style_args.ylim,
        **joint_kws,
    )

    # Add the marginal plots
    if marginal_kind == "hist":
        sns.histplot(
            data=data,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            binrange=style_args.xlim,
            legend=False,
            **marginal_kws,
        )
        sns.histplot(
            data=data,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            binrange=style_args.ylim,
            legend=False,
            **marginal_kws,
        )
    elif marginal_kind == "kde":
        sns.kdeplot(
            data=data,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kws,
        )
        sns.kdeplot(
            data=data,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kws,
        )

    # Set title
    if title is not None:
        g.ax_marg_x.set_title(title, pad=6.0)

    _set_style()
    _circumplex_grid(
        ax=g.ax_joint,
        xlim=style_args.get("xlim"),
        ylim=style_args.get("ylim"),
        xlabel=style_args.get("xlabel"),
        ylabel=style_args.get("ylabel"),
        diagonal_lines=style_args.get("diagonal_lines"),
        prim_ax_fontdict=style_args.get("prim_ax_fontdict"),
    )

    if legend is not None and hue is not None:
        _move_legend(ax=g.ax_joint, new_loc=style_args.get("legend_loc"))

    return g


def _pop_style_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Remove style parameters from kwargs dictionary.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments

    Returns
    -------
    dict[str, Any]
        Dictionary with style parameters removed

    """
    # Get style params / kwargs
    kwargs.pop("xlabel", DEFAULT_STYLE_PARAMS["xlabel"])
    kwargs.pop("ylabel", DEFAULT_STYLE_PARAMS["ylabel"])
    kwargs.pop("xlim", DEFAULT_STYLE_PARAMS["xlim"])
    kwargs.pop("ylim", DEFAULT_STYLE_PARAMS["ylim"])
    kwargs.pop("legend_loc", DEFAULT_STYLE_PARAMS["legend_loc"])
    kwargs.pop("diagonal_lines", DEFAULT_STYLE_PARAMS["diagonal_lines"])

    # Pull out any fontdict options which might be loose in the kwargs
    kwargs.pop("prim_ax_fontdict", DEFAULT_XY_LABEL_FONTDICT.copy())
    for key in DEFAULT_XY_LABEL_FONTDICT:
        if key in kwargs:
            kwargs.pop(key)

    return kwargs


def _move_legend(
    ax: Axes,
    new_loc: MplLegendLocType,
    **kwargs,
) -> None:
    """
    Move legend to desired relative location.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    new_loc : MplLegendLocType
        The location of the legend

    """
    old_legend = ax.get_legend()
    if old_legend is None:
        logger.debug("_move_legend: No legend found for axis.")
        return
    handles = [h for h in old_legend.legend_handles if isinstance(h, Artist | tuple)]
    if not handles:
        logger.debug("_move_legend: No legend handles found.")
        return

    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    if len(handles) != len(labels):
        labels = labels[: len(handles)]
    ax.legend(handles, labels, loc=new_loc, title=title, **kwargs)


def _set_style() -> None:
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})


def _circumplex_grid(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str | Literal[False],
    ylabel: str | Literal[False],
    *,
    line_weights: float = 1.5,
    diagonal_lines: bool = False,
    prim_ax_fontdict: dict[str, str] | None = DEFAULT_XY_LABEL_FONTDICT,
) -> None:
    """
    Create the base layer grids and label lines for the soundscape circumplex.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        plt subplot Axes to add the circumplex grids to
    xlabel: str | Literal[False]
        Label for the x-axis, can be set to False to omit.
    ylabel: str | Literal[False]
        Label for the y-axis, can be set to False to omit.
    prim_labels: bool, optional
        flag for whether to include the custom primary labels ISOPleasant and ISOEventful
            by default True
        If using your own x and y names, you should set this to False.
    diagonal_lines : bool, optional
        flag for whether the include the diagonal dimensions (calm, etc)
            by default False

    """  # noqa: E501
    # Setting up the grids
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # grids and ticks
    ax.get_xaxis().set_minor_locator(AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(AutoMinorLocator())

    ax.grid(visible=True, which="major", color="grey", alpha=0.5)
    ax.grid(
        visible=True,
        which="minor",
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.4,
        zorder=DEFAULT_STYLE_PARAMS["prim_lines_zorder"],
    )

    _primary_lines_and_labels(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        line_weights=line_weights,
        prim_ax_fontdict=prim_ax_fontdict,
    )
    if diagonal_lines:
        _diagonal_lines_and_labels(ax, line_weights=line_weights)


def _set_circum_title(
    ax: Axes, title: str, xlabel: str | Literal[False], ylabel: str | Literal[False]
) -> None:
    """
    Set the title for the circumplex plot.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    prim_labels: bool, optional
        Whether to include the custom primary labels ISOPleasant and ISOEventful
        by default True
        If using your own x and y names, you should set this to False.
    title : str | None
        Title to set

    """
    title_pad = 30.0 if (xlabel is not False or ylabel is not False) else 6.0
    ax.set_title(title, pad=title_pad, fontsize=DEFAULT_STYLE_PARAMS["title_fontsize"])


def _primary_lines_and_labels(
    ax: Axes,
    xlabel: str | Literal[False],
    ylabel: str | Literal[False],
    line_weights: float,
    prim_ax_fontdict: dict[str, str] | None = DEFAULT_XY_LABEL_FONTDICT,
) -> None:
    import re

    # Add lines and labels for circumplex model
    ## Primary Axes
    ax.axhline(  # Horizontal line
        y=0,
        color="grey",
        linestyle="dashed",
        alpha=1,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["prim_lines_zorder"],
    )
    ax.axvline(  # vertical line
        x=0,
        color="grey",
        linestyle="dashed",
        alpha=1,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["prim_lines_zorder"],
    )

    # Add labels for circumplex model
    # Check for math mode in labels
    if (re.search(r"\$.*\$", xlabel) if isinstance(xlabel, str) else False) or (
        re.search(r"\$.*\$", ylabel) if isinstance(ylabel, str) else False
    ):
        logger.warning(
            "parse_math is set to True, but $ $ indicates a math label. "
            "This may cause issues with the circumplex plot."
        )

    ax.set_xlabel(
        xlabel, fontdict=prim_ax_fontdict
    ) if xlabel is not False else ax.xaxis.label.set_visible(False)

    ax.set_ylabel(
        ylabel, fontdict=prim_ax_fontdict
    ) if ylabel is not False else ax.yaxis.label.set_visible(False)


def _diagonal_lines_and_labels(ax: Axes, line_weights: float) -> None:
    diag_ax_font = {
        "fontstyle": "italic",
        "fontsize": "small",
        "fontweight": "bold",
        "c": "black",
        "alpha": 0.5,
    }
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.plot(  # uppward diagonal
        [x_lim[0], x_lim[1]],
        [y_lim[0], y_lim[1]],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["diag_lines_zorder"],
    )
    ax.plot(  # downward diagonal
        [x_lim[0], x_lim[1]],
        [y_lim[1], y_lim[0]],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=DEFAULT_STYLE_PARAMS["diag_lines_zorder"],
    )

    ### Labels
    ax.text(  # Vibrant label
        x=x_lim[1] / 2,
        y=y_lim[1] / 2,
        s="(vibrant)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )
    ax.text(  # Chaotic label
        x=x_lim[0] / 2,
        y=y_lim[1] / 2,
        s="(chaotic)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )
    ax.text(  # monotonous label
        x=x_lim[0] / 2,
        y=y_lim[0] / 2,
        s="(monotonous)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )
    ax.text(  # calm label
        x=x_lim[1] / 2,
        y=y_lim[0] / 2,
        s="(calm)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=DEFAULT_STYLE_PARAMS["diag_labels_zorder"],
    )


def iso_annotation(
    ax: Axes,
    data: pd.DataFrame,
    location: str,
    *,
    x_adj: int = 0,
    y_adj: int = 0,
    x_key: str = DEFAULT_XCOL,
    y_key: str = DEFAULT_YCOL,
    ha: str = "center",
    va: str = "center",
    fontsize: str = "small",
    arrowprops: dict | None = None,
    **text_kwargs,
) -> None:
    """
    Add text annotations to circumplex plot based on coordinate values.

    Directly uses plt.annotate

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        existing plt axes to add to
    data : pd.Dataframe
        dataframe of coordinate points
    location : str
        name of the coordinate to plot
    x_adj : int, optional
        value to adjust x location by, by default 0
    y_adj : int, optional
        value to adjust y location by, by default 0
    x_key : str, optional
        name of x column, by default "ISOPleasant"
    y_key : str, optional
        name of y column, by default "ISOEventful"
    ha : str, optional
        horizontal alignment, by default "center"
    va : str, optional
        vertical alignment, by default "center"
    fontsize : str, optional
        by default "small"
    arrowprops : dict, optional
        dict of properties to send to plt.annotate,
        by default dict(arrowstyle="-", ec="black")

    """
    if arrowprops is None:
        arrowprops = {"arrowstyle": "-", "ec": "black"}

    # noinspection PyTypeChecker
    ax.annotate(
        text=data["LocationID"][location],
        xy=(
            data[x_key][location],
            data[y_key][location],
        ),
        xytext=(
            data[x_key][location] + x_adj,
            data[y_key][location] + y_adj,
        ),
        ha=ha,
        va=va,
        arrowprops=arrowprops,
        annotation_clip=True,
        fontsize=fontsize,
        **text_kwargs,
    )


def scatter_plot(*args, **kwargs) -> Axes:  # noqa: ANN002
    """
    Wrapper for the scatter function to maintain backwards compatibility.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the scatter function.
    **kwargs : dict
        Keyword arguments to pass to the scatter function.

    Returns
    -------
    Axes
        The Axes object containing the plot.

    """  # noqa: D401
    warnings.warn(
        "The `scatter_plot` function is deprecated and will be removed in a "
        "future version. Use `scatter` instead."
        "\nAs of v0.8, `scatter_plot` is an alias for `scatter` and does not maintain "
        "full backwards compatibility with v0.7. It may work, or some arguments may "
        "fail.",
        DeprecationWarning,
        stacklevel=2,
    )
    if kwargs.pop("backend", None) is not None:
        warnings.warn(
            "`Backend` is no longer supported in the scatter_plot function (v0.8+).",
            DeprecationWarning,
            stacklevel=2,
        )

    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("backend", "show_labels", "apply_styling")
    }

    return scatter(*args, **kwargs)


def density_plot(*args, **kwargs) -> Axes | np.ndarray | ISOPlot:  # noqa: ANN002
    """
    Wrapper for the density function to maintain backwards compatibility.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the density function.
    **kwargs : dict
        Keyword arguments to pass to the density function.

    Returns
    -------
    Axes
        The Axes object containing the plot.

    """  # noqa: D401
    warnings.warn(
        "The `density_plot` function is deprecated and will be removed in a "
        "future version. Use `density` instead."
        "\nAs of v0.8, `density_plot` is an alias for `density` and does not maintain "
        "full backwards compatibility with v0.7. It may work, or some arguments may "
        "fail.",
        DeprecationWarning,
        stacklevel=2,
    )
    if kwargs.pop("backend", None) is not None:
        warnings.warn(
            "`Backend` is no longer supported in the density_plot function (v0.8+).",
            DeprecationWarning,
            stacklevel=2,
        )
    filtered_args = [a for a in args if not isinstance(a, Backend)]

    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in (
            "backend",
            "show_labels",
            "apply_styling",
            "simple_density",
            "simple_density_thresh",
            "simple_density_levels",
            "simple_density_alpha",
        )
    }

    # Convert simple_density parameters to the new API if they exist
    if "density_type" not in kwargs and kwargs.pop("simple_density", False):
        kwargs["density_type"] = "simple"

    return density(*filtered_args, **kwargs)


def _setup_density_params(
    data: pd.DataFrame,
    x: str | None,
    y: str | None,
    density_type: str,
    hue: str | None,
    palette: SeabornPaletteType | None,
    legend: Literal["auto", "brief", "full", False],
    **kwargs,
) -> DensityParams:
    """
    Set up density parameters based on density type.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be plotted
    x : str or None
        Column name for x variable
    y : str or None
        Column name for y variable
    density_type : str
        Type of density plot ("simple" or "full")
    palette : SeabornPaletteType or None
        Color palette for the plot
    legend : "auto", "brief", "full", or False
        How to draw the legend
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    SimpleDensityParams
        Parameter object with density settings

    """
    if density_type == "simple":
        density_args = cast("DensityParams", SimpleDensityParams())
    else:
        density_args = DensityParams()

    density_args.update(
        data=data,
        x=x,
        y=y,
        palette=palette,
        hue=hue,
        legend=legend,
        extra="allow",
        ignore_null=False,
        **kwargs,
    )

    return density_args


def _valid_density(data: pd.DataFrame) -> None:
    """
    Check if the data is valid for density plots.

    Raises a warning if the dataset is too small for reliable density estimation.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be checked

    Raises
    ------
    UserWarning
        If the data is too small for density plots (< RECOMMENDED_MIN_SAMPLES).

    """
    if len(data) < RECOMMENDED_MIN_SAMPLES:
        warnings.warn(
            "Density plots are not recommended for "
            f"small datasets (<{RECOMMENDED_MIN_SAMPLES} samples).",
            UserWarning,
            stacklevel=2,
        )
