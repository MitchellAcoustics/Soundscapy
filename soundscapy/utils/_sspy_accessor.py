"""Soundscapy Dataframe Accessor

.. deprecated:: 0.5.0
    The SSPY Accessor is deprecated. Please refer to the `soundscapy.surveys` and `soundscapy.plotting` modules.

This module contains the Soundscapy Dataframe Accessor, which is used to add methods to Pandas Dataframes.
This method is used to simplify the direct manipulation of soundscape survey-type data, allowing methods to be chained.
The accessor is added to the dataframe by importing the module, and is accessed by using the `sspy` attribute of the dataframe.
The sspy accessor is intended to be general to any soundscape survey data, functions for specific survey types are contained in the `isd` and `araus` modules.
"""

# Add soundscapy to the Python path
from datetime import date
from typing import Union, Tuple, List, Dict

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

import soundscapy.databases.isd
import soundscapy.plotting.circumplex as sspyplot
import soundscapy.plotting.likert
import soundscapy.utils.surveys as db

# Define the names of the PAQ columns


# Default plot settings
diag_lines_zorder = 1
diag_labels_zorder = 4
prim_lines_zorder = 2
data_zorder = 3
default_bw_adjust = 1.2


@register_dataframe_accessor("sspy")
class SSPYAccessor:
    """Soundscapy Dataframe Accessor

    This class is used to add methods to Pandas Dataframes.
    This method is used to simplify the direct manipulation of soundscape survey-type data, allowing methods to be chained.
    The accessor is added to the dataframe by importing the module, and is accessed by using the `sspy` attribute of the dataframe.
    The sspy accessor is intended to be general to any soundscape survey data, functions for specific survey types are
    contained in the `isd` and `araus` modules.
    """

    def __init__(self, df):
        self._df = df
        self._analysis_date = date.today().isoformat()
        self._metadata = {}

    def validate_dataset(
        self,
        paq_aliases: Union[List, Dict] = None,
        allow_lockdown: bool = True,
        allow_na: bool = False,
        verbose: int = 1,
        val_range: Tuple = (1, 5),
    ):
        """Performs data quality checks and validates that the dataset fits the expected format

        Parameters
        ----------
        df : pd.DataFrame
            ISD style dataframe, incl PAQ data
        paq_aliases : list or dict, optional
            list of PAQ names (in order)
            or dict of PAQ names with new names as values, by default None
        allow_lockdown : bool, optional
            if True will keep Lockdown data in the df, by default True
        allow_paq_na : bool, optional
            remove rows which have any missing PAQ values
            otherwise will remove those with 50% missing, by default False    verbose : int, optional
            how much info to print while running, by default 1
        val_range : tuple, optional
            min and max range of the PAQ response values, by default (5, 1)

        Returns
        -------
        tuple
            cleaned dataframe, dataframe of excluded samples

        See Also
        --------
        :func:`soundscapy.database.validate_dataset`
        """
        raise DeprecationWarning(
            "The SSPY Accessor is deprecated. Please refer to the `soundscapy.surveys` and `soundscapy.plotting` modules."
        )
        # return soundscapy.databases.isd.validate_dataset(
        #     self._df, paq_aliases, allow_lockdown, allow_na, verbose, val_range
        # )

    def paq_data_quality(self, verbose=0):
        """
        Check the data quality of the PAQs in the dataframe

        Parameters
        ----------
        verbose : int, optional
            How much info to print, by default 0

        Returns
        -------
        pd.DataFrame
            Dataframe of the data quality checks

        See Also
        --------
        :func:`soundscapy.database.likert_data_quality`
        """
        return db.likert_data_quality(self._df, verbose)

    def filter(self, filter_by, condition):
        """Filter the dataframe by a condition

        Parameters
        ----------
        filter_by : str
            Column to filter by
        condition : str
            Condition to filter by

        Returns
        -------
        pd.DataFrame
        """
        if isinstance(condition, str):
            return self._df.query(f"{filter_by} == @condition")
        elif isinstance(condition, (list, tuple)):
            return self._df.query(f"{filter_by} in @condition")

    # TODO: add mean_responses method

    def convert_column_to_index(self, col, drop=False):
        """Convert a column to the index of the dataframe

        Parameters
        ----------
        col : str
            Column to convert to index
        drop : bool, optional
            Drop the column after conversion, by default False

        Returns
        -------
        pd.DataFrame
            Dataframe with the column converted to the index

        See Also
        --------
        :func:`soundscapy.database.convert_column_to_index`
        """
        return db.convert_column_to_index(self._df, col, drop)

    def return_paqs(self, incl_ids=True, other_cols=None):
        """Return only the PAQ columns

        Parameters
        ----------
        incl_ids : bool, optional
            whether to include ID cols too (i.e. RecordID, GroupID, etc), by default True
        other_cols : list, optional
            other columns to also include, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe with only the PAQ columns

        See Also
        --------
        :func:`soundscapy.database.return_paqs`
        """
        return db.return_paqs(self._df, incl_ids, other_cols)

    def add_paq_coords(
        self,
        scale_to_one: bool = True,
        val_range=(5, 1),
        projection: bool = True,
        names=("ISOPleasant", "ISOEventful"),
    ):
        """Calculate and add ISO coordinates as new columns in dataframe

        Calls `calculate_paq_coords()`

        Parameters
        ----------
        scale_to_one : bool, optional
            Should the coordinates be scaled to (-1, +1), by default True
        val_range: tuple, optional
            (max, min) range of original PAQ responses, by default (5, 1)
        projection : bool, optional
            Use the trigonometric projection (cos(45)) term for diagonal PAQs, by default True
        names : list, optional
            Names for new coordinate columns, by default ["ISOPleasant", "ISOEventful"]

        Returns
        -------
        pd.DataFrame
            Dataframe with new columns added

        See Also
        --------
        :func:`soundscapy.database.calculate_paq_coords`
        """
        raise DeprecationWarning(
            "The SSPY Accessor has been deprecated. Please use `soundscapy.surveys.add_iso_coords()` instead."
        )

    def soundscapy_describe(self, group_by, type="percent"):
        """Describe the dataframe by a column

        Parameters
        ----------
        group_by : str
            Column to group by
        type : str, optional
            Type of summary to return, by default "percent"

        Returns
        -------
        pd.DataFrame
            Dataframe of summary stats

        See Also
        --------
        :func:`soundscapy.database.soundscapy_describe`
        """
        res = {
            location: self.location_describe(location, type=type)
            for location in self._df[group_by].unique()
        }

        res = pd.DataFrame.from_dict(res, orient="index")
        return res

    def paq_radar(self, ax=None, index=None):
        """Plot a radar chart of the PAQs

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axis to plot on, by default None
        index : int, optional
            Index of the row to plot, by default None

        Returns
        -------
        matplotlib.axes
            Axis with the plot

        See Also
        --------
        :func:`soundscapy.database.paq_radar`
        """
        return soundscapy.plotting.likert.paq_radar_plot(self._df, ax, index)

    def scatter(
        self,
        x="ISOPleasant",
        y="ISOEventful",
        title="Soundscape Scatter Plot",
        diagonal_lines=False,
        xlim=(-1, 1),
        ylim=(-1, 1),
        figsize=(5, 5),
        legend_loc="lower left",
        hue=None,
        style=None,
        s=10,
        palette="colorblind",
        hue_order=None,
        hue_norm=None,
        sizes=None,
        size_order=None,
        size_norm=None,
        markers=True,
        style_order=None,
        alpha=None,
        legend="auto",
        ax=None,
        **scatter_kwargs,
    ):
        """Plot ISOcoordinates as scatter points on a soundscape circumplex grid

        Makes use of seaborn.scatterplot


         - Soundscapy specific parameters -
        We have made all of the `seaborn.scatterplot` arguments available, but have also added or changed some specific
        options for circumplex plotting.


        Parameters
        ----------
        x : vector or key in `data`, optional
            column name for x variable, by default "ISOPleasant"
        y : vector or key in `data`, optional
            column name for y variable, by default "ISOEventful"
        title : str, optional
            Title to add to circumplex plot, by default "Soundscape Scatter Plot"
        diagonal_lines : bool, optional
            whether to include diagonal dimension labels (e.g. calm, etc.), by default False
        xlim, ylim : tuple, optional
            Limits of the circumplex plot, by default (-1, 1)
            It's recommended to set these such that the x and y axes have the same aspect
        figsize : tuple, optional
            Size of the figure to return if `ax` is None, by default (5, 5)
        legend_loc : str, optional
            relative location of legend, by default "lower left"
        palette : string, list, dict or matplotlib.colors.Colormap, optional
            Method for choosing the colors to use when mapping the hue semantic. String values are passed to
            seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object
            implies numeric mapping.
            by default colorblind
        s : int, optional
            size of scatter points, by default 10
        hue : vector or key in data, optional
            Grouping variable that will produce points with different colors. Can be either categorical or numeric,
            although color mapping will behave differently in latter case, by default None
        style : vector or key in data
            Grouping variable that will produce points with different markers. Can have a numeric dtype but will always
            be treated as categorical.
        hue_order : vector of strings
            Specify the order of processing and plotting for categorical levels of the `hue` semantic
        hue_norm : tuple or matplotlib.colors.Normalize
            Either a pair of values that set the normalization range in data units or an object that will map from data
            units into a [0, 1] interval. Usage implies numeric mapping.
        sizes : list, dict, or tuple
            An object that determines how sizes are chosen when `size` is used. It can always be a list of size values or
            a dict mapping levels of the `size` variable to sizes. When `size` is numeric, it can also be a tuple
            specifying the minimum and maximum size to use such that other values are normalized within this range.
        size_order : list
            Specified order for appearance of the `size` variable levels, otherwise they are determined from the data. Not
            relevant when the `size` variable is numeric.
        size_norm : tuple or Normalization object
            Normalization in data units for scaling plot objects when the `size` variable is numeric.
        markers : boolean, list, or dictionary
            Object determining how to draw the markers for different levels of the `style` variable. Setting to `True` will
            use default markers, or you can pass a list of markers or a dictionary mapping levels of the `style` variable
            to markers. Setting to `False` will draw marker-less lines. Markers are specified as in matplotlib.
        style_order : list
            Specified order for appearance of the `style` variable levels otherwise they are determined from the data. Not
            relevant when the `style` variable is numeric.
        alpha : float
            Proportional opacity of the points.
        legend : {"auto", "brief", "full" or False}, optional
            How to draw the legend. If “brief”, numeric hue and size variables will be represented with a sample of evenly
            spaced values. If “full”, every group will get an entry in the legend. If “auto”, choose between brief or full
            representation based on number of levels. If False, no legend data is added and no legend is drawn.
            by default, "auto"
        ax : matplotlib.axes.Axes, optional
            Pre-existing matplotlib axes for the plot, by default None
            If `None` call `matplotlib.pyplot.subplots` with `figsize` internally.

        Returns
        -------
        matplotlib.axes.Axes

        See Also
        --------
        :func:`soundscapy.plotting.circumplex.scatter`

        `seaborn.kdeplot <https://seaborn.pydata.org/generated/seaborn.kdeplot.html>`_
        """
        return sspyplot.scatter(
            self._df,
            x=x,
            y=y,
            title=title,
            diagonal_lines=diagonal_lines,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            legend_loc=legend_loc,
            hue=hue,
            style=style,
            s=s,
            palette=palette,
            hue_order=hue_order,
            hue_norm=hue_norm,
            sizes=sizes,
            size_order=size_order,
            size_norm=size_norm,
            markers=markers,
            style_order=style_order,
            alpha=alpha,
            legend=legend,
            ax=ax,
            **scatter_kwargs,
        )

    def density(
        self,
        x: str = "ISOPleasant",
        y: str = "ISOEventful",
        incl_scatter: bool = True,
        density_type: str = "full",
        title: str = "Soundscapy Density Plot",
        diagonal_lines: bool = False,
        xlim: tuple = (-1, 1),
        ylim: tuple = (-1, 1),
        scatter_kws: dict = dict(s=20),
        incl_outline: bool = False,
        figsize: tuple = (5, 5),
        legend_loc: str = "lower left",
        alpha: float = 0.75,
        gridsize: int = 200,
        kernel: str = None,
        cut: Union[float, int] = 3,
        clip: Tuple[int] = None,
        legend: bool = False,
        cumulative: bool = False,
        cbar: bool = False,
        cbar_ax: matplotlib.axes.Axes = None,
        cbar_kws: dict = None,
        ax: matplotlib.axes.Axes = None,
        weights: str = None,
        hue: str = None,
        palette="colorblind",
        hue_order: List[str] = None,
        hue_norm=None,
        multiple: str = "layer",
        common_norm: bool = False,
        common_grid: bool = False,
        levels: int = 10,
        thresh: float = 0.05,
        bw_method="scott",
        bw_adjust: Union[float, int] = None,
        log_scale: Union[bool, int, float] = None,
        color: str = "blue",
        fill: bool = True,
        data2: Union[pd.DataFrame, np.ndarray] = None,
        warn_singular: bool = True,
        **kwargs,
    ):  # sourcery skip: default-mutable-arg
        """Plot a density plot of ISOCoordinates.

        Creates a wrapper around `seaborn.kdeplot` and adds functionality and styling to customise it for circumplex plots.
        The density plot is a combination of a kernel density estimate and a scatter plot.

         - Soundscapy specific parameters -
        We have made all of the `seaborn.scatterplot` arguments available, but have also added or changed some specific
        options for circumplex plotting.

        Parameters
        ----------
        data : pd.DataFrame, np.ndarray, mapping or sequence
            Input data structure. Either a long-form collection of vectors that can be assigned to
            named variables or a wide-form dataset that will be internally reshaped.
        x : vector or key in `data`, optional
            Column name for x variable, by default "ISOPleasant"
        y : vector or key in `data`, optional
            Column name for y variable, by default "ISOEventful"
        incl_scatter : bool, optional
            Whether to include a scatter plot of the data, by default True
        density_type : {"full", "simple"}, optional
            Type of density plot to draw, by default "full"
        title : str, optional
            Title to add to circumplex plot, by default "Soundscapy Density Plot"
        diagonal_lines : bool, optional
            Whether to include diagonal dimension labels (e.g. calm, etc.), by default False
        xlim, ylim : tuple, optional
            Limits of the circumplex plot, by default (-1, 1)
            It's recommended to set these such that the x and y axes have the same aspect
        scatter_kws : dict, optional
            Keyword arguments to pass to `seaborn.scatterplot`, by default dict(s=25, linewidth=0)
        incl_outline : bool, optional
        figsize : tuple, optional
            Size of the figure to return if `ax` is None, by default (5, 5)
        legend_loc : str, optional
            Relative location of legend, by default "lower left"
        alpha : float, optional
            Proportional opacity of the heatmap fill, by default 0.75
        gridsize : int, optional
            Nuber of points on each dimension of the evaluation grid, by default 200
        kernel : str, optional
            Function that defines the kernel, by default None
        cut : Union[float, int], optional
            Factor, multiplied by the smoothing bandwidth, that determines how far the evaluation grid extends past the
            extreme datapoints. When set to 0, truncate the curve at the data limits, by default 3
        clip : tuple[int], optional
            Do not evaluate the density outside of these limits, by default None
        legend : bool, optional
            If False, suppress the legend for semantic variables, by default True
        cumulative : bool, optional
            If True, estimate a cumulative distribution function, by default False
        cbar : bool, optional
            If True, add a colorbar to annotate the color mapping in a bivariate plot. Note: does not currently support
            plots with a `hue` variable well, by default False
        cbar_ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the colorbar, by default None
        cbar_kws : dict, optional
            Keyword arguments for the colorbar, by default None
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes object to use for the plot, by default None
        weights : vector or key in `data`, optional
            If provided, weight the kernel density estimation using these values, by default None
        hue : vector or key in `data`, optional
            Semantic variable that is mapped to determine the color of plot elements, by default None
        palette : Union[str, list, dict, matplotlib.colors.Colormap], optional
            Method for choosing the colors to use when mapping the hue semantic. String values are passed to
            seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object
            implies numeric mapping.
            by default colorblind
        hue_order : list[str], optional
            Specify the order of processing and plotting for categorical levels of the `hue` semantic, by default None
        hue_norm : Union[tuple, matplotlib.colors.Normalize], optional
            Either a pair of values that set the normalization range in data units or an object , by default None
        multiple : {"layer", "stack", "fill"}, optional
            Whether to plot multiple elements when semantic mapping creates subsets. Only relevant with univariate data,
            by default 'layer'
        common_norm : bool, optional
            If True, scale each conditional density by the number of observations such that the total area under all
            densities sums to 1. Otherwise, normalize each density independently, by default False
        common_grid : bool, optional
            If True, use the same evaluation grid for each kernel density estimate. Only relevant with univariate data.
            by default, False
        levels : int or vector, optional
            Number of contour levels or values to draw contours at. A vector argument must have increasing values in [0, 1].
             Levels correspond to iso-proportions of the density: e.g., 20% of the probability mass will lie below the
             contour drawn for 0.2. Only relevant with bivariate data.
             by default, 10
        thresh : number in [0, 1]
            Lowest iso-proportion level at which to draw a contour line. Ignored with `levels` is a vector. Only relevant
            with bivariate data.
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to `scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
            See Notes.
        log_scale : bool or number, or pair of bools or numbers, optional
            Set axis scale(s) to log. A single value sets the data axis for univariate distributions and both axes for
            bivariate distributions. A pair of values sets each axis independently. Numeric values are interpreted as the
            desired base (default 10). If False, defer to the existing Axes scale.
            by default None
        color : matplotlib color
            Single color specification for when hue mapping is not used. Otherwise the plot will try to hook into the
            matplotlib property cycle, by default "blue"
        fill : bool, optional
            If True, fill in the area under univariate density curves or between bivariate contours. If None, the default
            depends on `multiple`. by default True.
        data2 : Union[pd.DataFrame, np.ndarray] optional
            Second set of data to plot when `multiple` is "stack" or "fill".
        warn_singular : bool, optional
            If True, issue a warning when trying to estimate the density of data with zero variance, by default True
        **kwargs : dict, optional#
            Other keyword arguments are passed to one of the following matplotlib functions:
            - `matplotlib.axes.Axes.plot()` (univariate, `fill=False`),
            - `matplotlib.axes.fill_between()` (univariate, `fill=True`),
            - `matplotlib.axes.Axes.contour()` (bivariate, `fill=True`),
            - `matplotlib.axes.Axes.contourf()` (bivariate, `fill=True`).

        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing the plot.

        See Also
        --------
        `seaborn.scatterplot <https://seaborn.pydata.org/generated/seaborn.scatterplot.html>`_

        `seaborn.kdeplot <https://seaborn.pydata.org/generated/seaborn.kdeplot.html>`_

        """

        return sspyplot.density(
            data=self._df,
            x=x,
            y=y,
            incl_scatter=incl_scatter,
            density_type=density_type,
            title=title,
            diagonal_lines=diagonal_lines,
            xlim=xlim,
            ylim=ylim,
            scatter_kws=scatter_kws,
            incl_outline=incl_outline,
            figsize=figsize,
            legend_loc=legend_loc,
            alpha=alpha,
            gridsize=gridsize,
            kernel=kernel,
            cut=cut,
            clip=clip,
            legend=legend,
            cumulative=cumulative,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            ax=ax,
            weights=weights,
            hue=hue,
            palette=palette,
            hue_order=hue_order,
            hue_norm=hue_norm,
            multiple=multiple,
            common_norm=common_norm,
            common_grid=common_grid,
            levels=levels,
            thresh=thresh,
            bw_method=bw_method,
            bw_adjust=bw_adjust,
            log_scale=log_scale,
            color=color,
            fill=fill,
            warn_singular=warn_singular,
            **kwargs,
        )

    def jointplot(
        self,
        x="ISOPleasant",
        y="ISOEventful",
        incl_scatter=True,
        density_type="full",
        title="Soundscape Joint Plot",
        diagonal_lines=False,
        xlim=(-1, 1),
        ylim=(-1, 1),
        scatter_kws=dict(s=25, linewidth=0),
        incl_outline=False,
        legend_loc="lower left",
        alpha=0.75,
        color=None,
        joint_kws=dict(),
        marginal_kws={"fill": True, "common_norm": False},
        hue=None,
        palette="colorblind",
        hue_order=None,
        hue_norm=None,
        common_norm=False,
        fill=True,
        bw_adjust=None,
        thresh=0.1,
        levels=10,
        legend=False,
        marginal_kind="kde",
    ):  # sourcery skip: default-mutable-arg
        """Create a jointplot with distribution or scatter in the center and distributions on the margins.

        This method works by calling sns.jointplot() and creating a circumplex grid in the joint position, then
        overlaying a density or circumplex_scatter plot. The options for both the joint and marginal plots can be
        passed through the sns.jointplot() separately to customise them separately. The marginal distribution plots
        can be either a density or histogram.

        Parameters
        ----------
        data : pd.DataFrame, np.ndarray, mapping, or sequence
            Input data structure. Either a long-form collection of vectors that can be assigned to named variables or a
            wide-form dataset that will be internally reshaped.
        x : vector or key in `data`, optional
            column name for x variable, by default "ISOPleasant"
        y : vector or key in `data`, optional
            column name for y variable, by default "ISOEventful"
        incl_scatter : bool, optional
            Whether to include a scatter plot of the data, by default True
        density_type : str, optional
            Type of density plot to draw, by default "full"
        diagonal_lines : bool, optional
            whether to include diagonal dimension axis labels in the joint plot, by default False
        palette : str, optional
            [description], by default "colorblind"
        incl_scatter : bool, optional
            plot coordinate scatter underneath density plot, by default False
        fill : bool, optional
            whether to fill the density plot, by default True
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
            by default None
        bw_adjust : [type], optional
            [description], by default default_bw_adjust
        alpha : float, optional
            [description], by default 0.95
        legend : bool, optional
            whether to include the hue labels legend, by default False
        legend_loc : str, optional
            relative location of the legend, by default "lower left"
        marginal_kind : str, optional
            density or histogram plot in the margins, by default "kde"
        hue : vector or key in data, optional
            Grouping variable that will produce points with different colors. Can be either categorical or numeric,
            although color mapping will behave differently in latter case, by default None
        joint_kws : dict, optional
            Arguments to pass to density or scatter joint plot, by default {}
        marginal_kws : dict, optional
            Arguments to pass to marginal distribution plots, by default {"fill": True}
        hue : vector or key in `data`, optional
            Semantic variable that is mapped to determine the color of plot elements.
        palette : string, list, dict, or `matplotlib.colors.Colormap`, optional
            Method for choosing the colors to use when mapping the `hue` semantic. String values are passed to
            `color_palette()`. List or dict values imply categorical mapping, while a colormap object implies numeric
            mapping.
            by default, `"colorblind"`
        hue_order : vector of strings, optional.
            Specify the order of processing and plotting for categorical levels of the `hue` semantic.
        hue_norm : tuple or matplotlib.colors.Normalize, optional
            Either a pair of values that set the normalization range in data units or an object that will map from data
            units into a [0, 1] interval. Usage implies numeric mapping.
        common_norm : bool, optional
            If True, scale each conditional density by the number of observations such that the total area under all
            densities sums to 1. Otherwise, normalize each density independently, by default False.
        fill : bool, optional
            If True, fill in the area under univariate density curves or between bivariate contours. If None, the default
            depends on `multiple`. by default True
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
            See Notes. by default default_bw_adjust (1.2)
        thresh : number in [0, 1], optional
            Lowest iso-proportional level at which to draw a contour line. Ignored when `levels` is a vector. Only relevant
            with bivariate plots. by default 0.1
        levels : int or vector, optional
            Number of contour levels or values to draw contours at. A vector argument must have increasing values in [0, 1].
            Levels correspond to iso-proportionas of the density: e.g. 20% of the probability mass will lie below the
            contour drawn for 0.2. Only relevant with bivariate data.
            by default 10
        legend : bool, optional
            If False, suppress the legend for semantic variables, by default False
        legend_loc : str, optional
            Relative location of the legend, by default "lower left"
        marginal_kind : str, optional
            density or histogram plot in the margins, by default "kde"

        Returns
        -------
        plt.Axes

        See Also
        --------
        `seaborn.kdeplot <https://seaborn.pydata.org/generated/seaborn.kdeplot.html>`_

        `seaborn.jointplot <https://seaborn.pydata.org/generated/seaborn.jointplot.html>`_
        """

        return sspyplot.jointplot(
            data=self._df,
            x=x,
            y=y,
            incl_scatter=incl_scatter,
            density_type=density_type,
            title=title,
            diagonal_lines=diagonal_lines,
            xlim=xlim,
            ylim=ylim,
            scatter_kws=scatter_kws,
            incl_outline=incl_outline,
            legend_loc=legend_loc,
            alpha=alpha,
            color=color,
            joint_kws=joint_kws,
            marginal_kws=marginal_kws,
            hue=hue,
            palette=palette,
            hue_order=hue_order,
            hue_norm=hue_norm,
            common_norm=common_norm,
            fill=fill,
            bw_adjust=bw_adjust,
            thresh=thresh,
            levels=levels,
            legend=legend,
            marginal_kind=marginal_kind,
        )


# %%
