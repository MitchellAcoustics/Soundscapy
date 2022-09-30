"""ISD analysis functions
By: Andrew Mitchell - Research Fellow, UCL and Alan Turing Institute
andrew.mitchell.18@ucl.ac.uk

This module provides a collection of scripts and methods for analysing the 
soundscape assessment data contained in the International Soundscape Database
(ISD). 

This version of the code is provided alongside Mitchell et al. (2022) 'How to
analyse and represent soundscape perception'. JASA Express Letters. in order to 
replicate the results and figures presented in that article. This will 
eventually be superseded by a full Python package named Soundscapy.

The current version works by using pandas accessor functionality to attach 
methods to the pandas dataframe containing the ISD data. These can then be 
accessed and used in the same way as class methods. This API may change in the 
future as I find more stable ways to achieve this behaviour.

This module requires that `pandas`, `matplotlib`, `seaborn`, `pandas_flavor` 
and `janitor` be installed within the Python environment you are running this 
script in. In general the functions are provided as pandas_flavor style 
dataframe methods. This enables them to be accessed as if they were a class 
method of the pandas Dataframe, all that is needed is to import isd.py at the 
top of your file.

This file can also be imported as a module.

"""
# %%
# Add soundscapy to the Python path
import sys
from datetime import date
from typing import Union, Tuple, List

import janitor
import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
import matplotlib

import soundscapy.plotting.likert

sys.path.append("..")
import soundscapy.database as db
import soundscapy.plotting.circumplex as ssidplot

# Define the names of the PAQ columns
from soundscapy.parameters import PAQ_NAMES

# %%


# Default plot settings
diag_lines_zorder = 1
diag_labels_zorder = 4
prim_lines_zorder = 2
data_zorder = 3
default_bw_adjust = 1.2


def load_isd_dataset(version="latest"):
    """Automatically fetch and load the ISD dataset from Zenodo

    Parameters
    ----------
    version : str, optional
        version number of the dataset to fetch, by default "latest"

    Returns
    -------
    pd.Dataframe
        ISD data
    """
    return db.load_isd_dataset(version)


@register_dataframe_accessor("isd")
class ISDAccessor:
    def __init__(self, df):
        self._df = df
        self._analysis_date = date.today().isoformat()

    def validate_dataset(
        self,
        paq_aliases=None,
        allow_lockdown=True,
        allow_na=False,
        verbose=1,
        val_range=(5, 1),
    ):
        return db.validate_dataset(
            self._df, paq_aliases, allow_lockdown, allow_na, verbose, val_range
        )

    def paq_data_quality(self, verbose=0):
        return db.paq_data_quality(self._df, verbose)

    def filter_group_ids(self, group_ids):
        if isinstance(group_ids, str):
            return self._df.query("GroupID == @group_ids")
        elif isinstance(group_ids, (list, tuple)):
            return self._df.query("GroupID in @group_ids")

    def filter_record_ids(self, record_ids: Union[tuple, str]):
        if isinstance(record_ids, str):
            return self._df.query("RecordID == @record_ids")
        elif isinstance(record_ids, (list, tuple)):
            return self._df.query("RecordID in @record_ids")

    def filter_session_ids(self, session_ids: list, **kwargs):
        if isinstance(session_ids, str):
            return self._df.query("SessionID == @session_ids")
        elif isinstance(session_ids, (list, tuple)):
            return self._df.query("SessionID in @session_ids")

    def filter_location_ids(self, location_ids):
        if isinstance(location_ids, str):
            return self._df.query("LocationID == @location_ids")
        elif isinstance(location_ids, (list, tuple)):
            return self._df.query("LocationID in @location_ids")

    def filter_lockdown(self, is_lockdown=False):
        return (
            self._df.query("Lockdown == 1")
            if is_lockdown
            else self._df.query("Lockdown == 0")
        )

    # TODO: add mean_responses method

    def convert_column_to_index(self, col="GroupID", drop=False):
        return db.convert_column_to_index(self._df, col, drop)

    def return_paqs(self, incl_ids=True, other_cols=None):
        """Return only the PAQ columns

        Parameters
        ----------
        incl_ids : bool, optional
            whether to include ID cols too (i.e. RecordID, GroupID, etc), by default True
        other_cols : list, optional
            other columns to also include, by default None

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
        """
        isopl, isoev = db.calculate_paq_coords(
            self._df, scale_to_one, val_range, projection
        )
        return self._df.add_column(names[0], isopl).add_column(names[1], isoev)

    def location_describe(
        self, location, type="percent", pl_threshold=0, ev_threshold=0
    ):
        loc_df = self.filter_location_ids(location_ids=[location])
        count = len(loc_df)
        pl_count = len(loc_df[loc_df["ISOPleasant"] > pl_threshold])
        ev_count = len(loc_df[loc_df["ISOEventful"] > ev_threshold])
        vibrant_count = len(
            loc_df.query("ISOPleasant > @pl_threshold & ISOEventful > @ev_threshold")
        )
        chaotic_count = len(
            loc_df.query("ISOPleasant < @pl_threshold & ISOEventful > @ev_threshold")
        )
        mono_count = len(
            loc_df.query("ISOPleasant < @pl_threshold & ISOEventful < @ev_threshold")
        )
        calm_count = len(
            loc_df.query("ISOPleasant > @pl_threshold & ISOEventful < @ev_threshold")
        )

        res = {
            "count": count,
            "ISOPleasant": round(loc_df["ISOPleasant"].mean(), 3),
            "ISOEventful": round(loc_df["ISOEventful"].mean(), 3),
        }
        if type == "percent":
            res["pleasant"] = round(pl_count / count, 3)
            res["eventful"] = round(ev_count / count, 3)
            res["vibrant"] = round(vibrant_count / count, 3)
            res["chaotic"] = round(chaotic_count / count, 3)
            res["monotonous"] = round(mono_count / count, 3)
            res["calm"] = round(calm_count / count, 3)

        elif type == "count":
            res["pleasant"] = pl_count
            res["eventful"] = ev_count
            res["vibrant"] = vibrant_count
            res["chaotic"] = chaotic_count
            res["monotonous"] = mono_count
            res["calm"] = calm_count

        return res

    def soundscapy_describe(self, type="percent"):
        res = {
            location: self.location_describe(location, type=type)
            for location in self._df["LocationID"].unique()
        }

        res = pd.DataFrame.from_dict(res, orient="index")
        return res

    def paq_radar(self, ax=None, index=None):
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

        Parameters
        ----------
        x : vector or key in `data`, optional
            column name for x variable, by default "ISOPleasant"
        y : vector or key in `data`, optional
            column name for y variable, by default "ISOEventful"

         - Soundscapy specific parameters -
        We have made all of the `seaborn.scatterplot` arguments available, but have also added or changed some specific
        options for circumplex plotting.

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

         - `seaborn.scatterplot` parameters -

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
        """
        return ssidplot.scatter(
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

        Parameters
        ----------
        data : pd.DataFrame, np.ndarray, mapping or sequence
            Input data structure. Either a long-form collection of vectors that can be assigned to
            named variables or a wide-form dataset that will be internally reshaped.
        x : vector or key in `data`, optional
            Column name for x variable, by default "ISOPleasant"
        y : vector or key in `data`, optional
            Column name for y variable, by default "ISOEventful"

        - Soundscapy specific parameters -
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

         - Seaborn kdeplot arguments -

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
        """

        return ssidplot.density(
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
            data2=data2,
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

         - Soundscapy specific parameters -
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


         - Seaborn arguments -
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
        """

        return ssidplot.jointplot(
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


def simulation(n=3000, add_paq_coords=False, val_range=(5, 1), **coord_kwargs):
    """Generate random PAQ responses

    The PAQ responses will follow a uniform random distribution
    for each PAQ, meaning e.g. for calm either 1, 2, 3, 4, or 5
    is equally likely.

    Parameters
    ----------
    n : int, optional
        number of samples to simulate, by default 3000
    add_paq_coords : bool, optional
        should we also calculate the ISO coordinates, by default False
    val_range: tuple, optional
            (max, min) range of original PAQ responses, by default (5, 1)

    Returns
    -------
    pd.Dataframe
        dataframe of randomly generated PAQ response
    """
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randint(min(val_range), max(val_range) + 1, size=(n, 8)),
        columns=PAQ_NAMES,
    )
    if add_paq_coords:
        ISOPl, ISOEv = db.calculate_paq_coords(df, **coord_kwargs)
        df = janitor.add_columns(df, ISOPleasant=ISOPl, ISOEventful=ISOEv)
    return df


# %%
