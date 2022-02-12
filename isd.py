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

The current version works by using `pandas_flavor` and `janitor` to attach 
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

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi
import pandas_flavor as pf
import janitor


#%%

# Define the names of the PAQ columns
PAQ_COLS = [
    "pleasant",
    "vibrant",
    "eventful",
    "chaotic",
    "annoying",
    "monotonous",
    "uneventful",
    "calm",
]

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
    if version == "V0.2.1":
        url = "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx"
    elif version in ["V0.2.2", "latest"]:
        url = "https://zenodo.org/record/5705908/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"

    return pd.read_excel(url)


def calculate_paq_coords(
    results_df: pd.DataFrame,
    scale_to_one: bool = True,
    projection: bool = True,
):
    """Calculates the projected ISOPleasant and ISOEventful coordinates

    If a value is missing, by default it is replaced with neutral (3).
    The raw PAQ values should be Likert data from 1 to 5 and the column
    names should match the PAQ_cols given above.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe containing ISD formatted data
    scale_to_one : bool, optional
        Should the coordinates be scaled to (-1, +1), by default True
    projection : bool, optional
        Use the trigonometric projection (cos(45)) term for diagonal PAQs, by default True

    Returns
    -------
    tuple
        ISOPleasant and ISOEventful coordinate values
    """

    proj = np.cos(np.deg2rad(45)) if projection else 1
    scale = 4 + np.sqrt(32)

    # TODO: Add if statements for too much missing data
    # P =(p−a)+cos45°(ca−ch)+cos45°(v−m)
    complex_pleasant = (
        (results_df.pleasant.fillna(3) - results_df.annoying.fillna(3))
        + proj * (results_df.calm.fillna(3) - results_df.chaotic.fillna(3))
        + proj * (results_df.vibrant.fillna(3) - results_df.monotonous.fillna(3))
    )
    ISOPleasant = complex_pleasant / scale if scale_to_one else complex_pleasant

    # E =(e−u)+cos45°(ch−ca)+cos45°(v−m)
    complex_eventful = (
        (results_df.eventful.fillna(3) - results_df.uneventful.fillna(3))
        + proj * (results_df.chaotic.fillna(3) - results_df.calm.fillna(3))
        + proj * (results_df.vibrant.fillna(3) - results_df.monotonous.fillna(3))
    )
    ISOEventful = complex_eventful / scale if scale_to_one else complex_eventful

    return ISOPleasant, ISOEventful


def circumplex_grids(axes, diagonal_lines=False):
    """Create the base layer grids and label lines for the soundscape circumplex

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        plt subplot Axes to add the circumplex grids to
    diagonal_lines : bool, optional
        flag for whether the include the diagonal dimensions (calm, etc), by default False
    """
    sns.set_palette("Blues")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    line_weights = 1.5

    x_lim = axes.get_xlim()
    y_lim = axes.get_ylim()

    # grids and ticks
    axes.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    axes.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    axes.grid(b=True, which="major", color="grey", alpha=0.5)
    axes.grid(
        b=True,
        which="minor",
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.4,
    )

    # hide axis labels
    axes.xaxis.label.set_visible(False)
    axes.yaxis.label.set_visible(False)

    # axes label font dict
    prim_ax_font = {
        "fontstyle": "italic",
        "fontsize": "medium",
        "fontweight": "bold",
        "c": "grey",
        "alpha": 1,
    }

    # Add lines and labels for circumplex model
    ## Primary Axes
    axes.plot(  # Horizontal line
        [-1, 1],
        [0, 0],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
    )
    axes.plot(  # vertical line
        [0, 0],
        [1, -1],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
    )

    ### Labels
    axes.text(  # ISOPleasant Label
        x=x_lim[1] + 0.01,
        y=0,
        s="ISO\nPleasant",
        ha="left",
        va="center",
        fontdict=prim_ax_font,
    )
    axes.text(  # ISOEventful Label
        x=0,
        y=y_lim[1] + 0.01,
        s="ISO\nEventful",
        ha="center",
        va="bottom",
        fontdict=prim_ax_font,
    )

    ## Diagonal Axes

    if diagonal_lines:
        diag_ax_font = {
            "fontstyle": "italic",
            "fontsize": "small",
            "fontweight": "bold",
            "c": "grey",
            "alpha": 0.75,
        }

        axes.plot(  # uppward diagonal
            [-1, 1],
            [-1, 1],
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            lw=line_weights,
        )
        axes.plot(  # downward diagonal
            [-1, 1],
            [1, -1],
            linestyle="dashed",
            color="grey",
            alpha=0.5,
            lw=line_weights,
        )

        ### Labels
        # TODO: Add diagonal labels
        axes.text(  # Vibrant label
            x=x_lim[1] / 2,
            y=y_lim[1] / 2,
            s="(vibrant)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
        )
        axes.text(  # Chaotic label
            x=x_lim[0] / 2,
            y=y_lim[1] / 2,
            s="(chaotic)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
        )
        axes.text(  # monotonous label
            x=x_lim[0] / 2,
            y=y_lim[0] / 2,
            s="(monotonous)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
        )
        axes.text(  # calm label
            x=x_lim[1] / 2,
            y=y_lim[0] / 2,
            s="(calm)",
            ha="center",
            va="center",
            fontdict=diag_ax_font,
        )


def location_describe(loc_df, type="percent", pl_threshold=0, ev_threshold=0):
    count = len(loc_df)
    pl_count = len(loc_df[loc_df["ISOPleasant"] > pl_threshold])
    ev_count = len(loc_df[loc_df["ISOEventful"] > ev_threshold])
    vibrant_count = len(
        loc_df[loc_df["ISOPleasant"] > pl_threshold][
            loc_df["ISOEventful"] > ev_threshold
        ]
    )
    chaotic_count = len(
        loc_df[loc_df["ISOPleasant"] < pl_threshold][
            loc_df["ISOEventful"] > ev_threshold
        ]
    )
    mono_count = len(
        loc_df[loc_df["ISOPleasant"] < pl_threshold][
            loc_df["ISOEventful"] < ev_threshold
        ]
    )
    calm_count = len(
        loc_df[loc_df["ISOPleasant"] > pl_threshold][
            loc_df["ISOEventful"] < ev_threshold
        ]
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


def soundscapy_describe(df, type="percent"):
    res = {
        location: location_describe(df.filter_location_ids([location]), type=type)
        for location in df["LocationID"]
    }

    res = pd.DataFrame.from_dict(res, orient="index")
    return res


def soundscapy_describe(df, type="percent"):
    res = {
        location: location_describe(df.filter_location_ids([location]), type=type)
        for location in df["LocationID"]
    }

    res = pd.DataFrame.from_dict(res, orient="index")
    return res


def iso_annotation(
    ax,
    data,
    location,
    x_adj=0,
    y_adj=0,
    x_key="ISOPleasant",
    y_key="ISOEventful",
    ha="center",
    va="center",
    fontsize="small",
    arrowprops=dict(arrowstyle="-", ec="black"),
    **text_kwargs,
):
    """add text annotations to circumplex plot based on coordinate values

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
        dict of properties to send to plt.annotate, by default dict(arrowstyle="-", ec="black")
    """
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


def paq_radar_plot(data, ax=None):
    """Generate a radar/spider plot of PAQ values

    Parameters
    ----------
    data : pd.Dataframe
        dataframe of PAQ values
        recommended max number of values: 3
    ax : matplotlib.pyplot.Axes, optional
        existing subplot axes to plot to, by default None

    Returns
    -------
    plt.Axes
        matplotlib Axes with radar plot
    """
    if ax is None:
        ax = plt.gca(polar=True)
    # ---------- Part 1: create background
    # Number of variables
    categories = [
        "          pleasant",
        "    vibrant",
        "eventful",
        "chaotic    ",
        "annoying          ",
        "monotonous            ",
        "uneventful",
        "calm",
    ]
    N = len(categories)

    # What will be the angle of each axis in the plot (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the plots
    # fig = plt.figure(figsize=(16, 8))

    # initialise the spider plot
    # ax = plt.subplot(121, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(1, 5)

    # -------- Part 2: Add plots

    # Plot each individual = each line of the data
    fill_col = ["b", "r", "g"]
    for i in range(len(data.index)):
        # Ind1
        values = data.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=data.index[i])
        ax.fill(angles, values, fill_col[i], alpha=0.1)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    return ax


def simulation(n=3000, add_paq_coords=False, **coord_kwargs):
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

    Returns
    -------
    pd.Dataframe
        dataframe of randomly generated PAQ response
    """
    np.random.seed(42)
    df = pd.DataFrame(np.random.randint(1, 5, size=(n, 8)), columns=PAQ_COLS)
    if add_paq_coords:
        ISOPl, ISOEv = calculate_paq_coords(df, **coord_kwargs)
        df = janitor.add_columns(df, ISOPleasant=ISOPl, ISOEventful=ISOEv)
    return df


@pf.register_dataframe_method
def convert_column_to_index(self, col="GroupID", drop=False):
    """Reassign an existing column as the dataframe index

    Parameters
    ----------
    col : str, optional
        name of column to assign as index, by default "GroupID"
    drop : bool, optional
        whether to drop the column from the dataframe, by default False

    """
    assert col in self.columns, f"col: {col} not found in dataframe"
    self.index = self[col]
    if drop:
        self = self.drop(col, axis=1)
    return self


@pf.register_dataframe_method
def filter_group_ids(self, group_ids: list, **kwargs):
    return janitor.filter_column_isin(self, "GroupID", group_ids, **kwargs)


@pf.register_dataframe_method
def filter_record_ids(self, record_ids: list, **kwargs):
    return janitor.filter_column_isin(self, "RecordID", record_ids, **kwargs)


@pf.register_dataframe_method
def filter_session_ids(self, session_ids: list, **kwargs):
    return janitor.filter_column_isin(self, "SessionID", session_ids, **kwargs)


@pf.register_dataframe_method
def filter_location_ids(self, location_ids: list, **kwargs):
    return janitor.filter_column_isin(self, "LocationID", location_ids, **kwargs)


@pf.register_dataframe_method
def filter_lockdown(self, is_lockdown=False):
    complement = bool(is_lockdown)
    return janitor.filter_on(self, "Lockdown == 0", complement)


@pf.register_dataframe_method
def return_paqs(self, incl_ids=True, other_cols=None):
    """Return only the PAQ columns

    Parameters
    ----------
    incl_ids : bool, optional
        whether to include ID cols too (i.e. RecordID, GroupID, etc), by default True
    other_cols : list, optional
        other columns to also include, by default None

    """
    cols = PAQ_COLS
    if incl_ids:
        id_cols = [
            name
            for name in ["RecordID", "GroupID", "SessionID", "LocationID"]
            if name in self.columns
        ]

        cols = id_cols + cols
    if other_cols:
        cols = cols + other_cols
    return self[cols]


@pf.register_dataframe_method
def add_paq_coords(
    self,
    scale_to_one: bool = True,
    projection: bool = True,
    names=["ISOPleasant", "ISOEventful"],
):
    """Calculate and add ISO coordinates as new columns in dataframe

    Calls `calculate_paq_coords()`

    Parameters
    ----------
    scale_to_one : bool, optional
        Should the coordinates be scaled to (-1, +1), by default True
    projection : bool, optional
        Use the trigonometric projection (cos(45)) term for diagonal PAQs, by default True
    names : list, optional
        Names for new coordinate columns, by default ["ISOPleasant", "ISOEventful"]
    """
    isopl, isoev = calculate_paq_coords(self, scale_to_one, projection)
    self = self.add_column(names[0], isopl).add_column(names[1], isoev)
    return self


## Plotting


@pf.register_dataframe_method
def paq_radar(self, ax=None, index=None):
    # TODO: Resize the plot
    if index:
        self = self.convert_column_to_index(col=index)
    data = self[PAQ_COLS]
    if ax is None:
        ax = plt.axes(polar=True)
    # ---------- Part 1: create background
    # Number of variables
    categories = [
        "          pleasant",
        "    vibrant",
        "eventful",
        "chaotic    ",
        "annoying          ",
        "monotonous            ",
        "uneventful",
        "calm",
    ]
    N = len(categories)

    # What will be the angle of each axis in the plot (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(1, 5)

    # -------- Part 2: Add plots

    # Plot each individual = each line of the data
    fill_col = ["b", "r", "g"]
    for i in range(len(data.index)):
        # Ind1
        values = data.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=data.index[i])
        ax.fill(angles, values, fill_col[i], alpha=0.25)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    return ax


@pf.register_dataframe_method
def circumplex_scatter(
    self,
    ax=None,
    title="Soundscape Scatter Plot",
    hue=None,
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=True,
    diagonal_lines=False,
    figsize=(5, 5),
    palette=None,
    legend=False,
    legend_loc="lower left",
    s=10,
    **scatter_kwargs,
):
    """Plot ISOcoordinates as scatter points on a soundscape circumplex grid

    Makes use of seaborn.scatterplot

    Parameters
    ----------
    ax : plt.Axes, optional
        existing matplotlib axes, by default None
    title : str, optional
        , by default "Soundscape Scatter Plot"
    hue : vector or key in data, optional
        Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case, by default None
    x : str, optional
        column name for x variable, by default "ISOPleasant"
    y : str, optional
        column name for y variable, by default "ISOEventful"
    prim_labels : bool, optional
        whether to include ISOPleasant and ISOeventful labels, by default True
    diagonal_lines : bool, optional
        whether to include diagonal dimension labels (e.g. calm, etc.), by default False
    figsize : tuple, optional
        by default (5, 5)
    palette : string, list, dict or matplotlib.colors.Colormap, optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object implies numeric mapping.
        by default None
    legend : bool, optional
        whether to include legend with the hue values, by default False
    legend_loc : str, optional
        relative location of legend, by default "lower left"
    s : int, optional
        size of scatter points, by default 10

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if palette is None:
        n_colors = len(self[hue].unique()) if hue else len(self)
        palette = sns.color_palette("husl", n_colors, as_cmap=False)

    s = sns.scatterplot(
        data=self,
        x=x,
        y=y,
        hue=hue,
        s=s,
        ax=ax,
        legend=legend,
        palette=palette,
        zorder=data_zorder,
        **scatter_kwargs,
    )
    ax = _deal_w_default_labels(ax, prim_labels)
    _circumplex_grid(ax, prim_labels, diagonal_lines)
    _set_circum_title(ax, prim_labels, title)
    if legend:
        _move_legend(ax, legend_loc)
    return s


@pf.register_dataframe_method
def circumplex_density(
    self,
    ax=None,
    title="Soundscape Density Plot",
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=True,
    diagonal_lines=False,
    incl_scatter=False,
    incl_outline=False,
    figsize=(5, 5),
    palette="Blues",
    scatter_color="black",
    outline_color="black",
    fill_color="blue",
    fill=True,
    hue=None,
    common_norm=False,
    bw_adjust=default_bw_adjust,
    alpha=0.95,
    legend=False,
    legend_loc="lower left",
    s=10,
    scatter_kwargs={},
    **density_kwargs,
):
    """Create a bivariate distribution plot of ISOCoordinates

    This method works by creating a circumplex_grid, then overlaying a sns.kdeplot() using the ISOCoordinate data. IF a scatter is also included, it overlays a sns.scatterplot() using the given options underneath the density plot.

    If using a hue grouping, it is recommended to only plot the 50th percentile contour so as to not create a cluttered figure. This can be done with the options thresh = 0.5, levels = 2.

    Parameters
    ----------
    ax : plt.Axes, optional
        existing subplot axes, by default None
    title : str, optional
        by default "Soundscape Density Plot"
    x : str, optional
        column name for x variable, by default "ISOPleasant"
    y : str, optional
        column name for y variable, by default "ISOEventful"
    prim_labels : bool, optional
        whether to include ISOPleasant and ISOEventful axis labels, by default True
    diagonal_lines : bool, optional
        whether to include diagonal dimension axis labels (i.e. calm, etc.), by default False
    incl_scatter : bool, optional
        plot coordinate scatter underneath density plot, by default False
    incl_outline : bool, optional
        include a thicker outline around the density plot, by default False
    figsize : tuple, optional
        by default (5, 5)
    palette : str, optional
        Method for choosing the colors to use when mapping the hue semantic. String values are passed to seaborn.color_palette(). List or dict values imply categorical mapping, while a colormap object implies numeric mapping.
        by default "Blues"
    scatter_color : str, optional
        define a color for the scatter points. Does not work with a hue grouping variable, by default "black"
    outline_color : str, optional
        define a color for the add'l density outline, by default "black"
    fill_color : str, optional
        define a color for the density fill, does not work with a hue grouping variable, by default "blue"
    fill : bool, optional
        whether to fill the density plot, by default True
    hue : vector or key in data, optional
        Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case, by default None
    common_norm : bool, optional
        [description], by default False
    bw_adjust : [type], optional
        [description], by default default_bw_adjust
    alpha : float, optional
        [description], by default 0.95
    legend : bool, optional
        whether to include the hue labels legend, by default False
    legend_loc : str, optional
        relative location of the legend, by default "lower left"
    s : int, optional
        size of the scatter points, by default 10
    scatter_kwargs : dict, optional
        additional arguments for sns.scatterplot(), by default {}

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if incl_scatter:
        d = sns.scatterplot(
            data=self,
            x=x,
            y=y,
            hue=hue,
            s=s,
            ax=ax,
            legend=legend,
            color=scatter_color,
            palette=palette,
            zorder=data_zorder,
            **scatter_kwargs,
        )

    if incl_outline:
        d = sns.kdeplot(
            data=self,
            x=x,
            y=y,
            fill=False,
            ax=ax,
            alpha=1,
            color=outline_color,
            palette=palette,
            hue=hue,
            common_norm=common_norm,
            legend=legend,
            zorder=data_zorder,
            bw_adjust=bw_adjust,
            **density_kwargs,
        )

    d = sns.kdeplot(
        data=self,
        x=x,
        y=y,
        fill=fill,
        ax=ax,
        alpha=alpha,
        palette=palette,
        color=fill_color,
        hue=hue,
        common_norm=common_norm,
        legend=legend,
        zorder=data_zorder,
        bw_adjust=bw_adjust,
        **density_kwargs,
    )

    _circumplex_grid(ax, prim_labels, diagonal_lines)
    _set_circum_title(ax, prim_labels, title)
    _deal_w_default_labels(ax, prim_labels)
    if legend:
        _move_legend(ax, legend_loc)
    return d


@pf.register_dataframe_method
def circumplex_jointplot_density(
    self,
    title="Soundscape Joint Plot",
    x="ISOPleasant",
    y="ISOEventful",
    prim_labels=False,
    diagonal_lines=False,
    palette="Blues",
    incl_scatter=False,
    scatter_color="black",
    fill=True,
    bw_adjust=default_bw_adjust,
    alpha=0.95,
    legend=False,
    legend_loc="lower left",
    marginal_kind="kde",
    hue=None,
    joint_kwargs={},
    marginal_kwargs={"fill": True},
):
    """Create a jointplot with distribution or scatter in the center and distributions on the margins.

    This method works by calling sns.jointplot() and creating a circumplex grid in the joint position, then overlaying a circumplex_density or circumplex_scatter plot. The options for both the joint and marginal plots can be passed through the sns.jointplot() separately to customise them separately. The marginal distribution plots can be either a density or histogram.

    Parameters
    ----------
    title : str, optional
        by default "Soundscape Joint Plot"
    x : str, optional
        column name for x variable, by default "ISOPleasant"
    y : str, optional
        column name for y variable, by default "ISOEventful"
    prim_labels : bool, optional
        whether to include ISOPleasant and ISOEventful axis labels in the joint plot, by default False
    diagonal_lines : bool, optional
        whether to include diagonal dimension axis labels in the joint plot, by default False
    palette : str, optional
        [description], by default "Blues"
    incl_scatter : bool, optional
        plot coordinate scatter underneath density plot, by default False
    scatter_color : str, optional
        define a color for the scatter points, by default "black"
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
        Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case, by default None
    joint_kwargs : dict, optional
        Arguments to pass to density or scatter joint plot, by default {}
    marginal_kwargs : dict, optional
        Arguments to pass to marginal distribution plots, by default {"fill": True}

    Returns
    -------
    plt.Axes
    """
    g = sns.JointGrid()
    circumplex_density(
        self,
        g.ax_joint,
        title=None,
        x=x,
        y=y,
        prim_labels=prim_labels,
        diagonal_lines=diagonal_lines,
        palette=palette,
        incl_scatter=incl_scatter,
        scatter_color=scatter_color,
        hue=hue,
        fill=fill,
        bw_adjust=bw_adjust,
        alpha=alpha,
        legend=legend,
        **joint_kwargs,
    )
    if legend:
        _move_legend(g.ax_joint, legend_loc)

    if marginal_kind == "hist":
        sns.histplot(
            data=self,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            binrange=(-1, 1),
            legend=False,
            **marginal_kwargs,
        )
        sns.histplot(
            data=self,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            binrange=(-1, 1),
            legend=False,
            **marginal_kwargs,
        )
    elif marginal_kind == "kde":
        sns.kdeplot(
            data=self,
            x=x,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_x,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kwargs,
        )
        sns.kdeplot(
            data=self,
            y=y,
            hue=hue,
            palette=palette,
            ax=g.ax_marg_y,
            bw_adjust=bw_adjust,
            legend=False,
            **marginal_kwargs,
        )
    g.ax_marg_x.set_title(title, pad=6.0)

    return g


def _move_legend(ax, new_loc, **kws):
    """Moves legend to desired relative location.

    Parameters
    ----------
    ax : plt.Axes
        Existing axes object to adjust the legend on
    new_loc : str or pair of floats
        The location of the legend
    """
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


def _circumplex_grid(ax, prim_labels=True, diagonal_lines=False):

    # Setting up the grids
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    line_weights = 1.5
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # grids and ticks
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which="major", color="grey", alpha=0.5)
    ax.grid(
        b=True,
        which="minor",
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.4,
        zorder=prim_lines_zorder,
    )

    ax = _primary_lines_and_labels(ax, prim_labels, line_weights=line_weights)
    if diagonal_lines:
        ax = _diagonal_lines_and_labels(ax, line_weights=line_weights)

    return ax


def _set_circum_title(ax, prim_labels, title):
    title_pad = 30.0 if prim_labels is True else 6.0
    ax.set_title(title, pad=title_pad)
    return ax


def _deal_w_default_labels(ax, prim_labels):
    if prim_labels is True or prim_labels == "none":
        # hide axis labels
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

    return ax


def _primary_lines_and_labels(ax, prim_labels, line_weights):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # Add lines and labels for circumplex model
    ## Primary Axes
    ax.plot(  # Horizontal line
        [-1, 1],
        [0, 0],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
        zorder=prim_lines_zorder,
    )
    ax.plot(  # vertical line
        [0, 0],
        [1, -1],
        linestyle="dashed",
        color="grey",
        alpha=1,
        lw=line_weights,
        zorder=prim_lines_zorder,
    )

    if prim_labels is True:
        prim_ax_font = {
            "fontstyle": "italic",
            "fontsize": "medium",
            "fontweight": "bold",
            "c": "grey",
            "alpha": 1,
        }
        ### Labels
        ax.text(  # ISOPleasant Label
            x=x_lim[1] + 0.01,
            y=0,
            s="ISO\nPleasant",
            ha="left",
            va="center",
            fontdict=prim_ax_font,
        )
        ax.text(  # ISOEventful Label
            x=0,
            y=y_lim[1] + 0.01,
            s="ISO\nEventful",
            ha="center",
            va="bottom",
            fontdict=prim_ax_font,
        )

    return ax


def _diagonal_lines_and_labels(ax, line_weights):
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
        [-1, 1],
        [-1, 1],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=diag_lines_zorder,
    )
    ax.plot(  # downward diagonal
        [-1, 1],
        [1, -1],
        linestyle="dashed",
        color="grey",
        alpha=0.5,
        lw=line_weights,
        zorder=diag_lines_zorder,
    )

    ### Labels
    ax.text(  # Vibrant label
        x=x_lim[1] / 2,
        y=y_lim[1] / 2,
        s="(vibrant)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    ax.text(  # Chaotic label
        x=x_lim[0] / 2,
        y=y_lim[1] / 2,
        s="(chaotic)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    ax.text(  # monotonous label
        x=x_lim[0] / 2,
        y=y_lim[0] / 2,
        s="(monotonous)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    ax.text(  # calm label
        x=x_lim[1] / 2,
        y=y_lim[0] / 2,
        s="(calm)",
        ha="center",
        va="center",
        fontdict=diag_ax_font,
        zorder=diag_labels_zorder,
    )
    return ax


# %%
