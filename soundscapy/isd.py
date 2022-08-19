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
from typing import Union

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
        x_bins=None,
        y_bins=None,
        units=None,
        estimator=None,
        ci=95,
        n_boot=1000,
        alpha=None,
        x_jitter=None,
        y_jitter=None,
        legend="auto",
        ax=None,
        **scatter_kwargs,
    ):
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
            x_bins=x_bins,
            y_bins=y_bins,
            units=units,
            estimator=estimator,
            ci=ci,
            n_boot=n_boot,
            alpha=alpha,
            x_jitter=x_jitter,
            y_jitter=y_jitter,
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
        clip: tuple[int] = None,
        legend: bool = False,
        cumulative: bool = False,
        cbar: bool = False,
        cbar_ax: matplotlib.axes.Axes = None,
        cbar_kws: dict = None,
        ax: matplotlib.axes.Axes = None,
        weights: str = None,
        hue: str = None,
        palette="colorblind",
        hue_order: list[str] = None,
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
