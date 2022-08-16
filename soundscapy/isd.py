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
#%%
# Add soundscapy to the Python path
import sys
from datetime import date
from itertools import groupby
from typing import Union

import janitor
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.extensions import register_dataframe_accessor

sys.path.append("..")
import soundscapy.database as db
import soundscapy.plotting as ssidplot

# Define the names of the PAQ columns
from soundscapy.parameters import PAQ_IDS, PAQ_NAMES

#%%


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
        return db.return_paqs(df, incl_ids, other_cols)

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
        return ssidplot.paq_radar_plot(self._df, ax, index)

    def circumplex_scatter(
        self,
        ax=None,
        title="Soundscape Scatter Plot",
        hue=None,
        x="ISOPleasant",
        y="ISOEventful",
        prim_labels=True,
        diagonal_lines=False,
        xlim=(-1, 1),
        ylim=(-1, 1),
        figsize=(5, 5),
        palette="colorblind",
        legend=False,
        legend_loc="lower left",
        s=10,
        **scatter_kwargs,
    ):
        return ssidplot.circumplex_scatter(
            self._df,
            ax,
            title,
            hue,
            x,
            y,
            prim_labels,
            diagonal_lines,
            xlim,
            ylim,
            figsize,
            palette,
            legend,
            legend_loc,
            s,
            **scatter_kwargs,
        )

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
        xlim=(-1, 1),
        ylim=(-1, 1),
        figsize=(5, 5),
        palette="colorblind",
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
    ):  # sourcery skip: default-mutable-arg
        return ssidplot.circumplex_density(
            self._df,
            ax,
            title,
            x,
            y,
            prim_labels,
            diagonal_lines,
            xlim,
            ylim,
            incl_scatter,
            incl_outline,
            figsize,
            palette,
            scatter_color,
            outline_color,
            fill_color,
            fill,
            hue,
            common_norm,
            bw_adjust,
            alpha,
            legend,
            legend_loc,
            s,
            scatter_kwargs,
            **density_kwargs,
        )

    def circumplex_jointplot_density(
        self,
        title="Soundscape Joint Plot",
        x="ISOPleasant",
        y="ISOEventful",
        prim_labels=False,
        diagonal_lines=False,
        xlim=(-1, 1),
        ylim=(-1, 1),
        palette="colorblind",
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
    ):  # sourcery skip: default-mutable-arg
        return ssidplot.circumplex_jointplot_density(
            self._df,
            title,
            x,
            y,
            prim_labels,
            diagonal_lines,
            xlim,
            ylim,
            palette,
            incl_scatter,
            scatter_color,
            fill,
            bw_adjust,
            alpha,
            legend,
            legend_loc,
            marginal_kind,
            hue,
            joint_kwargs,
            marginal_kwargs,
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

    Returns
    -------
    pd.Dataframe
        dataframe of randomly generated PAQ response
    """
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randint(min(val_range), max(val_range), size=(n, 8)),
        columns=PAQ_NAMES,
    )
    if add_paq_coords:
        ISOPl, ISOEv = db.calculate_paq_coords(df, **coord_kwargs)
        df = janitor.add_columns(df, ISOPleasant=ISOPl, ISOEventful=ISOEv)
    return df


# %%
