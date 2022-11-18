"""
The module containing functions for dealing with soundscape survey data.
"""

# Add soundscapy to the Python path
import janitor
from pathlib import Path
from typing import Union, Tuple, List, Dict

import numpy as np
import pandas as pd

# Constants and Labels
from soundscapy.databases.parameters import PAQ_IDS, PAQ_NAMES

DEFAULT_CATS = [
    "indexing",
    "sound_source_dominance",
    "raw_PAQs",
    "overall_soundscape",
]

# General helper functions
_flatten = lambda t: [item for sublist in t for item in sublist]


###########

# %%
def return_paqs(df, incl_ids=True, other_cols=None):
    """Return only the PAQ columns

    Parameters
    ----------
    incl_ids : bool, optional
        whether to include ID cols too (i.e. RecordID, GroupID, etc), by default True
    other_cols : list, optional
        other columns to also include, by default None

    """
    cols = PAQ_IDS
    if incl_ids:
        id_cols = [
            name
            for name in ["RecordID", "GroupID", "SessionID", "LocationID"]
            if name in df.columns
        ]

        cols = id_cols + cols
    if other_cols:
        cols = cols + other_cols
    return df[cols]


def mean_responses(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Calculate the mean responses for each PAQ

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing ISD formatted data

    Returns
    -------
    pd.Dataframe
        Dataframe containing the mean responses for each PAQ
    """
    df = return_paqs(df, incl_ids=False, other_cols=[group])
    return df.groupby(group).mean()


# %%
def _circ_scale(range):
    diff = max(range) - min(range)
    return diff + diff * np.sqrt(2)


def convert_column_to_index(df, col: str, drop=False):
    """Reassign an existing column as the dataframe index"""
    assert col in df.columns, f"col: {col} not found in dataframe"
    df.index = df[col]
    if drop:
        df = df.drop(col, axis=1)
    return df


def validate_dataset(
    df: pd.DataFrame,
    paq_aliases: Union[List, Dict] = None,
    allow_lockdown: bool = False,
    allow_paq_na: bool = False,
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
    """
    if verbose > 0:
        print("Renaming PAQ columns.")
    df = rename_paqs(df, paq_aliases)

    if verbose > 0:
        print("Checking PAQ data quality.")
    l = paq_data_quality(df, verbose, allow_lockdown, allow_paq_na, val_range)
    if l is None:
        excl_df = None
    else:
        excl_df = df.iloc[l, :]
        df = df.drop(df.index[l])
    return df, excl_df


def rename_paqs(df, paq_aliases=None, verbose=0):
    if paq_aliases is None:
        if any(i in b for i in PAQ_IDS for b in df.columns):
            if verbose > 0:
                print("PAQs already correctly named.")
            return df
        if any(i in b for i in PAQ_NAMES for b in df.columns):
            paq_aliases = PAQ_NAMES

    if type(paq_aliases) == list:
        return df.rename(
            columns={
                paq_aliases[0]: PAQ_IDS[0],
                paq_aliases[1]: PAQ_IDS[1],
                paq_aliases[2]: PAQ_IDS[2],
                paq_aliases[3]: PAQ_IDS[3],
                paq_aliases[4]: PAQ_IDS[4],
                paq_aliases[5]: PAQ_IDS[5],
                paq_aliases[6]: PAQ_IDS[6],
                paq_aliases[7]: PAQ_IDS[7],
            }
        )
    elif type(paq_aliases) == dict:
        return df.rename(columns=paq_aliases)


def paq_data_quality(
    df, verbose=0, allow_lockdown=True, allow_na=False, val_range=(5, 1)
):
    paqs = return_paqs(df, incl_ids=False)
    l = []
    for i in range(len(paqs)):
        row = paqs.iloc[i]
        if "Lockdown" in df.columns:
            if allow_lockdown and df.iloc[i]["Lockdown"] == 1:
                continue
        if allow_na is False and row.isna().sum() > 0:
            l.append(i)
            continue
        if row["PAQ1"] == row["PAQ2"] == row["PAQ3"] == row["PAQ4"] == row[
            "PAQ5"
        ] == row["PAQ6"] == row["PAQ7"] == row["PAQ8"] and row.sum() != np.mean(
            val_range
        ):
            l.append(i)
        elif row.isna().sum() > 4:
            l.append(i)
        elif row.max() > max(val_range) or row.min() < min(val_range):
            l.append(i)
    if l:
        if verbose > 0:
            print(f"Identified {len(l)} samples to remove.\n{l}")
        return l
    if verbose > 0:
        print("PAQ quality confirmed. No rows dropped.")
    return None


def simulation(n=3000, val_range=(1, 5), add_paq_coords=False, **coord_kwargs):
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
        np.random.randint(min(val_range), max(val_range) + 1, size=(n, 8)),
        columns=PAQ_IDS,
    )
    if add_paq_coords:
        ISOPl, ISOEv = calculate_paq_coords(df, **coord_kwargs)
        df = janitor.add_columns(df, ISOPleasant=ISOPl, ISOEventful=ISOEv)
    return df


def calculate_paq_coords(
    results_df: pd.DataFrame,
    scale_to_one: bool = True,
    val_range: tuple = (5, 1),
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
    scale = _circ_scale(val_range) if scale_to_one else 1

    # TODO: Add if statements for too much missing data
    # P =(p−a)+cos45°(ca−ch)+cos45°(v−m)
    complex_pleasant = (
        (results_df.PAQ1.fillna(3) - results_df.PAQ5.fillna(3))
        + proj * (results_df.PAQ8.fillna(3) - results_df.PAQ4.fillna(3))
        + proj * (results_df.PAQ2.fillna(3) - results_df.PAQ6.fillna(3))
    )
    ISOPleasant = complex_pleasant / scale

    # E =(e−u)+cos45°(ch−ca)+cos45°(v−m)
    complex_eventful = (
        (results_df.PAQ3.fillna(3) - results_df.PAQ7.fillna(3))
        + proj * (results_df.PAQ4.fillna(3) - results_df.PAQ8.fillna(3))
        + proj * (results_df.PAQ2.fillna(3) - results_df.PAQ6.fillna(3))
    )
    ISOEventful = complex_eventful / scale

    return ISOPleasant, ISOEventful


def calculate_polar_coords(results_df: pd.DataFrame, val_range: tuple = (5, 1), scale_to_one: bool = True):
    """Calculates the polar coordinates

    If a value is missing, by default it is replaced with neutral (3).
    The raw PAQ values should be Likert data from 1 to 5 and the column
    names should match the PAQ_cols given above.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe containing ISD formatted data
    val_range : tuple, optional
        The range of values for the PAQs, by default (5, 1)
    scale_to_one : bool, optional
        Should the x, y coordinates be scaled to (-1, +1), by default True

    Returns
    -------
    tuple
        Polar coordinates
    """
    isopl, isoev = calculate_paq_coords(results_df, val_range=val_range, scale_to_one=scale_to_one)
    r, theta = _convert_to_polar_coords(isopl, isoev)
    return r, theta


def _convert_to_polar_coords(x, y):
    """Convert cartesian coordinates to polar coordinates

    Parameters
    ----------
    x : float
        x coordinate
    y : float
        y coordinate

    Returns
    -------
    tuple
        (r, theta) polar coordinates
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.rad2deg(np.arctan2(y, x))
    return r, theta

def ssm_metrics(
    df: pd.DataFrame,
    paq_cols: list = PAQ_IDS,
    val_range: tuple = (5, 1),
    scale_to_one: bool = True,
    projection: bool = True,
    verbose: int = 0,
):
    """Calculate the SSM metrics for each PAQ

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing ISD formatted data
    paq_cols : list, optional
        List of PAQ columns, by default PAQ_IDS
    val_range : tuple, optional
        The range of values for the PAQs, by default (5, 1)
    scale_to_one : bool, optional
        Should the x, y coordinates be scaled to (-1, +1), by default True
    projection : bool, optional
        Use the trigonometric projection (cos(45)) term for diagonal PAQs, by default True
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    pd.DataFrame
        Dataframe containing the SSM metrics
    """
    # Check that the PAQ columns are present
    if not set(paq_cols).issubset(df.columns):
        raise ValueError("PAQ columns are not present in the dataframe.")

    # # Check that the PAQ values are within the range
    # if not _check_paq_range(df, paq_cols, val_range, verbose):
    #     raise ValueError("PAQ values are not within the specified range.")

    # Calculate the coordinates
    r, theta = calculate_polar_coords(df, val_range=val_range, scale_to_one=scale_to_one)

    mean = np.mean(df[paq_cols], axis=1)
    mean = mean / abs(max(val_range) - min(val_range)) if scale_to_one else mean


    # Calculate the SSM metrics
    df = janitor.add_columns(
        df,
        amp=r,
        delta=theta,
        elev=mean,
    )
    return df


# %%
if __name__ == "__main__":
    import doctest

    TEST_DIR = Path("../../soundscapy/test/test_DB")
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
