"""
The module containing functions for dealing with soundscape survey data.

Notes
-----
The functions in this module are designed to be fairly general and can be used with any dataset in a similar format to
the ISD. The key to this is using a simple dataframe/sheet with the following columns:
    Index columns: e.g. LocationID, RecordID, GroupID, SessionID
    Perceptual attributes: PAQ1, PAQ2, ..., PAQ8
    Independent variables: e.g. Laeq, N5, Sharpness, etc.

The key functions of this module are designed to clean/validate datasets, calculate ISO coordinate values or SSM metrics,
filter on index columns. Functions and operations which are specific to a particular dataset are located in their own
modules under `soundscape.databases`.
"""

from pathlib import Path
from typing import Union, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import optimize

# Constants and Labels
from soundscapy.utils.parameters import PAQ_IDS, PAQ_NAMES

DEFAULT_CATS = [
    "indexing",
    "sound_source_dominance",
    "raw_PAQs",
    "overall_soundscape",
]

# General helper functions
_flatten = lambda t: [item for sublist in t for item in sublist]


###########
def return_paqs(df, incl_ids=True, other_cols=None):
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
        dataframe containing only the PAQ columns
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


def rename_paqs(
    df: pd.DataFrame, paq_aliases: Union[Tuple, Dict] = None, verbose: int = 0
) -> pd.DataFrame:
    """
    The rename_paqs function renames the PAQ columns in a dataframe.

    Soundscapy works with PAQ IDs (PAQ1, PAQ2, etc), so if you use labels such as pleasant, vibrant, etc. these will
    need to be renamed.

    It takes as input a pandas DataFrame and returns the same DataFrame with renamed columns.
    If no arguments are passed, it will attempt to rename all of the PAQs based on their column names.

    Parameters
    ----------
        df: pd.DataFrame
            Specify the dataframe to be renamed
        paq_aliases: tuple or dict, optional
            Specify which paqs are to be renamed, by default None.

            If None, will check if the column names are in our pre-defined options (i.e. pleasant, vibrant, etc).

            If a tuple is passed, the order of the tuple must match the order of the PAQs in the dataframe.

            Allow the function to be called with a dictionary of aliases if desired
        verbose: int, optional
            Print out a message if the paqs are already correctly named, by default 0

    Returns
    -------

        A pandas dataframe with the paq_ids column names

    """
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


def likert_data_quality(
    df: pd.DataFrame,
    verbose: int = 0,
    allow_na: bool = False,
    val_range: tuple = (1, 5),
) -> Union[List, None]:
    """Basic check of PAQ data quality

    The likert_data_quality function takes a DataFrame and returns a list of indices that
    should be dropped from the DataFrame. The function checks for:

    - Rows with all values equal to 1 (indicating no PAQ data)

    - Rows with more than 4 NaN values (indicating missing PAQ data)

    - Rows where any value is greater than 5 or less than 1 (indicating invalid PAQ data)

    Parameters
    ----------
        df: pd.DataFrame
            Specify the dataframe to be evaluated
        verbose: int, optional
            Determine whether or not the function should print out information about the data quality check, by default 0
        allow_na: bool
            Ensure that rows with any missing values are dropped, by default False
        val_range: tuple, optional
            Set the range of values that are considered to be valid, by default (1, 5).

    Returns
    -------

        A list of indices that need to be removed from the dataframe

    """

    paqs = return_paqs(df, incl_ids=False)
    l = []
    for i in range(len(paqs)):
        row = paqs.iloc[i]
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
        isopl, isoev = calculate_paq_coords(df, **coord_kwargs)
        df = df.assign(ISOPleasant=isopl, ISOEventful=isoev)
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


def add_iso_coords(
    data,
    scale_to_one: bool = True,
    val_range=(1, 5),
    projection: bool = True,
    names=("ISOPleasant", "ISOEventful"),
    overwrite=False,
):
    """Calculate and add ISO coordinates as new columns in dataframe

    Calls `calculate_paq_coords()`

    Parameters
    ----------
    data : pd.DataFrame
        ISD Dataframe
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
    if names[0] in data.columns:
        if overwrite:
            data = data.drop(names[0], axis=1)
        else:
            raise Warning(
                f"{names[0]} already in dataframe. Use `overwrite` to replace it."
            )
    if names[1] in data.columns:
        if overwrite:
            data = data.drop(names[1], axis=1)
        else:
            raise Warning(
                f"{names[1]} already in dataframe. Use `overwrite` to replace it."
            )
    isopl, isoev = calculate_paq_coords(
        data, scale_to_one=scale_to_one, val_range=val_range, projection=projection
    )
    data = data.assign(**{names[0]: isopl, names[1]: isoev})
    return data


def calculate_polar_coords(results_df: pd.DataFrame, scaling: str = "iso"):
    """Calculates the polar coordinates

    Based on the calculation given in Gurtman and Pincus (2003), pg. 416.

    The raw PAQ values should be Likert data from 1 to 5 and the column
    names should match the PAQ_cols given above.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe containing ISD formatted data
    scaling : str, optional
        The scaling to use for the polar coordinates, by default 'iso'

        Options are 'iso', 'gurtman', and 'none'

        For 'iso', the cartesian coordinates are scaled to (-1, +1) according to the basic
        method given in ISO12913.

        For 'gurtman', the polar coordinates are scaled according to the method given in
        Gurtman and Pincus (2003), pg. 416.

        For 'none', no scaling is applied.

    Returns
    -------
    tuple
        Polar coordinates
    """
    # raise error if scaling is not one of the options
    if scaling not in ["iso", "gurtman", "none"]:
        raise ValueError(
            f"Scaling must be one of 'iso', 'gurtman', or 'none', not {scaling}"
        )

    scale_to_one = True if scaling == "iso" else False
    isopl, isoev = calculate_paq_coords(results_df, scale_to_one=scale_to_one)

    if scaling == "gurtman":
        isopl = isopl * 0.25
        isoev = isoev * 0.25

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
    r = np.sqrt(x**2 + y**2)
    theta = np.rad2deg(np.arctan2(y, x))
    return r, theta


def ssm_metrics(
    df: pd.DataFrame,
    paq_cols: list = PAQ_IDS,
    method: str = "cosine",
    val_range: tuple = (5, 1),
    scale_to_one: bool = True,
    angles: Tuple = (0, 45, 90, 135, 180, 225, 270, 315),
):
    """Calculate the SSM metrics for each response

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing ISD formatted data
    paq_cols : list, optional
        List of PAQ columns, by default PAQ_IDS
    method : str, optional
        Method by which to calculate the SSM, by default 'cosine'
        'cosine' fits a cosine model to the data, using the Structural Summary Method developed
        by Gurtman (1994; Gurtman & Balakrishnan, 1998).
        'polar_conversion' directly converts the ISO coordinates to polar coordinates.

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

    if method == "polar":
        # Calculate the coordinates
        vl, theta = calculate_polar_coords(df)

        mean = np.mean(df[paq_cols], axis=1)
        mean = mean / abs(max(val_range) - min(val_range)) if scale_to_one else mean

        # Calculate the SSM metrics
        df = df.assign(
            vl=vl,
            theta=theta,
            mean_level=mean,
        )
        return df

    elif method == "cosine":
        ssm_df = df[paq_cols].apply(
            lambda y: ssm_cosine_fit(y, angles=angles), axis=1, result_type="expand"
        )

        df = df.assign(
            amp=ssm_df.iloc[:, 0],
            delta=ssm_df.iloc[:, 1],
            elev=ssm_df.iloc[:, 2],
            dev=ssm_df.iloc[:, 3],
            r2=ssm_df.iloc[:, 4],
        )
        return df

    else:
        raise ValueError("Method must be either 'polar' or 'cosine'.")


def ssm_cosine_fit(
    y,
    angles=(0, 45, 90, 135, 180, 225, 270, 315),
    bounds=([0, 0, 0, -np.inf], [np.inf, 360, np.inf, np.inf]),
):
    """Fit a cosine model to the data

    Parameters
    ----------
    angles : list
        List of angles
    y : list
        List of y values
    bounds : tuple
        Bounds for the parameters

    Returns
    -------
    tuple
        (amp, delta, elev, dev)
    """

    def form(theta, amp, delta, elev, dev):
        return elev + amp * np.cos(np.radians(theta - delta)) + dev

    param, covariance = optimize.curve_fit(
        form,
        xdata=angles,
        ydata=y,
        bounds=bounds,
    )
    r2 = _r2_score(y, form(angles, *param))
    amp, delta, elev, dev = param
    return amp, delta, elev, dev, r2


def _r2_score(y, y_hat):
    """Calculates the R2 score

    Parameters
    ----------
    y : np.ndarray
        Actual values
    y_hat : np.ndarray
        Predicted values

    Returns
    -------
    float
        R2 score
    """
    y_bar = np.mean(y)
    ss_tot = np.sum((y - y_bar) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    return 1 - (ss_res / ss_tot)


# %%
if __name__ == "__main__":
    import doctest

    TEST_DIR = Path("../../soundscapy/test/test_DB")
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
