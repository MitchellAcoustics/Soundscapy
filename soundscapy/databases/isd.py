# Customized functions specifically for the International Soundscape Database
import warnings
from datetime import date
from importlib import resources

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

# Constants and Labels
from soundscapy.utils.surveys import *
from soundscapy.utils.surveys import likert_data_quality, rename_paqs

# Add soundscapy to the Python path

_PAQ_ALIASES = {
    "pleasant": "PAQ1",
    "vibrant": "PAQ2",
    "eventful": "PAQ3",
    "chaotic": "PAQ4",
    "annoying": "PAQ5",
    "monotonous": "PAQ6",
    "uneventful": "PAQ7",
    "calm": "PAQ8",
}


def load():
    """Load example "ISD" [1]_ csv file to DataFrame

    Returns
    -------
    pd.DataFrame
        dataframe of ISD data

    References
    ----------
    .. [1] Mitchell, Andrew, Oberman, Tin, Aletta, Francesco, Erfanian, Mercede, Kachlicka, Magdalena, Lionello, Matteo, & Kang, Jian. (2022). The International Soundscape Database: An integrated multimedia database of urban soundscape surveys -- questionnaires with acoustical and contextual information (0.2.4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6331810
    """

    with resources.path("soundscapy.data", "ISD v1.0 Data.csv") as f:
        data = pd.read_csv(f)
    data = rename_paqs(data, _PAQ_ALIASES)

    return data


def load_zenodo(version="latest"):
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
    version = version.lower()

    version = "v1.0.1" if version == "latest" else version
    if version in ["v0.2.0", "v0.2.1"]:
        url = "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx"
        file_type = "excel"

    elif version in ["v0.2.2"]:
        url = "https://zenodo.org/record/5705908/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"
        file_type = "excel"

    elif version in ["v0.2.3"]:
        url = "https://zenodo.org/record/5914762/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"
        file_type = "excel"

    elif version in ["v1.0.0", "v1.0.1"]:
        url = "https://zenodo.org/records/10639661/files/ISD%20v1.0%20Data.csv"
        file_type = "csv"

    else:
        raise ValueError(f"Version {version} not recognised.")

    data = (
        pd.read_csv(url)
        if file_type == "csv"
        else pd.read_excel(url, engine="openpyxl")
    )

    data = rename_paqs(data, _PAQ_ALIASES)

    return data


def validate(
    df: pd.DataFrame,
    paq_aliases: Union[List, Dict] = _PAQ_ALIASES,
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
    l = likert_data_quality(df, verbose, allow_paq_na, val_range)
    if l is None:
        excl_df = None
    else:
        excl_df = df.iloc[l, :]
        df = df.drop(df.index[l])
    return df, excl_df


def _isd_select(data: pd.DataFrame, select_by, condition):
    """
    General function to select by ID variables.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe
    select_by : str
        Column name of the ID variable to select by
    condition : str, list, or tuple
        IDs to select from the dataframe

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    if isinstance(condition, (str, int)):
        return data.query(f"{select_by} == @condition", engine="python")
    elif isinstance(condition, (list, tuple)):
        return data.query(f"{select_by} in @condition")
    else:
        raise TypeError("Should be either a str, int, list, or tuple.")


def select_record_ids(data, record_ids):
    """Filter the dataframe by RecordID

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe
    record_ids : list
        List of RecordIDs to filter by

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return _isd_select(data, "RecordID", record_ids)


def select_group_ids(data, group_ids):
    """
    Filter the dataframe by GroupID

    Parameters
    ----------
    group_ids : list
        List of GroupIDs to filter by

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return _isd_select(data, "GroupID", group_ids)


def select_session_ids(data, session_ids):
    """
    Filter the dataframe by SessionID

    Parameters
    ----------
    session_ids : list
        List of SessionIDs to select

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return _isd_select(data, "SessionID", session_ids)


def select_location_ids(data, location_ids):
    """Filter the dataframe by LocationID

    Parameters
    ----------
    location_ids : list
        List of LocationIDs to filter by

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return _isd_select(data, "LocationID", location_ids)


def remove_lockdown(data):
    """
    Remove the Lockdown data from ISD data.

    Parameters
    ----------
    data : pd.DataFrame
        ISD DataFrame

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with Lockdown data removed.
    """
    warnings.warn(
        "Lockdown data no longer included in ISD past v1.0. This function will be deprecated in sspy v1.0",
        PendingDeprecationWarning,
        stacklevel=2,
    )
    return data.query("Lockdown == 0")


def describe_location(data, location, type="percent", pl_threshold=0, ev_threshold=0):
    """
    Return a summary of the data

    Parameters
    ----------
    location : str
        Location to describe
    type : str, optional
        Type of summary, by default "percent"
    pl_threshold : int, optional
        PL threshold, by default 0
    ev_threshold : int, optional
        EV threshold, by default 0

    Returns
    -------
    dict
        Summary of the data
    """
    loc_df = select_location_ids(data, location_ids=location)
    count = len(loc_df)
    pl_count = len(loc_df[loc_df["ISOPleasant"] > pl_threshold])
    ev_count = len(loc_df[loc_df["ISOEventful"] > ev_threshold])
    vibrant_count = len(
        loc_df.query(
            "ISOPleasant > @pl_threshold & ISOEventful > @ev_threshold", engine="python"
        )
    )
    chaotic_count = len(
        loc_df.query(
            "ISOPleasant < @pl_threshold & ISOEventful > @ev_threshold", engine="python"
        )
    )
    mono_count = len(
        loc_df.query(
            "ISOPleasant < @pl_threshold & ISOEventful < @ev_threshold", engine="python"
        )
    )
    calm_count = len(
        loc_df.query(
            "ISOPleasant > @pl_threshold & ISOEventful < @ev_threshold", engine="python"
        )
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


def soundscapy_describe(
    df: pd.DataFrame, group_by="LocationID", type="percent"
) -> pd.DataFrame:
    """Return a summary of the data

    Parameters
    ----------
    group_by : str, optional
        Column to group by, by default "LocationID"
    type : str, optional
        Type of summary, by default "percent"

    Returns
    -------
    pd.DataFrame
        Summary of the data
    """
    res = {
        location: describe_location(df, location) for location in df[group_by].unique()
    }

    return pd.DataFrame.from_dict(res, orient="index")
