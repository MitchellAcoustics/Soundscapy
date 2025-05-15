"""
Module for handling the International Soundscape Database (ISD).

This module provides functions for loading, validating, and analyzing data
from the International Soundscape Database. It includes utilities for
data retrieval, quality checks, and basic analysis operations.

Notes
-----
The ISD is a large-scale database of soundscape surveys and recordings
collected across multiple cities. This module is designed to work with
the specific structure and content of the ISD.

Examples
--------
>>> import soundscapy.databases.isd as isd
>>> df = isd.load()
>>> isinstance(df, pd.DataFrame)
True
>>> 'PAQ1' in df.columns
True

"""

from importlib import resources

import pandas as pd
from loguru import logger
from pandas import CategoricalDtype
from plot_likert.scales import Scale

from soundscapy.surveys.processing import (
    calculate_iso_coords,
    likert_data_quality,
)
from soundscapy.surveys.survey_utils import (
    LIKERT_SCALES,
    PAQ_IDS,
    PAQ_LABELS,
    rename_paqs,
)

# ISD-specific PAQ aliases
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


def load(locations: list[str] | None = None) -> pd.DataFrame:
    """
    Load the example "ISD" csv file to a DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ISD data.

    Notes
    -----
    This function loads the ISD data from a local CSV file included
    with the soundscapy package.

    References
    ----------
    Mitchell, A., Oberman, T., Aletta, F., Erfanian, M., Kachlicka, M.,
    Lionello, M., & Kang, J. (2022). The International Soundscape Database:
    An integrated multimedia database of urban soundscape surveys --
    questionnaires with acoustical and contextual information (0.2.4) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.6331810

    Examples
    --------
    >>> from soundscapy.surveys.survey_utils import PAQ_IDS
    >>> df = load()
    >>> isinstance(df, pd.DataFrame)
    True
    >>> set(PAQ_IDS).issubset(df.columns)
    True

    """
    isd_resource = resources.files("soundscapy.data").joinpath("ISD v1.0 Data.csv")
    with resources.as_file(isd_resource) as f:
        data = pd.read_csv(f)
    data = rename_paqs(data, _PAQ_ALIASES)
    logger.info("Loaded ISD data from Soundscapy's included CSV file.")

    if locations is not None:
        data = select_location_ids(data, locations)

    return data


def load_zenodo(version: str = "latest") -> pd.DataFrame:
    """
    Automatically fetch and load the ISD dataset from Zenodo.

    Parameters
    ----------
    version : str, optional
        Version number of the dataset to fetch, by default "latest".

    Returns
    -------
    pd.DataFrame
        DataFrame containing ISD data.

    Raises
    ------
    ValueError
        If the specified version is not recognized.

    Notes
    -----
    This function fetches the ISD data directly from Zenodo, allowing
    access to different versions of the dataset.

    Examples
    --------
    >>> from soundscapy.surveys.survey_utils import PAQ_IDS  # doctest: +SKIP
    >>> df = load_zenodo("v1.0.1")  # doctest: +SKIP
    >>> isinstance(df, pd.DataFrame)  # doctest: +SKIP
    True
    >>> set(PAQ_IDS).issubset(df.columns)  # doctest: +SKIP
    True

    """
    version = version.lower()
    version = "v1.0.1" if version == "latest" else version

    url_mapping = {
        "v0.2.0": "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx",
        "v0.2.1": "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx",
        "v0.2.2": "https://zenodo.org/record/5705908/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx",
        "v0.2.3": "https://zenodo.org/record/5914762/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx",
        "v1.0.0": "https://zenodo.org/records/10639661/files/ISD%20v1.0%20Data.csv",
        "v1.0.1": "https://zenodo.org/records/10639661/files/ISD%20v1.0%20Data.csv",
    }

    if version not in url_mapping:
        msg = f"Version {version} not recognised."
        raise ValueError(msg)

    url = url_mapping[version]
    file_type = "csv" if version in ["v1.0.0", "v1.0.1"] else "excel"

    data = (
        pd.read_csv(url)
        if file_type == "csv"
        else pd.read_excel(url, engine="openpyxl")
    )
    data = rename_paqs(data, _PAQ_ALIASES)

    logger.info(f"Loaded ISD data version {version} from Zenodo")
    return data


def validate(
    df: pd.DataFrame,
    paq_aliases: list | dict = _PAQ_ALIASES,
    val_range: tuple[int, int] = (1, 5),
    *,
    allow_paq_na: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Perform data quality checks and validate that the dataset fits the expected format.

    Parameters
    ----------
    df : pd.DataFrame
        ISD style dataframe, including PAQ data.
    paq_aliases : Union[List, Dict], optional
        List of PAQ names (in order) or dict of PAQ names with new names as values.
    allow_paq_na : bool, optional
        If True, allow NaN values in PAQ data, by default False.
    val_range : Tuple[int, int], optional
        Min and max range of the PAQ response values, by default (1, 5).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame | None]
        Tuple containing the cleaned dataframe
        and optionally a dataframe of excluded samples.

    Notes
    -----
    This function renames PAQ columns, checks PAQ data quality, and optionally
    removes rows with invalid or missing PAQ values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'PAQ1': [np.nan, 2, 3, 3], 'PAQ2': [3, 2, 6, 3], 'PAQ3': [2, 2, 3, 3],
    ...     'PAQ4': [1, 2, 3, 3], 'PAQ5': [5, 2, 3, 3], 'PAQ6': [3, 2, 3, 3],
    ...     'PAQ7': [4, 2, 3, 3], 'PAQ8': [2, 2, 3, 3]
    ... })
    >>> clean_df, excl_df = validate(df, allow_paq_na=True)
    >>> clean_df.shape[0]
    2
    >>> excl_df.shape[0]
    2

    """
    logger.info("Validating ISD data")
    data = rename_paqs(df, paq_aliases)

    invalid_indices = likert_data_quality(
        data, val_range=val_range, allow_na=allow_paq_na
    )

    if invalid_indices:
        excl_data = data.iloc[invalid_indices]
        data = data.drop(data.index[invalid_indices])
        logger.info(f"Removed {len(invalid_indices)} rows with invalid PAQ data")
    else:
        excl_data = None
        logger.info("All PAQ data passed quality checks")

    return data, excl_data


def match_col_to_likert_scale(col: str | None) -> Scale:  # noqa: PLR0911
    """
    Match a column in the DataFrame to the Likert scale.

    Parameters
    ----------
    col : str
        Column name to match.
    likert_scale : LikertScale
        Likert scale to match against.

    Returns
    -------
    Scale
        Likert scale object.

    """
    if col in PAQ_IDS or col in PAQ_LABELS:
        return LIKERT_SCALES.paq
    if col in ["traffic_noise", "other_noise", "human_sounds", "natural_sounds"]:
        return LIKERT_SCALES.source
    if col in ["overall_sound_environment"]:
        return LIKERT_SCALES.overall
    if col in ["appropriate"]:
        return LIKERT_SCALES.appropriate
    if col in ["perceived_loud"]:
        return LIKERT_SCALES.loud
    if col in ["visit_often"]:
        return LIKERT_SCALES.often
    if col in ["like_to_visit"]:
        return LIKERT_SCALES.visit

    msg = f"Column {col} does not match any known Likert scale."
    raise ValueError(msg)


def likert_categorical_from_data(
    data: pd.Series,
) -> pd.Categorical:
    """
    Get the Likert labels for a specific column in the DataFrame.

    Parameters
    ----------
    data : pd.Series
        Series containing the data.

    Returns
    -------
    pd.Series
        Series with Likert labels.

    Raises
    ------
    ValueError
        If the column does not match any known Likert scale.

    """
    likert_scale = match_col_to_likert_scale(str(data.name))
    if isinstance(data, pd.Categorical):
        return data

    data = data.astype("int") - 1  # Convert to zero-based index
    codes = data.to_list()

    return pd.Categorical.from_codes(
        codes,
        dtype=CategoricalDtype(categories=likert_scale, ordered=True),
    )


def _isd_select(
    data: pd.DataFrame, select_by: str, condition: str | int | list | tuple
) -> pd.DataFrame:
    """
    General function to select by ID variables.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe.
    select_by : str
        Column name of the ID variable to select by.
    condition : Union[str, int, List, Tuple]
        IDs to select from the dataframe.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Raises
    ------
    TypeError
        If condition is not a str, int, list, or tuple.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'ID': ['A', 'B', 'C', 'D'],
    ...     'Value': [1, 2, 3, 4]
    ... })
    >>> _isd_select(df, 'ID', ['A', 'C'])
      ID  Value
    0  A      1
    2  C      3

    """
    if isinstance(condition, str | int):
        return data.query(f"{select_by} == @condition", engine="python")
    if isinstance(condition, list | tuple):
        return data.query(f"{select_by} in @condition")
    msg = "Should be either a str, int, list, or tuple."
    raise TypeError(msg)


def select_record_ids(
    data: pd.DataFrame, record_ids: str | int | list | tuple
) -> pd.DataFrame:
    """
    Filter the dataframe by RecordID.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe.
    record_ids : Union[str, int, List, Tuple]
        RecordID(s) to filter by.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'RecordID': ['A', 'B', 'C', 'D'],
    ...     'Value': [1, 2, 3, 4]
    ... })
    >>> select_record_ids(df, ['A', 'C'])
      RecordID  Value
    0        A      1
    2        C      3

    """
    return _isd_select(data, "RecordID", record_ids)


def select_group_ids(
    data: pd.DataFrame, group_ids: str | int | list | tuple
) -> pd.DataFrame:
    """
    Filter the dataframe by GroupID.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe.
    group_ids : Union[str, int, List, Tuple]
        GroupID(s) to filter by.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'GroupID': ['G1', 'G1', 'G2', 'G2'],
    ...     'Value': [1, 2, 3, 4]
    ... })
    >>> select_group_ids(df, 'G1')
      GroupID  Value
    0      G1      1
    1      G1      2

    """
    return _isd_select(data, "GroupID", group_ids)


def select_session_ids(
    data: pd.DataFrame, session_ids: str | int | list | tuple
) -> pd.DataFrame:
    """
    Filter the dataframe by SessionID.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe.
    session_ids : Union[str, int, List, Tuple]
        SessionID(s) to filter by.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'SessionID': ['S1', 'S1', 'S2', 'S2'],
    ...     'Value': [1, 2, 3, 4]
    ... })
    >>> select_session_ids(df, ['S1', 'S2'])
      SessionID  Value
    0        S1      1
    1        S1      2
    2        S2      3
    3        S2      4

    """
    return _isd_select(data, "SessionID", session_ids)


def select_location_ids(
    data: pd.DataFrame, location_ids: str | int | list | tuple
) -> pd.DataFrame:
    """
    Filter the dataframe by LocationID.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe.
    location_ids : Union[str, int, List, Tuple]
        LocationID(s) to filter by.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'LocationID': ['L1', 'L1', 'L2', 'L2'],
    ...     'Value': [1, 2, 3, 4]
    ... })
    >>> select_location_ids(df, 'L2')
      LocationID  Value
    2         L2      3
    3         L2      4

    """
    return _isd_select(data, "LocationID", location_ids)


def describe_location(
    data: pd.DataFrame,
    location: str,
    calc_type: str = "percent",
    pl_threshold: float = 0,
    ev_threshold: float = 0,
) -> dict[str, int | float]:
    """
    Return a summary of the data for a specific location.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe.
    location : str
        Location to describe.
    calc_type : str, optional
        Type of summary, either "percent" or "count", by default "percent".
    pl_threshold : float, optional
        Pleasantness threshold, by default 0.
    ev_threshold : float, optional
        Eventfulness threshold, by default 0.

    Returns
    -------
    Dict[str, Union[int, float]]
        Summary of the data for the specified location.

    Examples
    --------
    >>> from soundscapy.surveys.processing import add_iso_coords
    >>> df = pd.DataFrame({
    ...     'LocationID': ['L1', 'L1', 'L2', 'L2'],
    ...     'PAQ1': [4, 2, 3, 5],
    ...     'PAQ2': [3, 5, 2, 4],
    ...     'PAQ3': [2, 4, 1, 3],
    ...     'PAQ4': [1, 3, 4, 2],
    ...     'PAQ5': [5, 1, 5, 1],
    ...     'PAQ6': [4, 2, 3, 5],
    ...     'PAQ7': [3, 5, 2, 4],
    ...     'PAQ8': [2, 4, 1, 3],
    ... })
    >>> df = add_iso_coords(df)
    >>> result = describe_location(df, 'L1')
    >>> set(result.keys()) == {
    ...     'count', 'ISOPleasant', 'ISOEventful', 'pleasant', 'eventful',
    ...     'vibrant', 'chaotic', 'monotonous', 'calm'
    ... }
    True
    >>> result['count']
    2

    """
    loc_df = select_location_ids(data, location_ids=location)
    count = len(loc_df)

    if "ISOPleasant" not in loc_df.columns or "ISOEventful" not in loc_df.columns:
        iso_pleasant, iso_eventful = calculate_iso_coords(loc_df)
        loc_df = loc_df.assign(ISOPleasant=iso_pleasant, ISOEventful=iso_eventful)

    pl_count = (loc_df["ISOPleasant"] > pl_threshold).sum()
    ev_count = (loc_df["ISOEventful"] > ev_threshold).sum()
    vibrant_count = (
        (loc_df["ISOPleasant"] > pl_threshold) & (loc_df["ISOEventful"] > ev_threshold)
    ).sum()
    chaotic_count = (
        (loc_df["ISOPleasant"] < pl_threshold) & (loc_df["ISOEventful"] > ev_threshold)
    ).sum()
    mono_count = (
        (loc_df["ISOPleasant"] < pl_threshold) & (loc_df["ISOEventful"] < ev_threshold)
    ).sum()
    calm_count = (
        (loc_df["ISOPleasant"] > pl_threshold) & (loc_df["ISOEventful"] < ev_threshold)
    ).sum()

    res = {
        "count": count,
        "ISOPleasant": loc_df["ISOPleasant"].mean(),
        "ISOEventful": loc_df["ISOEventful"].mean(),
    }

    if calc_type == "percent":
        res.update(
            {
                "pleasant": pl_count / count,
                "eventful": ev_count / count,
                "vibrant": vibrant_count / count,
                "chaotic": chaotic_count / count,
                "monotonous": mono_count / count,
                "calm": calm_count / count,
            }
        )
    elif calc_type == "count":
        res.update(
            {
                "pleasant": pl_count,
                "eventful": ev_count,
                "vibrant": vibrant_count,
                "chaotic": chaotic_count,
                "monotonous": mono_count,
                "calm": calm_count,
            }
        )
    else:
        msg = "Type must be either 'percent' or 'count'"
        raise ValueError(msg)

    return {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}


def soundscapy_describe(
    df: pd.DataFrame, group_by: str = "LocationID", calc_type: str = "percent"
) -> pd.DataFrame:
    """
    Return a summary of the data grouped by a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        ISD dataframe.
    group_by : str, optional
        Column to group by, by default "LocationID".
    type : str, optional
        Type of summary, either "percent" or "count", by default "percent".

    Returns
    -------
    pd.DataFrame
        Summary of the data.

    Examples
    --------
    >>> from soundscapy.surveys.processing import add_iso_coords
    >>> df = pd.DataFrame({
    ...     'LocationID': ['L1', 'L1', 'L2', 'L2'],
    ...     'PAQ1': [4, 2, 3, 5],
    ...     'PAQ2': [3, 5, 2, 4],
    ...     'PAQ3': [2, 4, 1, 3],
    ...     'PAQ4': [1, 3, 4, 2],
    ...     'PAQ5': [5, 1, 5, 1],
    ...     'PAQ6': [4, 2, 3, 5],
    ...     'PAQ7': [3, 5, 2, 4],
    ...     'PAQ8': [2, 4, 1, 3],
    ... })
    >>> df = add_iso_coords(df)
    >>> result = soundscapy_describe(df)
    >>> isinstance(result, pd.DataFrame)
    True
    >>> result.index.tolist()
    ['L1', 'L2']
    >>> set(result.columns) == {
    ...     'count', 'ISOPleasant', 'ISOEventful', 'pleasant', 'eventful',
    ...     'vibrant', 'chaotic', 'monotonous', 'calm'
    ... }
    True
    >>> result = soundscapy_describe(df, calc_type="count")
    >>> result.loc['L1', 'count']
    2

    """
    res = {
        location: describe_location(df, location, calc_type=calc_type)
        for location in df[group_by].unique()
    }
    return pd.DataFrame.from_dict(res, orient="index")
