"""
Core utility functions for processing soundscape survey data.

This module contains fundamental functions and constants used across
the soundscapy package for handling and analyzing soundscape survey data.
"""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from loguru import logger
from plot_likert.scales import Scale


class PAQ(Enum):
    """Enumeration of Perceptual Attribute Questions (PAQ) names and IDs."""

    PLEASANT = ("pleasant", "PAQ1")
    VIBRANT = ("vibrant", "PAQ2")
    EVENTFUL = ("eventful", "PAQ3")
    CHAOTIC = ("chaotic", "PAQ4")
    ANNOYING = ("annoying", "PAQ5")
    MONOTONOUS = ("monotonous", "PAQ6")
    UNEVENTFUL = ("uneventful", "PAQ7")
    CALM = ("calm", "PAQ8")

    def __init__(self, label: str, id: str) -> None:  # noqa: A002
        """
        Initialize a PAQ enum member.

        Parameters
        ----------
        label : str
            The descriptive label for the PAQ (e.g., 'pleasant').
        id : str
            The standard identifier for the PAQ (e.g., 'PAQ1').

        """
        self.label = label
        self.id = id


PAQ_LABELS = [paq.label for paq in PAQ]
PAQ_IDS = [paq.id for paq in PAQ]


@dataclass
class LikertScale:
    """
    Contains different Likert scale options for survey questions.

    This class provides standardized 5-point Likert scales questions commonly used
    in acoustic and soundscape surveys.

    Attributes:
        PAQ: Agreement scale from "Strongly disagree" to "Strongly agree"
        SOURCE: Source perception scale from "Not at all" to "Dominates completely"
        OVERALL: Quality assessment scale from "Very bad" to "Very good"
        APPROPRIATE: Appropriateness scale from "Not at all" to "Perfectly"
        LOUD: Loudness perception scale from "Not at all" to "Extremely"
        OFTEN: Frequency scale with first-time option from "Never / This is my first time here" to "Very often"
        VISIT: Standard frequency scale from "Never" to "Very often"

    """  # noqa: E501

    paq: Scale = field(
        default_factory=lambda: [
            "Strongly disagree",
            "Somewhat disagree",
            "Neutral",
            "Somewhat agree",
            "Strongly agree",
        ]
    )
    source: Scale = field(
        default_factory=lambda: [
            "Not at all",
            "A little",
            "Moderately",
            "A lot",
            "Dominates completely",
        ]
    )
    overall: Scale = field(
        default_factory=lambda: [
            "Very bad",
            "Bad",
            "Neither bad nor good",
            "Good",
            "Very good",
        ]
    )
    appropriate: Scale = field(
        default_factory=lambda: [
            "Not at all",
            "A little",
            "Moderately",
            "A lot",
            "Perfectly",
        ]
    )
    loud: Scale = field(
        default_factory=lambda: [
            "Not at all",
            "A little",
            "Moderately",
            "Very",
            "Extremely",
        ]
    )
    often: Scale = field(
        default_factory=lambda: [
            "Never / This is my first time here",
            "Rarely",
            "Sometimes",
            "Often",
            "Very often",
        ]
    )
    visit: Scale = field(
        default_factory=lambda: [
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Very often",
        ]
    )


LIKERT_SCALES = LikertScale()

EQUAL_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

# Language-specific angles for PAQs as defined in Aletta et. al. (2024)
LANGUAGE_ANGLES = {
    "eng": (0, 46, 94, 138, 177, 241, 275, 340),
    "arb": (0, 36, 45, 135, 167, 201, 242, 308),
    "cmn": (0, 18, 38, 154, 171, 196, 217, 318),
    "hrv": (0, 84, 93, 160, 173, 243, 273, 354),
    "nld": (0, 43, 111, 125, 174, 257, 307, 341),
    "deu": (0, 64, 97, 132, 182, 254, 282, 336),
    "ell": (0, 72, 86, 133, 161, 233, 267, 328),
    "ind": (0, 53, 104, 123, 139, 202, 284, 308),
    "ita": (0, 57, 104, 143, 170, 274, 285, 336),
    "spa": (0, 41, 103, 147, 174, 238, 279, 332),
    "swe": (0, 66, 87, 146, 175, 249, 275, 335),
    "tur": (0, 55, 97, 106, 157, 254, 289, 313),
}


def return_paqs(
    df: pd.DataFrame, other_cols: list[str] | None = None, *, incl_ids: bool = True
) -> pd.DataFrame:
    """
    Return only the PAQ columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing PAQ data.
    other_cols : List[str], optional
        Other columns to include in the output, by default None.
    incl_ids : bool, optional
        Whether to include ID columns (RecordID, GroupID, etc.), by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the PAQ columns and optionally ID and other specified
        columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'RecordID': [1, 2],
    ...     'PAQ1': [4, 3],
    ...     'PAQ2': [2, 5],
    ...     'PAQ3': [1, 2],
    ...     'PAQ4': [3, 4],
    ...     'PAQ5': [5, 1],
    ...     'PAQ6': [2, 3],
    ...     'PAQ7': [4, 5],
    ...     'PAQ8': [1, 2],
    ...     'OtherCol': ['A', 'B']
    ... })
    >>> return_paqs(df)
       RecordID  PAQ1  PAQ2  PAQ3  PAQ4  PAQ5  PAQ6  PAQ7  PAQ8
    0         1     4     2     1     3     5     2     4     1
    1         2     3     5     2     4     1     3     5     2
    >>> return_paqs(df, incl_ids=False, other_cols=['OtherCol'])
       PAQ1  PAQ2  PAQ3  PAQ4  PAQ5  PAQ6  PAQ7  PAQ8 OtherCol
    0     4     2     1     3     5     2     4     1        A
    1     3     5     2     4     1     3     5     2        B

    """
    cols = PAQ_IDS.copy()

    if incl_ids:
        id_cols = [
            name
            for name in ["RecordID", "GroupID", "SessionID", "LocationID"]
            if name in df.columns
        ]
        cols = id_cols + cols

    if other_cols:
        cols.extend(other_cols)

    logger.debug(f"Returning PAQ columns: {cols}")
    return df[cols]


def rename_paqs(
    df: pd.DataFrame, paq_aliases: list | tuple | dict | None = None
) -> pd.DataFrame:
    """
    Rename the PAQ columns in a DataFrame to standard PAQ IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing PAQ data.
    paq_aliases : Union[Tuple, Dict], optional
        Specify which PAQs are to be renamed. If None, will check if the column names
        are in pre-defined options. If a tuple, the order must match PAQ_IDS.
        If a dict, keys are current names and values are desired PAQ IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed PAQ columns.

    Raises
    ------
    ValueError
        If paq_aliases is not a tuple, list, or dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'pleasant': [4, 3],
    ...     'vibrant': [2, 5],
    ...     'other_col': [1, 2]
    ... })
    >>> rename_paqs(df)
       PAQ1  PAQ2  other_col
    0     4     2          1
    1     3     5          2
    >>> df_custom = pd.DataFrame({
    ...     'pl': [4, 3],
    ...     'vb': [2, 5],
    ... })
    >>> rename_paqs(df_custom, paq_aliases={'pl': 'PAQ1', 'vb': 'PAQ2'})
       PAQ1  PAQ2
    0     4     2
    1     3     5

    """
    if paq_aliases is None:
        if any(paq_id in df.columns for paq_id in PAQ_IDS):
            logger.info("PAQs already correctly named.")
            return df
        if any(paq_name in df.columns for paq_name in PAQ_LABELS):
            paq_aliases = PAQ_LABELS

    if isinstance(paq_aliases, list | tuple):
        rename_dict = dict(zip(paq_aliases, PAQ_IDS, strict=False))
    elif isinstance(paq_aliases, dict):
        rename_dict = paq_aliases
    else:
        msg = "paq_aliases must be a tuple, list, or dictionary."
        raise TypeError(msg)

    logger.debug(f"Renaming PAQs with the following mapping: {rename_dict}")
    return df.rename(columns=rename_dict)


def mean_responses(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    Calculate the mean responses for each PAQ group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing PAQ data.
    group : str
        Column name to group by.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean responses for each PAQ group.

    """
    data = return_paqs(df, other_cols=[group], incl_ids=False)
    return data.groupby(group).mean().reset_index()


# Add other utility functions here as needed

# %%
