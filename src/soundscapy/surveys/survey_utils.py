# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""
Core utility functions for processing soundscape survey data.

This module contains fundamental functions and constants used across
the soundscapy package for handling and analyzing soundscape survey data.
"""

from functools import partial

import pandas as pd
import pandera.pandas as pa
from loguru import logger
from pandera.typing.pandas import DataFrame, Series

from soundscapy import constants
from soundscapy._deprecated import moved_to

AllowNan = partial(pa.Field, nullable=True)

__getattr__ = moved_to(
    __name__,
    "soundscapy.constants",
    [
        "PAQ",
        "PAQ_IDS",
        "PAQ_LABELS",
        "LikertScale",
        "LIKERT_SCALES",
        "EQUAL_ANGLES",
        "LANGUAGE_ANGLES",
    ],
)


class PAQDfSchema(pa.DataFrameModel):
    """
    Pandera schema for validating PAQ (Perceptual Attribute Questions) DataFrames.

    This schema defines the expected structure and data types for DataFrames containing
    soundscape survey data with PAQ responses and associated metadata. It includes
    automatic column name coercion to standardize various input formats.

    Attributes
    ----------
    PAQ1-PAQ8 : Series[float]
        Perceptual Attribute Question responses (1-8) on a Likert scale.
        Nullable to allow for missing responses.
    language : Series[str] | None
        Language code for the survey responses. Optional field.
    location_id : Series[str] | None
        Identifier for the survey location. Optional field.
    session_id : Series[str] | None
        Identifier for the survey session. Optional field.
    group_id : Series[str] | None
        Identifier for the survey group. Optional field.
    record_id : Series[str] | None
        Unique identifier for each survey record. Optional field.

    """

    # PAQ response columns - float values representing Likert scale responses
    PAQ1: Series[float] = AllowNan()  # Pleasant
    PAQ2: Series[float] = AllowNan()  # Vibrant
    PAQ3: Series[float] = AllowNan()  # Eventful
    PAQ4: Series[float] = AllowNan()  # Chaotic
    PAQ5: Series[float] = AllowNan()  # Annoying
    PAQ6: Series[float] = AllowNan()  # Monotonous
    PAQ7: Series[float] = AllowNan()  # Uneventful
    PAQ8: Series[float] = AllowNan()  # Calm

    # Metadata columns - all optional string identifiers
    language: Series[str] | None = AllowNan()  # Survey language code
    location_id: Series[str] | None = AllowNan()  # Location identifier
    session_id: Series[str] | None = AllowNan()  # Session identifier
    group_id: Series[str] | None = AllowNan()  # Group identifier
    record_id: Series[str] | None = AllowNan()  # Record identifier

    @pa.dataframe_parser
    def column_name_coercion(cls, df: DataFrame) -> DataFrame:  # noqa: N805
        """
        Coerce column names to standardized format for PAQ data.

        This parser automatically renames columns to match the expected schema:

        - PAQ label names (e.g., 'pleasant') to PAQ IDs (e.g., 'PAQ1')
        - Legacy ID column names to lowercase snake_case format

        Parameters
        ----------
        cls
            The schema class (automatically passed by pandera).
        df
            Input DataFrame with potentially non-standard column names.

        Returns
        -------
        :
            DataFrame with standardized column names.

        """
        # Create mapping from PAQ labels to standard PAQ IDs
        rename_dict = dict(zip(constants.PAQ_LABELS, constants.PAQ_IDS, strict=False))

        # Add mappings for legacy ID column names to snake_case format
        rename_dict.update(
            {
                "LocationID": "location_id",
                "SessionID": "session_id",
                "GroupID": "group_id",
                "RecordID": "record_id",
            }
        )
        return df.rename(columns=rename_dict)


def return_paqs(
    df: pd.DataFrame, other_cols: list[str] | None = None, *, incl_ids: bool = True
) -> pd.DataFrame:
    """
    Return only the PAQ columns from a DataFrame.

    Parameters
    ----------
    df
        Input DataFrame containing PAQ data.
    other_cols
        Other columns to include in the output, by default None.
    incl_ids
        Whether to include ID columns (RecordID, GroupID, etc.), by default True.

    Returns
    -------
    :
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
    cols = constants.PAQ_IDS.copy()

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
    df
        Input DataFrame containing PAQ data.
    paq_aliases
        Specify which PAQs are to be renamed. If None, will check if the column names
        are in pre-defined options. If a tuple, the order must match PAQ_IDS.
        If a dict, keys are current names and values are desired PAQ IDs.

    Returns
    -------
    :
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
        if any(paq_id in df.columns for paq_id in constants.PAQ_IDS):
            logger.info("PAQs already correctly named.")
            return df
        if any(paq_name in df.columns for paq_name in constants.PAQ_LABELS):
            paq_aliases = constants.PAQ_LABELS

    if isinstance(paq_aliases, list | tuple):
        rename_dict = dict(zip(paq_aliases, constants.PAQ_IDS, strict=False))
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
    df
        Input DataFrame containing PAQ data.
    group
        Column name to group by.

    Returns
    -------
    :
        DataFrame with mean responses for each PAQ group.

    """
    data = return_paqs(df, other_cols=[group], incl_ids=False)
    return data.groupby(group).mean().reset_index()


# Add other utility functions here as needed
