"""
Module for handling the Soundscape Attributes Translation Project (SATP) database.

This module provides functions for loading and processing data from the
Soundscape Attributes Translation Project database. It includes utilities
for data retrieval from Zenodo and basic data loading operations.

Examples
--------
>>> import soundscapy.databases.satp as satp  # doctest: +SKIP
>>> df = satp.load_zenodo()  # doctest: +SKIP
>>> isinstance(df, pd.DataFrame)  # doctest: +SKIP
True
>>> 'Language' in df.columns  # doctest: +SKIP
True
>>> participants = satp.load_participants()  # doctest: +SKIP
>>> isinstance(participants, pd.DataFrame)  # doctest: +SKIP
True
>>> 'Country' in participants.columns  # doctest: +SKIP
True

"""

import pandas as pd
from loguru import logger


def _url_fetch(version: str) -> str:
    """
    Return the URL to fetch the SATP dataset from Zenodo.

    Parameters
    ----------
    version : str
        Version of the dataset to load.

    Returns
    -------
    str
        URL to fetch the SATP dataset from Zenodo.

    Raises
    ------
    ValueError
        If an invalid version is specified.

    Examples
    --------
    >>> _url_fetch("latest")
    'https://zenodo.org/record/7143599/files/SATP%20Dataset%20v1.2.xlsx'
    >>> _url_fetch("v1.2.1")
    'https://zenodo.org/record/7143599/files/SATP%20Dataset%20v1.2.xlsx'
    >>> _url_fetch("invalid")
    Traceback (most recent call last):
        ...
    ValueError: Invalid version. Should be either 'latest', 'v1.2.1', or 'v1.2'.

    """
    if version.lower() not in ["latest", "v1.2.1", "v1.2"]:
        msg = "Invalid version. Should be either 'latest', 'v1.2.1', or 'v1.2'."
        raise ValueError(msg)

    version = "v1.2.1" if version.lower() == "latest" else version.lower()
    url = "https://zenodo.org/record/7143599/files/SATP%20Dataset%20v1.2.xlsx"

    logger.debug(f"Fetching SATP dataset URL for version: {version}")
    return url


def load_zenodo(version: str = "latest") -> pd.DataFrame:
    """
    Load the SATP dataset from Zenodo.

    Parameters
    ----------
    version : str, optional
        Version of the dataset to load. The default is "latest".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the SATP dataset.

    """
    url = _url_fetch(version)
    data = pd.read_excel(url, engine="openpyxl", sheet_name="Main Merge")
    logger.info(f"Loaded SATP dataset version {version} from Zenodo")
    return data


def load_participants(version: str = "latest") -> pd.DataFrame:
    """
    Load the SATP participants dataset from Zenodo.

    Parameters
    ----------
    version : str, optional
        Version of the dataset to load. The default is "latest".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the SATP participants dataset.

    """
    url = _url_fetch(version)
    data = pd.read_excel(url, engine="openpyxl", sheet_name="Participants")
    data = data.drop(columns=["Unnamed: 3", "Unnamed: 4"])
    logger.info(f"Loaded SATP participants dataset version {version} from Zenodo")
    return data


if __name__ == "__main__":
    import xdoctest

    xdoctest.doctest_module(__file__)
