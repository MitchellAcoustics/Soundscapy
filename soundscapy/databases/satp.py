# Customized functions specifically for the Soundscape Attributes Translation Project (SATP) database

import pandas as pd


def _url_fetch(version: str) -> str:
    """
    Return the URL to fetch the SATP dataset from Zenodo.
    Parameters
    ----------
    version : str
        Version of the dataset to load.
    Returns
    -------
    url : str
        URL to fetch the SATP dataset from Zenodo.
    """
    if version.lower() not in ["latest", "v1.2.1", "v1.2"]:
        raise ValueError(
            "Invalid version. Should be either 'latest', 'v1.2.1', or 'v1.2'."
        )

    version = "v1.2.1" if version == "latest" else version.lower()
    if version in ["v1.2.1", "v1.2"]:
        url = "https://zenodo.org/record/7143599/files/SATP%20Dataset%20v1.2.xlsx"

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
    df : pandas.DataFrame
        Dataframe containing the SATP dataset.
    """
    url = _url_fetch(version)
    return pd.read_excel(url, engine="openpyxl", sheet_name="Main Merge")


def load_participants(version: str = "latest") -> pd.DataFrame:
    """
    Load the SATP participants dataset from Zenodo.
    Parameters
    ----------
    version : str, optional
        Version of the dataset to load. The default is "latest".
    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the SATP participants dataset.
    """
    url = _url_fetch(version)
    return pd.read_excel(url, engine="openpyxl", sheet_name="Participants").drop(
        columns=["Unnamed: 3", "Unnamed: 4"]
    )
