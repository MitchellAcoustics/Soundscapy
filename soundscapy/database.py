from datetime import time
from pathlib import Path
import csv 

import numpy as np
import pandas as pd

# Constants and Labels
from soundscapy.parameters import (
    CATEGORISED_VARS,
    IGNORE_LIST,
    LOCATION_IDS,
    PARAM_LIST,
)


# Survey database functions
def fill_missing_paqs(df: pd.DataFrame, features, fill_val=3):
    df[features] = df[features].fillna(value=fill_val)
    return df


def calculate_complex_paqs(
    df: pd.DataFrame,
    scale_to_one: bool = True,
    fill_na: bool = True,
    fill_val=3,
    append_var_names="",
):
    """Calculate the complex Pleasant and Eventful projections of the PAQs.
    Uses the projection formulae from ISO  12913 Part 3:

    P =(p−a)+cos45°*(ca−ch)+cos45°*(v−m)
    E =(e−u)+cos45°*(ch−ca)+cos45°*(v−m)

    Parameters
    ----------
    scale_to_one : bool, optional
        Scale the complex values from -1 to 1, by default True
    fill_na : bool, optional
        Fill missing raw_PAQ values, by default True
    fill_val : int, optional
        Value to fill missing raw_PAQs with, by default 3

    Returns
    -------
    (pd.Series, pd.Series)
        pandas Series containing the new complex Pleasant and Eventful vectors
    """
    features = CATEGORISED_VARS["raw_PAQs"]
    features = [var + append_var_names for var in features]

    if fill_na:
        df = fill_missing_paqs(df, features, fill_val=fill_val)

    # TODO: Add check for raw_PAQ column names
    # TODO: add handling for if sf already contains Pleasant and Eventful values

    proj = np.cos(np.deg2rad(45))
    scale = 4 + np.sqrt(32)

    # TODO: Add if statements for too much missing data
    # P =(p−a)+cos45°(ca−ch)+cos45°(v−m)
    complex_pleasant = (
        (df["pleasant" + append_var_names] - df["annoying" + append_var_names])
        + proj * (df["calm" + append_var_names] - df["chaotic" + append_var_names])
        + proj
        * (df["vibrant" + append_var_names] - df["monotonous" + append_var_names])
    )
    Pleasant = complex_pleasant / scale if scale_to_one else complex_pleasant

    # E =(e−u)+cos45°(ch−ca)+cos45°(v−m)
    complex_eventful = (
        (df["eventful" + append_var_names] - df["uneventful" + append_var_names])
        + proj * (df["chaotic" + append_var_names] - df["calm" + append_var_names])
        + proj
        * (df["vibrant" + append_var_names] - df["monotonous" + append_var_names])
    )
    Eventful = complex_eventful / scale if scale_to_one else complex_eventful

    return Pleasant, Eventful


# Parsing HEAD Acoustics files
def time_series_from_head_csv(filepath):
    timeseries = pd.read_csv(
        filepath,
        header=None,
        usecols=[0,1],
        index_col=0,
        sep=",",
        squeeze=True,
        skiprows=27
    )
    timeseries = timeseries.rename("data")
    with open(filepath) as csvDataFile:
        recording = list(csv.reader(csvDataFile))[0][0]
        recording = recording.split("	")[-1]

    return timeseries, recording


# Dealing with Directories!
def collect_all_dirs(
    root_directory: Path,
    location_ids: list,
    param_list: list,
    include_TS: bool = True,
    include_spectrum: bool = True,
    include_WAV: bool = True,
):
    """Iterate throughout the SSID DB file structure to extract TimeSeries, SpectrumData, WAV directory paths.

    Parameters
    ----------
    root_directory : Path
        The city-level SSID directory
    location_ids : list
        A subset of LocationIDs to include in the filepath collection
    param_list : list
        A subset of parameters to include in the filepath collection
    include_TS : bool, optional
        Collect TimeSeries dirs?, by default True
    include_spectrum : bool, optional
        Collect SpectrumData dirs?, by default True
    include_WAV : bool, optional
        Collect WAV dirs?, by default True

    Returns
    -------
    tuple of lists
        A tuple containing the full lists of TimeSeries, SpectrumData, and WAV directories.
        These lists contain WindowsPath objects.
    
    Examples
    ________
    >>> full_ts_list, full_spectrum_list, full_wav_list = collect_all_dirs(TEST_DIR, LOCATION_IDS["Test"], PARAM_LIST)
    >>> len(full_ts_list)
    33
    >>> len(full_spectrum_list)
    6
    >>> len(full_wav_list)
    3
    """

    # Input tests
    if not isinstance(root_directory, Path):
        raise TypeError(
            "The directory should be provided as a WindowsPath object from pathlib."
        )
    if not root_directory.is_dir():
        raise FileNotFoundError("Path does not exist.", root_directory.absolute())

    # Collect list of session id directories
    session_dirs = [session for session in root_directory.iterdir() if session.is_dir()]
    session_dirs = [session for session in session_dirs if "OFF" in session.name]

    new_session_dirs = [
        [session for location in location_ids if (location in session.name)]
        for session in session_dirs
    ]

    session_dirs = [
        val for sublist in new_session_dirs for val in sublist
    ]  # Remove blank entries

    bin_dirs = []
    for session in session_dirs:
        bin_dir = [
            child
            for child in session.iterdir()
            if child.is_dir() and "BIN" in child.name
        ][0]
        bin_dirs.append(bin_dir)

    if include_TS:
        # Collect Time Series parameter directories
        full_ts_list = _ts_dirs(bin_dirs, param_list)
    else:
        full_ts_list = []

    if include_spectrum:
        # Collect Spectrum directories
        full_spectrum_list = _spectrum_dirs(bin_dirs)
    else:
        full_spectrum_list = []

    if include_WAV:
        # Collect WAV directories
        full_wav_list = _wav_dirs(bin_dirs)
    else:
        full_wav_list = []

    return full_ts_list, full_spectrum_list, full_wav_list


def _spectrum_dirs(bin_dirs):
    spectrum_dirs = []
    for directory in bin_dirs:
        spectrum_dir = [
            child
            for child in directory.iterdir()
            if child.is_dir() and "SpectrumData" in child.name
        ][0]
        spectrum_dirs.append(spectrum_dir)

    full_spectrum_list = []
    for directory in spectrum_dirs:
        spectrum_dir = [child for child in directory.iterdir() if child.is_dir()]
        full_spectrum_list.append(spectrum_dir)

    full_spectrum_list = [val for sublist in full_spectrum_list for val in sublist]
    return full_spectrum_list


def _ts_dirs(bin_dirs, param_list):
    # Collect Time Series parameter directories
    ts_dirs = []
    for directory in bin_dirs:
        ts_dir = [
            child
            for child in directory.iterdir()
            if child.is_dir() and "TimeSeries" in child.name
        ][0]
        ts_dirs.append(ts_dir)

    param_dirs = []
    for directory in ts_dirs:
        param_dir = [child for child in directory.iterdir() if child.is_dir()]
        param_dirs.append(param_dir)
    param_dirs = [val for sublist in param_dirs for val in sublist]

    full_ts_list = [
        [directory for param in param_list if (param + "_TS" in directory.name)]
        for directory in param_dirs
    ]

    full_ts_list = [val for sublist in full_ts_list for val in sublist]
    return full_ts_list


def _wav_dirs(bin_dirs):
    # Collect Wav directories
    wav_dirs = []
    for directory in bin_dirs:
        wav_dir = [
            child
            for child in directory.iterdir()
            if child.is_dir() and "WAV" in child.name
        ][0]
        wav_dirs.append(wav_dir)

    return wav_dirs


if __name__ == "__main__":
    import doctest

    TEST_DIR = Path("test_DB")
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)

