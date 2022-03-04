import janitor

# Add soundscapy to the Python path
import sys

from matplotlib.pyplot import switch_backend
sys.path.append('..')

from pathlib import Path

import numpy as np
import pandas as pd

# Constants and Labels
from soundscapy.ssid.parameters import CATEGORISED_VARS, PAQ_IDS, PARAM_LIST, SURVEY_VARS, PAQ_NAMES
from soundscapy.ssid.plotting import default_bw_adjust, default_figsize

DEFAULT_CATS = [
    "indexing",
    "sound_source_dominance",
    "raw_PAQs",
    "overall_soundscape",
]

# General helper functions
_flatten = lambda t: [item for sublist in t for item in sublist]

###########

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
    version = "v0.2.3" if version == "latest" else version
    
    if version == "V0.2.1":
        url = "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx"
    elif version in ["V0.2.2", "v0.2.2"]:
        url = "https://zenodo.org/record/5705908/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"
    elif version in ["v0.2.3", "V0.2.3"]:
        url = "https://zenodo.org/record/5914762/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"

    return pd.read_excel(url)


def validate_dataset(df, paq_aliases=None, allow_na=False, verbose=1, val_range=(5,1)):
    if verbose > 0:
        print("Renaming PAQ columns.")
    df = rename_paqs(df, paq_aliases)

    if verbose > 0:
        print("Checking PAQ data quality.")
    if l := paq_data_quality(df, verbose, allow_na, val_range):
        df = df.drop(df.index[l])

    return df

def rename_paqs(df, paq_aliases=None, verbose=0):
    if paq_aliases is None:
        if any(i in b for i in PAQ_NAMES for b in df.columns):
            if verbose > 0:
                print("PAQs already correctly named.")
            return df
        if any(i in b for i in PAQ_IDS for b in df.columns):
            paq_aliases = PAQ_IDS
            
    if type(paq_aliases) == list:
        return df.rename(columns=
                  {
                      paq_aliases[0]: PAQ_NAMES[0],
                      paq_aliases[1]: PAQ_NAMES[1],
                      paq_aliases[2]: PAQ_NAMES[2],
                      paq_aliases[3]: PAQ_NAMES[3],
                      paq_aliases[4]: PAQ_NAMES[4],
                      paq_aliases[5]: PAQ_NAMES[5],
                      paq_aliases[6]: PAQ_NAMES[6],
                      paq_aliases[7]: PAQ_NAMES[7]
                  })
    elif type(paq_aliases) == dict:
        return df.rename(columns=paq_aliases)
    
def paq_data_quality(df, verbose=0, allow_na=False, val_range=(5,1)):
    paqs = df.isd.return_paqs(incl_ids=False)
    l = []
    for i in range(len(paqs)):
        row = paqs.iloc[i]
        if allow_na and row.isna().sum() == 8: # if we allow completely missing data (i.e. lockdown data)to be included, then we skip over it here
            continue
        if (
            row['pleasant']
            == row['vibrant']
            == row['eventful']
            == row['chaotic']
            == row['annoying']
            == row['monotonous']
            == row['uneventful']
            == row['calm']
            and row.sum() != np.mean(val_range)
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


def simulation(n=3000, add_paq_coords=False, **coord_kwargs):
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
    df = pd.DataFrame(np.random.randint(1, 5, size=(n, 8)), columns=PAQ_NAMES)
    if add_paq_coords:
        ISOPl, ISOEv = calculate_paq_coords(df, **coord_kwargs)
        df = janitor.add_columns(df, ISOPleasant=ISOPl, ISOEventful=ISOEv)
    return df


def calculate_paq_coords(
    results_df: pd.DataFrame,
    scale_to_one: bool = True,
    val_range: tuple = (5,1),
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
    scale = _circ_scale(val_range, proj) if scale_to_one else 1

    # TODO: Add if statements for too much missing data
    # P =(p−a)+cos45°(ca−ch)+cos45°(v−m)
    complex_pleasant = (
        (results_df.pleasant.fillna(3) - results_df.annoying.fillna(3))
        + proj * (results_df.calm.fillna(3) - results_df.chaotic.fillna(3))
        + proj * (results_df.vibrant.fillna(3) - results_df.monotonous.fillna(3))
    )
    ISOPleasant = complex_pleasant / scale

    # E =(e−u)+cos45°(ch−ca)+cos45°(v−m)
    complex_eventful = (
        (results_df.eventful.fillna(3) - results_df.uneventful.fillna(3))
        + proj * (results_df.chaotic.fillna(3) - results_df.calm.fillna(3))
        + proj * (results_df.vibrant.fillna(3) - results_df.monotonous.fillna(3))
    )
    ISOEventful = complex_eventful / scale

    return ISOPleasant, ISOEventful


def _circ_scale(range, proj):
    diff = max(range)-min(range)
    return diff + diff*np.sqrt(2)


def convert_column_to_index(df, col="GroupID", drop=False):
    """Reassign an existing column as the dataframe index"""
    assert col in df.columns, f"col: {col} not found in dataframe"
    df.index = df[col]
    if drop:
        df = df.drop(col, axis=1)
    return df


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
    >>> full_ts_list, full_spectrum_list, full_wav_list = collect_all_dirs(TEST_DIR, ["LocationA", "LocationB"], PARAM_LIST)
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
        raise FileNotFoundError("Path does not exist.")

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

    full_ts_list = _ts_dirs(bin_dirs, param_list) if include_TS else []
    full_spectrum_list = _spectrum_dirs(bin_dirs) if include_spectrum else []
    full_wav_list = _wav_dirs(bin_dirs) if include_WAV else []
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

    TEST_DIR = Path("../../soundscapy/test/test_DB")
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
