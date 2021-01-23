import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

import csv
from datetime import date
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Constants and Labels
from soundscapy.ssid.parameters import (
    CATEGORISED_VARS,
    IGNORE_LIST,
    LOCATION_IDS,
    PARAM_LIST,
    SURVEY_VARS,
)

DEFAULT_CATS = [
    "indexing",
    "sound_source_dominance",
    "raw_PAQs",
    "overall_soundscape",
]

# General helper functions
_flatten = lambda t: [item for sublist in t for item in sublist]

# Dealing with Surveys!
class SurveyFrame(pd.DataFrame):
    # TODO Add Documentation
    # TODO Add Example doctesting
    _analysis_date = date.today().isoformat()

    @property
    def _constructor(self):
        return SurveyFrame

    @classmethod
    def create_empty(
        self,
        variable_categories: list = DEFAULT_CATS,
        add_columns: list = [],
        index=None,
        dtype=None,
    ):
        # input sanity
        if not set(variable_categories).issubset(list(CATEGORISED_VARS.keys())):
            raise ValueError(
                "Category not found in defined sets of variables. See parameters.CATEGORISED_VARS"
            )

        cols = _flatten(
            [CATEGORISED_VARS.get(k, "col_missing") for k in variable_categories]
        )
        if add_columns:
            cols.extend(add_columns)

        return SurveyFrame(columns=cols, index=index, dtype=dtype)

    @classmethod
    def from_csv(
        self,
        filepath,
        clean_cols=False,  # TODO: Change to True when clean_cols func created
        use_RecordID_as_index: bool = True,
        variable_categories: list = DEFAULT_CATS,
        drop_columns=[],
        add_columns=[],
        nrows=None,
    ):
        # input sanity
        if not set(variable_categories).issubset(list(CATEGORISED_VARS.keys())):
            raise ValueError(
                "Category not found in defined sets of variables. See parameters.CATEGORISED_VARS"
            )

        use_cols = _flatten(
            [CATEGORISED_VARS.get(k, "col_missing") for k in variable_categories]
        )
        if add_columns:
            use_cols.extend(add_columns)

        if not variable_categories and not add_columns:
            use_cols = None

        all_cols = _flatten(CATEGORISED_VARS.values())
        index_col = "record_id" if use_RecordID_as_index else None

        # TODO Deal with use_cols not in file, right now raises ValueError from pandas
        # Deal with it using aliases. `try` block?
        # put into own function? def _pandas_read_csv_w_aliases()
        # or maybe just a `+check_col_names` function?

        use_cols = _check_csv_col_names(filepath, use_cols)

        df = pd.read_csv(
            filepath,
            sep=",",
            header=0,
            usecols=use_cols,
            index_col=index_col,
            skipinitialspace=True,
            nrows=nrows,
        )
        df = df.drop(drop_columns, axis=1)

        sf = SurveyFrame(df)
        sf._csv_file = filepath

        if clean_cols:
            sf = sf.clean_cols()

        return sf

    # TODO: clean_cols function
    def clean_cols():
        return None

    # TODO: complex_paqs function


def _check_csv_col_names(filepath, use_cols):
    """Compares the desired columns with those present in the csv file.

    Where a column is requested but not present, we check element-wise through the provided aliases from parameters.SURVEY_VARS and try those instead. If an alias is present in the file, we'll pass it to pandas to read in the csv.

    Parameters
    ----------
    filepath : filepath_or_buffer
        csv file to pass to pandas.read_csv()
    use_cols : list
        columns (variable) labels you'd like to read in

    Returns
    -------
    list
        use_cols with missing items replaced by their alias
    """
    # TODO Tests for _check_csv_col_names()
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)

    # Pull out items in cols but not in csv header
    missing_headers = np.setdiff1d(use_cols, headers)
    if len(missing_headers) > 0:
        for missing_item in missing_headers:
            if missing_item in SURVEY_VARS.keys():
                aliases = SURVEY_VARS[missing_item]["aliases"]
                for alias in aliases:
                    if alias in headers:
                        use_cols.remove(missing_item)
                        use_cols.append(alias)

            else:
                # TODO Search through all aliases to find which label it matches with
                # How?
                warnings.warn(
                    f"Warning: Can't find a matching alias for {missing_item} which is in the csv. Removing it before passing to pandas."
                )
                use_cols.remove(missing_item)

    return use_cols  # Exit if nothing is missing


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

    TEST_DIR = Path("../../soundscapy/test/test_DB")
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
