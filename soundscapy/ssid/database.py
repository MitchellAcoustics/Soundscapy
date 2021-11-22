import os
import sys

import janitor

from soundscapy.ssid import plotting

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

import csv
from datetime import date
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Constants and Labels
from .parameters import CATEGORISED_VARS, PARAM_LIST, SURVEY_VARS, PAQ_COLS
from .plotting import default_bw_adjust, default_figsize

DEFAULT_CATS = [
    "indexing",
    "sound_source_dominance",
    "raw_PAQs",
    "overall_soundscape",
]

# General helper functions
_flatten = lambda t: [item for sublist in t for item in sublist]


def load_isd_dataset(
    version="latest",
    clean_cols=False,
    use_RecordID_as_index: bool = True,
    drop_columns=[],
    add_columns=[],
    **read_kwargs,
):
    # TODO: Write docs
    version = "V0.2.2" if version == "latest" else version
    urls = {
        "V0.2.1": "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx",
        "V0.2.2": "https://zenodo.org/record/5705908/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx",
    }

    df = pd.read_excel(urls[version], header=0, **read_kwargs)
    df = df.drop(drop_columns, axis=1)
    sf = SurveyFrame(df)
    if use_RecordID_as_index:
        sf = sf.convert_column_to_index("RecordID")
    sf._version = version
    return sf


def simulated_dataset(
    n=3000, add_complex_paqs=False, **complex_kwargs,
):
    sf = SurveyFrame(np.random.randint(1, 5, size=(n, 8)), columns=PAQ_COLS)
    if add_complex_paqs:
        sf.add_complex_paqs(**complex_kwargs)
    return sf


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
    def load_isd_dataset(
        self,
        version="latest",
        clean_cols=False,
        use_RecordID_as_index: bool = True,
        drop_columns=[],
        add_columns=[],
        **read_kwargs,
    ):
        return load_isd_dataset()

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
        # TODO: Write docs
        # input sanity
        if not set(variable_categories).issubset(list(CATEGORISED_VARS.keys())):
            raise ValueError(
                "Category not found in defined sets of variables. See parameters.CATEGORISED_VARS"
            )

        use_cols = SurveyFrame._get_the_cols(variable_categories, add_columns)

        all_cols = _flatten(CATEGORISED_VARS.values())
        index_col = "record_id" if use_RecordID_as_index else None
        index_col = False if "record_id" not in use_cols else index_col

        # TODO Deal with use_cols not in file, right now raises ValueError from pandas
        # Deal with it using aliases. `try` block?
        # put into own function? def _pandas_read_csv_w_aliases()
        # or maybe just a `+check_col_names` function?

        use_cols = SurveyFrame._check_csv_col_names(filepath, use_cols)

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

    # ! from_excel not really tested!
    @classmethod
    def from_excel(
        self,
        filepath,
        clean_cols=False,  # TODO: Change to True when clean_cols func created
        use_RecordID_as_index: bool = True,
        variable_categories: list = DEFAULT_CATS,
        drop_columns=[],
        add_columns=[],
        nrows=None,
    ):
        # TODO: Write docs
        # input sanity
        if not set(variable_categories).issubset(list(CATEGORISED_VARS.keys())):
            raise ValueError(
                "Category not found in defined sets of variables. See parameters.CATEGORISED_VARS"
            )

        use_cols = SurveyFrame._get_the_cols(variable_categories, add_columns)

        all_cols = _flatten(CATEGORISED_VARS.values())
        index_col = "record_id" if use_RecordID_as_index else None
        index_col = False if "record_id" not in use_cols else index_col

        # TODO Deal with use_cols not in file, right now raises ValueError from pandas
        # Deal with it using aliases. `try` block?
        # put into own function? def _pandas_read_csv_w_aliases()
        # or maybe just a `+check_col_names` function?

        use_cols = SurveyFrame._check_excel_col_names(filepath, use_cols)

        df = pd.read_excel(
            filepath, header=0, usecols=use_cols, index_col=index_col, nrows=nrows,
        )
        df = df.drop(drop_columns, axis=1)

        sf = SurveyFrame(df)
        sf._csv_file = filepath

        if clean_cols:
            sf = sf.clean_cols()

        return sf

    @staticmethod
    def _get_the_cols(variable_categories, add_columns):
        use_cols = _flatten(
            [CATEGORISED_VARS.get(k, "col_missing") for k in variable_categories]
        )
        if add_columns:
            use_cols.extend(add_columns)

        if not variable_categories and not add_columns:
            use_cols = None
        return use_cols

    # TODO: clean_cols function
    # TODO: complex_paqs function

    @staticmethod
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

    @staticmethod
    def _check_excel_col_names(filepath, use_cols):
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
        headers = pd.read_excel(filepath, header=None, nrows=1).iloc[0].array

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

    def fill_missing_paqs(self, fill_val=3):
        self[CATEGORISED_VARS["raw_PAQs"]] = self[CATEGORISED_VARS["raw_PAQs"]].fillna(
            value=fill_val
        )
        return self

    def calculate_complex_paqs(
        self,
        scale_to_one: bool = True,
        projection: bool = True,
        fill_na: bool = True,
        fill_val=3,
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
        if fill_na:
            self = self.fill_missing_paqs(fill_val=fill_val)

        # TODO: Add check for raw_PAQ column names
        # TODO: add handling for if sf already contains ISOPleasant and ISOEventful values

        proj = np.cos(np.deg2rad(45)) if projection else 1
        scale = 4 + np.sqrt(32)

        # TODO: Add if statements for too much missing data
        # P =(p−a)+cos45°(ca−ch)+cos45°(v−m)
        complex_pleasant = (
            (self.pleasant.fillna(0) - self.annoying.fillna(0))
            + proj * (self.calm.fillna(0) - self.chaotic.fillna(0))
            + proj * (self.vibrant.fillna(0) - self.monotonous.fillna(0))
        )
        ISOPleasant = complex_pleasant / scale if scale_to_one else complex_pleasant

        # E =(e−u)+cos45°(ch−ca)+cos45°(v−m)
        complex_eventful = (
            (self.eventful.fillna(0) - self.uneventful.fillna(0))
            + proj * (self.chaotic.fillna(0) - self.calm.fillna(0))
            + proj * (self.vibrant.fillna(0) - self.monotonous.fillna(0))
        )
        ISOEventful = complex_eventful / scale if scale_to_one else complex_eventful

        return ISOPleasant, ISOEventful

    def convert_column_to_index(self, col="GroupID", drop=False):
        """Reassign an existing column as the dataframe index"""
        assert col in self.columns, f"col: {col} not found in dataframe"
        self.index = self[col]
        if drop:
            self = self.drop(col, axis=1)
        return self

    def add_complex_paqs(
        self,
        names=("ISOPleasant", "ISOEventful"),
        scale_to_one: bool = True,
        projection: bool = True,
        fill_na: bool = True,
        fill_val: int = 3,
    ):
        isopl, isoev = self.calculate_complex_paqs(
            scale_to_one, projection, fill_na, fill_val
        )
        self[names[0]] = isopl
        self[names[1]] = isoev
        return self

    def filter_record_ids(self, record_ids: list, **kwargs):
        return janitor.filter_column_isin(self, "RecordID", record_ids, **kwargs)

    def filter_group_ids(self, group_ids: list, **kwargs):
        return janitor.filter_column_isin(self, "GroupID", group_ids, **kwargs)

    def filter_session_ids(self, session_ids: list, **kwargs):
        return janitor.filter_column_isin(self, "SessionID", session_ids, **kwargs)

    def filter_location_ids(self, location_ids: list, **kwargs):
        return janitor.filter_column_isin(self, "LocationID", location_ids, **kwargs)

    def filter_lockdown(self, is_lockdown=False):
        complement = bool(is_lockdown)
        return janitor.filter_on(self, "Lockdown == 0", complement)

    def return_paqs(self, incl_ids: bool = True, other_cols: list = None):
        cols = PAQ_COLS
        if incl_ids:
            id_cols = [
                name
                for name in ["RecordID", "GroupID", "SessionID", "LocationID"]
                if name in self.columns
            ]

            cols = id_cols + cols
        if other_cols:
            cols = cols + other_cols
        return self[cols]

    # Plotting
    def circumplex_scatter(
        self,
        ax=None,
        title="Soundscape Scatter Plot",
        group=None,
        x="ISOPleasant",
        y="ISOEventful",
        prim_labels=True,
        diagonal_lines=False,
        palette=None,
        legend=False,
        legend_loc="lower left",
        s=100,
        figsize=default_figsize,
        **scatter_kwargs,
    ):
        return plotting.circumplex_scatter(
            self,
            ax=ax,
            title=title,
            group=group,
            x=x,
            y=y,
            prim_labels=prim_labels,
            diagonal_lines=diagonal_lines,
            palette=palette,
            legend=legend,
            legend_loc=legend_loc,
            s=s,
            figsize=figsize,
            **scatter_kwargs,
        )

    def circumplex_density(
        self,
        ax=None,
        title="Soundscape Density Plot",
        x="ISOPleasant",
        y="ISOEventful",
        prim_labels=True,
        diagonal_lines=False,
        palette="Blues",
        group=None,
        fill=True,
        bw_adjust=default_bw_adjust,
        alpha=0.95,
        legend=False,
        legend_loc="lower left",
        figsize=default_figsize,
        **density_kwargs,
    ):
        return plotting.circumplex_density(
            sf=self,
            ax=ax,
            title=title,
            x=x,
            y=y,
            prim_labels=prim_labels,
            diagonal_lines=diagonal_lines,
            palette=palette,
            group=group,
            fill=fill,
            bw_adjust=bw_adjust,
            alpha=alpha,
            legend=legend,
            legend_loc=legend_loc,
            figsize=figsize,
            **density_kwargs,
        )

    def circumplex_jointplot(
        self,
        title="Soundscape Joint Plot",
        x="ISOPleasant",
        y="ISOEventful",
        prim_labels=False,
        diagonal_lines=False,
        palette="Blues",
        fill=True,
        bw_adjust=default_bw_adjust,
        alpha=0.95,
        legend=False,
        legend_loc="lower left",
        s=100,
        marginal_kind="density",
        joint_kind="density",
        group=None,
        joint_kwargs={},
        marginal_kwargs={"fill": True},
    ):
        return plotting.circumplex_jointplot(
            self,
            title=title,
            x=x,
            y=y,
            prim_labels=prim_labels,
            diagonal_lines=diagonal_lines,
            palette=palette,
            fill=fill,
            bw_adjust=bw_adjust,
            alpha=alpha,
            legend=legend,
            legend_loc=legend_loc,
            s=s,
            marginal_kind=marginal_kind,
            joint_kind=joint_kind,
            group=group,
            joint_kwargs=joint_kwargs,
            marginal_kwargs=marginal_kwargs,
        )


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
