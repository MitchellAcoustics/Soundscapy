# Customized functions specifically for the International Soundscape Database

from datetime import date

# Add soundscapy to the Python path
from pathlib import Path

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

# Constants and Labels
from soundscapy.utils.surveys import *


@register_dataframe_accessor("isd")
class ISDAccessor:
    """Custom accessor for the International Soundscape Database (ISD) dataset

    Parameters
    ----------
    pandas_obj : pd.DataFrame
        Dataframe containing ISD formatted data

    Attributes
    ----------
    _obj : pd.DataFrame
        Dataframe containing ISD formatted data
    _analysis_data : str
        Date of analysis
    _metadata : dict
        Dictionary of metadata

    Methods
    -------
    validate_dataset
        Validate the dataset to ensure it is in the correct format.
    paq_data_quality
        Return the data quality of the PAQs
    filter_group_ids
        Filter the dataframe by GroupID
    filter_location_ids
        Filter the dataframe by LocationID
    filter_session_ids
        Filter the dataframe by SessionID
    filter_record_ids
        Filter the dataframe by RecordID
    filter_lockdown
        Filter the dataframe by Lockdown
    convert_group_ids_to_index
        Convert the GroupID column to the index
    return_paqs
        Return the PAQs as a dataframe
    soundscapy_describe
        Return a summary of the data
    mean_responses
        Calculate the mean responses for each PAQ
    convert_column_to_index
        Reassign an existing column as the dataframe index.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._analysis_data = date.today().isoformat()
        self._metadata = {}

    def validate_dataset(
        self,
        paq_aliases=PAQ_NAMES,
        allow_lockdown=False,
        allow_paq_na=False,
        verbose=1,
        val_range=(1, 5),
    ):
        """Validate the dataset to ensure it is in the correct format.

        Parameters
        ----------
        paq_aliases : list, optional
            List of aliases for the PAQ, by default PAQ_NAMES
        allow_lockdown : bool, optional
            Allow lockdown, by default False
        allow_paq_na : bool, optional
            Allow PAQ to be NA, by default False
        verbose : int, optional
            Print progress, by default 1
        val_range : tuple, optional
            Range of valid values, by default (1, 5)

        Returns
        -------
        pandas.DataFrame
            Validated dataframe
        """

        return validate_dataset(
            self._obj,
            paq_aliases=paq_aliases,
            allow_lockdown=allow_lockdown,
            allow_paq_na=allow_paq_na,
            verbose=verbose,
            val_range=val_range,
        )

    def paq_data_quality(self, verbose=0):
        """Return the data quality of the PAQs

        Parameters
        ----------
        verbose : int, optional
            Print progress, by default 0

        Returns
        -------
        pandas.DataFrame
        """
        return paq_data_quality(self._obj, verbose=verbose)

    def filter_group_ids(self, group_ids):
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
        return self._obj.sspy.filter("GroupID", group_ids)

    def filter_location_ids(self, location_ids):
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
        return self._obj.sspy.filter("LocationID", location_ids)

    def filter_session_ids(self, session_ids):
        """Filter the dataframe by SessionID

        Parameters
        ----------
        session_ids : list
            List of SessionIDs to filter by

        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        return self._obj.sspy.filter("SessionID", session_ids)

    def filter_record_ids(self, record_ids):
        """Filter the dataframe by RecordID

        Parameters
        ----------
        record_ids : list
            List of RecordIDs to filter by

        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        return self._obj.sspy.filter("RecordID", record_ids)

    def filter_lockdown(self, is_lockdown=False):
        """Filter the dataframe by Lockdown

        Parameters
        ----------
        is_lockdown : bool, optional
            Filter by lockdown, by default False

        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        return (
            self._obj.query("Lockdown == 1")
            if is_lockdown
            else self._obj.query("Lockdown == 0")
        )

    def convert_group_ids_to_index(self, drop=False):
        """Convert the GroupID column to the index

        Parameters
        ----------
        drop : bool, optional
            Drop the GroupID column, by default False

        Returns
        -------
        pd.DataFrame
            Dataframe with GroupID as index
        """
        return convert_column_to_index(self._obj, "Group_ID", drop=drop)

    def return_paqs(self, incl_ids=True, other_cols=None):
        """Return the PAQs as a dataframe

        Parameters
        ----------
        incl_ids : bool, optional
            Include the IDs, by default True
        other_cols : list, optional
            List of other columns to include, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing the PAQs
        """
        return return_paqs(self._obj, incl_ids=incl_ids, other_cols=other_cols)

    def location_describe(
        self, location, type="percent", pl_threshold=0, ev_threshold=0
    ):
        """Return a summary of the data

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
        pd.DataFrame
            Summary of the data
        """
        loc_df = self.filter_location_ids(location_ids=[location])
        count = len(loc_df)
        pl_count = len(loc_df[loc_df["ISOPleasant"] > pl_threshold])
        ev_count = len(loc_df[loc_df["ISOEventful"] > ev_threshold])
        vibrant_count = len(
            loc_df.query("ISOPleasant > @pl_threshold & ISOEventful > @ev_threshold")
        )
        chaotic_count = len(
            loc_df.query("ISOPleasant < @pl_threshold & ISOEventful > @ev_threshold")
        )
        mono_count = len(
            loc_df.query("ISOPleasant < @pl_threshold & ISOEventful < @ev_threshold")
        )
        calm_count = len(
            loc_df.query("ISOPleasant > @pl_threshold & ISOEventful < @ev_threshold")
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

    def soundscapy_describe(self, group_by="LocationID", type="percent"):
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
            location: self.location_describe(location, type=type)
            for location in self._obj[group_by].unique()
        }

        res = pd.DataFrame.from_dict(res, orient="index")
        return res

    def mean_responses(self, group="LocationID") -> pd.DataFrame:
        """Calculate the mean responses for each PAQ

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing ISD formatted data

        Returns
        -------
        pd.Dataframe
            Dataframe containing the mean responses for each PAQ
        """
        return mean_responses(self._obj, group=group)

    def convert_column_to_index(self, col="GroupID", drop=True):
        """Reassign an existing column as the dataframe index.

        Parameters
        ----------
        col : str, optional
            Column to convert to index, by default "GroupID"
        drop : bool, optional
            Drop the column, by default True

        Returns
        -------
        pd.DataFrame
            Dataframe with the column as the index
        """
        return convert_column_to_index(self._obj, col=col, drop=drop)

    def validate_isd(
        self,
        paq_aliases=None,
        allow_lockdown=False,
        allow_paq_na=False,
        verbose=1,
        val_range=(1, 5),
    ):
        """Validate the ISD dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing ISD formatted data
        paq_aliases : dict, optional
            Dictionary of PAQ aliases, by default None
        allow_lockdown : bool, optional
            Allow lockdown PAQs?, by default False
        allow_paq_na : bool, optional
            Allow missing PAQs?, by default False
        verbose : int, optional
            Verbosity level, by default 1
        val_range : tuple, optional
            Range of valid values, by default (1,5)

        Returns
        -------
        pd.DataFrame
            Dataframe containing the validation results
        """
        return validate_dataset(
            self._obj,
            paq_aliases=paq_aliases,
            allow_lockdown=allow_lockdown,
            allow_paq_na=allow_paq_na,
            verbose=verbose,
            val_range=val_range,
        )


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
    if version.lower() not in ["latest", "v0.2.3", "v0.2.2", "v0.2.1", "v0.2.0"]:
        raise ValueError(f"Version {version} not recognised.")

    version = "v0.2.3" if version == "latest" else version
    if version in ["V0.2.0", "v0.2.0", "V0.2.1", "v0.2.1"]:
        url = "https://zenodo.org/record/5578573/files/SSID%20Lockdown%20Database%20VL0.2.1.xlsx"

    elif version in ["V0.2.2", "v0.2.2"]:
        url = "https://zenodo.org/record/5705908/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"

    elif version in ["v0.2.3", "V0.2.3"]:
        url = "https://zenodo.org/record/5914762/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"

    return pd.read_excel(url, engine="openpyxl")


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
