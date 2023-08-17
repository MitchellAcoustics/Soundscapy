# Customized functions specifically for the International Soundscape Database

from datetime import date
from importlib import resources

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

# Constants and Labels
from soundscapy.utils.surveys import *
from soundscapy.utils.surveys import rename_paqs, likert_data_quality

# Add soundscapy to the Python path

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


def load():
    """Load example "ISD" [1]_ csv file to DataFrame

    Returns
    -------
    pd.DataFrame
        dataframe of ISD data

    References
    ----------
    .. [1] Mitchell, Andrew, Oberman, Tin, Aletta, Francesco, Erfanian, Mercede, Kachlicka, Magdalena, Lionello, Matteo, & Kang, Jian. (2022). The International Soundscape Database: An integrated multimedia database of urban soundscape surveys -- questionnaires with acoustical and contextual information (0.2.4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6331810
    """

    with resources.path("soundscapy.data", "ISD-v0.2.2.csv") as f:
        data = pd.read_csv(f)
    return data


def load_zenodo(version="latest"):
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


def validate(
    df: pd.DataFrame,
    paq_aliases: Union[List, Dict] = _PAQ_ALIASES,
    allow_paq_na: bool = False,
    verbose: int = 1,
    val_range: Tuple = (1, 5),
):
    """Performs data quality checks and validates that the dataset fits the expected format

    Parameters
    ----------
    df : pd.DataFrame
        ISD style dataframe, incl PAQ data
    paq_aliases : list or dict, optional
        list of PAQ names (in order)
        or dict of PAQ names with new names as values, by default None
    allow_lockdown : bool, optional
        if True will keep Lockdown data in the df, by default True
    allow_paq_na : bool, optional
        remove rows which have any missing PAQ values
        otherwise will remove those with 50% missing, by default False    verbose : int, optional
        how much info to print while running, by default 1
    val_range : tuple, optional
        min and max range of the PAQ response values, by default (5, 1)

    Returns
    -------
    tuple
        cleaned dataframe, dataframe of excluded samples
    """
    if verbose > 0:
        print("Renaming PAQ columns.")
    df = rename_paqs(df, paq_aliases)

    if verbose > 0:
        print("Checking PAQ data quality.")
    l = likert_data_quality(df, verbose, allow_paq_na, val_range)
    if l is None:
        excl_df = None
    else:
        excl_df = df.iloc[l, :]
        df = df.drop(df.index[l])
    return df, excl_df


def _isd_select(data, select_by, condition):
    """
    General function to select by ID variables.

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe
    select_by : str
        Column name of the ID variable to select by
    condition : str, list, or tuple
        IDs to select from the dataframe

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    if isinstance(condition, (str, int)):
        return data.query(f"{select_by} == @condition", engine="python")
    elif isinstance(condition, (list, tuple)):
        return data.query(f"{select_by} in @condition")
    else:
        raise TypeError("Should be either a str, int, list, or tuple.")


def select_record_ids(data, record_ids):
    """Filter the dataframe by RecordID

    Parameters
    ----------
    data : pd.DataFrame
        ISD dataframe
    record_ids : list
        List of RecordIDs to filter by

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return _isd_select(data, "RecordID", record_ids)


def select_group_ids(data, group_ids):
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
    return _isd_select(data, "GroupID", group_ids)


def select_session_ids(data, session_ids):
    """
    Filter the dataframe by SessionID

    Parameters
    ----------
    session_ids : list
        List of SessionIDs to select

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return _isd_select(data, "SessionID", session_ids)


def select_location_ids(data, location_ids):
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
    return _isd_select(data, "LocationID", location_ids)


def remove_lockdown(data):
    """
    Remove the Lockdown data from ISD data.

    Parameters
    ----------
    data : pd.DataFrame
        ISD DataFrame

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with Lockdown data removed.
    """
    return data.query("Lockdown == 0")


def describe_location(data, location, type="percent", pl_threshold=0, ev_threshold=0):
    """
    Return a summary of the data

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
    dict
        Summary of the data
    """
    loc_df = select_location_ids(data, location_ids=location)
    count = len(loc_df)
    pl_count = len(loc_df[loc_df["ISOPleasant"] > pl_threshold])
    ev_count = len(loc_df[loc_df["ISOEventful"] > ev_threshold])
    vibrant_count = len(
        loc_df.query(
            "ISOPleasant > @pl_threshold & ISOEventful > @ev_threshold", engine="python"
        )
    )
    chaotic_count = len(
        loc_df.query(
            "ISOPleasant < @pl_threshold & ISOEventful > @ev_threshold", engine="python"
        )
    )
    mono_count = len(
        loc_df.query(
            "ISOPleasant < @pl_threshold & ISOEventful < @ev_threshold", engine="python"
        )
    )
    calm_count = len(
        loc_df.query(
            "ISOPleasant > @pl_threshold & ISOEventful < @ev_threshold", engine="python"
        )
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


def soundscapy_describe(
    df: pd.DataFrame, group_by="LocationID", type="percent"
) -> pd.DataFrame:
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
        location: describe_location(df, location) for location in df[group_by].unique()
    }

    return pd.DataFrame.from_dict(res, orient="index")


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
    likert_data_quality
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

        # raise PendingDeprecationWarning("The ISD Accessor is being deprecated.")

    def validate_dataset(
        self,
        paq_aliases=PAQ_NAMES,
        allow_lockdown=False,
        allow_paq_na=False,
        verbose=1,
        val_range=(1, 5),
    ):
        """DEPRECATED - Use `soundscapy.isd.validate(data)` instead.

        Validate the dataset to ensure it is in the correct format.

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.validate(data)` instead."
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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.utils.likert_data_quality()` instead."
        )
        # return likert_data_quality(self._obj, verbose=verbose)

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.select_group_ids()` instead."
        )
        # return self._obj.sspy.filter("GroupID", group_ids)

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.select_location_ids()` instead."
        )
        # return self._obj.sspy.filter("LocationID", location_ids)

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.select_session_ids()` instead."
        )
        # return self._obj.sspy.filter("SessionID", session_ids)

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
        raise DeprecationWarning("The ISD accessor has been deprecated.")

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
        raise DeprecationWarning("The ISD accessor has been deprecated.")

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
        raise DeprecationWarning("The ISD accessor has been deprecated.")

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
        raise DeprecationWarning("The ISD accessor has been deprecated.")

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.describe_location()` instead."
        )

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.soundscapy_describe()` instead."
        )

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
        raise DeprecationWarning("The ISD accessor has been deprecated.")

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
        raise DeprecationWarning(
            "The ISD accessor has been deprecated. Please use `soundscapy.isd.validate(data)` instead."
        )
