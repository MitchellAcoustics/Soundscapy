import multiprocessing as mp
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from maad.features import all_spectral_alpha_indices, all_temporal_alpha_indices
from maad.sound import spectrogram
from mosqito.sq_metrics import (
    loudness_zwtv,
    roughness_dw,
    sharpness_din_from_loudness,
    sharpness_din_perseg,
    sharpness_din_tv,
)
from scipy import stats

DEFAULT_LABELS = {
    "LZeq": "LZeq",
    "LCeq": "LCeq",
    "LAeq": "LAeq",
    "Leq": "Leq",
    "SEL": "SEL",
    "loudness_zwtv": "N",
    "roughness_dw": "R",
    "sharpness_din_perseg": "S",
    "sharpness_din_from_loudness": "S",
    "sharpness_din_tv": "S",
}


# Metrics calculations
def _stat_calcs(
    label: str, ts_array: np.ndarray, res: dict, statistics: List[int | str]
) -> dict:
    """
    Calculate various statistics for a time series array and add them to a results dictionary.

    This function computes specified statistics (e.g., percentiles, mean, max, min, kurtosis, skewness)
    for a given time series and adds them to a results dictionary with appropriate labels.

    Args:
        label (str): Base label for the statistic in the results dictionary.
        ts_array (np.ndarray): 1D numpy array of the time series data.
        res (dict): Existing results dictionary to update with new statistics.
        statistics (List[Union[int, str]]): List of statistics to calculate. Can include percentiles
            (as integers) and string identifiers for other statistics (e.g., "avg", "max", "min", "kurt", "skew").

    Returns:
        dict: Updated results dictionary with newly calculated statistics.

    Example:
        >>> ts = np.array([1, 2, 3, 4, 5])
        >>> res = {}
        >>> updated_res = _stat_calcs("metric", ts, res, [50, "avg", "max"])
        >>> print(updated_res)
        {'metric_50': 3.0, 'metric_avg': 3.0, 'metric_max': 5}
    """

    for stat in statistics:
        if stat in ("avg", "mean"):
            res[f"{label}_{stat}"] = ts_array.mean()
        elif stat == "max":
            res[f"{label}_{stat}"] = ts_array.max()
        elif stat == "min":
            res[f"{label}_{stat}"] = ts_array.min()
        elif stat == "kurt":
            res[f"{label}_{stat}"] = stats.kurtosis(ts_array)
        elif stat == "skew":
            res[f"{label}_{stat}"] = stats.skew(ts_array)
        elif stat == "std":
            res[f"{label}_{stat}"] = np.std(ts_array)
        else:
            res[f"{label}_{stat}"] = np.percentile(ts_array, 100 - stat)
    return res


def mosqito_metric_1ch(
    s,
    metric: str,
    statistics: Tuple[Union[int, str, ...]] = (
        5,
        10,
        50,
        90,
        95,
        "avg",
        "max",
        "min",
        "kurt",
        "skew",
    ),
    label: Optional[str] = None,
    as_df: bool = False,
    return_time_series: bool = False,
    func_args: Dict = {},
) -> Dict | pd.DataFrame:
    """
    Calculate a MoSQITo psychoacoustic metric for a single channel signal.

    This function computes various psychoacoustic metrics using the MoSQITo library
    and calculates statistics on the results.

    Args:
        s (Signal): Single channel signal object to analyze.
        metric (str): Name of the metric to calculate. Options are "loudness_zwtv",
            "roughness_dw", "sharpness_din_from_loudness", "sharpness_din_perseg",
            or "sharpness_din_tv".
        statistics (Tuple[Union[int, str], ...]): Statistics to calculate on the metric results.
        label (Optional[str]): Label to use for the metric in the results. If None, uses a default label.
        as_df (bool): If True, return results as a pandas DataFrame. Otherwise, return a dictionary.
        return_time_series (bool): If True, include the full time series in the results.
        func_args (dict): Additional arguments to pass to the underlying MoSQITo function.

    Returns:
        Union[dict, pd.DataFrame]: Results of the metric calculation and statistics.

    Raises:
        ValueError: If the input signal is not single-channel or if an unrecognized metric is specified.

    Example:
        >>> # xdoctest: +SKIP
        >>> from soundscapy.audio import Binaural
        >>> signal = Binaural.from_wav("audio.wav")
        >>> results = mosqito_metric_1ch(signal[0], "loudness_zwtv", as_df=True)
    """
    # Checks and warnings
    if s.channels != 1:
        raise ValueError("Signal must be single channel")
    try:
        label = label or DEFAULT_LABELS[metric]
    except KeyError as e:
        raise ValueError(f"Metric {metric} not recognized.") from e
    if as_df and return_time_series:
        warnings.warn(
            "Cannot return both a dataframe and time series. Returning dataframe only."
        )
        return_time_series = False

    # Start the calc
    res = {}
    if metric == "loudness_zwtv":
        N, N_spec, bark_axis, time_axis = loudness_zwtv(s, s.fs, **func_args)
        res = _stat_calcs(label, N, res, statistics)
        if return_time_series:
            res[f"{label}_ts"] = (time_axis, N)
    elif metric == "roughness_dw":
        R, R_spec, bark_axis, time_axis = roughness_dw(s, s.fs, **func_args)
        res = _stat_calcs(label, R, res, statistics)
        if return_time_series:
            res[f"{label}_ts"] = (time_axis, R)
    elif metric == "sharpness_din_from_loudness":
        # The `sharpness_din_from_loudness` requires the loudness to be calculated first.
        field_type = func_args.get("field_type", "free")
        N, N_spec, bark_axis, time_axis = loudness_zwtv(s, s.fs, field_type=field_type)
        res = _stat_calcs("N", N, res, statistics)
        if return_time_series:
            res["N_ts"] = time_axis, N

        # Calculate the sharpness_din_from_loudness metric
        func_args.pop("field_type", None)
        S = sharpness_din_from_loudness(N, N_spec, **func_args)
        res = _stat_calcs(label, S, res, statistics)
        if return_time_series:
            res[f"{label}_ts"] = (time_axis, S)
    elif metric == "sharpness_din_perseg":
        S, time_axis = sharpness_din_perseg(s, s.fs, **func_args)
        res = _stat_calcs(label, S, res, statistics)
        if return_time_series:
            res[f"{label}_ts"] = (time_axis, S)
    elif metric == "sharpness_din_tv":
        S, time_axis = sharpness_din_tv(s, s.fs, **func_args)
        res = _stat_calcs(label, S, res, statistics)
        if return_time_series:
            res[f"{label}_ts"] = (time_axis, S)
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    # Return the results in the requested format
    if not as_df:
        return res
    try:
        rec = s.recording
        return pd.DataFrame(res, index=[rec])
    except AttributeError:
        return pd.DataFrame(res, index=[0])


def maad_metric_1ch(
    s, metric: str, as_df: bool = False, verbose: bool = False, func_args={}
):
    """Run a metric from the scikit-maad library (or suite of indices) on a single channel signal.

    Currently only supports running all of the alpha indices at once.

    Parameters
    ----------
    s : Signal or Binaural (single channel)
        Single channel signal to calculate the alpha indices for
    metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
        Metric to calculate
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    verbose : bool, optional
        Whether to print status updates, by default False
    **func_args : dict, optional
        Additional keyword arguments to pass to the metric function, by default {}

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame

    See Also
    --------
    maad.features.all_spectral_alpha_indices
    maad.features.all_temporal_alpha_indices
    """
    # Checks and status
    if s.channels != 1:
        raise ValueError("Signal must be single channel")
    if verbose:
        print(f" - Calculating scikit-maad {metric}")
    # Start the calc
    if metric == "all_spectral_alpha_indices":
        Sxx, tn, fn, ext = spectrogram(
            s, s.fs, **func_args
        )  # spectral requires the spectrogram
        res = all_spectral_alpha_indices(Sxx, tn, fn, extent=ext, **func_args)[0]

    elif metric == "all_temporal_alpha_indices":
        res = all_temporal_alpha_indices(s, s.fs, **func_args)
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    if not as_df:
        return res.to_dict("records")[0]
    try:
        res["Recording"] = s.recording
        res.set_index(["Recording"], inplace=True)
        return res
    except AttributeError:
        return res


def pyacoustics_metric_1ch(
    s,
    metric: str,
    statistics: List | Tuple = (
        5,
        10,
        50,
        90,
        95,
        "avg",
        "max",
        "min",
        "kurt",
        "skew",
    ),
    label: str = None,
    as_df: bool = False,
    return_time_series: bool = False,
    verbose: bool = False,
    func_args={},
):
    """Run a metric from the pyacoustics library on a single channel object.

    Parameters
    ----------
    s : Signal or Binaural (single channel slice)
        Single channel signal to calculate the metric for
    metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
        The metric to run
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc),
            by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False
        Cannot return time series if as_df is True
    verbose : bool, optional
        Whether to print status updates, by default False
    **func_args : dict, optional
        Additional keyword arguments to pass to the metric function, by default {}

    Returns
    -------
    dict
        dictionary of the calculated statistics.
        key is metric name + statistic (e.g. LZeq_5, LZeq_90, etc)
        value is the calculated statistic

    Raises
    ------
    ValueError
        Metric must be one of {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}

    See Also
    --------
    acoustics
    """
    if s.channels != 1:
        raise ValueError("Signal must be single channel")
    try:
        label = label or DEFAULT_LABELS[metric]
    except KeyError as e:
        raise ValueError(f"Metric {metric} not recognized.") from e
    if as_df and return_time_series:
        warnings.warn(
            "Cannot return both a dataframe and time series. Returning dataframe only."
        )

        return_time_series = False
    if verbose:
        print(f" - Calculating Python Acoustics: {metric} {statistics}")
    res = {}
    if metric in {"LZeq", "Leq", "LAeq", "LCeq"}:
        if metric in {"LZeq", "Leq"}:
            weighting = "Z"
        elif metric == "LAeq":
            weighting = "A"
        elif metric == "LCeq":
            weighting = "C"
        if "avg" in statistics or "mean" in statistics:
            stat = "avg" if "avg" in statistics else "mean"
            res[f"{label}"] = s.weigh(weighting).leq()
            statistics = list(statistics)
            statistics.remove(stat)
        if len(statistics) > 0:
            res = _stat_calcs(
                label, s.weigh(weighting).levels(**func_args)[1], res, statistics
            )

        if return_time_series:
            res[f"{label}_ts"] = s.weigh(weighting).levels(**func_args)
    elif metric == "SEL":
        res[f"{label}"] = s.sound_exposure_level()
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    if not as_df:
        return res
    try:
        rec = s.recording
        return pd.DataFrame(res, index=[rec])
    except AttributeError:
        return pd.DataFrame(res, index=[0])


# 2ch Metrics calculations
def pyacoustics_metric_2ch(
    b,
    metric: str,
    statistics: Tuple | List = (
        5,
        10,
        50,
        90,
        95,
        "avg",
        "max",
        "min",
        "kurt",
        "skew",
    ),
    label: str = None,
    channel_names: Tuple | List = ("Left", "Right"),
    as_df: bool = False,
    return_time_series: bool = False,
    verbose: bool = False,
    func_args={},
):
    """Run a metric from the python acoustics library on a Binaural object.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the metric for
    metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
        The metric to run
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc),
            by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    channel_names : tuple or list, optional
        Custom names for the channels, by default ("Left", "Right")
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False
        Cannot return time series if as_df is True
    verbose : bool, optional
        Whether to print status updates, by default False
    func_args : dict, optional
        Arguments to pass to the metric function, by default {}

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame

    See Also
    --------
    sq_metrics.pyacoustics_metric_1ch
    """
    if b.channels != 2:
        raise ValueError(
            "Must be 2 channel signal. Use `pyacoustics_metric_1ch instead`."
        )

    if verbose:
        print(f" - Calculating Python Acoustics metrics: {metric}")
    res_l = pyacoustics_metric_1ch(
        b[0],
        metric,
        statistics,
        label,
        as_df=False,
        return_time_series=return_time_series,
        func_args=func_args,
    )

    res_r = pyacoustics_metric_1ch(
        b[1],
        metric,
        statistics,
        label,
        as_df=False,
        return_time_series=return_time_series,
        func_args=func_args,
    )

    res = {channel_names[0]: res_l, channel_names[1]: res_r}
    if not as_df:
        return res
    try:
        rec = b.recording
    except AttributeError:
        rec = 0
    df = pd.DataFrame.from_dict(res, orient="index")
    df["Recording"] = rec
    df["Channel"] = df.index
    df.set_index(["Recording", "Channel"], inplace=True)
    return df


def _parallel_mosqito_metric_2ch(
    b,
    metric: str,
    statistics: Tuple | List = (
        5,
        10,
        50,
        90,
        95,
        "avg",
        "max",
        "min",
        "kurt",
        "skew",
    ),
    label: str = None,
    channel_names: Tuple | List = ("Left", "Right"),
    return_time_series: bool = False,
    func_args={},
):
    """Run a metric from the mosqito library on a Binaural object.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the metric for
    metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
        The metric to run
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc),
            by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    channel_names : tuple or list, optional
        Custom names for the channels, by default ("Left", "Right")
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False
        Cannot return time series if as_df is True
    func_args : dict, optional
        Arguments to pass to the metric function, by default {}

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame

    See Also
    --------
    sq_metrics.mosqito_metric_1ch
    """
    pool = mp.Pool(mp.cpu_count())
    result_objects = pool.starmap(
        mosqito_metric_1ch,
        [
            (
                b[i],
                metric,
                statistics,
                label,
                False,
                return_time_series,
                func_args,
            )
            for i in [0, 1]
        ],
    )

    pool.close()
    return {channel: result_objects[i] for i, channel in enumerate(channel_names)}


def mosqito_metric_2ch(
    b,
    metric: str,
    statistics: Tuple | List = (
        5,
        10,
        50,
        90,
        95,
        "avg",
        "max",
        "min",
        "kurt",
        "skew",
    ),
    label: str = None,
    channel_names: Tuple | List = ("Left", "Right"),
    as_df: bool = False,
    return_time_series: bool = False,
    parallel: bool = True,
    verbose: bool = False,
    func_args={},
):
    """function for calculating metrics from Mosqito.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the sound quality indices for
    metric : {"loudness_zwtv", "sharpness_din_from_loudness", "sharpness_din_perseg",
    "sharpness_din_tv", "roughness_dw"}
        Metric to calculate
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc.),
            by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    channel_names : tuple or list, optional
        Custom names for the channels, by default ("Left", "Right")
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False
        Only works for metrics that return a time series array.
        Cannot be returned in a dataframe. Will raise a warning if both `as_df`
        and `return_time_series` are True and will only return the DataFrame with the other stats.
    parallel : bool, optional
        Whether to run the channels in parallel, by default True
        If False, will run each channel sequentially.
        If being run as part of a larger parallel analysis (e.g. processing many recordings at once), this will
        automatically be set to False.
    verbose : bool, optional
        Whether to print status updates, by default False
    func_args : dict, optional
        Additional arguments to pass to the metric function, by default {}

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame
    """
    if b.channels != 2:
        raise ValueError("Must be 2 channel signal. Use `mosqito_metric_1ch` instead.")
    if verbose:
        if metric == "sharpness_din_from_loudness":
            print(
                " - Calculating MoSQITo metrics: `sharpness_din` from `loudness_zwtv`"
            )
        else:
            print(f" - Calculating MoSQITo metric: {metric}")

    # Make sure we're not already running in a parallel process
    # (e.g. if called from `parallel_process`)
    if mp.current_process().daemon:
        parallel = False
    if parallel:
        res = _parallel_mosqito_metric_2ch(
            b, metric, statistics, label, channel_names, return_time_series
        )

    else:
        res_l = mosqito_metric_1ch(
            b[0],
            metric,
            statistics,
            label,
            as_df=False,
            return_time_series=return_time_series,
            func_args=func_args,
        )

        res_r = mosqito_metric_1ch(
            b[1],
            metric,
            statistics,
            label,
            as_df=False,
            return_time_series=return_time_series,
            func_args=func_args,
        )

        res = {channel_names[0]: res_l, channel_names[1]: res_r}
    if not as_df:
        return res
    try:
        rec = b.recording
    except AttributeError:
        rec = 0
    df = pd.DataFrame.from_dict(res, orient="index")
    df["Recording"] = rec
    df["Channel"] = df.index
    df.set_index(["Recording", "Channel"], inplace=True)
    return df


def maad_metric_2ch(
    b,
    metric: str,
    channel_names: Tuple | List = ("Left", "Right"),
    as_df: bool = False,
    verbose: bool = False,
    func_args={},
):
    """Run a metric from the scikit-maad library (or suite of indices) on a binaural signal.

    Currently only supports running all the alpha indices at once.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the alpha indices for
    metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
        Metric to calculate
    channel_names : tuple or list, optional
        Custom names for the channels, by default ("Left", "Right").
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    verbose : bool, optional
        Whether to print status updates, by default False
    func_args : dict, optional
        Additional arguments to pass to the metric function, by default {}

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame

    See Also
    --------
    scikit-maad library
    `sq_metrics.maad_metric_1ch`

    """
    if b.channels != 2:
        raise ValueError("Must be 2 channel signal. Use `maad_metric_1ch` instead.")
    if verbose:
        print(f" - Calculating scikit-maad {metric}")
    res_l = maad_metric_1ch(b[0], metric, as_df=False)
    res_r = maad_metric_1ch(b[1], metric, as_df=False)
    res = {channel_names[0]: res_l, channel_names[1]: res_r}
    if not as_df:
        return res
    try:
        rec = b.recording
    except AttributeError:
        rec = 0
    df = pd.DataFrame.from_dict(res, orient="index")
    df["Recording"] = rec
    df["Channel"] = df.index
    df.set_index(["Recording", "Channel"], inplace=True)
    return df


# Analysis dataframe functions
def prep_multiindex_df(dictionary: dict, label: str = "Leq", incl_metric: bool = True):
    """df help to prepare a MultiIndex dataframe from a dictionary of results

    Parameters
    ----------
    dictionary : dict
        Dict of results with recording name as key, channels {"Left", "Right"} as second key, and Leq metric as value
    label : str, optional
        Name of metric included, by default "Leq"
    incl_metric : bool, optional
        Whether to include the metric value in the resulting dataframe, by default True
        If False, will only set up the DataFrame with the proper MultiIndex
    Returns
    -------
    pd.DataFrame
        Index includes "Recording" and "Channel" with a column for each index if `incl_metric`.

    """
    new_dict = {}
    for outerKey, innerDict in dictionary.items():
        for innerKey, values in innerDict.items():
            new_dict[(outerKey, innerKey)] = values
    idx = pd.MultiIndex.from_tuples(new_dict.keys())
    df = pd.DataFrame(new_dict.values(), index=idx, columns=[label])
    df.index.names = ["Recording", "Channel"]
    if not incl_metric:
        df = df.drop(columns=[label])
    return df


def add_results(results_df: pd.DataFrame, metric_results: pd.DataFrame):
    """Add results to MultiIndex dataframe

    Parameters
    ----------
    results_df : pd.DataFrame
        MultiIndex dataframe to add results to
    metric_results : pd.DataFrame
        MultiIndex dataframe of results to add

    Returns
    -------
    pd.DataFrame
        Index includes "Recording" and "Channel" with a column for each index.
    """
    # TODO: Add check for whether all of the recordings have rows in the dataframe
    # If not, add new rows first

    if not set(metric_results.columns).issubset(set(results_df.columns)):
        # Check if results_df already has the columns in results
        results_df = results_df.join(metric_results)
    else:
        results_df.update(metric_results, errors="ignore")
    return results_df


def process_all_metrics(
    b,
    analysis_settings: "AnalysisSettings",
    parallel: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Process all metrics specified in the analysis settings for a binaural signal.

    This function runs through all enabled metrics in the provided analysis settings,
    computes them for the given binaural signal, and compiles the results into a single DataFrame.

    Args:
        b (Binaural): Binaural signal object to process.
        analysis_settings (AnalysisSettings): Configuration object specifying which metrics
            to run and their parameters.
        parallel (bool): If True, run applicable calculations in parallel. Defaults to True.
        verbose (bool): If True, print progress information. Defaults to False.

    Returns:
        pd.DataFrame: A MultiIndex DataFrame containing results from all processed metrics.
                      The index includes "Recording" and "Channel" levels.

    Note:
        The parallel option primarily affects the MoSQITo metrics. Other metrics may not
        benefit from parallelization.

    Example:
        >>> # xdoctest: +SKIP
        >>> from soundscapy.audio import Binaural
        >>> from soundscapy import AnalysisSettings
        >>> signal = Binaural.from_wav("audio.wav")
        >>> settings = AnalysisSettings.from_yaml("settings.yaml")
        >>> results = process_all_metrics(signal, settings, verbose=True)
    """
    if verbose:
        print(f"Processing {b.recording}")

    idx = pd.MultiIndex.from_tuples(((b.recording, "Left"), (b.recording, "Right")))
    results_df = pd.DataFrame(index=idx)
    results_df.index.names = ["Recording", "Channel"]

    for library in ["PythonAcoustics", "MoSQITo", "scikit-maad"]:
        if library in analysis_settings.settings:
            for metric, metric_settings in analysis_settings.settings[library].items():
                if metric_settings.run:
                    if library == "PythonAcoustics":
                        results_df = pd.concat(
                            (
                                results_df,
                                b.pyacoustics_metric(
                                    metric,
                                    verbose=verbose,
                                    analysis_settings=analysis_settings,
                                ),
                            ),
                            axis=1,
                        )
                    elif library == "MoSQITo":
                        results_df = pd.concat(
                            (
                                results_df,
                                b.mosqito_metric(
                                    metric,
                                    parallel=parallel,
                                    verbose=verbose,
                                    analysis_settings=analysis_settings,
                                ),
                            ),
                            axis=1,
                        )
                    elif library == "scikit-maad":
                        results_df = pd.concat(
                            (
                                results_df,
                                b.maad_metric(
                                    metric,
                                    verbose=verbose,
                                    analysis_settings=analysis_settings,
                                ),
                            ),
                            axis=1,
                        )

    return results_df
