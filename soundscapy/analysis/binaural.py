import multiprocessing as mp

import pandas as pd

from soundscapy.analysis.metrics import *

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

# 2ch Metrics calculations


def pyacoustics_metric_2ch(
    b,
    metric: str,
    statistics: Union[tuple, list] = (
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
    channel_names: Union[tuple, list] = ("Left", "Right"),
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
    statistics: Union[tuple, list] = (
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
    channel_names: Union[tuple, list] = ("Left", "Right"),
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
    statistics: Union[tuple, list] = (
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
    channel_names: Union[tuple, list] = ("Left", "Right"),
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
    channel_names: Union[tuple, list] = ("Left", "Right"),
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
    analysis_settings,
    parallel: bool = True,
    verbose: bool = False,
):
    """Loop through all metrics included in `analysis_settings` and add results to `results_df`

    Parameters
    ----------
    b : Binaural
        Binaural signal to process
    analysis_settings : AnalysisSettings
        Settings for analysis, including `run` tag for whether to run a metric
    parallel : bool, optional
        Whether to run the channels in parallel for `binaural.mosqito_metric_2ch`, by default True
        If False, will run each channel sequentially.
        If being run as part of a larger parallel analysis (e.g. processing many recordings at once), this will
        automatically be set to False.
        Applies only to `binaural.mosqito_metric_2ch`. The other metrics are considered fast enough not to bother.
    verbose : bool, optional
        Whether to print status updates, by default False

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with results from all metrics for one Binaural recording
    """
    if verbose:
        print(f"Processing {b.recording}")

    idx = pd.MultiIndex.from_tuples(((b.recording, "Left"), (b.recording, "Right")))
    results_df = pd.DataFrame(index=idx)
    results_df.index.names = ["Recording", "Channel"]

    # Count number of metrics to run
    metric_count = 0
    for library in analysis_settings.keys():
        if library not in ["PythonAcoustics", "scikit-maad", "MoSQITo"]:
            pass
        else:
            for metric in analysis_settings[library].keys():
                if analysis_settings[library][metric]["run"]:
                    metric_count += 1

    # Loop through options in analysis_settings
    for library in analysis_settings.keys():
        # Python Acoustics metrics
        if library == "PythonAcoustics":
            for metric in analysis_settings[library].keys():
                results_df = pd.concat(
                    (
                        results_df,
                        b.pyacoustics_metric(
                            metric, verbose=verbose, analysis_settings=analysis_settings
                        ),
                    ),
                    axis=1,
                )
        # MoSQITO metrics
        elif library == "MoSQITo":
            for metric in analysis_settings[library].keys():
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
        # scikit-maad metrics
        elif library == "scikit-maad":
            for metric in analysis_settings[library].keys():
                results_df = pd.concat(
                    (
                        results_df,
                        b.maad_metric(
                            metric, verbose=verbose, analysis_settings=analysis_settings
                        ),
                    ),
                    axis=1,
                )

    return results_df


# %%
