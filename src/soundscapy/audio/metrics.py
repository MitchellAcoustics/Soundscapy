"""
soundscapy.audio.metrics
========================

This module provides functions for calculating various acoustic and psychoacoustic metrics
for audio signals. It includes implementations for single-channel and two-channel signals,
as well as wrapper functions for different libraries such as python-acoustics, MoSQITo,
and scikit-maad.

Functions
---------
_stat_calcs : Calculate various statistics for a time series array.
mosqito_metric_1ch : Calculate a MoSQITo psychoacoustic metric for a single channel signal.
maad_metric_1ch : Run a metric from the scikit-maad library on a single channel signal.
pyacoustics_metric_1ch : Run a metric from the pyacoustics library on a single channel object.
pyacoustics_metric_2ch : Run a metric from the python acoustics library on a Binaural object.
mosqito_metric_2ch : Calculate metrics from MoSQITo for a two-channel signal.
maad_metric_2ch : Run a metric from the scikit-maad library on a binaural signal.
prep_multiindex_df : Prepare a MultiIndex dataframe from a dictionary of results.
add_results : Add results to a MultiIndex dataframe.
process_all_metrics : Process all metrics specified in the analysis settings for a binaural signal.

Notes
-----
This module relies on external libraries such as numpy, pandas, maad, mosqito, and scipy.
Ensure these dependencies are installed before using this module.
"""

import concurrent.futures
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

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

from .analysis_settings import AnalysisSettings

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


def _stat_calcs(
    label: str, ts_array: np.ndarray, res: dict, statistics: List[Union[int, str]]
) -> dict:
    """
    Calculate various statistics for a time series array and add them to a results dictionary.

    Parameters
    ----------
    label : str
        Base label for the statistic in the results dictionary.
    ts_array : np.ndarray
        1D numpy array of the time series data.
    res : dict
        Existing results dictionary to update with new statistics.
    statistics : List[Union[int, str]]
        List of statistics to calculate. Can include percentiles
        (as integers) and string identifiers for other statistics
        (e.g., "avg", "max", "min", "kurt", "skew").

    Returns
    -------
    dict
        Updated results dictionary with newly calculated statistics.

    Examples
    --------
    >>> # xdoctest: +REQUIRES(env:AUDIO_DEPS='1')
    >>> ts = np.array([1, 2, 3, 4, 5])
    >>> res = {}
    >>> updated_res = _stat_calcs("metric", ts, res, [50, "avg", "max"])
    >>> print(updated_res)
    {'metric_50': 3.0, 'metric_avg': 3.0, 'metric_max': 5}
    """
    logger.debug(f"Calculating statistics for {label}")
    for stat in statistics:
        try:
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
        except Exception as e:
            logger.error(f"Error calculating {stat} for {label}: {str(e)}")
            res[f"{label}_{stat}"] = np.nan
    return res


def mosqito_metric_1ch(
    s,
    metric: str,
    statistics: Tuple[Union[int, str]] = (
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
) -> Union[Dict, pd.DataFrame]:
    """
    Calculate a MoSQITo psychoacoustic metric for a single channel signal.

    Parameters
    ----------
    s : Signal
        Single channel signal object to analyze.
    metric : str
        Name of the metric to calculate. Options are "loudness_zwtv",
        "roughness_dw", "sharpness_din_from_loudness", "sharpness_din_perseg",
        or "sharpness_din_tv".
    statistics : Tuple[Union[int, str], ...], optional
        Statistics to calculate on the metric results.
    label : str, optional
        Label to use for the metric in the results. If None, uses a default label.
    as_df : bool, optional
        If True, return results as a pandas DataFrame. Otherwise, return a dictionary.
    return_time_series : bool, optional
        If True, include the full time series in the results.
    func_args : dict, optional
        Additional arguments to pass to the underlying MoSQITo function.

    Returns
    -------
    Union[dict, pd.DataFrame]
        Results of the metric calculation and statistics.

    Raises
    ------
    ValueError
        If the input signal is not single-channel or if an unrecognized metric is specified.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> from soundscapy.audio import Binaural
    >>> signal = Binaural.from_wav("audio.wav", resample=480000)
    >>> results = mosqito_metric_1ch(signal[0], "loudness_zwtv", as_df=True)
    """
    logger.debug(f"Calculating MoSQITo metric: {metric}")

    # Checks and warnings
    if s.channels != 1:
        logger.error("Signal must be single channel")
        raise ValueError("Signal must be single channel")
    try:
        label = label or DEFAULT_LABELS[metric]
    except KeyError as e:
        logger.error(f"Metric {metric} not recognized")
        raise ValueError(f"Metric {metric} not recognized.") from e
    if as_df and return_time_series:
        logger.warning(
            "Cannot return both a dataframe and time series. Returning dataframe only."
        )
        return_time_series = False

    # Start the calc
    res = {}
    try:
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
            field_type = func_args.get("field_type", "free")
            N, N_spec, bark_axis, time_axis = loudness_zwtv(
                s, s.fs, field_type=field_type
            )
            res = _stat_calcs("N", N, res, statistics)
            if return_time_series:
                res["N_ts"] = time_axis, N

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
            logger.error(f"Metric {metric} not recognized")
            raise ValueError(f"Metric {metric} not recognized.")
    except Exception as e:
        logger.error(f"Error calculating {metric}: {str(e)}")
        raise

    # Return the results in the requested format
    if not as_df:
        return res
    try:
        rec = s.recording
        return pd.DataFrame(res, index=[rec])
    except AttributeError:
        return pd.DataFrame(res, index=[0])


def maad_metric_1ch(s, metric: str, as_df: bool = False, func_args={}):
    """
    Run a metric from the scikit-maad library (or suite of indices) on a single channel signal.

    Currently only supports running all of the alpha indices at once.

    Parameters
    ----------
    s : Signal or Binaural (single channel)
        Single channel signal to calculate the alpha indices for.
    metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
        Metric to calculate.
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    func_args : dict, optional
        Additional keyword arguments to pass to the metric function, by default {}.

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the signal is not single-channel or if an unrecognized metric is specified.

    See Also
    --------
    maad.features.all_spectral_alpha_indices
    maad.features.all_temporal_alpha_indices
    """
    logger.debug(f"Calculating MAAD metric: {metric}")

    # Checks and status
    if s.channels != 1:
        logger.error("Signal must be single channel")
        raise ValueError("Signal must be single channel")

    logger.debug(f"Calculating scikit-maad {metric}")

    # Start the calc
    try:
        if metric == "all_spectral_alpha_indices":
            Sxx, tn, fn, ext = spectrogram(s, s.fs, **func_args)
            res = all_spectral_alpha_indices(Sxx, tn, fn, extent=ext, **func_args)[0]
        elif metric == "all_temporal_alpha_indices":
            res = all_temporal_alpha_indices(s, s.fs, **func_args)
        else:
            logger.error(f"Metric {metric} not recognized")
            raise ValueError(f"Metric {metric} not recognized.")
    except Exception as e:
        logger.error(f"Error calculating {metric}: {str(e)}")
        raise

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
    statistics: List[Union[int, str]] = (
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
    func_args={},
):
    """
    Run a metric from the pyacoustics library on a single channel object.

    Parameters
    ----------
    s : Signal or Binaural (single channel slice)
        Single channel signal to calculate the metric for.
    metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
        The metric to run.
    statistics : List[Union[int, str]], optional
        List of level statistics to calculate (e.g. L_5, L_90, etc).
    label : str, optional
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False.
        Cannot return time series if as_df is True.
    func_args : dict, optional
        Additional keyword arguments to pass to the metric function, by default {}.

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of the calculated statistics or a pandas DataFrame.

    Raises
    ------
    ValueError
        If the signal is not single-channel or if an unrecognized metric is specified.

    See Also
    --------
    acoustics
    """
    logger.debug(f"Calculating pyacoustics metric: {metric}")

    if s.channels != 1:
        logger.error("Signal must be single channel")
        raise ValueError("Signal must be single channel")
    try:
        label = label or DEFAULT_LABELS[metric]
    except KeyError as e:
        logger.error(f"Metric {metric} not recognized")
        raise ValueError(f"Metric {metric} not recognized.") from e
    if as_df and return_time_series:
        logger.warning(
            "Cannot return both a dataframe and time series. Returning dataframe only."
        )

        return_time_series = False

    logger.debug(f"Calculating Python Acoustics: {metric} {statistics}")

    res = {}
    try:
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
            logger.error(f"Metric {metric} not recognized")
            raise ValueError(f"Metric {metric} not recognized.")
    except Exception as e:
        logger.error(f"Error calculating {metric}: {str(e)}")
        raise

    if not as_df:
        return res
    try:
        rec = s.recording
        return pd.DataFrame(res, index=[rec])
    except AttributeError:
        return pd.DataFrame(res, index=[0])


def pyacoustics_metric_2ch(
    b,
    metric: str,
    statistics: Union[Tuple, List] = (
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
    channel_names: Tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,
    return_time_series: bool = False,
    func_args={},
):
    """
    Run a metric from the python acoustics library on a Binaural object.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the metric for.
    metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
        The metric to run.
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc).
    label : str, optional
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    channel_names : tuple of str, optional
        Custom names for the channels, by default ("Left", "Right").
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False.
        Cannot return time series if as_df is True.
    func_args : dict, optional
        Arguments to pass to the metric function, by default {}.

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the input signal is not 2-channel.

    See Also
    --------
    pyacoustics_metric_1ch
    """
    logger.debug(f"Calculating pyacoustics metric for 2 channels: {metric}")

    if b.channels != 2:
        logger.error("Must be 2 channel signal. Use `pyacoustics_metric_1ch` instead.")
        raise ValueError(
            "Must be 2 channel signal. Use `pyacoustics_metric_1ch instead`."
        )

    logger.debug(f"Calculating Python Acoustics metrics: {metric}")

    try:
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
    except Exception as e:
        logger.error(f"Error calculating {metric} for 2 channels: {str(e)}")
        raise

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
    statistics: Union[Tuple, List] = (
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
    channel_names: Tuple[str, str] = ("Left", "Right"),
    return_time_series: bool = False,
    func_args={},
):
    """
    Run a metric from the mosqito library on a Binaural object in parallel.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the metric for.
    metric : str
        The metric to run.
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc).
    label : str, optional
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    channel_names : tuple of str, optional
        Custom names for the channels, by default ("Left", "Right").
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False.
    func_args : dict, optional
        Arguments to pass to the metric function, by default {}.

    Returns
    -------
    dict
        Dictionary of results for both channels.

    See Also
    --------
    mosqito_metric_1ch
    """
    logger.debug(f"Calculating MoSQITo metric in parallel: {metric}")

    pool = mp.Pool(mp.cpu_count())
    try:
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
    except Exception as e:
        logger.error(f"Error in parallel MoSQITo calculation: {str(e)}")
        pool.close()
        raise
    finally:
        pool.join()


def mosqito_metric_2ch(
    b,
    metric: str,
    statistics: Union[Tuple, List] = (
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
    channel_names: Tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,
    return_time_series: bool = False,
    parallel: bool = True,
    func_args={},
):
    """
    Calculate metrics from MoSQITo for a two-channel signal with optional parallel processing.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the sound quality indices for.
    metric : {"loudness_zwtv", "sharpness_din_from_loudness", "sharpness_din_perseg",
    "sharpness_din_tv", "roughness_dw"}
        Metric to calculate.
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc.).
    label : str, optional
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    channel_names : tuple of str, optional
        Custom names for the channels, by default ("Left", "Right").
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    return_time_series : bool, optional
        Whether to return the time series of the metric, by default False.
        Only works for metrics that return a time series array.
        Cannot be returned in a dataframe.
    parallel : bool, optional
        Whether to process channels in parallel, by default True.
    func_args : dict, optional
        Additional arguments to pass to the metric function, by default {}.

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the input signal is not 2-channel.
    """
    logger.debug(f"Calculating MoSQITo metric for 2 channels: {metric}")

    if b.channels != 2:
        logger.error("Must be 2 channel signal. Use `mosqito_metric_1ch` instead.")
        raise ValueError("Must be 2 channel signal. Use `mosqito_metric_1ch` instead.")

    if metric == "sharpness_din_from_loudness":
        logger.debug(
            "Calculating MoSQITo metrics: `sharpness_din` from `loudness_zwtv`"
        )
    else:
        logger.debug(f"Calculating MoSQITo metric: {metric}")

    try:
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_l = executor.submit(
                    mosqito_metric_1ch,
                    b[0],
                    metric,
                    statistics,
                    label,
                    False,
                    return_time_series,
                    func_args,
                )
                future_r = executor.submit(
                    mosqito_metric_1ch,
                    b[1],
                    metric,
                    statistics,
                    label,
                    False,
                    return_time_series,
                    func_args,
                )
                res_l = future_l.result()
                res_r = future_r.result()
        else:
            res_l = mosqito_metric_1ch(
                b[0],
                metric,
                statistics,
                label,
                False,
                return_time_series,
                func_args,
            )
            res_r = mosqito_metric_1ch(
                b[1],
                metric,
                statistics,
                label,
                False,
                return_time_series,
                func_args,
            )

        res = {channel_names[0]: res_l, channel_names[1]: res_r}
    except Exception as e:
        logger.error(
            f"Error calculating MoSQITo metric {metric} for 2 channels: {str(e)}"
        )
        raise

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
    channel_names: Tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,
    func_args={},
):
    """
    Run a metric from the scikit-maad library (or suite of indices) on a binaural signal.

    Currently only supports running all the alpha indices at once.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the alpha indices for.
    metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
        Metric to calculate.
    channel_names : tuple of str, optional
        Custom names for the channels, by default ("Left", "Right").
    as_df : bool, optional
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
    func_args : dict, optional
        Additional arguments to pass to the metric function, by default {}.

    Returns
    -------
    dict or pd.DataFrame
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the input signal is not 2-channel or if an unrecognized metric is specified.

    See Also
    --------
    scikit-maad library
    maad_metric_1ch
    """
    logger.debug(f"Calculating MAAD metric for 2 channels: {metric}")

    if b.channels != 2:
        logger.error("Must be 2 channel signal. Use `maad_metric_1ch` instead.")
        raise ValueError("Must be 2 channel signal. Use `maad_metric_1ch` instead.")

    logger.debug(f"Calculating scikit-maad {metric}")

    try:
        res_l = maad_metric_1ch(b[0], metric, as_df=False)
        res_r = maad_metric_1ch(b[1], metric, as_df=False)
        res = {channel_names[0]: res_l, channel_names[1]: res_r}
    except Exception as e:
        logger.error(f"Error calculating MAAD metric {metric} for 2 channels: {str(e)}")
        raise

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
    """
    Prepare a MultiIndex dataframe from a dictionary of results.

    Parameters
    ----------
    dictionary : dict
        Dict of results with recording name as key, channels {"Left", "Right"} as second key,
        and Leq metric as value.
    label : str, optional
        Name of metric included, by default "Leq".
    incl_metric : bool, optional
        Whether to include the metric value in the resulting dataframe, by default True.
        If False, will only set up the DataFrame with the proper MultiIndex.

    Returns
    -------
    pd.DataFrame
        Index includes "Recording" and "Channel" with a column for each index if `incl_metric`.

    Raises
    ------
    ValueError
        If the input dictionary is not in the expected format.
    """
    logger.info("Preparing MultiIndex DataFrame")
    try:
        new_dict = {}
        for outerKey, innerDict in dictionary.items():
            for innerKey, values in innerDict.items():
                new_dict[(outerKey, innerKey)] = values
        idx = pd.MultiIndex.from_tuples(new_dict.keys())
        df = pd.DataFrame(new_dict.values(), index=idx, columns=[label])
        df.index.names = ["Recording", "Channel"]
        if not incl_metric:
            df = df.drop(columns=[label])
        logger.debug("MultiIndex DataFrame prepared successfully")
        return df
    except Exception as e:
        logger.error(f"Error preparing MultiIndex DataFrame: {str(e)}")
        raise ValueError("Invalid input dictionary format") from e


def add_results(results_df: pd.DataFrame, metric_results: pd.DataFrame):
    """
    Add results to MultiIndex dataframe.

    Parameters
    ----------
    results_df : pd.DataFrame
        MultiIndex dataframe to add results to.
    metric_results : pd.DataFrame
        MultiIndex dataframe of results to add.

    Returns
    -------
    pd.DataFrame
        Index includes "Recording" and "Channel" with a column for each index.

    Raises
    ------
    ValueError
        If the input DataFrames are not in the expected format.
    """
    logger.info("Adding results to MultiIndex DataFrame")
    try:
        # TODO: Add check for whether all of the recordings have rows in the dataframe
        # If not, add new rows first

        if not set(metric_results.columns).issubset(set(results_df.columns)):
            # Check if results_df already has the columns in results
            results_df = results_df.join(metric_results)
        else:
            results_df.update(metric_results, errors="ignore")
        logger.debug("Results added successfully")
        return results_df
    except Exception as e:
        logger.error(f"Error adding results to DataFrame: {str(e)}")
        raise ValueError("Invalid input DataFrame format") from e


def process_all_metrics(
    b, analysis_settings: AnalysisSettings, parallel: bool = True
) -> pd.DataFrame:
    """
    Process all metrics specified in the analysis settings for a binaural signal.

    This function runs through all enabled metrics in the provided analysis settings,
    computes them for the given binaural signal, and compiles the results into a single DataFrame.

    Parameters
    ----------
    b : Binaural
        Binaural signal object to process.
    analysis_settings : AnalysisSettings
        Configuration object specifying which metrics to run and their parameters.
    parallel : bool, optional
        If True, run applicable calculations in parallel. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A MultiIndex DataFrame containing results from all processed metrics.
        The index includes "Recording" and "Channel" levels.

    Raises
    ------
    ValueError
        If there's an error processing any of the metrics.

    Notes
    -----
    The parallel option primarily affects the MoSQITo metrics. Other metrics may not
    benefit from parallelization.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> from soundscapy.audio import Binaural
    >>> from soundscapy import AnalysisSettings
    >>> signal = Binaural.from_wav("audio.wav", resample=480000)
    >>> settings = AnalysisSettings.from_yaml("settings.yaml")
    >>> results = process_all_metrics(signal,settings)
    """
    logger.info(f"Processing all metrics for {b.recording}")
    logger.debug(f"Parallel processing: {parallel}")

    idx = pd.MultiIndex.from_tuples(((b.recording, "Left"), (b.recording, "Right")))
    results_df = pd.DataFrame(index=idx)
    results_df.index.names = ["Recording", "Channel"]

    try:
        for (
            library,
            metrics_settings,
        ) in analysis_settings.get_enabled_metrics().items():
            for metric in metrics_settings.keys():
                logger.debug(f"Processing {library} metric: {metric}")
                if library == "PythonAcoustics":
                    results_df = pd.concat(
                        (
                            results_df,
                            b.pyacoustics_metric(
                                metric, metric_settings=metrics_settings[metric]
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
                                metric_settings=metrics_settings[metric],
                            ),
                        ),
                        axis=1,
                    )
                elif library == "scikit-maad" or library == "scikit_maad":
                    results_df = pd.concat(
                        (
                            results_df,
                            b.maad_metric(
                                metric, metric_settings=metrics_settings[metric]
                            ),
                        ),
                        axis=1,
                    )
        logger.info("All metrics processed successfully")
        return results_df
    except Exception as e:
        logger.error(f"Error processing metrics: {str(e)}")
        raise ValueError("Error processing metrics") from e


# Add any additional helper functions or constants here if needed

if __name__ == "__main__":
    # Add any script-level code or examples here
    pass
