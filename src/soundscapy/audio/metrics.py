"""
Functions for calculating various acoustic and psychoacoustic metrics for audio signals.

It includes implementations for single-channel and two-channel signals,
as well as wrapper functions for different libraries such as Acoustic Toolbox, MoSQITo,
and scikit-maad.

"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import warnings
from typing import TYPE_CHECKING, Any, Literal, TypedDict

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import pandas as pd
from loguru import logger
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

if TYPE_CHECKING:
    from acoustic_toolbox import Signal
    from numpy.typing import NDArray

    from soundscapy import Binaural
    from soundscapy.audio.analysis_settings import AnalysisSettings

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
    label: str,
    ts_array: NDArray[np.float64],
    res: dict,
    statistics: tuple[int | str, ...],
) -> dict:
    """
    Calculate various statistics for a time series array and add them to a dictionary.

    Parameters
    ----------
    label
        Base label for the statistic in the results dictionary.
    ts_array
        1D numpy array of the time series data.
    res
        Existing results dictionary to update with new statistics.
    statistics
        List of statistics to calculate. Can include percentiles
        (as integers) and string identifiers for other statistics
        (e.g., "avg", "max", "min", "kurt", "skew").

    Returns
    -------
    :
        Updated results dictionary with newly calculated statistics.

    Examples
    --------
    >>> # doctest: +REQUIRES(env:AUDIO_DEPS='1')
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
            elif isinstance(stat, int):
                res[f"{label}_{stat}"] = np.percentile(ts_array, 100 - stat)
            else:
                logger.error(f"Unrecognized statistic: {stat} for {label}")
                res[f"{label}_{stat}"] = np.nan
        except Exception as e:  # noqa: BLE001, PERF203
            logger.error(f"Error calculating {stat} for {label}: {e!s}")
            res[f"{label}_{stat}"] = np.nan
    return res


class _MosqitoMetricParams(TypedDict, total=False):
    field_type: str  # loudness_zwtv, sharpness_din_from_loudness, sharpness_din_tv
    weighting: str  # sharpness_din_from_loudness,sharpness_din_perseg,sharpness_din_tv
    overlap: float  # roughness_dw
    nperseg: int  # sharpness_din_perseg
    noverlap: int | None  # sharpness_din_perseg
    skip: float  # sharpness_din_tv


def mosqito_metric_1ch(
    s: Signal,
    metric: Literal[
        "loudness_zwtv",
        "roughness_dw",
        "sharpness_din_from_loudness",
        "sharpness_din_perseg",
        "sharpness_din_tv",
    ],
    statistics: tuple[int | str, ...] = (
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
    label: str | None = None,
    *,
    as_df: bool = False,
    return_time_series: bool = False,
    **kwargs: Unpack[_MosqitoMetricParams],
) -> dict | pd.DataFrame:
    """
    Calculate a MoSQITo psychoacoustic metric for a single channel signal.

    Parameters
    ----------
    s
        Single channel signal object to analyze.
    metric
        Name of the metric to calculate. Options are "loudness_zwtv",
        "roughness_dw", "sharpness_din_from_loudness", "sharpness_din_perseg",
        or "sharpness_din_tv".
    statistics
        Statistics to calculate on the metric results.
    label
        Label to use for the metric in the results. If None, uses a default label.
    as_df
        If True, return results as a pandas DataFrame. Otherwise, return a dictionary.
    return_time_series
        If True, include the full time series in the results.
    **kwargs
        Additional arguments to pass to the underlying MoSQITo function.

    Returns
    -------
    :
        Results of the metric calculation and statistics.

    Raises
    ------
    ValueError
        If the input signal is not single-channel
        or if an unrecognized metric is specified.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> from soundscapy.audio import Binaural
    >>> signal = Binaural.from_wav("audio.wav", resample=480000)
    >>> results = mosqito_metric_1ch(signal[0], "loudness_zwtv", as_df=True)

    """
    logger.debug(f"Calculating MoSQITo metric: {metric}")

    # Checks and warnings
    if s.channels != 1:
        logger.error("Signal must be single channel")
        msg = "Signal must be single channel"
        raise ValueError(msg)
    try:
        label = label or DEFAULT_LABELS[metric]
    except KeyError as e:
        logger.error(f"Metric {metric} not recognized")
        msg = f"Metric {metric} not recognized."
        raise ValueError(msg) from e
    if as_df and return_time_series:
        logger.warning(
            "Cannot return both a dataframe and time series. Returning dataframe only."
        )
        return_time_series = False

    # Start the calc
    res = {}
    try:
        if metric == "loudness_zwtv":
            # Prepare args specifically for loudness_zwtv
            loudness_args = {}
            if "field_type" in kwargs:
                loudness_args["field_type"] = kwargs["field_type"]
            # Call with filtered args
            N, N_spec, _, time_axis = loudness_zwtv(s, s.fs, **loudness_args)  # noqa: N806
            # TODO(MitchellAcoustics): Add the bark_axis back in
            # when we implement time series calcs
            # https://github.com/MitchellAcoustics/Soundscapy/issues/113
            res = _stat_calcs(label, N, res, statistics)
            if return_time_series:
                res[f"{label}_ts"] = (time_axis, N)

        elif metric == "roughness_dw":
            # Prepare args specifically for roughness_dw
            roughness_args = {}
            if "overlap" in kwargs:
                roughness_args["overlap"] = kwargs["overlap"]
            # Call with filtered args
            R, _, _, time_axis = roughness_dw(s, s.fs, **roughness_args)  # noqa: N806
            # TODO(MitchellAcoustics): Add the R_spec and bark_axis back in
            # when we implement time series calcs
            # https://github.com/MitchellAcoustics/Soundscapy/issues/113
            if isinstance(R, float | int):
                res[label] = R
            elif isinstance(R, np.ndarray) and len(R) == 1:
                res[label] = R[0]
            else:
                res = _stat_calcs(label, R, res, statistics)
            if return_time_series:
                res[f"{label}_ts"] = (time_axis, R)

        elif metric == "sharpness_din_from_loudness":
            # Prepare args for loudness_zwtv (needed first)
            loudness_args = {}
            if "field_type" in kwargs:
                loudness_args["field_type"] = kwargs["field_type"]
            N, N_spec, _, time_axis = loudness_zwtv(s, s.fs, **loudness_args)  # noqa: N806
            # TODO(MitchellAcoustics): Add the R_spec and bark_axis back in
            # when we implement time series calcs
            # https://github.com/MitchellAcoustics/Soundscapy/issues/113
            res = _stat_calcs("N", N, res, statistics)
            if return_time_series:
                res["N_ts"] = time_axis, N

            # Prepare args specifically for sharpness_din_from_loudness
            sharpness_args = {}
            if "weighting" in kwargs:
                sharpness_args["weighting"] = kwargs["weighting"]
            # Call with filtered args
            S = sharpness_din_from_loudness(N, N_spec, **sharpness_args)  # noqa: N806
            res = _stat_calcs(label, S, res, statistics)
            if return_time_series:
                res[f"{label}_ts"] = (time_axis, S)

        elif metric == "sharpness_din_perseg":
            # Prepare args specifically for sharpness_din_perseg
            sharpness_args = {}
            if "weighting" in kwargs:
                sharpness_args["weighting"] = kwargs["weighting"]
            if "nperseg" in kwargs:
                sharpness_args["nperseg"] = kwargs["nperseg"]
            if "noverlap" in kwargs:
                sharpness_args["noverlap"] = kwargs["noverlap"]
            # Call with filtered args
            S, time_axis = sharpness_din_perseg(s, s.fs, **sharpness_args)  # noqa: N806
            res = _stat_calcs(label, S, res, statistics)
            if return_time_series:
                res[f"{label}_ts"] = (time_axis, S)

        elif metric == "sharpness_din_tv":
            # Prepare args specifically for sharpness_din_tv
            sharpness_args = {}
            if "weighting" in kwargs:
                sharpness_args["weighting"] = kwargs["weighting"]
            if "skip" in kwargs:
                sharpness_args["skip"] = kwargs["skip"]
            # Call with filtered args
            S, time_axis = sharpness_din_tv(s, s.fs, **sharpness_args)  # noqa: N806
            res = _stat_calcs(label, S, res, statistics)
            if return_time_series:
                res[f"{label}_ts"] = (time_axis, S)
        else:
            msg = f"Metric {metric} not recognized."
            logger.error(msg)
            raise ValueError(msg)  # noqa: TRY301
    except Exception as e:
        logger.error(f"Error calculating {metric}: {e!s}")
        raise

    # Return the results in the requested format
    if not as_df:
        return res

    rec = getattr(s, "recording", None)
    return pd.DataFrame(res, index=[rec])


# noinspection PyPep8Naming
def maad_metric_1ch(
    s: Signal | Binaural,
    metric: Literal["all_temporal_alpha_indices", "all_spectral_alpha_indices"],
    as_df: bool = False,
    func_args: dict | None = None,
) -> Any:
    """
    Run a metric from the scikit-maad library (or suite of indices) on a single channel.

    Currently only supports running all the alpha indices at once.

    Parameters
    ----------
    s
        Single channel signal to calculate the alpha indices for.
    metric
        Metric to calculate.
    as_df
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe
        with ("Recording", "Channel") as the index.
    func_args
        Additional keyword arguments to pass to the metric function, by default {}.

    Returns
    -------
    :
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

    if func_args is None:
        func_args = {}

    # Checks and status
    if s.channels != 1:
        msg = "Signal must be single channel."
        logger.error(msg)
        raise ValueError(msg)

    logger.debug(f"Calculating scikit-maad {metric}")

    # Start the calc
    try:
        if metric == "all_spectral_alpha_indices":
            Sxx, tn, fn, ext = spectrogram(s, s.fs, **func_args)  # noqa: N806
            res = all_spectral_alpha_indices(Sxx, tn, fn, extent=ext, **func_args)[0]
        elif metric == "all_temporal_alpha_indices":
            res = all_temporal_alpha_indices(s, s.fs, **func_args)
        else:
            msg = f"Metric {metric} not recognized."
            logger.error(msg)
            raise ValueError(msg)  # noqa: TRY301
    except Exception as e:
        logger.error(f"Error calculating {metric}: {e!s}")
        raise

    if not as_df:
        return res.to_dict("records")[0]
    try:
        res["Recording"] = s.recording
        return res.set_index(["Recording"])
    except AttributeError:
        return res


def pyacoustics_metric_1ch(  # noqa: ANN201, D103
    s: Signal | Binaural,
    metric: str,
    statistics: list[int | str] = (
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
    label: str | None = None,
    as_df: bool = False,  # noqa: FBT001, FBT002
    return_time_series: bool = False,  # noqa: FBT001, FBT002
    func_args={},  # noqa: ANN001, B006
):
    """

    !!! warning "Deprecated"
        pyacoustics is deprecated. Use `acoustics_metric_1ch` instead.
    """
    warnings.warn(
        "pyacoustics is deprecated. Use acoustics_metric_1ch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return acoustics_metric_1ch(
        s,
        metric,
        statistics,
        label,
        as_df,
        return_time_series,
        func_args,
    )


def acoustics_metric_1ch(
    s: Signal | Binaural,
    metric: str,
    statistics: list[int | str] = (
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
    label: str | None = None,
    as_df: bool = False,
    return_time_series: bool = False,
    func_args: dict | None = None,
) -> dict | pd.DataFrame:
    """
    Run a metric from the acoustic_toolbox library on a single channel object.

    Parameters
    ----------
    s
        Single channel signal to calculate the metric for.
    metric
        The metric to run.
    statistics
        List of level statistics to calculate (e.g. L_5, L_90, etc).
    label
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    as_df
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe
        with ("Recording", "Channel") as the index.
    return_time_series
        Whether to return the time series of the metric, by default False.
        Cannot return time series if as_df is True.
    func_args
        Additional keyword arguments to pass to the metric function, by default {}.

    Returns
    -------
    :
        Dictionary of the calculated statistics or a pandas DataFrame.

    Raises
    ------
    ValueError
        If the signal is not single-channel or if an unrecognized metric is specified.

    See Also
    --------
    `acoustic_toolbox`

    """
    logger.debug(f"Calculating acoustics metric: {metric}")

    if func_args is None:
        func_args = {}

    if s.channels != 1:
        msg = "Signal must be single channel"
        logger.error(msg)
        raise ValueError(msg)
    try:
        label = label or DEFAULT_LABELS[metric]
    except KeyError as e:
        msg = f"Metric {metric} not recognized."
        logger.error(msg)
        raise ValueError(msg) from e
    if as_df and return_time_series:
        logger.warning(
            "Cannot return both a dataframe and time series. Returning dataframe only."
        )

        return_time_series = False

    logger.debug(f"Calculating Acoustic Toolbox: {metric} {statistics}")

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
            msg = f"Metric {metric} not recognized."
            logger.error(msg)
            raise ValueError(msg)  # noqa: TRY301
    except Exception as e:
        logger.error(f"Error calculating {metric}: {e!s}")
        raise

    if not as_df:
        return res
    try:
        rec = s.recording
        return pd.DataFrame(res, index=[rec])
    except AttributeError:
        return pd.DataFrame(res, index=[0])


def pyacoustics_metric_2ch(  # noqa: ANN201, D103
    b: Binaural,
    metric: str,
    statistics: tuple | list = (
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
    label: str | None = None,
    channel_names: tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,  # noqa: FBT001, FBT002
    return_time_series: bool = False,  # noqa: FBT001, FBT002
    func_args={},  # noqa: ANN001, B006
):
    """

    !!! warning "Deprecated"
        pyacoustics is deprecated. Use `acoustics_metric_2ch` instead.
    """
    warnings.warn(
        "pyacoustics is deprecated. Use acoustics_metric_2ch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return acoustics_metric_2ch(
        b,
        metric,
        statistics,
        label,
        channel_names,
        as_df,
        return_time_series,
        func_args,
    )


def acoustics_metric_2ch(
    b: Binaural,
    metric: str,
    statistics: tuple | list = (
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
    label: str | None = None,
    channel_names: tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,
    return_time_series: bool = False,
    func_args: dict | None = None,
) -> dict | pd.DataFrame:
    """
    Run a metric from the Acoustic Toolbox library on a Binaural object.

    Parameters
    ----------
    b
        Binaural signal to calculate the metric for.
    metric
        The metric to run.
    statistics
        List of level statistics to calculate (e.g. L_5, L_90, etc).
    label
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    channel_names
        Custom names for the channels, by default ("Left", "Right").
    as_df
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe
        with ("Recording", "Channel") as the index.
    return_time_series
        Whether to return the time series of the metric, by default False.
        Cannot return time series if as_df is True.
    func_args
        Arguments to pass to the metric function, by default {}.

    Returns
    -------
    :
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the input signal is not 2-channel.

    See Also
    --------
    `acoustics_metric_1ch`

    """
    logger.debug(f"Calculating acoustics metric for 2 channels: {metric}")

    if func_args is None:
        func_args = {}

    if b.channels != 2:  # noqa: PLR2004
        msg = "Must be 2 channel signal. Use `acoustics_metric_1ch instead`."
        logger.error(msg)
        raise ValueError(msg)

    logger.debug(f"Calculating Acoustic Toolbox metrics: {metric}")

    try:
        res_l = acoustics_metric_1ch(
            b[0],
            metric,
            statistics,
            label,
            as_df=False,
            return_time_series=return_time_series,
            func_args=func_args,
        )

        res_r = acoustics_metric_1ch(
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
        logger.error(f"Error calculating {metric} for 2 channels: {e!s}")
        raise

    if not as_df:
        return res
    try:
        rec = b.recording
    except AttributeError:
        rec = 0
    else:
        data = pd.DataFrame.from_dict(res, orient="index")
        data["Recording"] = rec
        data["Channel"] = data.index
        return data.set_index(["Recording", "Channel"])
    data = pd.DataFrame.from_dict(res, orient="index")
    data["Recording"] = rec
    data["Channel"] = data.index
    return data.set_index(["Recording", "Channel"])


def _parallel_mosqito_metric_2ch(
    b: Binaural,
    metric: str,
    statistics: tuple | list = (
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
    label: str | None = None,
    channel_names: tuple[str, str] = ("Left", "Right"),
    return_time_series: bool = False,
    func_args: dict | None = None,
) -> dict | None:
    """
    Run a metric from the mosqito library on a Binaural object in parallel.

    Parameters
    ----------
    b
        Binaural signal to calculate the metric for.
    metric
        The metric to run.
    statistics
        List of level statistics to calculate (e.g. L_5, L_90, etc).
    label
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    channel_names
        Custom names for the channels, by default ("Left", "Right").
    return_time_series
        Whether to return the time series of the metric, by default False.
    func_args
        Arguments to pass to the metric function, by default {}.

    Returns
    -------
    :
        Dictionary of results for both channels.

    See Also
    --------
    `mosqito_metric_1ch`

    """
    logger.debug(f"Calculating MoSQITo metric in parallel: {metric}")

    if func_args is None:
        func_args = {}

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
        logger.error(f"Error in parallel MoSQITo calculation: {e!s}")
        pool.close()
        raise
    finally:
        pool.join()


def mosqito_metric_2ch(
    b: Binaural,
    metric: Literal[
        "loudness_zwtv",
        "sharpness_din_from_loudness",
        "sharpness_din_perseg",
        "sharpness_din_tv",
        "roughness_dw",
    ],
    statistics: tuple | list = (
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
    label: str | None = None,
    channel_names: tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,
    return_time_series: bool = False,
    parallel: bool = True,
    func_args: dict | None = None,
) -> dict | pd.DataFrame:
    """
    Calculate metrics from MoSQITo for a two-channel signal with parallel processing.

    Parameters
    ----------
    b
        Binaural signal to calculate the sound quality indices for.
    metric
        Metric to calculate.
    statistics
        List of level statistics to calculate (e.g. L_5, L_90, etc.).
    label
        Label to use for the metric in the results dictionary.
        If None, will pull from default label for that metric given in DEFAULT_LABELS.
    channel_names
        Custom names for the channels, by default ("Left", "Right").
    as_df
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe
        with ("Recording", "Channel") as the index.
    return_time_series
        Whether to return the time series of the metric, by default False.
        Only works for metrics that return a time series array.
        Cannot be returned in a dataframe.
    parallel
        Whether to process channels in parallel, by default True.
    func_args
        Additional arguments to pass to the metric function, by default {}.

    Returns
    -------
    :
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the input signal is not 2-channel.

    """
    logger.debug(f"Calculating MoSQITo metric for 2 channels: {metric}")

    if func_args is None:
        func_args = {}

    if b.channels != 2:  # noqa: PLR2004
        msg = "Must be 2 channel signal. Use `mosqito_metric_1ch` instead."
        logger.error(msg)
        raise ValueError(msg)

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
                    as_df=False,
                    return_time_series=return_time_series,
                    **func_args,
                )
                future_r = executor.submit(
                    mosqito_metric_1ch,
                    b[1],
                    metric,
                    statistics,
                    label,
                    as_df=False,
                    return_time_series=return_time_series,
                    **func_args,
                )
                res_l = future_l.result()
                res_r = future_r.result()
        else:
            res_l = mosqito_metric_1ch(
                b[0],
                metric,
                statistics,
                label,
                as_df=False,
                return_time_series=return_time_series,
                **func_args,
            )
            res_r = mosqito_metric_1ch(
                b[1],
                metric,
                statistics,
                label,
                as_df=False,
                return_time_series=return_time_series,
                **func_args,
            )

        res = {channel_names[0]: res_l, channel_names[1]: res_r}
    except Exception as e:
        logger.error(f"Error calculating MoSQITo metric {metric} for 2 channels: {e!s}")
        raise

    if not as_df:
        return res
    try:
        rec = b.recording
    except AttributeError:
        rec = 0
    else:
        data = pd.DataFrame.from_dict(res, orient="index")
        data["Recording"] = rec
        data["Channel"] = data.index
        return data.set_index(["Recording", "Channel"])
    data = pd.DataFrame.from_dict(res, orient="index")
    data["Recording"] = rec
    data["Channel"] = data.index
    return data.set_index(["Recording", "Channel"])


def maad_metric_2ch(
    b: Binaural,
    metric: Literal["all_temporal_alpha_indices", "all_spectral_alpha_indices"],
    channel_names: tuple[str, str] = ("Left", "Right"),
    as_df: bool = False,
    func_args: dict | None = None,
) -> dict | pd.DataFrame:
    """
    Run a metric from scikit-maad library (or suite of indices) on a binaural signal.

    Currently only supports running all the alpha indices at once.

    Parameters
    ----------
    b
        Binaural signal to calculate the alpha indices for.
    metric
        Metric to calculate.
    channel_names
        Custom names for the channels, by default ("Left", "Right").
    as_df
        Whether to return a pandas DataFrame, by default False.
        If True, returns a MultiIndex Dataframe with ("Recording", "Channel")
        as the index.
    func_args
        Additional arguments to pass to the metric function, by default {}.

    Returns
    -------
    :
        Dictionary of results if as_df is False, otherwise a pandas DataFrame.

    Raises
    ------
    ValueError
        If the input signal is not 2-channel or if an unrecognized metric is specified.

    See Also
    --------
    `scikit-maad` library

    `maad_metric_1ch`

    """
    logger.debug(f"Calculating MAAD metric for 2 channels: {metric}")

    if func_args is None:
        func_args = {}

    if b.channels != 2:  # noqa: PLR2004
        logger.error("Must be 2 channel signal. Use `maad_metric_1ch` instead.")
        msg = "Must be 2 channel signal. Use `maad_metric_1ch` instead."
        raise ValueError(msg)

    logger.debug(f"Calculating scikit-maad {metric}")

    try:
        res_l = maad_metric_1ch(b[0], metric, as_df=False, **func_args)
        res_r = maad_metric_1ch(b[1], metric, as_df=False, **func_args)
        res = {channel_names[0]: res_l, channel_names[1]: res_r}
    except Exception as e:
        logger.error(f"Error calculating MAAD metric {metric} for 2 channels: {e!s}")
        raise

    if not as_df:
        return res
    try:
        rec = b.recording
    except AttributeError:
        rec = 0
    data = pd.DataFrame.from_dict(res, orient="index")
    data["Recording"] = rec
    data["Channel"] = data.index
    return data.set_index(["Recording", "Channel"])


# Analysis dataframe functions
def prep_multiindex_df(
    dictionary: dict, label: str = "Leq", incl_metric: bool = True
) -> pd.DataFrame:
    """
    Prepare a MultiIndex dataframe from a dictionary of results.

    Parameters
    ----------
    dictionary
        Dict of results with recording name as key,
        channels {"Left", "Right"} as second key, and Leq metric as value.
    label
        Name of metric included, by default "Leq".
    incl_metric
        Whether to include the metric value in the resulting dataframe, by default True.
        If False, will only set up the DataFrame with the proper MultiIndex.

    Returns
    -------
    :
        Index includes "Recording" and "Channel" with a column for each index
        if `incl_metric`.

    Raises
    ------
    ValueError
        If the input dictionary is not in the expected format.

    """
    logger.info("Preparing MultiIndex DataFrame")
    try:
        new_dict = {}
        for outer_key, inner_dict in dictionary.items():
            for inner_key, values in inner_dict.items():
                new_dict[(outer_key, inner_key)] = values
        idx = pd.MultiIndex.from_tuples(new_dict.keys())
        data = pd.DataFrame(new_dict.values(), index=idx, columns=[label])
        data.index.names = ["Recording", "Channel"]
        if not incl_metric:
            data = data.drop(columns=[label])
    except Exception as e:
        logger.error(f"Error preparing MultiIndex DataFrame: {e!s}")
        msg = "Invalid input dictionary format"
        raise ValueError(msg) from e
    logger.debug("MultiIndex DataFrame prepared successfully")
    return data


def add_results(results_df: pd.DataFrame, metric_results: pd.DataFrame) -> pd.DataFrame:
    """
    Add results to MultiIndex dataframe.

    Parameters
    ----------
    results_df
        MultiIndex dataframe to add results to.
    metric_results
        MultiIndex dataframe of results to add.

    Returns
    -------
    :
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
    except Exception as e:
        logger.error(f"Error adding results to DataFrame: {e!s}")
        msg = "Invalid input DataFrame format"
        raise ValueError(msg) from e
    logger.debug("Results added successfully")
    return results_df


def process_all_metrics(
    b: Binaural, analysis_settings: AnalysisSettings, parallel: bool = True
) -> pd.DataFrame:
    """
    Process all metrics specified in the analysis settings for a binaural signal.

    This function runs through all enabled metrics in the provided analysis settings,
    computes them for the given binaural signal, and compiles the results into a
    single DataFrame.

    Parameters
    ----------
    b
        Binaural signal object to process.
    analysis_settings
        Configuration object specifying which metrics to run and their parameters.
    parallel
        If True, run applicable calculations in parallel. Defaults to True.

    Returns
    -------
    :
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
    >>> # doctest: +SKIP
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
            for metric in metrics_settings:
                logger.debug(f"Processing {library} metric: {metric}")
                if library == "AcousticToolbox":
                    results_df = pd.concat(
                        (
                            results_df,
                            b.acoustics_metric(
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
                elif library in {"scikit-maad", "scikit_maad"}:
                    results_df = pd.concat(
                        (
                            results_df,
                            b.maad_metric(
                                metric, metric_settings=metrics_settings[metric]
                            ),
                        ),
                        axis=1,
                    )
    except Exception as e:
        logger.error(f"Error processing metrics: {e!s}")
        msg = "Error processing metrics"
        raise ValueError(msg) from e

    logger.info("All metrics processed successfully")
    return results_df
