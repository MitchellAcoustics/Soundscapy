# %%
import warnings
from typing import Union

from maad.sound import spectrogram
from maad.features import all_spectral_alpha_indices, all_temporal_alpha_indices
import numpy as np
import pandas as pd
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


# %%
# Metrics calculations
def _stat_calcs(
    label: str, ts_array: np.ndarray, res: dict, statistics: Union[list, tuple]
):
    """Process a time series array of metrics into a dictionary of statistics.

    Parameters
    ----------
    label : str
        Label to use for the metric in the results dictionary
    ts_array : np.ndarray
        Time series array of the metric. Must be a 1D array.
        Specific time step is not necessary
    res : dict
        results dictionary to add results to
    statistics : Union
        list of statistics to calculate

    Returns
    -------
    dict
        results dictionary with the calculated statistics added
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


# %%
def mosqito_metric_1ch(
    s,
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
    label=None,
    as_df: bool = False,
    return_time_series: bool = False,
    func_args={},
):
    """Calculating a metric and accompanying statistics from Mosqito.

    Parameters
    ----------
    s : Signal or Binaural (single channel)
        Single channel signal to calculate the sound quality indices for
    metric : {"loudness_zwtv", "sharpness_din_from_loudness", "sharpness_din_perseg", "sharpness_din_tv",
    "roughness_zwtv"}
        Metric to calculate
    statistics : tuple or list
        List of level statistics to calculate (e.g. L_5, L_90, etc.),
            by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    as_df : bool, optional
        Return the results as a dataframe, by default False
    return_time_series : bool, optional
        Return the time series array of the metric, by default False
        Only works for metrics that return a time series array.
        Cannot be returned in a dataframe. Will raise a warning if both `as_df`
        and `return_time_series` are True and will only return the DataFrame with the other stats.
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
        Signal must be single channel. Can be a slice of a multichannel signal.
    ValueError
        Metric is not recognized. Must be one of {"loudness_zwtv", "sharpness_din_from_loudness",
        "sharpness_din_perseg", "roughness_dw"}
    Warning

    See Also
    --------
    mosqito.sq_metrics.loudness_zwtv : MoSQito Loudness calculation
    mosqito.sq_metrics.roughness_dw : MoSQito Roughness calculation
    mosqito.sq_metrics.sharpness_din_from_loudness : MoSQito Sharpness calculation
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


# %%
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


# %%
def pyacoustics_metric_1ch(
    s,
    metric: str,
    statistics: Union[list, tuple] = (
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
