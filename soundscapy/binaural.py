# %%
from typing import Union
import pandas as pd

from sq_metrics import *
import multiprocessing as mp

# %%

# Only used once to create the calibration json file

# levels = pd.read_excel(base_path.joinpath("EuropeLZeq.xlsx"), header=1, index_col=0)
# levels = levels.drop(columns=["Unnamed: 1", "Unnamed: 3", "Unnamed: 5"])
# levels.index = levels.index.str.split(" ").str[0]
# levels = levels.to_dict(orient='index')
# with open(base_path.joinpath("Levels.json")) as f:
#     levels = json.load(f)


# %%

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
    **func_args,
):
    """Run a metric from the python acoustics library on a Binaural object.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the metric for
    metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
        The metric to run
    recording : str, optional
        Name of the recording to process, by default "Rec
    channel : Union, optional
        Which channels to process, by default None
        if None will process both channels
    statistics : tuple or list, optional
        List of level statistics to calculate (e.g. L_5, L_90, etc),
            by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    verbose : bool, optional
        Whether to print status updates, by default False

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with the results of the metric calculation
        Index includes "Recording" and "Channel"

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
        **func_args,
    )
    res_r = pyacoustics_metric_1ch(
        b[1],
        metric,
        statistics,
        label,
        as_df=False,
        return_time_series=return_time_series,
        **func_args,
    )

    res = {
        channel_names[0]: res_l,
        channel_names[1]: res_r,
    }
    if as_df:
        try:
            rec = b.recording
        except:
            rec = 0
        df = pd.DataFrame.from_dict(res, orient="index")
        df["Recording"] = rec
        df["Channel"] = df.index
        df.set_index(["Recording", "Channel"], inplace=True)
        return df
    else:
        return res


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
    **func_args,
):
    pool = mp.Pool(mp.cpu_count())
    results = {}

    result_objects = pool.starmap(
        mosqito_metric_1ch,
        [(b[i], metric, statistics, label, return_time_series) for i in [0, 1]],
    )
    pool.close()

    for i, channel in enumerate(channel_names):
        results[channel] = result_objects[i]

    return results


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
    **func_args,
):
    """function for calculating metrics from Mosqito.

    Parameters
    ----------
    b : Binaural
        Binaural signal to calculate the alpha indices for
    metric : {"loudness_zwtv", "sharpness_din_from_loudness", "sharpness_din_perseg",
    "sharpness_din_tv", "roughness_dw"}
        Metric to calculate
    recording : str, optional
        Name of the recording, by default None
    channel : tuple or list of str, or str, optional
        Which channels to process, by default None
    statistics : tuple or list
        Statistics to calculate
    label : str, optional
        Label to use for the metric in the results dictionary, by default None
        If None, will pull from default label for that metric given in DEFAULT_LABELS
    verbose : bool, optional
        Whether to print status updates, by default False

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame of results.
        Index includes "Recording" and "Channel" with a column for each statistic
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
    if mp.current_process().daemon:  # True if already a subprocess
        parallel = False

    if parallel:
        res = _parallel_mosqito_metric_2ch(
            b,
            metric,
            statistics,
            label,
            channel_names,
            return_time_series,
            **func_args,
        )

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

        res = {
            channel_names[0]: res_l,
            channel_names[1]: res_r,
        }
    if as_df:
        try:
            rec = b.recording
        except:
            rec = 0
        df = pd.DataFrame.from_dict(res, orient="index")
        df["Recording"] = rec
        df["Channel"] = df.index
        df.set_index(["Recording", "Channel"], inplace=True)
        return df
    else:
        return res


def maad_metric_2ch(
    b,
    metric: str,
    channel_names: Union[tuple, list] = ("Left", "Right"),
    as_df: bool = False,
    verbose: bool = False,
):
    """Specific function for calculating all alpha temporal indices according to scikit-maad's function.

    Parameters
    ----------
    s : Binaural
        Binary signal to calculate the alpha indices for
    recording : str, optional
        Recording name, by default None
    channel : tuple or list of str, or str, optional
        Which channels to process, by default None
    verbose : , optional
        Whether to print status updates, by default False

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame of results.
        Index includes "Recording" and "Channel" with a column for each statistic
    """
    if b.channels != 2:
        raise ValueError("Must be 2 channel signal. Use `maad_metric_1ch` instead.")

    if verbose:
        print(f" - Calculating scikit-maad {metric}")

    res_l = maad_metric_1ch(b[0], metric, as_df=False)
    res_r = maad_metric_1ch(b[1], metric, as_df=False)

    res = {
        channel_names[0]: res_l,
        channel_names[1]: res_r,
    }
    if as_df:
        try:
            rec = b.recording
        except:
            rec = 0
        df = pd.DataFrame.from_dict(res, orient="index")
        df["Recording"] = rec
        df["Channel"] = df.index
        df.set_index(["Recording", "Channel"], inplace=True)
        return df
    else:
        return res


# Analysis dataframe functions
def prep_multiindex_df(dictionary: dict, label: str = "Leq", incl_metric: bool = True):
    """df help to prepare a MultiIndex dataframe from a dictionary of results

    Parameters
    ----------
    dictionary : dict
        Dict of results with recording name as key, channels {"Left", "Right"} as second key, and Leq metric as value
    metric_name : str, optional
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


def add_results(results_df: pd.DataFrame, results: pd.DataFrame):
    """Add results to MultiIndex dataframe

    Parameters
    ----------
    results_df : pd.DataFrame
        MultiIndex dataframe to add results to
    results : pd.DataFrame
        MultiIndex dataframe of results to add

    Returns
    -------
    pd.DataFrame
        Index includes "Recording" and "Channel" with a column for each index.
    """
    if not set(results.columns).issubset(set(results_df.columns)):
        # Check if results_df already has the columns in results
        results_df = results_df.join(results)
    else:
        results_df.update(results, errors="ignore")
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

    # Loop through options in analysis_settings
    for library in analysis_settings.keys():
        # Ptyhon Acoustics metrics
        if library == "PythonAcoustics":
            for metric in analysis_settings[library].keys():
                results_df = pd.concat(
                    (
                        results_df,
                        b.pyacoustics_metric(
                            metric, analysis_settings=analysis_settings, verbose=verbose
                        ),
                    ),
                    axis=1,
                )
        # MosQITO metrics
        elif library == "MoSQITO":
            for metric in analysis_settings[library].keys():
                results_df = pd.concat(
                    (
                        results_df,
                        b.mosqito_metric(
                            metric,
                            analysis_settings=analysis_settings,
                            parallel=parallel,
                            verbose=verbose,
                        ),
                    ),
                    axis=1,
                )
        # scikit-maad metrics
        elif library == "maad":
            for metric in analysis_settings[library].keys():
                results_df = pd.concat(
                    (
                        results_df,
                        b.maad_metric(
                            metric, analysis_settings=analysis_settings, verbose=verbose
                        ),
                    ),
                    axis=1,
                )

    return results_df


# %%

# from acoustics import Signal
# from pathlib import Path

# wav_folder = Path("test", "data")
# binaural_wav = wav_folder.joinpath("CT101.wav")
# s = Signal.from_wav(binaural_wav)

# %%
