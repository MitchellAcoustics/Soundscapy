# %%
# from WavAnalysis import *
from pathlib import Path
from time import localtime, strftime
from typing import Union

import numpy as np
import pytest
import yaml
from acoustics import Signal
from pytest import approx

from binaural import *
from sq_metrics import *
from _AnalysisSettings import AnalysisSettings


# %%
class Binaural(Signal):
    """Binaural signal class for analysis of binaural signals.
    A signal consisting of 2D samples (array) and a sampling frequency (fs).

    Subclasses the Signal class from python acoustics.
    Also adds attributes for the recording name and analysis settings.
    Adds the ability to do binaural analysis using the acoustics, scikit-maad and mosqito libraries.
    Optimised for batch processing with analysis settings predefined in a yaml file and passed to the class via the AnalysisSettings class.

    See Also
    --------
    acoustics.Signal : Base class for binaural signal
    """

    def __new__(cls, data, fs, recording="Rec"):
        obj = super().__new__(cls, data, fs).view(cls)
        obj.recording = recording
        if obj.channels != 2:
            raise ValueError("Binaural class only supports 2 channels.")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fs = getattr(obj, "fs", None)
        self.recording = getattr(obj, "recording", None)


    def calibrate_to(self, decibel: Union[float, list, tuple], inplace: bool = False):
        """Binaural signal class for analysis of binaural signals.

        Parameters
        ----------
        decibel : float, list or tuple of float
            Value(s) to calibrate to in dB (Leq)
            Can also handle np.ndarray and pd.Series of length 2.
        inplace : bool, optional
            Whether to perform inplace or not, by default False

        Returns
        -------
        Binaural
            Calibrated Binaural signal

        Raises
        ------
        ValueError
            If decibel is not a (float, int) or a list or tuple of length 2.
        ValueError
            If decibel is not a (float, int) or a list or tuple of length 2.

        See Also
        --------
        acoustics.Signal.calibrate_to : Base method for calibration. Cannot handle 2ch calibration
        """
        if isinstance(decibel, (np.ndarray, pd.Series)):  # Force into tuple
            decibel = tuple(decibel)
        if isinstance(decibel, (list, tuple)):
            if len(decibel) == 2:  # Per-channel calibration (recommended)
                decibel = np.array(decibel)
                decibel = decibel[..., None]
                return super().calibrate_to(decibel, inplace)
            elif (
                len(decibel) == 1
            ):  # if one value given in tuple, assume same for both channels
                decibel = decibel[0]
            else:
                raise ValueError(
                    "decibel must either be a single value or a 2 value tuple"
                )
        if isinstance(decibel, (int, float)):  # Calibrate both channels to same value
            return super().calibrate_to(decibel, inplace)
        else:
            raise ValueError("decibel must be a single value or a 2 value tuple")

    @classmethod
    def from_wav(
        cls,
        filename: Union[Path, str],
        normalize: bool = True,
    ):
        """Load a wav file and return a Binaural object

        Overrides the Signal.from_wav method to return a
        Binaural object instead of a Signal object.

        Parameters
        ----------
        filename : Path, str
            Filename of wav file to load
        normalize : bool, optional
            Whether to normalize the signal, by default True

        Returns
        -------
        Binaural
            Binaural signal object of wav recording

        See Also
        --------
        acoustics.Signal.from_wav : Base method for loading wav files
        """
        s = super().from_wav(filename, normalize)
        return cls(s, s.fs, recording=filename.stem)

    def _get_channel(self, channel):
        if self.channels == 1:
            return self
        elif channel == None or channel == "both" or channel == ("Left", "Right") or channel == ["Left", "Right"]:
            return self
        elif channel == "Left" or channel == 0 or channel == "L":
            return self[0]
        elif channel == "Right" or channel == 1 or channel == "R":
            return self[1]
        else:
            warnings.warn("Channel not recognised. Returning Binaural object as is.")
            return self
        

    # Python Acoustics metrics
    def pyacoustics_metric(
        self,
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
        channel: Union[str, int, list, tuple] = ("Left", "Right"),
        verbose: bool = False,
        as_df: bool = True,
        return_time_series: bool = False,
        analysis_settings: AnalysisSettings = None,
        **func_args
    ):
        """Run a metric from the python acoustics library

        Parameters
        ----------
        metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
            The metric to run.
        channel : tuple, list, or str, optional
            Which channels to process, by default None
            If None, will process both channels
        statistics : tuple or list, optional
            List of level statistics to calulate (e.g. L_5, L_90, etc.),
                by default ( 5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew", )
        label : str, optional
            Label to use for the metric, by default None
            If None, will pull from default label for that metric given in WavAnalysis.DEFAULT_LABELS
        verbose : bool, optional
            Whether to print status updates, by default False
        analysis_settings : AnalysisSettings, optional
            Settings for analysis, by default None

            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.
        Returns
        -------
        pd.DataFrame
            MultiIndex Dataframe of results.
            Index includes "Recording" and "Channel" with a column for each statistic.

        See Also
        --------
        WavAnalysis.pyacoustics_metric
        acoustics.standards_iso_tr_25417_2007.equivalent_sound_pressure_level : Base method for Leq calculation
        acoustics.standards.iec_61672_1_2013.sound_exposure_level : Base method for SEL calculation
        acoustics.standards.iec_61672_1_2013.time_weighted_sound_level : Base method for Leq level time series calculation
        """
        if analysis_settings:
            (
                run,
                channel,
                statistics,
                label,
                func_args,
            ) = analysis_settings.parse_pyacoustics(metric)
            if run is False:
                return None

        channel = ("Left", "Right") if channel is None else channel
        s = self._get_channel(channel)

        if s.channels == 1:
            return pyacoustics_metric_1ch(
                s,
                metric,
                statistics,
                label,
                as_df,
                return_time_series,
                verbose,
                **func_args
            )

        else:
            return pyacoustics_metric_2ch(
                s,
                metric,
                statistics,
                label,
                channel,
                as_df,
                return_time_series,
                verbose,
                **func_args
            )

    # # Mosqito Metrics
    def mosqito_metric(
        self,
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
        channel: Union[int, tuple, list, str] = ("Left", "Right"),
        as_df: bool = True,
        return_time_series: bool = False,
        parallel: bool = True,
        verbose: bool = False,
        analysis_settings: AnalysisSettings = None,
        **func_args
    ):
        """Run a metric from the mosqito library

        Parameters
        ----------
        metric : {"loudness_zwtv", "sharpness_din_from_loudness", "sharpness_din_perseg",
        "sharpness_tv", "roughness_dw"}
            Metric to run from mosqito library.

            In the case of "sharpness_din_from_loudness", the "loudness_zwtv" metric
            will be calculated first and then the sharpness will be calculated from that.
            This is because the sharpness_from loudness metric requires the loudness metric to be
            calculated. Loudness will be returned as well
        channel : tuple or list of str or str, optional
            Which channels to process, by default None
        statistics : Union, optional
            List of level statistics to calulate (e.g. L_5, L_90, etc.),
                by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
        label : str, optional
            Label to use for the metric, by default None
            If None, will pull from default label for that metric given in WavAnalysis.DEFAULT_LABELS
        verbose : bool, optional
            Whether to print status updates, by default False
        analysis_settings : AnalysisSettings, optional
            Settings for analysis, by default None

            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.
        Returns
        -------
        pd.DataFrame
            Index includes "Recording" and "Channel" with a column for each index.
        """
        if analysis_settings:
            (
                run,
                channel,
                statistics,
                label,
                parallel,
                func_args,
            ) = analysis_settings.parse_mosqito(metric)
            if run is False:
                return None

        channel = ("Left", "Right") if channel is None else channel
        s = self._get_channel(channel)

        if s.channels == 1:
            return mosqito_metric_1ch(
                s,
                metric,
                statistics,
                label,
                as_df,
                return_time_series,
                verbose,
                **func_args
            )
        else:
            return mosqito_metric_2ch(
                s,
                metric,
                statistics,
                label,
                channel,
                as_df,
                return_time_series,
                parallel,
                verbose,
                **func_args
            )

    # scikit-maad metrics
    def maad_metric(
        self,
        metric: str,
        channel: Union[int, tuple, list, str] = ("Left", "Right"),
        as_df: bool = True,
        verbose: bool = False,
        analysis_settings: AnalysisSettings = None,
        **func_args
    ):
        """Run a metric from the scikit-maad library

        Currently only supports running all of the alpha indices at once.

        Parameters
        ----------
        metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
            The metric to run
        channel : tuple, list or str, optional
            Which channels to process, by default None
        verbose : bool, optional
            Whether to print status updates, by default False
        analysis_settings : AnalysisSettings, optional
            Settings for analysis, by default None

            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.
        Returns
        -------
        pd.DataFrame
            MultiIndex Dataframe of results.
            Index includes "Recording" and "Channel" with a column for each index.

        Raises
        ------
        ValueError
            If metric name is not recognised.
        """
        if analysis_settings:
            if metric in ("all_temporal_alpha_indices", "all_spectral_alpha_indices"):
                run, channel = analysis_settings.parse_maad_all_alpha_indices(metric)
            else:
                raise ValueError(f"Metric {metric} not recognised")
            if run is False:
                return None

        channel = ("Left", "Right") if channel is None else channel

        s = self._get_channel(channel)

        if s.channels == 1:
            return maad_metric_1ch(
                s, metric, as_df, verbose, **func_args
            )
        else:
            return maad_metric_2ch(
                s, metric, channel, as_df, verbose, **func_args
            )            


    def process_all_metrics(self, analysis_settings: AnalysisSettings, parallel:bool = False, verbose:bool = False):
        """Run all metrics specified in the AnalysisSettings object

        Parameters
        ----------
        analysis_settings : AnalysisSettings
            Analysis settings object
        verbose : bool, optional
            Whether to print status updates, by default False

        Returns
        -------
        pd.DataFrame
            MultiIndex Dataframe of results.
            Index includes "Recording" and "Channel" with a column for each metric.
        """
        return process_all_metrics(self, analysis_settings, parallel, verbose)


__all__ = ["Binaural"]
# %%
