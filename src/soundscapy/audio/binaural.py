import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from acoustics import Signal

from soundscapy.audio.metrics import (
    maad_metric_1ch,
    maad_metric_2ch,
    mosqito_metric_1ch,
    mosqito_metric_2ch,
    process_all_metrics,
    pyacoustics_metric_1ch,
    pyacoustics_metric_2ch,
)


class Binaural(Signal):
    """
    A class for processing and analyzing binaural audio signals.

    This class extends the Signal class from the acoustics library to provide
    specialized functionality for binaural recordings. It supports various
    psychoacoustic metrics and analysis techniques using libraries such as
    mosqito, maad, and python-acoustics.

    Attributes:
        fs (float): Sampling frequency of the signal.
        recording (str): Name or identifier of the recording.

    Inherits all attributes and methods from acoustics.Signal.

    Note:
        This class only supports 2-channel (stereo) audio signals.
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

    def calibrate_to(
        self, decibel: float | List[float] | Tuple[float, float], inplace: bool = False
    ) -> "Binaural":
        """
        Calibrate the binaural signal to predefined Leq/dB levels.

        This method allows calibration of both channels either to the same level
        or to different levels for each channel.

        Args:
            decibel (float | List[float] | Tuple[float, float]): Target calibration value(s) in dB (Leq).
                If a single value is provided, both channels will be calibrated to this level.
                If two values are provided, they will be applied to the left and right channels respectively.
            inplace (bool, optional): If True, modify the signal in place. If False, return a new calibrated signal.
                Defaults to False.

        Returns:
            Binaural: Calibrated Binaural signal. If inplace is True, returns self.

        Raises:
            ValueError: If decibel is not a float, or a list/tuple of two floats.

        Example:
            >>> signal = Binaural.from_wav("audio.wav")
            >>> calibrated_signal = signal.calibrate_to([60, 62])  # Calibrate left channel to 60 dB and right to 62 dB
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
        filename: Path | str,
        calibrate_to: float | List | Tuple = None,
        normalize: bool = False,
    ):
        """Load a wav file and return a Binaural object

        Overrides the Signal.from_wav method to return a
        Binaural object instead of a Signal object.

        Parameters
        ----------
        filename : Path, str
            Filename of wav file to load
        calibrate_to : float, list or tuple of float, optional
            Value(s) to calibrate to in dB (Leq)
            Can also handle np.ndarray and pd.Series of length 2.
            If only one value is passed, will calibrate both channels to the same value.
        normalize : bool, optional
            Whether to normalize the signal, by default False

        Returns
        -------
        Binaural
            Binaural signal object of wav recording

        See Also
        --------
        acoustics.Signal.from_wav : Base method for loading wav files
        """
        s = super().from_wav(filename, normalize)
        if calibrate_to is not None:
            s.calibrate_to(calibrate_to, inplace=True)
        return cls(s, s.fs, recording=filename.stem)

    def _get_channel(self, channel):
        """Get a single channel from the signal

        Parameters
        ----------
        channel : int
            Channel to get (0 or 1)

        Returns
        -------
        Signal
            Single channel signal
        """
        if self.channels == 1:
            return self
        elif (
            channel is None
            or channel == "both"
            or channel == ("Left", "Right")
            or channel == ["Left", "Right"]
        ):
            return self
        elif channel in ["Left", 0, "L"]:
            return self[0]
        elif channel in ["Right", 1, "R"]:
            return self[1]
        else:
            warnings.warn("Channel not recognised. Returning Binaural object as is.")
            return self

    # Python Acoustics metrics
    def pyacoustics_metric(
        self,
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
        channel: str | int | List | Tuple = ("Left", "Right"),
        as_df: bool = True,
        return_time_series: bool = False,
        verbose: bool = False,
        analysis_settings: "AnalysisSettings" = None,
        func_args={},
    ):
        """Run a metric from the python acoustics library

        Parameters
        ----------
        metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
            The metric to run.
        statistics : tuple or list, optional
            List of level statistics to calulate (e.g. L_5, L_90, etc.),
                by default ( 5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew", )
        label : str, optional
            Label to use for the metric, by default None
            If None, will pull from default label for that metric given in sq_metrics.DEFAULT_LABELS
        channel : tuple, list, or str, optional
            Which channels to process, by default None
            If None, will process both channels
        as_df: bool, optional
            Whether to return a dataframe or not, by default True
            If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
        return_time_series: bool, optional
            Whether to return the time series of the metric, by default False
            Cannot return time series if as_df is True
        verbose : bool, optional
            Whether to print status updates, by default False
        analysis_settings : AnalysisSettings, optional
            Settings for analysis, by default None

            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.
        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame

        See Also
        --------
        metrics.pyacoustics_metric
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
                func_args,
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
                func_args,
            )

    # # Mosqito Metrics
    def mosqito_metric(
        self,
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
        channel: int | Tuple | List | str = ("Left", "Right"),
        as_df: bool = True,
        return_time_series: bool = False,
        parallel: bool = True,
        verbose: bool = False,
        analysis_settings: "AnalysisSettings" = None,
        func_args={},
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
        statistics : tuple or list, optional
            List of level statistics to calculate (e.g. L_5, L_90, etc.),
                by default (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew")
        label : str, optional
            Label to use for the metric, by default None
            If None, will pull from default label for that metric given in sq_metrics.DEFAULT_LABELS
        channel : tuple or list of str or str, optional
            Which channels to process, by default ("Left", "Right")
        as_df: bool, optional
            Whether to return a dataframe or not, by default True
            If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
        return_time_series: bool, optional
            Whether to return the time series of the metric, by default False
            Cannot return time series if as_df is True
        parallel : bool, optional
            Whether to run the channels in parallel, by default True
            If False, will run each channel sequentially.
            If being run as part of a larger parallel analysis (e.g. processing many recordings at once), this will
            automatically be set to False.
        verbose : bool, optional
            Whether to print status updates, by default False
        analysis_settings : AnalysisSettings, optional
            Settings for analysis, by default None

            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.
        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame

        See Also
        --------
        binaural.mosqito_metric_2ch : Method for running metrics on 2 channels
        binaural.mosqito_metric_1ch : Method for running metrics on 1 channel
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
                s, metric, statistics, label, as_df, return_time_series, func_args
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
                func_args,
            )

    # scikit-maad metrics
    def maad_metric(
        self,
        metric: str,
        channel: int | Tuple | List | str = ("Left", "Right"),
        as_df: bool = True,
        verbose: bool = False,
        analysis_settings: "AnalysisSettings" = None,
        func_args={},
    ):
        """Run a metric from the scikit-maad library

        Currently only supports running all of the alpha indices at once.

        Parameters
        ----------
        metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
            The metric to run
        channel : tuple, list or str, optional
            Which channels to process, by default None
        as_df: bool, optional
            Whether to return a dataframe or not, by default True
            If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
        verbose : bool, optional
            Whether to print status updates, by default False
        analysis_settings : AnalysisSettings, optional
            Settings for analysis, by default None

            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.
        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame


        Raises
        ------
        ValueError
            If metric name is not recognised.
        """
        if analysis_settings:
            if metric in {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}:
                run, channel = analysis_settings.parse_maad_all_alpha_indices(metric)
            else:
                raise ValueError(f"Metric {metric} not recognised")
            if run is False:
                return None
        channel = ("Left", "Right") if channel is None else channel
        s = self._get_channel(channel)
        if s.channels == 1:
            return maad_metric_1ch(s, metric, as_df, verbose, func_args)
        else:
            return maad_metric_2ch(s, metric, channel, as_df, verbose, func_args)

    def process_all_metrics(
        self,
        analysis_settings: "AnalysisSettings",
        parallel: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Process all metrics specified in the analysis settings.

        This method runs all enabled metrics from the provided AnalysisSettings object
        and compiles the results into a single DataFrame.

        Args:
            analysis_settings (AnalysisSettings): Configuration object specifying which metrics to run and their parameters.
            parallel (bool, optional): Whether to run calculations in parallel where possible. Defaults to True.
            verbose (bool, optional): If True, print progress information. Defaults to False.

        Returns:
            pd.DataFrame: A MultiIndex DataFrame containing the results of all processed metrics.
                          The index includes "Recording" and "Channel" levels.

        Note:
            The parallel option primarily affects the MoSQITo metrics. Other metrics may not benefit from parallelization.

        Example:
            >>> signal = Binaural.from_wav("audio.wav")
            >>> settings = AnalysisSettings.from_yaml("settings.yaml")
            >>> results = signal.process_all_metrics(settings, verbose=True)
        """
        return process_all_metrics(self, analysis_settings, parallel, verbose)


__all__ = ["Binaural"]
