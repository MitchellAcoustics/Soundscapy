"""
soundscapy.audio.binaural
=========================

This module provides tools for working with binaural audio signals.

The main class, Binaural, extends the Signal class from the acoustics library
to provide specialized functionality for binaural recordings. It supports
various psychoacoustic metrics and analysis techniques using libraries such
as mosqito, maad, and python-acoustics.

Classes
-------
Binaural : A class for processing and analyzing binaural audio signals.

Notes
-----
This module requires the following external libraries:
- acoustics
- mosqito
- maad
- python-acoustics

Examples
--------
>>> # xdoctest: +SKIP
>>> from soundscapy.audio import Binaural
>>> signal = Binaural.from_wav("audio.wav")
>>> results = signal.process_all_metrics(analysis_settings)
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

import numpy as np
import pandas as pd
import scipy.signal
from acoustics import Signal

from .analysis_settings import AnalysisSettings, MetricSettings
from .metrics import (
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

    Attributes
    ----------
    fs : float
        Sampling frequency of the signal.
    recording : str
        Name or identifier of the recording.

    Notes
    -----
    This class only supports 2-channel (stereo) audio signals.
    """

    def __new__(cls, data, fs, recording="Rec"):
        """
        Create a new Binaural object.

        Parameters
        ----------
        data : array_like
            The audio data.
        fs : float
            Sampling frequency of the signal.
        recording : str, optional
            Name or identifier of the recording. Default is "Rec".

        Returns
        -------
        Binaural
            A new Binaural object.

        Raises
        ------
        ValueError
            If the input signal is not 2-channel.
        """
        obj = super().__new__(cls, data, fs).view(cls)
        obj.recording = recording
        if obj.channels != 2:
            logger.error(
                f"Attempted to create Binaural object with {obj.channels} channels"
            )
            raise ValueError("Binaural class only supports 2 channels.")
        logger.debug(f"Created new Binaural object: {recording}, fs={fs}")
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the new Binaural object.

        This method is called for all new Binaural objects.

        Parameters
        ----------
        obj : Binaural or None
            The object from which the new object was created.
        """
        if obj is None:
            return
        self.fs = getattr(obj, "fs", None)
        self.recording = getattr(obj, "recording", None)

    def calibrate_to(
        self,
        decibel: Union[float, List[float], Tuple[float, float]],
        inplace: bool = False,
    ) -> "Binaural":
        """
        Calibrate the binaural signal to predefined Leq/dB levels.

        This method allows calibration of both channels either to the same level
        or to different levels for each channel.

        Parameters
        ----------
        decibel : float or List[float] or Tuple[float, float]
            Target calibration value(s) in dB (Leq).
            If a single value is provided, both channels will be calibrated to this level.
            If two values are provided, they will be applied to the left and right channels respectively.
        inplace : bool, optional
            If True, modify the signal in place. If False, return a new calibrated signal.
            Default is False.

        Returns
        -------
        Binaural
            Calibrated Binaural signal. If inplace is True, returns self.

        Raises
        ------
        ValueError
            If decibel is not a float, or a list/tuple of two floats.

        Examples
        --------
        >>> # xdoctest: +SKIP
        >>> signal = Binaural.from_wav("audio.wav")
        >>> calibrated_signal = signal.calibrate_to([60, 62])  # Calibrate left channel to 60 dB and right to 62 dB
        """
        logger.info(f"Calibrating Binaural signal to {decibel} dB")
        if isinstance(decibel, (np.ndarray, pd.Series)):  # Force into tuple
            decibel = tuple(decibel)
        if isinstance(decibel, (list, tuple)):
            if len(decibel) == 2:  # Per-channel calibration (recommended)
                logger.debug(
                    f"Calibrating channels separately: Left={decibel[0]}dB, Right={decibel[1]}dB"
                )
                decibel = np.array(decibel)
                decibel = decibel[..., None]
                return super().calibrate_to(decibel, inplace)
            elif (
                len(decibel) == 1
            ):  # if one value given in tuple, assume same for both channels
                logger.debug(f"Calibrating both channels to {decibel[0]}dB")
                decibel = decibel[0]
            else:
                logger.error(f"Invalid calibration value: {decibel}")
                raise ValueError(
                    "decibel must either be a single value or a 2 value tuple"
                )
        if isinstance(decibel, (int, float)):  # Calibrate both channels to same value
            logger.debug(f"Calibrating both channels to {decibel}dB")
            return super().calibrate_to(decibel, inplace)
        else:
            logger.error(f"Invalid calibration value: {decibel}")
            raise ValueError("decibel must be a single value or a 2 value tuple")

    @classmethod
    def from_wav(
        cls,
        filename: Union[Path, str],
        calibrate_to: Optional[Union[float, List, Tuple]] = None,
        normalize: bool = False,
        resample: Optional[int] = None,
    ) -> "Binaural":
        """
        Load a wav file and return a Binaural object.

        Overrides the Signal.from_wav method to return a
        Binaural object instead of a Signal object.

        Parameters
        ----------
        filename : Path or str
            Filename of wav file to load.
        calibrate_to : float or List or Tuple, optional
            Value(s) to calibrate to in dB (Leq).
            Can also handle np.ndarray and pd.Series of length 2.
            If only one value is passed, will calibrate both channels to the same value.
        normalize : bool, optional
            Whether to normalize the signal. Default is False.
        resample : int, optional
            New sampling frequency to resample the signal to. Default is None

        Returns
        -------
        Binaural
            Binaural signal object of wav recording.

        See Also
        --------
        acoustics.Signal.from_wav : Base method for loading wav files.
        """
        logger.info(f"Loading WAV file: {filename}")
        s = super().from_wav(filename, normalize)
        b = cls(s, s.fs, recording=Path(filename).stem)
        if calibrate_to is not None:
            logger.info(f"Calibrating loaded signal to {calibrate_to} dB")
            b.calibrate_to(calibrate_to, inplace=True)
        if resample is not None:
            logger.debug(f"Resampling loaded signal to {resample} Hz")
            b = b.fs_resample(resample)
        return b

    def fs_resample(
        self,
        fs: float,
    ) -> "Binaural":
        """
        Resample the signal to a new sampling frequency.

        Parameters
        ----------
        fs : float
            New sampling frequency.


        Returns
        -------
        Binaural
            Resampled Binaural signal. If inplace is True, returns self.

        See Also
        --------
        acoustics.Signal.resample : Base method for resampling signals.
        """
        if fs == self.fs:
            logger.info(f"Signal already at {fs} Hz. No resampling needed.")
            return self
        logger.info(f"Resampling signal to {fs} Hz")
        resampled_channels = [
            scipy.signal.resample(channel, int(fs * len(channel) / self.fs))
            for channel in self
        ]
        resampled_channels = np.stack(resampled_channels)
        resampled_b = Binaural(resampled_channels, fs, recording=self.recording)
        return resampled_b

    def _get_channel(self, channel):
        """
        Get a single channel from the signal.

        Parameters
        ----------
        channel : int or str
            Channel to get (0 or 1, "Left" or "Right").

        Returns
        -------
        Signal
            Single channel signal.
        """
        if self.channels == 1:
            logger.debug("Returning single channel signal")
            return self
        elif (
            channel is None
            or channel == "both"
            or channel == ("Left", "Right")
            or channel == ["Left", "Right"]
        ):
            logger.debug("Returning both channels")
            return self
        elif channel in ["Left", 0, "L"]:
            logger.debug("Returning left channel")
            return self[0]
        elif channel in ["Right", 1, "R"]:
            logger.debug("Returning right channel")
            return self[1]
        else:
            logger.warning(
                f"Unrecognized channel specification: {channel}. Returning full Binaural object."
            )
            warnings.warn("Channel not recognised. Returning Binaural object as is.")
            return self

    def pyacoustics_metric(
        self,
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
        label: Optional[str] = None,
        channel: Union[str, int, List, Tuple] = ("Left", "Right"),
        as_df: bool = True,
        return_time_series: bool = False,
        metric_settings: Optional[MetricSettings] = None,
        func_args: Dict = {},
    ) -> Union[Dict, pd.DataFrame]:
        """
        Run a metric from the python acoustics library.

        Parameters
        ----------
        metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
            The metric to run.
        statistics : tuple or list, optional
            List of level statistics to calculate (e.g. L_5, L_90, etc.).
            Default is (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew").
        label : str, optional
            Label to use for the metric. If None, will pull from default label for that metric.
        channel : tuple, list, or str, optional
            Which channels to process. Default is ("Left", "Right").
        as_df : bool, optional
            Whether to return a dataframe or not. Default is True.
            If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
        return_time_series : bool, optional
            Whether to return the time series of the metric. Default is False.
            Cannot return time series if as_df is True.
        metric_settings : MetricSettings, optional
            Settings for metric analysis. Default is None.
        func_args : dict, optional
            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.

        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame.

        See Also
        --------
        metrics.pyacoustics_metric
        acoustics.standards_iso_tr_25417_2007.equivalent_sound_pressure_level : Base method for Leq calculation.
        acoustics.standards.iec_61672_1_2013.sound_exposure_level : Base method for SEL calculation.
        acoustics.standards.iec_61672_1_2013.time_weighted_sound_level : Base method for Leq level time series calculation.
        """
        if metric_settings:
            logger.debug("Using provided analysis settings")
            if not metric_settings.run:
                logger.info(f"Metric {metric} is disabled in analysis settings")
                return None

            channel = metric_settings.channel
            statistics = metric_settings.statistics
            label = metric_settings.label
            func_args = metric_settings.func_args

        channel = ("Left", "Right") if channel is None else channel
        s = self._get_channel(channel)

        if s.channels == 1:
            logger.debug("Processing single channel")
            return pyacoustics_metric_1ch(
                s, metric, statistics, label, as_df, return_time_series, func_args
            )
        else:
            logger.debug("Processing both channels")
            return pyacoustics_metric_2ch(
                s,
                metric,
                statistics,
                label,
                channel,
                as_df,
                return_time_series,
                func_args,
            )

    def mosqito_metric(
        self,
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
        label: Optional[str] = None,
        channel: Union[int, Tuple, List, str] = ("Left", "Right"),
        as_df: bool = True,
        return_time_series: bool = False,
        parallel: bool = True,
        metric_settings: Optional[MetricSettings] = None,
        func_args: Dict = {},
    ) -> Union[Dict, pd.DataFrame]:
        """
        Run a metric from the mosqito library.

        Parameters
        ----------
        metric : {"loudness_zwtv", "sharpness_din_from_loudness", "sharpness_din_perseg", "sharpness_tv", "roughness_dw"}
            Metric to run from mosqito library.
        statistics : tuple or list, optional
            List of level statistics to calculate (e.g. L_5, L_90, etc.).
            Default is (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew").
        label : str, optional
            Label to use for the metric. If None, will pull from default label for that metric.
        channel : tuple or list of str or str, optional
            Which channels to process. Default is ("Left", "Right").
        as_df : bool, optional
            Whether to return a dataframe or not. Default is True.
            If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
        return_time_series : bool, optional
            Whether to return the time series of the metric. Default is False.
            Cannot return time series if as_df is True.
        parallel : bool, optional
            Whether to run the channels in parallel. Default is True.
            If False, will run each channel sequentially.
        metric_settings : MetricSettings, optional
            Settings for metric analysis. Default is None.
        func_args : dict, optional
            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying python acoustics method.

        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame.

        See Also
        --------
        binaural.mosqito_metric_2ch : Method for running metrics on 2 channels.
        binaural.mosqito_metric_1ch : Method for running metrics on 1 channel.
        """
        logger.info(f"Running mosqito metric: {metric}")
        if metric_settings:
            logger.debug("Using provided analysis settings")
            if not metric_settings.run:
                logger.info(f"Metric {metric} is disabled in analysis settings")
                return None

            channel = metric_settings.channel
            statistics = metric_settings.statistics
            label = metric_settings.label
            parallel = metric_settings.parallel
            func_args = metric_settings.func_args

        channel = ("Left", "Right") if channel is None else channel
        s = self._get_channel(channel)

        if s.channels == 1:
            logger.debug("Processing single channel")
            return mosqito_metric_1ch(
                s, metric, statistics, label, as_df, return_time_series, func_args
            )
        else:
            logger.debug("Processing both channels")
            return mosqito_metric_2ch(
                s,
                metric,
                statistics,
                label,
                channel,
                as_df,
                return_time_series,
                parallel,
                func_args,
            )

    def maad_metric(
        self,
        metric: str,
        channel: Union[int, Tuple, List, str] = ("Left", "Right"),
        as_df: bool = True,
        metric_settings: Optional[MetricSettings] = None,
        func_args: Dict = {},
    ) -> Union[Dict, pd.DataFrame]:
        """
        Run a metric from the scikit-maad library.

        Currently only supports running all of the alpha indices at once.

        Parameters
        ----------
        metric : {"all_temporal_alpha_indices", "all_spectral_alpha_indices"}
            The metric to run.
        channel : tuple, list or str, optional
            Which channels to process. Default is ("Left", "Right").
        as_df : bool, optional
            Whether to return a dataframe or not. Default is True.
            If True, returns a MultiIndex Dataframe with ("Recording", "Channel") as the index.
        metric_settings : MetricSettings, optional
            Settings for metric analysis. Default is None.
        func_args : dict, optional
            Additional arguments to pass to the underlying scikit-maad method.

        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame.

        Raises
        ------
        ValueError
            If metric name is not recognised.

        See Also
        --------
        metrics.maad_metric_1ch
        metrics.maad_metric_2ch
        """
        logger.info(f"Running maad metric: {metric}")
        if metric_settings:
            logger.debug("Using provided analysis settings")
            if metric not in {
                "all_temporal_alpha_indices",
                "all_spectral_alpha_indices",
            }:
                logger.error(f"Invalid maad metric: {metric}")
                raise ValueError(f"Metric {metric} not recognised")

            if not metric_settings.run:
                logger.info(f"Metric {metric} is disabled in analysis settings")
                return None

            channel = metric_settings.channel
        channel = ("Left", "Right") if channel is None else channel
        s = self._get_channel(channel)
        if s.channels == 1:
            logger.debug("Processing single channel")
            return maad_metric_1ch(s, metric, as_df)
        else:
            logger.debug("Processing both channels")
            return maad_metric_2ch(s, metric, channel, as_df, func_args)

    def process_all_metrics(
        self,
        analysis_settings: AnalysisSettings = AnalysisSettings.default(),
        parallel: bool = True,
    ) -> pd.DataFrame:
        """
        Process all metrics specified in the analysis settings.

        This method runs all enabled metrics from the provided AnalysisSettings object
        and compiles the results into a single DataFrame.

        Parameters
        ----------
        analysis_settings : AnalysisSettings
            Configuration object specifying which metrics to run and their parameters.
        parallel : bool, optional
            Whether to run calculations in parallel where possible. Default is True.

        Returns
        -------
        pd.DataFrame
            A MultiIndex DataFrame containing the results of all processed metrics.
            The index includes "Recording" and "Channel" levels.

        Notes
        -----
        The parallel option primarily affects the MoSQITo metrics. Other metrics may not benefit from parallelization.

        TODO: Provide default settings to analysis_settings to make it optional.

        Examples
        --------
        >>> # xdoctest: +SKIP
        >>> signal = Binaural.from_wav("audio.wav")
        >>> settings = AnalysisSettings.from_yaml("settings.yaml")
        >>> results = signal.process_all_metrics(settings)
        """
        logger.info(f"Processing all metrics for {self.recording}")
        logger.debug(f"Parallel processing: {parallel}")
        return process_all_metrics(self, analysis_settings, parallel)


__all__ = ["Binaural"]
