"""
Provides tools for working with binaural audio signals.

The main class, Binaural, extends the Signal class from the Acoustic Toolbox library
to provide specialized functionality for binaural recordings. It supports
various psychoacoustic metrics and analysis techniques using libraries such
as mosqito, maad, and acoustic_toolbox.

Classes
-------
Binaural : A class for processing and analyzing binaural audio signals.

Notes
-----
This module requires the following external libraries:
- acoustics
- mosqito
- maad
- acoustic_toolbox

Examples
--------
>>> # xdoctest: +SKIP
>>> from soundscapy.audio import Binaural
>>> signal = Binaural.from_wav("audio.wav")
>>> results = signal.process_all_metrics(analysis_settings)

"""

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.signal
from acoustic_toolbox import Signal
from loguru import logger
from scipy.io import wavfile

from soundscapy._utils import ensure_input_path
from soundscapy.audio.analysis_settings import AnalysisSettings, MetricSettings
from soundscapy.audio.metrics import (
    acoustics_metric_1ch,
    acoustics_metric_2ch,
    maad_metric_1ch,
    maad_metric_2ch,
    mosqito_metric_1ch,
    mosqito_metric_2ch,
    process_all_metrics,
)

ALLOWED_BINAURAL_CHANNELS = 2


class Binaural(Signal):
    """
    A class for processing and analyzing binaural audio signals.

    This class extends the Signal class from the acoustic_toolbox library to provide
    specialized functionality for binaural recordings. It supports various
    psychoacoustic metrics and analysis techniques using libraries such as
    mosqito, maad, and acoustic_toolbox.

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

    def __new__(
        cls, data: np.ndarray, fs: float | None, recording: str = "Rec"
    ) -> "Binaural":
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
        if obj.channels != ALLOWED_BINAURAL_CHANNELS:
            logger.error(
                f"Attempted to create Binaural object with {obj.channels} channels"
            )
            msg = "Binaural class only supports 2 channels."
            raise ValueError(msg)
        logger.debug(f"Created new Binaural object: {recording}, fs={fs}")
        return obj

    def __array_finalize__(self, obj: "Binaural | None") -> None:
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
        self.recording = getattr(obj, "recording", "Rec")

    def calibrate_to(
        self,
        decibel: float | list[float] | tuple[float, float] | np.ndarray | pd.Series,
        inplace: bool = False,  # noqa: FBT001, FBT002 TODO(MitchellAcoustics): Change to keyword-only in acoustic_toolbox.Signal
    ) -> "Binaural":
        """
        Calibrate the binaural signal to predefined Leq/dB levels.

        This method allows calibration of both channels either to the same level
        or to different levels for each channel.

        Parameters
        ----------
        decibel : float or List[float] or Tuple[float, float]
            Target calibration value(s) in dB (Leq).
            If a single value is provided, both channels will be calibrated
            to this level.
            If two values are provided, they will be applied to the left and right
            channels respectively.
        inplace : bool, optional
            If True, modify the signal in place.
            If False, return a new calibrated signal.
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
        >>> # Calibrate left channel to 60 dB and right to 62 dB
        >>> calibrated_signal = signal.calibrate_to([60, 62])

        """
        logger.info(f"Calibrating Binaural signal to {decibel} dB")
        if isinstance(decibel, np.ndarray | pd.Series):  # Force into tuple
            decibel = tuple(decibel)
        if isinstance(decibel, list | tuple):
            if (
                len(decibel) == ALLOWED_BINAURAL_CHANNELS
            ):  # Per-channel calibration (recommended)
                logger.debug(
                    "Calibrating channels separately: "
                    f"Left={decibel[0]}dB, Right={decibel[1]}dB"
                )
                decibel = np.array(decibel)
                decibel = decibel[..., None]
                return super().calibrate_to(decibel, inplace)  # type: ignore[reportReturnType]
            if (
                len(decibel) == 1
            ):  # if one value given in tuple, assume same for both channels
                logger.debug(f"Calibrating both channels to {decibel[0]}dB")
                decibel = decibel[0]
            else:
                logger.error(f"Invalid calibration value: {decibel}")
                msg = "decibel must either be a single value or a 2 value tuple"
                raise TypeError(msg)
        if isinstance(decibel, int | float):  # Calibrate both channels to same value
            logger.debug(f"Calibrating both channels to {decibel}dB")
            return super().calibrate_to(decibel, inplace)  # type: ignore[reportReturnType]
        logger.error(f"Invalid calibration value: {decibel}")
        msg = "decibel must be a single value or a 2 value tuple"
        raise TypeError(msg)

    @classmethod
    def from_wav(
        cls,
        filename: Path | str,
        normalize: bool = False,  # noqa: FBT001, FBT002
        calibrate_to: float | list | tuple | None = None,
        resample: int | None = None,
        recording: str | None = None,
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
        acoustic_toolbox.Signal.from_wav : Base method for loading wav files.

        """
        filename = ensure_input_path(filename)
        if not filename.exists():
            logger.error(f"File not found: {filename}")
            msg = f"File not found: {filename}"
            raise FileNotFoundError(msg)

        logger.info(f"Loading WAV file: {filename}")
        fs, data = wavfile.read(filename)
        data = data.astype(np.float32, copy=False).T
        if normalize:
            data /= np.max(np.abs(data))

        recording = recording if recording is not None else Path(filename).stem
        b = cls(data, fs, recording=recording)

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
        original_fs: float | None = None,
    ) -> "Binaural":
        """
        Resample the signal to a new sampling frequency.

        Parameters
        ----------
        fs : float
            New sampling frequency.
        original_fs : float or None, optional
            Original sampling frequency.
            If None, it will be inferred from the signal (`Binaural.fs`).

        Returns
        -------
        Binaural
            Resampled Binaural signal. If inplace is True, returns self.

        See Also
        --------
        acoustic_toolbox.Signal.resample : Base method for resampling signals.

        """
        current_fs: float

        if original_fs is None:
            if hasattr(self, "fs") and self.fs is not None:
                current_fs = self.fs
            else:
                logger.error("Original sampling frequency not provided.")
                msg = "Original sampling frequency not provided."
                raise ValueError(msg)
        else:
            current_fs = original_fs

        if fs == current_fs:
            logger.info(f"Signal already at {current_fs} Hz. No resampling needed.")
            return self

        logger.info(f"Resampling signal to {fs} Hz")
        resampled_channels = [
            scipy.signal.resample(channel, int(fs * len(channel) / current_fs))
            for channel in self
        ]
        resampled_channels = np.stack(resampled_channels)
        return Binaural(resampled_channels, fs, recording=self.recording)

    def _get_channel(self, channel: int | str | None) -> Signal:
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
        if channel is None or channel in ("both", ("Left", "Right"), ["Left", "Right"]):
            logger.debug("Returning both channels")
            return self
        if channel in ["Left", 0, "L"]:
            logger.debug("Returning left channel")
            return self[0]
        if channel in ["Right", 1, "R"]:
            logger.debug("Returning right channel")
            return self[1]
        logger.warning(
            f"Unrecognized channel specification: {channel}."
            "Returning full Binaural object."
        )
        warnings.warn(
            "Channel not recognised. Returning Binaural object as is.", stacklevel=2
        )
        return self

    def pyacoustics_metric(
        self,
        metric: Literal["LZeq", "Leq", "LAeq", "LCeq", "SEL"],
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
        channel: str | int | list | tuple = ("Left", "Right"),
        as_df: bool = True,  # noqa: FBT001, FBT002
        return_time_series: bool = False,  # noqa: FBT001, FBT002
        metric_settings: MetricSettings | None = None,
        func_args: dict | None = None,
    ) -> dict | pd.DataFrame | None:
        """
        Run a metric from the pyacoustics library (deprecated).

        This method has been deprecated. Use `acoustics_metric` instead.
        All parameters are passed directly to `acoustics_metric`.

        Parameters
        ----------
        metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
            The metric to run.
        statistics : tuple or list, optional
            List of level statistics to calculate (e.g. L_5, L_90, etc.).
            Default is (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew").
        label : str, optional
            Label to use for the metric.
            If None, will pull from default label for that metric.
        channel : tuple, list, or str, optional
            Which channels to process. Default is ("Left", "Right").
        as_df : bool, optional
            Whether to return a dataframe or not. Default is True.
            If True, returns a MultiIndex Dataframe with
            ("Recording", "Channel") as the index.
        return_time_series : bool, optional
            Whether to return the time series of the metric. Default is False.
            Cannot return time series if as_df is True.
        metric_settings : MetricSettings, optional
            Settings for metric analysis. Default is None.
        func_args : dict, optional
            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying acoustic_toolbox method.

        Returns
        -------
        dict or pd.DataFrame
            Results of the metric calculation.

        See Also
        --------
        Binaural.acoustics_metric

        """
        if func_args is None:
            func_args = {}
        warnings.warn(
            "pyacoustics has been deprecated. Use acoustics_metric instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.acoustics_metric(
            metric,
            statistics,
            label,
            channel,
            as_df=as_df,
            return_time_series=return_time_series,
            metric_settings=metric_settings,
            func_args=func_args,
        )

    def acoustics_metric(
        self,
        metric: Literal["LZeq", "Leq", "LAeq", "LCeq", "SEL"],
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
        channel: str | int | list | tuple = ("Left", "Right"),
        metric_settings: MetricSettings | None = None,
        func_args: dict | None = None,
        *,
        as_df: bool = True,
        return_time_series: bool = False,
    ) -> dict | pd.DataFrame | None:
        """
        Run a metric from the acoustic_toolbox library.

        Parameters
        ----------
        metric : {"LZeq", "Leq", "LAeq", "LCeq", "SEL"}
            The metric to run.
        statistics : tuple or list, optional
            List of level statistics to calculate (e.g. L_5, L_90, etc.).
            Default is (5, 10, 50, 90, 95, "avg", "max", "min", "kurt", "skew").
        label : str, optional
            Label to use for the metric.
            If None, will pull from default label for that metric.
        channel : tuple, list, or str, optional
            Which channels to process. Default is ("Left", "Right").
        as_df : bool, optional
            Whether to return a dataframe or not. Default is True.
            If True, returns a MultiIndex Dataframe with
            ("Recording", "Channel") as the index.
        return_time_series : bool, optional
            Whether to return the time series of the metric. Default is False.
            Cannot return time series if as_df is True.
        metric_settings : MetricSettings, optional
            Settings for metric analysis. Default is None.
        func_args : dict, optional
            Any settings given here will override those in the other options.
            Can pass any *args or **kwargs to the underlying acoustic_toolbox method.

        Returns
        -------
        dict or pd.DataFrame
            Dictionary of results if as_df is False, otherwise a pandas DataFrame.

        See Also
        --------
        metrics.acoustics_metric
        acoustic_toolbox.standards_iso_tr_25417_2007.equivalent_sound_pressure_level :
            Base method for Leq calculation.
        acoustic_toolbox.standards.iec_61672_1_2013.sound_exposure_level :
            Base method for SEL calculation.
        acoustic_toolbox.standards.iec_61672_1_2013.time_weighted_sound_level :
            Base method for Leq level time series calculation.

        """
        if func_args is None:
            func_args = {}
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
            return acoustics_metric_1ch(
                s, metric, statistics, label, as_df, return_time_series, func_args
            )
        logger.debug("Processing both channels")
        return acoustics_metric_2ch(
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
        channel: int | tuple | list | str = ("Left", "Right"),
        as_df: bool = True,
        return_time_series: bool = False,
        parallel: bool = True,
        metric_settings: MetricSettings | None = None,
        func_args: dict = {},
    ) -> dict | pd.DataFrame:
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
            Can pass any *args or **kwargs to the underlying acoustic_toolbox method.

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
                s,
                metric,
                statistics,
                label,
                as_df=as_df,
                return_time_series=return_time_series,
                **func_args,
            )
        logger.debug("Processing both channels")
        return mosqito_metric_2ch(
            s,
            metric,
            statistics,
            label,
            channel,
            as_df=as_df,
            return_time_series=return_time_series,
            parallel=parallel,
            func_args=func_args,
        )

    def maad_metric(
        self,
        metric: str,
        channel: int | tuple | list | str = ("Left", "Right"),
        as_df: bool = True,
        metric_settings: MetricSettings | None = None,
        func_args: dict = {},
    ) -> dict | pd.DataFrame:
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
