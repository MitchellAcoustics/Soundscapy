from pathlib import Path

from acoustics import Signal

from soundscapy import AnalysisSettings
from soundscapy.analysis.binaural import *
from soundscapy.analysis.metrics import *


class Binaural(Signal):
    """Binaural signal class for analysis of binaural signals.
    A signal consisting of 2D samples (array) and a sampling frequency (fs).

    Subclasses the Signal class from python acoustics.
    Also adds attributes for the recording name.
    Adds the ability to do binaural analysis using the acoustics, scikit-maad and mosqito libraries.
    Optimised for batch processing with analysis settings predefined in a yaml file and passed to the class
    via the AnalysisSettings class.

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
        """Calibrate two channel signal to predefined Leq/dB levels.

        Parameters
        ----------
        decibel : float, list or tuple of float
            Value(s) to calibrate to in dB (Leq)
            Can also handle np.ndarray and pd.Series of length 2.
            If only one value is passed, will calibrate both channels to the same value.
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
        calibrate_to: Union[float, list, tuple] = None,
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
        as_df: bool = True,
        return_time_series: bool = False,
        verbose: bool = False,
        analysis_settings: AnalysisSettings = None,
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
        channel: Union[int, tuple, list, str] = ("Left", "Right"),
        as_df: bool = True,
        verbose: bool = False,
        analysis_settings: AnalysisSettings = None,
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
        analysis_settings: AnalysisSettings,
        parallel: bool = True,
        verbose: bool = False,
    ):
        """Run all metrics specified in the AnalysisSettings object

        Parameters
        ----------
        analysis_settings : AnalysisSettings
            Analysis settings object
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
            MultiIndex Dataframe of results.
            Index includes "Recording" and "Channel" with a column for each metric.
        """
        return process_all_metrics(self, analysis_settings, parallel, verbose)


__all__ = ["Binaural"]
