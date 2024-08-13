"""
soundscapy.audio._AnalysisSettings
==================================

This module provides the AnalysisSettings class, which is used to manage
and parse settings for various audio analysis methods.

The AnalysisSettings class is a dictionary-like object that stores settings
for different audio analysis libraries and metrics. It provides methods to
load settings from YAML files, parse settings for specific metrics, and
manage the execution of analysis tasks.

Classes:
    AnalysisSettings: Manages settings for audio analysis methods.

Functions:
    get_default_yaml: Retrieves default settings from the GitHub repository.
"""

import urllib.request
from pathlib import Path
from time import localtime, strftime
from typing import Union

import yaml


def get_default_yaml(save_as="default_settings.yaml"):
    """
    Retrieves the default settings for analysis from the GitHub repository
    and saves them to a file.

    Parameters
    ----------
    save_as : str, optional
        The name of the file to save the default settings to. Defaults to
        "default_settings.yaml".
    """
    print("Downloading default settings from GitHub...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/MitchellAcoustics/Soundscapy/main/soundscapy/analysis/default_settings.yaml",
        save_as,
    )


class AnalysisSettings(dict):
    """
    A dictionary-like class for managing settings for audio analysis methods.

    Each library has a dict of metrics, each of which has a dict of settings.
    This class provides methods to load settings from YAML files, parse settings
    for specific metrics, and manage the execution of analysis tasks.

    Attributes:
        run_stats (bool): Whether to include all stats or just the main metric.
        force_run_all (bool): Whether to force all metrics to run regardless of their settings.
        filepath (Union[str, Path]): Path to the YAML file containing the settings.

    Methods:
        from_yaml: Create an AnalysisSettings object from a YAML file.
        default: Create a default AnalysisSettings object.
        reload: Reload settings from the YAML file.
        to_yaml: Save settings to a YAML file.
        parse_maad_all_alpha_indices: Parse settings for MAAD alpha indices.
        parse_pyacoustics: Parse settings for pyacoustics metrics.
        parse_mosqito: Parse settings for MoSQITo metrics.
    """

    def __init__(
        self,
        data,
        run_stats=True,
        force_run_all=False,
        filepath: Union[str, Path] = None,
    ):
        super().__init__(data)
        self.run_stats = run_stats
        self.force_run_all = force_run_all
        self.filepath = filepath
        runtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
        super().__setitem__("runtime", runtime)

    @classmethod
    def from_yaml(cls, filename: Union[Path, str], run_stats=True, force_run_all=False):
        """
        Generate a settings object from a YAML file.

        Args:
            filename (Union[Path, str]): Filename of the YAML file.
            run_stats (bool, optional): Whether to include all stats listed or just return the main metric.
                Defaults to True.
            force_run_all (bool, optional): Whether to force all metrics to run regardless of their settings.
                Defaults to False.

        Returns:
            AnalysisSettings: An AnalysisSettings object.

        Note:
            If both mosqito:loudness_zwtv and mosqito:sharpness_din_from_loudness are present
            in the settings file, forcing all metrics to run will result in the loudness
            calculation being run twice.
        """
        with open(filename, "r") as f:
            return cls(
                yaml.load(f, Loader=yaml.Loader), run_stats, force_run_all, filename
            )

    @classmethod
    def default(cls, run_stats=True, force_run_all=False):
        """
        Generate a default settings object.

        Args:
            run_stats (bool, optional): Whether to include all stats listed or just return the main metric.
                Defaults to True.
            force_run_all (bool, optional): Whether to force all metrics to run regardless of their settings.
                Defaults to False.

        Returns:
            AnalysisSettings: A default AnalysisSettings object.

        Note:
            If both mosqito:loudness_zwtv and mosqito:sharpness_din_from_loudness are present
            in the settings file, forcing all metrics to run will result in the loudness
            calculation being run twice.
        """
        import soundscapy

        root = Path(soundscapy.__path__[0])
        return cls(
            AnalysisSettings.from_yaml(
                Path(root, "audio", "default_settings.yaml"),
                run_stats,
                force_run_all,
            )
        )

    def reload(self):
        """Reload the settings from the yaml file."""
        return self.from_yaml(self.filepath, self.run_stats, self.force_run_all)

    def to_yaml(self, filename: Union[Path, str]):
        """Save settings to a yaml file.

        Parameters
        ----------
        filename : Path object or str
            filename of the yaml file
        """
        with open(filename, "w") as f:
            yaml.dump(self, f)

    def parse_maad_all_alpha_indices(self, metric: str):
        """
        Generate relevant settings for the MAAD all_alpha_indices methods.

        Args:
            metric (str): Metric to prepare for. Must be either "all_temporal_alpha_indices"
                          or "all_spectral_alpha_indices".

        Returns:
            Tuple[bool, Union[Tuple[str, ...], List[str], str]]: A tuple containing:
                - run (bool): Whether to run the metric.
                - channel (Union[Tuple[str, ...], List[str], str]): Channel(s) to run the metric on.

        Raises:
            AssertionError: If the metric is not one of the supported alpha indices.
        """
        assert metric in [
            "all_temporal_alpha_indices",
            "all_spectral_alpha_indices",
        ], "metric must be all_temporal_alpha_indices or all_spectral_alpha_indices."

        lib_settings = self["scikit-maad"].copy()
        run = lib_settings[metric]["run"] or self.force_run_all
        channel = lib_settings[metric]["channel"].copy()
        return run, channel

    def parse_pyacoustics(self, metric: str):
        """Generate relevant settings for a pyacoustics metric.

        Parameters
        ----------
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        statistics: tuple or list of str, or str
            statistics to run the metric on.
            If run_stats is False, will only return the main statistic
        label: str
            label to use for the metric
        func_args: dict
            arguments to pass to the underlying metric function from python acoustics
        """
        return self._parse_method("PythonAcoustics", metric)

    def parse_mosqito(self, metric: str):
        """Generate relevant settings for a mosqito metric.

        Parameters
        ----------
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        statistics: tuple or list of str, or str
            statistics to run the metric on.
            If run_stats is False, will only return the main statistic
        label: str
            label to use for the metric
        func_args: dict
            arguments to pass to the underlying metric function from MoSQITo
        """
        assert metric in [
            "loudness_zwtv",
            "sharpness_din_from_loudness",
            "sharpness_din_perseg",
            "sharpness_din_tv",
            "roughness_dw",
        ], f"Metric {metric} not found."
        run, channel, statistics, label, func_args = self._parse_method(
            "MoSQITo", metric
        )
        try:
            parallel = self["MoSQITo"][metric]["parallel"]
        except KeyError:
            parallel = False
        # Check for sub metric
        # if sub metric is present, don't run this metric
        if (
            metric == "loudness_zwtv"
            and "sharpness_din_from_loudness" in self["MoSQITo"].keys()
            and self["MoSQITo"]["sharpness_din_from_loudness"]["run"]
            and self.force_run_all is False
        ):
            run = False
        return run, channel, statistics, label, parallel, func_args

    def _parse_method(self, library: str, metric: str):
        """Helper function to return relevant settings for a library

        Parameters
        ----------
        library : str
            Library containing the metric
        metric : str
            metric to prepare for

        Returns
        -------
        run: bool
            Whether to run the metric
        channel: tuple or list of str, or str
            channel(s) to run the metric on
        statistics: tuple or list of str, or str
            statistics to run the metric on.
            If run_stats is False, will only return the main statistic
        label: str
            label to use for the metric
        func_args: dict
            arguments to pass to the underlying metric function from the library
        """
        lib_settings = self[library].copy()
        channel = lib_settings[metric]["channel"].copy()
        statistics = lib_settings[metric]["statistics"].copy() if self.run_stats else []
        label = lib_settings[metric]["label"]
        func_args = (
            lib_settings[metric]["func_args"]
            if "func_args" in lib_settings[metric].keys()
            else {}
        )

        main_stat = lib_settings[metric]["main"]
        statistics = (
            (list(statistics) + [main_stat])
            if main_stat not in statistics
            else tuple(statistics)
        )

        # Override metric run settings if force_run_all is True
        run = lib_settings[metric]["run"] or self.force_run_all
        return run, channel, statistics, label, func_args
