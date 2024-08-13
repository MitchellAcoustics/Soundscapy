"""
soundscapy.audio.analysis_settings
==================================

This module provides the AnalysisSettings class for managing and parsing
settings for various audio analysis methods, with a focus on reproducibility
and shareability in scientific analysis.

Classes:
    MetricSettings: Dataclass for individual metric settings.
    AnalysisSettings: Main class for loading, validating, and accessing analysis settings.

Functions:
    get_default_yaml: Retrieves default settings from the GitHub repository.
"""

from __future__ import annotations

import dataclasses
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from schema import And, Optional, Or, Schema, Use


@dataclass
class MetricSettings:
    """
    Dataclass representing settings for an individual metric.

    Attributes:
        run (bool): Whether to run this metric.
        main (str | int): The main statistic to calculate.
        statistics (List[str]): List of statistics to calculate.
        channel (List[str]): List of channels to analyze.
        label (str): Label for the metric.
        parallel (bool): Whether to run the metric in parallel.
        func_args (Dict[str, Any]): Additional arguments for the metric function.
    """

    run: bool
    main: str | int = None
    statistics: List[str] = None
    channel: List[str] = field(default_factory=lambda: ["Left", "Right"])
    label: str = "label"
    parallel: bool = False
    func_args: Dict[str, Any] = field(default_factory=dict)


class AnalysisSettings:
    """
    A class for managing settings for audio analysis methods.

    This class handles loading, validating, and accessing configuration
    settings for audio analysis, with a focus on reproducibility in
    scientific analysis.

    Attributes:
        run_stats (bool): Whether to include all stats or just the main metric.
        force_run_all (bool): Whether to force all metrics to run regardless of their settings.
        filepath (Path): Path to the YAML file containing the settings.
        settings (Dict[str, Dict[str, MetricSettings]]): The loaded and validated settings.
    """

    CONFIG_SCHEMA = Schema(
        {
            Optional("version"): str,
            Optional("PythonAcoustics"): {
                str: {
                    "run": bool,
                    Optional("main"): Or(str, int),
                    Optional("statistics"): [
                        Or(
                            int,
                            And(
                                str,
                                Use(str.lower),
                                lambda s: s
                                in ["avg", "mean", "max", "min", "kurt", "skew"]
                                or s.isdigit(),
                            ),
                        )
                    ],
                    Optional("channel"): [str],
                    Optional("label"): str,
                    Optional("func_args"): dict,
                }
            },
            Optional("MoSQITo"): {
                str: {
                    "run": bool,
                    Optional("main"): Or(str, int),
                    Optional("statistics"): [
                        Or(
                            int,
                            And(
                                str,
                                Use(str.lower),
                                lambda s: s
                                in ["avg", "mean", "max", "min", "kurt", "skew"]
                                or s.isdigit(),
                            ),
                        )
                    ],
                    Optional("channel"): [str],
                    Optional("label"): str,
                    Optional("parallel"): bool,
                    Optional("func_args"): dict,
                }
            },
            Optional("scikit-maad"): {str: {"run": bool, Optional("channel"): [str]}},
        }
    )

    def __init__(
        self,
        config: Dict[str, Any] | Path | str,
        run_stats: bool = True,
        force_run_all: bool = False,
    ):
        """
        Initialize the AnalysisSettings object.

        Args:
            config (Dict[str, Any] | Path | str): Either a dictionary containing the configuration,
                                                  or a path to the YAML configuration file.
            run_stats (bool, optional): Whether to include all stats or just the main metric. Defaults to True.
            force_run_all (bool, optional): Whether to force all metrics to run. Defaults to False.

        Raises:
            FileNotFoundError: If a file path is provided and the file doesn't exist.
            ValueError: If the configuration is invalid.
        """
        self.run_stats = run_stats
        self.force_run_all = force_run_all
        self.settings: Dict[str, Dict[str, MetricSettings]] = {}

        if isinstance(config, (str, Path)):
            self.filepath = Path(config)
            if not self.filepath.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.filepath}"
                )
            with open(self.filepath, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            self.filepath = None
            config_dict = config

        self._load_and_validate_settings(config_dict)

    def _load_and_validate_settings(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration and load it into the settings attribute.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate and load.

        Raises:
            ValueError: If the configuration is invalid.
        """
        try:
            validated_config = self.CONFIG_SCHEMA.validate(config)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

        self.settings = {
            library: {
                metric: MetricSettings(**settings)
                for metric, settings in metrics.items()
            }
            for library, metrics in validated_config.items()
            if library != "version"
        }

    def get_enabled_metrics(self):
        enabled_metrics = {}
        for library, metrics in self.settings.items():
            enabled_metrics[library] = {
                metric: settings for metric, settings in metrics.items() if settings.run
            }
        return enabled_metrics

    def get_metric_settings(self, library: str, metric: str) -> MetricSettings:
        """
        Get the settings for a specific metric.

        Args:
            library (str): The name of the library (e.g., 'PythonAcoustics', 'MoSQITo').
            metric (str): The name of the metric.

        Returns:
            MetricSettings: The settings for the specified metric.

        Raises:
            KeyError: If the specified library or metric is not found in the settings.
        """
        try:
            metric_settings = self.settings[library][metric]
            run = metric_settings.run or self.force_run_all
            statistics = (
                metric_settings.statistics if self.run_stats else [metric_settings.main]
            )
            return MetricSettings(
                run=run,
                main=metric_settings.main,
                statistics=statistics,
                channel=metric_settings.channel,
                label=metric_settings.label,
                parallel=getattr(metric_settings, "parallel", False),
                func_args=metric_settings.func_args,
            )
        except KeyError:
            raise KeyError(f"Metric '{metric}' not found in library '{library}'")

    def parse_pyacoustics(self, metric: str) -> MetricSettings:
        """
        Parse settings for a Python Acoustics metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            MetricSettings: The settings for the specified metric.
        """
        return self.get_metric_settings("PythonAcoustics", metric)

    def parse_mosqito(self, metric: str) -> MetricSettings:
        """
        Parse settings for a MoSQITo metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            MetricSettings: The settings for the specified metric.

        Note:
            This method includes special handling for the 'loudness_zwtv' metric
            when 'sharpness_din_from_loudness' is also present.
        """
        settings = self.get_metric_settings("MoSQITo", metric)
        if (
            metric == "loudness_zwtv"
            and "sharpness_din_from_loudness" in self.settings["MoSQITo"]
            and self.settings["MoSQITo"]["sharpness_din_from_loudness"].run
            and not self.force_run_all
        ):
            settings.run = False
        return settings

    def parse_maad_all_alpha_indices(self, metric: str) -> MetricSettings:
        """
        Parse settings for MAAD alpha indices.

        Args:
            metric (str): The name of the metric.

        Returns:
            MetricSettings: The settings for the specified metric.

        Raises:
            ValueError: If the metric is not a valid MAAD alpha index.
        """
        if metric not in ["all_temporal_alpha_indices", "all_spectral_alpha_indices"]:
            raise ValueError(f"Invalid MAAD metric: {metric}")
        return self.get_metric_settings("scikit-maad", metric)

    @classmethod
    def from_dict(
        cls, config: Dict[str, Any], run_stats: bool = True, force_run_all: bool = False
    ) -> AnalysisSettings:
        """
        Create an AnalysisSettings object from a dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing the configuration.
            run_stats (bool, optional): Whether to include all stats or just the main metric. Defaults to True.
            force_run_all (bool, optional): Whether to force all metrics to run. Defaults to False.

        Returns:
            AnalysisSettings: An instance of AnalysisSettings.
        """
        return cls(config, run_stats, force_run_all)

    @classmethod
    def from_yaml(
        cls, filename: Path | str, run_stats: bool = True, force_run_all: bool = False
    ) -> AnalysisSettings:
        """
        Create an AnalysisSettings object from a YAML file.

        Args:
            filename (Path | str): Path to the YAML configuration file.
            run_stats (bool, optional): Whether to include all stats or just the main metric. Defaults to True.
            force_run_all (bool, optional): Whether to force all metrics to run. Defaults to False.

        Returns:
            AnalysisSettings: An instance of AnalysisSettings.
        """
        return cls(filename, run_stats, force_run_all)

    @classmethod
    def default(
        cls, run_stats: bool = True, force_run_all: bool = False
    ) -> AnalysisSettings:
        """
        Create a default AnalysisSettings object.

        This method uses the default settings file located in the soundscapy package.

        Args:
            run_stats (bool, optional): Whether to include all stats or just the main metric. Defaults to True.
            force_run_all (bool, optional): Whether to force all metrics to run. Defaults to False.

        Returns:
            AnalysisSettings: An instance of AnalysisSettings with default settings.
        """
        import soundscapy

        root = Path(soundscapy.__path__[0])
        return cls.from_yaml(
            root / "audio" / "default_settings.yaml", run_stats, force_run_all
        )

    def to_yaml(self, filename: Path | str) -> None:
        """
        Save the current settings to a YAML file.

        Args:
            filename (Path | str): Path to save the YAML file.
        """
        config = {
            library: {
                metric: dataclasses.asdict(settings)
                for metric, settings in metrics.items()
            }
            for library, metrics in self.settings.items()
        }
        config["version"] = self.settings.get(
            "version", "1.0"
        )  # Add version if it exists, else use default

        with open(filename, "w") as f:
            yaml.dump(config, f)


def get_default_yaml(save_as: str = "default_settings.yaml") -> None:
    """
    Retrieve the default settings from GitHub and save them to a file.

    Args:
        save_as (str, optional): Filename to save the default settings. Defaults to "default_settings.yaml".
    """
    print("Downloading default settings from GitHub...")
    url = os.getenv(
        "SOUNDSCAPY_DEFAULT_SETTINGS_URL",
        "https://raw.githubusercontent.com/MitchellAcoustics/Soundscapy/main/soundscapy/analysis/default_settings.yaml",
    )
    urllib.request.urlretrieve(url, save_as)


if __name__ == "__main__":
    # Example usage
    # From YAML file
    settings_from_yaml = AnalysisSettings.from_yaml("path/to/your/settings.yaml")

    # From dictionary
    config_dict = {
        "version": "1.0",
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": ["5", "95", "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
    }
    settings_from_dict = AnalysisSettings.from_dict(config_dict)

    print(settings_from_yaml.parse_pyacoustics("LAeq"))
    print(settings_from_dict.parse_pyacoustics("LAeq"))
