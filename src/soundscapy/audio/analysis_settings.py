"""
soundscapy.audio.analysis_settings
==================================

This module provides the AnalysisSettings class for managing and parsing
settings for various audio analysis methods, with a focus on reproducibility
and shareability in scientific analysis.

Classes
-------
MetricSettings : Dataclass for individual metric settings.
AnalysisSettings : Main class for loading, validating, and accessing analysis settings.

Functions
---------
get_default_yaml : Retrieves default settings from the GitHub repository.
"""

from __future__ import annotations

import dataclasses
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from schema import And, Optional, Or, Schema, Use
from soundscapy.logging import get_logger

logger = get_logger()


@dataclass
class MetricSettings:
    """
    Dataclass representing settings for an individual metric.

    Parameters
    ----------
    run : bool
        Whether to run this metric.
    main : str or int, optional
        The main statistic to calculate.
    statistics : List[str], optional
        List of statistics to calculate.
    channel : List[str], optional
        List of channels to analyze. Default is ["Left", "Right"].
    label : str, optional
        Label for the metric. Default is "label".
    parallel : bool, optional
        Whether to run the metric in parallel. Default is False.
    func_args : Dict[str, Any], optional
        Additional arguments for the metric function.
    """

    run: bool
    main: Union[str, int] = None
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

    Attributes
    ----------
    run_stats : bool
        Whether to include all stats or just the main metric.
    force_run_all : bool
        Whether to force all metrics to run regardless of their settings.
    filepath : Path
        Path to the YAML file containing the settings.
    settings : Dict[str, Dict[str, MetricSettings]]
        The loaded and validated settings.
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
        config: Union[Dict[str, Any], Path, str],
        run_stats: bool = True,
        force_run_all: bool = False,
    ):
        """
        Initialize the AnalysisSettings object.

        Parameters
        ----------
        config : Dict[str, Any] or Path or str
            Either a dictionary containing the configuration,
            or a path to the YAML configuration file.
        run_stats : bool, optional
            Whether to include all stats or just the main metric. Default is True.
        force_run_all : bool, optional
            Whether to force all metrics to run. Default is False.

        Raises
        ------
        FileNotFoundError
            If a file path is provided and the file doesn't exist.
        ValueError
            If the configuration is invalid.
        """
        self.run_stats = run_stats
        self.force_run_all = force_run_all
        self.settings: Dict[str, Dict[str, MetricSettings]] = {}

        if isinstance(config, (str, Path)):
            self.filepath = Path(config)
            if not self.filepath.exists():
                logger.error(f"Configuration file not found: {self.filepath}")
                raise FileNotFoundError(
                    f"Configuration file not found: {self.filepath}"
                )
            with open(self.filepath, "r") as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Loaded configuration from file: {self.filepath}")
        else:
            self.filepath = None
            config_dict = config
            logger.info("Loaded configuration from dictionary")

        self._load_and_validate_settings(config_dict)

    def _load_and_validate_settings(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration and load it into the settings attribute.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary to validate and load.

        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        try:
            validated_config = self.CONFIG_SCHEMA.validate(config)
            logger.debug("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Invalid configuration: {str(e)}")
            raise ValueError(f"Invalid configuration: {str(e)}")

        self.settings = {
            library: {
                metric: MetricSettings(**settings)
                for metric, settings in metrics.items()
            }
            for library, metrics in validated_config.items()
            if library != "version"
        }
        logger.info("Settings loaded and validated")

    def get_enabled_metrics(self) -> Dict[str, Dict[str, MetricSettings]]:
        """
        Get a dictionary of enabled metrics.

        Returns
        -------
        Dict[str, Dict[str, MetricSettings]]
            A dictionary of enabled metrics grouped by library.
        """
        enabled_metrics = {}
        for library, metrics in self.settings.items():
            enabled_metrics[library] = {
                metric: settings for metric, settings in metrics.items() if settings.run
            }
        logger.debug(f"Enabled metrics: {enabled_metrics}")
        return enabled_metrics

    def get_metric_settings(self, library: str, metric: str) -> MetricSettings:
        """
        Get the settings for a specific metric.

        Parameters
        ----------
        library : str
            The name of the library (e.g., 'PythonAcoustics', 'MoSQITo').
        metric : str
            The name of the metric.

        Returns
        -------
        MetricSettings
            The settings for the specified metric.

        Raises
        ------
        KeyError
            If the specified library or metric is not found in the settings.
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
            logger.error(f"Metric '{metric}' not found in library '{library}'")
            raise KeyError(f"Metric '{metric}' not found in library '{library}'")

    def parse_pyacoustics(self, metric: str) -> MetricSettings:
        """
        Parse settings for a Python Acoustics metric.

        Parameters
        ----------
        metric : str
            The name of the metric.

        Returns
        -------
        MetricSettings
            The settings for the specified metric.
        """
        return self.get_metric_settings("PythonAcoustics", metric)

    def parse_mosqito(self, metric: str) -> MetricSettings:
        """
        Parse settings for a MoSQITo metric.

        Parameters
        ----------
        metric : str
            The name of the metric.

        Returns
        -------
        MetricSettings
            The settings for the specified metric.

        Notes
        -----
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

        Parameters
        ----------
        metric : str
            The name of the metric.

        Returns
        -------
        MetricSettings
            The settings for the specified metric.

        Raises
        ------
        ValueError
            If the metric is not a valid MAAD alpha index.
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

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing the configuration.
        run_stats : bool, optional
            Whether to include all stats or just the main metric. Default is True.
        force_run_all : bool, optional
            Whether to force all metrics to run. Default is False.

        Returns
        -------
        AnalysisSettings
            An instance of AnalysisSettings.
        """
        return cls(config, run_stats, force_run_all)

    @classmethod
    def from_yaml(
        cls,
        filename: Union[Path, str],
        run_stats: bool = True,
        force_run_all: bool = False,
    ) -> AnalysisSettings:
        """
        Create an AnalysisSettings object from a YAML file.

        Parameters
        ----------
        filename : Path or str
            Path to the YAML configuration file.
        run_stats : bool, optional
            Whether to include all stats or just the main metric. Default is True.
        force_run_all : bool, optional
            Whether to force all metrics to run. Default is False.

        Returns
        -------
        AnalysisSettings
            An instance of AnalysisSettings.
        """
        return cls(filename, run_stats, force_run_all)

    @classmethod
    def default(
        cls, run_stats: bool = True, force_run_all: bool = False
    ) -> AnalysisSettings:
        """
        Create a default AnalysisSettings object.

        This method uses the default settings file located in the soundscapy package.

        Parameters
        ----------
        run_stats : bool, optional
            Whether to include all stats or just the main metric. Default is True.
        force_run_all : bool, optional
            Whether to force all metrics to run. Default is False.

        Returns
        -------
        AnalysisSettings
            An instance of AnalysisSettings with default settings.
        """
        import soundscapy

        root = Path(soundscapy.__path__[0])
        default_settings_path = root / "audio" / "default_settings.yaml"
        logger.info(f"Loading default settings from {default_settings_path}")
        return cls.from_yaml(default_settings_path, run_stats, force_run_all)

    def to_yaml(self, filename: Union[Path, str]) -> None:
        """
        Save the current settings to a YAML file.

        Parameters
        ----------
        filename : Path or str
            Path to save the YAML file.
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
        logger.info(f"Settings saved to {filename}")


def get_default_yaml(save_as: str = "default_settings.yaml") -> None:
    """
    Retrieve the default settings from GitHub and save them to a file.

    Parameters
    ----------
    save_as : str, optional
        Filename to save the default settings. Default is "default_settings.yaml".
    """
    print("Downloading default settings from GitHub...")
    url = os.getenv(
        "SOUNDSCAPY_DEFAULT_SETTINGS_URL",
        "https://raw.githubusercontent.com/MitchellAcoustics/Soundscapy/main/soundscapy/analysis/default_settings.yaml",
    )
    urllib.request.urlretrieve(url, save_as)
    logger.info(f"Default settings downloaded and saved to {save_as}")


if __name__ == "__main__":
    # Example usage
    # From YAML file
    settings_from_yaml = AnalysisSettings.from_yaml(
        "/Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/audio/default_settings.yaml"
    )

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
