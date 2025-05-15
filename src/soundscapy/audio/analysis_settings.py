"""
Module for managing audio analysis settings using Pydantic models.

This module defines Pydantic models for configuring analysis settings for different
audio processing libraries (AcousticToolbox, MoSQITo, scikit-maad).
It includes classes for individual metric settings, library settings, and overall
analysis settings. It also provides a ConfigManager class for loading, saving,
merging, and managing configurations from YAML files or dictionaries.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)


def _ensure_path(value: str | Path) -> Path:
    """Ensure the value is a Path object."""
    if isinstance(value, str):
        return Path(value)
    return value


class MetricSettings(BaseModel):
    """
    Settings for an individual metric.

    Parameters
    ----------
    run : bool
        Whether to run this metric.
    main : str | int | None
        The main statistic to calculate.
    statistics : list[str | int] | None
        List of statistics to calculate.
    channel : list[str]
        List of channels to analyze.
    label : str
        Label for the metric.
    parallel : bool
        Whether to run the metric in parallel.
    func_args : dict[str, Any]
        Additional arguments for the metric function.

    """

    run: bool = True
    main: str | int | None = None
    statistics: list[str | int] | None = None
    channel: list[str] = Field(default_factory=lambda: ["Left", "Right"])
    label: str = "label"
    parallel: bool = False
    func_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def check_main_in_statistics(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check that the main statistic is in the statistics list."""
        main = values.get("main")
        statistics = values.get("statistics", [])
        if main and main not in statistics:
            statistics.append(main)
            values["statistics"] = statistics
        return values


class LibrarySettings(RootModel):
    """Settings for a library of metrics."""

    root: dict[str, MetricSettings]

    def get_metric_settings(self, metric: str) -> MetricSettings:
        """
        Get the settings for a specific metric.

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
        KeyError
            If the specified metric is not found.

        """
        if metric in self.root:
            return self.root[metric]
        logger.error(f"Metric '{metric}' not found in library")
        msg = f"Metric '{metric}' not found in library"
        raise KeyError(msg)


class AnalysisSettings(BaseModel):
    """
    Settings for audio analysis methods.

    Parameters
    ----------
    version : str
        Version of the configuration.
    AcousticToolbox : LibrarySettings | None
        Settings for AcousticToolbox metrics.
    MoSQITo : LibrarySettings | None
        Settings for MoSQITo metrics.
    scikit_maad : LibrarySettings | None
        Settings for scikit-maad metrics.

    """

    version: str = "1.0"
    AcousticToolbox: LibrarySettings | None = Field(
        None, validation_alias=AliasChoices("AcousticToolbox", "PythonAcoustics")
    )
    MoSQITo: LibrarySettings | None = None
    scikit_maad: LibrarySettings | None = Field(
        None, validation_alias=AliasChoices("scikit-maad", "scikit_maad")
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def validate_library_settings(cls, v: dict | LibrarySettings) -> LibrarySettings:
        """Validate library settings."""
        if isinstance(v, dict):
            return LibrarySettings(root=v)
        return v

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> AnalysisSettings:
        """
        Create an AnalysisSettings object from a YAML file.

        Parameters
        ----------
        filepath : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        AnalysisSettings
            An instance of AnalysisSettings.

        """
        filepath = _ensure_path(filepath)
        logger.info(f"Loading configuration from {filepath}")
        with Path.open(filepath) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def default(cls) -> AnalysisSettings:
        """
        Create a default AnalysisSettings using the package default configuration file.

        Returns
        -------
        AnalysisSettings
            An instance of AnalysisSettings with default settings.

        """
        config_resource = resources.files("soundscapy.data").joinpath(
            "default_settings.yaml"
        )
        with resources.as_file(config_resource) as f:
            logger.info(f"Loading default configuration from {f}")
            return cls.from_yaml(f)

    @classmethod
    def from_dict(cls, d: dict) -> AnalysisSettings:
        """
        Create an AnalysisSettings object from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing the configuration settings.

        Returns
        -------
        AnalysisSettings
            An instance of AnalysisSettings.

        """
        return cls(**d)

    def to_yaml(self, filepath: str | Path) -> None:
        """
        Save the current settings to a YAML file.

        Parameters
        ----------
        filepath : str | Path
            Path to save the YAML file.

        """
        filepath = _ensure_path(filepath)
        logger.info(f"Saving configuration to {filepath}")
        with Path.open(filepath, "w") as f:
            yaml.dump(self.model_dump(by_alias=True), f)

    def update_setting(self, library: str, metric: str, **kwargs: dict) -> None:
        """
        Update the settings for a specific metric.

        Parameters
        ----------
        library : str
            The name of the library.
        metric : str
            The name of the metric.
        **kwargs
            Keyword arguments to update the metric settings.

        Raises
        ------
        KeyError
            If the specified library or metric is not found.

        """
        library_settings = getattr(self, library)
        if library_settings and metric in library_settings.root:
            metric_settings = library_settings.root[metric]
            for key, value in kwargs.items():
                if hasattr(metric_settings, key):
                    setattr(metric_settings, key, value)
                else:
                    logger.error(f"Invalid setting '{key}' for metric '{metric}'")
        else:
            logger.error(f"Metric '{metric}' not found in library '{library}'")
            msg = f"Metric '{metric}' not found in library '{library}'"
            raise KeyError(msg)

    def get_metric_settings(self, library: str, metric: str) -> MetricSettings:
        """
        Get the settings for a specific metric.

        Parameters
        ----------
        library : str
            The name of the library.
        metric : str
            The name of the metric.

        Returns
        -------
        MetricSettings
            The settings for the specified metric.

        Raises
        ------
        KeyError
            If the specified library or metric is not found.

        """
        library_settings = getattr(self, library)
        if library_settings and metric in library_settings.root:
            return library_settings.root[metric]
        logger.error(f"Metric '{metric}' not found in library '{library}'")
        msg = f"Metric '{metric}' not found in library '{library}'"
        raise KeyError(msg)

    def get_enabled_metrics(self) -> dict[str, dict[str, MetricSettings]]:
        """
        Get a dictionary of enabled metrics.

        Returns
        -------
        dict[str, dict[str, MetricSettings]]
            A dictionary of enabled metrics grouped by library.

        """
        enabled_metrics = {}
        for library in ["AcousticToolbox", "MoSQITo", "scikit_maad"]:
            library_settings = getattr(self, library)
            if library_settings:
                enabled_metrics[library] = {
                    metric: settings
                    for metric, settings in library_settings.root.items()
                    if settings.run
                }
        logger.debug(f"Enabled metrics: {enabled_metrics}")
        return enabled_metrics


class ConfigManager:
    """
    Manage configuration settings for audio analysis.

    Parameters
    ----------
    default_config_path : str | Path | None
        Path to the default configuration file.

    """

    def __init__(self, config_path: str | Path | None = None) -> None:  # noqa: D107
        self.config_path = _ensure_path(config_path) if config_path else None
        self.current_config: AnalysisSettings | None = None

    def load_config(self, config_path: str | Path | None = None) -> AnalysisSettings:
        """
        Load a configuration file or use the default configuration.

        Parameters
        ----------
        config_path : str | Path | None, optional
            Path to the configuration file. If None, uses the default configuration.

        Returns
        -------
        AnalysisSettings
            The loaded configuration.

        """
        if config_path:
            logger.info(f"Loading configuration from {config_path}")
            self.current_config = AnalysisSettings.from_yaml(config_path)
        elif self.config_path:
            logger.info(f"Loading configuration from {self.config_path}")
            self.current_config = AnalysisSettings.from_yaml(self.config_path)
        else:
            logger.info("Loading default configuration")
            self.current_config = AnalysisSettings.default()
        return self.current_config

    def save_config(self, filepath: str | Path) -> None:
        """
        Save the current configuration to a file.

        Parameters
        ----------
        filepath : str | Path
            Path to save the configuration file.

        Raises
        ------
        ValueError
            If no current configuration is loaded.

        """
        if self.current_config:
            logger.info(f"Saving configuration to {filepath}")
            self.current_config.to_yaml(filepath)
        else:
            logger.error("No current configuration to save")
            msg = "No current configuration to save."
            raise ValueError(msg)

    def merge_configs(self, override_config: dict) -> AnalysisSettings:
        """
        Merge the current config with override values and update the current_config.

        Parameters
        ----------
        override_config : dict
            Dictionary containing override configuration values.

        Returns
        -------
        AnalysisSettings
            The merged configuration.

        Raises
        ------
        ValueError
            If no base configuration is loaded.

        """
        if not self.current_config:
            logger.error("No base configuration loaded")
            msg = "No base configuration loaded."
            raise ValueError(msg)
        logger.info("Merging configurations")
        merged_dict = self.current_config.model_dump()
        self._deep_update(merged_dict, override_config)
        merged_config = AnalysisSettings(**merged_dict)
        self.current_config = merged_config  # Update the current_config
        return merged_config

    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """Recursively update a nested dictionary."""
        for key, value in update_dict.items():
            if (
                isinstance(value, dict)
                and key in base_dict
                and isinstance(base_dict[key], dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def generate_minimal_config(self) -> dict:
        """
        Generate a minimal configuration containing only changes from the default.

        Returns
        -------
        dict
            A dictionary containing the minimal configuration.

        Raises
        ------
        ValueError
            If no current configuration is loaded.

        """
        if not self.current_config:
            msg = "No current configuration loaded."
            raise ValueError(msg)
        default_config = AnalysisSettings.default()
        current_dict = self.current_config.model_dump()
        default_dict = default_config.model_dump()
        return self._get_diff(current_dict, default_dict)

    def _get_diff(self, current: dict, default: dict) -> dict:
        """Recursively find differences between two dictionaries."""
        diff = {}
        for key, value in current.items():
            if key not in default:
                diff[key] = value
            elif isinstance(value, dict) and isinstance(default[key], dict):
                nested_diff = self._get_diff(value, default[key])
                if nested_diff:
                    diff[key] = nested_diff
            elif value != default[key]:
                diff[key] = value
        return diff


if __name__ == "__main__":
    # Example usage
    logger.info("Starting configuration management example")

    # Initialize ConfigManager without specifying a config path
    config_manager = ConfigManager()

    # Load the configuration (will use default if no path specified)
    config = config_manager.load_config()

    # Print the loaded configuration
    logger.info(f"Loaded configuration: {config.model_dump()}")

    # Modify some settings
    override_config = {"AcousticToolbox": {"LAeq": {"run": False}}}

    # Merge the configurations
    merged_config = config_manager.merge_configs(override_config)
    logger.info(f"Merged configuration: {merged_config.model_dump()}")

    # Generate a minimal configuration
    minimal_config = config_manager.generate_minimal_config()
    logger.info(f"Minimal configuration: {minimal_config}")

    # Save the configuration
    config_manager.save_config("new_config.yaml")

    logger.info("Configuration management example completed")

    # Create a new config:
    # Create MetricSettings for AcousticToolbox
    laeq_settings = MetricSettings(
        run=True,
        main="avg",
        statistics=[5, 10, 50, 90, 95, "min", "max", "kurt", "skew"],
        channel=["Left", "Right"],
        label="LAeq",
        func_args={"time": 0.125, "method": "average"},
    )

    lzeq_settings = MetricSettings(
        run=True,
        main="avg",
        statistics=[5, 10, 50, 90, 95, "min", "max", "kurt", "skew"],
        channel=["Left", "Right"],
        label="LZeq",
        func_args={"time": 0.125, "method": "average"},
    )

    # Create LibrarySettings for AcousticToolbox
    acoustic_toolbox_settings = LibrarySettings(
        root={"LAeq": laeq_settings, "LZeq": lzeq_settings}
    )

    # Create MetricSettings for MoSQITo
    loudness_settings = MetricSettings(
        run=True,
        main=5,
        statistics=[10, 50, 90, 95, "min", "max", "kurt", "skew", "avg"],
        channel=["Left", "Right"],
        label="N",
        parallel=True,
        func_args={"field_type": "free"},
    )

    # Create LibrarySettings for MoSQITo
    mosqito_settings = LibrarySettings(root={"loudness_zwtv": loudness_settings})

    # Create MetricSettings for scikit-maad
    temporal_indices_settings = MetricSettings(run=True, channel=["Left", "Right"])

    # Create LibrarySettings for scikit-maad
    scikit_maad_settings = LibrarySettings(
        root={"all_temporal_alpha_indices": temporal_indices_settings}
    )

    # Create the AnalysisSettings object
    analysis_settings = AnalysisSettings(
        version="1.0",
        AcousticToolbox=acoustic_toolbox_settings,
        MoSQITo=mosqito_settings,
        scikit_maad=scikit_maad_settings,
    )

    # Print the created configuration
    print(analysis_settings.model_dump_json(indent=2))  # noqa: T201

    # Save the configuration to a YAML file
    output_path = Path("my_custom_config.yaml")
    analysis_settings.to_yaml(output_path)
    print(f"Configuration saved to {output_path}")  # noqa: T201

    # To use this configuration:

    config_manager = ConfigManager("my_custom_config.yaml")
    loaded_config = config_manager.load_config()
