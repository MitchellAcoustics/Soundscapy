"""
Audio analysis module for psychoacoustic analysis of audio files.

This module provides functionality for analyzing audio files using psychoacoustic
metrics. It includes the AudioAnalysis class for processing single files or entire
folders.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from soundscapy._utils import ensure_input_path, ensure_path_type
from soundscapy.audio.analysis_settings import ConfigManager
from soundscapy.audio.parallel_processing import load_analyse_binaural


class AudioAnalysis:
    """
    A class for performing psychoacoustic analysis on audio files.

    This class provides methods to analyze single audio files or entire folders
    of audio files using parallel processing. It handles configuration management,
    calibration, and saving of analysis results.

    Attributes
    ----------
    config_manager : ConfigManager
        Manages the configuration settings for audio analysis
    settings : dict
        The current configuration settings

    Methods
    -------
    analyze_file(file_path, calibration_levels, resample)
        Analyze a single audio file
    analyze_folder(folder_path, calibration_file, max_workers, resample)
        Analyze all audio files in a folder using parallel processing
    save_results(results, output_path)
        Save analysis results to a file
    update_config(new_config)
        Update the current configuration
    save_config(config_path)
        Save the current configuration to a file

    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """
        Initialize the AudioAnalysis with a configuration.

        Parameters
        ----------
        config_path : str, Path, or None
            Path to the configuration file. If None, uses default configuration.

        """
        self.config_manager = ConfigManager(config_path)
        self.settings = self.config_manager.load_config()
        logger.info(
            f"Psychoacoustic analysis initialized with configuration: {config_path}"
        )

    def analyze_file(
        self,
        file_path: str | Path,
        calibration_levels: dict[str, float] | list[float] | None = None,
        resample: int | None = None,
    ) -> pd.DataFrame:
        """
        Analyze a single audio file using the current configuration.

        Parameters
        ----------
        resample
        file_path : str or Path
            Path to the audio file to analyze.
        calibration_levels : dict, optional
            Dictionary containing calibration levels for left and right channels.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the analysis results.

        """
        file_path = ensure_input_path(file_path)

        logger.info(f"Analyzing file: {file_path}")
        return load_analyse_binaural(
            file_path,
            calibration_levels,
            self.settings,
            resample=resample,
        )

    @logger.catch
    def analyze_folder(
        self,
        folder_path: str | Path,
        calibration_file: str | Path | None = None,
        max_workers: int | None = None,
        resample: int | None = None,
    ) -> pd.DataFrame:
        """
        Analyze all audio files in a folder using parallel processing.

        Parameters
        ----------
        resample
        folder_path : str or Path
            Path to the folder containing audio files.
        calibration_file : str or Path, optional
            Path to a JSON file containing calibration levels for each audio file.
        max_workers : int, optional
            Maximum number of worker processes to use.
            If None, it will use the number of CPU cores.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the analysis results for all files.

        """
        folder_path = ensure_input_path(folder_path)
        audio_files = list(folder_path.glob("*.wav"))

        logger.info(
            f"Analyzing folder: {folder_path.name} of {len(audio_files)}"
            f"files in parallel (max_workers={max_workers})"
        ) if max_workers else logger.info(
            f"Analyzing folder: {folder_path}, {len(audio_files)} files"
        )

        calibration_levels = {}
        if calibration_file:
            calibration_file = ensure_input_path(calibration_file)
            with calibration_file.open() as f:
                calibration_levels = json.load(f)
            logger.debug(f"Loaded calibration levels from: {calibration_file}")

        all_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in audio_files:
                future = executor.submit(
                    load_analyse_binaural,
                    file,
                    calibration_levels,
                    self.settings,
                    resample,
                    parallel_mosqito=False,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Analyzing files"
            ):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:  # noqa: BLE001, PERF203
                    logger.error(f"Error processing file: {e!s}")

        combined_results = pd.concat(all_results)
        logger.info(
            f"Completed analysis for {len(audio_files)} files in folder: {folder_path}"
        )
        return combined_results

    def save_results(self, results: pd.DataFrame, output_path: str | Path) -> None:
        """
        Save analysis results to a file.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing the analysis results.
        output_path : str or Path
            Path to save the results file.

        """
        output_path = ensure_path_type(
            output_path
        )  # If doesn't already exist, pandas will create the file.
        if output_path.suffix == ".csv":
            results.to_csv(output_path)
        elif output_path.suffix == ".xlsx":
            results.to_excel(output_path)
        else:
            msg = "Unsupported file format. Use .csv or .xlsx"
            raise ValueError(msg)
        logger.info(f"Results saved to: {output_path}")

    def update_config(self, new_config: dict) -> "AudioAnalysis":
        """
        Update the current configuration.

        Parameters
        ----------
        new_config : dict
            Dictionary containing the new configuration settings.

        """
        self.settings = self.config_manager.merge_configs(new_config)
        logger.info("Configuration updated")
        return self

    def save_config(self, config_path: str | Path) -> None:
        """
        Save the current configuration to a file.

        Parameters
        ----------
        config_path : str or Path
            Path to save the configuration file.

        """
        self.config_manager.save_config(config_path)
        logger.info(f"Configuration saved to: {config_path}")


# Example usage
if __name__ == "__main__":
    from soundscapy.sspylogging import setup_logging

    setup_logging("INFO")

    # Initialize the analysis with default settings
    analysis = AudioAnalysis()

    # Analyze a single file
    single_file_result = analysis.analyze_file(
        "/Users/mitch/Documents/GitHub/Soundscapy/test/data/CT101.wav",
        calibration_levels=[79.0, 79.72],
    )
    analysis.save_results(single_file_result, "single_file_results.csv")

    # Analyze a folder of files in parallel
    folder_results = analysis.analyze_folder(
        "/Users/mitch/Documents/GitHub/Soundscapy/test/data",
        "/Users/mitch/Documents/GitHub/Soundscapy/test/data/Levels.json",
    )
    analysis.save_results(folder_results, "folder_results.xlsx")

    # Update configuration
    new_config = {"AcousticToolbox": {"LAeq": {"run": False}}}
    analysis.update_config(new_config)

    # Save updated configuration
    analysis.save_config("updated_config.yaml")
