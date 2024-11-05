import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from soundscapy.audio.analysis_settings import ConfigManager
from soundscapy.audio.parallel_processing import load_analyse_binaural


class AudioAnalysis:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_manager = ConfigManager(config_path)
        self.settings = self.config_manager.load_config()
        logger.info(
            f"Psychoacoustic analysis initialized with configuration: {config_path}"
        )

    def analyze_file(
        self,
        file_path: str | Path,
        calibration_levels: Optional[Dict[str, float] | List[float]] = None,
        resample: Optional[int] = None,
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
        if isinstance(file_path, str):
            file_path = Path(file_path)
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
        calibration_file: Optional[str | Path] = None,
        max_workers: Optional[int] = None,
        resample: Optional[int] = None,
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
            Maximum number of worker processes to use. If None, it will use the number of CPU cores.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the analysis results for all files.
        """

        folder_path = Path(folder_path)
        audio_files = list(folder_path.glob("*.wav"))

        logger.info(
            f"Analyzing folder: {folder_path.name} of {len(audio_files)} files in parallel (max_workers={max_workers})"
        ) if max_workers else logger.info(
            f"Analyzing folder: {folder_path}, {len(audio_files)} files"
        )

        calibration_levels = {}
        if calibration_file:
            with open(calibration_file, "r") as f:
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
                    False,
                    resample,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Analyzing files"
            ):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")

        combined_results = pd.concat(all_results)
        logger.info(
            f"Completed analysis for {len(audio_files)} files in folder: {folder_path}"
        )
        return combined_results

    def save_results(self, results: pd.DataFrame, output_path: Union[str, Path]):
        """
        Save analysis results to a file.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing the analysis results.
        output_path : str or Path
            Path to save the results file.
        """
        output_path = Path(output_path)
        if output_path.suffix == ".csv":
            results.to_csv(output_path)
        elif output_path.suffix == ".xlsx":
            results.to_excel(output_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        logger.info(f"Results saved to: {output_path}")

    def update_config(self, new_config: Dict):
        """
        Update the current configuration.

        Parameters
        ----------
        new_config : dict
            Dictionary containing the new configuration settings.
        """
        self.settings = self.config_manager.merge_configs(new_config)
        logger.info("Configuration updated")

    def save_config(self, config_path: Union[str, Path]):
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
    from soundscapy.logging import setup_logging

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
    new_config = {"PythonAcoustics": {"LAeq": {"run": False}}}
    analysis.update_config(new_config)

    # Save updated configuration
    analysis.save_config("updated_config.yaml")
