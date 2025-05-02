"""
Functions for parallel processing of binaural audio files.

It includes functions to load and analyze binaural files, as well as to process
multiple files in parallel using concurrent.futures.

Functions:
    load_analyse_binaural: Load and analyze a single binaural file.
    parallel_process: Process multiple binaural files in parallel.

Note:
    This module requires the tqdm library for progress bars and concurrent.futures
    for parallel processing. It uses loguru for logging.

"""

import concurrent.futures
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from .analysis_settings import AnalysisSettings
from .binaural import Binaural
from .metrics import (
    add_results,
    prep_multiindex_df,
    process_all_metrics,
)


def tqdm_write_sink(message: str) -> None:
    """
    Custom sink for loguru that writes messages using tqdm.write().

    This ensures that log messages don't interfere with tqdm progress bars.
    """  # noqa: D401
    tqdm.write(message, end="")


def load_analyse_binaural(
    wav_file: Path,
    levels: dict[str, float] | list[float] | None,
    analysis_settings: AnalysisSettings,
    resample: int | None = None,
    *,
    parallel_mosqito: bool = True,
) -> pd.DataFrame:
    """
    Load and analyze a single binaural audio file.

    Parameters
    ----------
    resample
    wav_file : Path
        Path to the WAV file.
    levels : Dict
        Dictionary with calibration levels for each channel.
    analysis_settings : AnalysisSettings
        Analysis settings object.
    parallel_mosqito : bool, optional
        Whether to process MoSQITo metrics in parallel. Defaults to True.

    Returns
    -------
    pd.DataFrame
        DataFrame with analysis results.

    """
    logger.info(f"Processing {wav_file}")
    try:
        b = Binaural.from_wav(wav_file, resample=resample)
        if levels is not None:
            if isinstance(levels, dict) and b.recording in levels:
                decibel = (levels[b.recording]["Left"], levels[b.recording]["Right"])
                b.calibrate_to(decibel, inplace=True)
            elif isinstance(levels, list | tuple):
                logger.debug(f"Calibrating {wav_file} to {levels} dB")
                b.calibrate_to(levels, inplace=True)
            else:
                logger.warning(f"No calibration levels found for {wav_file}")
        else:
            logger.warning(f"No calibration levels found for {wav_file}")
        return process_all_metrics(b, analysis_settings, parallel=parallel_mosqito)
    except Exception as e:
        logger.error(f"Error processing {wav_file}: {e!s}")
        raise


def parallel_process(
    wav_files: list[Path],
    results_df: pd.DataFrame,
    levels: dict,
    analysis_settings: AnalysisSettings,
    max_workers: int | None = None,
    resample: int | None = None,
    *,
    parallel_mosqito: bool = True,
) -> pd.DataFrame:
    """
    Process multiple binaural files in parallel.

    Parameters
    ----------
    resample
    wav_files : List[Path]
        List of WAV files to process.
    results_df : pd.DataFrame
        Initial results DataFrame to update.
    levels : Dict
        Dictionary with calibration levels for each file.
    analysis_settings : AnalysisSettings
        Analysis settings object.
    max_workers : int, optional
        Maximum number of worker processes. If None, it will default to the number of processors on the machine.
    parallel_mosqito : bool, optional
        Whether to process MoSQITo metrics in parallel within each file. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Updated results DataFrame with analysis results for all files.

    """
    logger.info(f"Starting parallel processing of {len(wav_files)} files")

    # Add a handler that uses tqdm.write for output
    tqdm_handler_id = logger.add(tqdm_write_sink, format="{message}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for wav_file in wav_files:
            future = executor.submit(
                load_analyse_binaural,
                wav_file,
                levels,
                analysis_settings,
                resample,
                parallel_mosqito=parallel_mosqito,
            )
            futures.append(future)

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results_df = add_results(results_df, result)
                except Exception as e:
                    logger.error(f"Error processing file: {e!s}")
                finally:
                    pbar.update(1)

    # Remove the tqdm-compatible handler
    logger.remove(tqdm_handler_id)

    logger.info("Parallel processing completed")
    return results_df


if __name__ == "__main__":
    # Example usage
    import json
    import warnings
    from datetime import datetime

    warnings.filterwarnings("ignore")

    from soundscapy.sspylogging import setup_logging

    setup_logging("DEBUG")

    base_path = Path().absolute().parent.parent.parent
    wav_folder = base_path.joinpath("test", "data")
    levels_file = wav_folder.joinpath("Levels.json")

    with open(levels_file) as f:
        levels = json.load(f)

    analysis_settings = AnalysisSettings.default()

    df = prep_multiindex_df(levels, incl_metric=True)

    wav_files = list(wav_folder.glob("*.wav"))

    df = parallel_process(wav_files[:4], df, levels, analysis_settings)

    output_file = base_path.joinpath(
        "test", f"ParallelTest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    df.to_excel(output_file)
    print(f"Results saved to {output_file}")
