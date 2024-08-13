"""
soundscapy.audio.parallel_processing
====================================

This module provides functions for parallel processing of binaural audio files.

It includes functions to load and analyze binaural files, as well as to process
multiple files in parallel using multiprocessing.

Functions:
    load_analyse_binaural: Load and analyze a single binaural file.
    parallel_process: Process multiple binaural files in parallel.

Note:
    This module requires the tqdm library for progress bars.
"""

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from soundscapy import AnalysisSettings, Binaural
from soundscapy.audio.metrics import (
    add_results,
    prep_multiindex_df,
    process_all_metrics,
)


def load_analyse_binaural(
    wav_file: Path,
    levels: Dict,
    analysis_settings: AnalysisSettings,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and analyze a binaural audio file.

    Args:
        wav_file (Path): Path to the WAV file.
        levels (Dict): Dictionary with calibration levels for each channel.
        analysis_settings (AnalysisSettings): Analysis settings object.
        verbose (bool, optional): Print progress information. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with analysis results.
    """
    print(f"Processing {wav_file.stem}")
    b = Binaural.from_wav(wav_file)
    decibel = (levels[b.recording]["Left"], levels[b.recording]["Right"])
    b.calibrate_to(decibel, inplace=True)
    return process_all_metrics(b, analysis_settings, parallel=False, verbose=verbose)


def parallel_process(
    wav_files: List[Path],
    results_df: pd.DataFrame,
    levels: Dict,
    analysis_settings: AnalysisSettings,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process multiple binaural files in parallel.

    Args:
        wav_files (List[Path]): List of WAV files to process.
        results_df (pd.DataFrame): Initial results DataFrame to update.
        levels (Dict): Dictionary with calibration levels for each file.
        analysis_settings (AnalysisSettings): Analysis settings object.
        verbose (bool, optional): Print progress information. Defaults to True.

    Returns:
        pd.DataFrame: Updated results DataFrame with analysis results for all files.
    """
    # Parallel processing with Pool.apply_async() without callback function

    pool = mp.Pool(mp.cpu_count() - 1)
    results = []
    result_objects = [
        pool.apply_async(
            load_analyse_binaural,
            args=(wav_file, levels, analysis_settings, verbose),
        )
        for wav_file in wav_files
    ]
    with tqdm(total=len(result_objects), desc="Processing files") as pbar:
        for r in result_objects:
            r.wait()
            results.append(r.get())
            pbar.update()
    # results = [r.get() for r in result_objects]

    pool.close()
    pool.join()

    for r in results:
        results_df = add_results(results_df, r)

    return results_df


if __name__ == "__main__":
    from datetime import datetime

    base_path = Path()
    wav_folder = base_path.joinpath("test", "data")
    levels = wav_folder.joinpath("Levels.json")
    with open(levels) as f:
        levels = json.load(f)

    analysis_settings = AnalysisSettings.default()

    df = prep_multiindex_df(levels, incl_metric=True)

    start = time.perf_counter()

    df = parallel_process(
        wav_folder.glob("*.wav"), df, levels, analysis_settings, verbose=True
    )

    end = time.perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")

    df.to_excel(base_path.joinpath("test", f"ParallelTest_{datetime.today()}.xlsx"))
