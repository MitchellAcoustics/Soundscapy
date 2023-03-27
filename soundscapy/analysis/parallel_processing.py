import json
import time
from pathlib import Path
from tqdm.auto import tqdm

from soundscapy import AnalysisSettings
from soundscapy import Binaural
from soundscapy.analysis.binaural import *


def load_analyse_binaural(wav_file, levels, analysis_settings, verbose=True):
    """Load and analyse binaural file

    Parameters
    ----------
    wav_file : str
        Path to wav file
    levels : list
        List of levels to analyse
    analysis_settings : AnalysisSettings
        Analysis settings
    verbose : bool, optional
        Print progress, by default True

    Returns
    -------
    results : dict
        Dictionary with results
    """
    print(f"Processing {wav_file.stem}")
    b = Binaural.from_wav(wav_file)
    decibel = (levels[b.recording]["Left"], levels[b.recording]["Right"])
    b.calibrate_to(decibel, inplace=True)
    return process_all_metrics(b, analysis_settings, parallel=False, verbose=verbose)


def parallel_process(wav_files, results_df, levels, analysis_settings, verbose=True):
    """
    Parallel processing of binaural files

    Parameters
    ----------
    wav_files : list
        List of wav files
    results_df : pandas.DataFrame
        Results dataframe
    levels : dict
        Dictionary with levels
    analysis_settings : AnalysisSettings
        Analysis settings
    verbose : bool, optional
        Print progress, by default True

    Returns
    -------
    results_df : pandas.DataFrame
        Results dataframe
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
    with tqdm(total = len(result_objects), desc="Processing files") as pbar:
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
