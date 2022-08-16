#%%
import multiprocessing as mp
from binaural import *
from _Binaural import Binaural
from _AnalysisSettings import AnalysisSettings
from pathlib import Path
import json
import time

#%%


def load_analyse_binaural(wav_file, levels, analysis_settings, verbose=True):
    recording = wav_file.name.strip(".wav")
    b = Binaural.from_wav(wav_file)
    decibel = (levels[recording]["Left"], levels[recording]["Right"])
    b.calibrate_to(decibel, inplace=True)
    res = process_all_metrics(b, analysis_settings, verbose=verbose)
    return res


#%%


def parallel_process(wav_files, results_df, levels, analysis_settings, verbose=True):
    # Parallel processing with Pool.apply_async() without callback function

    pool = mp.Pool(mp.cpu_count() - 2)
    results = []
    result_objects = [
        pool.apply_async(
            load_analyse_binaural, args=(wav_file, levels, analysis_settings, verbose)
        )
        for wav_file in wav_files
    ]
    results = [r.get() for r in result_objects]

    pool.close()
    pool.join()

    for r in results:
        results_df = add_results(results_df, r)

    return results_df


#%%
if __name__ == "__main__":
    from datetime import datetime

    base_path = Path()
    wav_folder = base_path.joinpath("test", "data")
    levels = wav_folder.joinpath("Levels.json")
    with open(levels) as f:
        levels = json.load(f)

    analysis_settings = AnalysisSettings.from_yaml(
        base_path.joinpath("default_settings.yaml"), run_stats=True
    )

    df = prep_multiindex_df(levels, incl_metric=True)

    start = time.perf_counter()
    df = parallel_process(
        wav_folder.glob("*.wav"), df, levels, analysis_settings, verbose=True
    )
    end = time.perf_counter()
    print(f"Time taken: {end-start:.2f} seconds")

    # df.to_excel(
    #     base_path.joinpath("Data", "Test", f"ParallelTest_{datetime.today()}.xlsx")
    # )

# %%
