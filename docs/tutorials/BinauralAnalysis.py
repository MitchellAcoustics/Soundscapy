# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.4 ('soundscapy-dev')
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Using Soundscapy for Binaural Recording Analysis

In v0.3.2, the ability to perform a huge suite of (psycho)acoustic analyses has been added to Soundscapy. This has been optimised for performing batch processing of many recordings, ease of use, and reproducibility. To do this, we rely on three packages to provide the analysis functions:

 * [Python Acoustics](https://github.com/python-acoustics/python-acoustics) (`acoustics`) : Python Acoustics is a library aimed at acousticians. It provides two huge benefits - first, all of the analysis functions are referenced directly to the relevant standard. Second, we subclass their `Signal` class to provide the Binaural functionality and any function available within the `Signal` class is also available to Soundscapy's `Binaural` class.
 * [scikit-maad](https://scikit-maad.github.io) (`maad`) : scikit-maad is a modular toolbox for quantitiative soundscape analysis, focussed on ecological soundscapes and bioacoustic indices. scikit-maad provides a huge suite of ecosoundscape focussed indices, including Acoustic Richness Index, Acoustic Complexity Index, Normalized Difference Soundscape Index (NDSI), and more.
 * [MoSQITo](https://github.com/Eomys/MoSQITo) (`mosqito`): MoSQITo is a modular framework of key sound quality metrics, providing the psychoacoustic metrics.

The metrics currently available are:
* Python Acoustics : $L_{Zeq}$, $L_{Aeq}$, $L_{Ceq}$, SEL, and all associated statistics ($L_5$ through $L_{95}$, $L_{max}$ and $L_{min}$, as well as [kurtosis](https://acousticstoday.org/wp-content/uploads/2020/12/Kurtosis-A-New-Tool-for-Noise-Analysis-Wei-Qiu-William-J.-Murphy-and-Alice-Suter.pdf) and skewness.
* scikit-maad : So far we have only implemented the combined `all_temporal_alpha_indices` and `all_spectral_alpha_indices` from `scikit-maad`; calculating them individually is not yet supported. `all_temporal_alpha_indices` comprises 16 temporal domain acoustic indices, such as temporal signal-to-noise-ratio, temporal entropy, temporal events. `all_spectral_alpha_indices` comprises 19 spectral domain acoustic indices, such as Bioacoustic Index, Acoustic Diversity Index, NDSI, Acoustic Evenness Index.
* MoSQITo :
    * Loudness (Zwicker time varying),
    * Sharpness (time varying, from loudness, and per segment with DIN, Aures, Bismarck, and Fastl weightings),
    * Roughness (Daniel and Weber, 1997).

Soundscapy combines all of these metrics and makes it easy and (relatively) fast to compute any or all of them for a binaural audio recording. These results have been preliminarily confirmed through comparison of results obtained from Head Acoustics ArtemiS suite on a set of real-world recordings.

## Consistent Analysis Settings

A primary goal when developing this library was to make it easy to save and document the settings used for all analyses. This is done through a `settings.yaml` file and the `AnalysisSettings` class. Although the settings for each metric can be set at runtime, the `settings.yaml` file allows you to set all of the settings at once and document exactly what settings were passed to each analysis function and to share these settings with collaborators or reviewers.

## Batch processing
The other primary goal was to make it simple and fast to perform this analysis on many recordings. One aspect of this is unifying the outputs from the underlying libraries and presenting them in an easy to parse format. The analysis functions from Soundscapy can return a MultiIndex pandas DataFrame with the Recording filename and Left and Right channels in the index and a column for each metric calculated. This dataframe can then be easily saved to a .csv or Excel spreadsheet. Alternatively, a dictionary can be returned for further processing within Python. The key point is that after calculating 100+ metrics for 1,000+ recordings, you'll be left with a single tidy spreadsheet.

The Soundscape Indices (SSID) project for which this was developed has over 3,000 recordings for which we needed to calculate a full suite of metrics for both channels. In particular, the MoSQITo functions can be quite slow, so running each recording one at a time can be prohibitively slow and only utilize a small portion of the available computing power. To help with this, a set of simple functions is provided to enable parallel processing, such that multiple recordings can be processed simultaneously by a multi-core CPU. In our initial tests on a 16-core AMD Ryzen 7 4800HS CPU (Fedora Linux 36), this increased the speed for processing 20 recordings by at least 8 times.

In testing, the MoSQITo functions are particularly slow, taking up to 3 minutes to calculate the Loudness for a 30s two-channel recording. When running only a single recording through, this has also been sped up by parallelizing the per-channel calculation, reducing the computation time to around 50s.
"""

# %% [markdown]
"""
## Getting Started

The basis of all of our analysis starts with the binaural signal, so we begin by importing the `Binaural` class. We'll also load up and examine our analysis settings. Throughout Soundscapy, we use `pathlib.Path` for defining filepaths.

"""

# %%
# Add soundscapy to the Python path
import sys
sys.path.append('../..')

# imports
from soundscapy import Binaural
from soundscapy import AnalysisSettings
from soundscapy.analysis.binaural import prep_multiindex_df, add_results, process_all_metrics
import json
from pathlib import Path


# %% [markdown]
"""
Set up where the data is located. In this case, we'll use the sample recordings located under the `test` folder.
"""

# %%
# May need to adjust for your system
wav_folder = Path().cwd().parent.joinpath("test", "data")

# %% [markdown]
"""

Ensuring that Soundscapy knows exactly how loud your recordings were onsite is crucial to getting correct answers. If you used equipment such as the Head Acoustics SqoBold, and were careful about how the recordings are exported to .wav, then they may already be correctly adjusted (as ours are here). However its best to be safe and calibrate each signal to their real-world dB level. To do this, we load in a .json that contains the per-channel correct dB $L_{eq}$ level.
"""

# %%
levels = wav_folder.joinpath("Levels.json")

with open(levels) as f:
    levels = json.load(f)

# Look at the first five sets of levels
list(levels.items())[:5]

# %% [markdown]
"""

### Prepping the results dataframe
The easiest way to organise and add the new data as it is processed is to prepare a dataframe ahead of time. We've provided a small function to convert a dictionary of calibration levels (`level`) into the properly formatted dataframe.
"""

# %%
df = prep_multiindex_df(levels, incl_metric=True)
df.head()

# %% [markdown]
"""
#### Load in a Binaural recording
Load in a binaural wav signal. We can use the `plot` function provided by the `acoustics.Signal` super-class.
"""

# %%
binaural_wav = wav_folder.joinpath("CT101.wav")
b = Binaural.from_wav(binaural_wav)
b.plot()

# %% [markdown]
"""
To ensure that the dB level is correct, and therefore any other metrics are correct, we start by calibrating the signal to precalculated levels.

"""

# %%
decibel = (levels[b.recording]["Left"], levels[b.recording]["Right"])
print(f"Calibration levels: {decibel}")
b.calibrate_to(decibel, inplace=True)

# %% [markdown]
"""
Now, check it by comparing it to what we already knew were the correct levels:
"""

# %%
print(f"Predefined levels: {levels[b.recording]}")
print(f"Calculated Levels: {b.pyacoustics_metric('Leq', statistics=['avg'], as_df=False)}")

# %% [markdown]
"""
## Calculating Acoustic Metrics

This brings us to how to calculate any of the many metrics available. Let's start simple with $L_{Aeq}$.

#### Python Acoustics

Since the $L_{Aeq}$ calc is provided by the Python Acoustics library, we'll be calling `pyacoustic_metric`. Then, we need to tell it what particular metric we want, what stats to calculate as well, what to label it, and what format to return the results in.
"""

# %%
metric = "LAeq"
stats = ("avg", 10, 50, 90, 95, "max")
label = "LAeq"
b.pyacoustics_metric(metric, stats, label, as_df=False)

# %% [markdown]
"""
If we want, we can get the results back as a pandas DataFrame instead:
"""

# %%
b.pyacoustics_metric(metric, stats, label, as_df=True)

# %% [markdown]
"""
And we can easily do the same for the C-weighting level:
"""

# %%
b.pyacoustics_metric("LCeq", stats, as_df=True)

# %% [markdown]
"""
#### MoSQITo

MoSQITo is very exciting as it is one of the first completely free and open-source libraries for calculating psychoacoustic features. Let's try out calculating the psychoacoustic loudness.

We start by defining many of the same options, but with two new ones. The first is our `func_args` to pass to `MoSQITo`. Since our test recording was collected in a public park, we need to select the correct field type: free or diffuse, and pass that to MoSQITo.

The second new argument is `parallel`. This just tells Soundscapy whether to try to calculate the Left and Right channels simultaneously to speed up processing.
"""

# %%
metric = "loudness_zwtv"
stats = (5, 50, 'avg', 'max')
func_args = {
    'field_type': 'free'
}

b.mosqito_metric(metric, statistics=stats, as_df=True, parallel=True, verbose=True, func_args=func_args)

# %% [markdown]
"""
`sharpness_din_from_loudness` is a bit of a special case to keep in mind. It can drastically speed up the processing time since it calculates the Sharpness values from pre-calculated Loudness results. If you are planning to do both analyses, I highly suggest using it. Soundscapy will handle it behind the scenes to make sure it doesn't accidentally calculate the Loudness values twice if you've asked for both of them. Let's try it out.
"""

# %%
b.mosqito_metric("sharpness_din_from_loudness", stats, as_df=True, parallel=True, verbose=True, func_args=func_args)

# %% [markdown]
"""
By default, the metrics will be calculated for both channels. But you may want only a single channel. This can be set with the `channel` option.
"""

# %%
b.pyacoustics_metric("LZeq", channel="Left")

# %%
b.maad_metric("all_spectral_alpha_indices", verbose=True)

# %% [markdown]
"""
### Defining Analysis Settings

Soundscapy provides the ability to predefine your analysis settings. These are defined in a separate `.yaml` file and are managed by Soundscapy using the `AnalysisSettings` class. These settings can then be passed to any of the analysis functions, rather than separately defining your settings as we did above. This will be particularly useful when performing our batch processing on an entire folder of wav recordings later.

Soundscapy provides a set of default settings which can be easily loaded in:
"""

# %%
analysis_settings = AnalysisSettings.default()

# %% [markdown]
"""
However, in your own analysis you'll probably want to define your own options and load that in. We'll show how this is done using the `example_settings.yaml' file. First, let's take a look at how it's laid out:

```
# Settings file for batch acoustic analysis.
# Split up according to which library performs which analysis.

# Python Acoustics
# Supported metrics: LAeq, LZeq, LCeq, SEL
# Supported stats: avg/mean, max, min, kurt, skew, any integer from 1-99
PythonAcoustics:
    LAeq:
        run: true
        main: 'avg'
        statistics: [5, 10, 50, 90, 95, 'min', 'max', 'kurt', 'skew']
        channel: ["Left", "Right"]
        label: 'LAeq'
        func_args:
            time: 0.125
            method: "average"

    LZeq:
        run: true
        main: 'avg'
        statistics: [5, 10, 50, 90, 95, 'min', 'max', 'kurt', 'skew']
        channel: ["Left", "Right"]
        label: 'LZeq'
        func_args:
            time: 0.125
            method: "average"

# MoSQITo
# supported metrics: loudness_zwtv, sharpness_din_from_loudness, roughness_dw
# supported stats: avg/mean, max, min, kurt, skew, any integer from 1-99
MoSQITo:
    loudness_zwtv:
        run: true
        main: 5
        statistics: [10, 50, 90, 95, 'min', 'max', 'kurt', 'skew', 'avg']
        channel: ["Left", "Right"]
        label: "N"
        parallel: true
        func_args:
            field_type: "free"

```
"""

# %% [markdown]
"""

The settings file is broken up according to the three libraries. Within these, we define separate options for each metric to calculate. The name of this metric should correspond exactly with what Soundscapy expects (in the case of PythonAcoustics) or what the underlying library calls its function.

Within each function, we then have a collection of settings that Soundscapy uses:
    * `run` : This tells Soundscapy whether or not to actually run this metric. This allows you to define and save the options you use for each metric without needing to run it.
    * `main` and `statistics` : These are the sub-statistics to calculate (e.g. $L_{5}$, $L_{90}$, etc.). `main` operates just like any of these, except you also have the option to return only the main statistic to simplify your results while still leaving your other preferences intact.
    * `label` : What label to assign that metric. For instance, Loudness is typically `'N'`. When calculated, the statistics will be appended like so: N_5, N_10, N_avg, ... N_{`stat`} and this will be the column name for that metric. If you pass nothing here, then Soundscapy will fall back to the labels defined in `sq_metrics.DEFAULT_LABELS`. **Warning**: Some functions share a label (e.g. `sharpness_din_tv` and `sharpness_din_perseg` are both 'S'), if you run both of these and don't define different labels, one will overwrite the other.

Finally, there is an opportunity to define arguments to pass to the underlying function itself. This is perhaps the most important part for consistency and reproducibility. This is where you define which standard is being used, what time or frequency weighting, or what spectrogram bins to use. These options are defined by the 3 analysis libraries used and are not documented fully in Soundscapy. When the `AnalysisSettings` is parsed, `func_args` will be returned as a `dict` with an entry for each option you'd like to pass. `func_args` is then passed as `**kwargs` to e.g. the `mosqito.sq_metrics.loudness_zwtv()` function. If `func_args` contains an option the function doesn't recognise it will throw an error, so be careful when defining these arguments.

---------------------------------------
Let's try loading in the `example_settings.yaml` file and see how `AnalysisSettings` handles it.
"""

# %%
ex_settings = AnalysisSettings.from_yaml(Path("../../examples/example_settings.yaml"))
ex_settings

# %% [markdown]
"""
`ex_settings` is just a Python `dict` with some class methods added on. One of these is a function to parse the settings object for a specific library:

"""

# %%
ex_settings.parse_pyacoustics(metric="LAeq")

# %% [markdown]
"""
This returns the value for `run`, `channel`, `statistics`, `label`, and `func_args` which will then be used by the `pyacoustic_metric()` function to calculate the $L_{Aeq}$ and its stats.

When passing your settings to an analysis function, it will start by automatically parsing and applying the settings for that particular metric. This will override any other settings passed to the function, so if you're using a settings file and you want to change anything, you should either change it in the `.yaml` and reload the settings. This also makes sure you keep a record of the settings for the last time you ran the analysis.

You can easily reload the settings `.yaml` after changing it:
"""

# %%
ex_settings = ex_settings.reload()

# %% [markdown]
"""
### Running a single metric with predefined settings

Now, with our settings loaded, we process a recording using those settings:
"""

# %%
b.pyacoustics_metric("LAeq", analysis_settings=ex_settings)

# %% [markdown]
"""
But this is just the start of what makes the analysis settings so useful.
"""

# %% [markdown]
"""
## Processing all the metrics at once, using predefined analysis settings.

Since we can define the settings for all the metrics, and specify which metrics we want to run, we can process all of our desired metrics at once. `process_all_metrics()` let's us do this with just a single line of code.
"""

# %%
b.process_all_metrics(ex_settings, verbose=True)

# %% [markdown]
"""
In this case we left out the MoSQITo metrics since they take a while, but let's say you do want to run those. You could either edit the `example_settings.yaml` file and use `ex_settings.reload()`, or what we'll do within this notebook is just directly edit the underlying dict.
"""

# %%
ex_settings['MoSQITo']["loudness_zwtv"]["run"] = True

b.process_all_metrics(ex_settings, verbose=True)

# %% [markdown]
"""
And now we have all of the same metrics from before, along with the psychoacoustic Loudness.
"""

# %% [markdown]
"""
## Batch processing a bunch of recordings.

The final step is to run all of these metrics on a whole bunch of wav recordings all at once. In the Soundscape Indices (SSID) project for which this package was developed, we use this data to train machine learning models, so it's necessary to process a large number of recordings. Using the predefined analysis settings, we can loop through an entire folder, add it all to the results dataframe and come away with a single spreadsheet of all of the metrics needed.

To start, we'll demonstrate this using a normal `for` loop. The `for` loop will process one recording at a time and add it to our results dataframe. For the demonstration, we'll reload all of the variables needed and use the default analysis settings to run all of the metrics and time it.
"""

# %%
import time # Just for timing

wav_folder = Path().cwd().parent.joinpath("test", "data")
levels = wav_folder.joinpath("Levels.json")
with open(levels) as f:
    levels = json.load(f)
df = prep_multiindex_df(levels, incl_metric=False)

analysis_settings = AnalysisSettings.default()

begin = time.perf_counter() # Start timer

# Loop through each wav file in the folder
for wav in wav_folder.glob("*.wav"):
    recording = wav.stem
    decibel = tuple(levels[recording].values())
    b = Binaural.from_wav(wav, calibrate_to=decibel)
    df = add_results(df, b.process_all_metrics(analysis_settings, verbose=True)) # Process all metrics and add to results df

end = time.perf_counter() # Stop timer

# %%
print("Run on 16 core AMD Ryzen 7 4800HS (Fedora Linux)")
print(f"Time taken (using for loop): {end-begin: .2f} seconds ({(end-begin)/60: .2f} minutes)")
df

# %% [markdown]
"""
As we can see, this took quite a long time. The problem is that the MoSQITo metrics can take up to a minute for each channel of each recording and the for loop only processes one recording at a time. Since most modern computers are multicore, this leaves a ton of processing power unused. On my machine, I have 16 cores and at any one time during this process, only two of those cores are running at 100%.

In order to take full advantage of the other cores, we need to tell the computer to process multiple recordings at once and add them all together later. This is called parallel processing and could theoretically speed up the analysis by 8x (usually we don't get this full speed up though). The reason two cores are running in parallel above is because we're already running both channels at the same time for the MoSQITo metrics.

To do this, we provide a function called `parallel_process()` which takes the path to your wav folder as an argument, then performs our full processing on multiple recordings in parallel and returns the full result dataframe at the end.

Note: Don't worry about the status updates looking  a bit jumbled - that's what happens with parallel processing.
"""

# %%
import time
from soundscapy.analysis.parallel_processing import parallel_process

# Redefine path etc. just for the example
wav_folder = Path().cwd().parent.joinpath("test", "data")
levels = wav_folder.joinpath("Levels.json")
with open(levels) as f:
    levels = json.load(f)
df = prep_multiindex_df(levels, incl_metric=False)
analysis_settings = AnalysisSettings.default()

start = time.perf_counter() # Start timer

df = parallel_process(
    wav_folder.glob("*.wav"), df, levels, analysis_settings, verbose=False
)

stop = time.perf_counter()

# %%
print("Run on 16 core AMD Ryzen 7 4800HS (Fedora Linux)")
print(f"Time taken: {stop-start:.2f} seconds ({(stop-start)/60: .2f} minutes)")
df

# %% [markdown]
"""
That's three times as fast! It should be noted that it appears any of the parallel operations, including the parallel channel processing, run faster (~25%) on Linux than on Windows on the same machine.

Then save it if you want to.
"""

# %% [markdown]
"""
That's three times as fast! It should be noted that it appears any of the parallel operations, including the parallel channel processing, run faster (~25%) on Linux than on Windows on the same machine.

Then save it if you want to.
"""

# %%
# from datetime import datetime
# df.to_excel(wav_folder.joinpath("test", f"ParallelTest_{datetime.today().strftime('%Y-%m-%d')}.xlsx"))

# %%


