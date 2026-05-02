# Using Soundscapy for Binaural Recording Analysis

Soundscapy has evolved to provide a comprehensive suite of acoustic and
psychoacoustic analyses. This tutorial will guide you through using the
new `AcousticAnalysis` class, which serves as the primary interface for
performing these analyses. The system is optimized for batch processing,
ease of use, and reproducibility.

## Background

Soundscapy relies on three main packages to provide its analysis
functions:

1. [Acoustic
    Toolbox](https://github.com/Universite-Gustave-Eiffel/acoustic-toolbox)
    (`acoustic_toolbox`): Provides standard acoustic metrics with direct
    references to relevant standards.
2. [scikit-maad](https://scikit-maad.github.io) (`maad`): Offers a
    suite of ecological soundscape and bioacoustic indices.
3. [MoSQITo](https://github.com/Eomys/MoSQITo) (`mosqito`): Provides
    key psychoacoustic metrics.

The metrics available include: - From Acoustic Toolbox:
*L*<sub>*Z**e**q*</sub>, *L*<sub>*A**e**q*</sub>,
*L*<sub>*C**e**q*</sub>, SEL, and associated statistics. - From
scikit-maad: Temporal and spectral alpha indices. - From MoSQITo:
Loudness, Sharpness, and Roughness.

Soundscapy combines all of these metrics and makes it easy and
(relatively) fast to compute any or all of them for a binaural audio
recording. These results have been preliminarily confirmed through
comparison of results obtained from Head Acoustics ArtemiS suite on a
set of real-world recordings.

## Getting Started

Let’s begin by importing the necessary modules and setting up our
environment:

``` python
import json
from pathlib import Path

# imports
from soundscapy import AnalysisSettings, AudioAnalysis
```

Set up where the data is located. In this case, we’ll use the sample
recordings located under the `test` folder.

``` python
# May need to adjust for your system
wav_folder = Path().cwd().parent.parent.joinpath("test", "data")
```

## Calibration Levels

Ensuring correct calibration is crucial for accurate analysis. If you
used equipment such as the Head Acoustics SqoBold, and were careful
about how the recordings are exported to .wav, then they may already be
correctly adjusted (as ours are here). However its best to be safe and
calibrate each signal to their real-world dB level. To do this, we load
in a .json that contains the per-channel correct dB *L*<sub>*e**q*</sub>
level.

``` python
levels_file = wav_folder.joinpath("Levels.json")

with levels_file.open("r", encoding="utf-8") as f:
    levels = json.load(f)

# Look at the first five sets of levels
list(levels.items())[:5]
```

    [('CT101', {'Left': 79.0, 'Right': 79.72}),
     ('CT102', {'Left': 79.35, 'Right': 79.88}),
     ('CT103', {'Left': 76.25, 'Right': 76.41}),
     ('CT104', {'Left': 79.9, 'Right': 79.93}),
     ('CT107', {'Left': 78.21, 'Right': 78.47})]

## Initializing AudioAnalysis

The `AudioAnalysis` class is our main interface for performing acoustic
analyses. Let’s initialize it with default settings:

``` python
analysis = AudioAnalysis()
```

By default, this loads the standard configuration. If you have a custom
configuration file, you can specify it:

``` python
# analysis = AudioAnalysis("path/to/custom_config.yaml")
```

## Analyzing a Single File

Let’s analyze a single audio file:

``` python
import time

start = time.perf_counter()

binaural_wav = wav_folder.joinpath("CT101.wav")
decibel = (levels["CT101"]["Left"], levels["CT101"]["Right"])

single_file_result = analysis.analyze_file(
    binaural_wav, calibration_levels=decibel, resample=48000
)
elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed:.2f} s")
single_file_result
```

    CPU times: user 2min 58s, sys: 36 s, total: 3min 34s
    Wall time: 3min

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">LAeq</th>
<th data-quarto-table-cell-role="th">LAeq_5</th>
<th data-quarto-table-cell-role="th">LAeq_10</th>
<th data-quarto-table-cell-role="th">LAeq_50</th>
<th data-quarto-table-cell-role="th">LAeq_90</th>
<th data-quarto-table-cell-role="th">LAeq_95</th>
<th data-quarto-table-cell-role="th">LAeq_min</th>
<th data-quarto-table-cell-role="th">LAeq_max</th>
<th data-quarto-table-cell-role="th">LAeq_kurt</th>
<th data-quarto-table-cell-role="th">LAeq_skew</th>
<th data-quarto-table-cell-role="th">...</th>
<th data-quarto-table-cell-role="th">TFSD</th>
<th data-quarto-table-cell-role="th">H_Havrda</th>
<th data-quarto-table-cell-role="th">H_Renyi</th>
<th data-quarto-table-cell-role="th">H_pairedShannon</th>
<th data-quarto-table-cell-role="th">H_gamma</th>
<th data-quarto-table-cell-role="th">H_GiniSimpson</th>
<th data-quarto-table-cell-role="th">RAOQ</th>
<th data-quarto-table-cell-role="th">AGI</th>
<th data-quarto-table-cell-role="th">ROItotal</th>
<th data-quarto-table-cell-role="th">ROIcover</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Recording</th>
<th data-quarto-table-cell-role="th">Channel</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT101</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.875703</td>
<td>72.257301</td>
<td>71.154342</td>
<td>68.113339</td>
<td>63.375091</td>
<td>62.366533</td>
<td>60.560166</td>
<td>77.382651</td>
<td>0.272011</td>
<td>-0.013877</td>
<td>...</td>
<td>0.596465</td>
<td>0.306220</td>
<td>1.254552</td>
<td>2.981413</td>
<td>1004.342366</td>
<td>0.767638</td>
<td>0.012133</td>
<td>1.502836</td>
<td>38</td>
<td>1.794776</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>69.953333</td>
<td>73.623236</td>
<td>72.578152</td>
<td>68.495399</td>
<td>64.533057</td>
<td>63.097659</td>
<td>60.520566</td>
<td>78.708783</td>
<td>0.473515</td>
<td>0.140450</td>
<td>...</td>
<td>0.596309</td>
<td>0.306565</td>
<td>1.260965</td>
<td>3.032816</td>
<td>1054.904824</td>
<td>0.771196</td>
<td>0.014212</td>
<td>1.505843</td>
<td>23</td>
<td>0.940919</td>
</tr>
</tbody>
</table>

<p>2 rows × 131 columns</p>
</div>

This performs all the analyses specified in our configuration on the
single file. The `calibration_levels` parameter ensures that the
analysis is calibrated correctly.

## Batch Processing

Now, let’s analyze all the WAV files in our folder:

``` python
import time

start = time.perf_counter()

folder_results = analysis.analyze_folder(
    wav_folder, calibration_file=levels_file, resample=48000
)

end = time.perf_counter()
print(f"Time taken: {end - start:.2f} seconds")
folder_results
```

    Analyzing files:   0%|          | 0/8 [00:00<?, ?it/s]

    Time taken: 234.62 seconds

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">LAeq</th>
<th data-quarto-table-cell-role="th">LAeq_5</th>
<th data-quarto-table-cell-role="th">LAeq_10</th>
<th data-quarto-table-cell-role="th">LAeq_50</th>
<th data-quarto-table-cell-role="th">LAeq_90</th>
<th data-quarto-table-cell-role="th">LAeq_95</th>
<th data-quarto-table-cell-role="th">LAeq_min</th>
<th data-quarto-table-cell-role="th">LAeq_max</th>
<th data-quarto-table-cell-role="th">LAeq_kurt</th>
<th data-quarto-table-cell-role="th">LAeq_skew</th>
<th data-quarto-table-cell-role="th">...</th>
<th data-quarto-table-cell-role="th">TFSD</th>
<th data-quarto-table-cell-role="th">H_Havrda</th>
<th data-quarto-table-cell-role="th">H_Renyi</th>
<th data-quarto-table-cell-role="th">H_pairedShannon</th>
<th data-quarto-table-cell-role="th">H_gamma</th>
<th data-quarto-table-cell-role="th">H_GiniSimpson</th>
<th data-quarto-table-cell-role="th">RAOQ</th>
<th data-quarto-table-cell-role="th">AGI</th>
<th data-quarto-table-cell-role="th">ROItotal</th>
<th data-quarto-table-cell-role="th">ROIcover</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Recording</th>
<th data-quarto-table-cell-role="th">Channel</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT108</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>70.589507</td>
<td>74.558756</td>
<td>73.843937</td>
<td>69.525786</td>
<td>64.824693</td>
<td>64.187174</td>
<td>62.768378</td>
<td>75.729902</td>
<td>-0.672714</td>
<td>-0.097663</td>
<td>...</td>
<td>0.601720</td>
<td>0.315433</td>
<td>1.462151</td>
<td>3.302710</td>
<td>1379.622896</td>
<td>0.816123</td>
<td>0.017529</td>
<td>1.460654</td>
<td>22</td>
<td>0.849617</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>70.112106</td>
<td>73.867805</td>
<td>73.206158</td>
<td>69.500252</td>
<td>63.767889</td>
<td>63.381380</td>
<td>62.296449</td>
<td>76.077951</td>
<td>-1.033197</td>
<td>-0.220893</td>
<td>...</td>
<td>0.597461</td>
<td>0.312580</td>
<td>1.388225</td>
<td>3.167239</td>
<td>1171.735670</td>
<td>0.799022</td>
<td>0.013242</td>
<td>1.623013</td>
<td>5</td>
<td>1.254925</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT107</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.044340</td>
<td>72.248420</td>
<td>71.395037</td>
<td>66.199040</td>
<td>62.448782</td>
<td>61.533067</td>
<td>60.039913</td>
<td>76.177939</td>
<td>-0.535311</td>
<td>0.386803</td>
<td>...</td>
<td>0.596434</td>
<td>0.306238</td>
<td>1.254894</td>
<td>3.044223</td>
<td>1338.067296</td>
<td>0.770212</td>
<td>0.014416</td>
<td>1.821029</td>
<td>31</td>
<td>3.092858</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>66.957640</td>
<td>71.154329</td>
<td>69.520778</td>
<td>65.567530</td>
<td>62.737097</td>
<td>62.105325</td>
<td>59.427035</td>
<td>73.699931</td>
<td>-0.229152</td>
<td>0.490734</td>
<td>...</td>
<td>0.600826</td>
<td>0.301908</td>
<td>1.180762</td>
<td>2.863631</td>
<td>875.440353</td>
<td>0.747898</td>
<td>0.008257</td>
<td>1.578739</td>
<td>45</td>
<td>2.396588</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT101</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.875703</td>
<td>72.257301</td>
<td>71.154342</td>
<td>68.113339</td>
<td>63.375091</td>
<td>62.366533</td>
<td>60.560166</td>
<td>77.382651</td>
<td>0.272011</td>
<td>-0.013877</td>
<td>...</td>
<td>0.596465</td>
<td>0.306220</td>
<td>1.254552</td>
<td>2.981413</td>
<td>1004.342366</td>
<td>0.767638</td>
<td>0.012133</td>
<td>1.502836</td>
<td>38</td>
<td>1.794776</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>69.953333</td>
<td>73.623236</td>
<td>72.578152</td>
<td>68.495399</td>
<td>64.533057</td>
<td>63.097659</td>
<td>60.520566</td>
<td>78.708783</td>
<td>0.473515</td>
<td>0.140450</td>
<td>...</td>
<td>0.596309</td>
<td>0.306565</td>
<td>1.260965</td>
<td>3.032816</td>
<td>1054.904824</td>
<td>0.771196</td>
<td>0.014212</td>
<td>1.505843</td>
<td>23</td>
<td>0.940919</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT109</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>69.629802</td>
<td>73.858804</td>
<td>72.535949</td>
<td>68.088413</td>
<td>65.395855</td>
<td>64.841201</td>
<td>63.608754</td>
<td>77.369593</td>
<td>0.101203</td>
<td>0.709280</td>
<td>...</td>
<td>0.594079</td>
<td>0.316275</td>
<td>1.486264</td>
<td>3.292027</td>
<td>1341.924154</td>
<td>0.820587</td>
<td>0.013846</td>
<td>1.545552</td>
<td>21</td>
<td>1.545712</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>68.140737</td>
<td>71.302562</td>
<td>70.788590</td>
<td>67.176560</td>
<td>64.900719</td>
<td>64.394600</td>
<td>62.678234</td>
<td>74.251971</td>
<td>-0.501266</td>
<td>0.377754</td>
<td>...</td>
<td>0.593684</td>
<td>0.310974</td>
<td>1.350959</td>
<td>3.076539</td>
<td>1161.508584</td>
<td>0.788654</td>
<td>0.014172</td>
<td>1.580760</td>
<td>43</td>
<td>2.021005</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT110</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.696864</td>
<td>73.688152</td>
<td>73.278617</td>
<td>66.395310</td>
<td>61.541000</td>
<td>61.077132</td>
<td>59.521031</td>
<td>75.734980</td>
<td>-0.902389</td>
<td>0.321096</td>
<td>...</td>
<td>0.596778</td>
<td>0.315119</td>
<td>1.453479</td>
<td>3.330735</td>
<td>1742.442706</td>
<td>0.817141</td>
<td>0.019447</td>
<td>1.635892</td>
<td>4</td>
<td>0.903032</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>68.086205</td>
<td>72.842216</td>
<td>72.233530</td>
<td>65.950830</td>
<td>60.897040</td>
<td>60.072192</td>
<td>58.683020</td>
<td>74.284699</td>
<td>-1.039466</td>
<td>0.153424</td>
<td>...</td>
<td>0.598589</td>
<td>0.306559</td>
<td>1.260850</td>
<td>2.989754</td>
<td>1138.129400</td>
<td>0.768657</td>
<td>0.011267</td>
<td>1.667605</td>
<td>3</td>
<td>2.584879</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT102</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>70.613447</td>
<td>74.541317</td>
<td>73.322597</td>
<td>69.297264</td>
<td>65.074481</td>
<td>64.561974</td>
<td>63.337182</td>
<td>78.922344</td>
<td>0.270694</td>
<td>0.530142</td>
<td>...</td>
<td>0.601338</td>
<td>0.315584</td>
<td>1.466408</td>
<td>3.338957</td>
<td>1570.553756</td>
<td>0.817882</td>
<td>0.018555</td>
<td>1.649929</td>
<td>2</td>
<td>3.491107</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>70.491840</td>
<td>75.681356</td>
<td>73.055797</td>
<td>69.131814</td>
<td>64.975452</td>
<td>64.209771</td>
<td>63.114758</td>
<td>81.634943</td>
<td>0.648596</td>
<td>0.601576</td>
<td>...</td>
<td>0.594765</td>
<td>0.314666</td>
<td>1.441181</td>
<td>3.266859</td>
<td>1171.085446</td>
<td>0.813338</td>
<td>0.012127</td>
<td>1.497619</td>
<td>50</td>
<td>2.569614</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT104</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>72.385734</td>
<td>76.823628</td>
<td>75.846019</td>
<td>70.883017</td>
<td>68.178606</td>
<td>67.906102</td>
<td>67.060970</td>
<td>78.252226</td>
<td>-0.551273</td>
<td>0.581452</td>
<td>...</td>
<td>0.601207</td>
<td>0.319373</td>
<td>1.586474</td>
<td>3.467531</td>
<td>1784.788259</td>
<td>0.840308</td>
<td>0.022362</td>
<td>1.607451</td>
<td>21</td>
<td>1.532046</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>70.661228</td>
<td>75.007596</td>
<td>72.920992</td>
<td>69.693243</td>
<td>67.615660</td>
<td>67.257368</td>
<td>65.945547</td>
<td>76.179642</td>
<td>0.158796</td>
<td>0.744622</td>
<td>...</td>
<td>0.592096</td>
<td>0.313899</td>
<td>1.421059</td>
<td>3.162598</td>
<td>1054.720367</td>
<td>0.803737</td>
<td>0.013435</td>
<td>1.517441</td>
<td>98</td>
<td>4.713390</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT103</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>66.330006</td>
<td>69.010487</td>
<td>68.241612</td>
<td>65.579475</td>
<td>63.680847</td>
<td>63.194765</td>
<td>62.103353</td>
<td>74.234566</td>
<td>2.236365</td>
<td>1.127518</td>
<td>...</td>
<td>0.591106</td>
<td>0.309768</td>
<td>1.324691</td>
<td>3.109320</td>
<td>1062.342802</td>
<td>0.785795</td>
<td>0.013687</td>
<td>1.484990</td>
<td>117</td>
<td>3.370993</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>66.320960</td>
<td>69.247212</td>
<td>68.344756</td>
<td>65.717681</td>
<td>64.145147</td>
<td>63.852638</td>
<td>62.685126</td>
<td>72.061583</td>
<td>1.008471</td>
<td>0.981831</td>
<td>...</td>
<td>0.583920</td>
<td>0.307667</td>
<td>1.281979</td>
<td>3.053171</td>
<td>1152.246760</td>
<td>0.775524</td>
<td>0.013850</td>
<td>1.521505</td>
<td>136</td>
<td>6.565286</td>
</tr>
</tbody>
</table>

<p>16 rows × 131 columns</p>
</div>

### Saving Results

We can easily save our results to a file:

``` python
analysis.save_results(folder_results, "acoustic_analysis_results.xlsx")
```

## Customizing the Analysis

### Updating Configuration

If we want to modify our analysis configuration, we can do so using the
`update_config` method:

``` python
new_config = {"AcousticToolbox": {"LAeq": {"run": False}}}

analysis.update_config(new_config)
print("Configuration updated")
```

    Configuration updated

This would disable the LAeq analysis in subsequent runs.

### Saving the Updated Configuration

We can save the updated configuration to a file:

``` python
analysis.save_config("updated_config.yaml")
print("Updated configuration saved to 'updated_config.yaml'")
```

Of course, you could also directly edit your config.yaml file instead.

## Advanced Usage

### Custom Analysis Settings

For more control, we can create a custom `AnalysisSettings`:

``` python
custom_settings = AnalysisSettings.from_yaml("example_settings.yaml")
custom_settings.update_setting("scikit_maad", "all_temporal_alpha_indices", run=True)
custom_settings.update_setting("scikit_maad", "all_spectral_alpha_indices", run=True)

# Create a new AudioAnalysis instance with the custom settings
custom_analysis = AudioAnalysis(config_path="example_settings.yaml")

# Or update an existing instance
analysis.update_config(custom_settings.model_dump())
```

## Parallel Processing Control

The `analyze_folder` method uses parallel processing by default. You can
control the number of worker processes using the `max_workers` argument.
Setting `max_workers = None` (the default) will use all available CPU
cores. Setting `max_workers = 1` will disable parallel processing, and
will take significantly longer to process:

``` python
start = time.perf_counter()

serial_analysis = AudioAnalysis()
folder_results = serial_analysis.analyze_folder(
    wav_folder, calibration_file=levels_file, max_workers=1, resample=48000
)

end = time.perf_counter()

print(f"Time taken: {end - start:.2f} seconds")
folder_results
```

    Analyzing files:   0%|          | 0/8 [00:00<?, ?it/s]

    Time taken: 1468.20 seconds

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">LAeq</th>
<th data-quarto-table-cell-role="th">LAeq_5</th>
<th data-quarto-table-cell-role="th">LAeq_10</th>
<th data-quarto-table-cell-role="th">LAeq_50</th>
<th data-quarto-table-cell-role="th">LAeq_90</th>
<th data-quarto-table-cell-role="th">LAeq_95</th>
<th data-quarto-table-cell-role="th">LAeq_min</th>
<th data-quarto-table-cell-role="th">LAeq_max</th>
<th data-quarto-table-cell-role="th">LAeq_kurt</th>
<th data-quarto-table-cell-role="th">LAeq_skew</th>
<th data-quarto-table-cell-role="th">...</th>
<th data-quarto-table-cell-role="th">TFSD</th>
<th data-quarto-table-cell-role="th">H_Havrda</th>
<th data-quarto-table-cell-role="th">H_Renyi</th>
<th data-quarto-table-cell-role="th">H_pairedShannon</th>
<th data-quarto-table-cell-role="th">H_gamma</th>
<th data-quarto-table-cell-role="th">H_GiniSimpson</th>
<th data-quarto-table-cell-role="th">RAOQ</th>
<th data-quarto-table-cell-role="th">AGI</th>
<th data-quarto-table-cell-role="th">ROItotal</th>
<th data-quarto-table-cell-role="th">ROIcover</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Recording</th>
<th data-quarto-table-cell-role="th">Channel</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT108</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>70.589507</td>
<td>74.558756</td>
<td>73.843937</td>
<td>69.525786</td>
<td>64.824693</td>
<td>64.187174</td>
<td>62.768378</td>
<td>75.729902</td>
<td>-0.672714</td>
<td>-0.097663</td>
<td>...</td>
<td>0.601720</td>
<td>0.315433</td>
<td>1.462151</td>
<td>3.302710</td>
<td>1379.622896</td>
<td>0.816123</td>
<td>0.017529</td>
<td>1.460654</td>
<td>22</td>
<td>0.849617</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>70.112106</td>
<td>73.867805</td>
<td>73.206158</td>
<td>69.500252</td>
<td>63.767889</td>
<td>63.381380</td>
<td>62.296449</td>
<td>76.077951</td>
<td>-1.033197</td>
<td>-0.220893</td>
<td>...</td>
<td>0.597461</td>
<td>0.312580</td>
<td>1.388225</td>
<td>3.167239</td>
<td>1171.735670</td>
<td>0.799022</td>
<td>0.013242</td>
<td>1.623013</td>
<td>5</td>
<td>1.254925</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT109</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>69.629802</td>
<td>73.858804</td>
<td>72.535949</td>
<td>68.088413</td>
<td>65.395855</td>
<td>64.841201</td>
<td>63.608754</td>
<td>77.369593</td>
<td>0.101203</td>
<td>0.709280</td>
<td>...</td>
<td>0.594079</td>
<td>0.316275</td>
<td>1.486264</td>
<td>3.292027</td>
<td>1341.924154</td>
<td>0.820587</td>
<td>0.013846</td>
<td>1.545552</td>
<td>21</td>
<td>1.545712</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>68.140737</td>
<td>71.302562</td>
<td>70.788590</td>
<td>67.176560</td>
<td>64.900719</td>
<td>64.394600</td>
<td>62.678234</td>
<td>74.251971</td>
<td>-0.501266</td>
<td>0.377754</td>
<td>...</td>
<td>0.593684</td>
<td>0.310974</td>
<td>1.350959</td>
<td>3.076539</td>
<td>1161.508584</td>
<td>0.788654</td>
<td>0.014172</td>
<td>1.580760</td>
<td>43</td>
<td>2.021005</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT101</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.875703</td>
<td>72.257301</td>
<td>71.154342</td>
<td>68.113339</td>
<td>63.375091</td>
<td>62.366533</td>
<td>60.560166</td>
<td>77.382651</td>
<td>0.272011</td>
<td>-0.013877</td>
<td>...</td>
<td>0.596465</td>
<td>0.306220</td>
<td>1.254552</td>
<td>2.981413</td>
<td>1004.342366</td>
<td>0.767638</td>
<td>0.012133</td>
<td>1.502836</td>
<td>38</td>
<td>1.794776</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>69.953333</td>
<td>73.623236</td>
<td>72.578152</td>
<td>68.495399</td>
<td>64.533057</td>
<td>63.097659</td>
<td>60.520566</td>
<td>78.708783</td>
<td>0.473515</td>
<td>0.140450</td>
<td>...</td>
<td>0.596309</td>
<td>0.306565</td>
<td>1.260965</td>
<td>3.032816</td>
<td>1054.904824</td>
<td>0.771196</td>
<td>0.014212</td>
<td>1.505843</td>
<td>23</td>
<td>0.940919</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT102</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>70.613447</td>
<td>74.541317</td>
<td>73.322597</td>
<td>69.297264</td>
<td>65.074481</td>
<td>64.561974</td>
<td>63.337182</td>
<td>78.922344</td>
<td>0.270694</td>
<td>0.530142</td>
<td>...</td>
<td>0.601338</td>
<td>0.315584</td>
<td>1.466408</td>
<td>3.338957</td>
<td>1570.553756</td>
<td>0.817882</td>
<td>0.018555</td>
<td>1.649929</td>
<td>2</td>
<td>3.491107</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>70.491840</td>
<td>75.681356</td>
<td>73.055797</td>
<td>69.131814</td>
<td>64.975452</td>
<td>64.209771</td>
<td>63.114758</td>
<td>81.634943</td>
<td>0.648596</td>
<td>0.601576</td>
<td>...</td>
<td>0.594765</td>
<td>0.314666</td>
<td>1.441181</td>
<td>3.266859</td>
<td>1171.085446</td>
<td>0.813338</td>
<td>0.012127</td>
<td>1.497619</td>
<td>50</td>
<td>2.569614</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT103</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>66.330006</td>
<td>69.010487</td>
<td>68.241612</td>
<td>65.579475</td>
<td>63.680847</td>
<td>63.194765</td>
<td>62.103353</td>
<td>74.234566</td>
<td>2.236365</td>
<td>1.127518</td>
<td>...</td>
<td>0.591106</td>
<td>0.309768</td>
<td>1.324691</td>
<td>3.109320</td>
<td>1062.342802</td>
<td>0.785795</td>
<td>0.013687</td>
<td>1.484990</td>
<td>117</td>
<td>3.370993</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>66.320960</td>
<td>69.247212</td>
<td>68.344756</td>
<td>65.717681</td>
<td>64.145147</td>
<td>63.852638</td>
<td>62.685126</td>
<td>72.061583</td>
<td>1.008471</td>
<td>0.981831</td>
<td>...</td>
<td>0.583920</td>
<td>0.307667</td>
<td>1.281979</td>
<td>3.053171</td>
<td>1152.246760</td>
<td>0.775524</td>
<td>0.013850</td>
<td>1.521505</td>
<td>136</td>
<td>6.565286</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT107</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.044340</td>
<td>72.248420</td>
<td>71.395037</td>
<td>66.199040</td>
<td>62.448782</td>
<td>61.533067</td>
<td>60.039913</td>
<td>76.177939</td>
<td>-0.535311</td>
<td>0.386803</td>
<td>...</td>
<td>0.596434</td>
<td>0.306238</td>
<td>1.254894</td>
<td>3.044223</td>
<td>1338.067296</td>
<td>0.770212</td>
<td>0.014416</td>
<td>1.821029</td>
<td>31</td>
<td>3.092858</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>66.957640</td>
<td>71.154329</td>
<td>69.520778</td>
<td>65.567530</td>
<td>62.737097</td>
<td>62.105325</td>
<td>59.427035</td>
<td>73.699931</td>
<td>-0.229152</td>
<td>0.490734</td>
<td>...</td>
<td>0.600826</td>
<td>0.301908</td>
<td>1.180762</td>
<td>2.863631</td>
<td>875.440353</td>
<td>0.747898</td>
<td>0.008257</td>
<td>1.578739</td>
<td>45</td>
<td>2.396588</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT104</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>72.385734</td>
<td>76.823628</td>
<td>75.846019</td>
<td>70.883017</td>
<td>68.178606</td>
<td>67.906102</td>
<td>67.060970</td>
<td>78.252226</td>
<td>-0.551273</td>
<td>0.581452</td>
<td>...</td>
<td>0.601207</td>
<td>0.319373</td>
<td>1.586474</td>
<td>3.467531</td>
<td>1784.788259</td>
<td>0.840308</td>
<td>0.022362</td>
<td>1.607451</td>
<td>21</td>
<td>1.532046</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>70.661228</td>
<td>75.007596</td>
<td>72.920992</td>
<td>69.693243</td>
<td>67.615660</td>
<td>67.257368</td>
<td>65.945547</td>
<td>76.179642</td>
<td>0.158796</td>
<td>0.744622</td>
<td>...</td>
<td>0.592096</td>
<td>0.313899</td>
<td>1.421059</td>
<td>3.162598</td>
<td>1054.720367</td>
<td>0.803737</td>
<td>0.013435</td>
<td>1.517441</td>
<td>98</td>
<td>4.713390</td>
</tr>
<tr>
<td rowspan="2" data-quarto-table-cell-role="th"
data-valign="top">CT110</td>
<td data-quarto-table-cell-role="th">Left</td>
<td>68.696864</td>
<td>73.688152</td>
<td>73.278617</td>
<td>66.395310</td>
<td>61.541000</td>
<td>61.077132</td>
<td>59.521031</td>
<td>75.734980</td>
<td>-0.902389</td>
<td>0.321096</td>
<td>...</td>
<td>0.596778</td>
<td>0.315119</td>
<td>1.453479</td>
<td>3.330735</td>
<td>1742.442706</td>
<td>0.817141</td>
<td>0.019447</td>
<td>1.635892</td>
<td>4</td>
<td>0.903032</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">Right</td>
<td>68.086205</td>
<td>72.842216</td>
<td>72.233530</td>
<td>65.950830</td>
<td>60.897040</td>
<td>60.072192</td>
<td>58.683020</td>
<td>74.284699</td>
<td>-1.039466</td>
<td>0.153424</td>
<td>...</td>
<td>0.598589</td>
<td>0.306559</td>
<td>1.260850</td>
<td>2.989754</td>
<td>1138.129400</td>
<td>0.768657</td>
<td>0.011267</td>
<td>1.667605</td>
<td>3</td>
<td>2.584879</td>
</tr>
</tbody>
</table>

<p>16 rows × 131 columns</p>
</div>

As we can see, on my system, enabling parallel processing reduces the
processing time for these 8 files from almost 25 minutes to less than 4
minutes. This will vary depending on your system and the number of files
you are processing. The more CPU cores and the more files, the more
beneficial parallel processing will be.

## Conclusion

The `AudioAnalysis` class provides a powerful and flexible interface for
performing acoustic and psychoacoustic analyses on binaural recordings.
It simplifies the process of batch analysis, configuration management,
and result handling, making it easier to process large datasets
consistently and efficiently.

Remember that the specific metrics calculated and their settings are
determined by the configuration. Always ensure your configuration
accurately reflects your analysis needs, keep in mind that the
psychoacoustic analyses in MoSQITo can be computationally intensive, and
don’t hesitate to customize it for your specific research or application
requirements.
