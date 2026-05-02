# Soundscape Assessment Tutorial

``` python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import soundscapy as sspy
from soundscapy import ISOPlot
from soundscapy.spi import DirectParams, MultiSkewNorm

# Set up plotting
warnings.simplefilter("always")
```

## Introduction

This tutorial will guide you through the process of conducting a
comprehensive soundscape assessment using Soundscapy. You’ll learn how
to analyze survey data, create visualizations, and interpret the results
to understand how people perceive and experience soundscapes in
different locations.

By the end of this tutorial, you’ll be able to: - Load and validate
soundscape survey data - Calculate ISO coordinates from perceptual
attribute questions (PAQs) - Create basic and advanced visualizations of
soundscape data - Perform statistical analysis of soundscape
perceptions - Apply the Soundscape Perception Index (SPI) to evaluate
soundscapes against target distributions - Create professional reports
and presentations of your findings

Let’s begin by loading some sample data and exploring its structure.

## 1. Loading and Exploring the Data

The first step in any soundscape assessment is to load and explore your
data. For this tutorial, we’ll use a sample dataset containing
soundscape survey responses from multiple locations.

``` python
# Load the demo data
data = pd.read_excel("../../scratch/DemoData.xlsx")

# Display the first few rows
data.head()
```

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
<th data-quarto-table-cell-role="th">LocationID</th>
<th data-quarto-table-cell-role="th">SessionID</th>
<th data-quarto-table-cell-role="th">GroupID</th>
<th data-quarto-table-cell-role="th">RecordID</th>
<th data-quarto-table-cell-role="th">start_time</th>
<th data-quarto-table-cell-role="th">end_time</th>
<th data-quarto-table-cell-role="th">traffic_noise</th>
<th data-quarto-table-cell-role="th">other_noise</th>
<th data-quarto-table-cell-role="th">human_sounds</th>
<th data-quarto-table-cell-role="th">natural_sounds</th>
<th data-quarto-table-cell-role="th">...</th>
<th data-quarto-table-cell-role="th">LAeq</th>
<th data-quarto-table-cell-role="th">LA10</th>
<th data-quarto-table-cell-role="th">LA90</th>
<th data-quarto-table-cell-role="th">LCeq</th>
<th data-quarto-table-cell-role="th">N5</th>
<th data-quarto-table-cell-role="th">sharpness</th>
<th data-quarto-table-cell-role="th">roughness</th>
<th data-quarto-table-cell-role="th">tonality</th>
<th data-quarto-table-cell-role="th">ISOPleasant</th>
<th data-quarto-table-cell-role="th">ISOEventful</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>CamdenTown</td>
<td>CamdenTown1</td>
<td>CT101</td>
<td>525</td>
<td>2019-05-02 11:40:44</td>
<td>2019-05-02 11:43:24</td>
<td>4.0</td>
<td>3.0</td>
<td>3.0</td>
<td>2.0</td>
<td>...</td>
<td>69.93</td>
<td>72.43</td>
<td>64.67</td>
<td>77.85</td>
<td>30.8</td>
<td>2.09</td>
<td>0.0363</td>
<td>0.304</td>
<td>-2.196699e-01</td>
<td>0.426777</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>CamdenTown</td>
<td>CamdenTown1</td>
<td>CT101</td>
<td>526</td>
<td>2019-05-02 11:41:57</td>
<td>2019-05-02 11:44:21</td>
<td>3.0</td>
<td>1.0</td>
<td>2.0</td>
<td>1.0</td>
<td>...</td>
<td>69.93</td>
<td>72.43</td>
<td>64.67</td>
<td>77.85</td>
<td>30.8</td>
<td>2.09</td>
<td>0.0363</td>
<td>0.304</td>
<td>5.748368e-17</td>
<td>0.250000</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>CamdenTown</td>
<td>CamdenTown1</td>
<td>CT101</td>
<td>561</td>
<td>2019-05-02 11:40:44</td>
<td>2019-05-02 11:43:24</td>
<td>4.0</td>
<td>3.0</td>
<td>4.0</td>
<td>2.0</td>
<td>...</td>
<td>69.93</td>
<td>72.43</td>
<td>64.67</td>
<td>77.85</td>
<td>30.8</td>
<td>2.09</td>
<td>0.0363</td>
<td>0.304</td>
<td>-4.696699e-01</td>
<td>0.176777</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>CamdenTown</td>
<td>CamdenTown1</td>
<td>CT102</td>
<td>560</td>
<td>2019-05-02 11:50:10</td>
<td>2019-05-02 11:53:03</td>
<td>3.0</td>
<td>2.0</td>
<td>4.0</td>
<td>1.0</td>
<td>...</td>
<td>70.60</td>
<td>73.72</td>
<td>65.11</td>
<td>77.21</td>
<td>33.2</td>
<td>2.14</td>
<td>0.0430</td>
<td>0.252</td>
<td>1.035534e-01</td>
<td>-0.750000</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>CamdenTown</td>
<td>CamdenTown1</td>
<td>CT103</td>
<td>527</td>
<td>2019-05-02 11:49:06</td>
<td>2019-05-02 11:54:24</td>
<td>4.0</td>
<td>2.0</td>
<td>4.0</td>
<td>1.0</td>
<td>...</td>
<td>66.36</td>
<td>68.36</td>
<td>64.39</td>
<td>74.23</td>
<td>24.6</td>
<td>1.98</td>
<td>0.0367</td>
<td>0.184</td>
<td>2.500000e-01</td>
<td>0.750000</td>
</tr>
</tbody>
</table>

<p>5 rows × 37 columns</p>
</div>

### Display basic information about the dataset

Dataset shape:

``` python
data.shape
```

    (456, 37)

Number of locations:

``` python
data["LocationID"].nunique()
```

    4

Locations:

``` python
data["LocationID"].unique()
```

    array(['CamdenTown', 'MarchmontGarden', 'TateModern', 'PancrasLock'],
          dtype=object)

### Understanding the Data Structure

Our dataset contains several types of columns:

1. **Index Columns**: Identify the survey, location, and respondent
    - `LocationID`: Identifier for the location
    - `RecordID`: Identifier for the audio recording
    - `GroupID`: Identifier for the group of respondents
    - `SessionID`: Identifier for the survey session
2. **Perceptual Attribute Questions (PAQs)**: Ratings on a 5-point
    Likert scale
    - `PAQ1` (pleasant): How pleasant is the soundscape?
    - `PAQ2` (vibrant): How vibrant is the soundscape?
    - `PAQ3` (eventful): How eventful is the soundscape?
    - `PAQ4` (chaotic): How chaotic is the soundscape?
    - `PAQ5` (annoying): How annoying is the soundscape?
    - `PAQ6` (monotonous): How monotonous is the soundscape?
    - `PAQ7` (uneventful): How uneventful is the soundscape?
    - `PAQ8` (calm): How calm is the soundscape?
3. **Sound Source Dominance**: Ratings of how dominant different sound
    sources are
    - `traffic_noise`: Dominance of traffic noise
    - `other_noise`: Dominance of other mechanical noise
    - `human_sounds`: Dominance of human sounds
    - `natural_sounds`: Dominance of natural sounds
4. **Acoustic Metrics**: Objective measurements of the sound
    environment
    - `LAeq`: A-weighted equivalent continuous sound level
    - Various other metrics like `N5`, `Sharpness`, etc.

## 2. Data Validation and ISO Coordinate Calculation

Before analyzing the data, it’s important to validate it to ensure
quality and consistency. Soundscapy provides functions for validating
soundscape data and calculating ISO coordinates.

``` python
# Validate the data
valid_data, invalid_indices = sspy.databases.isd.validate(data)

# Display validation results
print(f"Original dataset size: {len(data)}")
print(f"Valid dataset size: {len(valid_data)}")
print(f"Number of invalid records: {len(invalid_indices) if invalid_indices else 0}")

# If there are invalid records, display the first few
if invalid_indices:
    print("\nSample of invalid records:")
    print(data.loc[invalid_indices[:5]])
```

    Original dataset size: 456
    Valid dataset size: 456
    Number of invalid records: 0

### Calculating ISO Coordinates

The ISO 12913 standard defines a circumplex model for soundscape
perception, with two main dimensions: pleasantness and eventfulness.
Soundscapy can calculate these coordinates from the PAQ responses with a
single function call.

The ISO coordinates are calculated using a trigonometric projection of
the eight PAQ responses, where: - `ISOPleasant` represents the
horizontal axis (pleasant to unpleasant) - `ISOEventful` represents the
vertical axis (eventful to uneventful)

These coordinates allow us to position each response in the soundscape
circumplex model, which has four quadrants: - Pleasant and Eventful:
“Vibrant” - Unpleasant and Eventful: “Chaotic” - Unpleasant and
Uneventful: “Monotonous” - Pleasant and Uneventful: “Calm”

``` python
# Calculate ISO coordinates if not already present
valid_data = sspy.surveys.add_iso_coords(valid_data, overwrite=True)

# Display the first few rows with ISO coordinates
print("Data with ISO coordinates:")
valid_data[["LocationID", "ISOPleasant", "ISOEventful"]].head()
```

    Data with ISO coordinates:

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
<th data-quarto-table-cell-role="th">LocationID</th>
<th data-quarto-table-cell-role="th">ISOPleasant</th>
<th data-quarto-table-cell-role="th">ISOEventful</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>CamdenTown</td>
<td>-2.196699e-01</td>
<td>0.426777</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>CamdenTown</td>
<td>5.748368e-17</td>
<td>0.250000</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>CamdenTown</td>
<td>-4.696699e-01</td>
<td>0.176777</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>CamdenTown</td>
<td>1.035534e-01</td>
<td>-0.750000</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>CamdenTown</td>
<td>2.500000e-01</td>
<td>0.750000</td>
</tr>
</tbody>
</table>

</div>

### Save the data

After using Soundscapy to calculate the ISOCoordinates, we can easily
save our resulting DataFrame out to a file. We can easily save to an
Excel file, allowing you to use the tools you’re more comfortable with
for additional analysis:

``` python
data.to_excel("SoundscapyResults.xlsx", index=False)
```

## 3. Basic Visualization and Summary Statistics

Now that we have our validated data with ISO coordinates, we can create
basic visualizations and calculate summary statistics to understand the
soundscape perceptions at different locations.

``` python
# Get summary statistics for ISO coordinates by location
iso_stats = valid_data.groupby("LocationID")[["ISOPleasant", "ISOEventful"]].agg(
    ["mean", "std", "min", "max"]
)

# Display the summary statistics
print("Summary statistics for ISO coordinates by location:")
iso_stats
```

    Summary statistics for ISO coordinates by location:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr>
<th data-quarto-table-cell-role="th"></th>
<th colspan="4" data-quarto-table-cell-role="th"
data-halign="left">ISOPleasant</th>
<th colspan="4" data-quarto-table-cell-role="th"
data-halign="left">ISOEventful</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">mean</th>
<th data-quarto-table-cell-role="th">std</th>
<th data-quarto-table-cell-role="th">min</th>
<th data-quarto-table-cell-role="th">max</th>
<th data-quarto-table-cell-role="th">mean</th>
<th data-quarto-table-cell-role="th">std</th>
<th data-quarto-table-cell-role="th">min</th>
<th data-quarto-table-cell-role="th">max</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">LocationID</th>
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
<td data-quarto-table-cell-role="th">CamdenTown</td>
<td>-0.102571</td>
<td>0.289155</td>
<td>-0.780330</td>
<td>0.603553</td>
<td>0.364096</td>
<td>0.344868</td>
<td>-0.750000</td>
<td>1.000000</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">MarchmontGarden</td>
<td>0.276438</td>
<td>0.424521</td>
<td>-0.853553</td>
<td>1.000000</td>
<td>-0.036172</td>
<td>0.258432</td>
<td>-0.500000</td>
<td>0.707107</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">PancrasLock</td>
<td>0.254416</td>
<td>0.390951</td>
<td>-0.750000</td>
<td>0.926777</td>
<td>0.078584</td>
<td>0.284963</td>
<td>-0.707107</td>
<td>0.603553</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">TateModern</td>
<td>0.380585</td>
<td>0.288983</td>
<td>-0.426777</td>
<td>1.000000</td>
<td>0.210786</td>
<td>0.308650</td>
<td>-0.676777</td>
<td>1.000000</td>
</tr>
</tbody>
</table>

</div>

### Create a circumplex scatter plot

Creating a scatter plot of the Soundscape data is extremely easy in
Soundscapy. Let’s start by plotting a single location. Again, we use
`sspy.isd.select_location_ids()` to select a single location:

``` python
sspy.scatter(sspy.isd.select_location_ids(valid_data, "CamdenTown"))
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-12-output-1.png)

#### Customizing the plot

The Soundscapy plotting functions contain several customization options,
such as changing the color, palette, point size, and title:

``` python
sspy.scatter(
    sspy.isd.select_location_ids(valid_data, "CamdenTown"),
    color="purple",
    s=40,
    title="Circumplex Scatter Plot of Soundscape Perceptions",
    xlabel="Pleasantness",
    ylabel="Eventfulness",
    diagonal_lines=True,
)
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-13-output-1.png)

#### Color by location

One of the most useful settings is the ability to split the plot by some
grouping variable. In Soundscape data, this will often be the location,
but we can easily select any other categorical variable in the dataset.
Simply set the `hue` option to get a different color for each group:

``` python
# Create a basic scatter plot of ISO coordinates
sspy.scatter(
    valid_data,
    title="Soundscape Perceptions Across All Locations",
    hue="LocationID",
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-14-output-1.png)

``` python
# Split by a different variable:

sspy.scatter(...)
```

    TypeError: Data source must be a DataFrame or Mapping, not <class 'ellipsis'>.
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mTypeError[0m                                 Traceback (most recent call last)
    Cell [0;32mIn[15], line 3[0m
    [1;32m      1[0m [38;5;66;03m# Split by a different variable:[39;00m
    [0;32m----> 3[0m [43msspy[49m[38;5;241;43m.[39;49m[43mscatter[49m[43m([49m[38;5;241;43m.[39;49m[38;5;241;43m.[39;49m[38;5;241;43m.[39;49m[43m)[49m

    File [0;32m~/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:750[0m, in [0;36mscatter[0;34m(data, title, ax, x, y, hue, palette, legend, prim_labels, **kwargs)[0m
    [1;32m    747[0m [38;5;28;01mif[39;00m ax [38;5;129;01mis[39;00m [38;5;28;01mNone[39;00m:
    [1;32m    748[0m     _, ax [38;5;241m=[39m plt[38;5;241m.[39msubplots([38;5;241m1[39m, [38;5;241m1[39m, figsize[38;5;241m=[39msubplots_args[38;5;241m.[39mfigsize)
    [0;32m--> 750[0m p [38;5;241m=[39m [43msns[49m[38;5;241;43m.[39;49m[43mscatterplot[49m[43m([49m[43max[49m[38;5;241;43m=[39;49m[43max[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mscatter_args[49m[38;5;241;43m.[39;49m[43mas_dict[49m[43m([49m[43m)[49m[43m)[49m
    [1;32m    752[0m _set_style()
    [1;32m    753[0m _circumplex_grid(
    [1;32m    754[0m     ax[38;5;241m=[39max,
    [1;32m    755[0m     [38;5;241m*[39m[38;5;241m*[39mstyle_args[38;5;241m.[39mget_multiple(
    [1;32m    756[0m         [[38;5;124m"[39m[38;5;124mxlim[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mylim[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mxlabel[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mylabel[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mdiagonal_lines[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mprim_ax_fontdict[39m[38;5;124m"[39m]
    [1;32m    757[0m     ),
    [1;32m    758[0m )

    File [0;32m~/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/seaborn/relational.py:615[0m, in [0;36mscatterplot[0;34m(data, x, y, hue, size, style, palette, hue_order, hue_norm, sizes, size_order, size_norm, markers, style_order, legend, ax, **kwargs)[0m
    [1;32m    606[0m [38;5;28;01mdef[39;00m[38;5;250m [39m[38;5;21mscatterplot[39m(
    [1;32m    607[0m     data[38;5;241m=[39m[38;5;28;01mNone[39;00m, [38;5;241m*[39m,
    [1;32m    608[0m     x[38;5;241m=[39m[38;5;28;01mNone[39;00m, y[38;5;241m=[39m[38;5;28;01mNone[39;00m, hue[38;5;241m=[39m[38;5;28;01mNone[39;00m, size[38;5;241m=[39m[38;5;28;01mNone[39;00m, style[38;5;241m=[39m[38;5;28;01mNone[39;00m,
    [0;32m   (...)[0m
    [1;32m    612[0m     [38;5;241m*[39m[38;5;241m*[39mkwargs
    [1;32m    613[0m ):
    [0;32m--> 615[0m     p [38;5;241m=[39m [43m_ScatterPlotter[49m[43m([49m
    [1;32m    616[0m [43m        [49m[43mdata[49m[38;5;241;43m=[39;49m[43mdata[49m[43m,[49m
    [1;32m    617[0m [43m        [49m[43mvariables[49m[38;5;241;43m=[39;49m[38;5;28;43mdict[39;49m[43m([49m[43mx[49m[38;5;241;43m=[39;49m[43mx[49m[43m,[49m[43m [49m[43my[49m[38;5;241;43m=[39;49m[43my[49m[43m,[49m[43m [49m[43mhue[49m[38;5;241;43m=[39;49m[43mhue[49m[43m,[49m[43m [49m[43msize[49m[38;5;241;43m=[39;49m[43msize[49m[43m,[49m[43m [49m[43mstyle[49m[38;5;241;43m=[39;49m[43mstyle[49m[43m)[49m[43m,[49m
    [1;32m    618[0m [43m        [49m[43mlegend[49m[38;5;241;43m=[39;49m[43mlegend[49m
    [1;32m    619[0m [43m    [49m[43m)[49m
    [1;32m    621[0m     p[38;5;241m.[39mmap_hue(palette[38;5;241m=[39mpalette, order[38;5;241m=[39mhue_order, norm[38;5;241m=[39mhue_norm)
    [1;32m    622[0m     p[38;5;241m.[39mmap_size(sizes[38;5;241m=[39msizes, order[38;5;241m=[39msize_order, norm[38;5;241m=[39msize_norm)

    File [0;32m~/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/seaborn/relational.py:396[0m, in [0;36m_ScatterPlotter.__init__[0;34m(self, data, variables, legend)[0m
    [1;32m    387[0m [38;5;28;01mdef[39;00m[38;5;250m [39m[38;5;21m__init__[39m([38;5;28mself[39m, [38;5;241m*[39m, data[38;5;241m=[39m[38;5;28;01mNone[39;00m, variables[38;5;241m=[39m{}, legend[38;5;241m=[39m[38;5;28;01mNone[39;00m):
    [1;32m    388[0m
    [1;32m    389[0m     [38;5;66;03m# TODO this is messy, we want the mapping to be agnostic about[39;00m
    [1;32m    390[0m     [38;5;66;03m# the kind of plot to draw, but for the time being we need to set[39;00m
    [1;32m    391[0m     [38;5;66;03m# this information so the SizeMapping can use it[39;00m
    [1;32m    392[0m     [38;5;28mself[39m[38;5;241m.[39m_default_size_range [38;5;241m=[39m (
    [1;32m    393[0m         np[38;5;241m.[39mr_[[38;5;241m.5[39m, [38;5;241m2[39m] [38;5;241m*[39m np[38;5;241m.[39msquare(mpl[38;5;241m.[39mrcParams[[38;5;124m"[39m[38;5;124mlines.markersize[39m[38;5;124m"[39m])
    [1;32m    394[0m     )
    [0;32m--> 396[0m     [38;5;28;43msuper[39;49m[43m([49m[43m)[49m[38;5;241;43m.[39;49m[38;5;21;43m__init__[39;49m[43m([49m[43mdata[49m[38;5;241;43m=[39;49m[43mdata[49m[43m,[49m[43m [49m[43mvariables[49m[38;5;241;43m=[39;49m[43mvariables[49m[43m)[49m
    [1;32m    398[0m     [38;5;28mself[39m[38;5;241m.[39mlegend [38;5;241m=[39m legend

    File [0;32m~/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/seaborn/_base.py:634[0m, in [0;36mVectorPlotter.__init__[0;34m(self, data, variables)[0m
    [1;32m    629[0m [38;5;66;03m# var_ordered is relevant only for categorical axis variables, and may[39;00m
    [1;32m    630[0m [38;5;66;03m# be better handled by an internal axis information object that tracks[39;00m
    [1;32m    631[0m [38;5;66;03m# such information and is set up by the scale_* methods. The analogous[39;00m
    [1;32m    632[0m [38;5;66;03m# information for numeric axes would be information about log scales.[39;00m
    [1;32m    633[0m [38;5;28mself[39m[38;5;241m.[39m_var_ordered [38;5;241m=[39m {[38;5;124m"[39m[38;5;124mx[39m[38;5;124m"[39m: [38;5;28;01mFalse[39;00m, [38;5;124m"[39m[38;5;124my[39m[38;5;124m"[39m: [38;5;28;01mFalse[39;00m}  [38;5;66;03m# alt., used DefaultDict[39;00m
    [0;32m--> 634[0m [38;5;28;43mself[39;49m[38;5;241;43m.[39;49m[43massign_variables[49m[43m([49m[43mdata[49m[43m,[49m[43m [49m[43mvariables[49m[43m)[49m
    [1;32m    636[0m [38;5;66;03m# TODO Lots of tests assume that these are called to initialize the[39;00m
    [1;32m    637[0m [38;5;66;03m# mappings to default values on class initialization. I'd prefer to[39;00m
    [1;32m    638[0m [38;5;66;03m# move away from that and only have a mapping when explicitly called.[39;00m
    [1;32m    639[0m [38;5;28;01mfor[39;00m var [38;5;129;01min[39;00m [[38;5;124m"[39m[38;5;124mhue[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124msize[39m[38;5;124m"[39m, [38;5;124m"[39m[38;5;124mstyle[39m[38;5;124m"[39m]:

    File [0;32m~/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/seaborn/_base.py:679[0m, in [0;36mVectorPlotter.assign_variables[0;34m(self, data, variables)[0m
    [1;32m    674[0m [38;5;28;01melse[39;00m:
    [1;32m    675[0m     [38;5;66;03m# When dealing with long-form input, use the newer PlotData[39;00m
    [1;32m    676[0m     [38;5;66;03m# object (internal but introduced for the objects interface)[39;00m
    [1;32m    677[0m     [38;5;66;03m# to centralize / standardize data consumption logic.[39;00m
    [1;32m    678[0m     [38;5;28mself[39m[38;5;241m.[39minput_format [38;5;241m=[39m [38;5;124m"[39m[38;5;124mlong[39m[38;5;124m"[39m
    [0;32m--> 679[0m     plot_data [38;5;241m=[39m [43mPlotData[49m[43m([49m[43mdata[49m[43m,[49m[43m [49m[43mvariables[49m[43m)[49m
    [1;32m    680[0m     frame [38;5;241m=[39m plot_data[38;5;241m.[39mframe
    [1;32m    681[0m     names [38;5;241m=[39m plot_data[38;5;241m.[39mnames

    File [0;32m~/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/seaborn/_core/data.py:57[0m, in [0;36mPlotData.__init__[0;34m(self, data, variables)[0m
    [1;32m     51[0m [38;5;28;01mdef[39;00m[38;5;250m [39m[38;5;21m__init__[39m(
    [1;32m     52[0m     [38;5;28mself[39m,
    [1;32m     53[0m     data: DataSource,
    [1;32m     54[0m     variables: [38;5;28mdict[39m[[38;5;28mstr[39m, VariableSpec],
    [1;32m     55[0m ):
    [0;32m---> 57[0m     data [38;5;241m=[39m [43mhandle_data_source[49m[43m([49m[43mdata[49m[43m)[49m
    [1;32m     58[0m     frame, names, ids [38;5;241m=[39m [38;5;28mself[39m[38;5;241m.[39m_assign_variables(data, variables)
    [1;32m     60[0m     [38;5;28mself[39m[38;5;241m.[39mframe [38;5;241m=[39m frame

    File [0;32m~/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/seaborn/_core/data.py:278[0m, in [0;36mhandle_data_source[0;34m(data)[0m
    [1;32m    276[0m [38;5;28;01melif[39;00m data [38;5;129;01mis[39;00m [38;5;129;01mnot[39;00m [38;5;28;01mNone[39;00m [38;5;129;01mand[39;00m [38;5;129;01mnot[39;00m [38;5;28misinstance[39m(data, Mapping):
    [1;32m    277[0m     err [38;5;241m=[39m [38;5;124mf[39m[38;5;124m"[39m[38;5;124mData source must be a DataFrame or Mapping, not [39m[38;5;132;01m{[39;00m[38;5;28mtype[39m(data)[38;5;132;01m!r}[39;00m[38;5;124m.[39m[38;5;124m"[39m
    [0;32m--> 278[0m     [38;5;28;01mraise[39;00m [38;5;167;01mTypeError[39;00m(err)
    [1;32m    280[0m [38;5;28;01mreturn[39;00m data

    [0;31mTypeError[0m: Data source must be a DataFrame or Mapping, not <class 'ellipsis'>.

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-15-output-2.png)

## Density Plots

Of course, the most notable plot type in Soundscapy is the Density plot.
Just like scatter, this has its own simple function. Try it out:

``` python
# Create a basic density plot
sspy.density(...)
plt.show()
```

    TypeError: object of type 'ellipsis' has no len()
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mTypeError[0m                                 Traceback (most recent call last)
    Cell [0;32mIn[16], line 2[0m
    [1;32m      1[0m [38;5;66;03m# Create a basic density plot[39;00m
    [0;32m----> 2[0m [43msspy[49m[38;5;241;43m.[39;49m[43mdensity[49m[43m([49m[38;5;241;43m.[39;49m[38;5;241;43m.[39;49m[38;5;241;43m.[39;49m[43m)[49m
    [1;32m      3[0m plt[38;5;241m.[39mshow()

    File [0;32m~/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:958[0m, in [0;36mdensity[0;34m(data, title, ax, x, y, hue, incl_scatter, density_type, palette, scatter_kws, legend, prim_labels, **kwargs)[0m
    [1;32m    946[0m density_args [38;5;241m=[39m _setup_density_params(
    [1;32m    947[0m     data[38;5;241m=[39mdata,
    [1;32m    948[0m     x[38;5;241m=[39mx,
    [0;32m   (...)[0m
    [1;32m    954[0m     [38;5;241m*[39m[38;5;241m*[39mkwargs,
    [1;32m    955[0m )
    [1;32m    957[0m [38;5;66;03m# Check if dataset is large enough for density plots[39;00m
    [0;32m--> 958[0m [43m_valid_density[49m[43m([49m[43mdata[49m[43m)[49m
    [1;32m    960[0m [38;5;28;01mif[39;00m ax [38;5;129;01mis[39;00m [38;5;28;01mNone[39;00m:
    [1;32m    961[0m     _, ax [38;5;241m=[39m plt[38;5;241m.[39msubplots([38;5;241m1[39m, [38;5;241m1[39m, figsize[38;5;241m=[39msubplots_args[38;5;241m.[39mget([38;5;124m"[39m[38;5;124mfigsize[39m[38;5;124m"[39m))

    File [0;32m~/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:2020[0m, in [0;36m_valid_density[0;34m(data)[0m
    [1;32m   2003[0m [38;5;28;01mdef[39;00m[38;5;250m [39m[38;5;21m_valid_density[39m(data: pd[38;5;241m.[39mDataFrame) [38;5;241m-[39m[38;5;241m>[39m [38;5;28;01mNone[39;00m:
    [1;32m   2004[0m [38;5;250m    [39m[38;5;124;03m"""[39;00m
    [1;32m   2005[0m [38;5;124;03m    Check if the data is valid for density plots.[39;00m
    [1;32m   2006[0m
    [0;32m   (...)[0m
    [1;32m   2018[0m
    [1;32m   2019[0m [38;5;124;03m    """[39;00m
    [0;32m-> 2020[0m     [38;5;28;01mif[39;00m [38;5;28;43mlen[39;49m[43m([49m[43mdata[49m[43m)[49m [38;5;241m<[39m RECOMMENDED_MIN_SAMPLES:
    [1;32m   2021[0m         warnings[38;5;241m.[39mwarn(
    [1;32m   2022[0m             [38;5;124m"[39m[38;5;124mDensity plots are not recommended for [39m[38;5;124m"[39m
    [1;32m   2023[0m             [38;5;124mf[39m[38;5;124m"[39m[38;5;124msmall datasets (<[39m[38;5;132;01m{[39;00mRECOMMENDED_MIN_SAMPLES[38;5;132;01m}[39;00m[38;5;124m samples).[39m[38;5;124m"[39m,
    [1;32m   2024[0m             [38;5;167;01mUserWarning[39;00m,
    [1;32m   2025[0m             stacklevel[38;5;241m=[39m[38;5;241m2[39m,
    [1;32m   2026[0m         )

    [0;31mTypeError[0m: object of type 'ellipsis' has no len()

## 4. Creating Likert Style Plots

Likert plots are a useful way to visualize the distribution of responses
to Likert scale questions, such as the PAQs in our survey. Soundscapy
integrates with the `plot_likert` package to create these
visualizations.

We can use Soundscapy’s `paq_likert` plot to examine the distribution of
PAQ responses:

``` python
from soundscapy import PAQ_IDS

sspy.paq_likert(data[PAQ_IDS], title="Distribution of PAQ Responses in the Demo Data")
```

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/2117978073.py:3: ExperimentalWarning: This is an experimental function. It may change in the future.
      sspy.paq_likert(data[PAQ_IDS], title="Distribution of PAQ Responses in the Demo Data")
    /Users/mitch/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/plot_likert/plot_likert.py:257: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df.applymap(validate)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-17-output-2.png)

To get a better view of the data, we can also split it across the
Locations. To do this, we use the `select_location_ids()` function from
Soundscapy, and use a `for` loop to create a separate plot for each
location.

``` python
for location in data["LocationID"].unique():
    sspy.paq_likert(
        sspy.databases.isd.select_location_ids(data, location)[PAQ_IDS],
        title=f"Distribution of PAQ Responses in {location}",
    )
    plt.show()
```

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/2426594973.py:2: ExperimentalWarning: This is an experimental function. It may change in the future.
      sspy.paq_likert(
    /Users/mitch/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/plot_likert/plot_likert.py:257: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df.applymap(validate)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-18-output-2.png)

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/2426594973.py:2: ExperimentalWarning: This is an experimental function. It may change in the future.
      sspy.paq_likert(
    /Users/mitch/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/plot_likert/plot_likert.py:257: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df.applymap(validate)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-18-output-4.png)

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/2426594973.py:2: ExperimentalWarning: This is an experimental function. It may change in the future.
      sspy.paq_likert(
    /Users/mitch/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/plot_likert/plot_likert.py:257: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df.applymap(validate)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-18-output-6.png)

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/2426594973.py:2: ExperimentalWarning: This is an experimental function. It may change in the future.
      sspy.paq_likert(
    /Users/mitch/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/plot_likert/plot_likert.py:257: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df.applymap(validate)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-18-output-8.png)

### Analyzing Sound Source Dominance

Let’s examine the dominance of different sound sources at each location.

``` python
sspy.stacked_likert(valid_data, "traffic_noise", title="Traffic Noise Dominance")
```

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/2060333350.py:1: ExperimentalWarning: This is an experimental function. It may change in the future. Currently, this functio applies brute data cleaning, use with caution.
      sspy.stacked_likert(valid_data, "traffic_noise", title="Traffic Noise Dominance")
    /Users/mitch/Documents/GitHub/Soundscapy/.venv/lib/python3.12/site-packages/plot_likert/plot_likert.py:257: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df.applymap(validate)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-19-output-2.png)

Try this out with some of the other Likert scaled data:

``` python
sspy.stacked_likert(...)
```

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/3195878422.py:1: ExperimentalWarning: This is an experimental function. It may change in the future. Currently, this functio applies brute data cleaning, use with caution.
      sspy.stacked_likert(...)

    TypeError: 'ellipsis' object is not subscriptable
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mTypeError[0m                                 Traceback (most recent call last)
    Cell [0;32mIn[19], line 1[0m
    [0;32m----> 1[0m [43msspy[49m[38;5;241;43m.[39;49m[43mstacked_likert[49m[43m([49m[38;5;241;43m.[39;49m[38;5;241;43m.[39;49m[38;5;241;43m.[39;49m[43m)[49m

    File [0;32m~/Documents/GitHub/Soundscapy/src/soundscapy/plotting/likert.py:282[0m, in [0;36mstacked_likert[0;34m(data, column, title, legend, ax, plot_percentage, bar_labels, **kwargs)[0m
    [1;32m    264[0m [38;5;28;01mdef[39;00m[38;5;250m [39m[38;5;21mstacked_likert[39m(
    [1;32m    265[0m     data: pd[38;5;241m.[39mDataFrame,
    [1;32m    266[0m     column: [38;5;28mstr[39m [38;5;241m=[39m [38;5;124m"[39m[38;5;124mappropriate[39m[38;5;124m"[39m,
    [0;32m   (...)[0m
    [1;32m    273[0m     [38;5;241m*[39m[38;5;241m*[39mkwargs,
    [1;32m    274[0m ) [38;5;241m-[39m[38;5;241m>[39m [38;5;28;01mNone[39;00m:
    [1;32m    275[0m     warnings[38;5;241m.[39mwarn(
    [1;32m    276[0m         [38;5;124m"[39m[38;5;124mThis is an experimental function. It may change in the future. [39m[38;5;124m"[39m
    [1;32m    277[0m         [38;5;124m"[39m[38;5;124mCurrently, this functio applies brute data cleaning, use with caution. [39m[38;5;124m"[39m,
    [1;32m    278[0m         ExperimentalWarning,
    [1;32m    279[0m         stacklevel[38;5;241m=[39m[38;5;241m2[39m,
    [1;32m    280[0m     )
    [0;32m--> 282[0m     new_data [38;5;241m=[39m [43mdata[49m[43m[[49m[43mcolumn[49m[43m][49m[38;5;241m.[39mcopy()
    [1;32m    283[0m     new_data [38;5;241m=[39m new_data[38;5;241m.[39mdropna()
    [1;32m    285[0m     new_data [38;5;241m=[39m likert_categorical_from_data(new_data)  [38;5;66;03m# type: ignore[39;00m

    [0;31mTypeError[0m: 'ellipsis' object is not subscriptable

In addition to the more specialised Likert plots provided by Soundscapy,
we can of course apply more common analyses. For this, we need to do
some data processing and plotting in Pandas and Seaborn:

``` python
# Calculate mean sound source dominance by location
sound_sources = (
    valid_data.groupby("LocationID")[
        ["traffic_noise", "other_noise", "human_sounds", "natural_sounds"]
    ]
    .mean()
    .round(2)
)
```

#### Mean sound source dominance by location

``` python
sound_sources
```

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
<th data-quarto-table-cell-role="th">traffic_noise</th>
<th data-quarto-table-cell-role="th">other_noise</th>
<th data-quarto-table-cell-role="th">human_sounds</th>
<th data-quarto-table-cell-role="th">natural_sounds</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">LocationID</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">CamdenTown</td>
<td>3.77</td>
<td>2.68</td>
<td>3.27</td>
<td>1.34</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">MarchmontGarden</td>
<td>2.66</td>
<td>2.46</td>
<td>2.67</td>
<td>2.59</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">PancrasLock</td>
<td>2.42</td>
<td>3.28</td>
<td>2.48</td>
<td>2.39</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">TateModern</td>
<td>2.52</td>
<td>2.13</td>
<td>3.64</td>
<td>2.57</td>
</tr>
</tbody>
</table>

</div>

Since Soundscapy doesn’t implement its own functionality for the more
common plots, we fall back to the very nice Seaborn plotting library,
which we imported as `sns` to create a barplot of the mean sound source
responses (or you can of course do this in something like Excel):

``` python
# Create a bar chart
sound_sources_plot = sound_sources.reset_index().melt(
    id_vars=["LocationID"], var_name="Source", value_name="Dominance"
)

sns.barplot(
    data=sound_sources_plot,
    x="Source",
    y="Dominance",
    hue="LocationID",
    palette="colorblind",
)
plt.title("Sound Source Dominance by Location")
plt.xlabel("Sound Source")
plt.ylabel("Mean Dominance Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-23-output-1.png)

## 5. Creating Complex Plots with Hue and Subplots

Soundscapy makes it easy to create more complex visualizations that show
relationships between different variables. Let’s explore how sound
source dominance affects soundscape perception:

``` python
# Create subplots showing the impact of natural sounds on soundscape perception
sspy.create_iso_subplots(
    data=valid_data,
    subplot_by="LocationID",
    hue="natural_sounds",
    plot_layers=["scatter", "simple_density"],
    title="Impact of Natural Sounds on Soundscape Perception",
)
plt.tight_layout()
plt.show()
```

    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:986: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:987: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.to_outline().as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:986: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:987: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.to_outline().as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:986: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:987: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.to_outline().as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:986: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:987: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.to_outline().as_seaborn_kwargs())

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-24-output-2.png)

``` python
# Create subplots showing the impact of traffic noise on soundscape perception
sspy.create_iso_subplots(
    data=valid_data,
    subplot_by="LocationID",
    hue="traffic_noise",
    plot_layers=["scatter", "simple_density"],
    title="Impact of Traffic Noise on Soundscape Perception",
)
plt.tight_layout()
plt.show()
```

    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:986: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.as_seaborn_kwargs())
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/plot_functions.py:987: UserWarning: KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.
      d = sns.kdeplot(ax=ax, **density_args.to_outline().as_seaborn_kwargs())

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-25-output-2.png)

We can also examine the relationship between acoustic metrics and
soundscape perception:

``` python
# Create a scatter plot with regression line showing the relationship between LAeq and ISOPleasant
sns.lmplot(
    data=valid_data,
    x="LAeq",
    y="ISOPleasant",
    hue="LocationID",
    palette="colorblind",
    height=6,
    aspect=1.5,
)
plt.title("Relationship between Sound Level (LAeq) and Pleasantness")
plt.tight_layout()
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-26-output-1.png)

``` python
# Create a scatter plot with regression line showing the relationship between N5 and ISOEventful
sns.lmplot(
    data=valid_data,
    x="N5",
    y="ISOEventful",
    hue="LocationID",
    palette="colorblind",
    height=6,
    aspect=1.5,
)
plt.title("Relationship between Loudness (N5) and Eventfulness")
plt.tight_layout()
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-27-output-1.png)

## 6. Applying SPI Analysis

The Soundscape Perception Index (SPI) is a powerful tool for comparing
soundscapes to target distributions. It quantifies the similarity
between two soundscape distributions on a scale from 0 to 100, where 100
indicates perfect similarity.

Let’s define some target distributions and calculate SPI scores for our
locations:

``` python
# Define a "tranquil" target distribution
tranquil_target = DirectParams(
    xi=np.array([[0.8, -0.5]]),  # Pleasant and uneventful
    omega=np.array([[0.17, -0.04], [-0.04, 0.09]]),
    alpha=np.array([-8, 1]),
)

# Create a MultiSkewNorm instance from the parameters
tranquil_msn = MultiSkewNorm.from_params(tranquil_target)

# Generate sample data
tranquil_msn.sample(1000)

# Convert to DataFrame for visualization
tranquil_df = pd.DataFrame(
    tranquil_msn.sample_data, columns=["ISOPleasant", "ISOEventful"]
)

# Visualize the target distribution
plt.figure(figsize=(8, 8))
sspy.density(tranquil_df, title="Tranquil Target Distribution", color="red")
plt.show()
```

    <Figure size 800x800 with 0 Axes>

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-28-output-2.png)

``` python
# Define a "vibrant" target distribution
vibrant_target = DirectParams(
    xi=np.array([[0.7, 0.6]]),  # Pleasant and eventful
    omega=np.array([[0.15, 0.03], [0.03, 0.15]]),
    alpha=np.array([0, 0]),
)

# Create a MultiSkewNorm instance from the parameters
vibrant_msn = MultiSkewNorm.from_params(vibrant_target)

# Generate sample data
vibrant_msn.sample(1000)

# Convert to DataFrame for visualization
vibrant_df = pd.DataFrame(
    vibrant_msn.sample_data, columns=["ISOPleasant", "ISOEventful"]
)

# Visualize the target distribution
sspy.density(
    vibrant_df,
    title="Vibrant Target Distribution",
    color="purple",
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-29-output-1.png)

Now let’s calculate SPI scores for each location against both target
distributions:

``` python
# Calculate SPI scores for each location against the tranquil target
locations = valid_data["LocationID"].unique()
tranquil_spi_scores = {}
vibrant_spi_scores = {}

for location in locations:
    # Get data for this location
    location_data = sspy.databases.isd.select_location_ids(valid_data, location)

    # Calculate SPI against tranquil target
    tranquil_spi_scores[location] = tranquil_msn.spi_score(
        location_data[["ISOPleasant", "ISOEventful"]]
    )

    # Calculate SPI against vibrant target
    vibrant_spi_scores[location] = vibrant_msn.spi_score(
        location_data[["ISOPleasant", "ISOEventful"]]
    )

# Display the results
spi_results = pd.DataFrame(
    {
        "Tranquil SPI": tranquil_spi_scores,
        "Vibrant SPI": vibrant_spi_scores,
    }
).T

print("SPI Scores by Location:")
print(spi_results)

# Create a bar chart of SPI scores
plt.figure(figsize=(12, 6))
spi_results.plot(kind="bar", colormap="viridis")
plt.title("SPI Scores by Location")
plt.xlabel("Target Distribution")
plt.ylabel("SPI Score")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend(title="Location")
plt.tight_layout()
plt.show()
```

    SPI Scores by Location:
                  CamdenTown  MarchmontGarden  TateModern  PancrasLock
    Tranquil SPI          14               45          29           40
    Vibrant SPI           23               29          50           37

    <Figure size 1200x600 with 0 Axes>

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-30-output-3.png)

## 7. Using the ISOPlot Interface for SPI Analysis

Soundscapy’s `ISOPlot` class provides a more sophisticated interface for
creating SPI visualizations. Let’s use it to create a comprehensive SPI
analysis:

``` python
# Create an SPI plot comparing locations against the tranquil target
tranquil_plot = (
    ISOPlot(
        data=valid_data,
        title="Comparing Locations Against Tranquil Target",
    )
    .create_subplots(
        subplot_by="LocationID",
        figsize=(4, 4),
        auto_allocate_axes=True,
    )
    .add_scatter()
    .add_simple_density(fill=True)
    .add_spi(spi_target_data=tranquil_df, show_score="on axis")
    .style(legend_loc=False)
)
```

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/1322741312.py:3: ExperimentalWarning: `ISOPlot` is currently under development and should be considered experimental. `ISOPlot` implements an experimental API for creating layered soundscape circumplex plots. Use with caution.
      ISOPlot(
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/iso_plot.py:502: UserWarning: This is an experimental feature. The number of rows and columns may not be optimal.
      self._allocate_subplot_axes(subplot_titles)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-31-output-2.png)

``` python
# Create an SPI plot comparing locations against the vibrant target
vibrant_plot = (
    ISOPlot(
        data=valid_data,
        title="Comparing Locations Against Vibrant Target",
    )
    .create_subplots(
        subplot_by="LocationID",
        figsize=(4, 4),
        auto_allocate_axes=True,
    )
    .add_scatter()
    .add_simple_density(fill=True)
    .add_spi(spi_target_data=vibrant_df, show_score="on axis")
    .style(legend_loc=False)
)
```

    /var/folders/6t/7h8wn9n92w5f24ml_bkwck9m0000gn/T/ipykernel_74288/4209348228.py:3: ExperimentalWarning: `ISOPlot` is currently under development and should be considered experimental. `ISOPlot` implements an experimental API for creating layered soundscape circumplex plots. Use with caution.
      ISOPlot(
    /Users/mitch/Documents/GitHub/Soundscapy/src/soundscapy/plotting/iso_plot.py:502: UserWarning: This is an experimental feature. The number of rows and columns may not be optimal.
      self._allocate_subplot_axes(subplot_titles)

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-32-output-2.png)

## 8. Interpreting the Results

Now that we’ve analyzed our data and created various visualizations,
let’s interpret the results:

1. **Location Characteristics**:
    - Each location has a distinct soundscape character, as shown by its
      position in the circumplex model.
    - Some locations are more pleasant (higher ISOPleasant values),
      while others are more eventful (higher ISOEventful values).
2. **Sound Source Influence**:
    - Natural sounds tend to increase pleasantness, as shown by the
      relationship between natural sound dominance and ISOPleasant
      values.
    - Traffic noise tends to decrease pleasantness, as shown by the
      relationship between traffic noise dominance and ISOPleasant
      values.
3. **Acoustic Metrics**:
    - Higher sound levels (LAeq) are generally associated with lower
      pleasantness.
    - Higher loudness (N5) is generally associated with higher
      eventfulness.
4. **SPI Analysis**:
    - Some locations match better with the tranquil target, while others
      match better with the vibrant target.
    - This information can be used to identify which locations provide
      the desired soundscape experience.

These insights can inform soundscape design and management decisions,
such as: - Which locations to preserve or enhance for specific
soundscape experiences - Which sound sources to promote or mitigate at
different locations - How to design new spaces to achieve desired
soundscape characteristics

## 9. Gallery of Soundscapy Visualizations

Soundscapy offers a wide range of visualization options. Here’s a
gallery of additional plots you can create:

``` python
# Basic scatter plot
sspy.scatter(
    valid_data,
    title="Basic Scatter Plot",
    diagonal_lines=True,
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-33-output-1.png)

``` python
# Density plot
sspy.density(
    valid_data,
    title="Density Plot",
    diagonal_lines=True,
    fill=True,
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-34-output-1.png)

``` python
# Combined scatter and density plot
sspy.iso_plot(
    valid_data,
    title="Combined Scatter and Density Plot",
    plot_layers=["scatter", "density"],
    diagonal_lines=True,
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-35-output-1.png)

``` python
# Simple density plot with hue
sspy.density(
    valid_data,
    title="Simple Density Plot with Hue",
    density_type="simple",
    hue="LocationID",
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-36-output-1.png)

``` python
# Joint plot
sspy.jointplot(
    valid_data,
    title="Joint Plot",
)
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-37-output-1.png)

``` python
# Joint plot with histogram marginals
plt.figure(figsize=(10, 10))
sspy.jointplot(
    valid_data, title="Joint Plot with Histogram Marginals", marginal_kind="hist"
)
plt.show()
```

    <Figure size 1000x1000 with 0 Axes>

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-38-output-2.png)

``` python
# Joint plot with grouping
plt.figure(figsize=(10, 10))
sspy.jointplot(
    valid_data,
    title="Joint Plot with Grouping",
    hue="LocationID",
    density_type="simple",
)
plt.show()
```

    <Figure size 1000x1000 with 0 Axes>

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-39-output-2.png)

``` python
# Custom multi-panel visualization
fig = plt.figure(figsize=(15, 14))

# Create a 2x2 grid
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# Create a scatter plot of all locations in the first subplot
ax1 = fig.add_subplot(gs[0, 0])
sspy.scatter(
    valid_data,
    title="All Locations - Scatter",
    hue="LocationID",
    ax=ax1,
)

# Create a density plot of all locations in the second subplot
ax2 = fig.add_subplot(gs[0, 1])
sspy.density(
    valid_data,
    title="All Locations - Density",
    hue="LocationID",
    density_type="simple",
    incl_scatter=False,
    fill=False,
    ax=ax2,
)

# Create a scatter plot colored by LAeq in the third subplot
ax3 = fig.add_subplot(gs[1, 0])
sspy.scatter(
    valid_data,
    title="Sound Level (LAeq)",
    hue="LAeq",
    palette="viridis",
    ax=ax3,
)

# Create a density plot colored by natural sounds in the fourth subplot
ax4 = fig.add_subplot(gs[1, 1])
sspy.density(
    valid_data,
    title="Natural Sounds Dominance",
    hue="natural_sounds",
    density_type="simple",
    ax=ax4,
)

plt.tight_layout()
plt.show()
```

![](6_Soundscape_Assessment_Tutorial_files/figure-markdown_strict/cell-40-output-1.png)

## 10. Summary

In this tutorial, you’ve learned how to:

1. **Load and validate soundscape survey data**
    - Import data from Excel files
    - Validate data quality
    - Calculate ISO coordinates
2. **Create basic visualizations and summary statistics**
    - Scatter plots and density plots
    - Summary statistics by location
    - Sound source dominance analysis
3. **Create Likert style plots**
    - Convert numeric responses to categorical
    - Create side-by-side Likert plots for comparison
4. **Create complex plots with hue and subplots**
    - Visualize relationships between variables
    - Create multi-panel visualizations
    - Analyze the impact of sound sources on perception
5. **Apply SPI analysis**
    - Define target distributions
    - Calculate SPI scores
    - Compare locations against different targets
6. **Use the ISOPlot interface**
    - Create sophisticated SPI visualizations
    - Combine multiple layers in a single plot
    - Customize plot appearance

These skills will enable you to conduct comprehensive soundscape
assessments and communicate your findings effectively. By understanding
how people perceive and experience soundscapes, you can contribute to
the design and management of more pleasant and appropriate acoustic
environments.

## References

1. ISO 12913-1:2014. Acoustics — Soundscape — Part 1: Definition and
    conceptual framework.
2. ISO 12913-2:2018. Acoustics — Soundscape — Part 2: Data collection
    and reporting requirements.
3. ISO 12913-3:2019. Acoustics — Soundscape — Part 3: Data analysis.
4. Mitchell, A., Aletta, F., & Kang, J. (2022). How to analyse and
    represent quantitative soundscape data. JASA Express Letters,
    2, 37201. <https://doi.org/10.1121/10.0009794>
