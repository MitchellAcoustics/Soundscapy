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
#     display_name: Python 3.7.12 ('soundscapy-python3.7')
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# `Soundscapy` - Quick Start Guide

By Andrew Mitchell, Research Fellow, University College London

## Background

`Soundscapy` is a python toolbox for analysing quantitative soundscape data. Urban soundscapes are typically assessed through surveys which ask respondents how they perceive the given soundscape. Particularly when collected following the technical specification ISO 12913, these surveys can constitute quantitative data about the soundscape perception. As proposed in *How to analyse and represent quantitative soundscape data* [(Mitchell, Aletta, & Kang, 2022)](https://asa.scitation.org/doi/full/10.1121/10.0009794), in order to describe the soundscape perception of a group or of a location, we should consider the distribution of responses. `Soundscapy`'s approach to soundscape analysis follows this approach and makes it simple to process soundscape data and visualise the distribution of responses. 

For more information on the theory underlying the assessments and forms of data collection, please see ISO 12913-Part 2, *The SSID Protocol* [(Mitchell, *et al.*, 2020)](https://www.mdpi.com/2076-3417/10/7/2397), and *How to analyse and represent quantitative soundscape data*.

## This Notebook

The purpose of this notebook is to give a brief overview of how `Soundscapy` works and how to quickly get started using it to analyse your own soundscape data. The example dataset used is *The International Soundscape Database (ISD)* (Mitchell, *et al.*, 2021), which is publicly available at [Zenodo](https://zenodo.org/record/6331810) and is free to use. `Soundscapy` expects data to follow the format used in the ISD, but can be adapted for similar datasets.

----------

## Installation

`Soundscapy` is currently under active development and has not yet been released on PyPI. However, it is fairly straightforward to use `pip` to install directly from the Github page, or you can clone the git repo. To install with `pip`:

First, download the environment.yml and create the necessary conda environment:

```
conda env create -f environment.yml
conda activate soundscapy-dev
```

Then, install `Soundscapy` from Github using `pip`

```
pip install git+git://github.com/MitchellAcoustics/Soundscapy@main
```

----
"""

# %% [markdown]
"""
## Working with data

### Some technical notes
`Soundscapy` is built on top of `pandas` DataFrames for working with the data. The majority of the functionality is implemented using `pandas` extensions, like `@register_dataframe_method`. In `Soundscapy`, these methods are contained within the `.isd` accessor. If you're not familiar with [`pandas` extensions](https://pandas.pydata.org/pandas-docs/stable/development/extending.html), they effectively allow us to create custom methods to apply to `pandas` DataFrames. You can access these using the `isd` namespace, as we'll show below.

### Loading data

Data can be loaded as normal using `pandas`' `read_csv()` or `read_excel()` functions. However, we have made a built in function to access the ISD directly from the Zenodo URL, which will automatically default to the latest version of the ISD. 
"""

# %%
# Add soundscapy to the Python path, if working with a local copy of the `Soundscapy` repo. Not necessary if installed with pip
import sys
sys.path.append('../..')

# %load_ext autoreload
# %autoreload 2

# Import Soundscapy
from soundscapy import isd

df = isd.load_isd_dataset()
df

# %% [markdown]
"""
### Data included

The ISD contains two primary types of data - surveys and acoustic metrics. The surveys include several blocks of questions, the most important of which are the Perceptual Attribute Questions (PAQS). These form the 8 descriptors of the soundscape circumplex - pleasant, vibrant, eventful, chaotic, annoying, monotonous, uneventful, and calm. In addition, each survey includes other information about the soundscape and demographic characteristics (age, gender, etc.). Finally, the survey section includes identifiers of when and where the survey was conducted - the LocationID, SessionID, latitude, longitude, start_time, etc. 

The final bit of information for the survey is the `GroupID`. When stopping respondents in the survey space, they were often stopped as a group, for instance a couple walking through the space would be approached together and given the same `GroupID`. While each group completes the survey, a binaural audio recording is taken, typically lasting about 30 seconds. It is from these recordings that the acoustic data is calculated. Therefore, each `GroupID` can be connected to something like 1 to 10 surveys, and to one recording, or one set of acoustic features.

Within the acoustic data are a set of psychoacoustic analyses calculated for each recording. For each metric, originally one value is calculated for each channel (right and left ear), and the maximum of the to channels is what is shown here.

### Filtering data

The ISD includes survey data and the accompanying acoustic data collected from respondents *in situ* in 13 urban locations in London and Venice. It also include recording-only data taken during the COVID-19 lockdowns in 2020. Since for this example we'll only be looking at survey data, we can start by filtering out the lockdown data.

This is done using a method included in the `isd` `pandas` namespace.
"""

# %%
df = df.isd.filter_lockdown()
df.shape

# %% [markdown]
"""
### Validating the dataset
 
In order to validate that the dataset includes the data we would expect, and to check for missing or incorrect PAQ data, we use the `validate_dataset()` method. This method can also rename the PAQ columns if necessary.
"""

# %%
df, excl = df.isd.validate_dataset(allow_na=False)
df

# %% [markdown]
"""
When samples are found which need to be excluded based on the PAQ quality checks, a dataframe with these samples will be returned. Then we can take a look at which ones were excluded and why.
"""

# %%
excl.isd.return_paqs()

# %% [markdown]
"""
### Calculating the ISOPleasant and ISOEventful coordinate values

The analysis methods used by `Soundscapy` are based firstly on converting the 8 PAQs into their projected pleasantness and eventfulness coordinate values (called ISOPleasant and ISOEventful). Although the ISD already includes these values, we'll show how to calculate them from the raw PAQs.

Start by returning a version of the dataset which only includes the PAQs so we won't conflict with the pre-computed ISOCoordinate values.
"""

# %%
paqs = df.isd.return_paqs()
paqs

# %% [markdown]
"""
Now, calculate the ISOCoordinate values.
"""

# %%
paqs = paqs.isd.add_paq_coords()
paqs

# %% [markdown]
"""
`Soundscapy` expects the PAQ values to be Likert scale values ranging from 1 to 5 by default, as specified in ISO 12913 and the SSID Protocol. However, it is possible to use data which, although structured the same way, has a different range of values. For instance this could be a 7-point Likert scale, or a 0 to 100 scale. By passing these numbers both to `validate_dataset()` and `add_paq_coords()` as the `val_range`, `Soundscapy` will check that the data conforms to what is expected and will automatically scale the ISOCoordinates from -1 to +1 depending on the original value range. 

For example:
"""

# %%
import pandas as pd
val_range = (0, 100)
sample_transform = {
    "RecordID": ["EX1", "EX2"],
    "pleasant": [40, 25],
    "vibrant": [45, 31],
    "eventful": [41, 54],
    "chaotic": [24, 56],
    "annoying": [8, 52],
    "monotonous": [31, 55],
    "uneventful": [37, 31],
    "calm": [40, 10],
}
sample_transform = pd.DataFrame().from_dict(sample_transform)
sample_transform, excl = sample_transform.isd.validate_dataset(val_range=val_range)

# %%
sample_transform = sample_transform.isd.add_paq_coords(val_range=val_range)
sample_transform

# %% [markdown]
"""
### More filtering

`Soundscapy` includes methods for several filters that are normally needed, such as filtering by `LocationID` or `SessionID`.
"""

# %%
df.isd.filter_location_ids(['CamdenTown', 'PancrasLock'])

# %%
df.isd.filter_session_ids(['RegentsParkJapan1']).head()

# %% [markdown]
"""
However, if more complex filters or some other custom filter is needed, `pandas` provides a very nice approach with its `query()` method. For instance, if we wanted to filter by gender:
"""

# %%
df.query("Gender == 'Female'")

# %% [markdown]
"""
Or a more complex filter like women over 50:
"""

# %%
df.query("Gender == 'Female' and Age > 50")

# %% [markdown]
"""
All of these filters can also be chained together. So, for instance, to return surveys from women over 50 taken in Camden Town, we would do:
"""

# %%
df.isd.filter_location_ids(['CamdenTown']).query("Gender == 'Female' and Age > 50")

# %% [markdown]
"""
## Plotting

Probably the most important part of the `Soundscapy` package is its methods for plotting soundscape circumplex data. Making use of the `seaborn kdeplot()`, we can visualise the distribution of responses within the soundscape circumplex. 

### Scatter plots

The most basic plot is the `circumplex_scatter()`. 

First, we filter down to one location that we want to look at. Then, using the `circumplex_scatter()`, we can create a default formatted plot:
"""

# %%
df.isd.filter_location_ids(['RussellSq']).isd.scatter()

# %% [markdown]
"""
Each point in this scatter plot represents the ISOCoordinate values of one survey taken in Russell Square during all of the sessions. 

We can see that the `circumplex_scatter()` has added some customisations on top of the underlying `seaborn` plots. The first is to automatically scale the plot area to the -1 to +1 bounding of the circumplex. Second is the inclusion of a grid highlighting the typically quadrants of the circumplex. Finally, customised labels which make the relationship of the ISOPleasant and ISOEventful values more clear. 

This plot can be further customised though. For instance, if you don't like or need those custom primary labels, we can remove them by setting `prim_labels = False`. We could also add labels for the diagonal circumplex axes with `diagonal_lines = True`.
"""

# %%
df.isd.filter_location_ids(['RussellSq']).isd.scatter(diagonal_lines=True)

# %% [markdown]
"""
It's also often very useful to plot the different sessions taken in the same location with different colours. This is done with the `hue` parameter. At the same time, we'll also add a legend and make the scatter points larger.
"""

# %%
df.isd.filter_location_ids(['RussellSq']).isd.scatter(hue='SessionID', legend=True, s=20, title='RussellSq Sessions')

# %% [markdown]
"""
### Distribution Plots

The real power of `Soundscapy` is in creating plots of the distribution of soundscape assessments. The interface for doing this is the same as the scatter plots above.
"""

# %%
df.isd.filter_location_ids(['RussellSq']).isd.density()

# %% [markdown]
"""
This can be customised in the same ways as the scatter plots. To see how the scatter points and the density heatmap are related, we can add scatter points to the density plot.
"""

# %%
df.isd.filter_location_ids('RussellSq').isd.density(incl_scatter=True, alpha=0.75, hue="LocationID")

# %% [markdown]
"""
*How to analyse* proposes a method for simplifying this plot, allowing easy comparison between multiple soundscapes. In the simplified version, rather than showing the full distribution heatmap, we plot only the 50th percentile density curve, showing the general shape of the soundscape.

This is done by digging into `seaborn` `kdeplot()` and using its parameters `thresh` and `levels`. We'll also go ahead and customise some other aspects, such as the color palette.
"""

# %%
df.isd.filter_location_ids(['RegentsParkJapan']).isd.density(
    title="Median perception contour and scatter plot of individual assessments\n\n",
    density_type="simple",
    hue="LocationID",
    legend=True,
    palette="dark:gray",
)

# %% [markdown]
"""
As we said, this is particularly useful for comparing different soundscapes. So let's see how we can plot three different soundscapes at once.
"""

# %%
df.isd.filter_location_ids(
    ["CamdenTown", "RussellSq", "PancrasLock"]
).isd.density(
    title="Comparison of the soundscapes of three urban spaces\n\n",
    hue="LocationID",
    density_type="simple",
    incl_scatter=False,
    palette="husl",
)

# %% [markdown]
"""
### Jointplot
"""

# %% [markdown]
"""

"""

# %%
df.isd.filter_location_ids(["CamdenTown", "RussellSq"]).isd.jointplot(hue="LocationID", marginal_kind="kde", density_type="full")

# %%
