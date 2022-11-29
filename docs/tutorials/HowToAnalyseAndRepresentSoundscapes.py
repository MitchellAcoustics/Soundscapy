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
#     display_name: 'Python 3.9.2 64-bit (''GenDataSci'': conda)'
#     name: python3
# ---

# %% [markdown]
"""
# How to analyse and represent soundscape perception

Andrew Mitchell, Francesco Aletta, Jian Kang

This notebook provides examples for analysing and visualising soundscape assessment data from the International Soundscape Database (ISD). The custom functions created for this purpose are stored in the `isd.py` file. 

The ISD contains survey and acoustic data collected in urban public spaces with the goal of creating a unified dataset for the development of a predictive soundscape model and a set of soundscape indices. We have created a new visualisation method in order to properly analyse and examine the assessment of the locations included in the dataset. This method focuses on enabling sophisticated statistical analyses and ensuring the variety of responses in a location are properly considered.

In this notebook we will walk you through both using the code itself and interpreting the soundscape perception of urban spaces.
"""

# %%
# Add soundscapy to the Python path
import sys
sys.path.append('../..')
# imports
# %matplotlib inline
import soundscapy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

# %config InlineBackend.figure_format = 'svg'

# %% [markdown]
r"""
## The current ISO 12913 framework

Although different methods are proposed for data collection in ISO12913 Part 2, in the context of this study we focus on the questionnaire-based soundscape assessment (Method A), because it is underpinned by a theoretical relationship among the items of the questionnaire that compose it. The core of this questionnaire is the 8 perceptual attributes (PA) originally derived in Axlesson et al. (2010): pleasant, vibrant (or exciting), eventful, chaotic, annoying, monotonous, uneventful, and calm. In the questionnaire procedure, these PAs are assessed independently of each other, however, they are conceptually considered to form a two-dimensional circumplex with *Pleasantness* and *Eventfulness* on the x- and y-axis, respectively, where all regions of the space are equally likely to accomodate a given soundscape assessment. In Axelsson et al. (2010) a third primary dimension, *Familiarity* was also found, however this only accounted for 8% of the variance and is typically disregarded as part of the standard circumplex. As will be made clear throughout, the circumplex model has several aspects which make it useful for representing the soundscape perception of a space as a whole.

### Coordinate transformation

To facilitate the analysis of the perceptual attribute (PA) responses, the Likert scale responses are coded from 1 (Strongly disagree) to 5 (Strongly agree) as ordinal variables. In order to reduce the 8 PA values into a pair of coordinates which can be plotted on the Pleasant-Eventful axes, Part 3 of the ISO 12913 provides a trigonometric transformation, based on the $45\degree$ relationship between the diagonal axes and the pleasant and eventful axes. This tranformation projects the coded values from the individual PAs down onto the primary Pleasantness and Eventfulness dimensions, then adds them together to form a single coordinate pair. In theory, this coordinate pair then encapsulates information from all 8 PA dimensions onto a more easily understandable and analyzable 2 dimensions. 

The ISO coordinates are thus calculated by:

$$
ISOPleasant = [(pleasant - annoying) + \cos{45\degree} * (calm - chaotic) + \cos{45\degree} * (vibrant - monotonous)] * \frac{1}{(4 + \sqrt{32})}
$$

$$
ISOEventful = [(eventful - uneventful) + \cos{45\degree} * (chaotic - calm) + \cos{45\degree} * (vibrant - monotonous)] * \frac{1}{(4 + \sqrt{32})}
$$

where the PAs are arranged around the circumplex as shown in Figure 1. The $\cos{45\degree}$ term operates to project the diagonal terms down ono the x and y axes, and the $\frac{1}{4 + \sqrt{32}}$ scales the resulting coordinates to the range (-1, 1). The result of this transformation is demonstrated in Figure 1.

To give an example of this, we create two example survey responses, with different PAQ answers:
"""

# %%
sample_transform = {
    "RecordID": ["EX1", "EX2"],
    "pleasant": [4, 2],
    "vibrant": [4, 3],
    "eventful": [4, 5],
    "chaotic": [2, 5],
    "annoying": [1, 5],
    "monotonous": [3, 5],
    "uneventful": [3, 3],
    "calm": [4, 1],
}
sample_transform = pd.DataFrame().from_dict(sample_transform)
sample_transform = sample_transform.sspy.convert_column_to_index("RecordID")
sample_transform


# %% [markdown]
"""
We can visualise how these individual PAQ answers are arranged on the circumplex with a radar plot.
"""

# %%
fig = plt.figure(figsize=(4, 4))
plt.rcParams["figure.dpi"] = 350
sample_transform.sspy.paq_radar()


# %% [markdown]
"""
Now, we can apply the transform formula from above to calculate the ISOPleasant and ISOEventful values and add them to the dataframe.
"""

# %%
sample_transform = soundscapy.utils.surveys.rename_paqs(sample_transform)
sample_transform = sample_transform.sspy.add_paq_coords(scale_to_one=True)
sample_transform


# %% [markdown]
"""
Finally, we can plot these values on a two dimensional plane to visualise how the transform went from the 8 dimensions shown in the radar plot to the two ISO dimensions. This is done by calling the `circumplex_scatter()` function included in `isd.py`. This will create a plotting axis with the appropriate circumplex grid and labels, then plot the ISOPleasant and ISOEventful values as the x and y coordinates.

This treatment of the 8 PAs makes several assumptions and inferences about the relationships between the dimensions. As stated in the standard:

>  According to the two-dimensional model, vibrant soundscapes are both pleasant and eventful, chaotic soundscapes are both eventful and unpleasant, monotonous soundscapes are both unpleasant and uneventful, and finally calm soundscapes are both uneventful and pleasant.
"""

# %%
colors = ["b", "r"]
palette = sns.color_palette(colors)
sample_transform.sspy.scatter(
    hue="RecordID",
    legend="brief",
    s=100,
    palette=palette,
    alpha=0.45,
    diagonal_lines=True,
)


# %% [markdown]
"""
## The way forward: Probabilistic soundscape representation

Given the identified issues with the recommended methods for statistical analysis and their shortcomings in representing the variety in perception of the soundscape in a space, how then should we discuss or present the results of these soundscape assessments? Ideally the method will: 1) take advantage of the circumplex coordinates and their ability to be displayed on a scatter plot and be treated as continuous variables, 2)  scale from a dataset of 10 responses to thousands of responses, 3) facilitate the comparison of the soundscapes of different locations and conditions, and 4) encapsulate the nuances and diversity of soundscape perception by representing the distribution of responses.

We therefore present a series of visualisations of the soundscape assessments of several urban spaces included in the International Soundscape Database (ISD) which reflect these goals. The specific locations selected from the ISD are chosen for demonstration only and these methods can be applied to any location. Rather than attempting to represent a single individual's soundscape or of describing a location's soundscape as a single average assessment (as in Part 3 of the ISO technical specification), this representation shows the whole range of perception of the users of the space. 

To begin, we can load the dataset directly from the ISD:
"""

# %%
ssid = soundscapy.isd.load_isd_dataset()
ssid = ssid.isd.filter_lockdown()
ssid, excl = ssid.sspy.validate_dataset(allow_na=False)
ssid.head()


# %% [markdown]
"""
First, rather than calculating the median response to each PA in the location, then calculating the circumplex coordinates, the coordinates for each individual response are calculated. This results in a vector of ISOPleasant, ISOEventful values which are continuous variables from -1 to +1 and can be analysed statistically by calculating summary statistics (mean, standard deviation, quintiles, etc.) and through the use of regression modelling, which can often be simpler and more familiar than the recommended methods of analysing ordinal data. This also enables each individual's response to be placed within the pleasant-eventful space. All of the responses for a location can then be plotted, giving an overall scatter plot for a location, as demonstrated in (i). 
"""

# %%
ssid.isd.filter_location_ids(["PancrasLock"]).sspy.jointplot(
    title="(a) Example distribution of the soundscape perception of an urban park",
    diagonal_lines=True,
    hue="LocationID",
    legend=True,
    alpha=0.75,
)


# %% [markdown]
"""
Once these individual responses are plotted, we then overlay a heatmap of the bivariate distribution (with color maps for each decile) and marginal distribution plots. In this way, three primary characteristics of the soundscape perception can be seen: 

1) The distribution across both pleasantness and eventfulness, including the central tendency, the dispersion, and any skewness in the response;
2) The general shape of the soundscape within the space - in this case Russell Sq is almost entirely in the pleasant half, but is split relatively evenly across the eventfulness space, meaning while it is perceived as generally pleasant, it is not strongly calm or vibrant;
3) The degree of agreement about the soundscape perception - there appears to be a relatively high agreement about the character of Russell Sq, as demonstrated by the compactness of the distribution, but this is not the case for every location, as will be shown later.

Fig (i) includes several in-depth visualisations of the distribution of soundscape assessments, however the detail included can make further analysis difficult. In particular, a decile heatmap is so visually busy that, in our experience, it is not possible to plot more than one soundscape distribution at a time without the figure becoming overly busy. It also can make it difficult to truly grasp point 2, the general shape of the soundscape. To facilitate this, the same soundscape can be represented by its 50th percentile contour, as demonstrated in Fig (ii) where the shaded portion contains 50% of the responses. This simplified view of the distribution presents several advantages, as will be demonstrated in Figs. (iii and iv) and takes inspiration from the recommendation in the ISO standard to use the median as a summary statistic.

When visualised this way, it is possible to identify outliers and responses which are the result of anamolous sound events. For instance if, during a survey session at a calm park, a fleet of helicopters flies overhead, driving the participants to respond that the soundscape is highly chaotic, we would see a group of scatter points in the chaotic quadrant which appear obviously outside the general pattern of responses. In a simpler analysis method, these responses would either be entirely discarded as outliers or surveys and soundwalks would be halted entirely -- ignoring what is in fact a significant impact on that location, its soundscape, and how useful it may be for the community -- or would be included within the statistical analysis, significantly impacting the central tendency and dispersion metrics (i.e. median and range) without consideration for the context. This is the situation shown in Fig (ii) where it is obvious that there is strong agreement that Regents Park Fields is highly pleasant and calm, however we can see numerous responses which assessed it as highly chaotic when a series of military helicopter fly overs drastically changed the sound environment of the space for nearly 20 minutes.
"""

# %%
location = "RegentsParkFields"
fig, ax = plt.subplots(1,1, figsize=(7, 7))
ssid.isd.filter_location_ids([location]).sspy.density(
    title="(b) Median perception contour and scatter plot of individual assessments\n\n",
    ax=ax,
    hue="LocationID",
    legend=True,
    density_type="simple",
    palette="dark:gray",
)


# %% [markdown]
"""
Fig (iii) demonstrates how this simplified representation makes it possible to compare the soundscape of several locations in a sophisticated way. The soundscape assessments of three urban spaces, Camden Town, Pancras Lock, and Russell Square, are shown overlaid with each other. We can see that Camden Town, a busy and crowded street corner with high levels of traffic noise and amplified music, is generally perceived as chaotic, but the median contour shape which characterises it also crosses over into the vibrant quadrant. We can also see that, for a part of the sample, Russell Square and Pancras Lock are both perceived as similarly pleasant, however some portion of the responses perceived Pancras Lock as being somewhat chaotic and annoying. This kind of visualisation is able to highlight these similarities between the soundscapes in the locations and identify how they differ. From here, further investigation could lead us to answer what is it that led to those people perceiving the location as unpleasant, and what similarities does the soundscape of Pancras Lock have with Russell Square that could perhaps be enhanced to increase the proportion of people perceiving it as more pleasant.
"""

# %%
fig, ax = plt.subplots(1,1, figsize=(7,7))

ssid.isd.filter_location_ids(
    ["CamdenTown", "RussellSq", "PancrasLock"]
).sspy.density(
    title="(c) Comparison of the soundscapes of three urban spaces\n\n",
    ax=ax,
    hue="LocationID",
    alpha=0.5,
    palette="husl",
    legend=True,
    density_type="simple"
)


# %% [markdown]
"""
In addition to solely analysing the distributions of the perceptual responses themselves, this method can also be combined with other acoustic, environmental, and contextual data. The final example, in Fig (iv) demonstrates how this method can better demonstrate the complex relationships between acoustic features of the sound environment and the soundscape perception. The data in the ISD includes approximately 30 second long binaural audio recordings taken while each participant was responding to the soundscape survey, providing an indication of the exact sound environment they were exposed to. For Fig (iv) the entire dataset of 1,338 responses at all 13 locations has been split according to the analysis of these recordings giving a set of less than 65 dB LAeq and  a set of more than 65 dB. The bivariate distribution of these two conditions are then plotted. 
"""

# %%
ssid["dBLevel"] = pd.cut(
    ssid["LAeq_L(A)(dB(SPL))"],
    bins=(0, 63, 150),
    labels=("Under 63dB", "Over 63dB"),
    precision=1,
)

ssid.head()


# %%
ssid["dBLevel"].describe()


# %%
ssid.sspy.jointplot(
    marginal_kind="kde",
    title="(d) Soundscape perception as a function of sound level",
    diagonal_lines=False,
    alpha=0.5,
    palette="colorblind",
    hue="dBLevel",
    density_type="simple",
    incl_scatter=False,
    legend=True,  #
    marginal_kws={"common_norm": False, 'fill': True},
)


# %% [markdown]
"""
## Other examples

In addition to the visualisation demonstrations given above which were included in the JASA Express Letters article, we present a few examples of the uses of this distributional shape approach. 

### The soundscape shape of all 13 locations:
"""

# %%
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, location in enumerate(ssid.LocationID.unique()):
    ssid.isd.filter_location_ids(location_ids=[location]).sspy.density(
        ax=axes.flatten()[i], title=location
    )

plt.tight_layout()


# %% [markdown]
"""
### A comparison of two days in the same location:
"""

# %%
fig, ax = plt.subplots(1,1,figsize=(5,5))
location='RegentsParkFields'
ssid.isd.filter_location_ids(location_ids=[location]).sspy.density(ax=ax, title='Comparison of two days in Regents Park', density_type="simple", fill=False, incl_outline=True, hue='SessionID', legend=True)

# %% [markdown]
"""
### All of the survey days in every location:
"""

# %%
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i, location in enumerate(ssid["LocationID"].unique()):
    ssid.isd.filter_location_ids(location).sspy.density(
        ax=axes.flatten()[i],
        title=location,
        fill=False,
        hue="SessionID",
        legend=True,
        density_type="simple"
    )
    ssid.isd.filter_location_ids(location).sspy.scatter(
        ax=axes.flatten()[i],
        title=location,
        hue="SessionID",
        s=10,
    )
plt.tight_layout()


# %% [markdown]
"""
### Statistical analysis of the ISD dataset

Although the visualisations shown in the above figures are a powerful tool for viewing, analysing, and discussing the multi-dimensional aspects of soundscape perception, there are certainly cases where simpler metrics are needed to aid discussion and to set design goals. Taking inspiration from noise annoyance, we propose a move toward discussing the "percent of people likely to perceive" a soundscape as pleasant, vibrant, etc. when it is necessary to use numerical descriptions. In this way, a numerical design goal could also be set as e.g. "the soundscape should be likely to be perceived as pleasant by at least 75% of users" or the results of an intervention presented as e.g. "the likelihood of the soundscape being perceived as calm increased from 30% to 55%". These numbers can be drawn from either actual surveys or from the results of predictive models.

Finally, although acknowledging the distribution of responses is crucial, it is sometimes necessary to summarise locations down to a single point to compare many different locations and to easily investigate how the soundscape assessment has generally changed over time. For this purpose, the mean of the ISOPleasant and ISOEventful values across all respondents is calculated to result in a single coordinate point per location. This clearly mirrors the original intent of the coordinate transformation presented in the ISO, but by applying the transformation first to each individual assessment then calculating the mean value, it maintains a direct link to the distributions shown above. 

We have included a function for creating a numerical summary of each location. For this, we first calculate the mean ISOPleasant and ISOEventful value for the location, giving a single coordinate to describe the location in the circumplex. We then calculate the percentage of overall responses falling in the `pleasant` or `eventful` halfs, or in the `vibrant`, `chaotic`, etc. quadrants.
"""

# %%
results = ssid.isd.soundscapy_describe()
results


# %% [markdown]
"""
The standard `describe()` method in pandas can also still be used to calculate other summary statistics.
"""

# %%
ssid.describe()


# %% [markdown]
"""
Finally, the mean coordinate values for each location can be plotted in the soundscape circumplex.
"""

# %%
from soundscapy.utils.surveys import mean_responses
means = mean_responses(ssid, group="LocationID")
means = means.sspy.add_paq_coords()
means.sspy.scatter(hue="LocationID", s=40, legend=False, xlim=(-0.25, 0.75), ylim=(-0.25, 0.75))

# %%

# %%
