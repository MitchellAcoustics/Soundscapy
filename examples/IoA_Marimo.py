import marimo

__generated_with = "0.23.5"
app = marimo.App()


app._unparsable_cell(
    r"""
    !pip install -q condacolab
    import condacolab

    condacolab.install()
    """,
    name="_",
)


@app.cell
def _(condacolab):
    condacolab.check()
    return


app._unparsable_cell(
    r"""
    !mamba install -c https://prefix.dev/sonic-forge soundscapy -y
    """,
    name="_",
)


@app.cell
def _():
    from importlib.metadata import version

    version("soundscapy")
    return


app._unparsable_cell(
    r"""
    # Can manually install 'sn' and 'CircE' in the R environment:

    # !Rscript -e \"install.packages('sn')\"
    # !Rscript -e \"remotes::install_github('MitchellAcoustics/CircE-R')\"

    # Or set an environmental variable instructing soundscapy to do it for you
    # this will happen when you first import an R-dependent module in soundscapy
    import os

    os.environ["SOUNDSCAPY_AUTO_INSTALL_R"] = "true"

    # Now install soundscapy
    !pip install \"soundscapy[spi] == 0.8.0rc10\"
    """,
    name="_",
)


@app.cell
def _():
    import soundscapy

    soundscapy.__version__
    return


@app.cell
def _():
    # Let's confirm R is ready. If needed, Soundscapy will install R packages here.
    from soundscapy.r_wrapper._r_wrapper import get_r_session

    r_session = get_r_session()
    r_session.is_ready
    return


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    import soundscapy as sspy
    from soundscapy import ISOPlot
    from soundscapy.spi import DirectParams, MultiSkewNorm

    # Set up plotting
    warnings.filterwarnings("ignore")
    return DirectParams, ISOPlot, MultiSkewNorm, np, pd, plt, sns, sspy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Soundscape Assessment Tutorial

    ## Introduction

    This tutorial will guide you through the process of conducting a comprehensive soundscape assessment using Soundscapy. You'll learn how to analyze survey data, create visualizations, and interpret the results to understand how people perceive and experience soundscapes in different locations.

    By the end of this tutorial, you'll be able to:
    - Load and validate soundscape survey data
    - Calculate ISO coordinates from perceptual attribute questions (PAQs)
    - Create basic and advanced visualizations of soundscape data
    - Perform statistical analysis of soundscape perceptions
    - Apply the Soundscape Perception Index (SPI) to evaluate soundscapes against target distributions
    - Create professional reports and presentations of your findings

    Let's begin by loading some sample data and exploring its structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Loading and Exploring the Data

    The first step in any soundscape assessment is to load and explore your data. For this tutorial, we'll use a sample dataset containing soundscape survey responses from multiple locations.
    """)
    return


@app.cell
def _(pd):
    # Load the demo data
    data = pd.read_excel("/content/DemoData.xlsx")

    # Display the first few rows
    data.head()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Display basic information about the dataset

    Dataset shape:
    """)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Number of locations:
    """)
    return


@app.cell
def _(data):
    data["LocationID"].nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Locations:
    """)
    return


@app.cell
def _(data):
    data["LocationID"].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Understanding the Data Structure

    Our dataset contains several types of columns:

    1. **Index Columns**: Identify the survey, location, and respondent
       - `LocationID`: Identifier for the location
       - `RecordID`: Identifier for the audio recording
       - `GroupID`: Identifier for the group of respondents
       - `SessionID`: Identifier for the survey session

    2. **Perceptual Attribute Questions (PAQs)**: Ratings on a 5-point Likert scale
       - `PAQ1` (pleasant): How pleasant is the soundscape?
       - `PAQ2` (vibrant): How vibrant is the soundscape?
       - `PAQ3` (eventful): How eventful is the soundscape?
       - `PAQ4` (chaotic): How chaotic is the soundscape?
       - `PAQ5` (annoying): How annoying is the soundscape?
       - `PAQ6` (monotonous): How monotonous is the soundscape?
       - `PAQ7` (uneventful): How uneventful is the soundscape?
       - `PAQ8` (calm): How calm is the soundscape?

    3. **Sound Source Dominance**: Ratings of how dominant different sound sources are
       - `traffic_noise`: Dominance of traffic noise
       - `other_noise`: Dominance of other mechanical noise
       - `human_sounds`: Dominance of human sounds
       - `natural_sounds`: Dominance of natural sounds

    4. **Acoustic Metrics**: Objective measurements of the sound environment
       - `LAeq`: A-weighted equivalent continuous sound level
       - Various other metrics like `N5`, `Sharpness`, etc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data Validation and ISO Coordinate Calculation

    Before analyzing the data, it's important to validate it to ensure quality and consistency. Soundscapy provides functions for validating soundscape data and calculating ISO coordinates.
    """)
    return


@app.cell
def _(data, sspy):
    # Validate the data
    valid_data, invalid_indices = sspy.databases.isd.validate(data)

    # Display validation results
    print(f"Original dataset size: {len(data)}")
    print(f"Valid dataset size: {len(valid_data)}")
    print(
        f"Number of invalid records: {len(invalid_indices) if invalid_indices else 0}"
    )

    # If there are invalid records, display the first few
    if invalid_indices:
        print("\nSample of invalid records:")
        print(data.loc[invalid_indices[:5]])
    return (valid_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculating ISO Coordinates

    The ISO 12913 standard defines a circumplex model for soundscape perception, with two main dimensions: pleasantness and eventfulness. Soundscapy can calculate these coordinates from the PAQ responses with a single function call.

    The ISO coordinates are calculated using a trigonometric projection of the eight PAQ responses, where:
    - `ISOPleasant` represents the horizontal axis (pleasant to unpleasant)
    - `ISOEventful` represents the vertical axis (eventful to uneventful)

    These coordinates allow us to position each response in the soundscape circumplex model, which has four quadrants:
    - Pleasant and Eventful: "Vibrant"
    - Unpleasant and Eventful: "Chaotic"
    - Unpleasant and Uneventful: "Monotonous"
    - Pleasant and Uneventful: "Calm"
    """)
    return


@app.cell
def _(sspy, valid_data):
    valid_data_1 = sspy.surveys.add_iso_coords(valid_data, overwrite=True)
    print("Data with ISO coordinates:")
    valid_data_1.round(3)
    return (valid_data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save the data

    After using Soundscapy to calculate the ISOCoordinates, we can easily save our resulting DataFrame out to a file. We can easily save to an Excel file, allowing you to use the tools you're more comfortable with for additional analysis:
    """)
    return


@app.cell
def _(valid_data_1):
    valid_data_1.to_csv("SoundscapyResults.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Basic Visualization and Summary Statistics

    Now that we have our validated data with ISO coordinates, we can create basic visualizations and calculate summary statistics to understand the soundscape perceptions at different locations.
    """)
    return


@app.cell
def _(valid_data_1):
    iso_stats = valid_data_1.groupby("LocationID")[["ISOPleasant", "ISOEventful"]].agg(
        ["mean", "std", "min", "max"]
    )
    print("Summary statistics for ISO coordinates by location:")
    iso_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create a circumplex scatter plot

    Creating a scatter plot of the Soundscape data is extremely easy in Soundscapy. Let's start by plotting a single location. Again, we use `sspy.isd.select_location_ids()` to select a single location:
    """)
    return


@app.cell
def _(sspy, valid_data_1):
    subset_data = sspy.isd.select_location_ids(
        valid_data_1, ["CamdenTown", "PancrasLock"]
    )
    sspy.scatter(subset_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Customizing the plot

    The Soundscapy plotting functions contain several customization options, such as changing the color, palette, point size, and title:
    """)
    return


@app.cell
def _(sspy, valid_data_1):
    sspy.scatter(
        sspy.isd.select_location_ids(valid_data_1, "CamdenTown"),
        color="purple",
        s=40,
        title="Circumplex Scatter Plot of Soundscape Perceptions",
        xlabel="Pleasantness",
        ylabel="Eventfulness",
        diagonal_lines=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Color by location

    One of the most useful settings is the ability to split the plot by some grouping variable. In Soundscape data, this will often be the location, but we can easily select any other categorical variable in the dataset. Simply set the `hue` option to get a different color for each group:
    """)
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.scatter(
        valid_data_1,
        title="Soundscape Perceptions Across All Locations",
        hue="LocationID",
        palette="bright",
        figsize=(10, 10),
        s=50,
    )
    plt.show()
    return


@app.cell
def _():
    # Split by a different variable:

    # sspy.scatter(...)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Density Plots

    Of course, the most notable plot type in Soundscapy is the Density plot. Just like scatter, this has its own simple function. Try it out:
    """)
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.density(valid_data_1, hue="LocationID", density_type="simple", fill=False)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Creating Likert Style Plots

    Likert plots are a useful way to visualize the distribution of responses to Likert scale questions, such as the PAQs in our survey. Soundscapy integrates with the `plot_likert` package to create these visualizations.

    We can use Soundscapy's `paq_likert` plot to examine the distribution of PAQ responses:
    """)
    return


@app.cell
def _(sspy, valid_data_1):
    sspy.paq_likert(
        valid_data_1, title="Distribution of PAQ Responses in the Demo Data"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get a better view of the data, we can also split it across the Locations. To do this, we use the `select_location_ids()` function from Soundscapy, and use a `for` loop to create a separate plot for each location.
    """)
    return


@app.cell
def _(data):
    data["LocationID"].unique()
    return


@app.cell
def _(data, plt, sspy):
    for location in data["LocationID"].unique():
        sspy.paq_likert(
            sspy.databases.isd.select_location_ids(data, location),
            title=f"Distribution of PAQ Responses in {location}",
        )
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyzing Sound Source Dominance

    Let's examine the dominance of different sound sources at each location.
    """)
    return


@app.cell
def _(sspy, valid_data_1):
    sspy.stacked_likert(valid_data_1, "traffic_noise", title="Traffic Noise Dominance")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try this out with some of the other Likert scaled data:
    """)
    return


@app.cell
def _():
    # sspy.stacked_likert(...)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In addition to the more specialised Likert plots provided by Soundscapy, we can of course apply more common analyses. For this, we need to do some data processing and plotting in Pandas and Seaborn:
    """)
    return


@app.cell
def _(valid_data_1):
    sound_sources = (
        valid_data_1.groupby("LocationID")[
            ["traffic_noise", "other_noise", "human_sounds", "natural_sounds"]
        ]
        .mean()
        .round(2)
    )
    return (sound_sources,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mean sound source dominance by location:
    """)
    return


@app.cell
def _(sound_sources):
    sound_sources
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since Soundscapy doesn't implement its own functionality for the more common plots, we fall back to the very nice Seaborn plotting library, which we imported as `sns` to create a barplot of the mean sound source responses (or you can of course do this in something like Excel):
    """)
    return


@app.cell
def _(plt, sns, sound_sources, sspy):
    # Create a bar chart
    sound_sources_plot = sound_sources.reset_index().melt(
        id_vars=["LocationID"], var_name="Source", value_name="Dominance"
    )

    sns.barplot(
        data=sspy.isd.select_location_ids(sound_sources_plot, "CamdenTown"),
        x="Source",
        y="Dominance",
        # hue="LocationID",
        palette="colorblind",
    )
    plt.title("Sound Source Dominance by Location")
    plt.xlabel("Sound Source")
    plt.ylabel("Mean Dominance Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Creating Complex Plots with Hue and Subplots

    Soundscapy makes it easy to create more complex visualizations that show relationships between different variables. Let's explore how sound source dominance affects soundscape perception:
    """)
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.create_iso_subplots(
        data=valid_data_1,
        subplot_by="LocationID",
        hue="traffic_noise",
        plot_layers=["scatter", "simple_density"],
        title="Impact of Natural Sounds on Soundscape Perception",
        subplot_size=(5, 4),
        fill=False,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.create_iso_subplots(
        data=valid_data_1,
        subplot_by="LocationID",
        hue="traffic_noise",
        plot_layers=["scatter", "simple_density"],
        title="Impact of Traffic Noise on Soundscape Perception",
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also examine the relationship between acoustic metrics and soundscape perception:
    """)
    return


@app.cell
def _(plt, sns, valid_data_1):
    sns.lmplot(
        data=valid_data_1,
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
    return


@app.cell
def _(plt, sns, valid_data_1):
    sns.lmplot(
        data=valid_data_1,
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Applying SPI Analysis

    The Soundscape Perception Index (SPI) is a powerful tool for comparing soundscapes to target distributions. It quantifies the similarity between two soundscape distributions on a scale from 0 to 100, where 100 indicates perfect similarity.

    Let's define some target distributions and calculate SPI scores for our locations:
    """)
    return


@app.cell
def _(DirectParams, MultiSkewNorm, np, pd, plt, sspy):
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
    return tranquil_df, tranquil_msn


@app.cell
def _(DirectParams, MultiSkewNorm, np, pd, plt, sspy):
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
    return vibrant_df, vibrant_msn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's calculate SPI scores for each location against both target distributions:
    """)
    return


@app.cell
def _(pd, plt, sspy, tranquil_msn, valid_data_1, vibrant_msn):
    locations = valid_data_1["LocationID"].unique()
    tranquil_spi_scores = {}
    vibrant_spi_scores = {}
    for location_1 in locations:
        location_data = sspy.databases.isd.select_location_ids(valid_data_1, location_1)
        tranquil_spi_scores[location_1] = tranquil_msn.spi_score(
            location_data[["ISOPleasant", "ISOEventful"]]
        )
        vibrant_spi_scores[location_1] = vibrant_msn.spi_score(
            location_data[["ISOPleasant", "ISOEventful"]]
        )
    spi_results = pd.DataFrame(
        {"Tranquil SPI": tranquil_spi_scores, "Vibrant SPI": vibrant_spi_scores}
    ).T
    print("SPI Scores by Location:")
    print(spi_results)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Using the ISOPlot Interface for SPI Analysis

    Soundscapy's `ISOPlot` class provides a more sophisticated interface for creating SPI visualizations. Let's use it to create a comprehensive SPI analysis:
    """)
    return


@app.cell
def _(ISOPlot, tranquil_df, valid_data_1):
    tranquil_plot = (
        ISOPlot(data=valid_data_1, title="Comparing Locations Against Tranquil Target")
        .create_subplots(
            subplot_by="LocationID", figsize=(4, 4), auto_allocate_axes=True
        )
        .add_scatter()
        .add_simple_density(fill=True)
        .add_spi(spi_target_data=tranquil_df, show_score="on axis")
        .style(legend_loc=False)
    )
    return


@app.cell
def _(ISOPlot, valid_data_1, vibrant_df):
    vibrant_plot = (
        ISOPlot(data=valid_data_1, title="Comparing Locations Against Vibrant Target")
        .create_subplots(
            subplot_by="LocationID", figsize=(4, 4), auto_allocate_axes=True
        )
        .add_scatter()
        .add_simple_density(fill=True)
        .add_spi(spi_target_data=vibrant_df, show_score="on axis")
        .style(legend_loc=False)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Interpreting the Results

    Now that we've analyzed our data and created various visualizations, let's interpret the results:

    1. **Location Characteristics**:
       - Each location has a distinct soundscape character, as shown by its position in the circumplex model.
       - Some locations are more pleasant (higher ISOPleasant values), while others are more eventful (higher ISOEventful values).

    2. **Sound Source Influence**:
       - Natural sounds tend to increase pleasantness, as shown by the relationship between natural sound dominance and ISOPleasant values.
       - Traffic noise tends to decrease pleasantness, as shown by the relationship between traffic noise dominance and ISOPleasant values.

    3. **Acoustic Metrics**:
       - Higher sound levels (LAeq) are generally associated with lower pleasantness.
       - Higher loudness (N5) is generally associated with higher eventfulness.

    4. **SPI Analysis**:
       - Some locations match better with the tranquil target, while others match better with the vibrant target.
       - This information can be used to identify which locations provide the desired soundscape experience.

    These insights can inform soundscape design and management decisions, such as:
    - Which locations to preserve or enhance for specific soundscape experiences
    - Which sound sources to promote or mitigate at different locations
    - How to design new spaces to achieve desired soundscape characteristics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Gallery of Soundscapy Visualizations

    Soundscapy offers a wide range of visualization options. Here's a gallery of additional plots you can create:
    """)
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.scatter(valid_data_1, title="Basic Scatter Plot", diagonal_lines=True)
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.density(valid_data_1, title="Density Plot", diagonal_lines=True, fill=True)
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.iso_plot(
        valid_data_1,
        title="Combined Scatter and Density Plot",
        plot_layers=["scatter", "density"],
        diagonal_lines=True,
    )
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.density(
        valid_data_1,
        title="Simple Density Plot with Hue",
        density_type="simple",
        hue="LocationID",
    )
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    sspy.jointplot(valid_data_1, title="Joint Plot")
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    plt.figure(figsize=(10, 10))
    sspy.jointplot(
        valid_data_1, title="Joint Plot with Histogram Marginals", marginal_kind="hist"
    )
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    plt.figure(figsize=(10, 10))
    sspy.jointplot(
        valid_data_1,
        title="Joint Plot with Grouping",
        hue="LocationID",
        density_type="simple",
    )
    plt.show()
    return


@app.cell
def _(plt, sspy, valid_data_1):
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    sspy.scatter(
        valid_data_1, title="All Locations - Scatter", hue="LocationID", ax=ax1
    )
    ax2 = fig.add_subplot(gs[0, 1])
    sspy.density(
        valid_data_1,
        title="All Locations - Density",
        hue="LocationID",
        density_type="simple",
        incl_scatter=False,
        fill=False,
        ax=ax2,
    )
    ax3 = fig.add_subplot(gs[1, 0])
    sspy.scatter(
        valid_data_1, title="Sound Level (LAeq)", hue="LAeq", palette="viridis", ax=ax3
    )
    ax4 = fig.add_subplot(gs[1, 1])
    sspy.density(
        valid_data_1,
        title="Natural Sounds Dominance",
        hue="natural_sounds",
        density_type="simple",
        ax=ax4,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Summary

    In this tutorial, you've learned how to:

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

    These skills will enable you to conduct comprehensive soundscape assessments and communicate your findings effectively. By understanding how people perceive and experience soundscapes, you can contribute to the design and management of more pleasant and appropriate acoustic environments.

    ## References

    1. ISO 12913-1:2014. Acoustics — Soundscape — Part 1: Definition and conceptual framework.
    2. ISO 12913-2:2018. Acoustics — Soundscape — Part 2: Data collection and reporting requirements.
    3. ISO 12913-3:2019. Acoustics — Soundscape — Part 3: Data analysis.
    4. Mitchell, A., Aletta, F., & Kang, J. (2022). How to analyse and represent quantitative soundscape data. JASA Express Letters, 2, 37201. https://doi.org/10.1121/10.0009794
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
