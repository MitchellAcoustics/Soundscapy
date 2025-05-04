"""
Demo script showing the usage of the new SoundscapePlot class.

This script demonstrates the basic functionality of the SoundscapePlot class
and how it can be used to create soundscape visualizations with the
Seaborn Objects API.
"""

# %% Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

import soundscapy as sspy
from soundscapy.plotting.soundscape_plot import SoundscapePlot

# Set the style for the plots
sns.set_theme(style="whitegrid")

# %% Load the built-in ISD dataset
# Load the ISD dataset that comes with soundscapy
print("Loading ISD dataset...")
data = sspy.isd.load()

# Add ISO coordinates if they don't already exist
data = sspy.surveys.add_iso_coords(data, overwrite=True)

# Display dataset info
print(f"Dataset shape: {data.shape}")
print("\nColumns:")
print(data.columns.tolist())

# %% Create a sample subset of the data
# Select a subset of locations for demonstration
sample_data = sspy.isd.select_location_ids(data, ["CamdenTown", "RegentsParkJapan"])
print(f"Sample data shape: {sample_data.shape}")

# Show basic statistics for the ISO coordinates
print("\nBasic statistics for ISO coordinates:")
print(sample_data[["ISOPleasant", "ISOEventful"]].describe())

# %% Example 1: Basic scatter plot with grid
print("\nExample 1: Basic scatter plot with grid")

# Create a basic plot with default settings
plot1 = (
    SoundscapePlot(sample_data)
    .add(so.Dots(pointsize=25, alpha=0.7), color="location_id")
    .scale(color=so.Nominal("colorblind"))
    .add_grid(diagonal_lines=True)
    .label(title="Basic Soundscape Plot")
)

# Display the plot
plot1.show()

# %% Example 2: Customizing the plot
print("\nExample 2: Customized plot with different axes and styling")

# Create a customized plot with different configurations
plot2 = (
    SoundscapePlot(sample_data, x="ISOPleasant", y="ISOEventful")
    .add(so.Dots(pointsize=30), color="L_0_Pleasant")
    .scale(color=so.Continuous("viridis"))
    .add_grid(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), diagonal_lines=True)
    .label(title="Soundscape by Pleasantness Rating")
)

# Display the plot
plot2.show()

# %% Example 3: Adding multiple mark layers
print("\nExample 3: Multiple mark layers")

# Create location averages
location_means = (
    sample_data.groupby("location_id")[["ISOPleasant", "ISOEventful"]]
    .mean()
    .reset_index()
)

# Create multi-layer plot
plot3 = (
    SoundscapePlot(sample_data)
    # Add individual data points
    .add(so.Dots(alpha=0.6, pointsize=15), color="location_id")
    .scale(color=so.Nominal("colorblind"))
    # Add the means with larger markers
    .add(
        so.Dots(pointsize=100, alpha=0.9, marker="X"),
        data=location_means,
        color="location_id",
    )
    # Add connecting lines between points and means
    .add(
        so.Path(color="grey", alpha=0.3),
        so.Agg({"ISOPleasant": "mean", "ISOEventful": "mean"}, groupby="location_id"),
    )
    .add_grid(diagonal_lines=True)
    .label(title="Soundscape Points with Location Means")
)

# Display the plot
plot3.show()

# %% Example 4: Faceted plots
print("\nExample 4: Faceted plots by location")

plot4 = (
    SoundscapePlot(sample_data)
    .add(so.Dots(pointsize=30, alpha=0.8), color="L_0_Overall")
    .scale(color=so.Continuous("viridis"))
    .facet(col="location_id", wrap=3)
    .add_grid()
    .label(title="Soundscape by Location and Overall Rating")
)

# Display the plot
plot4.show()

# %% Example 5: Adding density contours
print("\nExample 5: Adding density contours")

plot5 = (
    SoundscapePlot(sample_data)
    # Add density contours
    .add(so.Area(alpha=0.3, fill=True), so.KDE(), color="location_id")
    # Add scatter points on top of density
    .add(so.Dots(pointsize=20, alpha=0.7), color="location_id")
    .scale(color=so.Nominal("colorblind"))
    .add_grid(diagonal_lines=True)
    .label(title="Soundscape Density and Points")
)

# Display the plot
plot5.show()

# %% Example 6: Demonstrating compatibility with standard so.Plot features
print("\nExample 6: Demonstrating compatibility with standard so.Plot features")

# Create a plot that uses both SoundscapePlot features and standard so.Plot methods
plot6 = (
    SoundscapePlot(sample_data)
    # Use paired data display
    .pair(x=["ISOPleasant", "L_0_Pleasant"], y=["ISOEventful", "L_0_Eventful"])
    # Add point layer
    .add(so.Dots(pointsize=25, alpha=0.7), color="location_id")
    # Add linear regression line
    .add(so.Line(), so.PolyFit(order=1))
    # Apply standard so.Plot styling
    .theme(
        {
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "gray",
            "grid.alpha": 0.5,
        }
    )
    .scale(color=so.Nominal("deep"))
    .label(title="Multi-variable Comparison")
)

# Display the plot
plot6.show()

# %% Example 7: Integration with Matplotlib
print("\nExample 7: Integration with Matplotlib")

# Create a plot and extract Matplotlib elements
fig, ax = plt.subplots(figsize=(8, 8))

# Create soundscape plot
base_plot = (
    SoundscapePlot(sample_data)
    .add(so.Dots(pointsize=30, alpha=0.7), color="location_id")
    .add_grid(diagonal_lines=True)
)

# Render to the provided axes
base_plot.plot(ax=ax)

# Add custom Matplotlib elements
plt.title("Soundscape Plot with Custom Matplotlib Elements", fontsize=14)
plt.figtext(0.5, 0.01, "Custom footer text", ha="center", fontsize=10)

# Add a text annotation
plt.annotate(
    "Important region",
    xy=(0.5, 0.5),
    xytext=(0.7, 0.8),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
)

# Display the plot
plt.tight_layout()
plt.show()
