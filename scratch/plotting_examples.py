# %%
"""
# Soundscapy Plotting Module: Developer's Guide

This notebook demonstrates how the Soundscapy plotting module works, starting with the internal
components and building up to the user-facing API. It uses the newly fixed CircumplexPlot class
that implements the grammar of graphics approach with Seaborn Objects.

RECENT FIXES TO CIRCUMPLEX PLOT:
1. Fixed palette handling to work with the latest Seaborn Objects API
2. Fixed the show() method to properly display plots in notebook contexts
3. Added property access to the underlying Seaborn Objects plot
4. Added a method to get matplotlib objects directly

These changes allow the CircumplexPlot class to work correctly while providing
a clean, builder-pattern API for creating layered visualizations.

See the proposed implementation details in the comment block below.

# CircumplexPlot enhancements - IMPLEMENTED
The CircumplexPlot class has been fixed with the following improvements:

1. Added @property to directly access the Seaborn Objects plot:
   @property
   def seaborn_plot(self):
       return self.plot

2. Added get_matplotlib_objects() method:
   def get_matplotlib_objects(self):
       fig, ax = plt.subplots(figsize=(6, 6))
       self.plot.plot(ax=ax)
       return fig, ax

3. Fixed the show() method to use pyplot=True:
   def show(self):
       self.plot.plot(pyplot=True)

4. Fixed palette handling to work with Seaborn Objects API by:
   - Storing palette_name instead of directly using palette
   - Using .scale(color=so.Nominal(palette_name)) instead of the palette parameter

These changes make CircumplexPlot work correctly in notebook contexts
while providing direct access to both the Seaborn Objects and matplotlib objects.

Structure:
1. Data preparation
2. Backend implementation with Seaborn Objects API
3. Builder pattern and grammar of graphics approach
4. Simple user-facing API
5. Advanced examples and integrations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so

# Import from soundscapy
import soundscapy as sspy
from soundscapy.plotting import (
    CircumplexPlot,
    create_circumplex_subplots,
    density_plot,
    joint_plot,
    scatter_plot,
)
from soundscapy.plotting.plotting_utils import DEFAULT_XLIM, DEFAULT_YLIM
from soundscapy.surveys.processing import add_iso_coords, simulation

# %%
# ## 1. Data Preparation


# Option 1: Load real data from the ISD database
def load_real_data():
    """Load real data from the International Soundscape Database."""
    # Load ISD data
    isd_data = sspy.isd.load()

    # Add ISO coordinates if not already present
    if "ISOPleasant" not in isd_data.columns:
        isd_data = add_iso_coords(isd_data)

    # Get data for a specific location
    location_data = sspy.isd.select_location_ids(isd_data, ["CamdenTown"])

    return isd_data, location_data


# Option 2: Create simulated data for demonstration
def create_simulated_data(n=200):
    """Generate random data for demonstration."""
    return simulation(n=n, incl_iso_coords=True)


# Create sample data for our examples
data = create_simulated_data(n=200)
print(f"Generated simulated data with {len(data)} samples")
# Look at the first few rows of our data
data.head()


# %%
# ## 2. Low-Level Backend Implementation

"""
This section shows the core building blocks of the Seaborn Objects-based implementation.
We'll demonstrate how to use Seaborn Objects directly to create plots, which is what
the CircumplexPlot builder class uses internally.
"""

# Create a basic seaborn objects plot manually
plot = so.Plot(data, x="ISOPleasant", y="ISOEventful")

# Add a scatter layer using Dots mark
plot = plot.add(so.Dots())

# Add styling manually (the raw way)
plot = plot.limit(x=DEFAULT_XLIM, y=DEFAULT_YLIM)
plot = plot.label(title="Direct Seaborn Objects Example")

# Display the plot
plot.show()

# %%
# Add a KDE layer to our plot (density plot)
# Note: In Seaborn Objects, KDE is a transform, not a mark
plot = so.Plot(data, x="ISOPleasant", y="ISOEventful")
plot = plot.add(
    so.Area(alpha=0.4, fill=True),  # Mark (what to draw)
    so.KDE(bw_adjust=1.2),  # Transform (how to process the data)
)

# To style our plot with grid lines, we can use matplotlib commands after rendering
# First, create a figure and axes to draw on
fig, ax = plt.subplots(figsize=(6, 6))

# Apply seaborn objects plot to this axes
plot = plot.on(ax)

# Now manually add styling to the axes
ax.grid(True, which="major", color="grey", alpha=0.5)
ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)

# Set axis limits and title
plot = plot.limit(x=DEFAULT_XLIM, y=DEFAULT_YLIM)
plot = plot.label(title="KDE Transform with Area Mark")

# Display the plot
plot.show()

# %%
# Add grouping with hue
# Create a categorical column to use for grouping
data["group"] = np.random.choice(["Group A", "Group B", "Group C"], size=len(data))

# Create a plot with color grouping using hue
plot = so.Plot(data, x="ISOPleasant", y="ISOEventful")

# Add dots with color grouping by 'group' column
plot = plot.add(
    so.Dots(pointsize=30, alpha=0.7),
    color="group",  # Use 'group' column for colors
)

# Apply a colorblind-friendly palette
plot = plot.scale(color=so.Nominal("colorblind"))

# Add a title and legend (in Seaborn Objects, legend labels come from the data)
plot = plot.label(title="Grouping with Color", group="Group")


# Apply styling through the .on() method with a function
def style_axes(ax):
    ax.grid(True, which="major", color="grey", alpha=0.5)
    ax.axhline(y=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
    ax.axvline(x=0, color="grey", linestyle="dashed", alpha=1, linewidth=1.5)
    return ax


# Create a new figure
plt.figure(figsize=(6, 6))
# Apply the function through pyplot
plot.plot(pyplot=True)
# Get the current axes and style it
ax = plt.gca()
style_axes(ax)

# We don't need this since we've already displayed the plot
# The axes have already been styled above

# %%
# ## 3. Builder Pattern / Grammar of Graphics Approach

"""
The CircumplexPlot class provides a builder pattern interface that wraps the 
Seaborn Objects API. It lets you build plots layer by layer with a fluent interface.
"""

# Example 1: Basic layer composition
print("Builder pattern - basic composition")

# Create a CircumplexPlot with default parameters
plot = CircumplexPlot(data)
plot.add_scatter(pointsize=30, alpha=0.7)
plot.add_grid(diagonal_lines=True)
plot.add_title("Grammar of Graphics: Basic Layers")

# Now we can simply use show() since it's been fixed
plot.show()

# %%
# Example 2: Multiple layer types - adding density + scatter
print("Builder pattern - multiple layer types")

plot = CircumplexPlot(data)
# Add a density layer first (background)
plot.add_density(alpha=0.3, fill=True, simple=True)
# Add scatter points on top
plot.add_scatter(pointsize=15, alpha=0.6)
# Add styling elements
plot.add_grid(diagonal_lines=True)
plot.add_title("Grammar of Graphics: Multiple Layers")

# Direct access to the Seaborn Objects plot (this is what a fixed CircumplexPlot.show() would do)
plot.plot.plot(pyplot=True)

# %%
# Example 3: Using hue for grouping with the builder pattern
print("Builder pattern - using hue for grouping")

# Step by step construction with assignments
plot = CircumplexPlot(data, hue="group")

# Add layers in order from background to foreground
plot.add_density(alpha=0.3, simple=True)
plot.add_scatter(pointsize=25, alpha=0.7)

# Add styling elements
plot.add_grid()
plot.add_title("Grouped by Category")
plot.add_legend(title="Category")

# Direct access to the Seaborn Objects plot (this is what a fixed CircumplexPlot.show() would do)
plot.plot.plot(pyplot=True)

# %%
# Example 4: Advanced customization with annotations
print("Builder pattern - advanced customization with annotations")

# Create a plot with both scatter and annotations
plot = CircumplexPlot(data)
plot.add_scatter(pointsize=20, alpha=0.7)
plot.add_grid(diagonal_lines=True)

# Add annotations for points of interest
plot.add_annotation(0, text="Point of interest", x_offset=0.2, y_offset=0.1)
plot.add_annotation(10, text="Another point", x_offset=-0.2, y_offset=0.2)

# Add title
plot.add_title("Grammar of Graphics: Adding Annotations")

# Direct access to the Seaborn Objects plot (this is what a fixed CircumplexPlot.show() would do)
plot.plot.plot(pyplot=True)

# %%
# Example 5: Faceting with the builder pattern
print("Builder pattern - faceting")

# Create a categorical variable for faceting
data["facet_var"] = np.random.choice(["Segment 1", "Segment 2"], size=len(data))

# Create a faceted plot using the builder pattern
plot = CircumplexPlot(data)  # Avoid hue for now to prevent palette issues
plot.add_scatter(pointsize=20, alpha=0.7)
plot.add_grid(diagonal_lines=True)
plot.facet(column="facet_var")  # The 'column' parameter creates column facets
plot.add_title("Faceted Plot by Segment")

# Direct access to the Seaborn Objects plot (this is what a fixed CircumplexPlot.show() would do)
plot.plot.plot(pyplot=True)

# %%
# ## 4. Simple User-Facing API Examples

"""
The module provides simplified high-level functions for common use cases.
These functions use the CircumplexPlot class internally, but provide a simpler
interface for basic use cases.
"""

# Simple scatter plot with the high-level API
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter_plot(data, title="Simple API: Scatter Plot")

# Simple density plot with the high-level API
plt.subplot(1, 2, 2)
density_plot(
    data,
    title="Simple API: Density Plot",
    diagonal_lines=True,
    simple_density=True,
    incl_scatter=True,
    scatter_alpha=0.3,
)
plt.tight_layout()

# %%
# Joint plot with marginals using the simple API
joint_plot(
    data,
    title="Simple API: Joint Plot with Marginals",
    plot_type="density",
    incl_scatter=True,
)

# %%
# Multiple subplots comparing different random data samples
datasets = [create_simulated_data(n=50) for _ in range(4)]
subtitles = ["Sample A", "Sample B", "Sample C", "Sample D"]

# Create multiple subplots with a single function call
create_circumplex_subplots(
    datasets,
    subtitles=subtitles,
    title="Simple API: Multiple Subplot Comparison",
    plot_type="density",
    incl_scatter=True,
    diagonal_lines=True,
)

# %%
# ## 5. Integration with Matplotlib

"""
The plotting functions can integrate with existing matplotlib workflows by
returning matplotlib objects or working with provided axes.
"""

# Create a custom layout with matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot directly on the first axis
scatter_plot(data, title="Scatter on Custom Axes", diagonal_lines=True, ax=axes[0])

# Plot on the second axis
density_plot(
    data,
    title="Density on Custom Axes",
    simple_density=True,
    incl_scatter=True,
    ax=axes[1],
)

# Add a global title
fig.suptitle("Integration with Matplotlib", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

# %%
# Integration with Seaborn Objects with as_objects=True parameter

# Get a Seaborn Objects plot directly
so_plot = scatter_plot(data, title="Getting Seaborn Objects Directly", as_objects=True)

# You can further customize the Seaborn Objects plot
so_plot = so_plot.theme({"axes.grid": True, "grid.color": ".8", "font.family": "serif"})

# And display it
so_plot.show()

# %%
# ## 6. Advanced Composite Examples

"""
These examples show more complex visualizations that combine multiple elements and techniques.
"""

# Create a dataset with groups for more complex examples
advanced_data = create_simulated_data(n=150)
advanced_data["group"] = np.random.choice(["A", "B", "C"], size=len(advanced_data))

# Example: Combine density contours with annotations
# Create a plot with both density contours and data points
plot = (
    CircumplexPlot(advanced_data, hue="group")
    .add_density(alpha=0.2, fill=True)
    .add_scatter(pointsize=15, alpha=0.6)
)

# Add grid with diagonal lines and quadrant labels
plot.add_grid(diagonal_lines=True)

# Add annotations for the group centroids
for group in advanced_data["group"].unique():
    group_data = advanced_data[advanced_data["group"] == group]
    x_mean = group_data["ISOPleasant"].mean()
    y_mean = group_data["ISOEventful"].mean()

    # Find the closest point to the centroid
    distances = np.sqrt(
        (group_data["ISOPleasant"] - x_mean) ** 2
        + (group_data["ISOEventful"] - y_mean) ** 2
    )
    closest_idx = distances.idxmin()

    # Add annotation
    plot.add_annotation(
        closest_idx,
        text=f"Group {group} centroid",
        x_offset=0.1 if group != "B" else -0.2,
        y_offset=0.1,
    )

# Add title and legend
plot.add_title("Advanced Example: Annotating Group Centroids")
plot.add_legend(title="Group")

# Direct access to the Seaborn Objects plot (this is what a fixed CircumplexPlot.show() would do)
plot.plot.plot(pyplot=True)

# %%
# Statistical comparison example between two datasets

# Create two datasets to compare
data1 = create_simulated_data(n=50)
data1["dataset"] = "Urban"
data2 = create_simulated_data(n=50)
data2["dataset"] = "Rural"

# Slightly shift data2 to make it more interesting
data2["ISOPleasant"] = data2["ISOPleasant"] + 0.3
data2["ISOEventful"] = data2["ISOEventful"] - 0.2

# Combine datasets
combined_data = pd.concat([data1, data2])

# Create a comparative plot without using hue in the constructor
plot = CircumplexPlot(combined_data)  # Avoid using hue in constructor
plot.hue = "dataset"  # Set hue after construction to avoid palette issue
plot.add_density(alpha=0.2, simple=True)
plot.add_scatter(pointsize=20, alpha=0.6)
plot.add_grid(diagonal_lines=True)
plot.add_title("Advanced Example: Comparing Urban vs Rural Soundscapes")
plot.add_legend(title="Location Type")

# Direct access to the Seaborn Objects plot (this is what a fixed CircumplexPlot.show() would do)
plot.plot.plot(pyplot=True)
