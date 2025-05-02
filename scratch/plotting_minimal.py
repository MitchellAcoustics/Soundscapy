# %%
"""
# Minimal CircumplexPlot Example with Fixed Implementation
"""

import matplotlib.pyplot as plt

from soundscapy.plotting import CircumplexPlot
from soundscapy.surveys.processing import simulation

# Create sample data
data = simulation(n=200, incl_iso_coords=True)
print(f"Generated simulated data with {len(data)} samples")

# Example 1: Basic Plot
plot = CircumplexPlot(data)
plot.add_scatter(pointsize=30, alpha=0.7)
plot.add_grid(diagonal_lines=True)
plot.add_title("Basic CircumplexPlot Example")
plot.show()  # This has been fixed to correctly display the plot

plt.figure()  # Create a new figure to avoid subplot errors

# Example 2: Adding hue
import numpy as np  # noqa: E402

data["group"] = np.random.choice(["Group A", "Group B", "Group C"], size=len(data))
plot2 = CircumplexPlot(data, hue="group")
plot2.add_scatter(pointsize=30, alpha=0.7)
plot2.add_grid(diagonal_lines=True)
plot2.add_title("CircumplexPlot with Hue Grouping")
plot2.show()  # Now works with color grouping
