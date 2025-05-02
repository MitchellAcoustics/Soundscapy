"""Script to generate baseline images for pytest-mpl comparisons."""

import os

import matplotlib.pyplot as plt
from numpy import random

from soundscapy.plotting import Backend, density_plot, scatter_plot
from soundscapy.surveys.processing import simulation

# Set random seed for reproducibility
random.seed(42)

# Create directory for baseline images if it doesn't exist
BASELINE_DIR = "/Users/mitch/Documents/GitHub/Soundscapy/test/baseline"
os.makedirs(BASELINE_DIR, exist_ok=True)

# Generate sample data
sample_data = simulation(n=100, incl_iso_coords=True)

# Generate and save scatter plot
scatter_ax = scatter_plot(sample_data, backend=Backend.SEABORN)
scatter_ax.figure.savefig(os.path.join(BASELINE_DIR, "test_scatter_plot.png"))
plt.close(scatter_ax.figure)

# Generate and save density plot
density_ax = density_plot(sample_data, backend=Backend.SEABORN)
density_ax.figure.savefig(os.path.join(BASELINE_DIR, "test_density_plot.png"))
plt.close(density_ax.figure)

print("Baseline images generated successfully.")
