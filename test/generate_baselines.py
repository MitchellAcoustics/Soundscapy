"""Script to generate baseline images for pytest-mpl comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt

from soundscapy.plotting import Backend, density_plot, scatter_plot
from soundscapy.sspylogging import get_logger
from soundscapy.surveys.processing import simulation

logger = get_logger()

# Create directory for baseline images if it doesn't exist
BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

# Generate sample data
sample_data = simulation(n=100, seed=42, incl_iso_coords=True)

# Generate and save scatter plot
scatter_ax = scatter_plot(sample_data, backend=Backend.SEABORN)
scatter_ax.figure.savefig(BASELINE_DIR.joinpath("test_scatter_plot.png"))
plt.close(scatter_ax.figure)

# Generate and save density plot
density_ax = density_plot(sample_data, backend=Backend.SEABORN)
density_ax.figure.savefig(BASELINE_DIR.joinpath("test_density_plot.png"))
plt.close(density_ax.figure)

logger.info("Baseline images generated successfully.")
