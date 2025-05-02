# %%
"""
# Soundscapy Plotting Module - Minimal Examples

This notebook demonstrates how the Soundscapy plotting module works
with minimal examples focused on the high-level API.
"""

import matplotlib.pyplot as plt

# Import from soundscapy
from soundscapy.plotting import (
    create_circumplex_subplots,
    density_plot,
    joint_plot,
    scatter_plot,
)
from soundscapy.surveys.processing import simulation

# %%
# Create sample data for our examples
data = simulation(n=200, incl_iso_coords=True)
print(f"Generated simulated data with {len(data)} samples")
data.head()

# %%
# Simple scatter and density plots

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter_plot(data, title="Scatter Plot")

plt.subplot(1, 2, 2)
density_plot(
    data,
    title="Density Plot",
    diagonal_lines=True,
    simple_density=True,
    incl_scatter=True,
    scatter_alpha=0.3,
)
plt.tight_layout()

# %%
# Joint plot with marginals
joint_plot(
    data, title="Joint Plot with Marginals", plot_type="density", incl_scatter=True
)

# %%
# Multiple subplots comparison
datasets = [simulation(n=50, incl_iso_coords=True) for _ in range(4)]
subtitles = ["Sample A", "Sample B", "Sample C", "Sample D"]

create_circumplex_subplots(
    datasets,
    subtitles=subtitles,
    title="Multiple Subplot Comparison",
    plot_type="density",
    incl_scatter=True,
    diagonal_lines=True,
)
