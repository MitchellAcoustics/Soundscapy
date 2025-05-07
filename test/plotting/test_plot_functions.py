import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import soundscapy as sspy
from soundscapy.plotting.plot_functions import scatter


@pytest.fixture
def data():
    data = sspy.isd.load()
    data, _ = sspy.isd.validate(data)
    return sspy.surveys.add_iso_coords(data)


@pytest.fixture
def simulated_data():
    data = sspy.surveys.simulation(1000)
    return sspy.surveys.add_iso_coords(data)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)  # For reproducibility
    return sspy.surveys.simulation(n=100, incl_iso_coords=True)


def test_scatter(simulated_data: pd.DataFrame):
    s = scatter(
        simulated_data,
        x="ISOPleasant",
        y="ISOEventful",
        xlabel="X Axis",
        ylabel="Y Axis",
    )
    assert type(s) is Axes


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", filename="test_scatter_plot.png"
)
def test_scatter_image(sample_data: pd.DataFrame):
    """Test scatter plot image comparison."""
    ax = scatter(sample_data)
    return ax.figure
