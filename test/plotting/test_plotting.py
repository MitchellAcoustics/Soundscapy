"""
Test suite for soundscapy plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn.objects as so

from soundscapy.plotting import (
    CircumplexPlot,
    PlotType,
    create_circumplex_subplots,
    density_plot,
    scatter_plot,
)
from soundscapy.surveys.processing import simulation


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)  # For reproducibility
    return simulation(n=100, incl_iso_coords=True)


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", filename="test_scatter_plot.png"
)
def test_scatter_plot_image(sample_data):
    """Test scatter plot image comparison."""
    ax = scatter_plot(sample_data)
    return ax.figure


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", filename="test_density_plot.png"
)
def test_density_plot_image(sample_data):
    """Test density plot image comparison."""
    ax = density_plot(sample_data)
    return ax.figure


def test_scatter_plot(sample_data):
    """Test scatter plot functionality."""
    # Create a figure and axis for the test
    fig, ax = plt.subplots()
    # Plot directly on the provided axis
    result_ax = scatter_plot(sample_data, ax=ax)
    # It should return the same axis
    assert result_ax is ax
    # Check axis properties
    assert ax.get_xlabel() == "ISOPleasant"
    assert ax.get_ylabel() == "ISOEventful"
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (-1, 1)
    assert ax.get_title() == "Soundscape Scatter Plot"


def test_scatter_plot_as_objects(sample_data):
    """Test scatter plot with objects return type."""
    plot = scatter_plot(sample_data, as_objects=True)
    assert isinstance(plot, so.Plot)


def test_density_plot(sample_data):
    """Test density plot functionality."""
    # Create a figure and axis for the test
    fig, ax = plt.subplots()
    # Plot directly on the provided axis
    result_ax = density_plot(sample_data, ax=ax)
    # It should return the same axis
    assert result_ax is ax
    # Check axis properties
    assert ax.get_xlabel() == "ISOPleasant"
    assert ax.get_ylabel() == "ISOEventful"
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (-1, 1)
    assert ax.get_title() == "Soundscape Density Plot"


def test_density_plot_as_objects(sample_data):
    """Test density plot with objects return type."""
    plot = density_plot(sample_data, as_objects=True)
    assert isinstance(plot, so.Plot)


def test_create_circumplex_subplots(sample_data):
    """Test creation of circumplex subplots."""
    fig = create_circumplex_subplots(
        [sample_data, sample_data], plot_type=PlotType.SCATTER
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_simple_density_plot_type(sample_data):
    fig = create_circumplex_subplots(
        [sample_data, sample_data], plot_type="simple_density", nrows=1, ncols=2
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_circumplex_plot_methods(sample_data):
    """Test CircumplexPlot methods."""
    # Test scatter method
    plot = CircumplexPlot(sample_data)
    plot.scatter()
    assert isinstance(plot.get_axes(), plt.Axes)

    # Test density method
    plot = CircumplexPlot(sample_data)
    plot.density()
    assert isinstance(plot.get_axes(), plt.Axes)

    # Test jointplot method
    plot = CircumplexPlot(sample_data)
    plot.jointplot()
    assert isinstance(plot.get_axes(), plt.Axes)


def test_plot_size(sample_data):
    """Test customizing plot size."""
    plot = CircumplexPlot(sample_data)
    plot.scatter()
    fig, _ = plot.build(as_objects=False)
    # Default size should be 6x6
    assert np.array_equal(fig.get_size_inches(), np.array((6, 6)))


def test_builder_pattern(sample_data):
    """Test the builder pattern API."""
    plot = CircumplexPlot(sample_data).add_scatter().add_grid().add_title("Test Title")

    # Verify the plot was built correctly
    assert plot.has_scatter is True
    assert plot.has_grid is True


def test_invalid_plot_type(sample_data):
    """Test invalid plot type gets treated as default."""
    # Invalid types no longer raise errors - they just fall back to default behavior
    fig = create_circumplex_subplots([sample_data], plot_type="invalid_plot_type")
    assert isinstance(fig, plt.Figure)


def test_simple_density(sample_data):
    """Test simple density plot functionality."""
    plot = CircumplexPlot(sample_data)
    plot.simple_density()
    assert isinstance(plot.get_axes(), plt.Axes)


def test_annotations(sample_data):
    """Test annotation functionality."""
    plot = CircumplexPlot(sample_data)
    plot.add_scatter()
    plot.add_annotation(0)
    plot.add_grid()

    # Just testing that it doesn't error since we can't easily check annotation
    assert plot.has_scatter is True
