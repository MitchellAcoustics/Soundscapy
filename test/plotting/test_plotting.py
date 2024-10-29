"""
Test suite for soundscapy plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest

from soundscapy.plotting import (
    Backend,
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
    ax = scatter_plot(sample_data, backend=Backend.SEABORN)
    return ax.figure


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", filename="test_density_plot.png"
)
def test_density_plot_image(sample_data):
    """Test density plot image comparison."""
    ax = density_plot(sample_data, backend=Backend.SEABORN)
    return ax.figure


def test_scatter_plot_seaborn(sample_data):
    """Test scatter plot with Seaborn backend."""
    ax = scatter_plot(sample_data, backend=Backend.SEABORN)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "ISOPleasant"
    assert ax.get_ylabel() == "ISOEventful"
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (-1, 1)
    assert ax.get_title() == "Soundscape Scatter Plot"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_scatter_plot_plotly(sample_data):
    """Test scatter plot with Plotly backend."""
    fig = scatter_plot(sample_data, backend=Backend.PLOTLY)
    assert isinstance(fig, go.Figure)


def test_density_plot_seaborn(sample_data):
    """Test density plot with Seaborn backend."""
    ax = density_plot(sample_data, backend=Backend.SEABORN)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "ISOPleasant"
    assert ax.get_ylabel() == "ISOEventful"
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (-1, 1)
    assert ax.get_title() == "Soundscape Density Plot"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_density_plot_plotly(sample_data):
    """Test density plot with Plotly backend."""
    # Update when density plots are implemented for Plotly
    with pytest.raises(NotImplementedError):
        fig = density_plot(sample_data, backend=Backend.PLOTLY)
        assert isinstance(fig, go.Figure)


def test_create_circumplex_subplots(sample_data):
    """Test creation of circumplex subplots."""
    fig = create_circumplex_subplots(
        [sample_data, sample_data], plot_type=PlotType.SCATTER
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_simple_density_plot_type(sample_data):
    fig = create_circumplex_subplots(
        [sample_data, sample_data], plot_type=PlotType.SIMPLE_DENSITY, nrows=1, ncols=2
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_circumplex_plot_seaborn(sample_data):
    """Test CircumplexPlot with Seaborn backend."""
    plot = CircumplexPlot(sample_data, backend=Backend.SEABORN)
    plot.scatter()
    assert isinstance(plot.get_axes(), plt.Axes)
    plot.density()
    assert isinstance(plot.get_axes(), plt.Axes)
    plot.jointplot()
    assert isinstance(plot.get_axes(), plt.Axes)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_circumplex_plot_plotly(sample_data):
    """Test CircumplexPlot with Plotly backend."""
    plot = CircumplexPlot(sample_data, backend=Backend.PLOTLY)
    plot.scatter()
    assert isinstance(plot.get_figure(), go.Figure)
    with pytest.raises(NotImplementedError):
        # Update when density plots are implemented for Plotly
        plot.density()
        assert isinstance(plot.get_figure(), go.Figure)


def test_style_options(sample_data):
    """Test updating style options."""
    plot = CircumplexPlot(sample_data, backend=Backend.SEABORN)
    plot.update_style_options(figsize=(8, 8))
    plot.scatter()
    fig = plot.get_figure()[0]
    assert np.array_equal(fig.get_size_inches(), np.array((8, 8)))


def test_invalid_backend():
    """Test invalid backend raises ValueError."""
    with pytest.raises(ValueError):
        CircumplexPlot(pd.DataFrame(), backend="invalid_backend")


def test_invalid_plot_type(sample_data):
    """Test invalid plot type raises ValueError."""
    with pytest.raises(KeyError):
        create_circumplex_subplots([sample_data], plot_type="invalid_plot_type")


def test_simple_density(sample_data):
    plot = CircumplexPlot(sample_data, backend=Backend.SEABORN)
    plot.simple_density()
    assert isinstance(plot.get_axes(), plt.Axes)


def test_simple_density_with_custom_params(sample_data):
    from soundscapy.plotting.circumplex_plot import CircumplexPlotParams

    params = CircumplexPlotParams(fill=False)
    plot = CircumplexPlot(sample_data, backend=Backend.SEABORN, params=params)
    plot.simple_density()
    assert isinstance(plot.get_axes(), plt.Axes)


def test_simple_density_with_custom_axes(sample_data):
    plot = CircumplexPlot(sample_data)
    fig, ax = plt.subplots()
    plot.simple_density(ax=ax)
    fig = plot.get_figure()
    assert isinstance(fig[0], plt.Figure)
