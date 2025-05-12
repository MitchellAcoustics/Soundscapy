"""Test suite for deprecated soundscapy plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import default_rng

from soundscapy import create_iso_subplots
from soundscapy.plotting import (
    Backend,
    CircumplexPlot,
    PlotType,
    create_circumplex_subplots,
    density_plot,
    scatter,
    scatter_plot,
)
from soundscapy.surveys.processing import simulation

rng = default_rng(42)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)  # For reproducibility  # noqa: NPY002
    return simulation(n=100, incl_iso_coords=True)


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", filename="test_scatter_plot.png"
)
def test_scatter_plot_image(sample_data: pd.DataFrame):
    """Test scatter plot image comparison."""
    with pytest.deprecated_call():
        ax = scatter_plot(sample_data, backend=Backend.SEABORN)
    return ax.figure


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", filename="test_density_plot.png"
)
def test_density_plot_image(sample_data: pd.DataFrame):
    """Test density plot image comparison."""
    with pytest.deprecated_call():
        ax = density_plot(sample_data, backend=Backend.SEABORN)
    return ax.figure


def test_scatter_plot_seaborn(sample_data: pd.DataFrame):
    """Test scatter plot with Seaborn backend."""
    ax = scatter(sample_data)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "$P_{ISO}$"
    assert ax.get_ylabel() == "$E_{ISO}$"
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (-1, 1)
    assert ax.get_title() == "Soundscape Scatter Plot"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_scatter_plot_plotly(sample_data: pd.DataFrame):
    """Test scatter plot with Plotly backend."""
    with pytest.deprecated_call():
        _ = scatter_plot(sample_data, backend=Backend.PLOTLY)


def test_density_plot_seaborn(sample_data: pd.DataFrame):
    """Test density plot with Seaborn backend."""
    with pytest.deprecated_call():
        ax = density_plot(sample_data, backend=Backend.SEABORN)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "$P_{ISO}$"
        assert ax.get_ylabel() == "$E_{ISO}$"
        assert ax.get_xlim() == (-1, 1)
        assert ax.get_ylim() == (-1, 1)
        assert ax.get_title() == "Soundscape Density Plot"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_density_plot_plotly(sample_data: pd.DataFrame):
    """DEPRECATED: Test density plot with Plotly backend."""
    with pytest.deprecated_call():
        fig = density_plot(sample_data, backend=Backend.PLOTLY)
    with pytest.raises(NameError):
        # plotly not installed, so go not defined
        assert isinstance(fig, go.Figure)


def test_create_circumplex_subplots(sample_data: pd.DataFrame):
    """Test creation of circumplex subplots."""
    with pytest.deprecated_call():
        fig = create_circumplex_subplots(
            [sample_data, sample_data], plot_type=PlotType.SCATTER
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2


def test_simple_density_plot_type(sample_data: pd.DataFrame):
    """Test creation of circumplex subplots with simple density."""
    with pytest.deprecated_call():
        fig = create_circumplex_subplots(
            [sample_data, sample_data],
            plot_type=PlotType.SIMPLE_DENSITY,
            nrows=1,
            ncols=2,
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2


def test_circumplex_plot_seaborn(sample_data: pd.DataFrame):
    """DEPRECATED: Test CircumplexPlot with Seaborn backend."""
    with pytest.raises(DeprecationWarning):  # noqa: PT012
        plot = CircumplexPlot(sample_data, backend=Backend.SEABORN)
        plot.scatter()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_axes(), Axes)  # type: ignore[reportAttributeAccessError]
        plot.density()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_axes(), Axes)  # type: ignore[reportAttributeAccessError]
        plot.jointplot()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_axes(), Axes)  # type: ignore[reportAttributeAccessError]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_circumplex_plot_plotly(sample_data: pd.DataFrame):
    """DEPRECATED: Test CircumplexPlot with Plotly backend."""
    with pytest.raises(DeprecationWarning):  # noqa: PT012
        plot = CircumplexPlot(sample_data, backend=Backend.PLOTLY)
        plot.scatter()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_figure(), go.Figure)  # type: ignore[reportAttributeAccessError]
        plot.density()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_figure(), go.Figure)  # type: ignore[reportAttributeAccessError]


def test_style_options(sample_data: pd.DataFrame):
    """Test updating style options."""
    with pytest.raises(DeprecationWarning):  # noqa: PT012
        plot = CircumplexPlot(sample_data, backend=Backend.SEABORN)
        plot.update_style_options(figsize=(8, 8))  # type: ignore[reportAttributeAccessError]
        plot.scatter()  # type: ignore[reportAttributeAccessError]
        fig = plot.get_figure()[0]  # type: ignore[reportAttributeAccessError]
        assert np.array_equal(fig.get_size_inches(), np.array((8, 8)))


def test_invalid_backend():
    """Test invalid backend raises ValueError."""
    with pytest.raises(DeprecationWarning):
        CircumplexPlot(pd.DataFrame(), backend="invalid_backend")


def test_invalid_plot_type(sample_data: pd.DataFrame):
    """Test invalid plot type raises ValueError."""
    with pytest.warns(UserWarning) as record:
        create_circumplex_subplots(
            [sample_data, sample_data], plot_type="invalid_plot_type"
        )


def test_no_subplots_needed_in_iso_subplots(sample_data: pd.DataFrame):
    """Test invalid plot type raises ValueError."""
    # TODO: This is actually testing _prepare_subplot_data
    with pytest.raises(ValueError, match="Only one subplot provided") as record:
        create_iso_subplots([sample_data], plot_layers="scatter")


def test_simple_density(sample_data: pd.DataFrame):
    """Test simple density plot."""
    with pytest.raises(DeprecationWarning):  # noqa: PT012
        plot = CircumplexPlot(sample_data, backend=Backend.SEABORN)
        plot.simple_density()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_axes(), Axes)  # type: ignore[reportAttributeAccessError]


def test_simple_density_with_custom_params(sample_data: pd.DataFrame):
    """Test simple density plot with custom parameters."""
    with pytest.raises(ImportError):
        from soundscapy.plotting.circumplex_plot import (  # type: ignore[reportMissingImports]
            CircumplexPlotParams,
        )

    with pytest.raises(DeprecationWarning):  # noqa: PT012
        from soundscapy.plotting import CircumplexPlotParams

        params = CircumplexPlotParams(fill=False)  # type: ignore[reportAttributeAccessError]
        plot = CircumplexPlot(sample_data, backend=Backend.SEABORN, params=params)
        plot.simple_density()  # type: ignore[reportAttributeAccessError]
        assert isinstance(plot.get_axes(), Axes)  # type: ignore[reportAttributeAccessError]


def test_simple_density_with_custom_axes(sample_data: pd.DataFrame):
    """Test simple density plot with custom axes."""
    with pytest.raises(DeprecationWarning):  # noqa: PT012
        plot = CircumplexPlot(sample_data)
        fig, ax = plt.subplots()
        plot.simple_density(ax=ax)  # type: ignore[reportAttributeAccessError]
        fig = plot.get_figure()  # type: ignore[reportAttributeAccessError]
        assert isinstance(fig[0], Figure)
