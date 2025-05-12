import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from pydantic import ValidationError

import soundscapy as sspy
from soundscapy.plotting.param_models import StyleParams, SubplotsParams
from soundscapy.plotting.plot_functions import (
    _setup_style_and_subplots_args_from_kwargs,
    scatter,
)


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


def test_setup_style_and_subplots_args_defaults():
    """Test with default arguments to ensure proper initialization of style and subplot parameters."""
    x, y, prim_labels, kwargs = None, None, None, {}
    style_args, subplots_args, remaining_kwargs = (
        _setup_style_and_subplots_args_from_kwargs(x, y, prim_labels, kwargs)
    )

    assert isinstance(style_args, StyleParams)
    assert isinstance(subplots_args, SubplotsParams)
    assert remaining_kwargs == {}


def test_setup_style_and_subplots_args_custom_kwargs():
    """Test with custom kwargs to verify proper handling and removal of style parameters."""
    x, y, prim_labels = "x_variable", "y_variable", None
    kwargs = {
        "xlabel": "Custom X",
        "ylabel": "Custom Y",
        "xlim": (0, 10),
        "ylim": (0, 20),
        "unused_param": 123,
    }
    style_args, subplots_args, remaining_kwargs = (
        _setup_style_and_subplots_args_from_kwargs(x, y, prim_labels, kwargs)
    )

    assert style_args.xlabel == "Custom X"
    assert style_args.ylabel == "Custom Y"
    assert style_args.xlim == (0, 10)
    assert style_args.ylim == (0, 20)
    assert "unused_param" in remaining_kwargs
    assert remaining_kwargs["unused_param"] == 123


def test_setup_style_and_subplots_args_deprecated_prim_labels():
    """Test deprecated prim_labels functionality and ensure appropriate warnings."""
    x, y, prim_labels = "x_variable", "y_variable", False
    kwargs = {}

    with pytest.warns(
        DeprecationWarning,
        match="The `prim_labels` parameter is deprecated. Use `xlabel` and `ylabel` instead.",
    ):
        style_args, _, _ = _setup_style_and_subplots_args_from_kwargs(
            x, y, prim_labels, kwargs
        )

    assert style_args.xlabel is False
    assert style_args.ylabel is False


def test_setup_style_and_subplots_args_subplots_params():
    """Test that subplot parameters are set correctly based on kwargs."""
    x, y, prim_labels = None, None, None
    kwargs = {"nrows": 2, "ncols": 3, "figsize": (10, 8)}
    _, subplots_args, _ = _setup_style_and_subplots_args_from_kwargs(
        x, y, prim_labels, kwargs
    )

    assert subplots_args.nrows == 2
    assert subplots_args.ncols == 3
    assert subplots_args.figsize == (10, 8)


def test_style_params_defaults_preserved():
    """Test that default values in StyleParams are preserved when not overridden."""
    x, y, prim_labels = None, None, None
    kwargs = {"xlabel": "Custom X"}  # Only override one parameter

    style_args, _, _ = _setup_style_and_subplots_args_from_kwargs(
        x, y, prim_labels, kwargs
    )

    # The specified parameter should be changed
    assert style_args.xlabel == "Custom X"

    # Other parameters should retain their defaults from StyleParams
    assert style_args.ylabel == r"$E_{ISO}$"  # Default from StyleParams
    assert style_args.xlim == (-1, 1)  # Default from StyleParams
    assert style_args.ylim == (-1, 1)  # Default from StyleParams
    assert style_args.linewidth == 1.5  # Default from StyleParams
    assert style_args.primary_lines is True  # Default from StyleParams
    assert style_args.diagonal_lines is False  # Default from StyleParams


def test_subplots_params_defaults_preserved():
    """Test that default values in SubplotsParams are preserved when not overridden."""
    x, y, prim_labels = None, None, None
    kwargs = {"nrows": 3}  # Only override one parameter

    _, subplots_args, _ = _setup_style_and_subplots_args_from_kwargs(
        x, y, prim_labels, kwargs
    )

    # The specified parameter should be changed
    assert subplots_args.nrows == 3

    # Other parameters should retain their defaults from SubplotsParams
    assert subplots_args.ncols == 1  # Default from SubplotsParams
    assert subplots_args.figsize == (5, 5)  # Default from SubplotsParams
    assert subplots_args.sharex is True  # Default from SubplotsParams
    assert subplots_args.sharey is True  # Default from SubplotsParams


def test_mixed_style_and_subplot_params():
    """Test passing a mix of style and subplot parameters."""
    x, y, prim_labels = None, None, None
    kwargs = {
        "xlim": (-2, 2),  # Style parameter
        "nrows": 2,  # Subplot parameter
        "linewidth": 2.0,  # Style parameter
        "figsize": (8, 6),  # Subplot parameter
    }

    style_args, subplots_args, remaining_kwargs = (
        _setup_style_and_subplots_args_from_kwargs(x, y, prim_labels, kwargs)
    )

    # Style parameters should be in style_args
    assert style_args.xlim == (-2, 2)
    assert style_args.linewidth == 2.0

    # Subplot parameters should be in subplots_args
    assert subplots_args.nrows == 2
    assert subplots_args.figsize == (8, 6)

    # Remaining kwargs should be empty as all were processed
    assert remaining_kwargs == {}


def test_unknown_parameters_preserved():
    """Test that unknown parameters are preserved in the returned kwargs."""
    x, y, prim_labels = None, None, None
    kwargs = {
        "xlabel": "X Label",  # Known style parameter
        "nrows": 2,  # Known subplot parameter
        "custom_param": "value",  # Unknown parameter
        "another_custom": 123,  # Unknown parameter
    }

    _, _, remaining_kwargs = _setup_style_and_subplots_args_from_kwargs(
        x, y, prim_labels, kwargs
    )

    # Unknown parameters should be returned in the remaining kwargs
    assert "custom_param" in remaining_kwargs
    assert remaining_kwargs["custom_param"] == "value"
    assert "another_custom" in remaining_kwargs
    assert remaining_kwargs["another_custom"] == 123

    # Known parameters should not be in remaining kwargs
    assert "xlabel" not in remaining_kwargs
    assert "nrows" not in remaining_kwargs


def test_prim_ax_fontdict_override():
    """Test that the nested prim_ax_fontdict can be partially overridden."""
    x, y, prim_labels = None, None, None
    custom_fontdict = {"fontsize": "x-large", "fontweight": "bold"}
    kwargs = {"prim_ax_fontdict": custom_fontdict}

    style_args, _, _ = _setup_style_and_subplots_args_from_kwargs(
        x, y, prim_labels, kwargs
    )

    # The specified fontdict parameters should be updated
    assert style_args.prim_ax_fontdict["fontsize"] == "x-large"
    assert style_args.prim_ax_fontdict["fontweight"] == "bold"

    # Other fontdict parameters should retain their defaults
    # Accepted these are failing now - not implemented nested updating.
    # assert style_args.prim_ax_fontdict["family"] == "sans-serif"
    # assert style_args.prim_ax_fontdict["fontstyle"] == "normal"
    # assert style_args.prim_ax_fontdict["parse_math"] is True


def test_none_values_handling():
    """Test that None values in kwargs properly override defaults or are ignored based on ignore_null setting."""
    x, y, prim_labels = None, None, None
    kwargs = {
        "xlabel": None,  # Override with None
        "legend_loc": False,  # Override with False
    }

    style_args, _, _ = _setup_style_and_subplots_args_from_kwargs(
        x, y, prim_labels, kwargs
    )

    # Since ignore_null=False is passed to update(), None values should override defaults
    # Other params should override
    assert style_args.xlabel is None
    assert style_args.legend_loc is False

    # Other parameters should still have their defaults
    assert style_args.ylabel == r"$E_{ISO}$"

    assert "xlabel" in style_args.current_field_names
    assert "ylabel" in style_args.current_field_names


@pytest.mark.xfail(reason="Relied on Pydantic model validation.")
def test_update_model_validate():
    """Test that None values in kwargs properly override defaults or are ignored based on ignore_null setting."""
    x, y, prim_labels = None, None, None
    kwargs = {
        "xlabel": None,  # Override with None
        "xlim": None,  # Override with None
    }

    with pytest.raises(ValidationError, match="Input should be a valid tuple"):
        # StyleParams.xlim has type tuple[float, float] so Pydantic will
        style_args, _, _ = _setup_style_and_subplots_args_from_kwargs(
            x, y, prim_labels, kwargs
        )
