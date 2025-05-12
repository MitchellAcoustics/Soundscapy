"""
Tests for the ParamModel class and its subclasses.
"""

import pytest

from soundscapy.plotting.param_models import (
    DensityParams,
    ScatterParams,
    SeabornParams,
    StyleParams,
)
from src.soundscapy.plotting.param_models import ParamModel


def test_create_instance_with_known_type():
    """Test that ParamModel.create correctly creates an instance with known type."""
    from soundscapy.plotting.param_models import ScatterParams

    instance = ScatterParams(param1=42)
    assert isinstance(instance, ScatterParams)
    assert instance.get("param1") == 42


def test_update_parameters():
    """Test the update method for modifying attributes."""
    model = ParamModel()
    model.update(param1=42, param2=None, extra="allow", ignore_null=True)

    assert model.get("param1") == 42
    assert model.get("param2") is None


def test_forbid_extra_parameters():
    """Test that update forbids extra fields when 'extra' is set to forbid."""
    model = ParamModel()

    with pytest.raises(ValueError, match="Unknown parameters: {'extra_param'}"):
        model.update(extra="forbid", extra_param=42)


def test_ignore_extra_parameters():
    """Test that update ignores extra fields when 'extra' is set to ignore."""
    model = ScatterParams()
    model.update(extra="ignore", s=42, extra_param=99)

    assert model.get("s") == 42
    assert model.get("extra_param") is None


def test_as_dict_conversion():
    """Test that as_dict properly converts model fields to a dictionary."""
    model = ParamModel(param1=42, param2="example")
    model_dict = model.as_dict()

    assert model_dict["param1"] == 42
    assert model_dict["param2"] == "example"


def test_get_changed_params():
    """Test that changed parameters are properly returned."""
    model = SeabornParams(param1=42, x="ISOPleasant")
    model.param1 = 100  # Modify a parameter
    changed_params = model.get_changed_params()

    assert "param1" in changed_params
    assert changed_params["param1"] == 100
    assert "x" not in changed_params


def test_getitem_behavior():
    """Test the dictionary-style access to model parameters."""
    model = ParamModel(param1=42)
    assert model["param1"] == 42

    with pytest.raises(KeyError):
        _ = model["missing"]


def test_pop_behavior():
    """Test pop method removes and returns the parameter value."""
    model = ParamModel(param1=42)
    value = model.pop("param1")

    assert value == 42
    assert "param1" not in model.get_changed_params()


def test_drop_behavior_with_existing_key():
    """Test drop method removes an existing key."""
    model = ParamModel(param1=42, param2="example")
    model.drop("param1")

    assert "param1" not in model.model_fields_set
    assert model.get("param1") is None


def test_drop_behavior_ignore_missing():
    """Test drop method ignores a missing key when ignore_missing=True."""
    model = ParamModel(param1=42)
    model.drop("missing_key", ignore_missing=True)

    assert model.get("param1") == 42


def test_field_names_property():
    """Test that defined_field_names property returns all field names."""
    # Note: defined_field_names refers to the predefined attributes of the model
    # Extra params do not count
    model = ScatterParams(s=42, param2="example")
    assert model["s"] == 42
    assert model.get("s") == 42
    assert "s" in model.defined_field_names
    assert "param2" not in model.defined_field_names


def test_param_models_instantiation():
    """Test creating parameter models using the factory method."""
    # Test creating various parameter types
    scatter_params = ScatterParams(x="test_x", y="test_y")
    density_params = DensityParams(alpha=0.7)
    style_params = StyleParams(linewidth=2.0)

    # Verify type-specific defaults and overrides work
    assert scatter_params.x == "test_x"  # Override
    assert scatter_params.y == "test_y"  # Override
    assert scatter_params.s == 20  # Default

    # Verify density defaults
    assert density_params.alpha == 0.7  # Override
    assert density_params.x == "ISOPleasant"  # Default
    assert density_params.fill is True  # Default

    # Verify style defaults
    assert style_params.linewidth == 2.0  # Override
    assert style_params.xlim == (-1, 1)  # Default
    assert style_params.primary_lines is True  # Default

    # Verify inheritance works
    assert scatter_params.alpha == 0.8  # Default from SeabornParams


def test_param_model_update():
    """Test updating parameter models."""
    # Create a parameter model and update it
    params = ScatterParams()

    # Initial defaults
    assert params.x == "ISOPleasant"

    # Update and self setting
    params.update(x="new_x", s=30)

    # Check updates applied
    assert params.x == "new_x"
    assert params.s == 30

    # None values should not override existing values
    params.update(x=None)
    assert params.x == "new_x"  # Still has the updated value


def test_scatter_params_as_dict():
    """Test converting parameter models to dictionaries."""
    # Create a parameter model with some custom values
    params = ScatterParams(x="custom_x", color="red")

    # Convert to dict
    params_dict = params.as_dict()

    # Dict should contain both defaults and custom values
    assert params_dict["x"] == "custom_x"
    assert params_dict["color"] == "red"
    assert params_dict["s"] == 20

    # Update and check dict again
    params.update(s=50)
    params_dict = params.as_dict()
    assert params_dict["s"] == 50
