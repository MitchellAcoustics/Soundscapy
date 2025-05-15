"""Tests for the ParamModel class and its subclasses."""

import pytest
from pydantic.dataclasses import dataclass as pydantic_dataclass

from soundscapy.plotting.param_models import (
    DensityParams,
    ParamModel,
    ScatterParams,
    SeabornParams,
    StyleParams,
)


@pydantic_dataclass(config={"extra": "allow", "arbitrary_types_allowed": True})
class TstParamModel(ParamModel):
    """A minimal subclass of ParamModel for testing with defined fields."""

    field1: str | None = None
    field2: int | None = 10  # Default value
    field3: float | None = None


# Basic ParamModel functionality tests
def test_create_instance_with_known_type():
    """Test that ParamModel.create correctly creates an instance with known type."""
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


def test_getitem_behavior():
    """Test the dictionary-style access to model parameters."""
    model = ParamModel(param1=42)
    assert model["param1"] == 42

    with pytest.raises(KeyError):
        _ = model["missing"]


# Tests for specific parameter model types
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


def test_get_changed_seaborn_params():
    """Test that changed parameters are properly returned."""
    model = SeabornParams(alpha=42, x="ISOPleasant")
    model.alpha = 100  # Modify a parameter
    changed_params = model.get_changed_params()

    assert "alpha" in changed_params
    assert changed_params["alpha"] == 100
    assert "x" not in changed_params


# Tests for update functionality
def test_update_valid_fields():
    """Test updating valid fields in the model."""
    model = TstParamModel()
    model.update(field1="value1", field2=123, extra="allow")
    assert model.get("field1") == "value1"
    assert model.get("field2") == 123


def test_update_ignore_null_functionality():
    """Test that None values are removed when ignore_null is True."""
    model = TstParamModel()
    # Test with ignore_null=True (default)
    model.update(field3=456, field2=None)
    assert model.get("field3") == 456
    assert model.get("field2") == 10  # Default value preserved

    # Test with ignore_null=False
    model.update(field2=None, ignore_null=False)
    assert model.get("field2") is None  # None value applied

    # Test with explicit ignore_null=True
    model = TstParamModel()
    model.update(field3=456, field2=None, ignore_null=True)
    assert model.get("field3") == 456
    assert model.get("field2") == 10  # Default value preserved


def test_update_overwrites_default():
    """Test that a default value is overwritten during update."""
    model = TstParamModel()
    assert model.get("field2") == 10  # Default value
    model.update(field2=99)
    assert model.get("field2") == 99  # Updated value


# Tests for get functionality
def test_get_with_default():
    """Test retrieving a field with a default fallback value."""
    model = TstParamModel()
    assert model.get("missing_field", default="default_value") == "default_value"


def test_getitem_existing_field():
    """Test dictionary-style access for existing fields."""
    model = TstParamModel(field1="value1")
    assert model["field1"] == "value1"


def test_getitem_nonexistent_field():
    """Test dictionary-style access raises KeyError for nonexistent fields."""
    model = TstParamModel()
    with pytest.raises(KeyError):
        _ = model["nonexistent"]


def test_as_dict_method():
    """Test that as_dict outputs the expected dictionary."""
    model = TstParamModel(field2=42, field3=3.14)
    result = model.as_dict()
    assert result["field2"] == 42
    assert result["field3"] == 3.14


def test_get_changed_params():
    """Test that changed parameters are returned correctly."""
    model = TstParamModel(field2=42)
    model.update(field2=99)
    changed_params = model.get_changed_params()
    assert changed_params["field2"] == 99


def test_get_multiple_fields():
    """Test retrieval of multiple fields as a dictionary."""
    model = TstParamModel(field1="value1", field2=99)
    multiple_fields = model.get_multiple(["field1", "field2", "nonexistent"])
    assert multiple_fields["field1"] == "value1"
    assert multiple_fields["field2"] == 99
    assert "nonexistent" not in multiple_fields


# Tests for pop functionality
def test_pop_behavior():
    """Test pop method removes and returns the parameter value."""
    model = ParamModel(param1=42)
    value = model.pop("param1")

    assert value == 42
    assert "param1" not in model.get_changed_params()


def test_pop_existing_field():
    """Test popping an existing field."""
    # Test popping a defined field
    model = TstParamModel(field3=3.14)
    value = model.pop("field3")
    assert value == 3.14
    # pop resets to default for defined fields
    assert "field3" in model.as_dict()

    # Test popping a custom field
    model.update(field4=20)
    value = model.pop("field4")
    assert value == 20
    with pytest.raises(KeyError):
        _ = model["field4"]

    # Test popping a defined field with default value
    model2 = TstParamModel(field2=20)
    value = model2.pop("field2")
    assert value == 20
    assert model2["field2"] == 10  # .pop Resets to default


def test_pop_nonexistent_field():
    """Test popping a nonexistent field raises KeyError."""
    model = TstParamModel()
    with pytest.raises(KeyError):
        model.pop("nonexistent")


# Tests for drop functionality
def test_drop_behavior_with_existing_key():
    """
    Test drop method removes an existing key.

    If dropping a defined field, drop resets to default
    """
    model = ParamModel(param1=42, param2="example")
    model.drop("param1")

    assert "param1" not in model.get_changed_params()
    assert model.get("param1") is None


def test_drop_behavior_ignore_missing():
    """Test drop method ignores a missing key when ignore_missing=True."""
    model = ParamModel(param1=42)
    model.drop("missing_key", ignore_missing=True)

    assert model.get("param1") == 42


def test_drop_single_field():
    """Test dropping a single field."""
    # Test dropping a defined field
    model = TstParamModel(field1="value1")
    model.drop("field1")
    with pytest.raises(KeyError):
        _ = model["field1"]

    # Test dropping a custom field
    model.field4 = "value4"
    assert model.field4 == "value4"
    model.drop("field4")
    assert "field4" not in model.as_dict()


def test_drop_multiple_fields():
    """Test dropping multiple fields."""
    # Test dropping defined fields
    model1 = TstParamModel(field2=42, field3=3.14)
    model1.drop(["field2", "field3"])
    assert "field2" not in model1.as_dict()
    assert "field3" not in model1.as_dict()

    # Test dropping custom fields
    model2 = TstParamModel(field4=42, field5=3.14)
    model2.drop(["field4", "field5"])
    assert "field4" not in model2.as_dict()
    assert "field5" not in model2.as_dict()


def test_drop_nonexistent_field_ignore():
    """Test dropping nonexistent fields with ignore_missing=True."""
    model = TstParamModel()
    model.drop("nonexistent", ignore_missing=True)  # Should not raise any exception


def test_drop_nonexistent_field_error():
    """Test dropping nonexistent fields with ignore_missing=False raises KeyError."""
    model = TstParamModel()
    with pytest.raises(KeyError):
        model.drop("nonexistent", ignore_missing=False)


# Tests for field name properties
def test_field_names_property():
    """Test that defined_field_names property returns all field names."""
    # Note: defined_field_names refers to the predefined attributes of the model
    # Extra params do not count
    model = ScatterParams(s=42, param2="example")
    assert model["s"] == 42
    assert model.get("s") == 42
    assert "s" in model.defined_field_names
    assert "param2" not in model.defined_field_names


def test_defined_field_names_property():
    """Test the defined_field_names property."""
    model = TstParamModel()
    defined_fields = model.defined_field_names
    assert "field1" in defined_fields
    assert "field2" in defined_fields
    assert "field3" in defined_fields


def test_current_field_names_property():
    """Test the current_field_names property."""
    model = TstParamModel(field1="value1")
    model.extra_field = "extra_value"
