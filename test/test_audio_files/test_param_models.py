import pytest

from src.soundscapy.plotting.param_models import ParamModel


class TestParamModel(ParamModel):
    """A minimal subclass of ParamModel for testing with defined fields."""

    field1: str | None = None
    field2: int | None = 10  # Default value
    field3: float | None = None


def test_update_valid_fields():
    """Test updating valid fields in the model."""
    model = TestParamModel()
    model.update(field1="value1", field2=123, extra="allow")
    assert model.get("field1") == "value1"
    assert model.get("field2") == 123


def test_update_ignore_null_functionality():
    """Test that None values are removed when ignore_null is True."""
    model = TestParamModel()
    model.update(field3=456, field2=None, ignore_null=True)
    assert model.get("field3") == 456
    assert model.get("field2") == 10

    model.update(field2=None, ignore_null=False)
    assert model.get("field2") is None


def test_update_overwrites_default():
    """Test that a default value is overwritten during update."""
    model = TestParamModel()
    assert model.get("field2") == 10  # Default value
    model.update(field2=99)
    assert model.get("field2") == 99  # Updated value


def test_get_with_default():
    """Test retrieving a field with a default fallback value."""
    model = TestParamModel()
    assert model.get("missing_field", default="default_value") == "default_value"


def test_getitem_existing_field():
    """Test dictionary-style access for existing fields."""
    model = TestParamModel(field1="value1")
    assert model["field1"] == "value1"


def test_getitem_nonexistent_field():
    """Test dictionary-style access raises KeyError for nonexistent fields."""
    model = TestParamModel()
    with pytest.raises(KeyError):
        _ = model["nonexistent"]


def test_as_dict_method():
    """Test that as_dict outputs the expected dictionary."""
    model = TestParamModel(field2=42, field3=3.14)
    result = model.as_dict()
    assert result["field2"] == 42
    assert result["field3"] == 3.14


def test_get_changed_params():
    """Test that changed parameters are returned correctly."""
    model = TestParamModel(field2=42)
    model.update(field2=99)
    changed_params = model.get_changed_params()
    assert changed_params["field2"] == 99


def test_get_multiple_fields():
    """Test retrieval of multiple fields as a dictionary."""
    model = TestParamModel(field1="value1", field2=99)
    multiple_fields = model.get_multiple(["field1", "field2", "nonexistent"])
    assert multiple_fields["field1"] == "value1"
    assert multiple_fields["field2"] == 99
    assert "nonexistent" not in multiple_fields


def test_pop_existing_field():
    """Test popping an existing field."""
    model = TestParamModel(field3=3.14)
    value = model.pop("field3")
    assert value == 3.14
    with pytest.raises(KeyError):
        _ = model["field3"]


def test_pop_nonexistent_field():
    """Test popping a nonexistent field raises KeyError."""
    model = TestParamModel()
    with pytest.raises(KeyError):
        model.pop("nonexistent")


def test_drop_single_field():
    """Test dropping a single field."""
    model = TestParamModel(field1="value1")
    model.drop("field1")
    assert "field1" not in model.as_dict()


def test_drop_multiple_fields():
    """Test dropping multiple fields."""
    model = TestParamModel(field2=42, field3=3.14)
    model.drop(["field2", "field3"])
    assert "field2" not in model.as_dict()
    assert "field3" not in model.as_dict()


def test_drop_nonexistent_field_ignore():
    """Test dropping nonexistent fields with ignore_missing=True."""
    model = TestParamModel()
    model.drop("nonexistent", ignore_missing=True)  # Should not raise any exception


def test_drop_nonexistent_field_error():
    """Test dropping nonexistent fields with ignore_missing=False raises KeyError."""
    model = TestParamModel()
    with pytest.raises(KeyError):
        model.drop("nonexistent", ignore_missing=False)


def test_defined_field_names_property():
    """Test the defined_field_names property."""
    model = TestParamModel()
    defined_fields = model.defined_field_names
    assert "field1" in defined_fields
    assert "field2" in defined_fields
    assert "field3" in defined_fields


def test_current_field_names_property():
    """Test the current_field_names property."""
    model = TestParamModel(field1="value1")
    current_fields = model.current_field_names
    assert "field1" in current_fields
