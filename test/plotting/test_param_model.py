"""
Tests for the ParamModel class and its subclasses.
"""

from soundscapy.plotting.plotting_types import ParamModel


def test_param_model_creation():
    """Test creating parameter models using the factory method."""
    # Test creating various parameter types
    scatter_params = ParamModel.create("scatter", x="test_x", y="test_y")
    density_params = ParamModel.create("density", alpha=0.7)
    style_params = ParamModel.create("style", linewidth=2.0)
    
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
    params = ParamModel.create("scatter")
    
    # Initial defaults
    assert params.x == "ISOPleasant"
    
    # Update and test fluent interface
    updated = params.update(x="new_x", s=30)
    
    # Check updates applied
    assert updated is params  # Should return self
    assert params.x == "new_x"
    assert params.s == 30
    
    # None values should not override existing values
    params.update(x=None)
    assert params.x == "new_x"  # Still has the updated value


def test_param_model_as_dict():
    """Test converting parameter models to dictionaries."""
    # Create a parameter model with some custom values
    params = ParamModel.create("scatter", x="custom_x", color="red")
    
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