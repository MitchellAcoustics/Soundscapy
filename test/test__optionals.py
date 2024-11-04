import pytest
import os


def test_optional_dependency_groups_defined():
    """Test that OPTIONAL_DEPENDENCIES has expected structure."""
    from soundscapy._optionals import OPTIONAL_DEPENDENCIES

    assert "audio" in OPTIONAL_DEPENDENCIES

    group = OPTIONAL_DEPENDENCIES["audio"]
    assert "packages" in group
    assert "install" in group
    assert "description" in group

    assert isinstance(group["packages"], tuple)
    assert len(group["packages"]) > 0
    assert all(isinstance(pkg, str) for pkg in group["packages"])

    assert group["install"] == "soundscapy[audio]"
    assert isinstance(group["description"], str)


def test_format_import_error():
    """Test error message formatting."""
    from soundscapy._optionals import format_import_error

    msg = format_import_error("audio")
    assert "requires additional dependencies" in msg
    assert "pip install soundscapy[audio]" in msg

    with pytest.raises(KeyError):
        format_import_error("nonexistent_group")


@pytest.mark.skipif(
    os.environ.get("AUDIO_DEPS") == "1",
    reason="Test requires audio dependencies to be missing",
)
def test_require_dependencies_missing():
    """Test behavior when dependencies are missing."""
    from soundscapy._optionals import require_dependencies

    with pytest.raises(ImportError) as exc_info:
        require_dependencies("audio")

    assert "requires additional dependencies" in str(exc_info.value)
    assert "pip install soundscapy[audio]" in str(exc_info.value)


@pytest.mark.optional_deps("audio")
def test_require_dependencies_present():
    """Test behavior when dependencies are present."""
    from soundscapy._optionals import require_dependencies

    packages = require_dependencies("audio")
    assert isinstance(packages, dict)
    assert all(pkg in packages for pkg in ["mosqito", "maad", "acoustics"])
    assert all(packages[pkg] is not None for pkg in packages)


def test_audio_module_import_behavior():
    """Test top-level audio module import behavior."""
    import importlib
    import soundscapy

    # Check if audio module is in __all__
    has_audio = "audio" in soundscapy.__all__

    # Should match whether dependencies are available
    try:
        # Attempt to import dependencies
        importlib.import_module("mosqito")
        importlib.import_module("maad")
        importlib.import_module("acoustics")

        # If all imports succeed, audio module should be present
        assert has_audio is True
    except ImportError:
        # If any import fails, audio module should not be present
        assert has_audio is False


def test_invalid_group():
    """Test behavior with invalid dependency group."""
    from soundscapy._optionals import require_dependencies

    with pytest.raises(KeyError) as exc_info:
        require_dependencies("nonexistent_group")
    assert "Unknown dependency group" in str(exc_info.value)
