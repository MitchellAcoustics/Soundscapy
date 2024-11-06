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


@pytest.mark.skipif(
    os.environ.get("AUDIO_DEPS") == "1",
    reason="Test requires audio dependencies to be missing",
)
def test_require_dependencies_missing():
    """Test behavior when dependencies are missing."""
    from soundscapy._optionals import require_dependencies

    with pytest.raises(ImportError) as exc_info:
        require_dependencies("audio")

    assert "Install with:" in str(exc_info.value)
    assert "pip install soundscapy[audio]" in str(exc_info.value)


@pytest.mark.optional_deps("audio")
def test_require_dependencies_present():
    """Test behavior when dependencies are present."""
    from soundscapy._optionals import require_dependencies

    packages = require_dependencies("audio")
    assert isinstance(packages, dict)
    assert all(pkg in packages for pkg in ["mosqito", "maad", "acoustics"])
    assert all(packages[pkg] is not None for pkg in packages)


def test_optional_import_behavior():
    """Test top-level optional component import behavior."""
    import importlib
    import soundscapy
    from soundscapy._optionals import OPTIONAL_IMPORTS

    # Test that optional components are listed in __all__
    for name in OPTIONAL_IMPORTS:
        assert name in soundscapy.__all__

    # Test import behavior
    def has_audio_deps():
        """Check if audio dependencies are available."""
        try:
            importlib.import_module("mosqito")
            importlib.import_module("maad")
            importlib.import_module("acoustics")
            return True
        except ImportError:
            return False

    # Try importing Binaural as an example component
    if has_audio_deps():
        # Should succeed when dependencies are available
        from soundscapy import Binaural

        assert hasattr(Binaural, "__module__")
    else:
        # Should raise helpful ImportError when dependencies missing
        import pytest

        with pytest.raises(ImportError) as exc_info:
            from soundscapy import Binaural
        assert "audio analysis functionality" in str(exc_info.value)
        assert "pip install soundscapy[audio]" in str(exc_info.value)


def test_invalid_group():
    """Test behavior with invalid dependency group."""
    from soundscapy._optionals import require_dependencies

    with pytest.raises(KeyError) as exc_info:
        require_dependencies("nonexistent_group")
    assert "Unknown dependency group" in str(exc_info.value)
