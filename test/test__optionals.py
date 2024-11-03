import pytest
from soundscapy._optionals import OptionalDependencyManager, MODULE_GROUPS


@pytest.fixture
def manager():
    """Provide clean OptionalDependencyManager instance"""
    return OptionalDependencyManager().get_instance()


def test_module_exists_success(manager):
    # Assuming 'sys' is always available
    module = manager.module_exists("sys")
    assert module is not None


def test_module_exists_failure_ignore(manager):
    module = manager.module_exists("non_existent_module", error="ignore")
    assert module is None


def test_module_exists_failure_warn(manager, caplog):
    """Test that a warning is logged when a non-existent module is checked with error='warn'."""
    with caplog.at_level("WARNING"):
        module = manager.module_exists("non_existent_module", error="warn")
        assert module is None
        assert "Missing optional dependency: non_existent_module" in caplog.text


def test_module_exists_failure_raise(manager):
    with pytest.raises(ImportError):
        manager.module_exists("non_existent_module", error="raise")


def test_check_module_group_success(manager):
    # Assuming 'sys' is always available and adding it to a test group
    MODULE_GROUPS["test_group"] = {
        "modules": ("sys",),
        "install": "soundscapy[test_group]",
        "description": "test group",
    }
    assert manager.check_module_group("test_group")


def test_check_module_group_failure_ignore(manager):
    MODULE_GROUPS["test_group"] = {
        "modules": ("non_existent_module",),
        "install": "soundscapy[test_group]",
        "description": "test group",
    }
    assert not manager.check_module_group("test_group", error="ignore")


def test_check_module_group_failure_warn(manager, caplog):
    """Test warning is logged for missing group dependencies"""
    MODULE_GROUPS["test_group"] = {
        "modules": ("non_existent_module",),
        "install": "soundscapy[test_group]",
        "description": "test group",
    }
    with caplog.at_level("WARNING"):
        assert not manager.check_module_group("test_group", error="warn")
        assert "Missing optional dependencies for test group" in caplog.text


def test_module_exists_caching(manager):
    """Test that module_exists caches its results."""
    # First call should actually import
    first_result = manager.module_exists("sys")
    # Second call should return cached result
    second_result = manager.module_exists("sys")
    assert first_result is second_result
    assert "sys" in manager._checked_modules


def test_module_exists_failed_import_caching(manager):
    """Test that failed imports are properly cached."""
    # First call
    first_result = manager.module_exists("non_existent_module")
    # Second call should use cached result
    second_result = manager.module_exists("non_existent_module")
    assert first_result is second_result
    assert first_result is None
    assert "non_existent_module" in manager._checked_modules


def test_module_exists_cached_failure_raise(manager):
    """Test raise behavior with cached failed import."""
    # Cache the failed import first
    manager.module_exists("non_existent_module", error="ignore")
    # Should raise on subsequent call with raise
    with pytest.raises(ImportError) as exc_info:
        manager.module_exists("non_existent_module", error="raise")
    assert "Required dependency non_existent_module not found" in str(exc_info.value)


def test_module_exists_cached_failure_warn(manager, caplog):
    """Test warning behavior with cached failed import."""
    # Cache the failed import first
    manager.module_exists("non_existent_module", error="ignore")
    # Should warn on subsequent call
    with caplog.at_level("WARNING"):
        result = manager.module_exists("non_existent_module", error="warn")
        assert result is None
        assert "Missing optional dependency: non_existent_module" in caplog.text


def test_module_exists_invalid_error_mode(manager):
    """Test that invalid error modes raise AssertionError."""
    with pytest.raises(AssertionError):
        manager.module_exists("sys", error="invalid_mode")
