"""Configure pytest for soundscapy testing."""

import os

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

# Cache the dependency check results
_dependency_cache = {}


def _check_dependencies(group: str) -> bool:
    """Check for dependencies of a group, caching the result."""
    if group not in _dependency_cache:
        try:
            if group == "audio":
                # Try importing audio-related modules using importlib.util for availability check
                import importlib.util

                deps = ["mosqito", "maad", "tqdm", "acoustic_toolbox"]
                all_available = all(
                    importlib.util.find_spec(dep) is not None for dep in deps
                )
                _dependency_cache[group] = all_available
                if all_available:
                    logger.debug(f"{group} dependencies found")
                else:
                    logger.debug(f"{group} dependencies missing")
            elif group == "spi":
                # Check SPI dependencies
                import importlib.util

                spi_available = importlib.util.find_spec("rpy2") is not None
                _dependency_cache[group] = spi_available
                if spi_available:
                    logger.debug(f"{group} dependencies found")
                else:
                    logger.debug(f"{group} dependencies missing")
            else:
                logger.debug(f"Unknown dependency group: {group}")
                _dependency_cache[group] = False
        except ImportError as e:
            logger.debug(f"Missing {group} dependencies: {e}")
            _dependency_cache[group] = False
    return _dependency_cache[group]


def pytest_ignore_collect(collection_path):
    """Control test collection for optional dependency modules."""
    path_str = str(collection_path)

    # Map module paths to their dependency groups
    module_deps = {
        "audio/": "audio",
        "spi/": "spi",
        # Add new optional module paths here
    }

    for module_path, dep_group in module_deps.items():
        if module_path in path_str:
            should_ignore = not _check_dependencies(dep_group)
            logger.debug(f"Collection check for {path_str}: ignore={should_ignore}")
            return should_ignore

    return None


def pytest_configure(config):
    """Register markers and configure test environment."""
    config.addinivalue_line(
        "markers", "optional_deps(group): mark tests requiring optional dependencies"
    )
    config.addinivalue_line(
        "markers",
        "skip_if_deps(group): mark tests to skip if dependencies are present",
    )

    # Define known dependency groups
    dependency_groups = ["audio", "spi"]
    for group in dependency_groups:
        env_var = f"{group.upper()}_DEPS"
        os.environ[env_var] = "1" if _check_dependencies(group) else "0"
        logger.debug(f"Set {env_var}={os.environ[env_var]}")

    # Configure xdoctest namespace
    namespace_setup = """
    import os
    import importlib
    """
    config.option.xdoctest_namespace = namespace_setup


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Override pytest's caplog to work with loguru."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


def pytest_runtest_setup(item):
    """Mark tests requiring optional dependencies as xfail if deps are missing."""
    for marker in item.iter_markers(name="optional_deps"):
        group = marker.args[0] if marker.args else marker.kwargs.get("group")
        if not group:
            pytest.fail("No dependency group specified for optional_deps marker")
        if not _check_dependencies(group):
            pytest.xfail(f"Missing optional dependencies for {group}")
