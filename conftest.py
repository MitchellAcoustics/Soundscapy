"""Configure pytest for soundscapy testing."""

import pytest
import os
from loguru import logger
from _pytest.logging import LogCaptureFixture

# Cache the dependency check results
_dependency_cache = {}


def _check_dependencies(group: str) -> bool:
    """Check for dependencies of a group, caching the result."""
    if group not in _dependency_cache:
        try:
            from soundscapy._optionals import require_dependencies

            required = require_dependencies(group)
            logger.debug(f"{group} dependencies found: {list(required.keys())}")
            _dependency_cache[group] = True
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

    # Set environment variables for each dependency group
    from soundscapy._optionals import OPTIONAL_DEPENDENCIES

    for group in OPTIONAL_DEPENDENCIES:
        env_var = f"{group.upper()}_DEPS"
        os.environ[env_var] = "1" if _check_dependencies(group) else "0"
        logger.debug(f"Set {env_var}={os.environ[env_var]}")

    # Configure xdoctest namespace with all dependency groups
    namespace_setup = """
    import os
    from soundscapy._optionals import require_dependencies
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
    """Skip tests marked as requiring optional dependencies."""
    for marker in item.iter_markers(name="optional_deps"):
        group = marker.args[0] if marker.args else marker.kwargs.get("group")
        if not group:
            pytest.fail("No dependency group specified for optional_deps marker")
        if not _check_dependencies(group):
            pytest.skip(f"Missing optional dependencies for {group}")
