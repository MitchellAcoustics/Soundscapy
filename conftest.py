"""Configure pytest for soundscapy testing."""

import pytest
import os
from loguru import logger
from _pytest.logging import LogCaptureFixture
from soundscapy._optionals import require_dependencies

# Cache the dependency check result
_has_audio = None


def _check_audio_deps():
    """Check for audio dependencies, caching the result."""
    global _has_audio
    if _has_audio is None:
        try:
            required = require_dependencies("audio")
            logger.debug(f"Audio dependencies found: {list(required.keys())}")
            _has_audio = True
        except ImportError as e:
            logger.debug(f"Missing audio dependencies: {e}")
            _has_audio = False
        logger.debug(f"Setting AUDIO_DEPS={_has_audio}")
    return _has_audio


def pytest_ignore_collect(collection_path):
    """Control test collection for optional dependency modules.

    Parameters
    ----------
    collection_path : Path
        Path to the file being considered for collection

    Returns
    -------
    bool
        True if the file should be ignored, False otherwise
    """
    path_str = str(collection_path)
    # Check if path is in the audio module
    if "audio/" in path_str:
        # Skip collection if audio dependencies are missing
        should_ignore = not _check_audio_deps()
        logger.debug(f"Collection check for {path_str}: ignore={should_ignore}")
        return should_ignore

    return None


def pytest_configure(config):
    """Register markers and configure test environment."""
    # Register only necessary markers
    config.addinivalue_line(
        "markers", "optional_deps(group): mark tests requiring optional dependencies"
    )

    # Set environment variable for xdoctest
    os.environ["AUDIO_DEPS"] = "1" if _check_audio_deps() else "0"

    # Configure xdoctest namespace
    config.option.xdoctest_namespace = """
    import os
    from soundscapy._optionals import require_dependencies
    """


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
        try:
            require_dependencies(group)
        except ImportError:
            pytest.skip(f"Missing optional dependencies for {group}")
