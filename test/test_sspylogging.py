# tests/test_logging.py
import tempfile
from pathlib import Path

from loguru import logger

from soundscapy.sspylogging import (
    disable_logging,
    enable_debug,
    get_logger,
    is_notebook,
    setup_logging,
)


def test_setup_logging_basic():
    """Test the basic logging setup."""
    setup_logging("INFO")
    assert logger.level("INFO").name == "INFO"


def test_setup_logging_with_file():
    """Test logging setup with a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        setup_logging("INFO", log_file=log_file)
        logger.info("Test message")
        # Force logger to flush any queued messages
        logger.complete()
        assert log_file.exists()
        assert "Test message" in log_file.read_text()


def test_setup_logging_with_format_levels():
    """Test different format levels for logging."""
    # Basic format
    setup_logging(format_level="basic")
    # Detailed format
    setup_logging(format_level="detailed")
    # Developer format
    setup_logging(format_level="developer")
    # Invalid format (should default to basic)
    setup_logging(format_level="invalid")


def test_enable_debug():
    """Test enabling debug logging."""
    enable_debug()
    assert logger.level("DEBUG").name == "DEBUG"


def test_disable_logging():
    """Test disabling logging."""
    import io

    # Create a test output buffer
    test_output = io.StringIO()

    # Set up logging with our test output
    logger.remove()
    logger.enable("soundscapy")
    logger.add(test_output, level="CRITICAL")

    # Try with logging enabled
    logger.critical("This should be logged")
    logger.complete()
    assert "This should be logged" in test_output.getvalue()

    # Now disable logging
    disable_logging()

    # Clear the output buffer
    test_output.seek(0)
    test_output.truncate(0)

    # Try to log at CRITICAL level again
    logger.critical("This should NOT be logged")
    logger.complete()

    # Verify nothing was logged after disabling
    assert test_output.getvalue() == ""

    # Reset for other tests
    setup_logging("INFO")


def test_get_logger():
    """Test getting the logger instance."""
    assert get_logger() is logger


def test_is_notebook():
    """Test detection of Jupyter notebook environment."""
    assert not is_notebook()  # When running pytest
