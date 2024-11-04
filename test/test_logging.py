# tests/test_logging.py
from loguru import logger
from soundscapy.logging import setup_logging, is_notebook


def test_setup_logging_basic():
    setup_logging("INFO")
    assert logger.level("INFO").name == "INFO"


def test_is_notebook():
    assert not is_notebook()  # When running pytest
