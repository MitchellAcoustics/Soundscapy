# tests/test_logging.py
from loguru import logger
from soundscapy.logging import setup_logging, is_notebook, LogFormatter

def test_setup_logging_basic():
    setup_logging("INFO")
    assert logger.level("INFO").name == "INFO"

def test_setup_logging_with_file(tmp_path):
    log_file = tmp_path / "test.log"
    setup_logging("DEBUG", str(log_file))
    assert log_file.exists()
    assert "Logging configured" in log_file.read_text()


def test_is_notebook():
    assert not is_notebook()  # When running pytest