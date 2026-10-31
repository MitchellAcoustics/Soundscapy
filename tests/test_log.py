"""Tests for the soundscapy logger."""

import logging
import subprocess
import sys
from collections.abc import Iterator

import pytest

from soundscapy._log import logger, set_log_level


@pytest.fixture
def restore_level() -> Iterator[None]:
    old = logger.level
    yield
    logger.setLevel(old)


def test_records_propagate_to_root(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        logger.info("hello")
    assert [r.getMessage() for r in caplog.records] == ["hello"]


@pytest.mark.usefixtures("restore_level")
def test_set_log_level_drops_info(caplog: pytest.LogCaptureFixture) -> None:
    set_log_level("WARNING")
    logger.info("quiet")
    logger.warning("loud")
    assert [r.getMessage() for r in caplog.records] == ["loud"]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # fixed interpreter, literal code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )


def test_default_handler_writes_to_stderr_without_root_handlers() -> None:
    result = _run("from soundscapy._log import logger; logger.info('once')")
    assert result.stderr.count("once") == 1
    assert result.stdout == ""


def test_default_handler_defers_to_root_handlers() -> None:
    result = _run(
        "import logging; logging.basicConfig()\n"
        "from soundscapy._log import logger; logger.info('once')"
    )
    assert result.stderr.count("once") == 1
