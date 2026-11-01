"""Tests for the soundscapy logger."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from soundscapy._log import _defer_to_application, logger, set_log_level

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def restore_level() -> Iterator[None]:
    old = logger.level
    yield
    logger.setLevel(old)
    logger.addFilter(_defer_to_application)  # set_log_level removes it; adding is idempotent


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
    # Printed once, by the application's handler, not a second time by soundscapy's.
    result = _run(
        "import logging; logging.basicConfig(level=logging.INFO)\n"
        "from soundscapy._log import logger; logger.info('once')"
    )
    assert result.stderr.count("once") == 1


# The application's logging configuration decides, as NFR-13 requires ("silence or redirect with
# one call or with standard logging configuration"): records reach a root handler only at or
# above the root level, unless soundscapy's level was set deliberately.
_EMIT = (
    "from soundscapy._log import logger; logger.info('info-line'); logger.warning('warn-line')\n"
)


def test_application_root_level_warning_silences_info() -> None:
    result = _run("import logging; logging.basicConfig(level=logging.WARNING)\n" + _EMIT)
    assert "info-line" not in result.stderr
    assert result.stderr.count("warn-line") == 1


def test_application_root_level_info_shows_info_once() -> None:
    result = _run("import logging; logging.basicConfig(level=logging.INFO)\n" + _EMIT)
    assert result.stderr.count("info-line") == 1


def test_set_log_level_overrides_the_application_root_level() -> None:
    result = _run(
        "import logging; logging.basicConfig(level=logging.WARNING)\n"
        "from soundscapy._log import set_log_level; set_log_level('INFO')\n" + _EMIT
    )
    assert result.stderr.count("info-line") == 1


def test_application_level_on_the_soundscapy_logger_is_honoured() -> None:
    result = _run(
        "import logging; logging.basicConfig(level=logging.WARNING)\n"
        "logging.getLogger('soundscapy').setLevel(logging.DEBUG)\n"
        "from soundscapy._log import logger; logger.debug('debug-line')\n"
    )
    assert result.stderr.count("debug-line") == 1
