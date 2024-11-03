import pytest
from loguru import logger
from _pytest.logging import LogCaptureFixture
from soundscapy.logging import setup_logging

logger.enable("soundscapy")  # TODO: Need to check if this is best practice for testing.
setup_logging("DEBUG")


# See: https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library
@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Override pytest's caplog to work with loguru"""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)
