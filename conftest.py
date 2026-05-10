"""Configure pytest for soundscapy testing."""

import importlib.util

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

# Skip xdoctest collection inside optional source subpackages when their
# extras aren't installed. Test directories under test/ are gated by their
# own conftest.py + pytest.importorskip.
collect_ignore_glob: list[str] = []
_audio_deps = ("acoustic_toolbox", "maad", "mosqito", "tqdm")
if any(importlib.util.find_spec(dep) is None for dep in _audio_deps):
    collect_ignore_glob.append("src/soundscapy/audio/*")
if importlib.util.find_spec("rpy2") is None:
    collect_ignore_glob += [
        "src/soundscapy/spi/*",
        "src/soundscapy/satp/*",
        "src/soundscapy/r_wrapper/*",
        # iso_plot.py contains xdoctests that import from soundscapy.spi
        "src/soundscapy/plotting/iso_plot.py",
    ]


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
