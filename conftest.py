# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Configure pytest for soundscapy testing."""

import importlib.util
from fnmatch import fnmatch

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
_has_rpy2 = importlib.util.find_spec("rpy2") is not None
if not _has_rpy2:
    collect_ignore_glob += [
        "src/soundscapy/spi/*",
        "src/soundscapy/satp/*",
        "src/soundscapy/r_wrapper/*",
    ]

# Doctests outside the R subpackages that still need R, matched on xdoctest
# item names. Gated here rather than with `# xdoctest: +REQUIRES(module:rpy2)`
# because xdoctest can't resolve namespace packages such as rpy2, and so the
# rendered docstrings stay free of test-harness directives.
_R_DOCTESTS = ("ISOPlot.add_spi:*",)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip doctests that need R when rpy2 isn't installed."""
    if _has_rpy2:
        return
    skip_r = pytest.mark.skip(reason="requires the soundscapy[r] extra")
    for item in items:
        if any(fnmatch(item.name, pattern) for pattern in _R_DOCTESTS):
            item.add_marker(skip_r)


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
