# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Configure pytest for soundscapy testing."""

import importlib.util
import os

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
# xdoctest's `module:` requirement can't see namespace packages such as rpy2,
# so doctests that need R gate on this flag instead.
_has_rpy2 = importlib.util.find_spec("rpy2") is not None
os.environ["SOUNDSCAPY_HAS_RPY2"] = "1" if _has_rpy2 else "0"
if not _has_rpy2:
    collect_ignore_glob += [
        "src/soundscapy/spi/*",
        "src/soundscapy/satp/*",
        "src/soundscapy/r_wrapper/*",
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
