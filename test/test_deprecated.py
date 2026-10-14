# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Tests for the moved-constant import shims built by ``_deprecated.moved_to``."""

import importlib

import pytest

from soundscapy import constants

MOVED = [
    ("soundscapy", "PAQ_IDS"),
    ("soundscapy", "PAQ_LABELS"),
    ("soundscapy.surveys", "LANGUAGE_ANGLES"),
    ("soundscapy.surveys", "PAQ_IDS"),
    ("soundscapy.surveys", "PAQ_LABELS"),
    ("soundscapy.surveys.survey_utils", "PAQ"),
    ("soundscapy.surveys.survey_utils", "PAQ_IDS"),
    ("soundscapy.surveys.survey_utils", "PAQ_LABELS"),
    ("soundscapy.surveys.survey_utils", "LikertScale"),
    ("soundscapy.surveys.survey_utils", "LIKERT_SCALES"),
    ("soundscapy.surveys.survey_utils", "EQUAL_ANGLES"),
    ("soundscapy.surveys.survey_utils", "LANGUAGE_ANGLES"),
]


@pytest.mark.parametrize(("module", "name"), MOVED)
def test_old_path_warns_and_forwards(module, name):
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=r"has moved to `soundscapy\.constants"):
        value = getattr(mod, name)
    assert value is getattr(constants, name)


def test_warning_points_at_caller():
    import soundscapy

    with pytest.warns(DeprecationWarning, match="has moved to") as record:
        # pyrefly: ignore [missing-attribute]
        _ = soundscapy.PAQ_IDS
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "module", ["soundscapy.surveys", "soundscapy.surveys.survey_utils"]
)
def test_unknown_name_raises(module):
    mod = importlib.import_module(module)
    with pytest.raises(AttributeError):
        _ = mod.not_a_real_name


def test_top_level_falls_back_to_lazy_loader():
    import soundscapy

    # Unknown names fall through to the lazy loader and raise.
    with pytest.raises(AttributeError):
        # pyrefly: ignore [missing-attribute]
        _ = soundscapy.not_a_real_name
