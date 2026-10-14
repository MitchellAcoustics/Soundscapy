# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Tests for the moved-name import shims built by ``_deprecated.moved_to``."""

import importlib

import pytest

_C = "soundscapy.constants"

# (old module, name, new module)
MOVED = [
    ("soundscapy", "PAQ_IDS", _C),
    ("soundscapy", "PAQ_LABELS", _C),
    ("soundscapy.surveys", "LANGUAGE_ANGLES", _C),
    ("soundscapy.surveys", "PAQ_IDS", _C),
    ("soundscapy.surveys", "PAQ_LABELS", _C),
    ("soundscapy.surveys.survey_utils", "PAQ", _C),
    ("soundscapy.surveys.survey_utils", "PAQ_IDS", _C),
    ("soundscapy.surveys.survey_utils", "PAQ_LABELS", _C),
    ("soundscapy.surveys.survey_utils", "LikertScale", _C),
    ("soundscapy.surveys.survey_utils", "LIKERT_SCALES", _C),
    ("soundscapy.surveys.survey_utils", "EQUAL_ANGLES", _C),
    ("soundscapy.surveys.survey_utils", "LANGUAGE_ANGLES", _C),
    ("soundscapy", "ConfigManager", "soundscapy.audio"),
    ("soundscapy", "add_results", "soundscapy.audio"),
    ("soundscapy", "parallel_process", "soundscapy.audio"),
    ("soundscapy", "prep_multiindex_df", "soundscapy.audio"),
    ("soundscapy", "process_all_metrics", "soundscapy.audio"),
    ("soundscapy", "CircE", "soundscapy.satp"),
    ("soundscapy", "CircEResults", "soundscapy.satp"),
    ("soundscapy", "CircModelE", "soundscapy.satp"),
    ("soundscapy", "normalize_polar_angles", "soundscapy.satp"),
    ("soundscapy", "CentredParams", "soundscapy.spi"),
    ("soundscapy", "DirectParams", "soundscapy.spi"),
    ("soundscapy", "MultiSkewNorm", "soundscapy.spi"),
    ("soundscapy", "cp2dp", "soundscapy.spi"),
    ("soundscapy", "dp2cp", "soundscapy.spi"),
    ("soundscapy", "msn", "soundscapy.spi"),
    ("soundscapy", "spi_score", "soundscapy.spi"),
]


@pytest.mark.parametrize(("module", "name", "new_module"), MOVED)
def test_old_path_warns_and_forwards(module, name, new_module):
    target = pytest.importorskip(new_module)
    mod = importlib.import_module(module)
    match = rf"has moved to `{new_module}\.{name}`"
    with pytest.warns(DeprecationWarning, match=match):
        value = getattr(mod, name)
    assert value is getattr(target, name)


@pytest.mark.parametrize(("module", "name"), [(m, n) for m, n, _ in MOVED])
def test_old_path_is_not_public(module, name):
    assert name not in getattr(importlib.import_module(module), "__all__", ())


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
