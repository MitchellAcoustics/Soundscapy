"""Profiles and settings files on the public entry points: carry-over, the default run, reruns."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from soundscapy import _data
from soundscapy._model import COL_INCLUDED
from soundscapy.instrument.angles import AngleSelection, AngleSetRef
from soundscapy.projection.project import ProjectionSettings, project
from soundscapy.survey.read import ImportSettings, read_survey
from tests._helpers import flag_mask, medium_warning, projection_settings, write_text

FIXTURES = Path(__file__).parent / "projection" / "fixtures"
HAND_CSV = FIXTURES / "hand_values.csv"
MAPPING = FIXTURES / "mapping.toml"

SAL = {"id": "sal-2026", "version": "1", "hash": _data.pin("profiles/sal-2026/1.toml")}
ISO_2019 = {
    "id": "iso-12913-3-2019",
    "version": "1",
    "hash": _data.pin("profiles/iso-12913-3-2019/1.toml"),
}
SATP = AngleSetRef(name="satp", version="1.5")


# --- The default run ------------------------------------------------------------------------------


def test_default_run_is_sal_2026_with_satp(isd_survey):
    with medium_warning() as record:
        coords = project(isd_survey)
    assert len(record) == 1
    assert coords.provenance["profile"] == SAL
    assert coords.provenance["overrides"] == {}
    assert "profile_inherited" not in coords.provenance
    assert coords.origin[0].angle_sets == ("satp@1.5",)
    assert coords.settings == ProjectionSettings(angles=AngleSelection(default=SATP))
    assert int(flag_mask(coords, "medium_confidence_language").sum()) == 69
    assert int(coords.labels[COL_INCLUDED].sum()) == 1536


def test_equal_keyword_is_warning_free_and_recorded_as_an_override(isd_survey):
    coords = project(isd_survey, angles="equal")  # filterwarnings = error: none fired
    assert coords.provenance["profile"] == SAL
    assert coords.provenance["overrides"] == {"projection.angles.default": "equal"}
    assert coords.origin[0].angle_sets == ("equal",)


def test_read_survey_records_the_default_profile(hand_survey):
    assert hand_survey.provenance["profile"] == SAL
    assert hand_survey.provenance["overrides"] == {}
    assert "profile_inherited" not in hand_survey.provenance


def test_read_survey_records_import_overrides():
    survey = read_survey(HAND_CSV, MAPPING, on_scale_violation="exclude")
    assert survey.provenance["overrides"] == {"import.on_scale_violation": "exclude"}


# --- Carry-over (spec 6.3.4) ----------------------------------------------------------------------


def test_profile_named_on_read_survey_carries_into_project():
    survey = read_survey(HAND_CSV, MAPPING, profile="iso-12913-3-2019")
    assert survey.provenance["profile"] == ISO_2019
    assert survey.provenance["overrides"] == {}
    coords = project(survey)
    assert coords.provenance["profile"] == ISO_2019
    assert "profile_inherited" not in coords.provenance
    assert coords.provenance["overrides"] == {}
    assert projection_settings(coords).formulation == "iso2019"
    assert coords.origin[0].angle_sets == ("equal",)


def test_explicit_profile_on_project_records_the_inherited_one():
    survey = read_survey(HAND_CSV, MAPPING, profile="iso-12913-3-2019")
    coords = project(survey, profile="sal-2026")
    assert coords.provenance["profile"] == SAL
    assert coords.provenance["profile_inherited"] == ISO_2019
    assert coords.provenance["overrides"] == {}
    assert projection_settings(coords).formulation == "canonical"
    assert coords.origin[0].angle_sets == ("satp@1.5",)


def test_a_survey_without_a_recorded_profile_falls_through_to_sal_2026(hand_survey):
    provenance = {
        key: value
        for key, value in hand_survey.provenance.items()
        if key not in ("profile", "overrides")
    }
    survey = dataclasses.replace(hand_survey, provenance=provenance)
    coords = project(survey)
    assert coords.provenance["profile"] == SAL
    assert "profile_inherited" not in coords.provenance
    assert coords.origin[0].angle_sets == ("satp@1.5",)


# --- Settings files -------------------------------------------------------------------------------


def test_settings_file_names_the_profile_and_applies_to_the_call_it_is_passed_to(tmp_path):
    path = write_text(
        tmp_path,
        'extends = "iso-12913-3-2019"\n'
        "[import]\n"
        'on_scale_violation = "exclude"\n'
        "[projection.angles]\n"
        'on_uncovered = "exclude"\n',
    )
    survey = read_survey(HAND_CSV, MAPPING, settings=path)
    assert survey.provenance["profile"] == ISO_2019
    assert survey.settings == ImportSettings(on_scale_violation="exclude")
    assert survey.provenance["overrides"] == {"import.on_scale_violation": "exclude"}

    with_file = project(survey, settings=path)
    assert with_file.provenance["profile"] == ISO_2019
    assert projection_settings(with_file).angles.on_uncovered == "exclude"
    assert with_file.provenance["overrides"] == {"projection.angles.on_uncovered": "exclude"}

    # The file given to read_survey does not reach project; only the profile carries over.
    without = project(survey)
    assert without.provenance["profile"] == ISO_2019
    assert projection_settings(without).angles.on_uncovered == "error"
    assert without.provenance["overrides"] == {}


def test_profile_keyword_wins_over_the_files_extends(tmp_path):
    path = write_text(
        tmp_path, 'extends = "iso-12913-3-2019"\n[projection.angles]\non_uncovered = "exclude"\n'
    )
    survey = read_survey(HAND_CSV, MAPPING)
    coords = project(survey, settings=path, profile="sal-2026")
    assert coords.provenance["profile"] == SAL
    assert projection_settings(coords).angles.on_uncovered == "exclude"
    assert coords.origin[0].angle_sets == ("satp@1.5",)


# --- Re-running with the recorded settings --------------------------------------------------------


def test_rerun_with_recorded_settings_reproduces_the_numbers(hand_survey):
    first = project(hand_survey, formulation="aletta2024", angles="equal")
    again = project(hand_survey, settings=projection_settings(first))
    np.testing.assert_array_equal(again.xy, first.xy)  # exact: the same inputs and settings
    assert again.origin[0].settings_hash == first.origin[0].settings_hash
    assert again.settings == first.settings
    assert again.provenance["overrides"] == first.provenance["overrides"]


def test_rerun_under_an_explicit_profile_reproduces_the_numbers():
    survey = read_survey(HAND_CSV, MAPPING, profile="iso-12913-3-2019")
    first = project(survey, profile="sal-2026")
    again = project(survey, settings=projection_settings(first), profile="sal-2026")
    np.testing.assert_array_equal(again.xy, first.xy)  # exact: the same inputs and settings
    assert again.origin[0].settings_hash == first.origin[0].settings_hash
