"""project with named angle sets: per-language resolution, overrides, warnings, provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soundscapy._data import pin
from soundscapy._model import COL_ANGLE_SET, COL_FLAGS, COL_INCLUDED, COL_LANGUAGE, COL_REASONS
from soundscapy.distribution.coordinates import n_valid
from soundscapy.errors import (
    DataReferenceError,
    EqualAnglesFallbackWarning,
    LowConfidenceLanguageError,
    MappingError,
    SettingsError,
    SoundscapyWarning,
    UncoveredLanguageError,
)
from soundscapy.instrument import AngleSet, AngleSetRef, angle_set, read_angle_set
from soundscapy.instrument.angles import AngleSelection
from soundscapy.projection.formulas import project_scores
from soundscapy.projection.project import ProjectionSettings, project
from soundscapy.provenance import hash_json
from soundscapy.survey.mapping import SurveyMapping
from soundscapy.survey.read import read_survey
from tests._helpers import SYNTHETIC_TIERS, medium_warning, raw_mapping, write_csv

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC = read_angle_set(SYNTHETIC_TIERS / "1.toml")
LABEL = SYNTHETIC.label
SATP = angle_set("satp", "1.5")
NOMINAL = np.arange(0.0, 360.0, 45.0)
# Row order of survey.csv: two qaa, one qab, three qac, two qad, one qae, one excluded blank.
LANGS = ["qaa", "qaa", "qab", "qac", "qac", "qac", "qad", "qad", "qae", ""]
QAA = [0, 1]
QAB = [2]
QAC = [3, 4, 5]
QAD = [6, 7]
QAE = [8]
BLANK = 9

# Every scenario passes the synthetic set, which never ships. qad and qae both follow
# on_uncovered, so a scenario about qac has to get past them first; once qac is dealt with
# too, only the Medium row warns.


@pytest.fixture(scope="module")
def synthetic_survey():
    return read_survey(SYNTHETIC_TIERS / "survey.csv", SYNTHETIC_TIERS / "mapping.toml")


@pytest.fixture(scope="module")
def hand_survey():
    return read_survey(FIXTURES / "hand_values.csv", FIXTURES / "mapping.toml")


def _settings(coords) -> ProjectionSettings:
    assert isinstance(coords.settings, ProjectionSettings)
    return coords.settings


def _data_refs(coords) -> list:
    refs = coords.provenance["data_refs"]
    assert isinstance(refs, list)
    return refs


def _has_flag(coords, flag: str) -> np.ndarray:
    return np.array([flag in cell.split(";") for cell in coords.labels[COL_FLAGS].tolist()])


def _user_set(**changes) -> AngleSet:
    data = {**SYNTHETIC.model_dump(mode="json", by_alias=True), **changes}
    return AngleSet.model_validate(data)


# --- the three on_uncovered branches ------------------------------------------------------------


def test_uncovered_languages_raise_by_default(synthetic_survey):
    with pytest.raises(UncoveredLanguageError) as info:
        project(synthetic_survey, angles=LABEL, angle_sets=[SYNTHETIC])
    message = str(info.value)
    assert "qad" in message
    assert "2 responses" in message
    assert f"qae (Low in {LABEL}, no adjusted angles): 1 response" in message
    assert "on_uncovered" in message
    assert "qaa" not in message
    assert synthetic_survey.status.counts()["included"] == 9  # the survey is untouched


def test_uncovered_exclude_writes_the_reason(synthetic_survey):
    with pytest.warns(SoundscapyWarning):  # qab is Medium and qac falls back to equal
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="exclude",
            on_low_confidence="equal",
        )
    reasons = coords.labels[COL_REASONS]
    assert reasons[QAD].tolist() == ["uncovered_language"] * 2
    assert reasons[QAE].tolist() == ["uncovered_language"]
    assert not coords.labels[COL_INCLUDED][QAD + QAE].any()
    assert np.isnan(coords.xy[QAD + QAE]).all()
    assert coords.labels[COL_ANGLE_SET][QAD + QAE].tolist() == ["", "", ""]
    assert n_valid(coords) == 9 - 3
    assert coords.labels[COL_REASONS][BLANK] == "incomplete_attributes"


def test_uncovered_equal_projects_with_nominal_angles_and_warns_once(synthetic_survey):
    with pytest.warns(SoundscapyWarning) as record:
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="equal",
            on_low_confidence="exclude",
        )
    fallbacks = [w for w in record if w.category is EqualAnglesFallbackWarning]
    assert len(fallbacks) == 1
    message = str(fallbacks[0].message)
    assert "qad" in message
    assert "qae" in message
    assert "equal_angles_fallback" in message
    assert coords.labels[COL_ANGLE_SET][QAD + QAE].tolist() == ["equal"] * 3
    assert coords.labels[COL_FLAGS][QAD + QAE].tolist() == ["equal_angles_fallback"] * 3
    assert coords.labels[COL_INCLUDED][QAD + QAE].all()
    scores = synthetic_survey.scores["attributes"][QAD + QAE]
    nominal = np.tile(NOMINAL, (3, 1))
    expected = project_scores(scores, nominal, coords.origin[0].effective_scale, "canonical")
    # exact: the same kernel on the same inputs
    np.testing.assert_array_equal(coords.xy[QAD + QAE], expected)


# --- the three on_low_confidence branches -------------------------------------------------------


def test_low_confidence_raises_once_uncovered_is_dealt_with(synthetic_survey):
    with pytest.raises(LowConfidenceLanguageError) as info:
        project(synthetic_survey, angles=LABEL, angle_sets=[SYNTHETIC], on_uncovered="exclude")
    message = str(info.value)
    assert f"qac (Low in {LABEL}): 3 responses" in message
    assert "on_low_confidence" in message


def test_low_confidence_exclude_writes_the_reason(synthetic_survey):
    with medium_warning():
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="exclude",
            on_low_confidence="exclude",
        )
    assert coords.labels[COL_REASONS][QAC].tolist() == ["low_confidence_language"] * 3
    assert np.isnan(coords.xy[QAC]).all()
    assert coords.labels[COL_ANGLE_SET][QAC].tolist() == [""] * 3
    assert n_valid(coords) == 9 - 3 - 3


def test_low_confidence_equal_falls_back_and_warns(synthetic_survey):
    with pytest.warns(SoundscapyWarning) as record:
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="exclude",
            on_low_confidence="equal",
        )
    fallbacks = [w for w in record if w.category is EqualAnglesFallbackWarning]
    assert len(fallbacks) == 1
    assert "qac" in str(fallbacks[0].message)
    assert "qad" not in str(fallbacks[0].message)
    assert coords.labels[COL_FLAGS][QAC].tolist() == ["equal_angles_fallback"] * 3
    assert coords.labels[COL_ANGLE_SET][QAC].tolist() == ["equal"] * 3


def test_low_confidence_exclude_alone_leaves_the_tier_only_language_raising(synthetic_survey):
    # A tier-only Low language has no angles, so it follows on_uncovered, not on_low_confidence.
    with pytest.raises(UncoveredLanguageError, match="qae"):
        project(synthetic_survey, angles=LABEL, angle_sets=[SYNTHETIC], on_low_confidence="exclude")


# --- the High and Medium rows -------------------------------------------------------------------


def test_high_and_medium_rows_project_with_the_set_angles(synthetic_survey):
    with medium_warning() as record:
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="exclude",
            on_low_confidence="exclude",
        )
    assert len(record) == 1
    message = str(record[0].message)
    assert f"qab (Medium in {LABEL}): 1 response" in message
    assert "medium_confidence_language" in message
    assert coords.labels[COL_ANGLE_SET][QAA + QAB].tolist() == [LABEL] * 3
    assert coords.labels[COL_FLAGS][QAA].tolist() == ["", ""]
    assert coords.labels[COL_FLAGS][QAB].tolist() == ["medium_confidence_language"]
    assert coords.labels[COL_INCLUDED][QAA + QAB].all()
    scores = synthetic_survey.scores["attributes"]
    scale = coords.origin[0].effective_scale
    for rows, code in ((QAA, "qaa"), (QAB, "qab")):
        angles = np.tile(SYNTHETIC.by_language[code].angles_deg, (len(rows), 1))
        # exact: the same kernel on the same inputs
        expected = project_scores(scores[rows], angles, scale, "canonical")
        np.testing.assert_array_equal(coords.xy[rows], expected)
    assert coords.n() == 10
    assert coords.labels[COL_LANGUAGE].tolist() == LANGS


def test_per_language_sends_one_language_to_another_set(synthetic_survey):
    selection = AngleSelection.model_validate(
        {
            "default": LABEL,
            "per_language": {"qab": "aletta-2024@1"},
            "on_uncovered": "exclude",
            "on_low_confidence": "exclude",
        }
    )
    coords = project(
        synthetic_survey, angles=selection, angle_sets=[SYNTHETIC]
    )  # no warning: qab is not Medium
    assert coords.labels[COL_REASONS][QAB].tolist() == ["uncovered_language"]
    assert coords.labels[COL_ANGLE_SET][QAA].tolist() == [LABEL] * 2
    assert coords.origin[0].angle_sets == (LABEL,)


def test_keywords_win_over_the_selection_and_settings(synthetic_survey):
    selection = AngleSelection.model_validate({"default": LABEL, "on_uncovered": "error"})
    with medium_warning():
        coords = project(
            synthetic_survey,
            angles=selection,
            settings=ProjectionSettings(formulation="aletta2024"),
            on_uncovered="exclude",
            on_low_confidence="exclude",
            angle_sets=[SYNTHETIC],
        )
    assert _settings(coords).formulation == "aletta2024"
    assert _settings(coords).angles == AngleSelection.model_validate(
        {"default": LABEL, "on_uncovered": "exclude", "on_low_confidence": "exclude"}
    )


def test_settings_mapping_supplies_the_selection(synthetic_survey):
    with medium_warning():
        coords = project(
            synthetic_survey,
            settings={"angles": {"default": LABEL, "on_uncovered": "exclude"}},
            on_low_confidence="exclude",
            angle_sets=[SYNTHETIC],
        )
    assert _settings(coords).angles.default == AngleSetRef(name="synthetic-tiers", version="1")
    assert _settings(coords).angles.on_uncovered == "exclude"
    assert _settings(coords).angles.on_low_confidence == "exclude"
    assert coords.provenance["settings_hash"] == hash_json(coords.settings)


# --- the status after resolution ----------------------------------------------------------------


def test_select_after_an_exclusion_keeps_the_new_reasons(synthetic_survey):
    with medium_warning():
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="exclude",
            on_low_confidence="exclude",
        )
    chosen = coords.select(np.isin(np.arange(10), QAC + QAD))
    assert (
        chosen.labels[COL_REASONS].tolist()
        == ["low_confidence_language"] * 3 + ["uncovered_language"] * 2
    )
    assert n_valid(chosen) == 0
    assert chosen.origin[0].angle_sets == (LABEL,)


def test_provenance_carries_the_survey_keys_and_the_set(synthetic_survey):
    with medium_warning():
        coords = project(
            synthetic_survey,
            angles=LABEL,
            angle_sets=[SYNTHETIC],
            on_uncovered="exclude",
            on_low_confidence="exclude",
        )
    prov = coords.provenance
    refs = _data_refs(coords)
    assert prov["quality_skipped"] == synthetic_survey.provenance["quality_skipped"]
    for key in ("input_hash", "mapping_hash", "response_id_source", "read", "id_columns"):
        assert prov[key] == synthetic_survey.provenance[key], key
    survey_refs = synthetic_survey.provenance["data_refs"]
    assert isinstance(survey_refs, list)
    assert refs[: len(survey_refs)] == survey_refs
    (set_ref,) = refs[len(survey_refs) :]
    assert set_ref == {
        "kind": "angle_set",
        "name": "synthetic-tiers",
        "version": "1",
        "hash": hash_json(SYNTHETIC),
        "content": SYNTHETIC.model_dump(mode="json", by_alias=True),
    }
    assert coords.origin[0].angle_sets == (LABEL,)


# --- passed sets --------------------------------------------------------------------------------


def test_a_passed_set_shadows_the_shipped_one(hand_survey):
    eng = SATP.by_language["eng"].angles_deg
    shifted = tuple(angle + 3.0 for angle in eng)
    user = _user_set(
        name="satp", version="1.5", languages={"eng": {"tier": "high", "angles_deg": shifted}}
    )
    shipped = project(hand_survey, angles="satp@1.5")
    coords = project(hand_survey, angles="satp@1.5", angle_sets=[user])
    assert coords.labels[COL_ANGLE_SET].tolist() == ["satp@1.5", "satp@1.5", ""]
    assert not np.allclose(coords.xy[:2], shipped.xy[:2])  # the user's angles were used
    scores = hand_survey.scores["attributes"][:2]
    scale = coords.origin[0].effective_scale
    expected = project_scores(scores, np.tile(shifted, (2, 1)), scale, "canonical")
    # exact: the same kernel on the same inputs
    np.testing.assert_array_equal(coords.xy[:2], expected)
    refs = [ref for ref in _data_refs(coords) if ref["kind"] == "angle_set"]
    assert refs == [
        {
            "kind": "angle_set",
            "name": "satp",
            "version": "1.5",
            "hash": hash_json(user),
            "content": user.model_dump(mode="json", by_alias=True),
        }
    ]
    assert pin("angle_sets/satp/1.5.toml") not in str(_data_refs(coords))


def test_a_set_for_another_circumplex_version_raises(hand_survey):
    user = _user_set(
        circumplex_version="2", languages={"eng": {"tier": "high", "angles_deg": list(NOMINAL)}}
    )
    with pytest.raises(DataReferenceError) as info:
        project(hand_survey, angles=LABEL, angle_sets=[user])
    assert "iso-12913-outdoor@2" in str(info.value)
    assert "iso-12913-outdoor@1" in str(info.value)


def test_angle_sets_rejects_a_string_and_a_bare_set(hand_survey):
    with pytest.raises(TypeError, match="angle_sets"):
        project(hand_survey, angles=LABEL, angle_sets="synthetic-tiers@1")  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime TypeError
    with pytest.raises(TypeError, match="angle_sets"):
        project(hand_survey, angles=LABEL, angle_sets=SYNTHETIC)  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime TypeError


def test_angle_sets_rejects_a_repeated_label(hand_survey):
    with pytest.raises(SettingsError, match=LABEL):
        project(hand_survey, angles=LABEL, angle_sets=[SYNTHETIC, SYNTHETIC])


def test_a_named_set_that_is_not_passed_is_looked_up_among_the_shipped(hand_survey):
    with pytest.raises(DataReferenceError, match="synthetic-tiers"):
        project(hand_survey, angles=LABEL)


# --- shipped sets on real surveys ---------------------------------------------------------------


def test_satp_projects_english_with_its_angles(hand_survey):
    coords = project(hand_survey, angles="satp@1.5")
    assert coords.labels[COL_ANGLE_SET].tolist() == ["satp@1.5", "satp@1.5", ""]
    eng = np.tile(SATP.by_language["eng"].angles_deg, (2, 1))
    scores = hand_survey.scores["attributes"][:2]
    # exact: the same kernel on the same inputs
    expected = project_scores(scores, eng, coords.origin[0].effective_scale, "canonical")
    np.testing.assert_array_equal(coords.xy[:2], expected)
    assert _settings(coords).angles.default == AngleSetRef(name="satp", version="1.5")
    assert coords.origin[0].angle_sets == ("satp@1.5",)
    assert _data_refs(coords)[-1] == {
        "kind": "angle_set",
        "name": "satp",
        "version": "1.5",
        "hash": pin("angle_sets/satp/1.5.toml"),
    }


def test_an_angle_set_ref_is_accepted(hand_survey):
    by_ref = project(hand_survey, angles=AngleSetRef(name="satp", version="1.5"))
    by_string = project(hand_survey, angles="satp@1.5")
    np.testing.assert_array_equal(by_ref.xy, by_string.xy)
    assert _settings(by_ref) == _settings(by_string)


def test_satp_on_the_subset_flags_exactly_the_cmn_rows(isd_survey):
    with medium_warning() as record:
        coords = project(isd_survey, angles="satp@1.5")
    assert len(record) == 1
    assert "cmn (Medium in satp@1.5): 69 responses" in str(record[0].message)
    flagged = _has_flag(coords, "medium_confidence_language")
    cmn = coords.labels[COL_LANGUAGE] == "cmn"
    included = coords.labels[COL_INCLUDED]
    np.testing.assert_array_equal(flagged, cmn & included)
    assert int(cmn.sum()) == 69
    assert int(flagged.sum()) == 69 - int((cmn & ~included).sum())
    assert coords.origin[0].angle_sets == ("satp@1.5",)
    included = coords.labels[COL_INCLUDED]
    assert set(coords.labels[COL_ANGLE_SET][included].tolist()) == {"satp@1.5"}
    assert n_valid(coords) == 1536


def test_aletta_2024_covers_every_subset_language_without_warning(isd_survey):
    coords = project(isd_survey, angles="aletta-2024@1")  # filterwarnings = error: none fired
    included = coords.labels[COL_INCLUDED]
    assert n_valid(coords) == int(included.sum()) == 1536
    assert set(coords.labels[COL_ANGLE_SET][included].tolist()) == {"aletta-2024@1"}
    assert not _has_flag(coords, "medium_confidence_language").any()
    assert not _has_flag(coords, "equal_angles_fallback").any()
    assert coords.origin[0].angle_sets == ("aletta-2024@1",)


def test_equal_default_stays_warning_free_on_the_subset(isd_survey):
    coords = project(isd_survey)
    assert _settings(coords).angles == AngleSelection()
    assert coords.origin[0].angle_sets == ("equal",)


# --- the mapping's language -----------------------------------------------------------------------


def test_a_named_set_needs_a_mapped_language(tmp_path):
    header = "ID,pleasant,vibrant,eventful,chaotic,annoying,monotonous,uneventful,calm"
    path = write_csv(tmp_path, header, "r1,5,3,3,3,1,3,3,3")
    survey = read_survey(path, SurveyMapping.model_validate(raw_mapping({"response": "ID"})))
    with pytest.raises(MappingError, match="language"):
        project(survey, angles="satp@1.5")
    assert project(survey).origin[0].angle_sets == ("equal",)  # equal never reads the language


def test_a_blank_language_on_an_included_row_raises_naming_it(tmp_path):
    header = "ID,lang,pleasant,vibrant,eventful,chaotic,annoying,monotonous,uneventful,calm"
    path = write_csv(tmp_path, header, "r1,eng,5,3,3,3,1,3,3,3", "r2,,4,5,2,1,2,1,4,5")
    mapping = {**raw_mapping({"response": "ID"}), "language": {"column": "lang"}}
    survey = read_survey(path, SurveyMapping.model_validate(mapping))
    with pytest.raises(MappingError, match="r2") as info:
        project(survey, angles="satp@1.5")
    assert "r1" not in str(info.value)
    assert "language" in str(info.value)
