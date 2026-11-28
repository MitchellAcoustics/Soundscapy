"""The S3 done-when (spec 9, row S3): angle sets, profiles, settings and labels.

It repeats assertions made elsewhere on purpose: ``uv run pytest tests/science/test_s3_done.py``
is the one command that shows the slice is done, as ``test_s2_done.py`` does for S2.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from soundscapy import (
    AngleSelection,
    AngleSetRef,
    ProjectionSettings,
    SurveyMapping,
    _data,
    circumplex,
    project,
    read_survey,
)
from soundscapy._model import COL_FLAGS, from_toml, to_toml
from soundscapy.errors import FormulationError, ProfileConstraintError, UncoveredLanguageError
from soundscapy.instrument import angle_set
from soundscapy.provenance import hash_json
from tests._helpers import COMPLETE, COMPLETE_B, LABELS, medium_warning, raw_mapping, write_csv

pytestmark = pytest.mark.science

FIXTURES = Path(__file__).parents[1] / "projection" / "fixtures"
HAND_CSV = FIXTURES / "hand_values.csv"
MAPPING = FIXTURES / "mapping.toml"
SCALE_1_7_CSV = FIXTURES / "scale_1_7.csv"
MAPPING_1_7 = FIXTURES / "mapping_1_7.toml"

SATP = AngleSetRef(name="satp", version="1.5")
# The 17 languages of Aletta et al. (2024) Table 4 besides English.
TRANSLATED = (
    "arb", "cmn", "hrv", "nld", "fra", "deu", "ell", "ind", "ita",
    "jpn", "kor", "zsm", "por", "spa", "swe", "tur", "vie",
)  # fmt: skip
# Medium and Low-tier languages of satp@1.5 with no published labels.
GAP_LANGUAGES = ("pol", "yue", "tha", "bra", "alb")


def _profile_id(provenance) -> str:
    return provenance["profile"]["id"]


def test_default_run_flags_the_mandarin_rows_once(isd_survey):
    # The default run is sal-2026 with satp@1.5, and the Medium warning fires once for cmn.
    with medium_warning() as record:
        coords = project(isd_survey)
    assert len(record) == 1
    flagged = ["medium_confidence_language" in cell.split(";") for cell in coords.labels[COL_FLAGS]]
    assert sum(flagged) == 69
    assert _profile_id(coords.provenance) == "sal-2026"
    assert coords.provenance["overrides"] == {}
    assert coords.origin[0].angle_sets == ("satp@1.5",)


def test_2019_profile_carries_over_from_read_survey_to_project():
    # The 2019 carry-over: the profile named on read_survey governs project.
    survey = read_survey(HAND_CSV, MAPPING, profile="iso-12913-3-2019")
    coords = project(survey)
    for provenance in (survey.provenance, coords.provenance):
        assert _profile_id(provenance) == "iso-12913-3-2019"
        assert provenance["overrides"] == {}
    assert isinstance(coords.settings, ProjectionSettings)
    assert coords.settings.formulation == "iso2019"
    assert coords.origin[0].angle_sets == ("equal",)


def test_iso2019_is_refused_on_a_seven_point_scale():
    # iso2019 is refused on 1..7.
    survey = read_survey(SCALE_1_7_CSV, MAPPING_1_7)
    with pytest.raises(FormulationError):
        project(survey, formulation="iso2019", angles="equal")


@pytest.mark.parametrize("profile", ["iso-12913-3-2019", "iso-12913-3-2024"])
def test_iso_profiles_refuse_aletta2024(profile):
    # The ISO profiles refuse aletta2024.
    survey = read_survey(HAND_CSV, MAPPING)
    with pytest.raises(ProfileConstraintError, match=re.escape("projection.formulation")):
        project(survey, profile=profile, formulation="aletta2024")


def test_sal_2026_accepts_aletta2024_and_records_the_override():
    # sal-2026 leaves the formulation open, so aletta2024 is an override it records.
    coords = project(read_survey(HAND_CSV, MAPPING), formulation="aletta2024", angles="equal")
    assert _profile_id(coords.provenance) == "sal-2026"
    assert coords.provenance["overrides"] == {
        "projection.formulation": "aletta2024",
        "projection.angles.default": "equal",
    }


def test_tier_only_language_is_uncovered_and_names_its_tier(tmp_path):
    # Tier-only languages resolve as uncovered and name their tier.
    path = write_csv(tmp_path, "ID," + ",".join(LABELS), f"k1,{COMPLETE}", f"k2,{COMPLETE_B}")
    raw = raw_mapping({"response": "ID"})
    raw["language"] = {"value": "kor"}
    survey = read_survey(path, SurveyMapping.model_validate(raw))
    with pytest.raises(UncoveredLanguageError) as info:
        project(survey, angles="satp@1.5")
    assert "kor" in str(info.value)
    assert "Medium" in str(info.value)


def test_translated_languages_have_labels_and_a_citation():
    # The sourced languages have eight labels each, with a citation.
    c = circumplex("iso-12913-outdoor", "1")
    by_language = {tr.language: tr for tr in c.translations}
    assert sorted(by_language) == sorted(TRANSLATED)
    for tr in by_language.values():
        assert len(tr.labels) == 8
        assert tr.citation.citation


@pytest.mark.parametrize("language", GAP_LANGUAGES)
def test_gap_languages_fall_back_to_english(language):
    # The gap languages fall back to English.
    assert circumplex("iso-12913-outdoor", "1").label("PAQ1", language) == "pleasant"


def test_every_shipped_file_matches_its_pin():
    # The committed angle-set and profile files match their pins (read_bytes verifies).
    for relpath in _data.PINS:
        _data.read_bytes(relpath)
    assert any(rel.startswith("angle_sets/") for rel in _data.PINS)
    assert any(rel.startswith("profiles/") for rel in _data.PINS)


@pytest.mark.parametrize(
    "model",
    [
        angle_set("satp", "1.5"),
        AngleSelection(default=SATP, per_language=(("cmn", SATP),), on_uncovered="exclude"),
        ProjectionSettings(formulation="aletta2024", angles=AngleSelection(default=SATP)),
    ],
    ids=lambda m: type(m).__name__,
)
def test_new_models_round_trip_through_toml_with_an_equal_hash(model):
    # Every model round-trips through TOML with an equal hash.
    back = from_toml(type(model), to_toml(model))
    assert back == model
    assert hash_json(back) == hash_json(model)
