"""Angle sets: the models, the shipped-set loader, references and the user-file reader."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from soundscapy._model import to_toml
from soundscapy.errors import DataReferenceError, SettingsError
from soundscapy.instrument import AngleSet, AngleSetRef, angle_set, read_angle_set
from soundscapy.instrument.angles import LanguageAngles

ANGLES = (0.0, 46.0, 94.0, 138.0, 177.0, 231.0, 275.0, 340.0)
SHIPPED_LABELS = ("aletta-2024@1", "satp@1.5")


def _set(**changes) -> AngleSet:
    data = {
        "schema": "soundscapy.angle_set",
        "schema_version": 1,
        "name": "test-set",
        "version": "1",
        "circumplex": "iso-12913-outdoor",
        "circumplex_version": "1",
        "languages": {
            "eng": {"tier": "high", "angles_deg": list(ANGLES)},
            "fra": {"tier": "medium", "note": "tier only"},
        },
    }
    return AngleSet.model_validate({**data, **changes})


def test_tier_only_language_accepted():
    s = _set()
    assert s.label == "test-set@1"
    assert s.by_language["fra"] == LanguageAngles(tier="medium", note="tier only")
    assert s.by_language["fra"].angles_deg == ()
    assert s.by_language["eng"].angles_deg == ANGLES
    assert s.by_language["eng"].has_angles
    assert not s.by_language["fra"].has_angles


@pytest.mark.parametrize(
    ("languages", "fragment"),
    [
        ({"eng": {"tier": "high"}}, "high"),
        ({"eng": {"tier": "high", "angles_deg": [*ANGLES[:-1], 360.0]}}, r"\[0, 360\)"),
        ({"eng": {"tier": "high", "angles_deg": [*ANGLES[:-1], -1.0]}}, r"\[0, 360\)"),
        ({"eng": {"tier": "high", "angles_deg": [0.0, 94.0, 46.0]}}, "strictly increasing"),
        (
            {
                "eng": {"tier": "high", "angles_deg": list(ANGLES)},
                "deu": {"tier": "high", "angles_deg": list(ANGLES[:4])},
            },
            "same number of angles",
        ),
        ({"en": {"tier": "medium"}}, "'en'"),
        ({"eng": {"tier": "top"}}, "tier"),
    ],
)
def test_validators_reject(languages, fragment):
    with pytest.raises(ValidationError, match=fragment):
        _set(languages=languages)


def test_name_equal_rejected():
    with pytest.raises(ValidationError, match="equal"):
        _set(name="equal")


def test_ref_parse_round_trips_to_label():
    ref = AngleSetRef.parse("satp@1.5")
    assert ref == AngleSetRef(name="satp", version="1.5")
    assert ref.label == "satp@1.5"
    assert AngleSetRef.parse(ref.label) == ref


@pytest.mark.parametrize("text", ["satp", "satp@", "@1.5", ""])
def test_ref_parse_rejects_a_form_without_a_version(text):
    with pytest.raises(DataReferenceError) as info:
        AngleSetRef.parse(text)
    for label in SHIPPED_LABELS:
        assert label in str(info.value)


def test_shipped_aletta_2024():
    s = angle_set("aletta-2024", "1")
    assert s.label == "aletta-2024@1"
    assert (s.circumplex, s.circumplex_version) == ("iso-12913-outdoor", "1")
    assert len(s.languages) == 18
    assert sum(entry.has_angles for _, entry in s.languages) == 12
    assert angle_set("aletta-2024", "1") is s  # cached


def test_shipped_satp_1_5():
    s = angle_set("satp", "1.5")
    assert s.label == "satp@1.5"
    assert len(s.languages) == 23


def test_unknown_angle_set_lists_shipped():
    with pytest.raises(DataReferenceError) as info:
        angle_set("nope", "1")
    for label in SHIPPED_LABELS:
        assert label in str(info.value)


def test_read_angle_set_round_trip(tmp_path):
    s = _set()
    path = tmp_path / "my-set.toml"
    path.write_text(to_toml(s), encoding="utf-8")
    assert read_angle_set(path) == s


def test_read_angle_set_bad_tier_names_the_field(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(to_toml(_set()).replace('tier = "medium"', 'tier = "top"'), encoding="utf-8")
    with pytest.raises(SettingsError, match="tier") as info:
        read_angle_set(path)
    assert "bad.toml" in str(info.value)
    assert "languages" in str(info.value)


def test_read_angle_set_invalid_toml(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("name = ", encoding="utf-8")
    with pytest.raises(SettingsError, match="not valid TOML"):
        read_angle_set(path)


def test_read_angle_set_missing_file(tmp_path):
    with pytest.raises(DataReferenceError, match=re.escape("missing.toml")):
        read_angle_set(tmp_path / "missing.toml")
