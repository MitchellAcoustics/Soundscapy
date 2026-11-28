"""read_survey with every group: the exclude path, quality rules, flags, age and settings."""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from soundscapy.errors import CodingError, MappingError, SettingsError
from soundscapy.projection.project import project
from soundscapy.survey.mapping import SurveyMapping, read_mapping
from soundscapy.survey.quality import QualitySettings
from soundscapy.survey.read import ImportSettings, read_survey
from soundscapy.survey.status import Flag, Reason
from tests._helpers import COMPLETE, LABELS, quality_settings, raw_mapping, write_csv

FIXTURES = Path(__file__).parent / "fixtures"
SCALE_0_4_CSV = FIXTURES / "survey_0_4.csv"
NAN = math.nan
SOURCE_COLUMNS = ("ssi01", "ssi02", "ssi03", "ssi04")
SOURCE_KEYS = ("traffic_noise", "other_noise", "human_sounds", "natural_sounds")
FLAG_COLUMNS = {"consent": "consent", "test": "test", "paper_input": "paper"}


def _mapping(ids: dict | None = None, **fields: object) -> SurveyMapping:
    raw = raw_mapping(ids if ids is not None else {"response": "ID"})
    raw.update(fields)
    return SurveyMapping.model_validate(raw)


def _full_mapping(source_keys: tuple[str, ...] = SOURCE_KEYS) -> SurveyMapping:
    """Every S2 field: attributes, a source group, age and the three flags."""
    raw = raw_mapping({"response": "ID"})
    raw["groups"]["source_dominance"] = {
        "columns": {
            key: col
            for key, col in zip(SOURCE_KEYS, SOURCE_COLUMNS, strict=True)
            if key in source_keys
        }
    }
    raw["age"] = "age"
    raw["flags"] = {
        name: {"column": col, "true_values": ["yes"]} for name, col in FLAG_COLUMNS.items()
    }
    return SurveyMapping.model_validate(raw)


TEN_HEADER = ",".join(("ID", *LABELS, *SOURCE_COLUMNS, "age", "consent", "test", "paper"))
# The ten rows of tests/survey/test_quality.py's fixture, as a user's file would hold them.
TEN_ROWS = (
    f"r0,{COMPLETE},3,3,3,3,40,yes,no,no",
    "r1,4,4,4,4,4,4,4,4,3,3,3,3,40,yes,no,no",
    "r2,3,3,3,3,3,3,3,3,3,3,3,3,40,yes,no,no",
    "r3,1,2,3,4,5,4,3,,3,3,3,3,40,yes,no,no",
    "r4,1,,,,,,,,,,,,40,yes,no,no",
    f"r5,{COMPLETE},3,3,3,3,40,,no,no",
    f"r6,{COMPLETE},3,3,3,3,40,yes, yes ,no",
    f"r7,{COMPLETE},3,3,3,3,40,yes,no,yes",
    f"r8,{COMPLETE},3,3,3,3,16,yes,no,no",
    f"r9,{COMPLETE},3,3,3,3,,yes,no,no",
)
TEN_REASONS = (
    (),
    (Reason.ALL_IDENTICAL,),
    (),
    (Reason.INCOMPLETE_ATTRIBUTES,),
    (Reason.EXCESSIVE_MISSING, Reason.INCOMPLETE_ATTRIBUTES),
    (Reason.NO_CONSENT,),
    (Reason.TEST_RESPONSE,),
    (),
    (),
    (),
)


def _ten(tmp_path: Path) -> Path:
    return write_csv(tmp_path, TEN_HEADER, *TEN_ROWS)


# --- the exclude path (Review Focus 4) ---------------------------------------------------------


def test_exclude_path_records_both_reasons():
    survey = read_survey(SCALE_0_4_CSV, _mapping(), on_scale_violation="exclude")
    both = (Reason.SCALE_VIOLATION, Reason.INCOMPLETE_ATTRIBUTES)
    assert survey.status.reasons == (both, both, both)
    nan = np.isnan(survey.scores["attributes"])
    # Position order: pleasant, vibrant, eventful, chaotic, annoying, monotonous, uneventful, calm.
    expected = np.zeros((3, 8), dtype=bool)
    expected[0, [0, 4]] = True  # a: pleasant 0, annoying 0
    expected[1, 7] = True  # b: calm 0
    expected[2, [0, 4]] = True  # c: pleasant 0, annoying blank
    assert nan.tolist() == expected.tolist()
    assert survey.settings.on_scale_violation == "exclude"
    assert project(survey).n_valid() == 0


def test_offset_fixes_the_0_4_export():
    raw = raw_mapping({"response": "ID"})
    raw["groups"]["attributes"]["coding"] = {"offset": 1.0}
    survey = read_survey(SCALE_0_4_CSV, SurveyMapping.model_validate(raw))
    assert survey.status.included.tolist() == [True, True, False]
    assert survey.status.reasons[2] == (Reason.INCOMPLETE_ATTRIBUTES,)


# --- a non-English, relabelled export (spec 8, FR-1.3) ----------------------------------------


def test_labelled_export_reads():
    survey = read_survey(
        FIXTURES / "labelled_export.csv", read_mapping(FIXTURES / "labelled_mapping.toml")
    )
    # Exact: labels map to integers, and the offset and reverse are integer arithmetic.
    np.testing.assert_array_equal(
        survey.scores["attributes"],
        [
            [5, 4, 4, 2, 1, 3, 2, 2],
            [5, 4, 4, 2, 1, 3, 2, NAN],
            [5, NAN, 4, 2, 1, 3, 2, 2],
            [5, 4, 4, 2, 1, 3, 2, 5],
        ],
    )
    np.testing.assert_array_equal(survey.scores["overall_quality"][:, 0], [4, 5, NAN, 1])
    assert survey.status.included.tolist() == [True, False, False, True]
    assert survey.status.reasons[1] == (Reason.INCOMPLETE_ATTRIBUTES,)
    assert survey.status.reasons[2] == (Reason.INCOMPLETE_ATTRIBUTES,)
    assert survey.language.tolist() == ["spa"] * 4
    assert set(survey.scores) == {"attributes", "overall_quality"}


# --- every rule through read_survey -----------------------------------------------------------


def test_every_rule_through_read_survey(tmp_path):
    survey = read_survey(_ten(tmp_path), _full_mapping())
    assert survey.status.reasons == TEN_REASONS
    assert [i for i, row in enumerate(survey.status.flags) if Flag.ALL_MIDPOINT in row] == [2]
    assert [i for i, row in enumerate(survey.status.flags) if Flag.PAPER_INPUT in row] == [7]
    assert [i for i, row in enumerate(survey.status.flags) if Flag.UNDER_AGE in row] == [8]
    assert np.flatnonzero(~survey.status.included).tolist() == [1, 3, 4, 5, 6]
    assert survey.provenance["quality_skipped"] == []
    assert set(survey.scores) == {"attributes", "source_dominance"}
    assert survey.scores["source_dominance"].shape == (10, 4)


def _missing(k: int) -> str:
    """A row's 10 item cells with the first ``k`` missing and the rest 3."""
    return ",".join([""] * k + ["3"] * (10 - k))


def test_partial_source_dominance_reads(tmp_path):
    header = ",".join(("ID", *LABELS, "ssi01", "ssi03"))
    path = write_csv(tmp_path, header, f"six,{_missing(6)}", f"five,{_missing(5)}")
    raw = raw_mapping({"response": "ID"})
    raw["groups"]["source_dominance"] = {
        "columns": {"traffic_noise": "ssi01", "human_sounds": "ssi03"}
    }
    survey = read_survey(path, SurveyMapping.model_validate(raw))
    assert survey.scores["source_dominance"].shape == (2, 2)
    assert survey.mapping.groups.source_dominance.columns == (
        ("traffic_noise", "ssi01"),
        ("human_sounds", "ssi03"),
    )
    assert Reason.EXCESSIVE_MISSING in survey.status.reasons[0]  # 6 of 10 missing, share 0.6
    assert Reason.EXCESSIVE_MISSING not in survey.status.reasons[1]  # 5 of 10, share 0.5


@pytest.mark.parametrize("text", ["prefer not to say", "25 years"])
def test_age_text_not_a_number_raises(tmp_path, text):
    header = ",".join(("ID", *LABELS, "age00"))
    path = write_csv(tmp_path, header, f"r1,{COMPLETE},{text}", f"r2,{COMPLETE},30")
    with pytest.raises(CodingError) as info:
        read_survey(path, _mapping(age="age00"))
    message = str(info.value)
    assert "'age00'" in message
    assert f"{text!r} x1" in message
    assert "r1" in message
    assert "leave age unmapped" in info.value.hint


def test_missing_flag_column_raises(tmp_path):
    path = write_csv(tmp_path, ",".join(("ID", *LABELS)), f"r1,{COMPLETE}")
    flags = {"consent": {"column": "consent", "true_values": ["yes"]}}
    with pytest.raises(MappingError, match=r"flags\.consent\.column"):
        read_survey(path, _mapping(flags=flags))


def test_missing_carry_column_raises(tmp_path):
    path = write_csv(tmp_path, ",".join(("ID", *LABELS)), f"r1,{COMPLETE}")
    with pytest.raises(MappingError, match=r"carry\[0\]"):
        read_survey(path, _mapping(carry=["start_time"]))


def test_quality_skipped_recorded(tmp_path):
    none_mapped = read_survey(_ten(tmp_path), _mapping())
    assert none_mapped.provenance["quality_skipped"] == [
        "no_consent",
        "test_response",
        "paper_input",
        "under_age",
    ]
    assert read_survey(_ten(tmp_path), _full_mapping()).provenance["quality_skipped"] == []


# --- settings ------------------------------------------------------------------------------------


def test_settings_precedence(tmp_path):
    path = _ten(tmp_path)
    survey = read_survey(
        path,
        _full_mapping(),
        settings={"quality": {"min_age": 16.0, "under_age": "exclude"}},
        quality={"under_age": "off"},
    )
    assert survey.settings.quality.min_age == 16.0
    assert survey.settings.quality.under_age == "off"
    survey = read_survey(
        path,
        _full_mapping(),
        settings={"on_scale_violation": "error"},
        on_scale_violation="exclude",
    )
    assert survey.settings.on_scale_violation == "exclude"
    survey = read_survey(
        path, _full_mapping(), settings=ImportSettings(on_scale_violation="exclude")
    )
    assert survey.settings.on_scale_violation == "exclude"
    survey = read_survey(
        path,
        _full_mapping(),
        settings={"quality": QualitySettings(min_age=16.0)},
        quality={"under_age": "off"},
    )
    assert (survey.settings.quality.min_age, survey.settings.quality.under_age) == (16.0, "off")
    with pytest.raises(SettingsError, match=r"quality\.all_identical"):
        read_survey(path, _full_mapping(), quality={"all_identical": "flag"})
    with pytest.raises(SettingsError, match="on_scale_violation"):
        read_survey(path, _full_mapping(), on_scale_violation="drop")  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime SettingsError
    with pytest.raises(SettingsError, match="settings file"):
        read_survey(path, _full_mapping(), settings="settings.toml")  # ty: ignore[invalid-argument-type]  # deliberate: the S2 type has no path member


def test_rerun_with_recorded_settings(tmp_path):
    path = _ten(tmp_path)
    first = read_survey(
        path, _full_mapping(), quality={"min_age": 16.0}, on_scale_violation="exclude"
    )
    again = read_survey(path, _full_mapping(), settings=first.settings)
    assert again.status.reasons == first.status.reasons
    assert again.status.flags == first.status.flags
    for name in first.scores:
        np.testing.assert_array_equal(again.scores[name], first.scores[name])  # exact: same bytes
    assert again.provenance["settings_hash"] == first.provenance["settings_hash"]


def test_info_summary_names_flags(caplog, tmp_path):
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        read_survey(_ten(tmp_path), _full_mapping())
    records = [r for r in caplog.records if r.name == "soundscapy"]
    assert len(records) == 1
    # Flags in Flag enum order: all_midpoint, under_age, paper_input.
    assert "; flagged (all_midpoint: 1, under_age: 1, paper_input: 1)" in records[0].getMessage()


# --- the row-count property (spec 7.1, the S2 done-when) ----------------------------------------

_CELL = st.sampled_from(["", "1", "2", "3", "4", "5", "0", "NA"])
_AGE = st.sampled_from(["", "12", "17", "18", "40"])


@st.composite
def _survey_files(draw):
    n = draw(st.integers(1, 12))
    rows = []
    for i in range(n):
        if draw(st.booleans()):
            rows.append("")  # a blank line the reader skips, keeping its number out of use
        cells = [draw(_CELL) for _ in range(12)]
        rows.append(",".join([f"r{i}", *cells, draw(_AGE)]))
    return n, rows, draw(quality_settings())


@given(_survey_files())
@settings(deadline=None, max_examples=60)  # per-example file I/O
def test_row_count_property(case):
    n, rows, quality = case
    header = ",".join(("ID", *LABELS, *SOURCE_COLUMNS, "age"))
    raw = raw_mapping({"response": "ID"})
    raw["groups"]["source_dominance"] = {
        "columns": dict(zip(SOURCE_KEYS, SOURCE_COLUMNS, strict=True))
    }
    raw["age"] = "age"
    mapping = SurveyMapping.model_validate(raw)
    with tempfile.TemporaryDirectory() as folder:
        path = write_csv(Path(folder), header, *rows)
        survey = read_survey(path, mapping, quality=quality, on_scale_violation="exclude")
    status = survey.status
    assert survey.n() == len(status.reasons) == len(status.flags) == project(survey).n() == n
    assert status.included.tolist() == [len(r) == 0 for r in status.reasons]
    assert survey.response_id.tolist() == [f"r{i}" for i in range(n)]
