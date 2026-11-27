"""read_survey from a CSV path: response ids, language, coding, scale check, status, provenance."""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tomlkit

from soundscapy import _data
from soundscapy.errors import (
    ColumnCollisionError,
    DuplicateResponseIdError,
    FileFormatError,
    MappingError,
    MissingResponseIdError,
    ScaleViolationError,
    SettingsError,
)
from soundscapy.provenance import envelope, hash_file
from soundscapy.survey.mapping import SurveyMapping
from soundscapy.survey.read import ImportSettings, ReadOptions, check_unique, read_survey
from soundscapy.survey.status import Reason
from tests._helpers import CSV_ORDER, ONES, raw_mapping, write_csv

FIXTURES = Path(__file__).parent / "fixtures"
KEY_CSV = FIXTURES / "survey_key.csv"
COLUMN_CSV = FIXTURES / "survey_column.csv"
ROW_CSV = FIXTURES / "survey_row.csv"
SCALE_0_4_CSV = FIXTURES / "survey_0_4.csv"
KEY = ("LocationID", "SessionID", "GroupID", "RecordID")
HEADER = ",".join((*KEY, "Language", *CSV_ORDER))
ID_HEADER = ",".join(("ID", *CSV_ORDER))
KEY_IDS = {
    "response_key": list(KEY),
    "location": "LocationID",
    "session": "SessionID",
    "group": "GroupID",
    "record": "RecordID",
}


def _raw(ids: dict | None = None, language: dict | None = None) -> dict:
    raw = raw_mapping(ids if ids is not None else {"response_key": list(KEY)}, columns=CSV_ORDER)
    if language is not None:
        raw["language"] = language
    return raw


def _mapping(ids: dict | None = None, language: dict | None = None) -> SurveyMapping:
    return SurveyMapping.model_validate(_raw(ids, language))


def _csv(tmp_path: Path, *rows: str, header: str = HEADER, name: str = "survey.csv") -> Path:
    return write_csv(tmp_path, header, *rows, name=name)


# --- reading by name ---------------------------------------------------------------------------


def test_reads_by_name_not_order():
    survey = read_survey(KEY_CSV, _mapping())
    scores = survey.scores["attributes"]
    assert scores.shape == (4, 8)
    assert scores.dtype == np.float64
    # Exact: small integers. Position order is pleasant, vibrant, eventful, chaotic,
    # annoying, monotonous, uneventful, calm; the first row is 1,2,3,4,5,1,2,3 in CSV order.
    assert scores[0].tolist() == [1.0, 3.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0]
    assert survey.n() == 4
    assert survey.settings == ImportSettings()
    assert survey.source.name == "survey_key.csv"
    assert list(survey.data.columns) == HEADER.split(",")


# --- response ids ------------------------------------------------------------------------------


def test_response_id_from_key():
    survey = read_survey(KEY_CSV, _mapping(KEY_IDS))
    assert survey.response_id.tolist() == ["L1/S1/G1/7", "L1/S1/G1/8", "L1/S2/G2/9", "L2/S1/G1/7"]
    assert survey.provenance["response_id_source"] == "key"
    assert set(survey.ids) == {"location", "session", "group", "record"}
    assert survey.ids["location"].tolist() == ["L1", "L1", "L1", "L2"]


def test_response_id_from_column():
    survey = read_survey(COLUMN_CSV, _mapping({"response": "ID"}))
    assert survey.response_id.tolist() == ["r1", "r2", "r3"]  # " r1 " is stripped
    assert survey.provenance["response_id_source"] == "column"
    assert survey.ids == {}


def test_response_id_from_row():
    # Header, two rows, a blank row, one row: the blank row's number stays out of use.
    survey = read_survey(ROW_CSV, _mapping({}))
    assert survey.response_id.tolist() == ["row-2", "row-3", "row-5"]
    assert survey.provenance["response_id_source"] == "row"


def test_blank_key_part_raises(tmp_path):
    path = _csv(tmp_path, f"L1,S1,G1,7,eng,{ONES}", f"L1,S1,G1, ,eng,{ONES}")
    with pytest.raises(MissingResponseIdError, match="RecordID") as info:
        read_survey(path, _mapping())
    assert "row 3" in str(info.value)


def test_blank_key_parts_in_several_columns_are_reported_together(tmp_path):
    path = _csv(
        tmp_path,
        f"L1,S1,G1,7,eng,{ONES}",
        f"L1,S1,G1, ,eng,{ONES}",
        f"L1,S1,,8,eng,{ONES}",
    )
    with pytest.raises(MissingResponseIdError) as info:
        read_survey(path, _mapping())
    message = str(info.value)
    assert "rows 3, 4" in message
    assert "'GroupID', 'RecordID'" in message


def test_blank_key_part_allowed(tmp_path):
    path = _csv(tmp_path, f"L1,S1,G1,7,eng,{ONES}", f"L1,S1,G1,,eng,{ONES}")
    ids = {"response_key": list(KEY), "response_key_blank_ok": ["RecordID"]}
    survey = read_survey(path, _mapping(ids))
    assert survey.response_id.tolist() == ["L1/S1/G1/7", "L1/S1/G1/"]
    assert survey.provenance["response_key_blank_ok"] == ["RecordID"]


def test_blank_response_column_raises(tmp_path):
    path = _csv(tmp_path, f"a,{ONES}", f"  ,{ONES}", header=ID_HEADER)
    with pytest.raises(MissingResponseIdError, match="row 3"):
        read_survey(path, _mapping({"response": "ID"}))


def test_duplicate_ids_raise(tmp_path):
    path = _csv(tmp_path, f"a,{ONES}", f"b,{ONES}", f"a,{ONES}", header=ID_HEADER)
    with pytest.raises(DuplicateResponseIdError, match="'a'") as info:
        read_survey(path, _mapping({"response": "ID"}))
    assert "rows 2, 4" in str(info.value)


def test_check_unique_lists_repeats_with_their_rows():
    seen: list[list[tuple[str, list[int]]]] = []

    def describe(repeated: list[tuple[str, list[int]]]) -> str:
        seen.append(repeated)
        return "repeats"

    check_unique(["a", "b"], [2, 3], error=SettingsError, describe=describe)
    assert seen == []
    with pytest.raises(SettingsError, match="repeats") as info:
        check_unique(
            ["b", "a", "b", "a", "c"],
            [2, 3, 4, 5, 6],
            error=SettingsError,
            describe=describe,
            hint="fix it",
        )
    assert info.value.hint == "fix it"
    # First-occurrence order, each value with every row it is at.
    assert seen == [[("b", [2, 4]), ("a", [3, 5])]]


def test_key_part_with_slash_raises(tmp_path):
    path = _csv(tmp_path, f"L1,S1,G1/x,7,eng,{ONES}")
    with pytest.raises(MappingError, match="GroupID") as info:
        read_survey(path, _mapping())
    assert "'G1/x'" in str(info.value)


def test_id_error_precedes_coding_error(tmp_path):
    path = _csv(tmp_path, f"a,{ONES}", f"a,loud,{ONES[2:]}", header=ID_HEADER)
    with pytest.raises(DuplicateResponseIdError):
        read_survey(path, _mapping({"response": "ID"}))


# --- language ----------------------------------------------------------------------------------


def test_language_from_column_with_coding():
    language = {"column": "Language", "coding": {"English": "eng"}}
    survey = read_survey(COLUMN_CSV, _mapping({"response": "ID"}, language))
    assert survey.language.tolist() == ["eng", "", "eng"]


def test_language_uncoded_text_raises():
    with pytest.raises(MappingError, match="'English' x1") as info:
        read_survey(COLUMN_CSV, _mapping({"response": "ID"}, {"column": "Language"}))
    assert "r1" in str(info.value)


def test_language_from_value():
    survey = read_survey(KEY_CSV, _mapping(language={"value": "eng"}))
    assert survey.language.tolist() == ["eng"] * 4


def test_language_unmapped_is_blank():
    survey = read_survey(KEY_CSV, _mapping())
    assert survey.language.tolist() == [""] * 4


# --- input checks ------------------------------------------------------------------------------


def test_ssp_column_collision_raises(tmp_path):
    # The file also has an id error; the collision check runs before anything else.
    path = _csv(tmp_path, f"a,1,{ONES}", f"a,1,{ONES}", header=f"ID,ssp_included,{ID_HEADER[3:]}")
    with pytest.raises(ColumnCollisionError, match="ssp_included"):
        read_survey(path, _mapping({"response": "ID"}))


def test_missing_mapped_column_raises(tmp_path):
    header = HEADER.replace("calm", "Calm").replace("Language", "Lang")
    path = _csv(tmp_path, f"L1,S1,G1,7,eng,{ONES}", header=header)
    language = {"column": "Language"}
    with pytest.raises(MappingError, match="'calm'") as info:
        read_survey(path, _mapping(KEY_IDS, language))
    message = str(info.value)
    assert "groups.attributes.columns.PAQ8" in message
    assert "language.column" in message
    assert "'Language'" in message


def test_unknown_suffix_raises(tmp_path):
    path = _csv(tmp_path, f"L1,S1,G1,7,eng,{ONES}", name="survey.txt")
    with pytest.raises(FileFormatError, match=re.escape("'.txt'")):
        read_survey(path, _mapping())
    # An explicit format reads it.
    assert read_survey(path, _mapping(), read_options={"format": "csv"}).n() == 1


def test_tsv_by_suffix(tmp_path):
    rows = [f"L1\tS1\tG1\t7\teng\t{ONES.replace(',', chr(9))}"]
    path = _csv(tmp_path, *rows, header=HEADER.replace(",", "\t"), name="survey.tsv")
    survey = read_survey(path, _mapping())
    assert survey.response_id.tolist() == ["L1/S1/G1/7"]
    assert survey.source.read_options == ReadOptions(format="tsv", delimiter="\t")
    assert survey.provenance["read"] == {
        "format": "tsv",
        "encoding": "utf-8-sig",
        "delimiter": "\t",
    }


def test_explicit_delimiter(tmp_path):
    path = _csv(
        tmp_path, f"L1;S1;G1;7;eng;{ONES.replace(',', ';')}", header=HEADER.replace(",", ";")
    )
    survey = read_survey(path, _mapping(), read_options=ReadOptions(delimiter=";"))
    assert survey.n() == 1


def test_bad_read_options_raise_settings_error():
    with pytest.raises(SettingsError, match="delimiter"):
        read_survey(KEY_CSV, _mapping(), read_options={"delimiter": ";;"})
    with pytest.raises(SettingsError, match="format"):
        read_survey(KEY_CSV, _mapping(), read_options={"format": "xlsx"})


def test_read_options_not_a_mapping_raises_settings_error():
    with pytest.raises(
        SettingsError, match="read_options must be a ReadOptions or a mapping, got str"
    ):
        read_survey(KEY_CSV, _mapping(), read_options="csv")  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime SettingsError


def test_survey_constructor_deep_copies_provenance():
    survey = read_survey(KEY_CSV, _mapping())
    provenance = {**survey.provenance, "id_columns": {"location": "LocationID"}}
    rebuilt = dataclasses.replace(survey, provenance=provenance)
    provenance["id_columns"]["location"] = "X"
    assert rebuilt.provenance["id_columns"] == {"location": "LocationID"}


def test_unknown_encoding_raises_settings_error():
    with pytest.raises(SettingsError, match="unknown encoding 'latin-99'"):
        read_survey(KEY_CSV, _mapping(), read_options={"encoding": "latin-99"})


def test_bad_mapping_type_raises():
    with pytest.raises(TypeError, match="SurveyMapping or a path"):
        read_survey(KEY_CSV, {"ids": {}})  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime TypeError


def test_survey_rejects_length_mismatch():
    survey = read_survey(KEY_CSV, _mapping())
    with pytest.raises(ValueError, match="differ in length"):
        dataclasses.replace(survey, response_id=survey.response_id[:2])


def test_frame_source_raises_type_error():
    frame = pd.read_csv(KEY_CSV, dtype=str)
    with pytest.raises(TypeError, match="file path"):
        read_survey(frame, _mapping())  # ty: ignore[invalid-argument-type]  # deliberate: exercises the S1 path-only TypeError


def test_mapping_path(tmp_path):
    path = tmp_path / "mapping.toml"
    path.write_text(tomlkit.dumps(_raw(KEY_IDS)), encoding="utf-8")
    survey = read_survey(KEY_CSV, path)
    assert survey.mapping == _mapping(KEY_IDS)


def test_mapping_path_validation_error_is_mapping_error(tmp_path):
    raw = _raw()
    raw["groups"]["attributes"]["columns"]["PAQ1"] = "other"  # pleasant and PAQ1 both map PAQ1
    path = tmp_path / "mapping.toml"
    path.write_text(tomlkit.dumps(raw), encoding="utf-8")
    with pytest.raises(
        MappingError,
        match=re.escape(
            "groups.attributes.columns: attribute keys 'pleasant' and 'PAQ1' both map PAQ1"
        ),
    ) as info:
        read_survey(KEY_CSV, path)
    assert str(path) in str(info.value)


def test_mapping_path_field_validator_keeps_field_path(tmp_path):
    raw = _raw({"response": "ID", "response_key": list(KEY)})
    path = tmp_path / "mapping.toml"
    path.write_text(tomlkit.dumps(raw), encoding="utf-8")
    with pytest.raises(MappingError, match="not both") as info:
        read_survey(KEY_CSV, path)
    assert "\nids\n" in str(info.value)  # pydantic's field path line


# --- scale check and status --------------------------------------------------------------------


def test_scale_violation_raises():
    with pytest.raises(ScaleViolationError) as info:
        read_survey(SCALE_0_4_CSV, _mapping({"response": "ID"}))
    message = str(info.value)
    assert "'pleasant' (PAQ1): 0 x2 (responses a, c)" in message
    assert "'calm' (PAQ8): 0 x1 (responses b)" in message
    # Row c's blank annoying cell is missing, not a violation; annoying's 0 is in row a only.
    assert "'annoying' (PAQ5): 0 x1 (responses a)" in message
    assert "'vibrant'" not in message
    hint = info.value.hint
    assert "missing" in hint
    assert "offset" in hint
    assert "on_scale_violation" not in hint


def test_incomplete_attributes_excluded(tmp_path):
    path = _csv(tmp_path, f"a,{ONES}", f"b,{ONES[:-1]}", header=ID_HEADER)
    survey = read_survey(path, _mapping({"response": "ID"}))
    assert survey.status.included.tolist() == [True, False]
    assert survey.status.reasons[1] == (Reason.INCOMPLETE_ATTRIBUTES,)
    assert np.isnan(survey.scores["attributes"][1, 5])  # monotonous is PAQ6


def test_info_summary_logged(caplog, tmp_path):
    path = _csv(tmp_path, f"a,{ONES}", ",,,,,,,,", f"b,{ONES[:-1]}", header=ID_HEADER)
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        read_survey(path, _mapping({"response": "ID"}))
    records = [r for r in caplog.records if r.name == "soundscapy"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert "2 rows" in message
    assert "1 blank row skipped" in message
    assert "1 included" in message
    assert "incomplete_attributes: 1" in message


def test_survey_arrays_read_only():
    survey = read_survey(KEY_CSV, _mapping(KEY_IDS, {"value": "eng"}))
    arrays = [
        survey.scores["attributes"],
        survey.response_id,
        survey.language,
        survey.status.included,
        *survey.ids.values(),
    ]
    for array in arrays:
        with pytest.raises(ValueError, match="read-only"):
            array[0] = array[1]


# --- provenance --------------------------------------------------------------------------------


def _expected_keys() -> set[str]:
    base = envelope(operation="x", settings=ImportSettings(), input_hash="")
    extra = {"response_id_source", "read", "id_columns", "response_id_columns"}
    return set(base) | extra | {"response_key_blank_ok"}


@pytest.mark.parametrize(
    ("path", "ids", "source", "id_columns", "response_id_columns"),
    [
        (
            KEY_CSV,
            KEY_IDS,
            "key",
            {"location": "LocationID", "session": "SessionID", "group": "GroupID"}
            | {"record": "RecordID"},
            list(KEY),
        ),
        (COLUMN_CSV, {"response": "ID"}, "column", {}, ["ID"]),
        (ROW_CSV, {}, "row", {}, []),
    ],
)
def test_provenance_keys(path, ids, source, id_columns, response_id_columns):
    survey = read_survey(path, _mapping(ids))
    prov = survey.provenance
    assert set(prov) == _expected_keys()
    assert prov["operation"] == "read_survey"
    assert prov["input_hash"] == hash_file(path)
    assert prov["input_hash"] == survey.source.hash
    assert prov["response_id_source"] == source
    assert prov["id_columns"] == id_columns
    assert prov["response_id_columns"] == response_id_columns
    assert prov["response_key_blank_ok"] == []
    assert prov["read"] == {"format": "csv", "encoding": "utf-8-sig", "delimiter": ","}
    assert prov["data_refs"] == [
        {
            "kind": "circumplex",
            "name": "iso-12913-outdoor",
            "version": "1",
            "hash": _data.PINS["circumplexes/iso-12913-outdoor/1.toml"],
        }
    ]
