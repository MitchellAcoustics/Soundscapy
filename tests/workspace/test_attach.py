"""attach: the projected coordinates and status columns appended to the user's own frame."""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from soundscapy._frames import column_text, to_eager
from soundscapy._model import (
    COL_ANGLE_SET,
    COL_FLAGS,
    COL_INCLUDED,
    COL_LANGUAGE,
    COL_REASONS,
)
from soundscapy.datasets import isd
from soundscapy.errors import AlignmentError, ColumnCollisionError
from soundscapy.projection.project import project
from soundscapy.survey.mapping import SurveyMapping
from soundscapy.survey.read import read_survey
from soundscapy.workspace import attach
from tests._helpers import LABELS, ONES, raw_mapping, write_csv

SUBSET_CSV = Path(isd.__file__).parent.parent / "_data" / "datasets" / "isd" / "subset-v1.0.csv"
FIVES = ",".join(["5"] * 8)
X, Y = "ssp_ISOPleasant", "ssp_ISOEventful"
ADDED = (X, Y, COL_INCLUDED, COL_REASONS, COL_FLAGS, COL_LANGUAGE, COL_ANGLE_SET)


def _mapping(ids: dict) -> SurveyMapping:
    return SurveyMapping.model_validate(raw_mapping(ids))


def _attach_log(caplog: pytest.LogCaptureFixture, level: int = logging.INFO) -> str:
    records = [r for r in caplog.records if r.name == "soundscapy" and "attach" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == level
    return records[0].getMessage()


def _null(native: object, column: str) -> np.ndarray:
    if isinstance(native, pd.DataFrame):
        return native[column].isna().to_numpy()
    assert isinstance(native, pl.DataFrame)
    return native[column].is_null().to_numpy()


# --- the subset, by id -------------------------------------------------------------------------


def test_attach_to_survey_data(isd_survey, isd_coords):
    survey, coords = isd_survey, isd_coords
    before = survey.data.copy()
    out = attach(survey.data, coords)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1552, 40 + 7)
    assert list(out.columns[40:]) == list(ADDED)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[:, 0])
    np.testing.assert_array_equal(out[Y].to_numpy(), coords.xy[:, 1])
    assert np.isnan(out[X].to_numpy()[~coords.labels[COL_INCLUDED]]).all()
    assert out[COL_INCLUDED].tolist() == coords.labels[COL_INCLUDED].tolist()
    assert out[COL_REASONS].tolist() == coords.labels[COL_REASONS].tolist()
    assert out[COL_LANGUAGE].tolist() == coords.labels[COL_LANGUAGE].tolist()
    assert out[COL_ANGLE_SET].tolist() == coords.labels[COL_ANGLE_SET].tolist()
    pd.testing.assert_frame_equal(survey.data, before)


def test_attach_to_reread_frame_with_int_ids(isd_coords):
    coords = isd_coords
    frame = pd.read_csv(SUBSET_CSV).iloc[::-1].reset_index(drop=True)
    assert frame["RecordID"].dtype == np.int64
    # An int64 id reads as the CSV cell's text ("22"), not "22.0".
    assert column_text(to_eager(frame), "RecordID")[-1] == "22"
    out = attach(frame, coords)
    assert out[COL_INCLUDED].notna().all()
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[::-1, 0])
    assert out[COL_REASONS].tolist() == coords.labels[COL_REASONS][::-1].tolist()


def test_attach_polars(isd_survey, isd_coords, frame_lib):
    survey, coords = isd_survey, isd_coords
    frame = frame_lib({name: survey.data[name].tolist() for name in survey.data.columns})
    out = attach(frame, coords)
    assert type(out) is type(frame)
    assert out.shape == (1552, 47)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(np.asarray(out[X].to_numpy(), dtype=float), coords.xy[:, 0])
    assert out[COL_INCLUDED].to_list() == coords.labels[COL_INCLUDED].tolist()


def test_attach_partial_frame(isd_survey, isd_coords, caplog):
    survey, coords = isd_survey, isd_coords
    half = survey.data.iloc[::2]
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        out = attach(half, coords)
    assert out[COL_INCLUDED].notna().all()
    assert list(out.index) == list(half.index)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[::2, 0])
    message = _attach_log(caplog, logging.WARNING)
    assert "id" in message
    assert "776 of 776" in message
    assert "776 survey responses matched no frame row" in message


def test_attach_blank_id_part_is_unmatched(isd_survey, isd_coords, caplog):
    survey, coords = isd_survey, isd_coords
    frame = survey.data.copy()
    frame.loc[3, "RecordID"] = "  "
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        out = attach(frame, coords)
    assert out.loc[3, list(ADDED)].isna().all()
    assert out.drop(index=3)[COL_INCLUDED].notna().all()
    message = _attach_log(caplog, logging.WARNING)
    assert "1551 of 1552" in message
    assert "1 survey response matched no frame row" in message


def test_attach_no_unmatched_survey_rows_logs_info(isd_survey, isd_coords, caplog):
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        attach(isd_survey.data, isd_coords)
    assert "1552 of 1552" in _attach_log(caplog, logging.INFO)


def test_attach_unmatched_survey_rows_warning_names_the_remedy(tmp_path, caplog):
    # A user re-reading the CSV with pd.read_csv turns "007" into 7; that is not
    # recoverable, so the warning names the cause and the remedy.
    header = ",".join(("ID", *LABELS))
    path = write_csv(tmp_path, header, f"007,{ONES}", f"8,{FIVES}")
    coords = project(read_survey(path, _mapping({"response": "ID"})))
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        out = attach(pd.read_csv(path), coords)
    assert out[COL_INCLUDED].isna().tolist() == [True, False]
    message = _attach_log(caplog, logging.WARNING)
    assert "1 survey response matched no frame row" in message
    assert "leading zeros" in message
    assert "pd.read_csv(path, dtype=str)" in message
    out = attach(pd.read_csv(path, dtype=str), coords)
    assert out[COL_INCLUDED].notna().all()


def test_attach_no_match_hint_names_the_remedy(tmp_path):
    header = ",".join(("ID", *LABELS))
    path = write_csv(tmp_path, header, f"007,{ONES}", f"008,{FIVES}")
    coords = project(read_survey(path, _mapping({"response": "ID"})))
    with pytest.raises(AlignmentError, match="no frame row") as info:
        attach(pd.read_csv(path), coords)
    assert "leading zeros" in info.value.hint
    assert "pd.read_csv(path, dtype=str)" in info.value.hint


def test_attach_never_overwrites(isd_survey, isd_coords):
    survey, coords = isd_survey, isd_coords
    frame = survey.data.assign(ssp_ISOPleasant=0.0)
    with pytest.raises(ColumnCollisionError, match="ssp_ISOPleasant"):
        attach(frame, coords)


def test_attach_missing_key_column_raises(isd_survey, isd_coords):
    survey, coords = isd_survey, isd_coords
    assert coords.provenance["response_id_columns"][2] == "GroupID"
    with pytest.raises(AlignmentError, match="'GroupID'"):
        attach(survey.data.drop(columns="GroupID"), coords)


def test_attach_no_match_raises(isd_survey, isd_coords):
    survey, coords = isd_survey, isd_coords
    frame = survey.data.assign(LocationID="Nowhere")
    with pytest.raises(AlignmentError, match="no frame row"):
        attach(frame, coords)


def test_attach_duplicate_frame_ids_raise(isd_survey, isd_coords):
    survey, coords = isd_survey, isd_coords
    frame = pd.concat([survey.data, survey.data.iloc[:1]], ignore_index=True)
    with pytest.raises(AlignmentError, match="not unique"):
        attach(frame, coords)


# --- small surveys: each id source -------------------------------------------------------------


@pytest.fixture
def column_coords(tmp_path):
    header = ",".join(("ID", "site", *LABELS))
    path = write_csv(tmp_path, header, f"a,s1,{ONES}", f"b,s1,{FIVES}", f"c,s2,{ONES}")
    return project(read_survey(path, _mapping({"response": "ID", "location": "site"})))


def test_attach_column_source(tmp_path):
    header = ",".join(("ID", "site", *LABELS))
    path = write_csv(tmp_path, header, f"a,s1,{ONES}", f"b,s2,{FIVES}")
    # ID holds no id role, only the response id.
    coords = project(read_survey(path, _mapping({"response": "ID", "location": "site"})))
    assert coords.provenance["response_id_source"] == "column"
    out = attach(pd.DataFrame({"ID": [" b ", "a"]}), coords)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[::-1, 0])


def test_attach_frame_with_extra_rows(column_coords, frame_lib, caplog):
    frame = frame_lib({"ID": ["c", "zz", "a", "b"], "note": [1, 2, 3, 4]})
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        out = attach(frame, column_coords)
    for column in ADDED:
        assert _null(out, column).tolist() == [False, True, False, False], column
    assert "3 of 4" in _attach_log(caplog)


def test_attach_unmatched_rows_still_filter_on_ssp_included_in_pandas(column_coords):
    # The unmatched row is pd.NA in a nullable "boolean" column, so the usual filter works and
    # drops it; with an object column of NaN, pandas refuses to filter (review F3).
    frame = pd.DataFrame({"ID": ["c", "zz", "a", "b"]})
    out = attach(frame, column_coords)
    assert str(out[COL_INCLUDED].dtype) == "boolean"
    assert out[out[COL_INCLUDED]]["ID"].tolist() == ["c", "a", "b"]


def test_attach_pandas_ssp_included_is_nullable_boolean_when_every_row_matches(column_coords):
    # The dtype does not depend on whether any row was unmatched.
    out = attach(pd.DataFrame({"ID": ["a", "b", "c"]}), column_coords)
    assert str(out[COL_INCLUDED].dtype) == "boolean"


def test_attach_blank_response_cell_is_unmatched(column_coords):
    out = attach(pd.DataFrame({"ID": ["a", None, ""]}), column_coords)
    assert out[COL_INCLUDED].isna().tolist() == [False, True, True]


def test_attach_key_not_equal_to_roles(tmp_path):
    header = ",".join(("site", "resp_no", *LABELS))
    path = write_csv(tmp_path, header, f"s1,1,{ONES}", f"s1,2,{FIVES}", f"s2,1,{FIVES}")
    mapping = _mapping({"response_key": ["site", "resp_no"], "location": "site"})
    coords = project(read_survey(path, mapping))
    assert coords.provenance["response_id_columns"] == ["site", "resp_no"]
    frame = pd.DataFrame({"resp_no": [1, 1, 2], "site": ["s2", "s1", "s1"]})
    out = attach(frame, coords)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[[2, 0, 1], 0])


def test_attach_integer_ids_made_float_by_a_null(tmp_path):
    # pd.read_csv turns an integer column with a blank cell into float64 (1 -> 1.0);
    # the frame id must still read "1", as the survey's CSV cell does.
    header = ",".join(("site", "resp_no", *LABELS))
    path = write_csv(tmp_path, header, f"s1,1,{ONES}", f"s1,2,{FIVES}", f"s2,1,{FIVES}")
    coords = project(read_survey(path, _mapping({"response_key": ["site", "resp_no"]})))
    frame = pd.DataFrame({"site": ["s2", "s1", "s1", "s1"], "resp_no": [1, 1, 2, None]})
    assert frame["resp_no"].dtype == np.float64
    out = attach(frame, coords)
    assert out[COL_INCLUDED].isna().tolist() == [False, False, False, True]
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy()[:3], coords.xy[[2, 0, 1], 0])


def test_attach_non_integral_float_ids_keep_their_text(tmp_path):
    header = ",".join(("ID", *LABELS))
    path = write_csv(tmp_path, header, f"1.5,{ONES}", f"2,{FIVES}")
    coords = project(read_survey(path, _mapping({"response": "ID"})))
    out = attach(pd.DataFrame({"ID": [2.0, 1.5, np.nan]}), coords)
    assert out[COL_INCLUDED].isna().tolist() == [False, False, True]


def test_attach_blank_ok_key_part_matches(tmp_path):
    header = ",".join(("site", "resp_no", *LABELS))
    path = write_csv(tmp_path, header, f"s1,,{ONES}", f"s2,1,{FIVES}")
    mapping = _mapping({"response_key": ["site", "resp_no"], "response_key_blank_ok": ["resp_no"]})
    coords = project(read_survey(path, mapping))
    out = attach(pd.DataFrame({"site": ["s2", "s1"], "resp_no": ["1", None]}), coords)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[::-1, 0])


def test_attach_row_ids_by_position(tmp_path, caplog):
    path = write_csv(tmp_path, ",".join(LABELS), ONES, FIVES, ONES)
    coords = project(read_survey(path, _mapping({})))
    assert coords.provenance["response_id_source"] == "row"
    with caplog.at_level(logging.INFO, logger="soundscapy"):
        out = attach(pd.DataFrame({"note": ["x", "y", "z"]}), coords)
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(out[X].to_numpy(), coords.xy[:, 0])
    message = _attach_log(caplog)
    assert "position" in message
    assert "3 of 3" in message
    with pytest.raises(AlignmentError, match=re.escape("2 rows; the survey has 3")):
        attach(pd.DataFrame({"note": ["x", "y"]}), coords)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"response_id_source": "hash"}, "no known response id source"),
        ({"response_id_columns": "ID"}, "not a list of column names"),
    ],
)
def test_attach_malformed_provenance_raises(column_coords, changes, match):
    broken = dataclasses.replace(column_coords, provenance={**column_coords.provenance, **changes})
    with pytest.raises(AlignmentError, match=match):
        attach(pd.DataFrame({"ID": ["a"]}), broken)
