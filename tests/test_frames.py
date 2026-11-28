"""Tests for the frame boundary, the CSV text reader and ``frame_hash``."""

from __future__ import annotations

import csv
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from soundscapy._frames import (
    TextTable,
    column_text,
    columns_float,
    float_column,
    frame_hash,
    new_frame,
    read_text_table,
    require_frame,
    with_columns,
)
from soundscapy.errors import (
    CodingError,
    ColumnCollisionError,
    EncodingError,
    FileFormatError,
    MappingError,
)
from tests._helpers import fake_frame

FRAME_VECTOR = "h1-sha256:0ef558dd8a05e2aa709ff3e49550a1d91eb4aeab621fde47153bfd3be5028ee8"


# require_frame


@pytest.mark.parametrize("module", ["polars.dataframe.frame", "pyarrow.lib"])
def test_require_frame_names_to_pandas_for_polars_and_arrow(module):
    with pytest.raises(TypeError) as excinfo:
        require_frame(fake_frame(module))
    assert str(excinfo.value) == (
        "only pandas DataFrames are accepted; convert with .to_pandas() first"
    )


@pytest.mark.parametrize("obj", [{"a": [1]}, [1, 2], pd.Series([1])])
def test_require_frame_refuses_non_frames(obj):
    with pytest.raises(TypeError, match=f"expected a pandas DataFrame, got {type(obj).__name__}"):
        require_frame(obj)


def test_require_frame_returns_the_frame_itself(frame_lib):
    frame = frame_lib({"a": [1.0]})
    assert require_frame(frame) is frame


def test_duplicate_column_names_raise_collision():
    with pytest.raises(ColumnCollisionError, match="'a'"):
        require_frame(pd.DataFrame([[1, 2]], columns=["a", "a"]))


# columns_float, float_column and column_text


def test_columns_float_copies_and_is_read_only(frame_lib):
    frame = frame_lib({"x": [1.0, None, float("nan")], "y": [1, 2, 3]})
    arr = columns_float(frame, ["x", "y"])
    assert arr.shape == (3, 2)
    assert arr.dtype == np.float64
    assert arr.flags.writeable is False
    # Exact: whole numbers and NaN placement involve no arithmetic.
    np.testing.assert_array_equal(arr, [[1.0, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    frame.loc[0, "x"] = 99
    assert arr[0, 0] == 1.0  # exact: the copy is untouched


@pytest.mark.parametrize(
    ("column", "values"), [("t", ["a", "b"]), ("b", [True, False]), ("o", ["1", None])]
)
def test_columns_float_refuses_text_and_bool(frame_lib, column, values):
    with pytest.raises(CodingError, match=f"'{column}'"):
        columns_float(frame_lib({column: values}), [column])


def test_missing_column_names_it(frame_lib):
    frame = frame_lib({"a": [1.0]})
    with pytest.raises(MappingError, match="'zz'"):
        columns_float(frame, ["zz"])
    with pytest.raises(MappingError, match="'zz'"):
        column_text(frame, "zz")
    with pytest.raises(MappingError, match="'zz'"):
        float_column(frame, "zz")


def test_float_column_reads_only_float_dtypes():
    frame = pd.DataFrame(
        {
            "f": [1.0, np.nan],
            "nf": pd.array([2.0, None], dtype="Float64"),
            "i": [1, 2],
            "ni": pd.array([1, None], dtype="Int64"),
            "b": [True, False],
            "t": ["1.0", "2.0"],
        }
    )
    floats = float_column(frame, "f")
    assert floats is not None
    assert floats.flags.writeable is False
    # Exact: the values are copied, not computed.
    np.testing.assert_array_equal(floats, [1.0, np.nan])
    np.testing.assert_array_equal(float_column(frame, "nf"), [2.0, np.nan])
    for name in ("i", "ni", "b", "t"):
        assert float_column(frame, name) is None, name


def test_column_text_nulls_are_empty(frame_lib):
    frame = frame_lib({"t": ["a", None, "é"], "f": [1.5, None, 2.0]})
    text = column_text(frame, "t")
    assert text.dtype == np.object_
    assert text.flags.writeable is False
    assert text.tolist() == ["a", "", "é"]
    assert column_text(frame, "f").tolist() == ["1.5", "", "2.0"]


# with_columns and new_frame


def test_with_columns_collision(frame_lib):
    frame = frame_lib({"a": [1.0, 2.0]})
    with pytest.raises(ColumnCollisionError, match="'a'"):
        with_columns(frame, {"a": np.array([3.0, 4.0])})


def test_with_columns_length_mismatch(frame_lib):
    frame = frame_lib({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="3 values"):
        with_columns(frame, {"b": np.array([3.0, 4.0, 5.0])})


def test_with_columns_returns_a_new_pandas_frame(frame_lib):
    frame = frame_lib({"a": [1.0, 2.0]})
    out = with_columns(frame, {"b": np.array([3.0, 4.0]), "c": np.array(["x", "y"], dtype=object)})
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b", "c"]
    assert out["c"].tolist() == ["x", "y"]
    assert frame.shape == (2, 1)  # the input frame is not changed


def test_with_columns_keeps_pandas_index():
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=[10, 20])
    out = with_columns(frame, {"b": np.array([3.0, 4.0])})
    assert out["b"].tolist() == [3.0, 4.0]
    assert out.index.tolist() == [10, 20]
    masked = with_columns(frame, {"b": np.array([3.0, 4.0])}, valid=np.array([False, True]))
    assert masked.index.tolist() == [10, 20]
    assert masked["b"].isna().tolist() == [True, False]


def test_with_columns_valid_mask_gives_nulls(frame_lib):
    # The added columns' dtypes do not depend on the dtypes of the user's own columns.
    frame = frame_lib({"a": [1.0, 2.0, 3.0]})
    columns = {
        "b": np.array([True, False, True]),
        "f": np.array([0.5, 1.5, 2.5]),
        "t": np.array(["x", "y", "z"], dtype=object),
        "i": np.array([1, 2, 3]),
    }
    out = with_columns(frame, columns, valid=np.array([True, False, True]))
    for name in columns:
        assert out[name].isna().tolist() == [False, True, False], name
    assert out["b"].tolist()[::2] == [True, True]
    assert out["f"].tolist()[::2] == [0.5, 2.5]
    assert out["t"].tolist()[::2] == ["x", "z"]
    assert out["i"].tolist()[::2] == [1, 3]


@pytest.mark.parametrize("n_rows", [0, 3])
def test_with_columns_all_masked_keeps_dtypes(frame_lib, n_rows):
    columns = {
        "f": np.arange(n_rows, dtype=np.float64),
        "b": np.ones(n_rows, dtype=bool),
        "t": np.array(["x"] * n_rows, dtype=object),
    }
    frame = frame_lib({"a": [1.0] * n_rows})
    populated = with_columns(frame, columns).dtypes
    masked = with_columns(frame, columns, valid=np.zeros(n_rows, dtype=bool))
    for name in columns:
        assert masked[name].isna().all()  # vacuously true with no rows
    assert masked.dtypes["f"] == populated["f"] == np.float64
    assert masked.dtypes["t"] == populated["t"]
    assert str(masked.dtypes["b"]) == str(populated["b"]) == "boolean"
    floats = columns_float(masked, ["f"])
    assert floats.shape == (n_rows, 1)
    assert np.isnan(floats).all()


def test_new_frame_builds_pandas():
    columns = {"x": np.array([1.0, 2.0]), "t": np.array(["a", "b"], dtype=object)}
    frame = new_frame(columns)
    assert isinstance(frame, pd.DataFrame)
    assert frame["x"].tolist() == [1.0, 2.0]
    assert frame["t"].tolist() == ["a", "b"]
    columns["x"][0] = 99.0
    assert frame["x"][0] == 1.0  # exact: the frame owns a copy


# frame_hash


def test_frame_hash_vector():
    content = {"i": [1, None, 3], "f": [0.5, 1.5, -2.0], "t": ["a", None, "é"]}
    classic = pd.DataFrame(content)
    nullable = pd.DataFrame(content).convert_dtypes()
    # The integer column has two representations, so equal hashes prove normalisation.
    assert str(classic.dtypes["i"]) == "float64"
    assert str(nullable.dtypes["i"]) == "Int64"
    assert frame_hash(classic) == FRAME_VECTOR
    assert frame_hash(nullable) == FRAME_VECTOR


def test_frame_hash_normalisation():
    ints = frame_hash(pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)}))
    assert ints == frame_hash(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert ints == frame_hash(pd.DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")}))
    assert ints != frame_hash(pd.DataFrame({"a": ["1", "2", "3"]}))

    no_null = frame_hash(pd.DataFrame({"b": [True, False, True]}))
    with_null = {"b": [True, None, True]}
    assert no_null != frame_hash(pd.DataFrame(with_null))

    naive = dt.datetime(2020, 1, 1, 12, 30)  # noqa: DTZ001  # deliberate: a naive datetime column
    assert frame_hash(pd.DataFrame({"d": [naive, None]})) == frame_hash(
        pd.DataFrame({"d": pd.Series([naive, None], dtype=object)})
    )

    other = {"day": [dt.date(2020, 1, 2)], "time": [dt.time(1, 2, 3)]}
    assert frame_hash(pd.DataFrame(other)) == frame_hash(
        pd.DataFrame({"day": ["2020-01-02"], "time": ["01:02:03"]})
    )


def test_frame_hash_categorical_matches_text():
    text = frame_hash(pd.DataFrame({"c": ["a", "b", None]}))
    assert text == frame_hash(pd.DataFrame({"c": pd.Categorical(["a", "b", None])}))


def test_frame_hash_refuses_non_frames():
    with pytest.raises(TypeError, match="to_pandas"):
        frame_hash(fake_frame("polars.dataframe.frame"))


def test_frame_hash_column_order_and_name_matter():
    base = frame_hash(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    assert base != frame_hash(pd.DataFrame({"b": [2.0], "a": [1.0]}))
    assert base != frame_hash(pd.DataFrame({"a": [1.0], "c": [2.0]}))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"x": ["a"], "y": ["b"]}, {"x": ["ab"], "y": [""]}),
        ({"x": ["a\x1fb", "c"]}, {"x": ["a", "b\x1fc"]}),
        ({"x": [None]}, {"x": ["\x00"]}),
    ],
)
def test_frame_hash_text_does_not_collide(left, right):
    left_frame = pd.DataFrame(left, dtype=object)
    right_frame = pd.DataFrame(right, dtype=object)
    assert frame_hash(left_frame) != frame_hash(right_frame)


# pandas dtype families: the hashes and nulls that soundscapy's own column rules must reproduce.
# Literal hashes were taken from the narwhals boundary before pandas became the only library.


def _hash(columns):
    return frame_hash(pd.DataFrame(columns))


def test_frame_hash_nullable_numbers_match_classic_floats():
    assert _hash({"i": pd.array([1, None, 3], dtype="Int64")}) == _hash({"i": [1.0, np.nan, 3.0]})
    assert _hash({"f": pd.array([0.5, None], dtype="Float64")}) == _hash({"f": [0.5, np.nan]})
    assert _hash({"u": np.array([1, 2], dtype=np.uint8)}) == _hash({"u": [1.0, 2.0]})


def test_frame_hash_nullable_boolean_is_bool_only_without_nulls():
    bools = _hash({"b": np.array([True, False])})
    assert _hash({"b": pd.array([True, False], dtype="boolean")}) == bools
    with_null = _hash({"b": pd.array([True, None], dtype="boolean")})
    assert with_null == _hash({"b": pd.Series([True, None], dtype=object)})
    assert with_null != _hash({"b": pd.array([True, False], dtype="boolean")})


def test_frame_hash_string_dtypes_match_object_text():
    text = _hash({"t": pd.Series(["a", None], dtype=object)})
    assert _hash({"t": pd.array(["a", None], dtype="string")}) == text
    assert _hash({"t": pd.Series(["a", None], dtype="str")}) == text


def test_frame_hash_datetime_truncates_to_microseconds():
    nanos = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 00:00:00.000000500"])})
    whole = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 00:00:00"])})
    assert str(nanos.dtypes["d"]) == "datetime64[ns]"
    assert frame_hash(nanos) == frame_hash(whole)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (
            {"td": pd.Series([pd.Timedelta("1s"), None])},
            "h1-sha256:5c5f0830905ab4ff610892f9d298d9f42fff1d1220fa64e78f816fb993655022",
        ),
        (
            {"p": pd.period_range("2020-01", periods=2, freq="M")},
            "h1-sha256:ef070d0672c733d2f9d18f5ff17a79debc5187775d702e430055e95f0f788a8f",
        ),
        (
            {0: [1.0]},
            "h1-sha256:de56d6626d80ba04d2551b2cc27da0781923c19e8a329a3c6a58b2b3cf562253",
        ),
    ],
    ids=["timedelta-as-text", "period-as-text", "integer-column-name"],
)
def test_frame_hash_pinned_text_and_name_cases(columns, expected):
    assert _hash(columns) == expected


def test_with_columns_masked_row_nulls_by_kind():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    columns = {
        "b": np.array([True, False, True]),
        "f": np.array([0.5, 1.5, 2.5]),
        "t": np.array(["x", "y", "z"], dtype=object),
        "i": np.array([1, 2, 3]),
    }
    out = with_columns(frame, columns, valid=np.array([True, False, True]))
    assert str(out["b"].dtype) == "boolean"
    assert out["b"][1] is pd.NA
    assert out["f"].dtype == np.float64
    assert np.isnan(out["f"][1])
    # A masked integer column becomes float64, the classic pandas way to hold a null int.
    assert out["i"].dtype == np.float64
    assert np.isnan(out["i"][1])
    # A text column takes pandas' default text dtype (object on pandas 2, str on pandas 3).
    assert out["t"].dtype == pd.Series(["x"]).dtype
    assert out["t"].isna().tolist() == [False, True, False]
    unmasked = with_columns(frame, columns)
    assert unmasked["i"].dtype == np.int64
    assert str(unmasked["b"].dtype) == "boolean"


# read_text_table


def test_read_text_table_numbers_records(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_bytes(b"\xef\xbb\xbfid,score\r\na,1\r\nb,2\r\n\r\n , \r\nc,3\r\n")
    table = read_text_table(path, encoding="utf-8-sig", delimiter=",")
    assert isinstance(table, TextTable)
    assert table.row_numbers.tolist() == [2, 3, 6]
    assert table.row_numbers.dtype == np.int64
    assert table.row_numbers.flags.writeable is False
    assert table.n_blank == 2
    assert table.delimiter == ","
    assert table.encoding == "utf-8-sig"
    assert list(table.frame.columns) == ["id", "score"]
    assert table.frame["score"].tolist() == ["1", "2", "3"]


def test_read_text_table_quoted_newline_is_one_record(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_bytes(b'id;note\na;"two\nlines"\nb;x\n')  # bytes: write_text gives CRLF on Windows
    table = read_text_table(path, encoding="utf-8", delimiter=";")
    assert table.row_numbers.tolist() == [2, 3]
    assert table.frame["note"].tolist() == ["two\nlines", "x"]


def test_read_text_table_ragged_rows(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("a,b,c\n1,2\n", encoding="utf-8")
    table = read_text_table(path, encoding="utf-8", delimiter=",")
    assert table.frame.iloc[0].tolist() == ["1", "2", ""]

    path = tmp_path / "long.csv"
    path.write_text("a,b\n1,2\n1,2,3\n", encoding="utf-8")
    with pytest.raises(FileFormatError, match="record 3"):
        read_text_table(path, encoding="utf-8", delimiter=",")


@pytest.fixture
def small_field_limit():
    """Lower csv.field_size_limit to 8 characters for one test, then restore it."""
    previous = csv.field_size_limit(8)
    yield
    csv.field_size_limit(previous)


@pytest.mark.usefixtures("small_field_limit")
def test_read_text_table_csv_error_is_file_format_error(tmp_path):
    path = tmp_path / "big.csv"
    path.write_text("a,b\n1,2\n3,123456789\n", encoding="utf-8")
    with pytest.raises(FileFormatError, match=r"record 3 of .*big\.csv.*field larger") as info:
        read_text_table(path, encoding="utf-8", delimiter=",")
    assert isinstance(info.value.__cause__, csv.Error)


def test_read_text_table_duplicate_header(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("a,b,a\n1,2,3\n", encoding="utf-8")
    with pytest.raises(FileFormatError, match="'a'"):
        read_text_table(path, encoding="utf-8", delimiter=",")


def test_read_text_table_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(FileFormatError, match="header"):
        read_text_table(path, encoding="utf-8", delimiter=",")


def test_read_text_table_header_only(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("a,b\n", encoding="utf-8")
    table = read_text_table(path, encoding="utf-8", delimiter=",")
    assert list(table.frame.columns) == ["a", "b"]
    assert len(table.frame) == 0
    assert table.row_numbers.tolist() == []


def test_read_text_table_bad_encoding(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("name\ncafé\n".encode("cp1252"))
    with pytest.raises(EncodingError) as excinfo:
        read_text_table(path, encoding="utf-8", delimiter=",")
    assert "cp1252" in excinfo.value.hint
    table = read_text_table(path, encoding="cp1252", delimiter=",")
    assert table.frame["name"].tolist() == ["café"]
