"""Tests for the frame boundary, the CSV text reader and ``frame_hash``."""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
import sys

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from soundscapy._frames import (
    TextTable,
    column_text,
    columns_float,
    frame_hash,
    new_frame,
    read_text_table,
    to_eager,
    to_native,
    with_columns,
)
from soundscapy.errors import (
    CodingError,
    ColumnCollisionError,
    EncodingError,
    FileFormatError,
    MappingError,
)

FRAME_VECTOR = "h1-sha256:0ef558dd8a05e2aa709ff3e49550a1d91eb4aeab621fde47153bfd3be5028ee8"


# to_eager


def test_lazy_frame_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
        to_eager(pl.LazyFrame({"a": [1]}))
    assert str(excinfo.value) == "lazy frames are not accepted; call .collect() first"


@pytest.mark.parametrize("obj", [{"a": [1]}, pd.Series([1]), pl.Series([1])])
def test_non_frames_raise_type_error(obj):
    with pytest.raises(TypeError, match="data frame"):
        to_eager(obj)


def test_to_eager_and_to_native_round_trip(frame_lib):
    native = frame_lib({"a": [1.0]})
    assert to_native(to_eager(native)) is native


def test_frames_works_without_polars():
    code = (
        "import sys; sys.modules['polars'] = None; sys.modules['pyarrow'] = None; "
        "import soundscapy._frames"
    )
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_duplicate_column_names_raise_collision():
    with pytest.raises(ColumnCollisionError, match="'a'"):
        to_eager(pd.DataFrame([[1, 2]], columns=["a", "a"]))


# columns_float and column_text


def test_columns_float_copies_and_is_read_only(frame_lib):
    native = frame_lib({"x": [1.0, None, float("nan")], "y": [1, 2, 3]})
    arr = columns_float(to_eager(native), ["x", "y"])
    assert arr.shape == (3, 2)
    assert arr.dtype == np.float64
    assert arr.flags.writeable is False
    # Exact: whole numbers and NaN placement involve no arithmetic.
    np.testing.assert_array_equal(arr, [[1.0, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    if isinstance(native, pd.DataFrame):
        native.loc[0, "x"] = 99.0
    else:
        native[0, "x"] = 99.0
    assert arr[0, 0] == 1.0  # exact: the copy is untouched


@pytest.mark.parametrize(
    ("column", "values"), [("t", ["a", "b"]), ("b", [True, False]), ("o", ["1", None])]
)
def test_columns_float_refuses_text_and_bool(frame_lib, column, values):
    frame = to_eager(frame_lib({column: values}))
    with pytest.raises(CodingError, match=f"'{column}'"):
        columns_float(frame, [column])


def test_missing_column_names_it(frame_lib):
    frame = to_eager(frame_lib({"a": [1.0]}))
    with pytest.raises(MappingError, match="'zz'"):
        columns_float(frame, ["zz"])
    with pytest.raises(MappingError, match="'zz'"):
        column_text(frame, "zz")


def test_column_text_nulls_are_empty(frame_lib):
    frame = to_eager(frame_lib({"t": ["a", None, "é"], "f": [1.5, None, 2.0]}))
    text = column_text(frame, "t")
    assert text.dtype == np.object_
    assert text.flags.writeable is False
    assert text.tolist() == ["a", "", "é"]
    assert column_text(frame, "f").tolist() == ["1.5", "", "2.0"]


# with_columns and new_frame


def test_with_columns_collision(frame_lib):
    native = frame_lib({"a": [1.0, 2.0]})
    with pytest.raises(ColumnCollisionError, match="'a'"):
        with_columns(native, {"a": np.array([3.0, 4.0])})


def test_with_columns_length_mismatch(frame_lib):
    native = frame_lib({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="3 values"):
        with_columns(native, {"b": np.array([3.0, 4.0, 5.0])})


def test_with_columns_adds_in_same_library(frame_lib):
    native = frame_lib({"a": [1.0, 2.0]})
    out = with_columns(native, {"b": np.array([3.0, 4.0]), "c": np.array(["x", "y"], dtype=object)})
    assert type(out) is type(native)
    frame = to_eager(out)
    assert frame.columns == ["a", "b", "c"]
    assert frame["c"].to_list() == ["x", "y"]
    assert native.shape == (2, 1)  # the input frame is not changed


def test_with_columns_keeps_pandas_index():
    native = pd.DataFrame({"a": [1.0, 2.0]}, index=[10, 20])
    out = with_columns(native, {"b": np.array([3.0, 4.0])})
    assert out["b"].tolist() == [3.0, 4.0]
    assert out.index.tolist() == [10, 20]


def test_with_columns_valid_mask_gives_nulls(frame_lib):
    native = frame_lib({"a": [1.0, 2.0, 3.0]})
    columns = {
        "b": np.array([True, False, True]),
        "f": np.array([0.5, 1.5, 2.5]),
        "t": np.array(["x", "y", "z"], dtype=object),
        "i": np.array([1, 2, 3]),
    }
    valid = np.array([True, False, True])
    frame = to_eager(with_columns(native, columns, valid=valid))
    for name in columns:
        assert frame[name].is_null().to_list() == [False, True, False], name
    assert frame["b"].to_list()[::2] == [True, True]
    assert frame["f"].to_list()[::2] == [0.5, 2.5]
    assert frame["t"].to_list()[::2] == ["x", "z"]
    assert frame["i"].to_list()[::2] == [1, 3]
    # Nulls follow the library's own model, as narwhals presents it (review F3), except that a
    # pandas boolean column becomes the nullable "boolean" dtype (pd.NA), so it still filters.
    assert frame.schema["b"] == nw.Boolean
    if isinstance(native, pd.DataFrame):
        assert str(frame.to_native()["b"].dtype) == "boolean"
        assert frame.schema["i"] == nw.Float64  # pandas' classic dtypes hold a null int as float
    else:
        assert frame.schema["i"] == nw.Int64


def test_new_series_with_dtype_turns_a_pandas_null_boolean_false():
    # The trap with_columns avoids by masking after building: narwhals' pandas new_series
    # with a Boolean dtype silently turns None into False. If narwhals changes this, the
    # masking in with_columns can be revisited.
    series = nw.new_series("b", [True, None], dtype=nw.Boolean, backend="pandas")
    assert series.to_list() == [True, False]


@pytest.mark.parametrize("n_rows", [0, 3])
def test_with_columns_all_masked_keeps_dtypes(frame_lib, n_rows):
    columns = {
        "f": np.arange(n_rows, dtype=np.float64),
        "b": np.ones(n_rows, dtype=bool),
        "t": np.array(["x"] * n_rows, dtype=object),
    }
    native = frame_lib({"a": [1.0] * n_rows})
    populated = to_eager(with_columns(native, columns)).schema
    masked = to_eager(with_columns(native, columns, valid=np.zeros(n_rows, dtype=bool)))
    for name in columns:
        assert masked[name].is_null().all()  # vacuously true with no rows
    assert masked.schema["f"] == populated["f"] == nw.Float64
    assert masked.schema["t"] == populated["t"]
    assert masked.schema["b"] == populated["b"] == nw.Boolean
    floats = columns_float(masked, ["f"])
    assert floats.shape == (n_rows, 1)
    assert np.isnan(floats).all()


def test_new_frame_follows_like():
    columns = {"x": np.array([1.0, 2.0]), "t": np.array(["a", "b"], dtype=object)}
    default = new_frame(columns)
    assert isinstance(default, pd.DataFrame)
    assert default["x"].tolist() == [1.0, 2.0]
    polars = new_frame(columns, like=pl.DataFrame({"z": [0]}))
    assert isinstance(polars, pl.DataFrame)
    assert polars["t"].to_list() == ["a", "b"]


# frame_hash


def test_frame_hash_vector():
    content = {"i": [1, None, 3], "f": [0.5, 1.5, -2.0], "t": ["a", None, "é"]}
    pandas_frame = pd.DataFrame(content)
    polars_frame = pl.DataFrame(content)
    # The integer column has two representations, so equal hashes prove normalisation.
    assert str(pandas_frame.dtypes["i"]) == "float64"
    assert polars_frame.schema["i"] == pl.Int64
    assert frame_hash(pandas_frame) == FRAME_VECTOR
    assert frame_hash(polars_frame) == FRAME_VECTOR


def test_frame_hash_normalisation():
    ints = frame_hash(pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)}))
    assert ints == frame_hash(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert ints == frame_hash(pl.DataFrame({"a": [1, 2, 3]}))
    assert ints != frame_hash(pd.DataFrame({"a": ["1", "2", "3"]}))

    no_null = frame_hash(pd.DataFrame({"b": [True, False, True]}))
    with_null = {"b": [True, None, True]}
    assert no_null != frame_hash(pd.DataFrame(with_null))
    assert frame_hash(pd.DataFrame(with_null)) == frame_hash(pl.DataFrame(with_null))
    assert no_null == frame_hash(pl.DataFrame({"b": [True, False, True]}))

    naive = dt.datetime(2020, 1, 1, 12, 30)  # noqa: DTZ001  # deliberate: a naive datetime column
    dates = {"d": [naive, None]}
    assert frame_hash(pd.DataFrame(dates)) == frame_hash(pl.DataFrame(dates))

    nanos = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 00:00:00.000000500"])})
    assert str(nanos.dtypes["d"]) == "datetime64[ns]"
    nanos_pl = pl.from_pandas(nanos)
    assert nanos_pl.schema["d"] == pl.Datetime("ns")
    assert frame_hash(nanos) == frame_hash(nanos_pl)

    other = {"day": [dt.date(2020, 1, 2)], "time": [dt.time(1, 2, 3)]}
    assert frame_hash(pd.DataFrame(other)) == frame_hash(pl.DataFrame(other))


def test_frame_hash_categorical_matches_text():
    text = frame_hash(pd.DataFrame({"c": ["a", "b", None]}))
    assert text == frame_hash(pd.DataFrame({"c": pd.Categorical(["a", "b", None])}))
    assert text == frame_hash(pl.DataFrame({"c": ["a", "b", None]}, schema={"c": pl.Categorical}))


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
    path.write_text('id;note\na;"two\nlines"\nb;x\n', encoding="utf-8")
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
