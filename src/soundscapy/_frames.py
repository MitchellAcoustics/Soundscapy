"""The frame boundary: the only module that touches pandas or polars.

Every frame enters through :func:`to_eager` and leaves through :func:`to_native`,
:func:`with_columns` or :func:`new_frame`. Library dispatch goes through
``narwhals.stable.v2``; this module imports pandas (for frames read from paths) and never
polars or pyarrow, so ``import soundscapy`` needs only the runtime dependencies.
"""

import csv
import datetime as dt
import os
import struct
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.stable.v2.typing import IntoDataFrame
from numpy.typing import ArrayLike, DTypeLike, NDArray

from soundscapy.errors import (
    CodingError,
    ColumnCollisionError,
    EncodingError,
    FileFormatError,
    MappingError,
)
from soundscapy.provenance import canonical_json, hash_bytes

__all__ = [
    "Frame",
    "TextTable",
    "column_text",
    "columns_float",
    "frame_hash",
    "new_frame",
    "read_text_table",
    "readonly_copy",
    "to_eager",
    "to_native",
    "with_columns",
]

Frame: TypeAlias = IntoDataFrame
"""A native eager data frame narwhals accepts, such as a pandas or polars ``DataFrame``."""

_LAZY_MESSAGE: Final = "lazy frames are not accepted; call .collect() first"
_QUIET_NAN: Final = np.uint64(0x7FF8000000000000)
_NULL_TEXT: Final = struct.pack("<Q", 0xFFFFFFFFFFFFFFFF)


def readonly_copy(values: ArrayLike, dtype: DTypeLike | None = None) -> NDArray[Any]:
    """Return a read-only copy of an array.

    Parameters
    ----------
    values : array-like
        The values to copy.
    dtype : numpy dtype, optional
        The dtype of the copy; that of ``values`` when omitted.

    Returns
    -------
    numpy.ndarray
        A new array that does not share memory with ``values``, with writing disabled.
    """
    out = np.array(values, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def to_eager(obj: object) -> nw.DataFrame[Any]:
    """Wrap a native eager data frame for use inside soundscapy.

    Parameters
    ----------
    obj : object
        A pandas or polars ``DataFrame`` (or a narwhals ``DataFrame``).

    Returns
    -------
    narwhals.stable.v2.DataFrame
        The wrapped frame; it shares data with ``obj``.

    Raises
    ------
    TypeError
        If ``obj`` is a lazy frame, a series, a dict or any other object.
    ColumnCollisionError
        If two columns of ``obj`` share a name.
    """
    try:
        # narwhals checks the type at runtime; the cast only selects the eager overload.
        return nw.from_native(cast("Frame", obj), eager_only=True)
    except nw.exceptions.DuplicateError as err:
        names = sorted({str(n) for n, k in Counter(getattr(obj, "columns", ())).items() if k > 1})
        msg = f"the frame has duplicate column names: {', '.join(map(repr, names))}"
        raise ColumnCollisionError(msg) from err
    except TypeError as err:
        # pass_through only classifies the refusal: a lazy frame comes back wrapped.
        if isinstance(nw.from_native(obj, pass_through=True), nw.LazyFrame):
            raise TypeError(_LAZY_MESSAGE) from err
        msg = f"expected a pandas or polars data frame, got {type(obj).__name__}"
        raise TypeError(msg) from err


def to_native(frame: nw.DataFrame[Any]) -> Any:  # noqa: ANN401  # the user's own frame type
    """Return the native frame behind a narwhals frame.

    Parameters
    ----------
    frame : narwhals.stable.v2.DataFrame
        A frame from :func:`to_eager`.

    Returns
    -------
    object
        The frame in the user's library.
    """
    return nw.to_native(frame)


def _column(frame: nw.DataFrame[Any], name: str) -> nw.Series[Any]:
    """Return a column, raising a backend-neutral error when it is missing."""
    if name not in frame.columns:
        msg = f"column {name!r} is not in the frame; it has {frame.columns}"
        raise MappingError(msg)
    return frame.get_column(name)


def _float_values(series: nw.Series[Any]) -> NDArray[np.float64]:
    """Return a numeric series as a fresh float64 array, null and NaN as NaN."""
    null = series.is_null().to_numpy()
    values = np.array(series.cast(nw.Float64).to_numpy(), dtype=np.float64, copy=True)
    values[null] = np.nan
    return values


def columns_float(frame: nw.DataFrame[Any], names: Sequence[str]) -> NDArray[np.float64]:
    """Read numeric columns as a float64 matrix.

    Parameters
    ----------
    frame : narwhals.stable.v2.DataFrame
        A frame from :func:`to_eager`.
    names : sequence of str
        The columns to read, in order.

    Returns
    -------
    numpy.ndarray
        A read-only copy of shape ``(n_rows, len(names))``; null and NaN are both NaN.

    Raises
    ------
    CodingError
        If a column is boolean, text or of any other non-numeric type; text must be coded.
    MappingError
        If a column is not in the frame.
    """
    out = np.empty((len(frame), len(names)), dtype=np.float64)
    for index, name in enumerate(names):
        series = _column(frame, name)
        if not series.dtype.is_numeric():
            msg = (
                f"column {name!r} holds {series.dtype} values, not numbers; "
                "text and boolean columns must be coded first"
            )
            raise CodingError(msg)
        out[:, index] = _float_values(series)
    out.setflags(write=False)
    return out


def _value_text(value: object) -> str:
    """Return the backend-independent text of one non-null value."""
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.replace(nanosecond=0).isoformat()
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return str(value)


def _texts(series: nw.Series[Any]) -> list[str | None]:
    """Return each value's text, with ``None`` for a null."""
    null = series.is_null().to_list()
    return [
        None if is_null else _value_text(value)
        for value, is_null in zip(series.to_list(), null, strict=True)
    ]


def column_text(frame: nw.DataFrame[Any], name: str) -> NDArray[np.object_]:
    """Read one column as text.

    Text comes from each value's Python form: ``str`` as is, datetimes, dates and times
    truncated to whole microseconds and written by ``isoformat()``, anything else by
    ``str()``. The result is the same for pandas and polars.

    Parameters
    ----------
    frame : narwhals.stable.v2.DataFrame
        A frame from :func:`to_eager`.
    name : str
        The column to read.

    Returns
    -------
    numpy.ndarray
        A read-only object array of Python ``str``, with nulls as ``""``.

    Raises
    ------
    MappingError
        If the column is not in the frame.
    """
    texts = _texts(_column(frame, name))
    out = np.empty(len(texts), dtype=object)
    out[:] = ["" if text is None else text for text in texts]
    out.setflags(write=False)
    return out


def _pandas_column(
    values: NDArray[Any], valid: NDArray[np.bool_] | None, index: pd.Index
) -> pd.Series:
    """Build a pandas column, null where ``valid`` is False."""
    if valid is None:
        return pd.Series(values, index=index, copy=True)
    invalid = ~valid
    if values.dtype.kind == "f":
        return pd.Series(np.where(invalid, np.nan, values), index=index)
    if values.dtype.kind == "b":
        return pd.Series(pd.array(values, dtype="boolean"), index=index).mask(invalid)
    if values.dtype.kind in "iu":
        return pd.Series(pd.array(values, dtype="Int64"), index=index).mask(invalid)
    objects = values.astype(object)
    objects[invalid] = None
    return pd.Series(objects, index=index, dtype=object)


def _nw_dtype(values: NDArray[Any]) -> Any:  # noqa: ANN401  # a narwhals dtype class
    """Return the narwhals dtype of an added column, so all-null columns keep their type."""
    kind = values.dtype.kind
    if kind == "f":
        return nw.Float64
    if kind == "b":
        return nw.Boolean
    if kind in "iu":
        return nw.Int64
    return nw.String


def _list_with_nulls(values: NDArray[Any], valid: NDArray[np.bool_] | None) -> list[Any]:
    """Return the values as a Python list with ``None`` where ``valid`` is False."""
    items = values.tolist()
    if valid is None:
        return items
    return [item if ok else None for item, ok in zip(items, valid.tolist(), strict=True)]


def with_columns(
    frame: object,
    columns: Mapping[str, NDArray[Any]],
    *,
    valid: NDArray[np.bool_] | None = None,
) -> Any:  # noqa: ANN401  # the user's own frame type
    """Return a copy of a native frame with columns added.

    Parameters
    ----------
    frame : object
        A pandas or polars ``DataFrame``; it is not changed.
    columns : mapping of str to numpy.ndarray
        The new columns, each with one value per row.
    valid : numpy.ndarray of bool, optional
        Rows where this is False are null in every added column: NaN in a pandas float
        column, ``pd.NA`` in a pandas ``boolean`` or ``Int64`` column, ``None`` in a pandas
        object column, and null in polars.

    Returns
    -------
    object
        A frame of the same library as ``frame``.

    Raises
    ------
    ColumnCollisionError
        If a new column's name is already in the frame.
    ValueError
        If a column or ``valid`` does not have one value per row.
    """
    eager = to_eager(frame)
    n_rows = len(eager)
    clashes = [name for name in columns if name in eager.columns]
    if clashes:
        msg = f"columns {clashes} are already in the frame; rename or drop them first"
        raise ColumnCollisionError(msg)
    sized = {f"column {name!r}": values for name, values in columns.items()}
    if valid is not None:
        sized["valid"] = valid
    for label, values in sized.items():
        if len(values) != n_rows:
            msg = f"{label} has {len(values)} values; the frame has {n_rows} rows"
            raise ValueError(msg)
    native = to_native(eager)
    if isinstance(native, pd.DataFrame):
        added = {name: _pandas_column(v, valid, native.index) for name, v in columns.items()}
        return native.assign(**added)
    backend = nw.get_native_namespace(eager)
    series = [
        nw.new_series(
            name, _list_with_nulls(values, valid), dtype=_nw_dtype(values), backend=backend
        )
        for name, values in columns.items()
    ]
    return to_native(eager.with_columns(*series))


def new_frame(columns: Mapping[str, NDArray[Any]], like: object | None = None) -> Any:  # noqa: ANN401  # pandas or the library of like
    """Build a frame from columns.

    Parameters
    ----------
    columns : mapping of str to numpy.ndarray
        The columns, each of the same length.
    like : object, optional
        A frame whose library the result uses; pandas when omitted.

    Returns
    -------
    object
        A new native frame; it owns copies of the arrays.
    """
    backend = pd if like is None else nw.get_native_namespace(to_eager(like))
    copied = {name: np.array(values, copy=True) for name, values in columns.items()}
    return to_native(nw.from_dict(copied, backend=backend))


@dataclass(frozen=True)
class TextTable:
    """A delimited text file read with every cell as text.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per non-blank record; every cell is a Python ``str``.
    row_numbers : numpy.ndarray of int64
        Each row's physical record number in the file, the header being record 1;
        read-only. Skipped blank records keep their numbers out of use.
    n_blank : int
        How many blank records were skipped.
    delimiter : str
        The delimiter the file was read with.
    encoding : str
        The encoding the file was decoded with.
    """

    frame: pd.DataFrame
    row_numbers: NDArray[np.int64]
    n_blank: int
    delimiter: str
    encoding: str


def read_text_table(path: str | os.PathLike[str], *, encoding: str, delimiter: str) -> TextTable:
    """Read a delimited text file with every cell as text.

    The first record is the header. A record whose cells are all empty after stripping is
    skipped and counted. A record with fewer cells than the header is padded with ``""``.

    Parameters
    ----------
    path : str or os.PathLike
        The file to read.
    encoding : str
        The text encoding, for example ``"utf-8-sig"`` or ``"cp1252"``.
    delimiter : str
        The single-character field delimiter.

    Returns
    -------
    TextTable
        The records with their record numbers.

    Raises
    ------
    EncodingError
        If the file cannot be decoded with ``encoding``.
    FileFormatError
        If the file has no header, the header repeats a name, or a record has more cells
        than the header.
    """
    try:
        with Path(path).open(encoding=encoding, newline="") as handle:
            records = list(csv.reader(handle, delimiter=delimiter))
    except UnicodeDecodeError as err:
        msg = f"{os.fspath(path)!r} cannot be decoded as {encoding}: {err.reason}"
        raise EncodingError(
            msg, hint='pass an explicit encoding, such as encoding="cp1252"'
        ) from err
    if not records or _is_blank(records[0]):
        msg = f"{os.fspath(path)!r} has no header record"
        raise FileFormatError(msg)
    header = records[0]
    repeated = sorted(name for name, count in Counter(header).items() if count > 1)
    if repeated:
        msg = f"the header of {os.fspath(path)!r} repeats {', '.join(map(repr, repeated))}"
        raise FileFormatError(msg)
    rows, numbers = [], []
    for number, record in enumerate(records[1:], start=2):
        if _is_blank(record):
            continue
        if len(record) > len(header):
            msg = (
                f"record {number} of {os.fspath(path)!r} has {len(record)} cells; "
                f"the header has {len(header)}"
            )
            raise FileFormatError(msg)
        rows.append(record + [""] * (len(header) - len(record)))
        numbers.append(number)
    return TextTable(
        frame=pd.DataFrame(rows, columns=header, dtype=object),
        row_numbers=readonly_copy(numbers, np.int64),
        n_blank=len(records) - 1 - len(rows),
        delimiter=delimiter,
        encoding=encoding,
    )


def _is_blank(record: Sequence[str]) -> bool:
    return all(not cell.strip() for cell in record)


def _hash_chunks(frame: nw.DataFrame[Any]) -> Iterator[bytes]:
    """Yield the normalised byte stream of a frame (header, then column by column)."""
    dtypes: list[str] = []
    bodies: list[Iterable[bytes]] = []
    for name in frame.columns:
        series = frame.get_column(name)
        if series.dtype.is_numeric():
            values = _float_values(series).astype("<f8")
            bits = values.view("<u8")
            bits[np.isnan(values)] = _QUIET_NAN
            dtypes.append("float64")
            bodies.append([bits.tobytes()])
        elif series.dtype == nw.Boolean and not series.is_null().any():
            dtypes.append("bool")
            bodies.append([series.to_numpy().astype(np.uint8).tobytes()])
        else:
            dtypes.append("string")
            bodies.append(_text_bytes(_texts(series)))
    header = {"columns": list(frame.columns), "dtypes": dtypes, "n_rows": len(frame)}
    yield canonical_json(header)
    yield b"\x00"
    for body in bodies:
        yield from body


def _text_bytes(texts: Iterable[str | None]) -> Iterator[bytes]:
    """Yield each text as a uint64 byte length and its UTF-8 bytes, a null as all ones."""
    for text in texts:
        if text is None:
            yield _NULL_TEXT
        else:
            data = text.encode("utf-8")
            yield struct.pack("<Q", len(data))
            yield data


def frame_hash(frame: object) -> str:
    """Hash a frame's content, independent of its library and integer representation.

    Numeric columns become float64 with null as NaN; boolean columns without nulls stay
    boolean; every other column becomes text from each value's Python form. The hashed
    stream is the canonical JSON of ``{"columns", "dtypes", "n_rows"}``, a 0x00 byte, then
    each column: float64 little-endian with every NaN as the quiet NaN
    ``0x7FF8000000000000``, booleans as ``uint8``, and text value by value as a
    little-endian ``uint64`` UTF-8 byte length followed by the bytes, a null as the
    ``uint64`` ``0xFFFFFFFFFFFFFFFF`` alone.

    Parameters
    ----------
    frame : object
        A pandas or polars ``DataFrame``.

    Returns
    -------
    str
        The ``h1`` hash.

    Raises
    ------
    TypeError
        If ``frame`` is not an eager data frame.
    """
    return hash_bytes(_hash_chunks(to_eager(frame)))
