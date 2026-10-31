"""Moving results in and out of the user's workspace: attaching them to the user's frames."""

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from soundscapy._frames import Frame, to_eager, with_columns
from soundscapy._log import logger
from soundscapy._model import (
    COL_ANGLE_SET,
    COL_FLAGS,
    COL_INCLUDED,
    COL_LANGUAGE,
    COL_REASONS,
    COL_RESPONSE_ID,
    axis_column,
)
from soundscapy.distribution.coordinates import Coordinates, n
from soundscapy.errors import AlignmentError, _plural, _shown
from soundscapy.survey.read import check_unique, join_id_parts, stripped_text

__all__ = ["attach"]

_LABELS: Final = (COL_INCLUDED, COL_REASONS, COL_FLAGS, COL_LANGUAGE, COL_ANGLE_SET)
_REREAD_REMEDY: Final = (
    "if the frame was re-read from the survey file, its ids may have lost their text "
    '(leading zeros, as when "007" reads as 7): re-read the file with every column as '
    "text, e.g. pd.read_csv(path, dtype=str)"
)


def attach(frame: Frame, result: Coordinates) -> Any:  # noqa: ANN401  # the user's own frame type
    """Append a result's ``ssp_`` columns to the user's own frame.

    Appends ``ssp_<axis id>`` for the x and the y axis (for example ``ssp_ISOPleasant``
    and ``ssp_ISOEventful``, dimensionless), then ``ssp_included``, ``ssp_reasons``,
    ``ssp_flags``, ``ssp_language`` and ``ssp_angle_set``. Rows are aligned by the rule
    that built the result's response ids, ``provenance["response_id_source"]``:

    - ``"column"`` or ``"key"``: each frame row's id is rebuilt from the columns in
      ``provenance["response_id_columns"]``, in that order, as the survey was read: each
      cell's text, stripped, the parts joined with ``/``. A float cell holding a whole
      number reads as the integer's text (``1.0`` as ``"1"``), since pandas stores an
      integer id column with a null as float. The frame's row order is kept. A row whose
      id is not in the result, or whose id has a blank part other than in a column
      listed in ``provenance["response_key_blank_ok"]``, gets nulls in every added
      column.
    - ``"row"``: rows are aligned by position.

    One line on the ``soundscapy`` logger records the alignment (``"id"`` or
    ``"position"``) and the number of matched rows. It is an INFO line when every survey
    response found a frame row, and otherwise a WARNING that gives the number of
    responses left out and names a common cause: a frame re-read from the survey file
    with inferred types loses ids' text, such as leading zeros (``"007"`` reads as 7),
    which cannot be recovered; re-read it with every column as text, for example
    ``pd.read_csv(path, dtype=str)``. Attaching to ``survey.data`` always aligns.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        The user's frame; it is not changed.
    result : Coordinates
        Projected responses from a survey.

    Returns
    -------
    object
        A new frame of the same library as ``frame``, with the added columns last.

    Raises
    ------
    AlignmentError
        If an id column is missing from the frame, the frame's ids are not unique, no
        frame row matches (the hint names the re-reading remedy above), the frame's row
        count differs from the survey's when aligning by position, or the result records
        no known id source.
    ColumnCollisionError
        If an added column's name is already in the frame; ``attach`` never overwrites.
    """
    eager = to_eager(frame)
    n_rows = len(eager)
    x_axis, y_axis = result.circumplex.axes
    columns: dict[str, NDArray[Any]] = {
        axis_column(x_axis.id): result.xy[:, 0],
        axis_column(y_axis.id): result.xy[:, 1],
        **{name: result.labels[name] for name in _LABELS},
    }
    source = result.provenance.get("response_id_source")
    if source == "row":
        if n_rows != n(result):
            msg = "aligning by position needs the survey's row count: the frame has "
            msg += f"{n_rows} {_plural(n_rows, 'row')}; the survey has {n(result)}"
            raise AlignmentError(msg, hint="attach to survey.data, which always aligns")
        alignment, index, matched = "position", np.arange(n_rows), np.ones(n_rows, dtype=bool)
    elif source in ("column", "key"):
        alignment = "id"
        index, matched = _match_ids(eager, result)
    else:
        msg = f"the result records no known response id source: {source!r}"
        raise AlignmentError(msg)
    gathered = {name: values[index] for name, values in columns.items()}
    out = with_columns(frame, gathered, valid=matched)
    n_matched = int(matched.sum())
    message = f"attach: aligned by {alignment}; {n_matched} of {n_rows} frame rows matched"
    # Frame ids are unique, so each matched frame row is a distinct survey response.
    left_out = n(result) - n_matched
    if left_out:
        message += f"; {left_out} survey {_plural(left_out, 'response')} matched no frame row"
        logger.warning("%s; %s", message, _REREAD_REMEDY)
    else:
        logger.info("%s", message)
    return out


def _str_list(result: Coordinates, key: str) -> list[str]:
    """Return a provenance entry that must be a list of column names."""
    value = result.provenance.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"the result's provenance[{key!r}] is not a list of column names: {value!r}"
        raise AlignmentError(msg)
    return list(value)


def _match_ids(eager: Any, result: Coordinates) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:  # noqa: ANN401  # a narwhals frame
    """Return, per frame row, its row in the result (0 if none) and whether it matched."""
    id_columns = _str_list(result, "response_id_columns")
    blank_ok = set(_str_list(result, "response_key_blank_ok"))
    missing = [column for column in id_columns if column not in eager.columns]
    if missing or not id_columns:
        msg = f"the frame lacks the response id columns {missing or id_columns}; "
        msg += f"the result's ids were built from {id_columns}"
        raise AlignmentError(msg, hint="attach to survey.data, which always aligns")
    n_rows = len(eager)
    parts = {column: _id_text(eager, column) for column in id_columns}
    ids, blank = join_id_parts(parts, blank_ok)
    usable = ~blank
    rows = np.flatnonzero(usable).tolist()
    check_unique(ids[usable].tolist(), rows, error=AlignmentError, describe=_describe_repeats)
    position = {rid: i for i, rid in enumerate(result.labels[COL_RESPONSE_ID].tolist())}
    index = np.zeros(n_rows, dtype=np.intp)
    matched = np.zeros(n_rows, dtype=bool)
    for row in rows:
        hit = position.get(ids[row])
        if hit is not None:
            index[row], matched[row] = hit, True
    if not matched.any():
        msg = f"no frame row matches a response id of the result, built from {id_columns}"
        hint = f"check the id columns hold the survey's values; {_REREAD_REMEDY}"
        raise AlignmentError(msg, hint=hint)
    return index, matched


def _id_text(eager: Any, column: str) -> NDArray[np.object_]:  # noqa: ANN401  # a narwhals frame
    """Read a frame's id column as :func:`stripped_text` does, whole floats as integers.

    pandas stores an integer column that holds a null as float64, so ``1`` comes back as
    ``1.0``; the survey's CSV cell read ``"1"``.
    """
    series = eager.get_column(column)
    if not series.dtype.is_float():
        return stripped_text(eager, column)
    null = series.is_null().to_list()
    out = np.empty(len(null), dtype=object)
    out[:] = [
        "" if is_null else str(int(value)) if value.is_integer() else str(value)
        for value, is_null in zip(series.to_list(), null, strict=True)
    ]
    return out


def _describe_repeats(repeated: list[tuple[str, list[int]]]) -> str:
    shown = _shown([repr(rid) for rid in sorted(rid for rid, _ in repeated)])
    return f"the frame's response ids are not unique: {shown} repeat"
