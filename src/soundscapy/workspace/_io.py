"""Moving results in and out of the user's workspace: attaching them to the user's frames."""

from collections import Counter
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
from soundscapy.survey.read import join_id_parts, stripped_text

__all__ = ["attach"]

_LABELS: Final = (COL_INCLUDED, COL_REASONS, COL_FLAGS, COL_LANGUAGE, COL_ANGLE_SET)


def attach(frame: Frame, result: Coordinates) -> Any:  # noqa: ANN401  # the user's own frame type
    """Append a result's ``ssp_`` columns to the user's own frame.

    Appends ``ssp_<axis id>`` for the x and the y axis (for example ``ssp_ISOPleasant``
    and ``ssp_ISOEventful``, dimensionless), then ``ssp_included``, ``ssp_reasons``,
    ``ssp_flags``, ``ssp_language`` and ``ssp_angle_set``. Rows are aligned by the rule
    that built the result's response ids, ``provenance["response_id_source"]``:

    - ``"column"`` or ``"key"``: each frame row's id is rebuilt from the columns in
      ``provenance["response_id_columns"]``, in that order, as the survey was read: each
      cell's text, stripped, the parts joined with ``/``. The frame's row order is kept.
      A row whose id is not in the result, or whose id has a blank part other than in a
      column listed in ``provenance["response_key_blank_ok"]``, gets nulls in every
      added column.
    - ``"row"``: rows are aligned by position.

    One INFO line on the ``soundscapy`` logger records the alignment (``"id"`` or
    ``"position"``) and the number of matched rows. Attaching to ``survey.data`` always
    aligns.

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
        frame row matches, the frame's row count differs from the survey's when aligning
        by position, or the result records no known id source.
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
    logger.info(
        "attach: aligned by %s; %d of %d frame rows matched",
        alignment,
        int(matched.sum()),
        n_rows,
    )
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
    parts = {column: stripped_text(eager, column) for column in id_columns}
    ids, blank = join_id_parts(parts, blank_ok)
    usable = ~blank
    _check_unique(ids[usable].tolist())
    position = {rid: i for i, rid in enumerate(result.labels[COL_RESPONSE_ID].tolist())}
    index = np.zeros(n_rows, dtype=np.intp)
    matched = np.zeros(n_rows, dtype=bool)
    for row in np.flatnonzero(usable).tolist():
        hit = position.get(ids[row])
        if hit is not None:
            index[row], matched[row] = hit, True
    if not matched.any():
        msg = f"no frame row matches a response id of the result, built from {id_columns}"
        raise AlignmentError(msg, hint="check the id columns hold the survey's values")
    return index, matched


def _check_unique(ids: list[str]) -> None:
    repeated = sorted(rid for rid, count in Counter(ids).items() if count > 1)
    if repeated:
        msg = f"the frame's response ids are not unique: {_shown(list(map(repr, repeated)))} repeat"
        raise AlignmentError(msg)
