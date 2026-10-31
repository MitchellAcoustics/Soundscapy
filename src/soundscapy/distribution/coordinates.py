"""Projected responses: their (x, y) coordinates, per-response labels and origin.

:func:`valid_mask`, :func:`n` and :func:`n_valid` are the one definition of a valid row
and of the two counts; every consumer calls them.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, StrictInt, StrictStr

from soundscapy._frames import new_frame
from soundscapy._model import COL_INCLUDED, COL_LANGUAGE, FrozenModel, axis_column
from soundscapy.errors import MappingError
from soundscapy.instrument import Circumplex, Scale

__all__ = ["Coordinates", "OriginSegment", "n", "n_valid", "valid_mask"]

_XY_COLUMNS = 2


class OriginSegment(FrozenModel):
    """Where a run of rows of :class:`Coordinates` came from.

    Attributes
    ----------
    kind : {"survey"}
        The kind of source.
    dataset : str
        The source's name: the file name of a survey read from a file.
    n : int
        The number of rows in the segment.
    input_hash, mapping_hash : str
        The ``h1`` hashes of the input and of its mapping.
    effective_scale : Scale
        The response scale the scores were projected with.
    angle_sets : tuple of str
        The angle-set labels used, for example ``("equal",)``.
    settings_hash : str
        The ``h1`` hash of the projection settings.
    """

    kind: Literal["survey"]
    dataset: StrictStr
    n: StrictInt = Field(ge=0)
    input_hash: StrictStr
    mapping_hash: StrictStr
    effective_scale: Scale
    angle_sets: tuple[StrictStr, ...]
    settings_hash: StrictStr


def _frozen(values: NDArray[Any]) -> NDArray[Any]:
    out = np.array(values, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False, slots=True)
class Coordinates:
    """Responses projected onto the two circumplex axes, one row per response.

    Rows align with the rows of the survey they were projected from. The arrays are
    copied and made read-only, and the provenance deep-copied, on construction.

    Parameters
    ----------
    xy : numpy.ndarray of float64, shape (n, 2)
        The x and y coordinates, dimensionless; NaN on rows not projected.
    circumplex : Circumplex
        The instrument; its axes name the two columns.
    labels : dict of str to numpy.ndarray
        Per-response columns of length n (bool, float64, or object arrays of str), in
        column order. For a survey: ``ssp_response_id``, ``ssp_included``,
        ``ssp_reasons``, ``ssp_flags``, ``ssp_language``, ``ssp_angle_set`` (``""`` on
        rows not projected), then each mapped id role under the user's column name.
    origin : tuple of OriginSegment
        Where the rows came from; the segments' ``n`` sum to the number of rows.
    settings : FrozenModel
        The settings of the operation that made the coordinates.
    provenance : dict
        The provenance envelope; ``provenance["id_columns"]`` maps each id role to its
        label column.

    Raises
    ------
    TypeError
        If ``xy`` is not a float64 array.
    ValueError
        If ``xy`` is not of shape (n, 2), a label is not one-dimensional of length n, or
        the origin counts do not sum to n.
    """

    xy: NDArray[np.float64]
    circumplex: Circumplex
    labels: dict[str, NDArray[Any]]
    origin: tuple[OriginSegment, ...]
    settings: FrozenModel
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        """Copy the arrays read-only and the provenance deeply; check shapes and counts."""
        xy = np.asarray(self.xy)
        if xy.dtype != np.float64:
            msg = f"xy must be a float64 array, got {xy.dtype}"
            raise TypeError(msg)
        if xy.ndim != _XY_COLUMNS or xy.shape[1] != _XY_COLUMNS:
            msg = f"xy must have shape (n, 2), got {xy.shape}"
            raise ValueError(msg)
        rows = xy.shape[0]
        for name, values in self.labels.items():
            if np.ndim(values) != 1 or len(values) != rows:
                msg = f"label {name!r} must be one-dimensional of length {rows}, got shape "
                msg += f"{np.shape(values)}"
                raise ValueError(msg)
        total = sum(segment.n for segment in self.origin)
        if total != rows:
            msg = f"the origin segments count {total} rows, but xy has {rows}"
            raise ValueError(msg)
        object.__setattr__(self, "xy", _frozen(xy))
        labels = {name: _frozen(values) for name, values in self.labels.items()}
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "origin", tuple(self.origin))
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))

    def n(self) -> int:
        """Return the number of rows; see :func:`n`."""
        return n(self)

    def n_valid(self) -> int:
        """Return the number of valid rows; see :func:`n_valid`."""
        return n_valid(self)

    def valid_mask(self) -> NDArray[np.bool_]:
        """Return which rows are valid; see :func:`valid_mask`."""
        return valid_mask(self)

    def to_frame(self, like: object | None = None) -> Any:  # noqa: ANN401  # pandas or the library of like
        """Return the labels and coordinates as a data frame.

        Parameters
        ----------
        like : object, optional
            A frame whose library the result uses; pandas when omitted.

        Returns
        -------
        object
            The label columns, then ``ssp_<axis id>`` for the x and the y axis (for
            example ``ssp_ISOPleasant`` and ``ssp_ISOEventful``).
        """
        x_axis, y_axis = self.circumplex.axes
        columns = {
            **self.labels,
            axis_column(x_axis.id): self.xy[:, 0],
            axis_column(y_axis.id): self.xy[:, 1],
        }
        return new_frame(columns, like=like)

    def select(self, mask: ArrayLike) -> "Coordinates":
        """Return the rows where ``mask`` is true.

        Parameters
        ----------
        mask : array-like of bool
            One entry per row.

        Returns
        -------
        Coordinates
            The chosen rows, with the origin's ``n`` updated; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If ``mask`` is not a boolean array with one entry per row.
        NotImplementedError
            If the coordinates have more than one origin segment.
        """
        keep = np.asarray(mask)
        if keep.shape != (n(self),) or keep.dtype != np.bool_:
            msg = f"mask must be a boolean array of shape ({n(self)},), got {keep.dtype} "
            msg += f"{keep.shape}"
            raise ValueError(msg)
        if len(self.origin) != 1:
            msg = "selecting from coordinates with more than one origin segment is not supported"
            raise NotImplementedError(msg)
        segment = OriginSegment.model_validate(
            {**self.origin[0].model_dump(), "n": int(keep.sum())}
        )
        return Coordinates(
            xy=self.xy[keep],
            circumplex=self.circumplex,
            labels={name: values[keep] for name, values in self.labels.items()},
            origin=(segment,),
            settings=self.settings,
            provenance=self.provenance,
        )

    def split(self, by: str) -> dict[str, "Coordinates"]:
        """Split the rows into groups by an id role, the language or a label column.

        Parameters
        ----------
        by : str
            An id role (``"location"``, ``"session"``, ``"group"``, ``"record"``) that
            the mapping names, ``"language"``, or a label column name.

        Returns
        -------
        dict of str to Coordinates
            One entry per distinct value of the column, in sorted key order.

        Raises
        ------
        MappingError
            If ``by`` is none of these; the message lists the choices.
        """
        roles = _id_columns(self)
        if by in roles:
            column = roles[by]
        elif by == "language":
            column = COL_LANGUAGE
        elif by in self.labels:
            column = by
        else:
            choices = [*roles, "language", *self.labels]
            msg = f"cannot split by {by!r}; choose one of {choices}"
            raise MappingError(msg)
        keys = np.array([str(value) for value in self.labels[column].tolist()], dtype=object)
        return {key: self.select(keys == key) for key in sorted(set(keys.tolist()))}


def _id_columns(coords: Coordinates) -> dict[str, str]:
    """Return ``{role: label column}`` from the provenance, ``{}`` when none is recorded."""
    recorded = coords.provenance.get("id_columns", {})
    if not isinstance(recorded, Mapping):
        msg = f'provenance["id_columns"] must be a mapping, got {type(recorded).__name__}'
        raise TypeError(msg)
    return {str(role): str(column) for role, column in recorded.items()}


def valid_mask(coords: Coordinates) -> NDArray[np.bool_]:
    """Return which rows are valid.

    A row is valid when it is included (``ssp_included``) and both its x and y are finite.

    Parameters
    ----------
    coords : Coordinates
        The coordinates.

    Returns
    -------
    numpy.ndarray of bool
        One read-only entry per row.
    """
    mask = np.asarray(coords.labels[COL_INCLUDED], dtype=np.bool_) & np.isfinite(coords.xy).all(
        axis=1
    )
    mask.setflags(write=False)
    return mask


def n(coords: Coordinates) -> int:
    """Return the number of rows, valid or not.

    Parameters
    ----------
    coords : Coordinates
        The coordinates.

    Returns
    -------
    int
        The number of rows.
    """
    return int(coords.xy.shape[0])


def n_valid(coords: Coordinates) -> int:
    """Return the number of valid rows (see :func:`valid_mask`).

    Parameters
    ----------
    coords : Coordinates
        The coordinates.

    Returns
    -------
    int
        The number of rows that are included and have finite x and y.
    """
    return int(valid_mask(coords).sum())
