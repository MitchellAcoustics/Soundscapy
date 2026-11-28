"""Reading a survey export into a coded, validated ``Survey``.

``read_survey()`` reads a CSV or TSV file with every cell as text, builds a response id
for every row, codes the language and the attribute items, checks the coded values
against the response scale and records each response's inclusion status.
"""

from __future__ import annotations

import codecs
import copy
import os
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NoReturn

import numpy as np
from pydantic import StrictStr, field_validator

from soundscapy import _data
from soundscapy._frames import column_text, read_text_table, readonly_copy, take_rows, with_columns
from soundscapy._log import logger
from soundscapy._model import (
    _UNSET,
    COL_RESPONSE_ID,
    SSP_PREFIX,
    FrozenModel,
    _Unset,
    merge_settings,
    settings_layer,
    validate,
)
from soundscapy.errors import (
    CodingError,
    ColumnCollisionError,
    DuplicateResponseIdError,
    FileFormatError,
    MappingError,
    MissingResponseIdError,
    ScaleViolationError,
    SettingsError,
    SoundscapyError,
    _plural,
    _shown,
)
from soundscapy.provenance import envelope, hash_file, hash_json
from soundscapy.survey.mapping import (
    _LANGUAGE_CODE,
    _NOT_LANGUAGE_CODE,
    Coding,
    Groups,
    IdSpec,
    ItemGroup,
    LanguageSpec,
    SurveyMapping,
    apply_coding,
    read_mapping,
)
from soundscapy.survey.quality import QualityInputs, QualitySettings, apply_quality
from soundscapy.survey.status import Flag, Reason, Status

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ImportSettings",
    "ReadOptions",
    "SourceRef",
    "Survey",
    "read_survey",
    "resolve_import_settings",
]

_ROLES: Final = ("location", "session", "group", "record")
_SUFFIX_FORMATS: Final[Mapping[str, Literal["csv", "tsv"]]] = {".csv": "csv", ".tsv": "tsv"}
_DEFAULT_DELIMITERS: Final[Mapping[str, str]] = {"csv": ",", "tsv": "\t"}
_FLAG_NAMES: Final = ("consent", "test", "paper_input")
_SCALE_HINT: Final = (
    "add a missing_values code for a value that means no answer, set coding.offset to "
    'shift the scale (a 0..4 export needs offset = 1.0), or pass on_scale_violation="exclude" '
    "to make each violating cell missing"
)


class ReadOptions(FrozenModel):
    """How a survey file is read.

    Attributes
    ----------
    format : {"auto", "csv", "tsv"}, optional
        The file format; ``"auto"`` takes it from the file suffix (``.csv`` or ``.tsv``).
    encoding : str, optional
        The text encoding, for example ``"utf-8-sig"`` (the default, which also reads plain
        UTF-8) or ``"cp1252"``.
    delimiter : str, optional
        One field-delimiter character; ``"auto"`` means ``,`` for csv and a tab for tsv.

    Raises
    ------
    pydantic.ValidationError
        If ``encoding`` is not a known codec or ``delimiter`` is neither ``"auto"`` nor
        one character.
    """

    format: Literal["auto", "csv", "tsv"] = "auto"
    encoding: StrictStr = "utf-8-sig"
    delimiter: StrictStr = "auto"

    @field_validator("encoding")
    @classmethod
    def _known_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError:
            msg = f"unknown encoding {value!r}"
            raise ValueError(msg) from None
        return value

    @field_validator("delimiter")
    @classmethod
    def _one_character(cls, value: str) -> str:
        if value != "auto" and len(value) != 1:
            msg = f'delimiter must be "auto" or one character, got {value!r}'
            raise ValueError(msg)
        return value


class ImportSettings(FrozenModel):
    """Settings of ``read_survey()`` (spec 6.3.4).

    Attributes
    ----------
    quality : QualitySettings, optional
        The quality rules' actions and thresholds (spec 6.7).
    on_scale_violation : {"error", "exclude"}, optional
        What a coded value outside its response scale does: ``"error"`` raises
        ``ScaleViolationError`` naming every offending column; ``"exclude"`` makes each
        violating cell missing and gives the response reason ``scale_violation``.
    """

    quality: QualitySettings = QualitySettings()
    on_scale_violation: Literal["error", "exclude"] = "error"


class SourceRef(FrozenModel):
    """The file a survey was read from.

    Attributes
    ----------
    name : str
        The file name, without its directory.
    hash : str
        The ``h1`` hash of the file's raw bytes.
    read_options : ReadOptions
        The options the file was read with, with ``format`` and ``delimiter`` resolved.
    """

    name: StrictStr
    hash: StrictStr
    read_options: ReadOptions


@dataclass(frozen=True, eq=False, slots=True)
class Survey:
    """A survey read and coded against its mapping, one entry per response.

    Built by ``read_survey()``. The arrays are copied and made read-only, and the
    provenance deep-copied, on construction.

    Parameters
    ----------
    data : pandas.DataFrame
        The file as read: every column, every cell as text, blank rows skipped.
    mapping : SurveyMapping
        The normalised mapping.
    scores : dict of str to numpy.ndarray
        Per item group, the coded scores as float64 of shape ``(n, k)``, NaN where
        missing; ``"attributes"`` holds the attribute items in position order.
    response_id : numpy.ndarray of str
        Each response's unique id.
    ids : dict of str to numpy.ndarray
        For each mapped id role (``location``, ``session``, ``group``, ``record``), its
        column's text, stripped.
    language : numpy.ndarray of str
        Each response's ISO 639-3 language code, ``""`` when unknown or not mapped.
    status : Status
        Each response's inclusion status.
    settings : ImportSettings
        The settings the survey was read with.
    source : SourceRef
        The file read.
    provenance : dict
        The provenance envelope.

    Raises
    ------
    ValueError
        If the per-response arrays and the status differ in length.
    """

    data: pd.DataFrame
    mapping: SurveyMapping
    scores: dict[str, NDArray[np.float64]]
    response_id: NDArray[np.object_]
    ids: dict[str, NDArray[np.object_]]
    language: NDArray[np.object_]
    status: Status
    settings: ImportSettings
    source: SourceRef
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        """Copy the arrays read-only and the provenance deeply; check array lengths."""
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))
        object.__setattr__(self, "response_id", readonly_copy(self.response_id, object))
        object.__setattr__(self, "language", readonly_copy(self.language, object))
        scores = {name: readonly_copy(v, np.float64) for name, v in self.scores.items()}
        object.__setattr__(self, "scores", scores)
        ids = {role: readonly_copy(v, object) for role, v in self.ids.items()}
        object.__setattr__(self, "ids", ids)
        lengths = {
            "response_id": len(self.response_id),
            "language": len(self.language),
            "status": len(self.status.included),
            **{f"scores[{name!r}]": len(v) for name, v in self.scores.items()},
            **{f"ids[{role!r}]": len(v) for role, v in self.ids.items()},
        }
        if len(set(lengths.values())) > 1:
            msg = f"per-response arrays differ in length: {lengths}"
            raise ValueError(msg)

    def n(self) -> int:
        """Return the number of responses.

        Returns
        -------
        int
            The number of rows read, blank rows excluded.
        """
        return len(self.response_id)

    def to_frame(self) -> pd.DataFrame:
        """Return the data with the response id and status columns appended.

        Returns
        -------
        pandas.DataFrame
            ``data``'s columns, then ``ssp_response_id``, ``ssp_included``, ``ssp_reasons``
            and ``ssp_flags``; ``data`` itself is unchanged.
        """
        columns = {COL_RESPONSE_ID: self.response_id, **self.status.label_arrays()}
        return with_columns(self.data, columns)

    def select(self, mask: ArrayLike) -> Survey:
        """Return the responses where ``mask`` is true, in order.

        Parameters
        ----------
        mask : array-like of bool
            One entry per response.

        Returns
        -------
        Survey
            The chosen responses: every per-response array, score table, id column and the
            status restricted to them, and ``data`` with a fresh index. The mapping,
            settings, source and provenance are unchanged; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If ``mask`` is not a boolean array with one entry per response.
        """
        keep = np.asarray(mask)
        if keep.shape != (self.n(),) or keep.dtype != np.bool_:
            msg = f"mask must be a boolean array of shape ({self.n()},), got {keep.dtype} "
            msg += f"{keep.shape}"
            raise ValueError(msg)
        return Survey(
            data=take_rows(self.data, keep),
            mapping=self.mapping,
            scores={name: values[keep] for name, values in self.scores.items()},
            response_id=self.response_id[keep],
            ids={role: values[keep] for role, values in self.ids.items()},
            language=self.language[keep],
            status=self.status.select(keep),
            settings=self.settings,
            source=self.source,
            provenance=self.provenance,
        )


def read_survey(
    source: str | os.PathLike[str],
    mapping: SurveyMapping | str | os.PathLike[str],
    *,
    read_options: ReadOptions | Mapping[str, object] | _Unset = _UNSET,
    quality: QualitySettings | Mapping[str, object] | _Unset = _UNSET,
    on_scale_violation: Literal["error", "exclude"] | _Unset = _UNSET,
    settings: ImportSettings | Mapping[str, object] | _Unset = _UNSET,
) -> Survey:
    """Read a survey export into a coded, validated survey.

    The file is read with every cell as text, and then, in order: a column starting with
    ``ssp_`` is refused; every mapped column must be in the header; each row gets a
    response id; the language and every mapped item group are coded; coded values are
    checked against their group's scale (raising, or under ``on_scale_violation="exclude"``
    making the cell missing); the age and flag columns are read; and the quality rules of
    ``settings.quality`` write each response's status, a response with a missing attribute
    item always getting reason ``incomplete_attributes``. Id errors come before coding
    errors, because coding and scale errors name responses by id. One INFO line on the
    ``soundscapy`` logger summarises the rows read, blank rows skipped, responses included
    and excluded by reason, and the flags given.

    Settings resolve in the order of spec 6.3.4: an explicit keyword (``quality``,
    ``on_scale_violation``) beats ``settings``, which beats the model defaults.

    The response id is, by the first rule that applies: the mapped ``response`` column,
    stripped; the mapped ``response_key`` columns, each stripped, joined with ``/``; or
    ``row-<n>``, where ``n`` is the record number in the file with the header as row 1,
    so blank rows do not shift later ids. The rule used is recorded as
    ``provenance["response_id_source"]``.

    Parameters
    ----------
    source : str or os.PathLike
        A ``.csv`` or ``.tsv`` file. Data frames are not accepted in this version.
    mapping : SurveyMapping, str or os.PathLike
        The mapping, or a mapping TOML file to read with ``read_mapping()``.
    read_options : ReadOptions or mapping, optional
        How the file is read; a mapping is validated into ``ReadOptions``.
    quality : QualitySettings or mapping, optional
        The quality rules; a mapping holds the fields to change.
    on_scale_violation : {"error", "exclude"}, optional
        What a coded value outside its scale does; see ``ImportSettings``.
    settings : ImportSettings or mapping, optional
        Every import setting at once; the two keywords above override its fields.
        Settings files arrive with profiles.

    Returns
    -------
    Survey
        One entry per non-blank row of the file.

    Raises
    ------
    TypeError
        If ``source`` is not a path, or ``mapping`` is neither a mapping nor a path.
    SettingsError
        If ``read_options`` or a settings argument has the wrong type or fails validation.
    FileFormatError
        If the format cannot be told from the suffix, or the file is malformed.
    EncodingError
        If the file cannot be decoded.
    MappingError
        If the mapping file is invalid, a mapped column is not in the header, a key part
        holds ``/``, or a language text is not an ISO 639-3 code after coding.
    ColumnCollisionError
        If a column name starts with ``ssp_``.
    MissingResponseIdError
        If a response id, or a key part not listed in ``response_key_blank_ok``, is blank.
    DuplicateResponseIdError
        If two rows share a response id.
    CodingError
        If an item text is neither missing, a value-map key nor a number, or an age text is
        not a number.
    ScaleViolationError
        If a coded value lies outside its group's scale and ``on_scale_violation`` is
        ``"error"``.

    References
    ----------
    The items and their response scales: [@iso12913-2-2018]. Excluding responses with
    missing attribute items before analysis: [@mitchell2022how]. The quality rules:
    [@erfanian2021psychological].
    """
    if not isinstance(source, str | os.PathLike):
        msg = (
            "read_survey reads a survey from a file path (str or os.PathLike) in this "
            f"version; in-memory data frames are not supported yet, got {type(source).__name__}"
        )
        raise TypeError(msg)
    if isinstance(mapping, SurveyMapping):
        resolved = mapping
    elif isinstance(mapping, str | os.PathLike):
        resolved = read_mapping(mapping)  # validation failures arrive as MappingError
    else:
        msg = f"mapping must be a SurveyMapping or a path, got {type(mapping).__name__}"
        raise TypeError(msg)
    import_settings = resolve_import_settings(
        quality=quality, on_scale_violation=on_scale_violation, settings=settings
    )
    return _read_survey(
        source, resolved, read_options=_read_options(read_options), settings=import_settings
    )


def resolve_import_settings(
    *,
    quality: QualitySettings | Mapping[str, object] | _Unset = _UNSET,
    on_scale_violation: Literal["error", "exclude"] | _Unset = _UNSET,
    settings: ImportSettings | Mapping[str, object] | _Unset = _UNSET,
) -> ImportSettings:
    """Resolve the import settings from the keywords ``read_survey()`` and the loaders take.

    Parameters
    ----------
    quality : QualitySettings or mapping, optional
        The quality rules, or the fields of them to change.
    on_scale_violation : {"error", "exclude"}, optional
        What a coded value outside its scale does.
    settings : ImportSettings or mapping, optional
        Every import setting at once.

    Returns
    -------
    ImportSettings
        The layers merged, the keywords winning over ``settings`` and both over the model
        defaults (spec 6.3.4 ranks 1 and 2), and validated once.

    Raises
    ------
    SettingsError
        If an argument has the wrong type, ``settings`` is a path (settings files arrive
        with profiles), or the merged settings fail validation.
    """
    layers: list[Mapping[str, object]] = []
    if settings is not _UNSET:
        if isinstance(settings, str | os.PathLike):
            msg = (
                f"settings={os.fspath(settings)!r} is a path, but settings files arrive with "
                "profiles; pass an ImportSettings or a mapping of its fields"
            )
            raise SettingsError(msg)
        layers.append(settings_layer(settings, model=ImportSettings, name="settings"))
    if quality is not _UNSET:
        layers.append({"quality": settings_layer(quality, model=QualitySettings, name="quality")})
    if on_scale_violation is not _UNSET:
        layers.append({"on_scale_violation": on_scale_violation})
    return merge_settings(ImportSettings, layers, context="invalid import settings")


def _read_options(value: ReadOptions | Mapping[str, object] | _Unset) -> ReadOptions:
    if value is _UNSET:
        return ReadOptions()
    if isinstance(value, ReadOptions):
        return value
    if not isinstance(value, Mapping):
        msg = f"read_options must be a ReadOptions or a mapping, got {type(value).__name__}"
        raise SettingsError(msg)
    return validate(ReadOptions, dict(value), context="invalid read_options")


def _resolve(options: ReadOptions, path: Path) -> ReadOptions:
    """Return the options with ``format`` and ``delimiter`` resolved for ``path``."""
    fmt = options.format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix not in _SUFFIX_FORMATS:
            msg = (
                f"cannot tell the format of {path.name!r} from its suffix {suffix!r}; "
                "this version reads .csv and .tsv files"
            )
            hint = 'pass read_options={"format": "csv"} for a comma-separated file'
            raise FileFormatError(msg, hint=hint)
        fmt = _SUFFIX_FORMATS[suffix]
    delimiter = _DEFAULT_DELIMITERS[fmt] if options.delimiter == "auto" else options.delimiter
    return ReadOptions(format=fmt, encoding=options.encoding, delimiter=delimiter)


def _read_survey(
    source: str | os.PathLike[str],
    mapping: SurveyMapping,
    *,
    read_options: ReadOptions,
    settings: ImportSettings,
    data_refs: Sequence[Mapping[str, str]] = (),
) -> Survey:
    """Read a survey from a path with resolved inputs; ``data_refs`` follow the circumplex's."""
    path = Path(source)
    options = _resolve(read_options, path)
    table = read_text_table(path, encoding=options.encoding, delimiter=options.delimiter)
    frame = table.frame
    _check_columns(list(frame.columns), mapping)
    ids = mapping.ids
    role_columns = {role: getattr(ids, role) for role in _mapped_roles(ids)}
    stripped = {
        column: stripped_text(frame, column)
        for column in (ids.response, *ids.response_key, *role_columns.values())
        if column
    }
    rows = table.row_numbers
    response_id, id_source, id_columns = _response_ids(ids, stripped, rows)
    language = _languages(mapping.language, frame, response_id)
    coded = {
        name: _code_group(group, frame, response_id) for name, group in _mapped_groups(mapping)
    }
    status = Status.all_included(len(response_id))
    violated = [name for name, (_, outside) in coded.items() if outside.any()]
    if violated and settings.on_scale_violation == "error":
        groups = dict(_mapped_groups(mapping))
        raise ScaleViolationError(
            _scale_violation_message(
                [(name, groups[name], *coded[name]) for name in violated], response_id
            ),
            hint=_SCALE_HINT,
        )
    scores: dict[str, NDArray[np.float64]] = {}
    violating_rows = np.zeros(len(response_id), dtype=np.bool_)
    for name, (values, outside) in coded.items():
        values[outside] = np.nan
        violating_rows |= outside.any(axis=1)
        scores[name] = values
    status = status.with_reason(violating_rows, Reason.SCALE_VIOLATION)
    attribute_scale = mapping.groups.attributes.scale
    inputs = QualityInputs(
        attributes=scores["attributes"],
        midpoint=(attribute_scale.minimum + attribute_scale.maximum) / 2,
        source_dominance=scores.get("source_dominance"),
        consent=_flag(frame, mapping, "consent"),
        test=_flag(frame, mapping, "test"),
        paper_input=_flag(frame, mapping, "paper_input"),
        age=_age(frame, mapping, response_id),
    )
    status, skipped = apply_quality(status, inputs, settings.quality)
    _log_summary(path.name, table.n_blank, status)

    source_ref = SourceRef(name=path.name, hash=hash_file(path), read_options=options)
    circumplex_ref = {
        "kind": "circumplex",
        "name": mapping.instrument,
        "version": mapping.instrument_version,
        "hash": _data.pin(f"circumplexes/{mapping.instrument}/{mapping.instrument_version}.toml"),
    }
    blank_ok = set(ids.response_key_blank_ok)
    provenance = envelope(
        operation="read_survey",
        settings=settings,
        input_hash=source_ref.hash,
        mapping_hash=hash_json(mapping),
        data_refs=[circumplex_ref, *data_refs],
        response_id_source=id_source,
        read=options.model_dump(mode="json"),
        id_columns=role_columns,
        response_id_columns=id_columns,
        response_key_blank_ok=[c for c in ids.response_key if c in blank_ok],
        quality_skipped=list(skipped),
    )
    return Survey(
        data=frame,
        mapping=mapping,
        scores=scores,
        response_id=response_id,
        ids={role: stripped[column] for role, column in role_columns.items()},
        language=language,
        status=status,
        settings=settings,
        source=source_ref,
        provenance=provenance,
    )


def _mapped_roles(ids: IdSpec) -> list[str]:
    return [role for role in _ROLES if getattr(ids, role)]


def _mapped_groups(mapping: SurveyMapping) -> list[tuple[str, ItemGroup]]:
    """Return the item groups with columns, attributes first, then in ``Groups`` field order."""
    return [
        (name, group)
        for name in Groups.model_fields
        if (group := getattr(mapping.groups, name)).columns
    ]


def _check_columns(header: list[str], mapping: SurveyMapping) -> None:
    """Refuse ``ssp_`` columns, then mapped columns missing from the header."""
    clashes = [name for name in header if name.startswith(SSP_PREFIX)]
    if clashes:
        msg = (
            f"columns {clashes} start with {SSP_PREFIX!r}, which soundscapy reserves for "
            "the columns it derives; rename them in the file"
        )
        raise ColumnCollisionError(msg)
    ids = mapping.ids
    mapped = [("ids.response", ids.response)] if ids.response else []
    mapped += [(f"ids.response_key[{i}]", col) for i, col in enumerate(ids.response_key)]
    mapped += [(f"ids.{role}", getattr(ids, role)) for role in _mapped_roles(ids)]
    if mapping.language.column:
        mapped.append(("language.column", mapping.language.column))
    for name, group in _mapped_groups(mapping):
        mapped += [(f"groups.{name}.columns.{key}", column) for key, column in group.columns]
    if mapping.age:
        mapped.append(("age", mapping.age))
    for name in _FLAG_NAMES:
        column = getattr(mapping.flags, name).column
        if column:
            mapped.append((f"flags.{name}.column", column))
    mapped += [(f"carry[{i}]", column) for i, column in enumerate(mapping.carry)]
    columns = set(header)
    missing = [f"{key} = {column!r}" for key, column in mapped if column not in columns]
    if missing:
        msg = f"the mapping names columns the file does not have: {'; '.join(missing)}"
        raise MappingError(msg, hint=f"the file's columns are {header}")


def _rows(numbers: Sequence[int]) -> str:
    """``row 3`` or ``rows 3, 5``, at most five numbers then how many more."""
    return f"{_plural(len(numbers), 'row')} {_shown(numbers)}"


def _check_not_blank(
    values: NDArray[np.object_], rows: NDArray[np.int64], what: str, hint: str
) -> None:
    blank = [int(rows[i]) for i, value in enumerate(values.tolist()) if not value]
    if blank:
        msg = f"{what} is blank at {_rows(blank)}"
        raise MissingResponseIdError(msg, hint=hint)


def stripped_text(frame: pd.DataFrame, name: str) -> NDArray[np.object_]:
    """Read an id column as text, each cell stripped of surrounding whitespace.

    Parameters
    ----------
    frame : pandas.DataFrame
        The frame.
    name : str
        The column.

    Returns
    -------
    numpy.ndarray of str
        One stripped text per row, ``""`` for a null.
    """
    out = np.empty(len(frame), dtype=object)
    out[:] = [text.strip() for text in column_text(frame, name).tolist()]
    return out


def join_id_parts(
    parts: Mapping[str, NDArray[np.object_]], blank_ok: Collection[str]
) -> tuple[NDArray[np.object_], NDArray[np.bool_]]:
    """Join response-key parts into response ids, the way ``read_survey()`` builds them.

    Parameters
    ----------
    parts : mapping of str to numpy.ndarray of str
        Per key column, in key order, its stripped texts; every array has one entry per
        row.
    blank_ok : collection of str
        The key columns that may be blank.

    Returns
    -------
    ids : numpy.ndarray of str
        Each row's parts joined with ``/``.
    blank : numpy.ndarray of bool
        Whether the row has a blank part in a column not in ``blank_ok``.
    """
    texts = [part.tolist() for part in parts.values()]
    n_rows = len(texts[0]) if texts else 0
    blank = np.zeros(n_rows, dtype=np.bool_)
    for column, part in parts.items():
        if column not in blank_ok:
            blank |= np.array([not text for text in part.tolist()], dtype=np.bool_)
    ids = np.empty(n_rows, dtype=object)
    ids[:] = ["/".join(key) for key in zip(*texts, strict=True)]
    return ids, blank


def _response_ids(
    ids: IdSpec, stripped: Mapping[str, NDArray[np.object_]], rows: NDArray[np.int64]
) -> tuple[NDArray[np.object_], str, list[str]]:
    """Return the response ids, the rule used and the columns they were built from.

    ``stripped`` holds the stripped text of every mapped id column.
    """
    if ids.response:
        values = stripped[ids.response]
        hint = "fill in every id, or map a response_key instead"
        _check_not_blank(values, rows, f"the response id column {ids.response!r}", hint)
        source, columns = "column", [ids.response]
    elif ids.response_key:
        parts = {column: stripped[column] for column in ids.response_key}
        values, blank = join_id_parts(parts, ids.response_key_blank_ok)
        if blank.any():
            _raise_blank_key(parts, blank, rows, ids.response_key_blank_ok)
        for column, part in parts.items():
            _check_no_slash(part, rows, column)
        source, columns = "key", list(ids.response_key)
    else:
        values = np.empty(len(rows), dtype=object)
        values[:] = [f"row-{n}" for n in rows.tolist()]
        source, columns = "row", []
    _check_unique(values, rows)
    return values, source, columns


def _raise_blank_key(
    parts: Mapping[str, NDArray[np.object_]],
    blank: NDArray[np.bool_],
    rows: NDArray[np.int64],
    blank_ok: Collection[str],
) -> NoReturn:
    """Raise for the rows ``blank`` marks, naming the key columns blank there."""
    columns = [
        column
        for column, part in parts.items()
        if column not in blank_ok and not all(part[blank].tolist())
    ]
    where = _rows(rows[blank].tolist())
    msg = f"the response key is blank at {where}, in {_plural(len(columns), 'column')} "
    msg += ", ".join(map(repr, columns))
    hint = "fill in every key part, or list the column in ids.response_key_blank_ok"
    raise MissingResponseIdError(msg, hint=hint)


def _check_no_slash(part: NDArray[np.object_], rows: NDArray[np.int64], column: str) -> None:
    hits = [i for i, value in enumerate(part.tolist()) if "/" in value]
    if hits:
        shown = _shown([f"{part[i]!r} at row {rows[i]}" for i in hits])
        msg = f"the response key column {column!r} holds '/', which joins the key parts: {shown}"
        raise MappingError(msg, hint="map a response column instead, or clean the key column")


def check_unique(
    values: Sequence[str],
    rows: Sequence[int],
    *,
    error: type[SoundscapyError],
    describe: Callable[[list[tuple[str, list[int]]]], str],
    hint: str = "",
) -> None:
    """Raise when a response id repeats.

    Parameters
    ----------
    values : sequence of str
        The ids.
    rows : sequence of int
        Each id's row number, as the message should show it.
    error : type[SoundscapyError]
        The error raised.
    describe : callable
        Turns the repeats, ``(id, rows)`` pairs in first-occurrence order, into the
        error message.
    hint : str, optional
        The error's hint.

    Raises
    ------
    SoundscapyError
        ``error``, if any id repeats.
    """
    where: dict[str, list[int]] = {}
    for value, row in zip(values, rows, strict=True):
        where.setdefault(value, []).append(row)
    repeated = [(value, numbers) for value, numbers in where.items() if len(numbers) > 1]
    if repeated:
        raise error(describe(repeated), hint=hint)


def _check_unique(values: NDArray[np.object_], rows: NDArray[np.int64]) -> None:
    def describe(repeated: list[tuple[str, list[int]]]) -> str:
        shown = _shown([f"{value!r} at {_rows(numbers)}" for value, numbers in repeated], sep="; ")
        return f"response ids must be unique; {len(repeated)} repeat: {shown}"

    check_unique(
        values.tolist(),
        rows.tolist(),
        error=DuplicateResponseIdError,
        describe=describe,
        hint="map a response column that is unique, or more columns into ids.response_key",
    )


def _languages(
    spec: LanguageSpec,
    frame: pd.DataFrame,
    response_ids: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Return each response's ISO 639-3 code, ``""`` where blank or not mapped."""
    out = np.empty(len(response_ids), dtype=object)
    if not spec.column:
        out[:] = [spec.value] * len(response_ids)
        return out
    coding = dict(spec.coding)
    bad: Counter[str] = Counter()
    bad_rows: list[int] = []
    codes = []
    for i, cell in enumerate(column_text(frame, spec.column).tolist()):
        text = cell.strip()
        code = coding.get(text, text) if text else ""
        if code and not _LANGUAGE_CODE.match(code):
            bad[code] += 1
            bad_rows.append(i)
        codes.append(code)
    if bad:
        listed = ", ".join(f"{text!r} x{count}" for text, count in bad.items())
        ids = _shown([response_ids[i] for i in bad_rows])
        msg = (
            f"in the language column {spec.column!r}, each of these texts {_NOT_LANGUAGE_CODE}: "
            f"{listed} (responses {ids})"
        )
        hint = "add a [language.coding] entry giving each text's ISO 639-3 code"
        raise MappingError(msg, hint=hint)
    out[:] = codes
    return out


def _code_group(
    group: ItemGroup,
    frame: pd.DataFrame,
    response_ids: NDArray[np.object_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Code an item group's columns; return the scores and which cells are off its scale."""
    cells = np.empty((len(response_ids), len(group.columns)), dtype=object)
    for j, (_, column) in enumerate(group.columns):
        cells[:, j] = column_text(frame, column)
    keys = [key for key, _ in group.columns]
    scores = apply_coding(
        cells, group.coding, group.scale, item_keys=keys, response_ids=response_ids
    )
    outside = ~np.isnan(scores) & ~group.scale.contains(scores)
    return scores, outside


def _scale_violation_message(
    violated: Sequence[tuple[str, ItemGroup, NDArray[np.float64], NDArray[np.bool_]]],
    response_ids: NDArray[np.object_],
) -> str:
    """One message for every group with a violation, in coding order (spec 6.4.3)."""
    segments = []
    for name, group, scores, outside in violated:
        entries = []
        for j, (key, column) in enumerate(group.columns):
            hit = outside[:, j]
            if not hit.any():
                continue
            counts = Counter(scores[hit, j].tolist())
            listed = ", ".join(f"{value:g} x{counts[value]}" for value in sorted(counts))
            ids = _shown(response_ids[hit].tolist())
            entries.append(f"{column!r} ({key}): {listed} (responses {ids})")
        scale = group.scale
        segments.append(f"{name} ({scale.minimum:g}..{scale.maximum:g}): " + "; ".join(entries))
    return "coded values outside their scales: " + " | ".join(segments)


def _age(
    frame: pd.DataFrame, mapping: SurveyMapping, response_ids: NDArray[np.object_]
) -> NDArray[np.float64] | None:
    """Code the age column as numbers, NaN where missing; ``None`` when unmapped."""
    if not mapping.age:
        return None
    cells = column_text(frame, mapping.age)[:, np.newaxis]
    try:
        coded = apply_coding(
            cells,
            Coding(),
            mapping.groups.attributes.scale,  # unused: nothing is reversed
            item_keys=("age",),
            response_ids=response_ids,
        )
    except CodingError as err:
        msg = f"the age column {mapping.age!r} holds texts that are not numbers; {err}"
        hint = "map age to a column of numbers, or leave age unmapped"
        raise CodingError(msg, hint=hint) from err
    return coded[:, 0]


def _flag(frame: pd.DataFrame, mapping: SurveyMapping, name: str) -> NDArray[np.bool_] | None:
    """Whether each response's flag cell, stripped, is a true value; ``None`` when unmapped."""
    spec = getattr(mapping.flags, name)
    if not spec.column:
        return None
    return np.isin(stripped_text(frame, spec.column), list(spec.true_values))


def _log_summary(name: str, n_blank: int, status: Status) -> None:
    counts = status.counts()
    by_reason = [
        f"{reason.value}: {counts[f'reason:{reason.value}']}"
        for reason in Reason
        if counts[f"reason:{reason.value}"]
    ]
    by_flag = [
        f"{flag.value}: {counts[f'flag:{flag.value}']}"
        for flag in Flag
        if counts[f"flag:{flag.value}"]
    ]
    logger.info(
        "read_survey: %d %s read from %s, %d blank %s skipped; %d included, %d excluded (%s); "
        "flagged (%s)",
        counts["n"],
        _plural(counts["n"], "row"),
        name,
        n_blank,
        _plural(n_blank, "row"),
        counts["included"],
        counts["excluded"],
        ", ".join(by_reason) or "none",
        ", ".join(by_flag) or "none",
    )
