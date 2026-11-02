"""Survey mappings: how a survey export's columns map onto an instrument, and how they are coded.

A ``SurveyMapping`` names the id, language and item columns of a user's export.
``apply_coding()`` turns the item texts into numbers on the instrument scale, and
``read_mapping()`` and ``write_mapping()`` store a mapping as TOML.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pydantic
import tomlkit
from pydantic import Field, StrictFloat, StrictStr, model_validator

from soundscapy._model import FloatPairs, FrozenModel, StrPairs, _to_pairs, parse_toml, validate
from soundscapy.errors import CodingError, MappingError, _shown
from soundscapy.instrument import Circumplex, Scale, circumplex

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "Coding",
    "Groups",
    "IdSpec",
    "ItemGroup",
    "LanguageSpec",
    "SurveyMapping",
    "apply_coding",
    "read_mapping",
    "write_mapping",
]

_LANGUAGE_CODE = re.compile(r"^[a-z]{3}$")
_NOT_LANGUAGE_CODE = "is not an ISO 639-3 code (three lowercase letters)"
_DEFAULT_INSTRUMENT = "iso-12913-outdoor"
_DEFAULT_INSTRUMENT_VERSION = "1"
_IDENTITY_FIELDS = frozenset({"schema_name", "schema_version", "instrument", "instrument_version"})
_NUMBER = {
    ".": re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"),
    ",": re.compile(r"^[+-]?(?:\d+(?:,\d*)?|,\d+)(?:[eE][+-]?\d+)?$"),
}


class IdSpec(FrozenModel):
    """Which columns identify a response, and which hold its location, session and group.

    Attributes
    ----------
    response : str, optional
        Column holding a unique response id. Not together with ``response_key``.
    response_key : tuple of str, optional
        Columns whose texts, joined with ``/``, form the response id.
    response_key_blank_ok : tuple of str, optional
        Columns of ``response_key`` that may be blank; a proper subset of it.
    location, session, group, record : str, optional
        Columns holding each id role; blank means not mapped.
    """

    response: StrictStr = ""
    response_key: tuple[StrictStr, ...] = ()
    response_key_blank_ok: tuple[StrictStr, ...] = ()
    location: StrictStr = ""
    session: StrictStr = ""
    group: StrictStr = ""
    record: StrictStr = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.response and self.response_key:
            msg = "name a response column or a response_key, not both"
            raise ValueError(msg)
        blank_ok = set(self.response_key_blank_ok)
        if blank_ok and (not blank_ok < set(self.response_key)):
            msg = (
                f"response_key_blank_ok {list(self.response_key_blank_ok)} must be a proper "
                f"subset of response_key {list(self.response_key)}"
            )
            raise ValueError(msg)
        return self


class LanguageSpec(FrozenModel):
    """Where each response's language comes from.

    Attributes
    ----------
    column : str, optional
        Column holding the language. Not together with ``value``.
    value : str, optional
        One ISO 639-3 code for every response.
    coding : pairs of str, optional
        Raw column text to ISO 639-3 code, read from and written as a table.
    """

    column: StrictStr = ""
    value: StrictStr = ""
    coding: StrPairs = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.column and self.value:
            msg = "name a language column or a language value, not both"
            raise ValueError(msg)
        if self.value and not _LANGUAGE_CODE.match(self.value):
            msg = f"language value {self.value!r} {_NOT_LANGUAGE_CODE}"
            raise ValueError(msg)
        for text, code in self.coding:
            if not _LANGUAGE_CODE.match(code):
                msg = f"language coding {text!r} = {code!r}: {code!r} {_NOT_LANGUAGE_CODE}"
                raise ValueError(msg)
        return self


class Coding(FrozenModel):
    """How the texts of an item group's columns become numbers.

    Attributes
    ----------
    missing_values : tuple of str, optional
        Texts, after stripping whitespace, that mean a missing response.
    value_map : pairs of str and float, optional
        Text to number, for label responses; read from and written as a table.
    decimal : {".", ","}, optional
        Decimal mark used when parsing numbers.
    offset : float, optional
        Added to every coded value.
    reverse : tuple of str, optional
        Item keys whose scores are reversed against the group's scale.
    """

    missing_values: tuple[StrictStr, ...] = ("", "NA", "N/A", "NaN", "nan", "#N/A")
    value_map: FloatPairs = ()
    decimal: Literal[".", ","] = "."
    offset: StrictFloat = 0.0
    reverse: tuple[StrictStr, ...] = ()


class ItemGroup(FrozenModel):
    """A group of survey items: their columns, response scale and coding.

    Attributes
    ----------
    columns : pairs of str
        Item key to column name, read from and written as a table.
    scale : Scale
        The response scale of the items. Inside a ``SurveyMapping`` it defaults to
        the instrument's scale; built directly it must be given.
    coding : Coding, optional
        How the column texts are coded.
    """

    columns: StrPairs
    scale: Scale
    coding: Coding = Coding()


class Groups(FrozenModel):
    """The item groups a survey maps.

    Attributes
    ----------
    attributes : ItemGroup
        The perceptual attributes. Keys are positions (``PAQ1``) or reference-language
        labels (``pleasant``) of the instrument, normalised to positions; every position
        must be mapped.
    """

    attributes: ItemGroup


class SurveyMapping(FrozenModel):
    """How a survey export maps onto an instrument.

    Attribute keys and ``reverse`` keys are normalised to the instrument's positions in
    position order, so ``pleasant = "x"`` and ``PAQ1 = "x"`` give equal models with
    equal hashes. A missing attribute scale is filled in from the instrument.

    Attributes
    ----------
    schema_name : str
        Always ``"soundscapy.mapping"``; the file key is ``schema``.
    schema_version : int
        Always ``1``.
    instrument, instrument_version : str, optional
        The circumplex instrument the attributes belong to.
    ids : IdSpec
        Id columns.
    language : LanguageSpec, optional
        Language source.
    groups : Groups
        Item groups.

    Raises
    ------
    DataReferenceError
        If the instrument does not ship.
    """

    model_config = FrozenModel.model_config | {"populate_by_name": True}

    schema_name: Literal["soundscapy.mapping"] = Field(alias="schema")
    schema_version: Literal[1]
    instrument: StrictStr = _DEFAULT_INSTRUMENT
    instrument_version: StrictStr = _DEFAULT_INSTRUMENT_VERSION
    ids: IdSpec
    language: LanguageSpec = LanguageSpec()
    groups: Groups

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:  # noqa: ANN401  # pydantic passes raw input of any type
        if not isinstance(data, Mapping):
            return data  # a SurveyMapping instance, or bad input pydantic reports
        name = data.get("instrument", _DEFAULT_INSTRUMENT)
        version = data.get("instrument_version", _DEFAULT_INSTRUMENT_VERSION)
        groups = _as_dict(data.get("groups"))
        if groups is None or not (isinstance(name, str) and isinstance(version, str)):
            return data  # pydantic reports the bad or missing field
        attributes = _as_dict(groups.get("attributes"))
        if attributes is None:
            return data
        circ = circumplex(name, version)  # DataReferenceError propagates unwrapped
        attributes.setdefault("scale", circ.scale)
        columns = _pairs(attributes.get("columns"))
        if columns is not None:
            attributes["columns"] = _normalise_columns(columns, circ)
        coding = _as_dict(attributes.get("coding"))
        if coding is not None and "reverse" in coding:
            coding["reverse"] = _normalise_reverse(coding["reverse"], circ)
            attributes["coding"] = coding
        return {**data, "groups": {**groups, "attributes": attributes}}


def _as_dict(value: object) -> dict[str, Any] | None:
    """Copy a table or a model's fields into a dict; ``None`` for anything else."""
    if isinstance(value, pydantic.BaseModel | Mapping):
        return dict(value)
    return None


def _pairs(value: object) -> list[tuple[str, Any]] | None:
    """Key-value pairs of a table or pair sequence with string keys; ``None`` if malformed.

    For ``None`` pydantic's own validation of the field reports the problem.
    """
    try:
        items = _to_pairs(value)
    except ValueError:  # a repeated key
        return None
    if not isinstance(items, tuple):
        return None
    if all(isinstance(p, tuple | list) and len(p) == 2 and isinstance(p[0], str) for p in items):  # noqa: PLR2004  # a pair
        return [(p[0], p[1]) for p in items]
    return None


def _normalise_columns(
    columns: list[tuple[str, Any]], circ: Circumplex
) -> tuple[tuple[str, Any], ...]:
    """Attribute keys as positions in position order; every position mapped exactly once."""
    by_position: dict[str, tuple[str, Any]] = {}
    for key, column in columns:
        position = circ.position_for(key)
        if position in by_position:
            other = by_position[position][0]
            msg = "groups.attributes.columns: "
            msg += f"attribute keys {other!r} and {key!r} both map {position}"
            raise ValueError(msg)
        by_position[position] = (key, column)
    for position in circ.positions:
        if position not in by_position:
            msg = "groups.attributes.columns: "
            msg += f"attribute {position} ({circ.label(position)!r}) is not mapped"
            raise ValueError(msg)
    return tuple((p, by_position[p][1]) for p in circ.positions)


def _normalise_reverse(reverse: object, circ: Circumplex) -> object:
    """Reverse keys as positions in position order; a repeated item is an error."""
    if not isinstance(reverse, Sequence) or isinstance(reverse, str):
        return reverse  # pydantic reports the type
    if not all(isinstance(key, str) for key in reverse):
        return reverse
    positions: dict[str, str] = {}
    for key in reverse:
        position = circ.position_for(key)
        if position in positions:
            msg = "groups.attributes.coding.reverse: "
            msg += f"reverse keys {positions[position]!r} and {key!r} repeat {position}"
            raise ValueError(msg)
        positions[position] = key
    return tuple(p for p in circ.positions if p in positions)


# --- coding (normative order) -----------------------------------------------------------------


def apply_coding(
    texts: NDArray[np.object_],
    coding: Coding,
    scale: Scale,
    *,
    item_keys: Sequence[str],
    response_ids: NDArray[np.object_],
) -> NDArray[np.float64]:
    """Code an item group's texts as numbers.

    Each cell goes through, in order: strip whitespace and treat a ``missing_values`` text
    as missing (NaN); replace a ``value_map`` key by its number; otherwise parse a number
    with ``decimal`` as the decimal mark; add ``offset``; and for item keys in ``reverse``
    take ``minimum + maximum - score`` against ``scale``. The scale check is not done here.

    Parameters
    ----------
    texts : numpy.ndarray of str, shape (n, k)
        Cell texts, one column per item key.
    coding : Coding
        The coding rules.
    scale : Scale
        The response scale reversed against.
    item_keys : sequence of str
        The item key of each column, length k.
    response_ids : numpy.ndarray of str, shape (n,)
        Response ids, used to name offending rows in errors.

    Returns
    -------
    numpy.ndarray of float64, shape (n, k)
        The coded scores, NaN where missing.

    Raises
    ------
    CodingError
        If a text is neither missing, a value-map key nor a finite number; the message
        lists the distinct texts per item with counts and up to five response ids.
    MappingError
        If ``coding.reverse`` names a key not in ``item_keys``.
    ValueError
        If the shapes of ``texts``, ``item_keys`` and ``response_ids`` disagree.
    TypeError
        If a cell is not a ``str``.

    References
    ----------
    The attribute items and their response scale: [@iso12913-2-2018].
    """
    keys = tuple(item_keys)
    if texts.ndim != 2 or texts.shape[1] != len(keys):  # noqa: PLR2004  # an (n, k) table
        msg = f"texts of shape {texts.shape} do not match {len(keys)} item_keys"
        raise ValueError(msg)
    if response_ids.shape != (texts.shape[0],):
        msg = f"response_ids of shape {response_ids.shape} do not match {texts.shape[0]} rows"
        raise ValueError(msg)
    unknown = [key for key in coding.reverse if key not in keys]
    if unknown:
        msg = f"reverse names {unknown}, which are not among the item keys {list(keys)}"
        raise MappingError(msg)
    rules = _Rules(
        missing=frozenset(coding.missing_values),
        value_map=dict(coding.value_map),
        number=_NUMBER[coding.decimal],
        decimal=coding.decimal,
    )
    out = np.empty(texts.shape, dtype=np.float64)
    failures: list[str] = []
    for j, key in enumerate(keys):
        failure = _code_column(texts[:, j], out[:, j], rules, key, response_ids)
        if failure:
            failures.append(failure)
    if failures:
        msg = "cannot code these texts as numbers: " + "; ".join(failures)
        hint = "add a value_map entry for label responses, or a missing_values code"
        raise CodingError(msg, hint=hint)
    out += coding.offset
    reversed_columns = [j for j, key in enumerate(keys) if key in coding.reverse]
    out[:, reversed_columns] = scale.minimum + scale.maximum - out[:, reversed_columns]
    return out


@dataclass(frozen=True, slots=True)
class _Rules:
    """Steps 1 to 3 of the coding order, prepared once per item group."""

    missing: frozenset[str]
    value_map: dict[str, float]
    number: re.Pattern[str]
    decimal: str

    def code(self, text: str) -> float | None:
        """Code one stripped text; ``None`` when it cannot be coded."""
        if text in self.missing:
            return np.nan
        if text in self.value_map:
            return self.value_map[text]
        if not self.number.match(text):
            return None
        value = float(text.replace(self.decimal, "."))
        return value if np.isfinite(value) else None


def _code_column(
    column: NDArray[np.object_],
    out: NDArray[np.float64],
    rules: _Rules,
    key: str,
    response_ids: NDArray[np.object_],
) -> str:
    """Code one column into ``out``; return a description of its failures, or ``""``."""
    bad: Counter[str] = Counter()
    bad_rows: list[int] = []
    for i, cell in enumerate(column.tolist()):
        if not isinstance(cell, str):
            msg = f"item {key}: every cell must be a str, got {cell!r}"
            raise TypeError(msg)
        text = cell.strip()
        value = rules.code(text)
        if value is None:
            bad[text] += 1
            bad_rows.append(i)
            value = np.nan
        out[i] = value
    if not bad:
        return ""
    listed = ", ".join(f"{text!r} x{count}" for text, count in bad.items())
    return f"{key}: {listed} (responses {_shown([response_ids[i] for i in bad_rows])})"


# --- TOML ------------------------------------------------------------------------------------


def read_mapping(path: str | os.PathLike[str]) -> SurveyMapping:
    """Read a survey mapping from a TOML file.

    Parameters
    ----------
    path : str or os.PathLike
        The TOML file, UTF-8 encoded.

    Returns
    -------
    SurveyMapping
        The validated, normalised mapping.

    Raises
    ------
    MappingError
        If the file is not valid TOML, holds a blank value (a TODO left in a draft), or
        fails validation; the message keeps pydantic's field paths.
    DataReferenceError
        If the instrument does not ship.
    DataIntegrityError
        If the shipped instrument file does not match its pinned hash.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = parse_toml(text, error=MappingError, source=os.fspath(path))
    _check_no_blanks(data, "")  # before validation, which would hide a blank behind its own error
    return validate(
        SurveyMapping, data, error=MappingError, context=f"invalid mapping {os.fspath(path)}"
    )


def write_mapping(mapping: SurveyMapping, path: str | os.PathLike[str]) -> None:
    """Write a survey mapping as TOML.

    The output is deterministic: the identity keys are always written, other fields at
    their default are omitted (the attribute scale when it equals the instrument's),
    scalars and arrays come before tables, each in field order, and attribute keys are
    written as the instrument's reference-language labels.

    Parameters
    ----------
    mapping : SurveyMapping
        The mapping.
    path : str or os.PathLike
        The file to write, UTF-8 encoded with LF line endings.

    Raises
    ------
    MappingError
        If the mapping holds a blank value, which ``read_mapping()`` would reject.
    """
    circ = circumplex(mapping.instrument, mapping.instrument_version)
    data = _prune(mapping, mapping.model_dump(mode="json", by_alias=True))
    attributes = data["groups"]["attributes"]
    if mapping.groups.attributes.scale == circ.scale:
        del attributes["scale"]
    attributes["columns"] = {circ.label(p): col for p, col in mapping.groups.attributes.columns}
    if "coding" in attributes and "reverse" in attributes["coding"]:
        attributes["coding"]["reverse"] = [
            circ.label(p) for p in mapping.groups.attributes.coding.reverse
        ]
    _check_no_blanks(data, "")
    # tomlkit places plain keys before tables itself when it builds a document.
    text = tomlkit.dumps(data)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def _prune(model: pydantic.BaseModel, dumped: dict[str, Any]) -> dict[str, Any]:
    """Drop dumped fields at their default, keeping the identity keys."""
    out: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        if (
            name not in _IDENTITY_FIELDS
            and not field.is_required()
            and value == field.get_default(call_default_factory=True)
        ):
            continue
        key = field.alias or name
        entry = dumped[key]
        out[key] = _prune(value, entry) if isinstance(value, pydantic.BaseModel) else entry
    return out


def _check_no_blanks(data: Mapping[str, Any], prefix: str) -> None:
    """Raise for a blank string held as a scalar or table value; array entries may be blank."""
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, str) and not value.strip():
            msg = f"{dotted} is blank; fill it in or remove the key"
            raise MappingError(msg)
        if isinstance(value, Mapping):
            _check_no_blanks(value, f"{dotted}.")
