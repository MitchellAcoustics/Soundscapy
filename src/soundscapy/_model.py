"""Bottom-layer building blocks: ``FrozenModel``, pair types, TOML I/O, column namespace.

This module imports nothing from the rest of the package except ``soundscapy.errors``.
"""

from __future__ import annotations

import enum
import os
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, TypeVar

import pydantic
import tomlkit
from pydantic import BeforeValidator, PlainSerializer, StrictFloat, StrictStr, ValidationError

from soundscapy.errors import (
    DataIntegrityError,
    DataReferenceError,
    MappingError,
    SettingsError,
    SoundscapyError,
)

# --- Derived-column namespace (spec 6.3.3) -----------------------------------------------------

SSP_PREFIX: Final = "ssp_"
COL_RESPONSE_ID: Final = "ssp_response_id"
COL_INCLUDED: Final = "ssp_included"
COL_REASONS: Final = "ssp_reasons"
COL_FLAGS: Final = "ssp_flags"
COL_LANGUAGE: Final = "ssp_language"
COL_ANGLE_SET: Final = "ssp_angle_set"


def axis_column(axis_id: str) -> str:
    """Name the derived column holding a circumplex axis.

    Parameters
    ----------
    axis_id : str
        Axis identifier, for example ``"ISOPleasant"``.

    Returns
    -------
    str
        ``"ssp_<axis_id>"``.
    """
    return f"{SSP_PREFIX}{axis_id}"


def versioned_label(name: str, version: str) -> str:
    """Label a versioned object, such as an angle set or a profile.

    Parameters
    ----------
    name : str
        The name or id.
    version : str
        The version.

    Returns
    -------
    str
        ``"<name>@<version>"``.
    """
    return f"{name}@{version}"


# --- Sentinel for "argument not given" (spec 6.3.4) ----------------------------------------------


class _Unset(enum.Enum):
    """Type of the ``_UNSET`` sentinel; an enum so ``x is _UNSET`` narrows for type checkers."""

    UNSET = enum.auto()


_UNSET: Final = _Unset.UNSET

# --- FrozenModel (spec 6.2.1) --------------------------------------------------------------------


class FrozenModel(pydantic.BaseModel):
    """Immutable, closed pydantic model; ±inf and NaN are rejected in every float field."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Citation(FrozenModel):
    """Citation block of a shipped data file (spec 6.3.7).

    Every field is an empty string when unknown.

    Attributes
    ----------
    citation : str
        Full reference text.
    doi_or_url : str
        DOI or URL of the source.
    licence : str
        Licence of the data.
    date : str
        Date of the source.
    notes : str
        Free-form notes.
    """

    citation: StrictStr = ""
    doi_or_url: StrictStr = ""
    licence: StrictStr = ""
    date: StrictStr = ""
    notes: StrictStr = ""


# --- Key-to-value pair types (spec 6.2.1, 6.4.4) -------------------------------------------------


def _to_pairs(value: object) -> object:
    """Turn a table (mapping) into a tuple of pairs; reject a repeated key in either form."""
    if isinstance(value, Mapping):
        return tuple(value.items())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        pairs = tuple(value)
        seen: set[object] = set()
        for pair in pairs:
            if not isinstance(pair, tuple | list) or len(pair) != 2:  # noqa: PLR2004  # a pair
                return pairs  # let the type validation report the malformed entry
            key = pair[0]
            if key in seen:
                msg = f"repeated key {key!r}"
                raise ValueError(msg)  # a value error, not a type error
            seen.add(key)
        return pairs
    return value


def _to_table(pairs: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Serialise pairs as a table, converting inner tuples to lists for TOML."""
    return {key: list(val) if isinstance(val, tuple) else val for key, val in pairs}


StrPairs = Annotated[
    tuple[tuple[str, str], ...],
    BeforeValidator(_to_pairs),
    PlainSerializer(_to_table, when_used="json"),
]
"""``tuple[tuple[str, str], ...]`` read from and written as a TOML table."""

StrTuplePairs = Annotated[
    tuple[tuple[str, tuple[str, ...]], ...],
    BeforeValidator(_to_pairs),
    PlainSerializer(_to_table, when_used="json"),
]
"""Key to tuple-of-strings pairs (scale labels) read from and written as a TOML table."""

FloatPairs = Annotated[
    tuple[tuple[str, StrictFloat], ...],
    BeforeValidator(_to_pairs),
    PlainSerializer(_to_table, when_used="json"),
]
"""Key to finite-float pairs (value maps) read from and written as a TOML table."""

# --- TOML (spec 6.2.1, 6.4.4) --------------------------------------------------------------------

M = TypeVar("M", bound=FrozenModel)


def dump_json(model: FrozenModel) -> dict[str, Any]:
    """Dump a model in its file form: JSON-compatible values under the file keys.

    Parameters
    ----------
    model : FrozenModel
        The model to dump.

    Returns
    -------
    dict
        ``model.model_dump(mode="json", by_alias=True)``.
    """
    return model.model_dump(mode="json", by_alias=True)


def to_toml(model: FrozenModel) -> str:
    """Write a model as TOML text.

    Parameters
    ----------
    model : FrozenModel
        The model to write.

    Returns
    -------
    str
        ``dump_json(model)`` rendered by tomlkit.
    """
    return tomlkit.dumps(dump_json(model))


def parse_toml(
    text: str, *, error: type[SoundscapyError] = SettingsError, source: str = ""
) -> dict[str, Any]:
    """Parse TOML text into a table.

    Parameters
    ----------
    text : str
        TOML text.
    error : type[SoundscapyError], optional
        The error raised for invalid TOML.
    source : str, optional
        Where the text came from, such as a file path; it starts the message.

    Returns
    -------
    dict
        The parsed table.

    Raises
    ------
    SoundscapyError
        ``error``, if the text is not valid TOML.
    """
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        msg = f"{source} is not valid TOML: {err}" if source else f"not valid TOML: {err}"
        raise error(msg) from err


def validate(
    cls: type[M], data: object, *, error: type[SoundscapyError] = SettingsError, context: str = ""
) -> M:
    """Validate data into a model, raising a soundscapy error on failure.

    Parameters
    ----------
    cls : type[FrozenModel]
        The model class to build.
    data : object
        The data, such as a parsed TOML table.
    error : type[SoundscapyError], optional
        The error raised when validation fails.
    context : str, optional
        What is validated; it starts the message as ``"<context>: "``.

    Returns
    -------
    FrozenModel
        An instance of ``cls``.

    Raises
    ------
    SoundscapyError
        ``error``, if validation fails; the message keeps pydantic's field paths.
    DataIntegrityError, DataReferenceError, MappingError
        Unchanged, when a validator raised one (for example while loading the
        instrument a mapping names); pydantic would otherwise wrap it.
    """
    try:
        return cls.model_validate(data)
    except ValidationError as err:
        failure = err
    # Outside the except block, so the original keeps its own cause and context.
    original = _passthrough_error(failure)
    if original is not None:
        raise original
    raise error(f"{context}: {failure}" if context else str(failure)) from failure


# Errors a validator can raise that are not about the validated data's shape: they reach
# the caller as raised, not as a field error of the model.
_PASSTHROUGH: Final = (DataIntegrityError, DataReferenceError, MappingError)


def _passthrough_error(err: ValidationError) -> SoundscapyError | None:
    """Return the first error of a ``_PASSTHROUGH`` class that ``err`` wraps."""
    for detail in err.errors():
        original = detail.get("ctx", {}).get("error")
        if isinstance(original, _PASSTHROUGH):
            return original
    return None


def from_toml(
    cls: type[M], text: str, *, error: type[SoundscapyError] = SettingsError, context: str = ""
) -> M:
    """Read a model from TOML text.

    Parameters
    ----------
    cls : type[FrozenModel]
        The model class to build.
    text : str
        TOML text.
    error : type[SoundscapyError], optional
        The error raised on failure.
    context : str, optional
        What is read, such as ``"angle set <path>"``; it starts every message.

    Returns
    -------
    FrozenModel
        An instance of ``cls``.

    Raises
    ------
    SoundscapyError
        ``error`` (by default ``SettingsError``), if the text is not valid TOML or
        fails validation; a validation message keeps pydantic's field paths.
    """
    table = parse_toml(text, error=error, source=context)
    return validate(cls, table, error=error, context=context)


def read_user_file(path: str | os.PathLike[str], *, what: str, error: type[SoundscapyError]) -> str:
    """Read a user's text file, UTF-8 encoded.

    Parameters
    ----------
    path : str or os.PathLike
        The file.
    what : str
        What the file holds, such as ``"settings file"``; it starts the error message.
    error : type[SoundscapyError]
        The error raised when the file does not exist.

    Returns
    -------
    str
        The file's text.

    Raises
    ------
    SoundscapyError
        ``error``, if ``path`` is not a file; the message names the path.
    """
    file = Path(path)
    if not file.is_file():
        msg = f"{what} {os.fspath(path)!r} does not exist"
        raise error(msg)
    return file.read_text(encoding="utf-8")


# --- Settings layers (spec 6.3.4, ranks 1 and 2) -------------------------------------------------


def settings_layer(value: object, *, model: type[FrozenModel], name: str) -> dict[str, Any]:
    """Turn a settings argument into one layer for ``merge_settings()``.

    Parameters
    ----------
    value : object
        An instance of ``model``, or a mapping of its fields. A nested mapping becomes a
        ``dict``; a nested ``FrozenModel`` instance becomes its set fields, so
        ``{"quality": QualitySettings(min_age=16.0)}`` merges leaf by leaf.
    model : type[FrozenModel]
        The settings model the layer is for.
    name : str
        The argument's name, for the error message.

    Returns
    -------
    dict
        A fresh dict: an instance contributes only the fields that were set
        (``exclude_unset``), in JSON form; a mapping is deep-copied.

    Raises
    ------
    SettingsError
        If ``value`` is neither an instance of ``model`` nor a mapping.
    """
    if isinstance(value, model) or isinstance(value, Mapping):  # noqa: SIM101  # `model` is a variable, not a class literal
        return _layer(value)
    msg = (
        f"{name} must be a {model.__name__} or a mapping of its fields, got {type(value).__name__}"
    )
    raise SettingsError(msg)


def _layer(value: Any) -> Any:  # noqa: ANN401  # a settings value of any JSON-like type
    """Copy a layer: a model becomes its set fields, a mapping a fresh dict, recursively."""
    if isinstance(value, FrozenModel):
        return value.model_dump(mode="json", by_alias=True, exclude_unset=True)
    if isinstance(value, Mapping):
        return {key: _layer(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], layer: dict[str, Any]) -> None:
    """Merge ``layer``, a fresh copy from ``_layer``, into ``base``; a later value wins."""
    for key, item in layer.items():
        current = base.get(key)
        if isinstance(item, dict) and isinstance(current, dict):
            _deep_merge(current, item)
        else:
            base[key] = item


def merge_settings(model: type[M], layers: Sequence[Mapping[str, Any]], *, context: str) -> M:
    """Merge settings layers, a later layer winning, and validate the result.

    Parameters
    ----------
    model : type[FrozenModel]
        The settings model to build.
    layers : sequence of mapping
        Layers from ``settings_layer()``, lowest precedence first; each is copied before
        it is merged, so the caller's tables are never adopted. Nested mappings merge key
        by key; any other value replaces the earlier one.
    context : str
        What is validated; it starts the error message.

    Returns
    -------
    FrozenModel
        An instance of ``model``.

    Raises
    ------
    SettingsError
        If the merged settings fail validation; the message keeps pydantic's field paths.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        _deep_merge(merged, _layer(layer))
    return validate(model, merged, error=SettingsError, context=context)
