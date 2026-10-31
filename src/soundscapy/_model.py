"""Bottom-layer building blocks: ``FrozenModel``, pair types, TOML I/O, column namespace.

This module imports nothing from the rest of the package except :mod:`soundscapy.errors`.
"""

import enum
import tomllib
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Final, TypeVar

import pydantic
import tomlkit
from pydantic import BeforeValidator, PlainSerializer, StrictFloat, StrictStr, ValidationError

from soundscapy.errors import SettingsError, SoundscapyError

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


def to_toml(model: FrozenModel) -> str:
    """Write a model as TOML text.

    Parameters
    ----------
    model : FrozenModel
        The model to write.

    Returns
    -------
    str
        ``model_dump(mode="json", by_alias=True)`` rendered by tomlkit.
    """
    return tomlkit.dumps(model.model_dump(mode="json", by_alias=True))


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
    """
    try:
        return cls.model_validate(data)
    except ValidationError as err:
        raise error(f"{context}: {err}" if context else str(err)) from err


def from_toml(cls: type[M], text: str, *, error: type[SoundscapyError] = SettingsError) -> M:
    """Read a model from TOML text.

    Parameters
    ----------
    cls : type[FrozenModel]
        The model class to build.
    text : str
        TOML text.
    error : type[SoundscapyError], optional
        The error raised on failure.

    Returns
    -------
    FrozenModel
        An instance of ``cls``.

    Raises
    ------
    SoundscapyError
        ``error`` (by default :class:`SettingsError`), if the text is not valid TOML or
        fails validation; a validation message keeps pydantic's field paths.
    """
    return validate(cls, parse_toml(text, error=error), error=error)
