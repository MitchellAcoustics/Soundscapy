"""Profiles, user settings files and the one settings-resolution rule (spec 6.3.4).

A profile is a versioned, hash-pinned data file, ``_data/profiles/<id>/<version>.toml``,
holding one section of defaults per operation. Every operation resolves its settings
through ``resolve_settings``: the base profile's section, then the section of a user
settings file, then a passed settings model or mapping, then explicit keywords, merged and
validated once. The result records which profile produced the numbers and every field that
differs from it.

This module sits in the bottom layer beside ``_model`` and ``_data``; each operation
validates its own section with its own model, so no operation module is imported here
except lazily by ``save_settings``, which checks the two sections that have models.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, Self, TypeVar

import tomlkit
from pydantic import Field, StrictStr

from soundscapy import _data
from soundscapy._model import (
    _UNSET,
    Citation,
    FrozenModel,
    _layer,
    _Unset,
    merge_settings,
    parse_toml,
    settings_layer,
    validate,
)
from soundscapy.errors import DataReferenceError, ProfileConstraintError, SettingsError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "SECTIONS",
    "ProfileHeader",
    "ProfileRef",
    "ResolvedSettings",
    "list_profiles",
    "read_profile",
    "read_settings_file",
    "resolve_settings",
    "save_settings",
]

SECTIONS: Final = (
    "import",
    "projection",
    "describe",
    "density",
    "contours",
    "item_statistics",
    "correlate",
    "compare_groups",
    "who5",
    "source_profile",
)
"""The section names a profile or settings file may hold, one per operation (spec 6.3.4)."""

DEFAULT_PROFILE: Final = "sal-2026"
"""The profile used when nothing names one."""

_DATA_PREFIX: Final = "profiles/"
_EXTENDS: Final = "extends"
_SUGGESTION: Final = f'pass profile="{DEFAULT_PROFILE}" to change it'

M = TypeVar("M", bound=FrozenModel)
Sections = dict[str, dict[str, Any]]
"""Section name to its raw table."""


# --- Models ---------------------------------------------------------------------------------------


class ProfileHeader(FrozenModel):
    """Everything in a profile file except its operation sections.

    Attributes
    ----------
    schema_name : str
        Always ``"soundscapy.profile"``; the file key is ``schema``.
    schema_version : int
        Always ``1``.
    id, version : str
        Identity of the profile; the label is ``id@version``.
    locked : tuple of str, optional
        Dotted keys (``"projection.formulation"``) whose resolved value must equal the
        profile's; a key names a field or a whole table.
    notes : str, optional
        What the profile follows and any reading it depends on.
    citation : Citation
        The source the profile reproduces.

    References
    ----------
    ``sal-2026`` follows [@mitchell2026translation]; ``iso-12913-3-2019`` follows
    [@iso12913-3-2019]; ``iso-12913-3-2024`` reads the 2024 revision of that standard,
    pending the owner's check of the text.
    """

    model_config = FrozenModel.model_config | {"populate_by_name": True}

    schema_name: Literal["soundscapy.profile"] = Field(alias="schema")
    schema_version: Literal[1]
    id: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)
    locked: tuple[StrictStr, ...] = ()
    notes: StrictStr = ""
    citation: Citation = Citation()

    @property
    def label(self) -> str:
        """``id@version``."""
        return f"{self.id}@{self.version}"


class ProfileRef(FrozenModel):
    """The profile a result was computed under, as provenance records it.

    Attributes
    ----------
    id, version : str
        Identity of the profile.
    hash : str
        The ``h1`` hash of the profile file, its ``_data`` pin.
    """

    id: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)
    hash: StrictStr = Field(min_length=1)

    @property
    def label(self) -> str:
        """``id@version``."""
        return f"{self.id}@{self.version}"

    @classmethod
    def from_record(cls, value: Mapping[str, object] | Self) -> Self:
        """Accept an instance or the ``{id, version, hash}`` mapping provenance stores."""
        if isinstance(value, cls):
            return value
        return validate(cls, value, context="an inherited profile record")


@dataclasses.dataclass(frozen=True)
class ResolvedSettings(Generic[M]):
    """What ``resolve_settings`` returns.

    Attributes
    ----------
    settings : FrozenModel
        The operation's validated settings.
    profile : ProfileRef
        The base profile; ``profile.model_dump()`` is the provenance record.
    profile_inherited : ProfileRef or None
        The input's profile when an explicit, different ``profile=`` replaced it.
    overrides : dict
        Dotted key to JSON value for every field that differs from the base profile, such
        as ``{"projection.angles.on_uncovered": "exclude"}``.
    """

    settings: M
    profile: ProfileRef
    profile_inherited: ProfileRef | None
    overrides: dict[str, Any]


# --- Shipped profiles -----------------------------------------------------------------------------


def _shipped() -> Iterator[tuple[str, str]]:
    """Yield ``(id, version)`` of every pinned profile file."""
    for relpath in _data.PINS:
        if relpath.startswith(_DATA_PREFIX):
            ident, _, version = relpath.removeprefix(_DATA_PREFIX).partition("/")
            yield ident, version.removesuffix(".toml")


def list_profiles() -> tuple[str, ...]:
    """List the shipped profiles.

    Returns
    -------
    tuple of str
        The sorted ``id@version`` labels.
    """
    return tuple(sorted(f"{ident}@{version}" for ident, version in _shipped()))


def _version_key(version: str) -> tuple[int, ...]:
    """Order dotted numeric versions part by part, so ``"1.10"`` is after ``"1.9"``."""
    return tuple(int(part) for part in version.split("."))


def _latest(ident: str) -> str:
    versions = [version for shipped, version in _shipped() if shipped == ident]
    if not versions:
        msg = f"no profile {ident!r}; shipped: {list(list_profiles())}"
        raise DataReferenceError(msg)
    return max(versions, key=_version_key)


def _parse_ref(ref: str) -> tuple[str, str]:
    """Split ``id@version`` or a bare id, which means the latest shipped version."""
    ident, sep, version = ref.partition("@")
    if not ident:
        msg = f"a profile is named 'id' or 'id@version', got {ref!r}"
        raise DataReferenceError(msg)
    if not sep:
        return ident, _latest(ident)
    if f"{_DATA_PREFIX}{ident}/{version}.toml" not in _data.PINS:
        msg = f"no profile {ident!r} version {version!r}; shipped: {list(list_profiles())}"
        raise DataReferenceError(msg)
    return ident, version


def read_profile(ref: str) -> tuple[ProfileHeader, Sections, str]:
    """Read a shipped profile.

    Parameters
    ----------
    ref : str
        ``"id@version"``, or a bare id meaning the latest shipped version. Versions are
        dotted numbers and compare part by part as integers.

    Returns
    -------
    header : ProfileHeader
        The validated header.
    sections : dict
        Section name to its raw table, for the sections the file holds; each operation
        validates its own with its own model.
    hash : str
        The file's ``h1`` pin.

    Raises
    ------
    DataReferenceError
        If no such profile ships; the message lists the ones that do.
    DataIntegrityError
        If the shipped file does not match its pin.
    """
    ident, version = _parse_ref(ref)
    relpath = f"{_DATA_PREFIX}{ident}/{version}.toml"
    raw = _data.read_toml(relpath)
    sections: Sections = {key: raw.pop(key) for key in SECTIONS if key in raw}
    header = validate(ProfileHeader, raw, context=f"profile {relpath}")
    return header, sections, _data.pin(relpath)


# --- Settings files -------------------------------------------------------------------------------


def read_settings_file(path: str | os.PathLike[str]) -> tuple[str, Sections]:
    """Read a user settings file: ``extends`` plus operation sections.

    Parameters
    ----------
    path : str or os.PathLike
        A TOML file, UTF-8 encoded, as ``save_settings`` writes it.

    Returns
    -------
    extends : str
        The profile the file extends, ``"id@version"`` or a bare id.
    sections : dict
        Section name to its table, for the sections the file holds.

    Raises
    ------
    SettingsError
        If the file does not exist, is not valid TOML, has no ``extends``, has a top-level
        key that is not ``extends`` or a section name, or a section that is not a table.
        Every message names the path.
    """
    file = Path(path)
    name = os.fspath(path)
    if not file.is_file():
        msg = f"settings file {name!r} does not exist"
        raise SettingsError(msg)
    raw = parse_toml(file.read_text(encoding="utf-8"), source=f"settings file {name}")
    unknown = sorted(key for key in raw if key != _EXTENDS and key not in SECTIONS)
    if unknown:
        msg = (
            f"settings file {name} has unknown top-level keys {unknown}; sections: {list(SECTIONS)}"
        )
        raise SettingsError(msg)
    extends = raw.pop(_EXTENDS, None)
    if not isinstance(extends, str):
        what = "no 'extends'" if extends is None else f"extends = {extends!r}, not a string"
        msg = (
            f"settings file {name} has {what}; it must name the profile it extends, one of "
            f"{list(list_profiles())}; save_settings writes such a file"
        )
        raise SettingsError(msg)
    for key, table in raw.items():
        if not isinstance(table, dict):
            msg = f"settings file {name}: [{key}] must be a table, got {table!r}"
            raise SettingsError(msg)
    return extends, raw


# --- Resolution -----------------------------------------------------------------------------------


def _dotted(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _diff(base: Any, resolved: Any, prefix: str, out: dict[str, Any]) -> None:  # noqa: ANN401  # JSON values
    """Record where ``resolved`` differs from ``base``, recursing into tables of the same shape."""
    if isinstance(base, dict) and isinstance(resolved, dict) and base.keys() == resolved.keys():
        for key in base:
            _diff(base[key], resolved[key], _dotted(prefix, key), out)
    elif base != resolved:
        out[prefix] = resolved


def _lookup(table: Mapping[str, Any], dotted: str) -> object:
    """Follow a dotted key through nested tables; ``_UNSET`` where the path does not exist."""
    current: object = table
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _UNSET
        current = current[part]
    return current


def _check_locked(
    header: ProfileHeader, section: str, base: Mapping[str, Any], resolved: Mapping[str, Any]
) -> None:
    """Raise if a locked key under ``section`` resolved to another value than the profile's."""
    for key in header.locked:
        head, _, rest = key.partition(".")
        if head != section:
            continue
        expected = _lookup(base, rest)
        actual = _lookup(resolved, rest)
        if actual != expected:
            msg = (
                f"{key} is locked by profile {header.label} to {expected!r}, got {actual!r}; "
                f"{_SUGGESTION}"
            )
            raise ProfileConstraintError(msg)


def _dump(model: FrozenModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True)


def _base_label(profile: str | _Unset, inherited: ProfileRef | None, extends: str | None) -> str:
    # An explicit settings file's extends ranks with an explicit profile (spec 11.4), so it
    # beats the profile the input recorded.
    if profile is not _UNSET:
        return profile
    if extends is not None:
        return extends
    return inherited.label if inherited is not None else DEFAULT_PROFILE


def resolve_settings(
    model: type[M],
    section: str,
    *,
    keywords: Mapping[str, object],
    settings: object | _Unset = _UNSET,
    profile: str | _Unset = _UNSET,
    inherited: Mapping[str, object] | ProfileRef | None = None,
) -> ResolvedSettings[M]:
    """Resolve one operation's settings by the one precedence rule (spec 6.3.4).

    The base profile is ``profile``, else the ``extends`` of a settings file passed as
    ``settings`` (an explicit file ranks with an explicit profile), else ``inherited``, else
    ``sal-2026``. Then each field takes the first value set
    in this order: a keyword, ``settings``, the file's section, the profile's section, the
    model default; the layers merge table by table and the result validates once.

    Parameters
    ----------
    model : type[FrozenModel]
        The operation's settings model.
    section : str
        The operation's section name, one of ``SECTIONS``.
    keywords : mapping
        Field name to the explicit keyword value; a value of ``_UNSET`` changes nothing. A
        nested model instance contributes only its set fields.
    settings : object, optional
        An instance of ``model`` (contributing only its set fields), a mapping of its
        fields, or the path of a settings file (``str`` or ``os.PathLike``).
    profile : str, optional
        ``"id@version"`` or a bare id meaning the latest shipped version. It wins over the
        file's ``extends``, whose sections still apply.
    inherited : mapping or ProfileRef, optional
        The profile recorded in the input's provenance, ``{id, version, hash}``.

    Returns
    -------
    ResolvedSettings
        The validated settings, the profile, the replaced inherited profile (or None) and
        the overrides: every field whose value differs from the profile's section merged
        onto the model defaults.

    Raises
    ------
    DataReferenceError
        If the profile does not ship.
    SettingsError
        If ``settings`` has the wrong type, the settings file is invalid, or the merged
        settings fail validation; the message keeps the field path.
    ProfileConstraintError
        If a field the profile locks resolved to another value than the profile's; the
        message names the field and suggests ``profile="sal-2026"``.
    """
    file_sections: Sections = {}
    extends: str | None = None
    if isinstance(settings, str | os.PathLike):
        extends, file_sections = read_settings_file(settings)
    previous = None if inherited is None else ProfileRef.from_record(inherited)
    header, sections, digest = read_profile(_base_label(profile, previous, extends))
    ref = ProfileRef(id=header.id, version=header.version, hash=digest)

    # Each merge gets its own copy of the profile section: merging adopts nested tables.
    profile_section = sections.get(section, {})
    base = merge_settings(
        model, [_layer(profile_section)], context=f"profile {ref.label} [{section}]"
    )
    layers: list[Mapping[str, Any]] = [
        _layer(profile_section),
        _layer(file_sections.get(section, {})),
    ]
    if settings is not _UNSET and extends is None:
        layers.append(settings_layer(settings, model=model, name="settings"))
    layers.append({key: _layer(value) for key, value in keywords.items() if value is not _UNSET})
    resolved = merge_settings(model, layers, context=f"{section} settings")

    base_dump, resolved_dump = _dump(base), _dump(resolved)
    _check_locked(header, section, base_dump, resolved_dump)
    overrides: dict[str, Any] = {}
    _diff(base_dump, resolved_dump, section, overrides)
    replaced = previous if previous is not None and previous != ref else None
    return ResolvedSettings(resolved, ref, replaced, overrides)


# --- Saving ---------------------------------------------------------------------------------------


def _section_model(section: str) -> type[FrozenModel] | None:
    """Give the settings model of a section, for the operations that have one so far.

    The operation modules sit above this one, so they are imported here, when a file is
    saved, rather than at import time.
    """
    if section == "import":
        from soundscapy.survey.read import ImportSettings  # noqa: PLC0415  # layered above

        return ImportSettings
    if section == "projection":
        from soundscapy.projection.project import ProjectionSettings  # noqa: PLC0415  # layered above

        return ProjectionSettings
    return None


def _nest(overrides: Mapping[str, Any]) -> Sections:
    """Turn dotted override keys into nested section tables."""
    out: Sections = {}
    for dotted, value in overrides.items():
        section, _, rest = dotted.partition(".")
        if section not in SECTIONS or not rest:
            msg = (
                f"override {dotted!r} must be '<section>.<field>' with a section in "
                f"{list(SECTIONS)}"
            )
            raise SettingsError(msg)
        table: dict[str, Any] = out.setdefault(section, {})
        *parents, leaf = rest.split(".")
        for part in parents:
            table = table.setdefault(part, {})
        table[leaf] = value
    return out


def save_settings(
    path: str | os.PathLike[str], *, extends: str, overrides: Mapping[str, Any]
) -> None:
    """Write a settings file: the profile it extends and the overrides as sections.

    Parameters
    ----------
    path : str or os.PathLike
        Where to write; an existing file is overwritten.
    extends : str
        ``"id@version"`` or a bare id, written as ``id@version`` of the latest version.
    overrides : mapping
        Dotted key to value, as ``ResolvedSettings.overrides`` holds them; each becomes a
        key in its section's table.

    Raises
    ------
    DataReferenceError
        If the profile does not ship.
    SettingsError
        If a key names no section, or a section with a settings model fails validation
        when merged onto the profile. A section without a model yet is written unchecked.
    ProfileConstraintError
        If an override changes a field the profile locks, as the operation would refuse it.

    References
    ----------
    Spec 6.3.4: a settings file holds ``extends`` and only the overrides.
    """
    header, sections, _ = read_profile(extends)
    nested = _nest(overrides)
    for section, table in nested.items():
        model = _section_model(section)
        if model is None:
            continue
        profile_section = sections.get(section, {})
        context = f"[{section}] under profile {header.label}"
        base = merge_settings(model, [_layer(profile_section)], context=context)
        resolved = merge_settings(model, [_layer(profile_section), _layer(table)], context=context)
        _check_locked(header, section, _dump(base), _dump(resolved))
    document: dict[str, Any] = {_EXTENDS: header.label, **nested}
    Path(path).write_text(tomlkit.dumps(document), encoding="utf-8")
