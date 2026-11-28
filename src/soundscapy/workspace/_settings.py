"""Writing a settings file: the profile it extends and the overrides as sections."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import tomlkit

from soundscapy.errors import SettingsError
from soundscapy.profiles import EXTENDS, SECTIONS, Sections, _resolve_section, read_profile
from soundscapy.projection.project import ProjectionSettings
from soundscapy.survey.read import ImportSettings

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping

    from soundscapy._model import FrozenModel

__all__ = ["save_settings"]

# The settings model of each section an operation has so far; other sections are written
# unchecked.
_SECTION_MODELS: Final[dict[str, type[FrozenModel]]] = {
    "import": ImportSettings,
    "projection": ProjectionSettings,
}


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
        model = _SECTION_MODELS.get(section)
        if model is None:
            continue
        context = f"[{section}] under profile {header.label}"
        _resolve_section(
            model, header, section, sections.get(section, {}), [table], context=context
        )
    document: dict[str, Any] = {EXTENDS: header.label, **nested}
    Path(path).write_text(tomlkit.dumps(document), encoding="utf-8")
