"""Which attribute angles a projection uses: angle sets, their references and the selection."""

from __future__ import annotations

import functools
import itertools
import os
import re
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BeforeValidator,
    Field,
    PlainSerializer,
    StrictFloat,
    StrictStr,
    model_validator,
)

from soundscapy import _data
from soundscapy._model import Citation, FrozenModel, _to_pairs, _to_table, from_toml
from soundscapy.errors import DataReferenceError, SettingsError

__all__ = [
    "AngleSelection",
    "AngleSet",
    "AngleSetRef",
    "LanguageAngles",
    "angle_set",
    "read_angle_set",
]

_LANGUAGE_CODE = re.compile(r"^[a-z]{3}$")
_FULL_TURN_DEG = 360.0
_DATA_PREFIX: Final = "angle_sets/"
Tier = Literal["high", "medium", "low"]


class LanguageAngles(FrozenModel):
    """What an angle set says about one language.

    Attributes
    ----------
    tier : {"high", "medium", "low"}
        The source's confidence in the language's translation of the attributes.
    angles_deg : tuple of float, optional
        Adjusted attribute angles in degrees, in position order, each in ``[0, 360)`` and
        strictly increasing. Empty for a tier-only language, which the source assessed
        without publishing angles; a ``high`` language always has angles.
    note : str, optional
        Where the source's reading needs a remark, such as a table that prints another
        tier than the paper's stated rule gives.
    """

    tier: Tier
    angles_deg: tuple[StrictFloat, ...] = ()
    note: StrictStr = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.tier == "high" and not self.angles_deg:
            msg = "a high-tier language must have angles_deg"
            raise ValueError(msg)
        for angle in self.angles_deg:
            if not 0.0 <= angle < _FULL_TURN_DEG:
                msg = f"angle {angle} is outside [0, 360)"
                raise ValueError(msg)
        if any(b <= a for a, b in itertools.pairwise(self.angles_deg)):
            msg = f"angles_deg must be strictly increasing, got {list(self.angles_deg)}"
            raise ValueError(msg)
        return self

    @property
    def has_angles(self) -> bool:
        """Whether the language has published angles (``False`` for a tier-only language)."""
        return bool(self.angles_deg)


LanguagePairs = Annotated[
    tuple[tuple[str, LanguageAngles], ...],
    BeforeValidator(_to_pairs),
    PlainSerializer(_to_table, when_used="json"),
]
"""Language code to ``LanguageAngles`` pairs, read from and written as ``[languages.<code>]``."""


class AngleSet(FrozenModel):
    """Published, language-specific attribute angles for one circumplex.

    The file form is ``_data/angle_sets/<name>/<version>.toml``; a user's set uses the same
    format and is read with ``read_angle_set``. Identity is ``(name, version)`` together with
    the circumplex it applies to; whether the set matches a survey's circumplex and angle
    count is checked where the two meet, not here. The identity keys ``schema`` and
    ``schema_version`` are required and always written.

    Attributes
    ----------
    schema_name : str
        Always ``"soundscapy.angle_set"``; the file key is ``schema``.
    schema_version : int
        Always ``1``.
    name, version : str
        Identity of the set; ``name`` is never ``"equal"``, which names the circumplex's
        nominal angles in an ``AngleSelection``.
    circumplex, circumplex_version : str
        The circumplex whose positions the angles belong to.
    citation : Citation
        Source of the angles and tiers.
    languages : pairs of str to LanguageAngles
        One entry per ISO 639-3 code (``^[a-z]{3}$``). Every entry with angles has the
        same number of them.

    References
    ----------
    Aletta et al. (2024) publish such a set for eighteen languages [@aletta2024soundscape];
    Mitchell et al. (2026) revise it [@mitchell2026translation].
    """

    model_config = FrozenModel.model_config | {"populate_by_name": True}

    schema_name: Literal["soundscapy.angle_set"] = Field(alias="schema")
    schema_version: Literal[1]
    name: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)
    circumplex: StrictStr = Field(min_length=1)
    circumplex_version: StrictStr = Field(min_length=1)
    citation: Citation = Citation()
    languages: LanguagePairs

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.name == "equal":
            msg = "an angle set cannot be named 'equal': that names the nominal angles"
            raise ValueError(msg)
        for code, _ in self.languages:
            if not _LANGUAGE_CODE.match(code):
                msg = f"language key {code!r} is not a three-letter lowercase ISO 639-3 code"
                raise ValueError(msg)
        counts = {len(entry.angles_deg) for _, entry in self.languages if entry.has_angles}
        if len(counts) > 1:
            msg = f"every language with angles must have the same number of angles, got {counts}"
            raise ValueError(msg)
        return self

    @property
    def label(self) -> str:
        """``name@version``."""
        return f"{self.name}@{self.version}"

    @property
    def by_language(self) -> dict[str, LanguageAngles]:
        """The entries as a dict keyed by language code."""
        return dict(self.languages)


class AngleSetRef(FrozenModel):
    """A reference to an angle set by name and version, as settings hold it.

    Attributes
    ----------
    name, version : str
        Identity of the set. The string form, ``label``, always carries the version.
    """

    name: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)

    @property
    def label(self) -> str:
        """``name@version``."""
        return f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse a ``name@version`` label.

        Parameters
        ----------
        text : str
            The label, such as ``"satp@1.5"``; a bare name is not accepted.

        Returns
        -------
        AngleSetRef
            The reference.

        Raises
        ------
        DataReferenceError
            If ``text`` has no version; the message lists the shipped labels.
        """
        name, sep, version = text.partition("@")
        if not (name and sep and version):
            msg = f"an angle set is named 'name@version', got {text!r}"
            msg = f"{msg}; shipped: {_shipped_labels()}"
            raise DataReferenceError(msg)
        return cls(name=name, version=version)


def _shipped_labels() -> list[str]:
    return sorted(
        p.removeprefix(_DATA_PREFIX).removesuffix(".toml").replace("/", "@")
        for p in _data.PINS
        if p.startswith(_DATA_PREFIX)
    )


@functools.cache
def angle_set(name: str, version: str) -> AngleSet:
    """Load a shipped angle set.

    The set is immutable, so each name and version is read and verified once per process
    and the same object returned after that.

    Parameters
    ----------
    name : str
        Set name, such as ``"satp"``.
    version : str
        Set version, such as ``"1.5"``.

    Returns
    -------
    AngleSet
        The set, read from a hash-verified data file.

    Raises
    ------
    DataReferenceError
        If no such set ships; the message lists the ones that do.
    DataIntegrityError
        If the shipped file does not match its pin.

    References
    ----------
    ``aletta-2024@1`` is Table 4 of [@aletta2024soundscape]; ``satp@1.5`` is Tables 4.7 and
    4.8 of [@mitchell2026translation].
    """
    relpath = f"{_DATA_PREFIX}{name}/{version}.toml"
    if relpath not in _data.PINS:
        msg = f"no angle set {name!r} version {version!r}; shipped: {_shipped_labels()}"
        raise DataReferenceError(msg)
    return from_toml(AngleSet, _data.read_bytes(relpath).decode("utf-8"))


def read_angle_set(path: str | os.PathLike[str]) -> AngleSet:
    """Read a user's angle set from a TOML file in the shipped format.

    Parameters
    ----------
    path : str or os.PathLike
        The TOML file, UTF-8 encoded.

    Returns
    -------
    AngleSet
        The validated set.

    Raises
    ------
    DataReferenceError
        If the file does not exist.
    SettingsError
        If the file is not valid TOML or fails validation; the message names the file and
        keeps pydantic's field paths.
    """
    file = Path(path)
    if not file.is_file():
        msg = f"angle set file {os.fspath(path)!r} does not exist"
        raise DataReferenceError(msg)
    text = file.read_text(encoding="utf-8")
    try:
        return from_toml(AngleSet, text, error=SettingsError)
    except SettingsError as err:
        msg = f"invalid angle set {os.fspath(path)}: {err}"
        raise SettingsError(msg, hint=err.hint) from err


class AngleSelection(FrozenModel):
    """The angle set each response is projected with.

    Attributes
    ----------
    default : {"equal"}, optional
        The angle set of every response. ``"equal"`` uses the circumplex's nominal angles
        and does not consult the response's language. Named, language-specific angle sets
        are not available in this version.
    """

    default: Literal["equal"] = "equal"
