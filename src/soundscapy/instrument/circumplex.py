"""Circumplex instruments: attributes at angles, axes, quadrants and the shipped data."""

import itertools
import re
from typing import Literal, Self

from pydantic import Field, StrictFloat, StrictStr, model_validator

from soundscapy import _data
from soundscapy._model import Citation, FrozenModel, StrPairs, from_toml
from soundscapy.errors import DataReferenceError, MappingError
from soundscapy.instrument.scale import Scale

__all__ = ["Attribute", "Axis", "Circumplex", "Translation", "circumplex"]

_POSITION = re.compile(r"^PAQ[1-9][0-9]*$")
_QUADRANT_KEYS = frozenset({"++", "-+", "--", "+-"})
_FULL_TURN_DEG = 360.0


class Attribute(FrozenModel):
    """One perceptual attribute of a circumplex.

    Attributes
    ----------
    position : str
        Position name, ``PAQ`` followed by a number without leading zero.
    angle_deg : float
        Nominal angle in degrees, in ``[0, 360)``.
    labels : pairs of str
        Label of the attribute in the circumplex's reference language only, keyed by
        ISO 639-3 code. Other languages go in :class:`Translation`.
    """

    position: StrictStr = Field(pattern=_POSITION.pattern)
    angle_deg: StrictFloat
    labels: StrPairs

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 0.0 <= self.angle_deg < _FULL_TURN_DEG:
            msg = f"angle_deg {self.angle_deg} for {self.position} is outside [0, 360)"
            raise ValueError(msg)
        return self


class Axis(FrozenModel):
    """An axis of the circumplex, running from a negative to a positive attribute.

    Attributes
    ----------
    id : str
        Identifier, used in column names (for example ``ISOPleasant``).
    positive, negative : str
        Positions of the attributes at the two ends of the axis.
    labels : pairs of str, optional
        Axis titles by language; empty means the id is the title and the end labels
        come from the attributes.
    """

    id: StrictStr = Field(min_length=1)
    positive: StrictStr
    negative: StrictStr
    labels: StrPairs = ()


class Translation(FrozenModel):
    """Attribute labels of one non-reference language.

    Attributes
    ----------
    language : str
        ISO 639-3 code (or the code the angle sets use).
    labels : pairs of str
        One label per position, keyed by position.
    citation : Citation
        Source of these labels.
    """

    language: StrictStr = Field(min_length=1)
    labels: StrPairs
    citation: Citation = Citation()


class Circumplex(FrozenModel):
    """A circumplex instrument.

    The identity keys ``schema`` and ``schema_version`` are required and always written.
    A changed copy is built with
    ``Circumplex.model_validate({**c.model_dump(), **changes})``.

    Attributes
    ----------
    schema_name : str
        Always ``"soundscapy.circumplex"``; the file key is ``schema``.
    schema_version : int
        Always ``1``.
    name, version : str
        Identity of the instrument.
    reference_language : str, optional
        ISO 639-3 code of the language of the attribute labels.
    attributes : tuple of Attribute
        Positions ``PAQ1``..``PAQn`` in order, with strictly increasing angles.
    axes : tuple of Axis
        The x axis then the y axis.
    scale : Scale
        The default response scale.
    quadrants : pairs of str
        Sign key (``"++"``, ``"-+"``, ``"--"``, ``"+-"``, x sign first) to the position
        whose label names the quadrant.
    translations : tuple of Translation, optional
        Labels in languages other than the reference language.
    citation : Citation
        Source of the instrument.
    """

    model_config = FrozenModel.model_config | {"populate_by_name": True}

    schema_name: Literal["soundscapy.circumplex"] = Field(alias="schema")
    schema_version: Literal[1]
    name: StrictStr
    version: StrictStr
    reference_language: StrictStr = "eng"
    attributes: tuple[Attribute, ...]
    axes: tuple[Axis, Axis]
    scale: Scale
    quadrants: StrPairs
    translations: tuple[Translation, ...] = ()
    citation: Citation = Citation()

    @model_validator(mode="after")
    def _check(self) -> Self:
        self._check_attributes()
        positions = set(self.positions)
        for axis in self.axes:
            for end in (axis.positive, axis.negative):
                if end not in positions:
                    msg = f"axis {axis.id!r} names unknown position {end!r}"
                    raise ValueError(msg)
        keys = [key for key, _ in self.quadrants]
        if set(keys) != _QUADRANT_KEYS:
            msg = f"quadrants must have exactly the keys {sorted(_QUADRANT_KEYS)}, got {keys}"
            raise ValueError(msg)
        for key, position in self.quadrants:
            if position not in positions:
                msg = f"quadrant {key!r} names unknown position {position!r}"
                raise ValueError(msg)
        self._check_translations()
        return self

    def _check_attributes(self) -> None:
        if not self.attributes:
            msg = "a circumplex needs at least one attribute"
            raise ValueError(msg)
        for index, attr in enumerate(self.attributes, start=1):
            if attr.position != f"PAQ{index}":
                msg = (
                    f"positions must run PAQ1..PAQn in order; got {attr.position} at place {index}"
                )
                raise ValueError(msg)
            if tuple(lang for lang, _ in attr.labels) != (self.reference_language,):
                msg = (
                    f"{attr.position} labels must hold the reference language "
                    f"{self.reference_language!r} only"
                )
                raise ValueError(msg)
        angles = [a.angle_deg for a in self.attributes]
        if any(b <= a for a, b in itertools.pairwise(angles)):
            msg = f"angles must be strictly increasing, got {angles}"
            raise ValueError(msg)

    def _check_translations(self) -> None:
        seen: set[str] = set()
        for tr in self.translations:
            if tr.language == self.reference_language:
                msg = f"a translation for the reference language {tr.language!r} is not allowed"
                raise ValueError(msg)
            if tr.language in seen:
                msg = f"repeated translation for {tr.language!r}"
                raise ValueError(msg)
            seen.add(tr.language)
            if tuple(p for p, _ in tr.labels) != self.positions:
                msg = f"translation {tr.language!r} must label every position in order"
                raise ValueError(msg)

    @property
    def positions(self) -> tuple[str, ...]:
        """Position names in order, ``("PAQ1", ..., "PAQn")``."""
        return tuple(a.position for a in self.attributes)

    @property
    def angles_deg(self) -> tuple[float, ...]:
        """Nominal angles in degrees, in position order."""
        return tuple(a.angle_deg for a in self.attributes)

    def label(self, position: str, language: str = "") -> str:
        """Return the label of a position in a language.

        Parameters
        ----------
        position : str
            A position such as ``"PAQ6"``.
        language : str, optional
            ISO 639-3 code; blank means the reference language. A language with no
            translation falls back to the reference language.

        Returns
        -------
        str
            The label.

        Raises
        ------
        MappingError
            If the position does not exist.
        """
        if position not in self.positions:
            msg = f"unknown position {position!r}; positions are {self.positions}"
            raise MappingError(msg)
        if language and language != self.reference_language:
            for tr in self.translations:
                if tr.language == language:
                    return dict(tr.labels)[position]
        return dict(self.attributes[self.positions.index(position)].labels)[self.reference_language]

    def position_for(self, key: str) -> str:
        """Resolve a position name or reference-language label to a position.

        Parameters
        ----------
        key : str
            A position (``"PAQ3"``) or a reference label (``"Calm"``), case-insensitive.

        Returns
        -------
        str
            The position.

        Raises
        ------
        MappingError
            If the key matches no position and no reference label.
        """
        wanted = key.casefold()
        for position in self.positions:
            if wanted in {position.casefold(), self.label(position).casefold()}:
                return position
        msg = f"{key!r} is neither a position nor a {self.reference_language} label"
        raise MappingError(msg, hint=f"positions: {self.positions}")


def circumplex(name: str = "iso-12913-outdoor", version: str = "1") -> Circumplex:
    """Load a shipped circumplex instrument.

    Parameters
    ----------
    name : str, optional
        Instrument name.
    version : str, optional
        Instrument version.

    Returns
    -------
    Circumplex
        The instrument, read from a hash-verified data file.

    Raises
    ------
    DataReferenceError
        If no such instrument ships; the message lists the ones that do.
    DataIntegrityError
        If the shipped file does not match its pin.
    """
    relpath = f"circumplexes/{name}/{version}.toml"
    if relpath not in _data.PINS:
        shipped = sorted(
            p.removeprefix("circumplexes/").removesuffix(".toml").replace("/", "@")
            for p in _data.PINS
            if p.startswith("circumplexes/")
        )
        msg = f"no circumplex {name!r} version {version!r}; shipped: {shipped}"
        raise DataReferenceError(msg)
    return from_toml(Circumplex, _data.read_bytes(relpath).decode("utf-8"))
