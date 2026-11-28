"""Per-response angle resolution: languages and an ``AngleSelection`` to angles and outcomes.

A pure function over arrays. It never imports the survey layer: ``project`` turns the
masks it returns into status reasons and flags, raises for ``error`` outcomes and warns
for the others, so the resolution rule itself stays testable on plain arrays.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from soundscapy.errors import DataReferenceError
from soundscapy.instrument.angles import AngleSelection, AngleSetRef

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from soundscapy.instrument.angles import AngleSet, LanguageAngles, Policy, Tier
    from soundscapy.instrument.circumplex import Circumplex

__all__ = ["LanguageOutcome", "ResolvedAngles", "resolve_angles"]

_EQUAL: Final = "equal"

Outcome = Literal["error_low", "error_uncovered", "medium", "low_equal", "uncovered_equal"]
# The error and the equal-angles outcome of each kind of language a policy applies to.
_OUTCOMES: Final[dict[str, tuple[Outcome, Outcome]]] = {
    "low": ("error_low", "low_equal"),
    "uncovered": ("error_uncovered", "uncovered_equal"),
}


@dataclasses.dataclass(frozen=True)
class LanguageOutcome:
    """One language's outcome under a named set, for ``project``'s errors and warnings.

    Attributes
    ----------
    language : str
        The ISO 639-3 code.
    outcome : {"error_low", "error_uncovered", "medium", "low_equal", "uncovered_equal"}
        What happened to its included responses: an ``error_*`` outcome is one
        ``project`` raises for; the others are one flag each, warned once per call.
        Excluded rows produce no entry; their masks say what happened.
    count : int
        How many included responses have the language.
    tier : {"high", "medium", "low"} or None
        The set's tier for the language; ``None`` when the set does not list it.
    tier_only : bool
        Whether the set lists the language with a tier but no angles.
    set_label : str
        The ``name@version`` of the set that was consulted.
    """

    language: str
    outcome: Outcome
    count: int
    tier: Tier | None
    tier_only: bool
    set_label: str


@dataclasses.dataclass(frozen=True)
class ResolvedAngles:
    """What ``resolve_angles`` returns, one entry per survey response throughout.

    Attributes
    ----------
    angles_deg : ndarray of float64, shape (n, k)
        The angles each response is projected with: a set's angles, the circumplex's
        nominal angles where the row's set is ``"equal"`` or a fallback applies, NaN
        where the row is excluded or gets no angles.
    labels : ndarray of object, shape (n,)
        The row's ``ssp_angle_set`` label: ``"name@version"``, ``"equal"``, or ``""``
        where the row has no angles.
    medium : ndarray of bool
        Included rows projected with a medium-tier language's angles.
    low_exclude, low_equal : ndarray of bool
        Included rows of a low-tier language with angles, under ``on_low_confidence``
        ``"exclude"`` and ``"equal"``.
    uncovered_exclude, uncovered_equal : ndarray of bool
        Included rows of a language the set does not cover or lists without angles,
        under ``on_uncovered`` ``"exclude"`` and ``"equal"``.
    blank_language : ndarray of bool
        Included rows whose set is named but whose language is ``""``; they get no
        angles, and ``project`` refuses them.
    outcomes : tuple of LanguageOutcome
        One entry per language and outcome that needs a message, sorted by language.
    """

    angles_deg: NDArray[np.float64]
    labels: NDArray[np.object_]
    medium: NDArray[np.bool_]
    low_exclude: NDArray[np.bool_]
    low_equal: NDArray[np.bool_]
    uncovered_exclude: NDArray[np.bool_]
    uncovered_equal: NDArray[np.bool_]
    blank_language: NDArray[np.bool_]
    outcomes: tuple[LanguageOutcome, ...]


def resolve_angles(
    languages: NDArray[np.object_],
    included: NDArray[np.bool_],
    selection: AngleSelection,
    sets: Mapping[str, AngleSet],
    circumplex: Circumplex,
) -> ResolvedAngles:
    """Resolve each included response's angles from its language and the selection.

    The set of a response is ``selection.per_language[language]`` if listed, else
    ``selection.default``. ``"equal"`` gives the nominal angles and the label ``equal``
    without consulting the language. Under a named set a high-tier language gets the
    set's angles and label; a medium-tier one the same plus ``medium``; a low-tier one
    with angles follows ``on_low_confidence``; a language absent from the set, or listed
    with a tier and no angles (every low-tier language of the shipped sets), follows
    ``on_uncovered``. Under ``"error"`` the rows get no angles and an ``error_*`` outcome
    is reported for ``project`` to raise on; ``"exclude"`` sets the ``*_exclude`` mask;
    ``"equal"`` gives the nominal angles, the label ``equal`` and the ``*_equal`` mask.
    Rows not included are never consulted.

    Parameters
    ----------
    languages : ndarray of object, shape (n,)
        Each response's ISO 639-3 code, ``""`` where blank or unmapped.
    included : ndarray of bool, shape (n,)
        Which responses are projected.
    selection : AngleSelection
        The selection to apply.
    sets : mapping of str to AngleSet
        Every set the selection names, keyed by ``name@version``.
    circumplex : Circumplex
        The survey's circumplex: the nominal angles and the position count.

    Returns
    -------
    ResolvedAngles
        Angles, labels, one mask per outcome and the per-language outcome records.

    Raises
    ------
    DataReferenceError
        If a named set is for another circumplex or circumplex version than the
        survey's, or has another number of angles than the circumplex has positions;
        the message names both.

    References
    ----------
    The tiers and the rule that a language without validated angles is not projected
    with another language's: [@mitchell2026translation], chapter 4; the language-specific
    angles the rule is built on: [@aletta2024soundscape], Table 4.
    """
    languages = np.asarray(languages, dtype=object)
    included = np.asarray(included, dtype=bool)
    per_language = dict(selection.per_language)
    for ref in {*per_language.values(), selection.default}:
        if isinstance(ref, AngleSetRef):
            _check_match(sets[ref.label], circumplex)

    builder = _Builder(len(included), np.asarray(circumplex.angles_deg, dtype=np.float64))
    for language, count in sorted(Counter(languages[included].tolist()).items()):
        rows = included & (languages == language)
        choice = per_language.get(language, selection.default)
        if choice == _EQUAL:
            builder.equal(rows)
        else:
            builder.named(rows, language, count, sets[choice.label], selection)
    return builder.freeze()


def _check_match(angle_set: AngleSet, circumplex: Circumplex) -> None:
    """Raise unless the set is for this circumplex and has one angle per position."""
    if (angle_set.circumplex, angle_set.circumplex_version) != (
        circumplex.name,
        circumplex.version,
    ):
        msg = (
            f"angle set {angle_set.label} is for circumplex "
            f"{angle_set.circumplex}@{angle_set.circumplex_version}, the survey's is "
            f"{circumplex.name}@{circumplex.version}"
        )
        raise DataReferenceError(msg)
    counts = {len(entry.angles_deg) for _, entry in angle_set.languages if entry.has_angles}
    positions = len(circumplex.positions)
    if counts and counts != {positions}:
        msg = (
            f"angle set {angle_set.label} has {counts.pop()} angles per language, "
            f"the circumplex {circumplex.name}@{circumplex.version} has {positions} positions"
        )
        raise DataReferenceError(msg)


class _Builder:
    """Mutable arrays filled language by language, frozen into a ``ResolvedAngles``."""

    def __init__(self, n: int, nominal: NDArray[np.float64]) -> None:
        self.nominal = nominal
        self.angles = np.full((n, len(nominal)), np.nan, dtype=np.float64)
        self.labels = np.full(n, "", dtype=object)
        self.masks = {
            name: np.zeros(n, dtype=np.bool_)
            for name in (
                "medium",
                "low_exclude",
                "low_equal",
                "uncovered_exclude",
                "uncovered_equal",
                "blank_language",
            )
        }
        self.outcomes: list[LanguageOutcome] = []

    def equal(self, rows: NDArray[np.bool_]) -> None:
        self.angles[rows] = self.nominal
        self.labels[rows] = _EQUAL

    def named(
        self,
        rows: NDArray[np.bool_],
        language: str,
        count: int,
        angle_set: AngleSet,
        selection: AngleSelection,
    ) -> None:
        if not language:
            self.masks["blank_language"][rows] = True
            return
        entry = angle_set.by_language.get(language)
        record = _Record(language, count, entry, angle_set.label)
        if entry is None or not entry.has_angles:
            self._policy(selection.on_uncovered, "uncovered", rows, record)
        elif entry.tier == "low":
            self._policy(selection.on_low_confidence, "low", rows, record)
        else:
            self.angles[rows] = entry.angles_deg
            self.labels[rows] = angle_set.label
            if entry.tier == "medium":
                self.masks["medium"][rows] = True
                self.outcomes.append(record.outcome("medium"))

    def _policy(
        self,
        policy: Policy,
        kind: Literal["low", "uncovered"],
        rows: NDArray[np.bool_],
        record: _Record,
    ) -> None:
        error, equal = _OUTCOMES[kind]
        if policy == "error":
            self.outcomes.append(record.outcome(error))
        elif policy == "exclude":
            self.masks[f"{kind}_exclude"][rows] = True
        else:
            self.equal(rows)
            self.masks[f"{kind}_equal"][rows] = True
            self.outcomes.append(record.outcome(equal))

    def freeze(self) -> ResolvedAngles:
        return ResolvedAngles(
            angles_deg=self.angles,
            labels=self.labels,
            outcomes=tuple(self.outcomes),
            **self.masks,
        )


@dataclasses.dataclass(frozen=True)
class _Record:
    """What one language looks like in its set, before the outcome is known."""

    language: str
    count: int
    entry: LanguageAngles | None
    set_label: str

    def outcome(self, outcome: Outcome) -> LanguageOutcome:
        return LanguageOutcome(
            language=self.language,
            outcome=outcome,
            count=self.count,
            tier=None if self.entry is None else self.entry.tier,
            tier_only=self.entry is not None and not self.entry.has_angles,
            set_label=self.set_label,
        )
