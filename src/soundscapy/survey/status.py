"""Per-response inclusion status: why a response is excluded and what is flagged.

No response row is ever dropped. Each row carries a tuple of exclusion
``Reason`` codes (the row is included exactly when the tuple is empty) and a tuple
of ``Flag`` codes (the row is kept, with a note).
"""

import enum
import itertools
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from soundscapy._frames import new_frame, readonly_copy
from soundscapy._model import COL_FLAGS, COL_INCLUDED, COL_REASONS

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Flag", "Reason", "Status"]

_E = TypeVar("_E", bound=enum.StrEnum)


class Reason(enum.StrEnum):
    """Why a response is excluded from analysis.

    The members are listed in their normative order; codes inside a row's tuple follow it.
    """

    SCALE_VIOLATION = "scale_violation"
    ALL_IDENTICAL = "all_identical"
    EXCESSIVE_MISSING = "excessive_missing"
    NO_CONSENT = "no_consent"
    TEST_RESPONSE = "test_response"
    UNDER_AGE = "under_age"
    PAPER_INPUT = "paper_input"
    INCOMPLETE_ATTRIBUTES = "incomplete_attributes"
    UNCOVERED_LANGUAGE = "uncovered_language"
    LOW_CONFIDENCE_LANGUAGE = "low_confidence_language"


class Flag(enum.StrEnum):
    """A note on a response that is kept in the analysis.

    The members are listed in their normative order; codes inside a row's tuple follow it.
    """

    ALL_MIDPOINT = "all_midpoint"
    UNDER_AGE = "under_age"
    PAPER_INPUT = "paper_input"
    MEDIUM_CONFIDENCE_LANGUAGE = "medium_confidence_language"
    EQUAL_ANGLES_FALLBACK = "equal_angles_fallback"


def _ordered_unique(codes: Iterable[Any], order: type[enum.StrEnum]) -> bool:
    """Return whether ``codes`` are members of ``order``, unique and in enum order."""
    codes = tuple(codes)
    # StrEnum members equal their str values, so Reason and Flag codes (and plain str)
    # would match each other in index(); require the enum type itself.
    if not all(isinstance(c, order) for c in codes):
        return False
    members = list(order)
    try:
        positions = [members.index(c) for c in codes]
    except ValueError:
        return False
    return all(a < b for a, b in itertools.pairwise(positions))


def _add_code(
    rows: tuple[tuple[_E, ...], ...], mask: NDArray[np.bool_], code: _E, order: type[_E]
) -> tuple[tuple[_E, ...], ...]:
    """Return ``rows`` with ``code`` added where ``mask`` is true, in enum order.

    The result skips the per-row checks, so ``code`` is checked here, once per call:
    StrEnum members equal their str values, so without the type check a Flag (or a
    plain str) would pass as a Reason.
    """
    if not isinstance(code, order):
        msg = f"code must be a {order.__name__}, got {code!r} ({type(code).__name__})"
        raise ValueError(msg)  # noqa: TRY004  # Status reports every bad code as ValueError, as its constructor does
    members = list(order)
    return tuple(
        tuple(sorted({*row, code}, key=members.index)) if hit else row
        for row, hit in zip(rows, mask, strict=True)
    )


@dataclass(frozen=True, eq=False, slots=True)
class Status:
    """Inclusion status of every response.

    Parameters
    ----------
    included : numpy.ndarray of bool
        Whether each response is included; read-only after construction. True exactly
        when the response has no exclusion reason.
    reasons : tuple of tuple of Reason
        Per response, the exclusion reasons: unique and in enum order.
    flags : tuple of tuple of Flag
        Per response, the flags: unique and in enum order.

    Raises
    ------
    ValueError
        If the three sequences differ in length, ``included`` disagrees with
        ``reasons``, or a row's codes are repeated or out of enum order.
    """

    included: NDArray[np.bool_]
    reasons: tuple[tuple[Reason, ...], ...]
    flags: tuple[tuple[Flag, ...], ...]

    def __post_init__(self) -> None:
        """Validate the invariants and freeze the ``included`` array."""
        included = readonly_copy(self.included, np.bool_)
        if included.ndim != 1:
            msg = f"included must be one-dimensional, got {included.ndim} dimensions"
            raise ValueError(msg)
        if not len(included) == len(self.reasons) == len(self.flags):
            msg = (
                "included, reasons and flags must have the same length, got "
                f"{len(included)}, {len(self.reasons)} and {len(self.flags)}"
            )
            raise ValueError(msg)
        expected = np.array([len(r) == 0 for r in self.reasons], dtype=np.bool_)
        if not np.array_equal(included, expected):
            msg = "included must be true exactly where a response has no exclusion reason"
            raise ValueError(msg)
        for row in self.reasons:
            if not _ordered_unique(row, Reason):
                msg = f"reason codes must be unique and in enum order, got {row!r}"
                raise ValueError(msg)
        for row in self.flags:
            if not _ordered_unique(row, Flag):
                msg = f"flag codes must be unique and in enum order, got {row!r}"
                raise ValueError(msg)
        object.__setattr__(self, "included", included)

    @classmethod
    def _unchecked(
        cls,
        included: NDArray[np.bool_],
        reasons: tuple[tuple[Reason, ...], ...],
        flags: tuple[tuple[Flag, ...], ...],
    ) -> Self:
        """Build a status that is valid by construction, skipping the per-row checks."""
        status = object.__new__(cls)
        object.__setattr__(status, "included", readonly_copy(included, np.bool_))
        object.__setattr__(status, "reasons", reasons)
        object.__setattr__(status, "flags", flags)
        return status

    @classmethod
    def all_included(cls, n: int) -> Self:
        """Return a status in which all ``n`` responses are included and unflagged.

        Parameters
        ----------
        n : int
            Number of responses.

        Returns
        -------
        Status
            Every response included, with no reasons and no flags.
        """
        return cls._unchecked(np.ones(n, dtype=np.bool_), ((),) * n, ((),) * n)

    def _mask(self, mask: ArrayLike) -> NDArray[np.bool_]:
        array = np.asarray(mask)
        if array.shape != self.included.shape or (array.size and array.dtype != np.bool_):
            msg = (
                f"mask must be a boolean array of shape {self.included.shape}, "
                f"got {array.dtype} {array.shape}"
            )
            raise ValueError(msg)
        return array.astype(np.bool_)

    def with_reason(self, mask: ArrayLike, reason: Reason) -> Self:
        """Return a new status with ``reason`` added where ``mask`` is true.

        Parameters
        ----------
        mask : array-like of bool
            Which responses to exclude; one entry per response.
        reason : Reason
            The exclusion reason. A response that already has it keeps one copy.

        Returns
        -------
        Status
            A new status; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If ``mask`` is not a boolean array with one entry per response, or
            ``reason`` is not a ``Reason`` (a ``Flag`` or a plain str is
            rejected).
        """
        hit = self._mask(mask)
        reasons = _add_code(self.reasons, hit, reason, Reason)
        return self._unchecked(self.included & ~hit, reasons, self.flags)

    def with_flag(self, mask: ArrayLike, flag: Flag) -> Self:
        """Return a new status with ``flag`` added where ``mask`` is true.

        Parameters
        ----------
        mask : array-like of bool
            Which responses to flag; one entry per response.
        flag : Flag
            The flag. A response that already has it keeps one copy.

        Returns
        -------
        Status
            A new status; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If ``mask`` is not a boolean array with one entry per response, or ``flag``
            is not a ``Flag`` (a ``Reason`` or a plain str is rejected).
        """
        hit = self._mask(mask)
        return self._unchecked(self.included, self.reasons, _add_code(self.flags, hit, flag, Flag))

    def counts(self) -> dict[str, int]:
        """Count responses overall, by reason and by flag.

        Returns
        -------
        dict of str to int
            ``"n"``, ``"included"`` and ``"excluded"``, then ``"reason:<code>"`` for every
            ``Reason`` and ``"flag:<code>"`` for every ``Flag``, zero when absent.
            The keys do not depend on the data.
        """
        n_included = int(self.included.sum())
        out = {
            "n": len(self.included),
            "included": n_included,
            "excluded": len(self.included) - n_included,
        }
        # Codes are unique within a row, so counting codes counts rows.
        reasons = Counter(code for row in self.reasons for code in row)
        flags = Counter(code for row in self.flags for code in row)
        out.update({f"reason:{reason.value}": reasons[reason] for reason in Reason})
        out.update({f"flag:{flag.value}": flags[flag] for flag in Flag})
        return out

    def label_arrays(self) -> dict[str, NDArray[Any]]:
        """Return the three status columns as read-only arrays.

        Returns
        -------
        dict of str to numpy.ndarray
            ``ssp_included`` as a bool array; ``ssp_reasons`` and ``ssp_flags`` as object
            arrays of codes joined with ``";"``, the empty string when none.
        """
        return {
            COL_INCLUDED: self.included,
            COL_REASONS: _joined(self.reasons),
            COL_FLAGS: _joined(self.flags),
        }

    def to_frame(self) -> "pd.DataFrame":
        """Return the status as a pandas data frame.

        Returns
        -------
        pandas.DataFrame
            One row per response with the columns of ``label_arrays()``.
        """
        return new_frame(self.label_arrays())


def _joined(rows: tuple[tuple[enum.StrEnum, ...], ...]) -> NDArray[np.object_]:
    out = np.empty(len(rows), dtype=object)
    out[:] = [";".join(row) for row in rows]
    out.setflags(write=False)
    return out
