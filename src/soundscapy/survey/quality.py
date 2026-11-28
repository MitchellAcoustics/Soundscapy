"""Response quality rules (spec 6.7): settings, inputs and the rules as numpy functions.

Every rule runs on every row, so a row collects every code that applies. A rule writes a
``Reason`` under ``"exclude"`` and a ``Flag`` under ``"flag"``; a rule set to ``"off"``
writes nothing. ``incomplete_attributes`` always runs and has no setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import numpy as np
from pydantic import StrictFloat, field_validator

from soundscapy._model import FrozenModel
from soundscapy.survey.status import Flag, Reason, Status

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["RULES", "Action", "QualityInputs", "QualitySettings", "apply_quality"]

Action: TypeAlias = Literal["exclude", "flag", "off"]
"""What a rule writes when it fires: an exclusion reason, a flag, or nothing."""

RULES: Final = (
    "all_identical",
    "all_midpoint",
    "excessive_missing",
    "incomplete_attributes",
    "no_consent",
    "test_response",
    "paper_input",
    "under_age",
)
"""The rule names in the order of spec 6.7's table."""


class QualitySettings(FrozenModel):
    """Actions and thresholds of the quality rules (spec 6.7).

    Each rule takes only the actions its codes allow: ``all_identical``,
    ``excessive_missing``, ``no_consent`` and ``test_response`` exclude or are off;
    ``all_midpoint`` flags or is off; ``paper_input`` and ``under_age`` take all three.
    ``incomplete_attributes`` always excludes and has no field. The defaults reproduce
    the processing the 0.x tutorials applied to ISD v1.0 (PRD FR-1.5).

    Attributes
    ----------
    all_identical : {"exclude", "off"}
        Every attribute item present and equal, and not the scale midpoint.
    all_midpoint : {"flag", "off"}
        Every attribute item present and equal to the scale midpoint.
    excessive_missing : {"exclude", "off"}
        The missing share over the mapped attribute and source-dominance items is above
        ``excessive_missing_share``.
    excessive_missing_share : float
        The threshold the missing share must exceed, in ``[0, 1]``; default 0.5.
    no_consent : {"exclude", "off"}
        The consent cell is not one of the mapping's true values.
    test_response : {"exclude", "off"}
        The test cell is one of the mapping's true values.
    paper_input : {"exclude", "flag", "off"}
        The paper-input cell is one of the mapping's true values.
    under_age : {"exclude", "flag", "off"}
        The age is below ``min_age``.
    min_age : float
        The age below which ``under_age`` fires (strictly less than), in years; default 18.

    References
    ----------
    Erfanian et al. (2021) [@erfanian2021psychological] for the all-identical, midpoint and
    missing-share rules; Mitchell, Aletta and Kang (2022) [@mitchell2022how] for
    ``incomplete_attributes``.
    """

    all_identical: Literal["exclude", "off"] = "exclude"
    all_midpoint: Literal["flag", "off"] = "flag"
    excessive_missing: Literal["exclude", "off"] = "exclude"
    excessive_missing_share: StrictFloat = 0.5
    no_consent: Literal["exclude", "off"] = "exclude"
    test_response: Literal["exclude", "off"] = "exclude"
    paper_input: Action = "flag"
    under_age: Action = "flag"
    min_age: StrictFloat = 18.0

    @field_validator("excessive_missing_share")
    @classmethod
    def _share_in_unit_interval(cls, share: float) -> float:
        if not 0.0 <= share <= 1.0:
            msg = f"excessive_missing_share must be between 0 and 1, got {share}"
            raise ValueError(msg)
        return share


@dataclass(frozen=True, slots=True)
class QualityInputs:
    """The per-response inputs the quality rules read.

    Parameters
    ----------
    attributes : numpy.ndarray of float64
        Coded attribute scores, shape ``(n, k)``; a missing or excluded cell is NaN.
    midpoint : float
        The attribute scale's midpoint.
    source_dominance : numpy.ndarray of float64 or None
        Coded source-dominance scores, shape ``(n, m)``; ``None`` when the group is unmapped.
    consent, test, paper_input : numpy.ndarray of bool or None
        Whether each response's cell is one of the mapping's true values; ``None`` when
        the flag column is unmapped.
    age : numpy.ndarray of float64 or None
        Age in years, NaN where missing; ``None`` when unmapped.
    """

    attributes: NDArray[np.float64]
    midpoint: float
    source_dominance: NDArray[np.float64] | None
    consent: NDArray[np.bool_] | None
    test: NDArray[np.bool_] | None
    paper_input: NDArray[np.bool_] | None
    age: NDArray[np.float64] | None


def _masks(inputs: QualityInputs, settings: QualitySettings) -> dict[str, NDArray[np.bool_] | None]:
    """Return, per rule, which rows it fires on, or ``None`` when its input is unmapped."""
    attributes = inputs.attributes
    missing = np.isnan(attributes)
    complete = ~missing.any(axis=1)
    constant = complete & (attributes == attributes[:, :1]).all(axis=1)
    first = attributes[:, 0]
    items = [missing]
    if inputs.source_dominance is not None:
        items.append(np.isnan(inputs.source_dominance))
    cells = np.concatenate(items, axis=1)
    share = cells.mean(axis=1)
    return {
        "all_identical": constant & (first != inputs.midpoint),
        "all_midpoint": constant & (first == inputs.midpoint),
        "excessive_missing": share > settings.excessive_missing_share,
        "incomplete_attributes": ~complete,
        "no_consent": None if inputs.consent is None else ~inputs.consent,
        "test_response": inputs.test,
        "paper_input": inputs.paper_input,
        "under_age": None if inputs.age is None else inputs.age < settings.min_age,
    }


def apply_quality(
    status: Status, inputs: QualityInputs, settings: QualitySettings
) -> tuple[Status, tuple[str, ...]]:
    """Run every quality rule and record its code on the rows it fires on.

    Parameters
    ----------
    status : Status
        The status to add codes to; it is not changed.
    inputs : QualityInputs
        The coded scores and flag cells the rules read.
    settings : QualitySettings
        Each rule's action and the thresholds.

    Returns
    -------
    Status
        A new status with every firing rule's code added.
    tuple of str
        The rules skipped because their input is unmapped, in ``RULES`` order. A rule
        set to ``"off"`` is never listed; the settings record it.

    Notes
    -----
    Exact float equality is right here: coded scores are the parsed numbers, plus an
    integer offset or reversed against integer-valued bounds. ``under_age`` compares with
    a strict ``<``, and a missing age never fires it.

    References
    ----------
    Erfanian et al. (2021) [@erfanian2021psychological]: ``all_identical``, ``all_midpoint``
    and ``excessive_missing``. Mitchell, Aletta and Kang (2022) [@mitchell2022how]:
    ``incomplete_attributes``. The defaults reproduce the 0.x tutorials' processing of
    ISD v1.0 (PRD FR-1.5).
    """
    masks = _masks(inputs, settings)
    skipped: list[str] = []
    for rule in RULES:
        action = "exclude" if rule == "incomplete_attributes" else getattr(settings, rule)
        if action == "off":
            continue
        mask = masks[rule]
        if mask is None:
            skipped.append(rule)
            continue
        if action == "exclude":
            status = status.with_reason(mask, Reason(rule))
        else:
            status = status.with_flag(mask, Flag(rule))
    return status, tuple(skipped)
