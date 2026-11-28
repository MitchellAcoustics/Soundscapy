"""The quality rules of spec 6.7: every allowed rule x action pair, defaults, skips, invariants."""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from soundscapy._model import merge_settings
from soundscapy.errors import SettingsError
from soundscapy.survey.quality import RULES, QualityInputs, QualitySettings, apply_quality
from soundscapy.survey.status import Flag, Reason, Status
from tests._helpers import quality_settings

NAN = math.nan
# Row 0 is clean; every other row repeats it except where its comment says.
_ATTRIBUTES = np.array(
    [
        [1, 2, 3, 4, 5, 4, 3, 2],  # 0 clean
        [4, 4, 4, 4, 4, 4, 4, 4],  # 1 all_identical
        [3, 3, 3, 3, 3, 3, 3, 3],  # 2 all_midpoint
        [1, 2, 3, 4, 5, 4, 3, NAN],  # 3 one attribute missing
        [1, NAN, NAN, NAN, NAN, NAN, NAN, NAN],  # 4 11 of 12 items missing
        [1, 2, 3, 4, 5, 4, 3, 2],  # 5 no consent
        [1, 2, 3, 4, 5, 4, 3, 2],  # 6 test response
        [1, 2, 3, 4, 5, 4, 3, 2],  # 7 paper input
        [1, 2, 3, 4, 5, 4, 3, 2],  # 8 age 16
        [1, 2, 3, 4, 5, 4, 3, 2],  # 9 age missing
    ],
    dtype=np.float64,
)
_SOURCE = np.full((10, 4), 3.0)
_SOURCE[4] = NAN
_CONSENT = np.array([True] * 10)
_CONSENT[5] = False
_TEST = np.array([False] * 10)
_TEST[6] = True
_PAPER = np.array([False] * 10)
_PAPER[7] = True
_AGE = np.full(10, 40.0)
_AGE[8] = 16.0
_AGE[9] = NAN
N = 10

TRIGGER_ROW = {
    "all_identical": 1,
    "all_midpoint": 2,
    "excessive_missing": 4,
    "no_consent": 5,
    "test_response": 6,
    "paper_input": 7,
    "under_age": 8,
}


def _inputs(**overrides: object) -> QualityInputs:
    fields = {
        "attributes": _ATTRIBUTES,
        "midpoint": 3.0,
        "source_dominance": _SOURCE,
        "consent": _CONSENT,
        "test": _TEST,
        "paper_input": _PAPER,
        "age": _AGE,
    }
    fields.update(overrides)
    return QualityInputs(**fields)


def _apply(
    settings: QualitySettings | None = None, **overrides: object
) -> tuple[Status, tuple[str, ...]]:
    return apply_quality(
        Status.all_included(N), _inputs(**overrides), settings or QualitySettings()
    )


def _action_fields() -> list[tuple[str, str]]:
    pairs = []
    for rule in RULES:
        field = QualitySettings.model_fields.get(rule)
        if field is None:
            continue
        pairs.extend((rule, action) for action in get_args(field.annotation))
    return pairs


_PAIRS = _action_fields()
_REASON_CODES = {r.value for r in Reason}
_FLAG_CODES = {f.value for f in Flag}


def test_the_matrix_has_sixteen_allowed_pairs():
    assert len(_PAIRS) == 16  # 4x2 + 1x2 + 2x3, spec 6.7
    assert "incomplete_attributes" not in QualitySettings.model_fields


def _mask(codes: tuple[tuple[object, ...], ...], code: object) -> np.ndarray:
    return np.array([code in row for row in codes], dtype=bool)


def _rows_with(codes: tuple[tuple[object, ...], ...], code: object) -> list[int]:
    return np.flatnonzero(_mask(codes, code)).tolist()


@pytest.mark.parametrize(("rule", "action"), _PAIRS, ids=[f"{r}-{a}" for r, a in _PAIRS])
def test_rule_action_matrix(rule, action):
    status, _ = _apply(QualitySettings.model_validate({rule: action}))
    reason_rows = _rows_with(status.reasons, Reason(rule)) if rule in _REASON_CODES else []
    flag_rows = _rows_with(status.flags, Flag(rule)) if rule in _FLAG_CODES else []
    expected = [TRIGGER_ROW[rule]]
    assert reason_rows == (expected if action == "exclude" else [])
    assert flag_rows == (expected if action == "flag" else [])


_REFUSED = [
    ("all_identical", "flag"),
    ("excessive_missing", "flag"),
    ("no_consent", "flag"),
    ("test_response", "flag"),
    ("all_midpoint", "exclude"),
    ("incomplete_attributes", "off"),
]


@pytest.mark.parametrize(("rule", "action"), _REFUSED, ids=[f"{r}-{a}" for r, a in _REFUSED])
def test_rule_action_refused(rule, action):
    with pytest.raises(ValidationError):
        QualitySettings.model_validate({rule: action})
    with pytest.raises(SettingsError, match=rule):
        merge_settings(QualitySettings, [{rule: action}], context="quality")


def test_defaults_on_the_fixture():
    status, skipped = _apply()
    assert skipped == ()
    assert status.reasons == (
        (),
        (Reason.ALL_IDENTICAL,),
        (),
        (Reason.INCOMPLETE_ATTRIBUTES,),
        (Reason.EXCESSIVE_MISSING, Reason.INCOMPLETE_ATTRIBUTES),
        (Reason.NO_CONSENT,),
        (Reason.TEST_RESPONSE,),
        (),
        (),
        (),
    )
    assert _rows_with(status.flags, Flag.ALL_MIDPOINT) == [2]
    assert _rows_with(status.flags, Flag.PAPER_INPUT) == [7]
    assert _rows_with(status.flags, Flag.UNDER_AGE) == [8]
    assert np.flatnonzero(~status.included).tolist() == [1, 3, 4, 5, 6]


def test_skipped_rules():
    _, skipped = _apply(source_dominance=None, consent=None, test=None, paper_input=None, age=None)
    assert skipped == ("no_consent", "test_response", "paper_input", "under_age")
    _, skipped = _apply(
        QualitySettings(no_consent="off"),
        source_dominance=None,
        consent=None,
        test=None,
        paper_input=None,
        age=None,
    )
    assert skipped == ("test_response", "paper_input", "under_age")


def _missing_share_fixture(n_missing: int) -> QualityInputs:
    attributes = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]])
    source = np.array([[3.0, 3.0]])
    cells = np.concatenate([attributes, source], axis=1)
    cells[0, :n_missing] = NAN
    return QualityInputs(
        attributes=cells[:, :8],
        midpoint=3.0,
        source_dominance=cells[:, 8:],
        consent=None,
        test=None,
        paper_input=None,
        age=None,
    )


def _fires(inputs: QualityInputs, **fields: object) -> bool:
    status, _ = apply_quality(
        Status.all_included(1), inputs, QualitySettings.model_validate(fields)
    )
    return Reason.EXCESSIVE_MISSING in status.reasons[0]


def test_excessive_missing_denominator_counts_mapped_items():
    # 8 attributes + 2 mapped source items = 10; the default share is 0.5, strict >.
    assert _fires(_missing_share_fixture(6))
    assert not _fires(_missing_share_fixture(5))
    assert not _fires(_missing_share_fixture(10), excessive_missing_share=1.0)
    assert _fires(_missing_share_fixture(1), excessive_missing_share=0.0)
    for bad in (1.5, -0.1):
        with pytest.raises(ValidationError, match="excessive_missing_share"):
            QualitySettings.model_validate({"excessive_missing_share": bad})


def test_under_age_boundary():
    age = _AGE.copy()
    age[0] = 17.9
    age[1] = 18.0
    age[2] = 15.9
    status, _ = _apply(age=age)
    assert _rows_with(status.flags, Flag.UNDER_AGE) == [0, 2, 8]  # strict <, spec 6.7
    status, _ = _apply(QualitySettings(min_age=16.0), age=age)
    assert _rows_with(status.flags, Flag.UNDER_AGE) == [2]  # 16.0 itself no longer fires


@given(
    attributes=st.lists(
        st.lists(st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, NAN]), min_size=8, max_size=8),
        min_size=0,
        max_size=30,
    ),
    settings_=quality_settings(),
)
@settings(max_examples=200)
def test_quality_invariants(attributes, settings_):
    n = len(attributes)
    values = np.array(attributes, dtype=np.float64).reshape(n, 8)
    inputs = QualityInputs(
        attributes=values,
        midpoint=3.0,
        source_dominance=None,
        consent=None,
        test=None,
        paper_input=None,
        age=None,
    )
    status, _ = apply_quality(Status.all_included(n), inputs, settings_)
    assert len(status.reasons) == len(status.flags) == n
    assert status.included.tolist() == [len(r) == 0 for r in status.reasons]
    identical = _mask(status.reasons, Reason.ALL_IDENTICAL)
    midpoint = _mask(status.flags, Flag.ALL_MIDPOINT)
    assert not (identical & midpoint).any()
    any_nan = np.isnan(values).any(axis=1)
    assert _mask(status.reasons, Reason.INCOMPLETE_ATTRIBUTES).tolist() == any_nan.tolist()
    if settings_.all_identical == "exclude" and settings_.all_midpoint == "flag":
        constant = ~any_nan & (values == values[:, :1]).all(axis=1)
        assert (identical | midpoint).tolist() == constant.tolist()
    if settings_.excessive_missing == "exclude":
        looser = settings_.model_copy(
            update={"excessive_missing_share": min(1.0, settings_.excessive_missing_share + 0.25)}
        )
        strict_rows = _mask(status.reasons, Reason.EXCESSIVE_MISSING)
        loose_status, _ = apply_quality(Status.all_included(n), inputs, looser)
        loose_rows = _mask(loose_status.reasons, Reason.EXCESSIVE_MISSING)
        assert not (loose_rows & ~strict_rows).any()  # raising the share never adds a row
