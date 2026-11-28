"""The per-response angle resolver, tested on arrays with the synthetic three-tier set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soundscapy.errors import DataReferenceError
from soundscapy.instrument import AngleSet, angle_set, circumplex, read_angle_set
from soundscapy.instrument.angles import AngleSelection
from soundscapy.projection._angles import LanguageOutcome, resolve_angles

SYNTHETIC = read_angle_set(Path(__file__).parents[1] / "fixtures" / "synthetic_tiers" / "1.toml")
ALETTA = angle_set("aletta-2024", "1")
CIRC = circumplex("iso-12913-outdoor", "1")
NOMINAL = np.asarray(CIRC.angles_deg, dtype=np.float64)
SETS = {SYNTHETIC.label: SYNTHETIC, ALETTA.label: ALETTA}

# One row per branch: High, Medium, Low with angles, absent, Low tier-only, a blank language
# on an included row, and a blank language on an excluded row.
LANGUAGES = np.array(["qaa", "qab", "qac", "qad", "qae", "", ""], dtype=object)
INCLUDED = np.array([True, True, True, True, True, True, False])
QAA, QAB, QAC, QAD, QAE, BLANK, EXCLUDED = range(7)
MASKS = ("medium", "low_exclude", "low_equal", "uncovered_exclude", "uncovered_equal")


def _selection(**fields) -> AngleSelection:
    return AngleSelection.model_validate({"default": SYNTHETIC.label, **fields})


def _set_masks(result) -> set[str]:
    return {name for name in (*MASKS, "blank_language") if getattr(result, name).any()}


def test_equal_default_uses_nominal_angles_and_never_reads_the_language():
    result = resolve_angles(LANGUAGES, INCLUDED, AngleSelection(), SETS, CIRC)
    assert result.angles_deg.shape == (7, 8)
    assert result.angles_deg.dtype == np.float64
    np.testing.assert_array_equal(result.angles_deg[INCLUDED], np.tile(NOMINAL, (6, 1)))
    assert np.isnan(result.angles_deg[EXCLUDED]).all()
    assert result.labels.tolist() == ["equal"] * 6 + [""]
    assert _set_masks(result) == set()
    assert result.outcomes == ()


def test_high_row_gets_the_set_angles_and_label():
    result = resolve_angles(LANGUAGES, INCLUDED, _selection(), SETS, CIRC)
    np.testing.assert_array_equal(result.angles_deg[QAA], SYNTHETIC.by_language["qaa"].angles_deg)
    assert result.labels[QAA] == "synthetic-tiers@1"
    assert not any(getattr(result, name)[QAA] for name in MASKS)


def test_medium_row_gets_the_set_angles_and_the_medium_mask():
    result = resolve_angles(LANGUAGES, INCLUDED, _selection(), SETS, CIRC)
    np.testing.assert_array_equal(result.angles_deg[QAB], SYNTHETIC.by_language["qab"].angles_deg)
    assert result.labels[QAB] == "synthetic-tiers@1"
    assert result.medium.tolist() == [False, True, False, False, False, False, False]
    assert (
        LanguageOutcome(
            language="qab",
            outcome="medium",
            count=1,
            tier="medium",
            tier_only=False,
            set_label="synthetic-tiers@1",
        )
        in result.outcomes
    )


def _expected(policy: str, kind: str, row: int, result) -> None:
    """Check one row under ``policy`` for ``kind`` in {"low", "uncovered"}."""
    if policy == "error":
        assert np.isnan(result.angles_deg[row]).all()
        assert result.labels[row] == ""
        assert not any(getattr(result, name)[row] for name in MASKS)
        assert any(o.outcome == f"error_{kind}" and o.count == 1 for o in result.outcomes)
    elif policy == "exclude":
        assert np.isnan(result.angles_deg[row]).all()
        assert result.labels[row] == ""
        assert getattr(result, f"{kind}_exclude")[row]
    else:
        np.testing.assert_array_equal(result.angles_deg[row], NOMINAL)
        assert result.labels[row] == "equal"
        assert getattr(result, f"{kind}_equal")[row]
        assert any(o.outcome == f"{kind}_equal" and o.count == 1 for o in result.outcomes)


@pytest.mark.parametrize("on_low_confidence", ["error", "exclude", "equal"])
@pytest.mark.parametrize("on_uncovered", ["error", "exclude", "equal"])
def test_low_and_uncovered_follow_their_overrides(on_uncovered, on_low_confidence):
    selection = _selection(on_uncovered=on_uncovered, on_low_confidence=on_low_confidence)
    result = resolve_angles(LANGUAGES, INCLUDED, selection, SETS, CIRC)
    _expected(on_low_confidence, "low", QAC, result)
    _expected(on_uncovered, "uncovered", QAD, result)
    # Low tier-only follows on_uncovered: there are no angles to project with.
    _expected(on_uncovered, "uncovered", QAE, result)
    by_language = {o.language: o for o in result.outcomes}
    if on_low_confidence != "exclude":
        assert by_language["qac"].tier == "low"
        assert by_language["qac"].tier_only is False
    if on_uncovered != "exclude":
        assert by_language["qad"].tier is None
        assert by_language["qad"].tier_only is False
        assert by_language["qae"].tier == "low"
        assert by_language["qae"].tier_only is True


def test_outcomes_count_rows_per_language_and_are_sorted():
    languages = np.array(["qad", "qac", "qad", "qad"], dtype=object)
    included = np.ones(4, dtype=bool)
    result = resolve_angles(languages, included, _selection(), SETS, CIRC)
    assert [(o.language, o.outcome, o.count) for o in result.outcomes] == [
        ("qac", "error_low", 1),
        ("qad", "error_uncovered", 3),
    ]


def test_per_language_picks_that_set_for_its_rows_only():
    selection = _selection(per_language={"qab": ALETTA.label}, on_uncovered="exclude")
    result = resolve_angles(LANGUAGES, INCLUDED, selection, SETS, CIRC)
    # aletta-2024@1 has no qab, so the qab row is uncovered like qad and qae; qaa keeps the
    # default set.
    assert result.uncovered_exclude.tolist() == [False, True, False, True, True, False, False]
    assert result.labels[QAB] == ""
    assert result.labels[QAA] == "synthetic-tiers@1"
    # Each outcome names the set that was consulted for its language.
    fallback = _selection(per_language={"qab": ALETTA.label}, on_uncovered="equal")
    result = resolve_angles(LANGUAGES, INCLUDED, fallback, SETS, CIRC)
    assert {o.language: o.set_label for o in result.outcomes if o.outcome == "uncovered_equal"} == {
        "qab": "aletta-2024@1",
        "qad": "synthetic-tiers@1",
        "qae": "synthetic-tiers@1",
    }


def test_per_language_can_name_a_set_while_the_default_is_equal():
    selection = AngleSelection(per_language={"qab": SYNTHETIC.label})
    result = resolve_angles(LANGUAGES, INCLUDED, selection, SETS, CIRC)
    assert result.labels.tolist() == ["equal", "synthetic-tiers@1", *["equal"] * 4, ""]
    assert result.medium[QAB]
    assert not result.blank_language.any()  # the blank row's set is "equal"


def _variant(**changes) -> AngleSet:
    data = SYNTHETIC.model_dump(by_alias=True)
    return AngleSet.model_validate({**data, **changes})


def test_set_for_another_circumplex_version_names_both():
    other = _variant(circumplex_version="2")
    with pytest.raises(DataReferenceError) as info:
        resolve_angles(LANGUAGES, INCLUDED, _selection(), {other.label: other}, CIRC)
    assert "iso-12913-outdoor@2" in str(info.value)
    assert "iso-12913-outdoor@1" in str(info.value)


def test_set_with_another_angle_count_names_both_counts():
    languages = {
        "qaa": {"tier": "high", "angles_deg": [0.0, 46.0, 94.0, 138.0, 177.0, 231.0, 275.0]}
    }
    other = _variant(languages=languages)
    with pytest.raises(DataReferenceError) as info:
        resolve_angles(LANGUAGES, INCLUDED, _selection(), {other.label: other}, CIRC)
    assert "7 angles" in str(info.value)
    assert "8 positions" in str(info.value)


def test_tier_only_set_passes_the_count_check():
    tiers_only = _variant(name="tiers-only", languages={"qaa": {"tier": "medium"}})
    selection = _selection(per_language={"qaa": tiers_only.label}, on_uncovered="equal")
    sets = {**SETS, tiers_only.label: tiers_only}
    result = resolve_angles(LANGUAGES, INCLUDED, selection, sets, CIRC)
    assert result.uncovered_equal[QAA]


def test_blank_language_is_reported_on_included_rows_under_a_named_set_only():
    result = resolve_angles(LANGUAGES, INCLUDED, _selection(), SETS, CIRC)
    assert result.blank_language.tolist() == [False, False, False, False, False, True, False]
    assert np.isnan(result.angles_deg[BLANK]).all()
    assert result.labels[BLANK] == ""
    assert np.isnan(result.angles_deg[EXCLUDED]).all()
    assert result.labels[EXCLUDED] == ""


def test_excluded_rows_are_never_resolved():
    languages = np.array(["qad", "qac", ""], dtype=object)
    included = np.zeros(3, dtype=bool)
    result = resolve_angles(languages, included, _selection(), SETS, CIRC)
    assert np.isnan(result.angles_deg).all()
    assert result.labels.tolist() == ["", "", ""]
    assert _set_masks(result) == set()
    assert result.outcomes == ()
