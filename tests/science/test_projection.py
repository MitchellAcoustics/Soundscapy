"""Science tests of the projection kernel against its sources.

The ISO/TS 12913-3:2019 formula is Eqs. 1 and 2 of Mitchell, Aletta & Kang (2022); the
canonical formulation is Mitchell et al. (2026), chapter 4, proof p. 23, section 4.5.
Items are in position order (p, v, e, ch, a, m, u, ca) = PAQ1..PAQ8 at 0, 45, ..., 315 degrees.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from soundscapy.instrument.scale import Scale
from soundscapy.projection.formulas import project_scores
from tests._helpers import NOMINAL_ANGLES_DEG, SCALE_1_5

pytestmark = pytest.mark.science

_SQRT2 = math.sqrt(2.0)


def _one(scores, angles, scale, formulation):
    """Project a single response; return (x, y)."""
    xy = project_scores(
        np.asarray([scores], dtype=np.float64),
        np.asarray([angles], dtype=np.float64),
        scale,
        formulation,
    )
    return float(xy[0, 0]), float(xy[0, 1])


@pytest.mark.parametrize("formulation", ["iso2019", "canonical"])
@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((5, 3, 3, 3, 1, 3, 3, 3), (_SQRT2 - 1.0, 0.0)),
        (
            (4, 5, 2, 1, 2, 1, 4, 5),
            ((1.0 + 2.0 * _SQRT2) / (2.0 + 2.0 * _SQRT2), -1.0 / (2.0 + 2.0 * _SQRT2)),
        ),
    ],
)
def test_iso2019_hand_values(scores, expected, formulation):
    """Hand values derived from the ISO/TS 12913-3:2019 formula.

    The formula is Eqs. 1 and 2 of Mitchell, Aletta & Kang (2022). For (5,3,3,3,1,3,3,3),
    x = (5 - 1) / (4 + sqrt 32) = sqrt 2 - 1 and y = 0. For (4,5,2,1,2,1,4,5),
    x = (2 + 4c + 4c) / (4 + sqrt 32) = (1 + 2 sqrt 2) / (2 + 2 sqrt 2) and
    y = (-2 - 4c + 4c) / (4 + sqrt 32) = -1 / (2 + 2 sqrt 2), with c = cos 45 degrees.
    """
    x, y = _one(scores, NOMINAL_ANGLES_DEG, SCALE_1_5, formulation)
    assert x == pytest.approx(
        expected[0], abs=1e-12
    )  # float64 sums of 8 terms; closed forms are exact
    assert y == pytest.approx(
        expected[1], abs=1e-12
    )  # float64 sums of 8 terms; closed forms are exact


def test_canonical_equals_iso2019_exhaustively():
    """Canonical equals ISO/TS 12913-3:2019 at 45 degrees on 1..5, for all 5^8 responses.

    At the nominal angles the canonical formula of Mitchell et al. (2026, section 4.5) reduces
    algebraically to Eqs. 1 and 2 of Mitchell, Aletta & Kang (2022), since the cosines and sines
    sum to 0 and their absolute values to 2 + 2 sqrt 2.
    """
    scores = np.indices((5,) * 8, dtype=np.float64).reshape(8, -1).T + 1.0
    assert scores.shape == (390_625, 8)
    angles = np.broadcast_to(NOMINAL_ANGLES_DEG, scores.shape)
    canon = project_scores(scores, angles, SCALE_1_5, "canonical")
    iso = project_scores(scores, angles, SCALE_1_5, "iso2019")
    np.testing.assert_allclose(canon, iso, rtol=0, atol=1e-12)  # spec §7.2; measured max 2.8e-16


# --- Property tests (named invariants) ------------------------------------------------------


@st.composite
def _scales(draw):
    if draw(st.booleans()):
        lo = draw(st.integers(-10, 9))
        hi = draw(st.integers(lo + 1, 10))
        return Scale(minimum=float(lo), maximum=float(hi))
    lo = draw(st.floats(-10.0, 9.5))
    hi = draw(st.floats(lo + 0.5, 10.0))
    return Scale(minimum=lo, maximum=hi, kind="interval")


# Eight strictly increasing angles in [0, 360), on a 0.01 degree grid.
_angle_sets = st.lists(st.integers(0, 35_999), min_size=8, max_size=8, unique=True).map(
    lambda hundredths: np.array(sorted(hundredths), dtype=np.float64) / 100.0
)


def _in_scale(scale, size=8):
    if scale.kind == "ordinal":
        item = st.integers(int(scale.minimum), int(scale.maximum)).map(float)
    else:
        item = st.floats(scale.minimum, scale.maximum)
    return st.lists(item, min_size=size, max_size=size)


@st.composite
def _canonical_case(draw):
    scale = draw(_scales())
    return scale, draw(_angle_sets), draw(_in_scale(scale))


@st.composite
def _canonical_shift_case(draw):
    scale, angles, scores = draw(_canonical_case())
    lo, hi = scale.minimum - min(scores), scale.maximum - max(scores)
    shift = (
        float(draw(st.integers(int(lo), int(hi))))
        if scale.kind == "ordinal"
        else draw(st.floats(lo, hi))
    )
    return scale, angles, scores, shift


@st.composite
def _iso_shift_case(draw):
    scores = draw(st.lists(st.integers(1, 5), min_size=8, max_size=8))
    shift = draw(st.integers(1 - min(scores), 5 - max(scores)))
    return [float(s) for s in scores], float(shift)


@given(_canonical_case())
def test_canonical_square_bounds(case):
    """|x| <= 1 and |y| <= 1 for any scale, angle set and in-scale scores (canonical).

    Each axis is a weighted mean of centred scores bounded by the half-range, normalised by
    the summed absolute weights (Mitchell et al. 2026, section 4.5).
    """
    scale, angles, scores = case
    x, y = _one(scores, angles, scale, "canonical")
    assert abs(x) <= 1 + 1e-12  # spec §7.1: float rounding of sums of 8 terms
    assert abs(y) <= 1 + 1e-12  # spec §7.1: float rounding of sums of 8 terms


@given(_canonical_case())
def test_neutral_maps_to_origin_canonical(case):
    """All items at the declared midpoint map to (0, 0) under canonical, for any angles.

    The canonical formulation centres each score at the scale midpoint (Mitchell et al. 2026,
    section 4.5).
    """
    scale, angles, _ = case
    x, y = _one([scale.midpoint] * 8, angles, scale, "canonical")
    assert x == pytest.approx(0.0, abs=1e-12)  # float rounding of sums of 8 terms
    assert y == pytest.approx(0.0, abs=1e-12)  # float rounding of sums of 8 terms


def test_neutral_maps_to_origin_iso2019():
    """All items at 3 on 1..5 map to (0, 0) under Eqs. 1 and 2 of Mitchell et al. (2022)."""
    x, y = _one([3.0] * 8, NOMINAL_ANGLES_DEG, SCALE_1_5, "iso2019")
    assert x == pytest.approx(0.0, abs=1e-12)  # float rounding of sums of 8 terms
    assert y == pytest.approx(0.0, abs=1e-12)  # float rounding of sums of 8 terms


@given(_canonical_shift_case())
def test_constant_shift_canonical(case):
    """Adding c to every item moves x by c·Σcos θ_i / (h·Σ|cos θ_i|), and y likewise.

    Follows from the linearity of the canonical formula (Mitchell et al. 2026, section 4.5).
    """
    scale, angles, scores, shift = case
    x0, y0 = _one(scores, angles, scale, "canonical")
    x1, y1 = _one([s + shift for s in scores], angles, scale, "canonical")
    cos, sin = np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))
    h = scale.half_range
    assert x1 - x0 == pytest.approx(  # float rounding of sums of 8 terms
        shift * cos.sum() / (h * np.abs(cos).sum()), abs=1e-12
    )
    assert y1 - y0 == pytest.approx(  # float rounding of sums of 8 terms
        shift * sin.sum() / (h * np.abs(sin).sum()), abs=1e-12
    )


@given(_iso_shift_case())
def test_constant_shift_iso2019(case):
    """Adding c to every item leaves (x, y) unchanged under the 2019 formula.

    Eqs. 1 and 2 of Mitchell et al. (2022) take differences of opposite items, and
    Σcos θ_i = Σsin θ_i = 0 at 45 degree steps.
    """
    scores, shift = case
    x0, y0 = _one(scores, NOMINAL_ANGLES_DEG, SCALE_1_5, "iso2019")
    x1, y1 = _one([s + shift for s in scores], NOMINAL_ANGLES_DEG, SCALE_1_5, "iso2019")
    assert x1 - x0 == pytest.approx(0.0, abs=1e-12)  # float rounding of sums of 8 terms
    assert y1 - y0 == pytest.approx(0.0, abs=1e-12)  # float rounding of sums of 8 terms
