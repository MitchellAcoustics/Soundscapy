"""project_scores: input checks, refusals, NaN rows and the absence of clamping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from soundscapy.errors import FormulationError
from soundscapy.instrument.scale import Scale
from soundscapy.projection.formulas import project_scores
from tests._helpers import NOMINAL_ANGLES_DEG, SCALE_1_5

_FORMULATIONS = ["canonical", "iso2019", "aletta2024"]


def _nominal(n):
    return np.broadcast_to(NOMINAL_ANGLES_DEG, (n, 8))


@pytest.mark.parametrize("formulation", _FORMULATIONS)
def test_output_float64_shape(formulation):
    scores = np.full((3, 8), 3.0)
    xy = project_scores(scores, _nominal(3), SCALE_1_5, formulation)
    assert xy.dtype == np.float64
    assert xy.shape == (3, 2)


@pytest.mark.parametrize("formulation", _FORMULATIONS)
@pytest.mark.parametrize("item", range(8))
def test_nan_item_gives_nan_row_only(formulation, item):
    scores = np.array([[5, 3, 3, 3, 1, 3, 3, 3]] * 3, dtype=np.float64)
    scores[1, item] = math.nan
    xy = project_scores(scores, _nominal(3), SCALE_1_5, formulation)
    assert np.isnan(xy[1]).all()
    assert np.isfinite(xy[[0, 2]]).all()


def test_iso2019_refuses_other_scale():
    with pytest.raises(FormulationError, match="1 to 5"):
        project_scores(
            np.full((1, 8), 4.0), _nominal(1), Scale(minimum=1.0, maximum=7.0), "iso2019"
        )


@pytest.mark.parametrize(
    "angles",
    [
        np.array([[0.0, 46.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]]),
        np.vstack([NOMINAL_ANGLES_DEG, NOMINAL_ANGLES_DEG + 1.0]),
        np.array([[0.0, 90.0, 180.0, 270.0]]),
    ],
)
def test_iso2019_refuses_non_nominal_angles(angles):
    scores = np.full(angles.shape, 3.0)
    with pytest.raises(FormulationError, match="nominal"):
        project_scores(scores, angles, SCALE_1_5, "iso2019")


@pytest.mark.parametrize(
    ("scores", "angles"),
    [
        (np.full((2, 8), 3.0), np.full((2, 7), 0.0)),
        (np.full((2, 8), 3.0), np.full((3, 8), 0.0)),
        (np.full(8, 3.0), np.full(8, 0.0)),
    ],
)
def test_shape_mismatch_raises(scores, angles):
    with pytest.raises(ValueError, match="shape"):
        project_scores(scores, angles, SCALE_1_5, "canonical")


def test_non_float64_raises():
    with pytest.raises(TypeError, match="float64"):
        project_scores(
            np.full((1, 8), 3, dtype=np.int64),  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime TypeError
            _nominal(1),
            SCALE_1_5,
            "canonical",
        )


def test_non_finite_angles_raise():
    angles = np.array([NOMINAL_ANGLES_DEG])
    angles[0, 3] = math.nan
    with pytest.raises(ValueError, match="finite"):
        project_scores(np.full((1, 8), 3.0), angles, SCALE_1_5, "canonical")


def test_unknown_formulation_raises():
    with pytest.raises(ValueError, match="nope") as info:
        project_scores(
            np.full((1, 8), 3.0),
            _nominal(1),
            SCALE_1_5,
            "nope",  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime ValueError
        )

    for name in _FORMULATIONS:
        assert repr(name) in str(info.value)


def test_aletta2024_accepts_other_scales_and_angles():
    # Hand value on 1..7 at 0, 46, ..., 340: with rho = 6, lambda_P = 3 * sum|cos| and
    # lambda_E = 3 * sum|sin|, and x, y are the raw weighted sums over those.
    angles = np.array([[0.0, 46.0, 94.0, 138.0, 177.0, 231.0, 275.0, 340.0]])
    scores = np.array([[7.0, 4.0, 4.0, 4.0, 1.0, 4.0, 4.0, 4.0]])
    xy = project_scores(scores, angles, Scale(minimum=1.0, maximum=7.0), "aletta2024")
    cos, sin = np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))
    expected = [
        (cos * scores[0]).sum() / (3.0 * np.abs(cos).sum()),
        (sin * scores[0]).sum() / (3.0 * np.abs(sin).sum()),
    ]
    np.testing.assert_allclose(xy[0], expected, rtol=0, atol=1e-12)  # float64 sums of 8 terms


@pytest.mark.parametrize("formulation", ["canonical", "aletta2024"])
@pytest.mark.parametrize(
    ("angles", "axis"),
    [((90.0, 270.0), "x axis"), ((0.0, 180.0), "y axis")],
)
def test_degenerate_angle_sums_raise(angles, axis, formulation):
    # Sums are float64 residues of about 1e-16 here, never exactly 0.
    rows = np.array([[0.0, 90.0], angles])
    with pytest.raises(FormulationError, match=axis):
        project_scores(np.full((2, 2), 3.0), rows, SCALE_1_5, formulation)


@pytest.mark.parametrize("formulation", _FORMULATIONS)
def test_no_clamping(formulation):
    scores = np.array([[5, 5, 5, 1, 1, 1, 1, 5]], dtype=np.float64)
    x, y = project_scores(scores, _nominal(1), SCALE_1_5, formulation)[0]
    assert x == pytest.approx(1.0, abs=1e-12)  # closed form; float64 sums of 8 terms
    assert y == pytest.approx(
        math.sqrt(2.0) - 1.0, abs=1e-12
    )  # closed form; float64 sums of 8 terms
    radius = math.sqrt(4.0 - 2.0 * math.sqrt(2.0))
    assert np.hypot(x, y) == pytest.approx(
        radius, abs=1e-12
    )  # closed form; float64 sums of 8 terms
    assert np.hypot(x, y) > 1.08  # a clip to the unit circle would fail this
