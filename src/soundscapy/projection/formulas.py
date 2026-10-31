"""The projection kernel: attribute scores to (x, y) circumplex coordinates."""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from soundscapy.errors import FormulationError
from soundscapy.instrument.scale import Scale

__all__ = ["project_scores"]

_FORMULATIONS = ("canonical", "iso2019")
_NOMINAL_ANGLES_DEG = np.arange(0.0, 360.0, 45.0)
# Below any real angle set's sums, which are O(1), and far above the float64 residues
# (up to 1.8e-16 per term) that cos and sin leave at multiples of 90 degrees.
_DEGENERATE_SUM = 1e-9


def project_scores(
    scores: NDArray[np.float64],
    angles_deg: NDArray[np.float64],
    scale: Scale,
    formulation: Literal["canonical", "iso2019"],
) -> NDArray[np.float64]:
    """Project attribute scores onto the x (pleasant) and y (eventful) axes.

    With μ = ``scale.midpoint`` and h = ``scale.half_range`` from the declared scale, never
    from the data, the ``"canonical"`` formulation is

    ```
    x = Σ_i cos θ_i (s_i - μ) / (h · Σ_i |cos θ_i|)
    y = Σ_i sin θ_i (s_i - μ) / (h · Σ_i |sin θ_i|)
    ```

    The ``"iso2019"`` formulation is the ISO/TS 12913-3:2019 formula, computed from its own
    equations with c = cos 45° and items PAQ1..PAQ8 in position order:

    ```
    x = [(PAQ1 - PAQ5) + c(PAQ8 - PAQ4) + c(PAQ2 - PAQ6)] / (4 + √32)
    y = [(PAQ3 - PAQ7) + c(PAQ4 - PAQ8) + c(PAQ2 - PAQ6)] / (4 + √32)
    ```

    Both bound each axis separately to [-1, 1] for in-scale scores; the radius can exceed 1
    (up to 1.0824 at 45° on 1 to 5), and nothing is clipped or clamped. At the nominal angles
    on a 1 to 5 scale the two formulations are algebraically identical. Sums run in position
    order in float64.

    Parameters
    ----------
    scores : numpy.ndarray of float64, shape (n, k)
        Attribute scores, one row per response, columns in circumplex position order, in the
        units of ``scale``. A row with any NaN projects to (NaN, NaN).
    angles_deg : numpy.ndarray of float64, shape (n, k)
        Angle of each attribute for each response, in degrees; must be finite.
    scale : Scale
        The declared response scale of the scores.
    formulation : {"canonical", "iso2019"}
        Which projection formula to apply.

    Returns
    -------
    numpy.ndarray of float64, shape (n, 2)
        Columns x and y, dimensionless.

    Raises
    ------
    TypeError
        If ``scores`` or ``angles_deg`` is not a float64 array.
    ValueError
        If the arrays are not two-dimensional with equal shapes, an angle is not finite, or
        the formulation is unknown.
    FormulationError
        If ``"iso2019"`` is given a scale other than 1 to 5 or any row's angles differ from
        (0, 45, ..., 315); or, for any formulation, if in some row Σ|cos θ_i| (x axis) or
        Σ|sin θ_i| (y axis) is below 1e-9.

    References
    ----------
    The canonical formulation: [@mitchell2026translation], proof p. 23, §4.5. The 2019
    formulation: [@iso12913-3-2019], as written in [@mitchell2022how], Eqs. 1 and 2.
    """
    for name, arr in (("scores", scores), ("angles_deg", angles_deg)):
        if arr.dtype != np.float64:
            msg = f"{name} must be a float64 array, got {arr.dtype}"
            raise TypeError(msg)
    if scores.ndim != 2 or scores.shape != angles_deg.shape:  # noqa: PLR2004  # (n, k)
        msg = (
            f"scores and angles_deg must be 2-D with equal shape (n, k), "
            f"got {scores.shape} and {angles_deg.shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(angles_deg).all():
        msg = "angles_deg must be finite"
        raise ValueError(msg)
    if formulation not in _FORMULATIONS:
        msg = f"unknown formulation {formulation!r}; expected one of {_FORMULATIONS}"
        raise ValueError(msg)

    if formulation == "iso2019":
        _check_iso2019(angles_deg, scale)

    theta = np.deg2rad(angles_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    abs_cos, abs_sin = np.abs(cos).sum(axis=1), np.abs(sin).sum(axis=1)
    for axis, sums, label in (("x", abs_cos, "cos"), ("y", abs_sin, "sin")):
        low = np.flatnonzero(sums < _DEGENERATE_SUM)
        if low.size:
            msg = (
                f"degenerate angles: Σ|{label} θ_i| = {sums[low[0]]:.3g} is below "
                f"{_DEGENERATE_SUM:g} in row {low[0]} ({low.size} rows), so the {axis} axis "
                "is undefined"
            )
            raise FormulationError(msg)

    if formulation == "canonical":
        centred = scores - scale.midpoint
        x = (cos * centred).sum(axis=1) / (scale.half_range * abs_cos)
        y = (sin * centred).sum(axis=1) / (scale.half_range * abs_sin)
    else:
        x, y = _iso2019(scores)

    out = np.column_stack([x, y])
    out[np.isnan(scores).any(axis=1)] = np.nan
    return out


def _check_iso2019(angles_deg: NDArray[np.float64], scale: Scale) -> None:
    if (scale.minimum, scale.maximum) != (1.0, 5.0):
        msg = (
            f"iso2019 needs a 1 to 5 scale, got {scale.minimum:g} to {scale.maximum:g}; "
            "use the canonical formulation"
        )
        raise FormulationError(msg)
    if (
        angles_deg.shape[1] != _NOMINAL_ANGLES_DEG.size
        or not (angles_deg == _NOMINAL_ANGLES_DEG).all()
    ):
        msg = (
            "iso2019 needs the nominal angles (0, 45, ..., 315) in every row; "
            "use the canonical formulation"
        )
        raise FormulationError(msg)


def _iso2019(scores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    c = math.cos(math.radians(45.0))
    norm = 4.0 + math.sqrt(32.0)
    p1, p2, p3, p4, p5, p6, p7, p8 = scores.T
    x = ((p1 - p5) + c * (p8 - p4) + c * (p2 - p6)) / norm
    y = ((p3 - p7) + c * (p4 - p8) + c * (p2 - p6)) / norm
    return x, y
