"""The S1 done-when check: the bundled ISD subset, projected and drawn by location."""

from __future__ import annotations

import matplotlib.axes
import numpy as np
import pytest

import soundscapy as sspy
from soundscapy.distribution.coordinates import valid_mask

pytestmark = pytest.mark.science


def test_thin_slice():
    """Load, project and plot the ISD subset; the plot draws the array it is given."""
    coords = sspy.project(sspy.load_isd(subset=True), angles="equal")
    ax = sspy.plot.scatter(coords, by="location")
    assert isinstance(ax, matplotlib.axes.Axes)

    groups = coords.split("location")
    drawn = np.vstack([collection.get_offsets() for collection in ax.collections])
    expected = np.vstack([g.xy[valid_mask(g)] for g in groups.values()])
    # Exact: scatter stores the coordinates it is handed, so no arithmetic separates them.
    np.testing.assert_array_equal(drawn, expected)
    assert len(drawn) == coords.n_valid()
