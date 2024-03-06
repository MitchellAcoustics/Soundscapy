import matplotlib.pyplot as plt
import pytest

import soundscapy as sspy
from soundscapy.plotting.circumplex import _circumplex_grid


@pytest.fixture()
def data():
    data = sspy.isd.load()
    data, excl_data = sspy.isd.validate(data)
    data = sspy.surveys.add_iso_coords(data)
    return data


@pytest.fixture()
def simulated_data():
    data = sspy.surveys.simulation(1000)
    data = sspy.surveys.add_iso_coords(data)
    return data


def test__circumplex_grid():
    """Test the circumplex grid."""
    fig, ax = plt.subplots()
    ax = _circumplex_grid(ax)
    assert type(ax) is plt.Subplot


def test_scatter(simulated_data):
    s = sspy.plotting.scatter(simulated_data, "ISOPleasant", "ISOEventful")
    assert type(s) is plt.Subplot
