import pytest

from soundscapy import databases as db


@pytest.mark.slow
def test_load_zenodo():
    data = db.satp.load_zenodo(version="v1.2.1")
    assert data.shape == (17441, 16)
