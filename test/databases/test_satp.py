import pytest

from soundscapy import databases as db


@pytest.mark.slow
def test_load_zenodo():
    df = db.satp.load_zenodo(version="v1.2.1")
    assert df.shape == (17441, 16)
