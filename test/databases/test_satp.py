import pytest

from soundscapy import satp


@pytest.mark.slow
def test_load_zenodo():
    df = satp.load_zenodo(version="v1.2.1")
    assert df.shape == (17441, 16)
