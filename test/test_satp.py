from soundscapy import satp


def test_load_zenodo():
    df = satp.load_zenodo(version="v1.2.1")
    assert df.shape == (17441, 16)
