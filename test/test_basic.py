import soundscapy


def test_soundscapy_import():
    assert soundscapy.__version__ is not None, "Soundscapy version should be defined"


def test_soundscapy_modules():
    assert hasattr(soundscapy, "audio"), "Soundscapy should have an audio module"
    assert hasattr(soundscapy, "plotting"), "Soundscapy should have a plotting module"
    assert hasattr(soundscapy, "surveys"), "Soundscapy should have a surveys module"
    assert hasattr(soundscapy, "databases"), "Soundscapy should have a databases module"
