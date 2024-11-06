import soundscapy
import pytest


def test_soundscapy_import():
    assert soundscapy.__version__ is not None, "Soundscapy version should be defined"


def test_core_soundscapy_modules():
    assert hasattr(soundscapy, "plotting"), "Soundscapy should have a plotting module"
    assert hasattr(soundscapy, "surveys"), "Soundscapy should have a surveys module"
    assert hasattr(soundscapy, "databases"), "Soundscapy should have a databases module"


@pytest.mark.optional_deps("audio")
def test_soundscapy_audio_module():
    assert hasattr(soundscapy, "audio"), "Soundscapy should have an audio module"
