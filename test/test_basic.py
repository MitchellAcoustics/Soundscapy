import pytest

import soundscapy


def test_soundscapy_import():
    assert soundscapy.__version__ is not None, "Soundscapy version should be defined"


def test_core_soundscapy_modules():
    assert hasattr(soundscapy, "plotting"), "Soundscapy should have a plotting module"
    assert hasattr(soundscapy, "surveys"), "Soundscapy should have a surveys module"
    assert hasattr(soundscapy, "databases"), "Soundscapy should have a databases module"


def test_soundscapy_audio_module():
    pytest.importorskip("mosqito")
    assert hasattr(soundscapy, "audio"), "Soundscapy should have an audio module"
    assert hasattr(soundscapy, "Binaural")
    assert hasattr(soundscapy, "AudioAnalysis")
    assert hasattr(soundscapy, "AnalysisSettings")
    assert hasattr(soundscapy, "ConfigManager")


def test_soundscapy_spi_module():
    """Test that the SPI module can be imported when dependencies are available."""
    pytest.importorskip("rpy2")
    assert hasattr(soundscapy, "spi"), "Soundscapy should have an spi module"
    assert hasattr(soundscapy, "MultiSkewNorm"), "MultiSkewNorm should be available"
    assert hasattr(soundscapy, "dp2cp"), "dp2cp should be available"
    assert hasattr(soundscapy, "spi_score"), "spi_score should be available"


def test_soundscapy_satp_module():
    """Test that the SATP module can be imported when dependencies are available."""
    pytest.importorskip("rpy2")
    assert hasattr(soundscapy, "satp"), "Soundscapy should have a satp module"
    assert hasattr(soundscapy, "fit_circe"), "fit_circe should be available"
    assert hasattr(soundscapy, "CircModelE"), "CircModelE should be available"
