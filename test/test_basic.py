import os

import pytest

import soundscapy


def test_soundscapy_import():
    assert soundscapy.__version__ is not None, "Soundscapy version should be defined"


def test_core_soundscapy_modules():
    assert hasattr(soundscapy, "plotting"), "Soundscapy should have a plotting module"
    assert hasattr(soundscapy, "surveys"), "Soundscapy should have a surveys module"
    assert hasattr(soundscapy, "databases"), "Soundscapy should have a databases module"


@pytest.mark.optional_deps("audio")
def test_soundscapy_audio_module():
    assert hasattr(soundscapy, "audio"), "Soundscapy should have an audio module"
    # Test that the key classes are available
    assert hasattr(soundscapy, "Binaural")
    assert hasattr(soundscapy, "AudioAnalysis")
    assert hasattr(soundscapy, "AnalysisSettings")
    assert hasattr(soundscapy, "ConfigManager")


@pytest.mark.optional_deps("spi")
def test_soundscapy_spi_module():
    """Test that the SPI module can be imported when dependencies are available."""
    assert hasattr(soundscapy, "spi"), "Soundscapy should have an spi module"
    # Test top-level imports
    assert hasattr(soundscapy, "MultiSkewNorm"), "MultiSkewNorm should be available"
    assert hasattr(soundscapy, "dp2cp"), "dp2cp should be available"
    # assert hasattr(soundscapy, "calculate_spi"), "calculate_spi should be available"
    # assert hasattr(soundscapy, "calculate_spi_from_data"), (
    #     "calculate_spi_from_data should be available"
    # )


def test_spi_import_error():
    """Test that helpful error message is shown when SPI dependencies are missing."""
    # Skip if dependencies are actually installed
    if os.environ.get("SPI_DEPS") == "1":
        pytest.skip("SPI dependencies are installed")

    # Since direct imports are now used instead of __getattr__, we need to test
    # through direct access to the module which would trigger ImportError
    with pytest.raises(ImportError) as excinfo:
        import soundscapy.spi  # noqa: F401

    # Check error message contains helpful instructions
    assert "SPI functionality requires" in str(excinfo.value)
    assert "soundscapy[spi]" in str(excinfo.value)
