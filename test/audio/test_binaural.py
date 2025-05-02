from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soundscapy.audio import AnalysisSettings
from soundscapy.audio.binaural import Binaural

TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_audio_files"


@pytest.fixture
def test_binaural_signal():
    """Create a Binaural signal from a test WAV file."""
    test_file = TEST_AUDIO_DIR / "trimmed_CT101.wav"
    return Binaural.from_wav(test_file)


@pytest.fixture
def analysis_settings():
    """Create a simple AnalysisSettings object for testing."""
    settings = {
        "AcousticToolbox": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": [5, 95, "avg"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
        "scikit-maad": {
            "all_temporal_alpha_indices": {"run": True, "channel": ["Left", "Right"]}
        },
    }
    return AnalysisSettings(**settings)


def test_binaural_initialization(test_binaural_signal):
    """Test proper initialization of Binaural object."""
    assert test_binaural_signal.fs == 48000
    assert test_binaural_signal.channels == 2
    assert test_binaural_signal.recording == "trimmed_CT101"


def test_binaural_from_wav():
    test_file = TEST_AUDIO_DIR / "trimmed_CT101.wav"
    b = Binaural.from_wav(test_file)
    assert isinstance(b, Binaural)
    assert b.channels == 2
    assert b.fs == 48000
    assert b.recording == "trimmed_CT101"


def test_binaural_from_wav_resample():
    test_file = TEST_AUDIO_DIR / "trimmed_CT101.wav"
    b = Binaural.from_wav(test_file, resample=44100)
    assert isinstance(b, Binaural)
    assert b.channels == 2
    assert b.fs == 44100
    assert b.recording == "trimmed_CT101"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_binaural_calibration(test_binaural_signal):
    """Test calibration of Binaural signal."""
    calibrated = test_binaural_signal.calibrate_to(70)
    assert isinstance(calibrated, Binaural)

    # Test different calibration for each channel
    calibrated = test_binaural_signal.calibrate_to([68, 72])
    assert isinstance(calibrated, Binaural)

    # Test invalid input
    with pytest.raises(TypeError):
        test_binaural_signal.calibrate_to([60, 62, 64])


def test_acoustics_metric(test_binaural_signal):
    """Test acoustics metric calculation."""
    result = test_binaural_signal.acoustics_metric("LAeq")
    assert isinstance(result, pd.DataFrame)
    assert "LAeq" in result.columns
    assert result.index.get_level_values(0).tolist() == [
        "trimmed_CT101",
        "trimmed_CT101",
    ]
    assert result.index.get_level_values(1).tolist() == ["Left", "Right"]


def test_maad_metric(test_binaural_signal):
    """Test MAAD metric calculation."""
    result = test_binaural_signal.maad_metric("all_temporal_alpha_indices")
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == [
        "ZCR",
        "MEANt",
        "VARt",
        "SKEWt",
        "KURTt",
        "LEQt",
        "BGNt",
        "SNRt",
        "MED",
        "Ht",
        "ACTtFraction",
        "ACTtCount",
        "ACTtMean",
        "EVNtFraction",
        "EVNtMean",
        "EVNtCount",
    ]
    assert result.index.get_level_values(0).tolist() == [
        "trimmed_CT101",
        "trimmed_CT101",
    ]
    assert result.index.get_level_values(1).tolist() == ["Left", "Right"]


def test_process_all_metrics(test_binaural_signal, analysis_settings):
    """Test processing of all metrics."""
    result = test_binaural_signal.process_all_metrics(analysis_settings)
    assert isinstance(result, pd.DataFrame)
    assert "LAeq" in result.columns
    assert "LEQt" in result.columns
    assert result.index.get_level_values(0).tolist() == [
        "trimmed_CT101",
        "trimmed_CT101",
    ]
    assert result.index.get_level_values(1).tolist() == ["Left", "Right"]


@pytest.mark.slow
def test_mosqito_metric(test_binaural_signal):
    """Test MoSQITo metric calculation."""
    result = test_binaural_signal.mosqito_metric("loudness_zwtv", parallel=True)
    assert isinstance(result, pd.DataFrame)
    assert "N_5" in result.columns
    assert result.index.get_level_values(0).tolist() == [
        "trimmed_CT101",
        "trimmed_CT101",
    ]
    assert result.index.get_level_values(1).tolist() == ["Left", "Right"]


@pytest.mark.slow
def test_mosqito_metric_parallel_false(test_binaural_signal):
    """Test MoSQITo metric calculation with parallel stereo disabled."""
    result = test_binaural_signal.mosqito_metric("loudness_zwtv", parallel=False)
    assert isinstance(result, pd.DataFrame)
    assert "N_5" in result.columns
    assert result.index.get_level_values(0).tolist() == [
        "trimmed_CT101",
        "trimmed_CT101",
    ]
    assert result.index.get_level_values(1).tolist() == ["Left", "Right"]


def test_fs_resample_to_different_frequency():
    original_data = np.random.rand(2, 1000)
    original_fs = 44100
    new_fs = 48000
    binaural_signal = Binaural(original_data, original_fs)
    resampled_signal = binaural_signal.fs_resample(new_fs)
    assert resampled_signal.fs == new_fs
    assert resampled_signal.shape[0] == int(new_fs * len(original_data) / original_fs)
    assert isinstance(resampled_signal, Binaural)
    assert resampled_signal.duration == pytest.approx(
        binaural_signal.duration, abs=1e-2, rel=1e-1
    )


def test_fs_resample_to_same_frequency():
    original_data = np.random.rand(2, 1000)
    original_fs = 44100
    binaural_signal = Binaural(original_data, original_fs)
    resampled_signal = binaural_signal.fs_resample(original_fs)
    assert resampled_signal.fs == original_fs
    assert np.array_equal(resampled_signal, binaural_signal)
    assert isinstance(resampled_signal, Binaural)


def test_fs_resample_with_invalid_frequency():
    original_data = np.random.rand(2, 1000)
    original_fs = 44100
    binaural_signal = Binaural(original_data, original_fs)
    with pytest.raises(ValueError):
        binaural_signal.fs_resample(-1000)


if __name__ == "__main__":
    pytest.main()
