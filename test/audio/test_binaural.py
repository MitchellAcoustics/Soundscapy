from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from soundscapy.audio import AnalysisSettings
from soundscapy.audio.binaural import Binaural


@pytest.fixture
def mock_binaural_signal():
    """Create a mock Binaural signal for testing."""
    fs = 44100
    duration = 1  # 1 second
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    left_channel = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    right_channel = np.sin(2 * np.pi * 880 * t)  # 880 Hz sine wave
    data = np.stack((left_channel, right_channel))
    return Binaural(data, fs, recording="test_recording")


@pytest.fixture
def analysis_settings():
    """Create a simple AnalysisSettings object for testing."""
    settings = {
        "PythonAcoustics": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": [5, 95],
                "channel": ["Left", "Right"],
                "label": "LAeq",
            }
        },
        "MoSQITo": {
            "loudness_zwtv": {
                "run": True,
                "main": 5,
                "statistics": [5, 95],
                "channel": ["Left", "Right"],
                "label": "N",
                "parallel": True,
            }
        },
        "scikit-maad": {
            "all_temporal_alpha_indices": {"run": True, "channel": ["Left", "Right"]}
        },
    }
    return AnalysisSettings(settings)


def test_binaural_initialization(mock_binaural_signal):
    """Test proper initialization of Binaural object."""
    assert mock_binaural_signal.fs == 44100
    assert mock_binaural_signal.channels == 2
    assert mock_binaural_signal.recording == "test_recording"


def test_binaural_calibration(mock_binaural_signal):
    """Test calibration of Binaural signal."""
    calibrated = mock_binaural_signal.calibrate_to(70)
    assert isinstance(calibrated, Binaural)

    # Test different calibration for each channel
    calibrated = mock_binaural_signal.calibrate_to([68, 72])
    assert isinstance(calibrated, Binaural)

    # Test invalid input
    with pytest.raises(ValueError):
        mock_binaural_signal.calibrate_to([60, 62, 64])


@patch("soundscapy.audio.binaural.pyacoustics_metric_2ch")
def test_pyacoustics_metric(mock_pyacoustics, mock_binaural_signal):
    """Test pyacoustics metric calculation."""
    mock_pyacoustics.return_value = pd.DataFrame(
        {"metric": [60, 62]}, index=["Left", "Right"]
    )

    result = mock_binaural_signal.pyacoustics_metric("LAeq")
    assert isinstance(result, pd.DataFrame)
    mock_pyacoustics.assert_called_once()


@patch("soundscapy.audio.binaural.mosqito_metric_2ch")
def test_mosqito_metric(mock_mosqito, mock_binaural_signal):
    """Test MoSQITo metric calculation."""
    mock_mosqito.return_value = pd.DataFrame(
        {"metric": [5, 6]}, index=["Left", "Right"]
    )

    result = mock_binaural_signal.mosqito_metric("loudness_zwtv")
    assert isinstance(result, pd.DataFrame)
    mock_mosqito.assert_called_once()


@patch("soundscapy.audio.binaural.maad_metric_2ch")
def test_maad_metric(mock_maad, mock_binaural_signal):
    """Test MAAD metric calculation."""
    mock_maad.return_value = pd.DataFrame(
        {"metric": [0.5, 0.6]}, index=["Left", "Right"]
    )

    result = mock_binaural_signal.maad_metric("all_temporal_alpha_indices")
    assert isinstance(result, pd.DataFrame)
    mock_maad.assert_called_once()


@patch("soundscapy.audio.binaural.Binaural.pyacoustics_metric")
@patch("soundscapy.audio.binaural.Binaural.mosqito_metric")
@patch("soundscapy.audio.binaural.Binaural.maad_metric")
def test_process_all_metrics(
    mock_pyacoustics, mock_mosqito, mock_maad, mock_binaural_signal, analysis_settings
):
    """Test processing of all metrics."""
    mock_pyacoustics.return_value = pd.DataFrame(
        {"LAeq": [60, 62]}, index=["Left", "Right"]
    )
    mock_mosqito.return_value = pd.DataFrame({"N": [5, 6]}, index=["Left", "Right"])
    mock_maad.return_value = pd.DataFrame(
        {"alpha": [0.5, 0.6]}, index=["Left", "Right"]
    )

    result = mock_binaural_signal.process_all_metrics(analysis_settings)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"LAeq", "N", "alpha"}
    mock_pyacoustics.assert_called()
    mock_mosqito.assert_called()
    mock_maad.assert_called()


if __name__ == "__main__":
    pytest.main()
