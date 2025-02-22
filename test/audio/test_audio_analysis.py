from pathlib import Path

import pandas as pd
import pytest
import yaml

from soundscapy.audio.analysis_settings import AnalysisSettings
from soundscapy.audio.audio_analysis import AudioAnalysis

TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_audio_files"


@pytest.fixture
def sample_config():
    return {
        "version": "1.0",
        "AcousticToolbox": {
            "LAeq": {
                "run": True,
                "main": "avg",
                "statistics": [5, 10, 50, 90, 95, "avg", "min", "max", "kurt", "skew"],
                "channel": ["Left", "Right"],
                "label": "LAeq",
                "func_args": {"time": 0.125, "method": "average"},
            }
        },
        "scikit-maad": {
            "all_temporal_alpha_indices": {"run": True, "channel": ["Left", "Right"]}
        },
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def audio_analysis(temp_config_file):
    return AudioAnalysis(temp_config_file)


class TestAudioAnalysis:
    def test_initialization(self, audio_analysis, temp_config_file):
        assert isinstance(
            audio_analysis.config_manager.current_config, AnalysisSettings
        )
        assert audio_analysis.config_manager.config_path == temp_config_file

    def test_analyze_file(self, audio_analysis):
        test_file = TEST_AUDIO_DIR / "trimmed_CT101.wav"
        result = audio_analysis.analyze_file(test_file)
        assert isinstance(result, pd.DataFrame)
        assert "LAeq" in result.columns
        assert "LEQt" in result.columns  # One of the MAAD indices

    @pytest.mark.slow
    def test_analyze_folder_single_worker(self, audio_analysis):
        # Count the number of WAV files in the test directory
        expected_file_count = len(list(TEST_AUDIO_DIR.glob("*.wav")))

        result = audio_analysis.analyze_folder(TEST_AUDIO_DIR, max_workers=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "No results were produced"
        assert "LAeq" in result.columns
        assert "LEQt" in result.columns

        # Check if the number of rows in the result matches the number of WAV files
        analyzed_files = set(result.index.get_level_values(0))
        assert len(analyzed_files) == expected_file_count, (
            f"Expected {expected_file_count} files to be analyzed, but got {len(analyzed_files)}"
        )

        # Check if all expected files were analyzed
        expected_files = {f.stem for f in TEST_AUDIO_DIR.glob("*.wav")}
        assert analyzed_files == expected_files, (
            f"Mismatch in analyzed files. Expected: {expected_files}, Got: {analyzed_files}"
        )

    @pytest.mark.slow
    def test_analyze_folder_multi_worker(self, audio_analysis):
        # Count the number of WAV files in the test directory
        expected_file_count = len(list(TEST_AUDIO_DIR.glob("*.wav")))

        result = audio_analysis.analyze_folder(TEST_AUDIO_DIR, max_workers=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "No results were produced"
        assert "LAeq" in result.columns
        assert "LEQt" in result.columns

        # Check if the number of rows in the result matches the number of WAV files
        analyzed_files = set(result.index.get_level_values(0))
        assert len(analyzed_files) == expected_file_count, (
            f"Expected {expected_file_count} files to be analyzed, but got {len(analyzed_files)}"
        )

        # Check if all expected files were analyzed
        expected_files = {f.stem for f in TEST_AUDIO_DIR.glob("*.wav")}
        assert analyzed_files == expected_files, (
            f"Mismatch in analyzed files. Expected: {expected_files}, Got: {analyzed_files}"
        )

    def test_save_results(self, audio_analysis, tmp_path):
        df = pd.DataFrame({"test": [1, 2]})
        csv_path = tmp_path / "results.csv"
        audio_analysis.save_results(df, csv_path)
        assert csv_path.exists()

        xlsx_path = tmp_path / "results.xlsx"
        audio_analysis.save_results(df, xlsx_path)
        assert xlsx_path.exists()

        with pytest.raises(ValueError):
            audio_analysis.save_results(df, tmp_path / "results.txt")

    def test_update_config(self, audio_analysis):
        new_config = {"AcousticToolbox": {"LAeq": {"run": False}}}
        audio_analysis.update_config(new_config)
        assert not audio_analysis.settings.AcousticToolbox.root["LAeq"].run

    def test_save_config(self, audio_analysis, tmp_path):
        config_path = tmp_path / "new_config.yaml"
        audio_analysis.save_config(config_path)
        assert config_path.exists()

    @pytest.mark.slow
    def test_mosqito_metric(self, audio_analysis):
        mosqito_config = {
            "MoSQITo": {
                "loudness_zwtv": {
                    "run": True,
                    "main": 5,
                    "statistics": [
                        5,
                        10,
                        50,
                        90,
                        95,
                        "min",
                        "max",
                        "kurt",
                        "skew",
                        "avg",
                    ],
                    "channel": ["Left", "Right"],
                    "label": "N",
                    "parallel": True,
                    "func_args": {"field_type": "free"},
                }
            }
        }
        audio_analysis.update_config(mosqito_config)

        test_file = TEST_AUDIO_DIR / "trimmed_CT101.wav"
        result = audio_analysis.analyze_file(test_file, resample=48000)

        assert isinstance(result, pd.DataFrame)
        assert "N_5" in result.columns


if __name__ == "__main__":
    pytest.main()
