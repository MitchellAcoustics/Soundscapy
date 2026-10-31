import importlib.metadata
import subprocess
import sys

import soundscapy


def test_version_matches_metadata():
    assert soundscapy.__version__ == importlib.metadata.version("soundscapy")


def test_import_is_silent():
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import soundscapy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_matplotlib_not_imported():
    code = 'import sys; import soundscapy; assert "matplotlib" not in sys.modules'
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_import_without_polars_or_pyarrow():
    code = (
        "import sys; sys.modules['polars'] = None; sys.modules['pyarrow'] = None; import soundscapy"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_all_names_resolve():
    missing = [name for name in soundscapy.__all__ if not hasattr(soundscapy, name)]
    assert missing == []
