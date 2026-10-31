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
