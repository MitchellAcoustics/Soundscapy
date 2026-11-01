from __future__ import annotations

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


def test_matplotlib_and_seaborn_not_imported():
    code = (
        "import sys; import soundscapy; "
        'assert "matplotlib" not in sys.modules; assert "seaborn" not in sys.modules'
    )
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


# The public surface of slice S1, exactly; each later slice adds its own names to this set.
S1_NAMES = frozenset(
    {
        "AngleSelection", "Circumplex", "Coding", "Coordinates", "ProjectionSettings",
        "ReadOptions", "Scale", "Survey", "SurveyMapping", "__version__", "attach",
        "circumplex", "datasets", "errors", "load_isd", "plot", "project", "read_mapping",
        "read_survey", "set_log_level", "show_versions", "workspace", "write_mapping",
    }
)  # fmt: skip


def test_all_is_the_s1_surface():
    assert len(soundscapy.__all__) == len(S1_NAMES) == 23
    assert set(soundscapy.__all__) == S1_NAMES
    assert soundscapy.__all__ == sorted(soundscapy.__all__)


def test_all_names_resolve():
    missing = [name for name in soundscapy.__all__ if not hasattr(soundscapy, name)]
    assert missing == []
