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


def test_import_and_project_without_narwhals_polars_or_pyarrow():
    # A None entry in sys.modules makes importing that name fail.
    code = (
        "import sys\n"
        "for name in ('narwhals', 'polars', 'pyarrow'):\n"
        "    sys.modules[name] = None\n"
        "import soundscapy as sspy\n"
        "sspy.project(sspy.load_isd(subset=True))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# The public surface of slice S1, exactly; each later slice adds its own names to this set.
S2_NAMES = frozenset(
    {
        "AngleSelection", "Circumplex", "Coding", "Coordinates", "ProjectionSettings",
        "QualitySettings", "ReadOptions", "Scale", "Survey", "SurveyMapping", "__version__",
        "attach", "circumplex", "datasets", "errors", "load_isd", "plot", "project",
        "read_mapping", "read_survey", "set_log_level", "show_versions", "workspace",
        "write_mapping",
    }
)  # fmt: skip


def test_all_is_the_s2_surface():
    assert len(soundscapy.__all__) == len(S2_NAMES) == 24
    assert set(soundscapy.__all__) == S2_NAMES
    assert soundscapy.__all__ == sorted(soundscapy.__all__)


def test_all_names_resolve():
    missing = [name for name in soundscapy.__all__ if not hasattr(soundscapy, name)]
    assert missing == []
