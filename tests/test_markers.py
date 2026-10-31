"""Pin pytest's marker-selection behaviour that the project's addopts rely on."""

import subprocess
import sys
import textwrap


def test_later_m_overrides_addopts(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""
        [tool.pytest.ini_options]
        addopts = ["--strict-markers", "-m", "not reference and not network"]
        markers = ["network: needs a download", "reference: slow"]
    """)
    )
    (tmp_path / "test_x.py").write_text(
        textwrap.dedent("""
        import pytest
        def test_plain(): pass
        @pytest.mark.network
        def test_net(): pass
    """)
    )

    def run(*args):
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", "-q", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        ).stdout

    assert "1 passed, 1 deselected" in run()
    assert "1 passed, 1 deselected" in run("-m", "network")
