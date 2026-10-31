"""Tests for the Claude Code hook scripts."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "stop_check.py"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_stop_hook_falls_back_to_uv_run_task(tmp_path):
    # PATH holds only uv, so `task` is missing and the hook must use `uv run task check:fast`.
    uv = shutil.which("uv")
    assert uv is not None
    uv_dir = tmp_path / "bin"
    uv_dir.mkdir()
    (uv_dir / "uv").symlink_to(uv)
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(HOOK)],
        input="{}",
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(uv_dir)},
        cwd=HOOK.parents[2],
        check=False,
    )
    assert "neither `task` nor `uv`" not in result.stderr
    assert "`task` (go-task) is not on PATH" not in result.stderr
