"""Tests for the Claude Code hook scripts."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "stop_check.py"


def _run_hook(tmp_path: Path, path_dir: Path) -> subprocess.CompletedProcess[str]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "Taskfile.yml").write_text("")
    env = {**os.environ, "PATH": str(path_dir), "CLAUDE_PROJECT_DIR": str(project)}
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input="{}",
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="the fake task is a /bin/sh script")
def test_stop_hook_runs_task_check_fast(tmp_path):
    # PATH holds only a fake task that records its arguments; the hook must call
    # `task check:fast`.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv = tmp_path / "argv.txt"
    task = bin_dir / "task"
    task.write_text(f'#!/bin/sh\necho "$@" > "{argv}"\nexit 0\n')
    task.chmod(task.stat().st_mode | stat.S_IXUSR)
    result = _run_hook(tmp_path, bin_dir)
    assert result.returncode == 0, result.stderr
    assert argv.read_text().strip() == "check:fast"


def test_stop_hook_reports_when_task_is_not_on_path(tmp_path):
    # Runs on every OS: PATH is one empty directory, and the hook is started through the
    # absolute sys.executable, so no lookup on PATH is needed to launch it. Exit 1 does not
    # block the agent from stopping; the message says how to install go-task.
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run_hook(tmp_path, empty)
    assert result.returncode == 1
    assert "uv tool install go-task-bin" in result.stderr
