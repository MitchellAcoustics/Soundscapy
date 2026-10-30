#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stop hook: run the fast project checks (``task check:fast``) before Claude stops.

Exits 0 immediately when ``stop_hook_active`` is set, which prevents the hook
from re-triggering itself. Exits 0 when the project has no ``Taskfile.yml``
yet. On failure the check output goes to stderr with exit code 2 so the agent
is asked to fix it before finishing. Tests are deliberately not included;
``task check`` is the full definition of done.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _clean_env() -> dict[str, str]:
    """Drop this script's own uv environment so `uv run` targets the project's .venv."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

    if payload.get("stop_hook_active"):
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())
    if not (project / "Taskfile.yml").exists():
        return 0
    if shutil.which("task") is None:
        print("stop_check hook: `task` (go-task) is not on PATH; skipped check:fast", file=sys.stderr)
        return 1

    try:
        result = subprocess.run(
            ["task", "check:fast"], cwd=project, capture_output=True, text=True, timeout=170, env=_clean_env()
        )
    except subprocess.TimeoutExpired:
        print("stop_check hook: `task check:fast` timed out", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print("`task check:fast` failed. Fix these before stopping:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
