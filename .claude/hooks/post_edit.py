#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""PostToolUse hook: format and lint the single Python file that was just edited.

Reads the hook payload from stdin. For a ``.py`` file it runs ``ruff format``
then ``ruff check --fix`` on that file only. Unfixable problems are
printed to stderr with exit code 2 so the agent sees them.

The hook is a no-op until the project has a ``pyproject.toml`` (the tools are
run through ``uv run`` from the project environment).
"""

from __future__ import annotations

import json
import os
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
        return 0

    tool_input = payload.get("tool_input") or {}
    raw_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not raw_path:
        return 0

    path = Path(raw_path)
    if path.suffix != ".py" or not path.exists():
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())
    if not (project / "pyproject.toml").exists():
        return 0  # toolchain not set up yet; nothing to run

    try:
        relative = path.resolve().relative_to(project.resolve())
    except ValueError:
        return 0  # edited file is outside the project
    if relative.parts and relative.parts[0] == ".claude":
        return 0  # hook scripts and skills are not project code

    commands = [
        ["uv", "run", "ruff", "format", str(path)],
        ["uv", "run", "ruff", "check", "--fix", str(path)],
    ]

    problems: list[str] = []
    for cmd in commands:
        try:
            result = subprocess.run(cmd, cwd=project, capture_output=True, text=True, timeout=50, env=_clean_env())
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"post_edit hook could not run {' '.join(cmd[:3])}: {exc}", file=sys.stderr)
            return 1  # non-blocking: tool missing or hung, surface it without stopping work
        if result.returncode != 0:
            problems.append(f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}".rstrip())

    if problems:
        print(f"Remaining problems in {relative} after auto-fix:", file=sys.stderr)
        print("\n\n".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
