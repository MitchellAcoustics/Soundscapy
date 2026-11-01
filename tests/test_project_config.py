from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]

# Dependencies taken from a pinned git commit in [tool.uv.sources] rather than from an index,
# so a version specifier would be ignored. great-docs: main at the merge of PR #487.
GIT_SOURCED = {"great-docs"}

FLOOR_OPERATORS = {">=", "==", "~="}


def _all_requirements() -> list[tuple[str, str]]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entries = [("project.dependencies", r) for r in pyproject["project"]["dependencies"]]
    for group, items in pyproject["dependency-groups"].items():
        # {include-group = "..."} tables pull in another group, which is checked on its own.
        entries.extend((group, r) for r in items if isinstance(r, str))
    return entries


def test_git_sourced_exceptions_are_real_uv_sources():
    sources = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]
    for name in GIT_SOURCED:
        assert "git" in sources[name], name


def test_every_dependency_has_a_floor():
    for where, entry in _all_requirements():
        requirement = Requirement(entry)
        if requirement.name in GIT_SOURCED:
            continue
        operators = {spec.operator for spec in requirement.specifier}
        assert operators & FLOOR_OPERATORS, f"{where}: {entry} has no lower bound"


def test_taskfile_defines_the_normative_tasks():
    text = (ROOT / "Taskfile.yml").read_text()
    for task in (
        "default",
        "setup",
        "fmt",
        "lint",
        "typecheck",
        "test",
        "test:cov",
        "test:reference",
        "test:network",
        "check:fast",
        "check",
        "docs",
        "docs:papers",
        "docs:preview",
        "build",
        "clean",
        "repo-review",
    ):
        assert re.search(rf"^  {re.escape(task)}:", text, re.MULTILINE), task


# A version such as 0.13.6 or 3.54.
VERSION = r"\d+(?:\.\d+)+"


def _uv_floor() -> str:
    required = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"][
        "required-version"
    ]
    match = re.fullmatch(rf">=({VERSION})", required)
    assert match, f"[tool.uv] required-version {required!r} is not a single >= floor"
    return match.group(1)


def _go_task_floor() -> str:
    text = (ROOT / "Taskfile.yml").read_text()
    match = re.search(rf"^version: '({VERSION})'$", text, re.MULTILINE)
    assert match, "Taskfile.yml has no version line"
    return match.group(1)


def test_contributing_states_the_uv_and_go_task_floors():
    # pyproject.toml and Taskfile.yml set these floors; CONTRIBUTING.md repeats them for
    # readers, so this test keeps the copies in step.
    text = (ROOT / "CONTRIBUTING.md").read_text()
    for tool, floor in (("uv", _uv_floor()), ("go-task", _go_task_floor())):
        line = rf"^- \[{re.escape(tool)}\]\([^)]+\) {re.escape(floor)} or later\b"
        assert re.search(line, text, re.MULTILINE), f"CONTRIBUTING.md: {tool} {floor} or later"


def test_cloud_setup_installs_at_least_the_uv_floor():
    text = (ROOT / ".claude" / "cloud" / "setup.sh").read_text()
    assert re.search(rf"^uv_min={re.escape(_uv_floor())}$", text, re.MULTILINE)
