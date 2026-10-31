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
        "repo-review",
    ):
        assert re.search(rf"^  {re.escape(task)}:", text, re.MULTILINE), task
