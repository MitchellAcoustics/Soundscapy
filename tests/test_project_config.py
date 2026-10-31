import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_dev_and_test_dependency_has_a_floor():
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]
    for name in ("dev", "test"):
        for requirement in groups[name]:
            assert re.search(r"(>=|==)", requirement), f"{name}: {requirement} has no lower bound"


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
