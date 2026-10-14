# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Guard that each ``__init__.pyi``'s literal ``__all__`` matches the runtime one.

See the header of ``soundscapy/__init__.pyi`` for why the stubs carry the API.
"""

import ast
from functools import cache
from pathlib import Path

import pytest

import soundscapy

ROOT = Path(soundscapy.__file__).parent
STUBBED = sorted(
    ".".join(("soundscapy", *p.parent.relative_to(ROOT).parts))
    for p in ROOT.rglob("__init__.pyi")
)
# Subpackages gated on optional extras, found from their ``require_deps`` call.
OPTIONAL = {
    p.parent.name
    for p in ROOT.glob("*/__init__.py")
    if "require_deps(" in p.read_text(encoding="utf-8")
}


@cache
def _stub(package: str) -> ast.Module:
    path = ROOT.joinpath(*package.split(".")[1:], "__init__.pyi")
    return ast.parse(path.read_text(encoding="utf-8"))


def _stub_all(package: str) -> list[str]:
    for node in _stub(package).body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "__all__"
        ):
            return ast.literal_eval(node.value)
    msg = f"No literal __all__ in {package}/__init__.pyi"
    raise AssertionError(msg)


@pytest.mark.parametrize("package", STUBBED)
def test_stub_all_matches_runtime_all(package):
    module = pytest.importorskip(package)
    assert sorted(_stub_all(package)) == sorted(module.__all__)


def test_eager_names_are_bound_without_lazy_import():
    """Core names must exist in the module namespace, not via lazy __getattr__."""
    lazy = OPTIONAL | {
        alias.asname or alias.name
        for node in _stub("soundscapy").body
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] in OPTIONAL
        for alias in node.names
    }
    eager = set(soundscapy.__all__) - lazy
    assert eager <= set(vars(soundscapy)), eager - set(vars(soundscapy))
