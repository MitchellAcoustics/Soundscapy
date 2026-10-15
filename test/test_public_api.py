# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Guard that each ``__init__.pyi``'s literal ``__all__`` matches the runtime one.

See the header of ``soundscapy/__init__.pyi`` for why the stubs carry the API.
"""

import ast
import contextlib
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


def _lazy_root_names() -> set[str]:
    """Top-level names that come from optional subpackages (and so load lazily)."""
    return OPTIONAL | {
        alias.asname or alias.name
        for node in _stub("soundscapy").body
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] in OPTIONAL
        for alias in node.names
    }


def test_eager_names_are_bound_without_lazy_import():
    """Core names must exist in the module namespace, not via lazy __getattr__."""
    eager = set(soundscapy.__all__) - _lazy_root_names()
    assert eager <= set(vars(soundscapy)), eager - set(vars(soundscapy))


@pytest.mark.parametrize("package", STUBBED)
def test_every_stub_name_resolves(package):
    """``__all__`` is derived from the stub, so check each name really exists."""
    module = pytest.importorskip(package)
    # Names that come from an optional subpackage can only resolve when its extra
    # is installed; in a slim install they raise the uniform gate ImportError.
    optional = _lazy_root_names() if package == "soundscapy" else set()
    for name in module.__all__:
        if name in optional:
            with contextlib.suppress(ImportError):
                getattr(module, name)
        else:
            getattr(module, name)
