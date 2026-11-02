"""Every runtime dependency has a recorded, non-empty licence (NFR-7, spec section 6.19)."""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_LICENCES = tomllib.loads((Path(__file__).parent / "licences.toml").read_text("utf-8"))["licences"]
_PYPROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text("utf-8"))


def _runtime_names() -> list[str]:
    names = []
    for spec in _PYPROJECT["project"]["dependencies"]:
        match = re.match(r"[A-Za-z0-9._-]+", spec)
        assert match is not None, spec
        names.append(match.group(0).lower().replace("_", "-"))
    return names


def test_every_runtime_dependency_has_a_licence_entry():
    missing = [name for name in _runtime_names() if name not in _LICENCES]
    assert not missing


@pytest.mark.parametrize("name", _runtime_names())
def test_installed_metadata_declares_a_licence(name):
    meta = metadata.metadata(name)
    expression = meta.get_all("License-Expression") or []
    licence = meta.get_all("License") or []
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    assert any(v.strip() for v in [*expression, *licence, *classifiers])
