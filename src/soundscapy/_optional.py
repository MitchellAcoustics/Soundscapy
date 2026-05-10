"""
Optional-dependency gates for soundscapy.

Each optional subpackage's ``__init__.py`` calls :func:`require_deps` on
its first non-comment line so that direct imports either succeed or fail
with a uniform, actionable :class:`ImportError`.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Import name -> PyPI distribution name. Only needed when they differ.
_DIST_NAME = {
    "acoustic_toolbox": "acoustic-toolbox",
    "maad": "scikit-maad",
}


def _install_hint(extra: str, missing: Iterable[str]) -> str:
    items = [_DIST_NAME.get(m, m) for m in missing]
    pretty = ", ".join(repr(m) for m in items)
    return (
        f"{pretty} required for soundscapy[{extra}], not installed. "
        f"Install with:  pip install 'soundscapy[{extra}]'"
    )


def require_deps(modules: Iterable[str], *, extra: str) -> None:
    """
    Raise ``ImportError`` if any of ``modules`` is not installed.

    Uses :func:`importlib.util.find_spec` so the check itself does not
    trigger the listed modules' import side effects.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise ImportError(_install_hint(extra, missing))
