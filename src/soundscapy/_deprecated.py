# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Helpers for deprecating import paths."""

import importlib
import warnings
from collections.abc import Callable, Iterable


def moved_to(
    module_name: str,
    new_module: str,
    names: Iterable[str],
    fallback: Callable[[str], object] | None = None,
) -> Callable[[str], object]:
    """
    Build a module ``__getattr__`` that forwards moved names with a warning.

    Parameters
    ----------
    module_name
        ``__name__`` of the module the names used to live in.
    new_module
        Dotted path of the module the names now live in.
    names
        The names that moved.
    fallback
        ``__getattr__`` to defer to for any other name (e.g. a lazy loader).
        If omitted, other names raise ``AttributeError``.

    Returns
    -------
    Callable[[str], object]
        A function suitable for assigning to a module-level ``__getattr__``.

    """
    moved = frozenset(names)

    def getattr_(name: str) -> object:
        if name not in moved:
            if fallback is not None:
                return fallback(name)
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        warnings.warn(
            f"`{module_name}.{name}` has moved to `{new_module}.{name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(new_module), name)

    return getattr_
