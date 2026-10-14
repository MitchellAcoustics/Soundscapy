# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Helpers for deprecating import paths."""

import importlib
import warnings
from collections.abc import Callable, Iterable


def moved_to(
    module_name: str, new_module: str, names: Iterable[str]
) -> Callable[..., object]:
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

    Returns
    -------
    Callable[..., object]
        A function suitable for assigning to a module-level ``__getattr__``.

    """
    moved = frozenset(names)

    def getattr_(name: str, stacklevel: int = 2) -> object:
        if name not in moved:
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        warnings.warn(
            f"`{module_name}.{name}` has moved to `{new_module}.{name}`.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return getattr(importlib.import_module(new_module), name)

    return getattr_
