# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""Helpers for deprecating import paths."""

import importlib
import types
import warnings
from collections.abc import Callable, Iterable, Mapping


def moved_to(
    module_name: str,
    moves: Mapping[str, Iterable[str]],
    fallback: Callable[[str], object] | None = None,
    renamed_modules: Mapping[str, str] | None = None,
) -> Callable[[str], object]:
    """
    Build a module ``__getattr__`` that forwards moved names with a warning.

    Parameters
    ----------
    module_name
        ``__name__`` of the module the names used to live in.
    moves
        Maps the dotted path of each new module to the names that moved there.
    fallback
        ``__getattr__`` to defer to for any other name (e.g. a lazy loader).
        If omitted, other names raise ``AttributeError``.
    renamed_modules
        Maps an old attribute name to the dotted path of the module that
        replaced it (e.g. ``{"db": "soundscapy.databases"}``).

    Returns
    -------
    Callable[[str], object]
        A function suitable for assigning to a module-level ``__getattr__``.

    """
    moved = {name: new_module for new_module, names in moves.items() for name in names}
    renamed = renamed_modules or {}

    def getattr_(name: str) -> object:
        replacement = renamed.get(name)
        if replacement is not None:
            warnings.warn(
                f"`{module_name}.{name}` has been renamed to `{replacement}`.",
                DeprecationWarning,
                stacklevel=2,
            )
            return importlib.import_module(replacement)
        new_module = moved.get(name)
        if new_module is None:
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


def moved_module(
    module_name: str, moves: Mapping[str, Iterable[str]]
) -> types.ModuleType:
    """
    Build a stand-in for a module that no longer exists.

    Register the result in ``sys.modules[module_name]`` so that
    ``from <module_name> import X`` still works, with a warning. Use this instead
    of a stub file when the old name is now an attribute of the parent package.

    Parameters
    ----------
    module_name
        Dotted name of the removed module.
    moves
        Maps the dotted path of each new module to the names that moved there.

    Returns
    -------
    types.ModuleType
        A module whose attribute access forwards the moved names with a warning.

    """
    module = types.ModuleType(module_name)
    module.__getattr__ = moved_to(module_name, moves)  # type: ignore[method-assign]
    return module
