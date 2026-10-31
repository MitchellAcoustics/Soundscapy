"""Shipped data files and their hash pins.

Every read verifies the file's ``h1`` hash against :data:`PINS`, so an edited installed
file fails loudly at run time, not only in the test suite. Adding or changing a shipped
file means adding or changing its pin in the same commit; a correction after release is a
new version of the file, never an edit.
"""

import importlib.resources
import importlib.resources.abc
import tomllib
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Final

from soundscapy.errors import DataIntegrityError, DataReferenceError
from soundscapy.provenance import hash_bytes

__all__ = ["PINS", "as_path", "pin", "read_bytes", "read_toml"]

PINS: Final[Mapping[str, str]] = {
    "circumplexes/iso-12913-outdoor/1.toml": (
        "h1-sha256:4ef2208390747505a2ddb326802995891e8481927fa0832e7e99740d444b8686"
    ),
}
"""Posix path relative to ``_data/`` mapped to the ``h1`` hash of the file's bytes."""


def _resource(relpath: str) -> importlib.resources.abc.Traversable:
    return importlib.resources.files(__name__).joinpath(relpath)


def _raw_bytes(relpath: str) -> bytes:
    """Read a shipped file without verifying it (the seam the tests tamper with)."""
    return _resource(relpath).read_bytes()


def pin(relpath: str) -> str:
    """Return the registered hash of a shipped file.

    Parameters
    ----------
    relpath : str
        Posix path relative to the ``_data`` directory.

    Returns
    -------
    str
        The ``h1`` hash recorded in :data:`PINS`.

    Raises
    ------
    DataReferenceError
        If the path is not registered.
    """
    try:
        return PINS[relpath]
    except KeyError:
        msg = f"{relpath!r} is not a registered data file"
        raise DataReferenceError(msg, hint=f"registered: {sorted(PINS)}") from None


def read_bytes(relpath: str) -> bytes:
    """Read a shipped file and verify it against its pin.

    Parameters
    ----------
    relpath : str
        Posix path relative to the ``_data`` directory.

    Returns
    -------
    bytes
        The file's bytes.

    Raises
    ------
    DataReferenceError
        If the path is not registered.
    DataIntegrityError
        If the bytes do not match the pin; the message names both hashes.
    """
    expected = pin(relpath)
    data = _raw_bytes(relpath)
    actual = hash_bytes(data)
    if actual != expected:
        msg = f"data file {relpath!r} has hash {actual}, expected {expected}"
        raise DataIntegrityError(msg)
    return data


def read_toml(relpath: str) -> dict[str, Any]:
    """Read a shipped TOML file, verified against its pin.

    Parameters
    ----------
    relpath : str
        Posix path relative to the ``_data`` directory.

    Returns
    -------
    dict
        The parsed TOML table.

    Raises
    ------
    DataReferenceError
        If the path is not registered.
    DataIntegrityError
        If the bytes do not match the pin.
    """
    return tomllib.loads(read_bytes(relpath).decode("utf-8"))


def as_path(relpath: str) -> AbstractContextManager[Path]:
    """Give a verified shipped file as a filesystem path.

    Parameters
    ----------
    relpath : str
        Posix path relative to the ``_data`` directory.

    Returns
    -------
    contextlib.AbstractContextManager
        A context manager yielding the file's :class:`~pathlib.Path`.

    Raises
    ------
    DataReferenceError
        If the path is not registered.
    DataIntegrityError
        If the bytes do not match the pin.
    """
    read_bytes(relpath)  # verify before handing out a path
    return importlib.resources.as_file(_resource(relpath))
