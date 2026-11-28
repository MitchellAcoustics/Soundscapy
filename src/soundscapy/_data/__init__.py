"""Shipped data files and their hash pins.

Every read verifies the file's ``h1`` hash against ``PINS``, so an edited installed
file fails loudly at run time, not only in the test suite. Adding or changing a shipped
file means adding or changing its pin in the same commit; a correction after release is a
new version of the file, never an edit.
"""

from __future__ import annotations

import importlib.resources
import importlib.resources.abc
import tomllib
from typing import TYPE_CHECKING, Any, Final

from soundscapy.errors import DataIntegrityError, DataReferenceError
from soundscapy.provenance import hash_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager
    from pathlib import Path

__all__ = ["PINS", "as_path", "pin", "read_bytes", "read_toml"]

PINS: Final[Mapping[str, str]] = {
    "angle_sets/aletta-2024/1.toml": (
        "h1-sha256:9677722398d03cd02e109d0b25795495e4e2a9eaf9fa7f74126f1af6c0eb5e99"
    ),
    "angle_sets/satp/1.5.toml": (
        "h1-sha256:7757bdf1ad40b017f0b75aa3e9394cef3a11cacd4c6a12c84383fdf024d5e6ce"
    ),
    "circumplexes/iso-12913-outdoor/1.toml": (
        "h1-sha256:4ef2208390747505a2ddb326802995891e8481927fa0832e7e99740d444b8686"
    ),
    "datasets/course_demo/course_demo.csv": (
        "h1-sha256:7f9e0933a0b6da290a612659420586617c0a1b0ddcd74d69c4e66c314c8b7386"
    ),
    "datasets/course_demo/course_demo.toml": (
        "h1-sha256:3d6ebbbd72fa4333834de9ade671aa0561e0af9870f8d1eaabf4dd156d32bc42"
    ),
    "datasets/course_demo/mapping.toml": (
        "h1-sha256:08554ad29d5b264b875d132c659fb80e843bc8a5e3cd691db664779678b3b31b"
    ),
    "datasets/isd/isd_mapping.toml": (
        "h1-sha256:1c8c7af20fa8f75ad1fd81397cdd67d7fba05c47efd0fa1bd816aee96ab75974"
    ),
    "datasets/isd/registry.toml": (
        "h1-sha256:16118d3824d5a43feb8c5e383315a449b5326a76c46140bb991a0540927d783c"
    ),
    "datasets/isd/subset-v1.0.csv": (
        "h1-sha256:5bd19f7909750eddd8812350300267e974638e35ababc92660816435538783fb"
    ),
    "datasets/isd/subset-v1.0.toml": (
        "h1-sha256:5e68eaadad19212470b07159affb340ba58547fd55b4adc52144013fe745ebef"
    ),
    "profiles/iso-12913-3-2019/1.toml": (
        "h1-sha256:740750313499412ec15ed1391d35ce4ee78e43f1298e219f445160d4120c5fa8"
    ),
    "profiles/iso-12913-3-2024/1.toml": (
        "h1-sha256:2a042e7e9e15de96a3d99a613193ac21bf10c2b401a3d019e1eaede6973743d7"
    ),
    "profiles/sal-2026/1.toml": (
        "h1-sha256:7efcda72c0d0b15c97143e449da8bef4731af3d70fc45da4dbe7c40681aa4934"
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
        The ``h1`` hash recorded in ``PINS``.

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
        A context manager yielding the file's ``pathlib.Path``.

    Raises
    ------
    DataReferenceError
        If the path is not registered.
    DataIntegrityError
        If the bytes do not match the pin.
    """
    read_bytes(relpath)  # verify before handing out a path
    return importlib.resources.as_file(_resource(relpath))
