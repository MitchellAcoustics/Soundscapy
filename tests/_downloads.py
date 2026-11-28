"""Downloading the full ISD for the nightly network tests, with stdlib urllib only.

``datasets.isd.fetch`` and pooch arrive in slice S9, which replaces this module. Until
then the two registry rows below are test constants; only 1.0.1-alpha.1 is in the shipped
registry.
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING, Final

from soundscapy.provenance import hash_file

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

ISD_FILE: Final = "ISD v1.0 Data.csv"
ISD_FILES: Final[Mapping[str, tuple[str, str]]] = {
    "1.0.1-alpha.1": (
        "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv",
        "27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735",
    ),
    "1.0.1-alpha": (
        "https://zenodo.org/records/10639661/files/ISD%20v1.0%20Data.csv",
        "1e54e41c694b7054e137210dc632a7d0c0aa6eca91e4b5e6a59605ea6411e3cf",
    ),
}
"""Per ISD version, the Zenodo URL and sha256 of ``ISD v1.0 Data.csv`` (spec 6.20.3)."""


def isd_csv(version: str, cache: Path) -> Path:
    """Return the path of the ISD CSV for ``version``, downloading it into ``cache`` once.

    A cached copy is reused when its hash matches; otherwise the file is downloaded to a
    temporary name, checked, and renamed into place.

    Raises
    ------
    AssertionError
        If the download's hash is not the pinned one, naming both hashes.
    """
    url, sha256 = ISD_FILES[version]
    expected = "h1-sha256:" + sha256
    path = cache / "isd" / version / ISD_FILE
    if path.exists() and hash_file(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310  # fixed https URL from the spec's registry table
        partial.write_bytes(response.read())
    actual = hash_file(partial)
    assert actual == expected, f"{url}: downloaded {actual}, expected {expected}"
    partial.replace(path)
    return path
