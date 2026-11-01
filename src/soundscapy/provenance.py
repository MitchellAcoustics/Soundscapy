"""Hashing and the provenance envelope.

This is the only module that imports ``hashlib``. Every hash is
``h1-sha256:<64 lowercase hex>``.
"""

import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias

import pydantic

from soundscapy._model import FrozenModel
from soundscapy.errors import HashSchemeError

__all__ = [
    "HASH_PREFIX",
    "OPTIONAL_KEYS",
    "JSONValue",
    "canonical_json",
    "check_hash",
    "envelope",
    "hash_bytes",
    "hash_file",
    "hash_json",
    "show_versions",
]

HASH_PREFIX: Final = "h1-sha256:"

JSONValue: TypeAlias = (
    bool | int | float | str | Sequence["JSONValue"] | Mapping[str, "JSONValue"] | None
)

OPTIONAL_KEYS: Final = frozenset(
    {
        "input_hashes",
        "upstream",
        "quality_skipped",
        "profile",
        "profile_inherited",
        "overrides",
        "response_id_source",
        "read",
        "id_columns",
        "response_id_columns",
        "response_key_blank_ok",
    }
)

_LIBRARIES: Final = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "narwhals",
    "pydantic",
    "contourpy",
)
_HEX_DIGITS: Final = frozenset("0123456789abcdef")
_DIGEST_LEN: Final = 64


def hash_bytes(data: bytes | Iterable[bytes]) -> str:
    """Hash bytes, or the concatenation of an iterable of byte chunks.

    Parameters
    ----------
    data : bytes or iterable of bytes
        The content; an iterable is hashed as its concatenation, so large
        content can be streamed.

    Returns
    -------
    str
        ``h1-sha256:`` followed by 64 lowercase hex digits.
    """
    digest = hashlib.sha256()
    if isinstance(data, bytes):
        digest.update(data)
    else:
        for chunk in data:
            digest.update(chunk)
    return HASH_PREFIX + digest.hexdigest()


def hash_file(path: str | os.PathLike[str]) -> str:
    """Hash a file's raw bytes.

    Parameters
    ----------
    path : str or os.PathLike
        The file to hash.

    Returns
    -------
    str
        The ``h1`` hash of the file's bytes.
    """
    with Path(path).open("rb") as handle:
        return HASH_PREFIX + hashlib.file_digest(handle, "sha256").hexdigest()


def _check_json(value: object, path: str) -> None:
    """Raise if ``value`` is outside ``JSONValue`` or holds a non-finite float."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"non-finite float {value!r} at {path or '<root>'}"
            raise ValueError(msg)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"mapping key {key!r} at {path or '<root>'} is {type(key).__name__}, not str"
                raise TypeError(msg)
            _check_json(item, f"{path}.{key}" if path else key)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]")
        return
    msg = f"{type(value).__name__} at {path or '<root>'} is not a JSON value"
    raise TypeError(msg)


def canonical_json(value: pydantic.BaseModel | JSONValue) -> bytes:
    """Serialise a model or plain JSON value to its canonical UTF-8 bytes.

    Keys are sorted, separators are compact, non-ASCII text is kept, and
    non-finite floats are refused. Models are dumped with
    ``model_dump(mode="json", by_alias=True)``.

    Parameters
    ----------
    value : pydantic.BaseModel or JSONValue
        What to serialise.

    Returns
    -------
    bytes
        The canonical JSON.

    Raises
    ------
    ValueError
        If a float is not finite; the message names its path.
    TypeError
        If a mapping key is not a string or a value is not a JSON value; the
        message names its path.
    """
    plain = (
        value.model_dump(mode="json", by_alias=True)
        if isinstance(value, pydantic.BaseModel)
        else value
    )
    _check_json(plain, "")
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_json(value: pydantic.BaseModel | JSONValue) -> str:
    """Hash the canonical JSON of a model or plain JSON value.

    Parameters
    ----------
    value : pydantic.BaseModel or JSONValue
        What to hash.

    Returns
    -------
    str
        The ``h1`` hash of ``canonical_json()``.
    """
    return hash_bytes(canonical_json(value))


def check_hash(value: str) -> str:
    """Validate an ``h1`` hash string and return its hex digest.

    Parameters
    ----------
    value : str
        The hash string.

    Returns
    -------
    str
        The 64-character lowercase hex digest.

    Raises
    ------
    HashSchemeError
        If the prefix is not ``h1-sha256:`` or the digest is malformed.
    """
    if not value.startswith(HASH_PREFIX):
        msg = f"hash {value!r} does not use the {HASH_PREFIX!r} scheme"
        raise HashSchemeError(msg)
    digest = value[len(HASH_PREFIX) :]
    if len(digest) != _DIGEST_LEN or not set(digest) <= _HEX_DIGITS:
        msg = f"hash {value!r} does not hold 64 lowercase hex digits"
        raise HashSchemeError(msg)
    return digest


def _library_versions(extra_libraries: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (*_LIBRARIES, *extra_libraries):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue  # not installed (later slices add it as a dependency)
    return versions


def envelope(  # the spec's normative keys are separate inputs
    *,
    operation: str,
    settings: FrozenModel,
    input_hash: str,
    mapping_hash: str = "",
    data_refs: Sequence[Mapping[str, str]] = (),
    extra_libraries: Sequence[str] = (),
    upstream_run: str = "",
    **extra: object,
) -> dict[str, object]:
    """Build the provenance envelope of a result.

    Parameters
    ----------
    operation : str
        The public function name, for example ``"project"``.
    settings : FrozenModel
        The operation's settings; its hash is recorded as ``settings_hash``.
    input_hash : str
        The ``h1`` hash of the input.
    mapping_hash : str, optional
        The ``h1`` hash of the mapping, ``""`` where none is involved.
    data_refs : sequence of mapping, optional
        ``{kind, name, version, hash}`` for each shipped or registered data
        item used.
    extra_libraries : sequence of str, optional
        Further installed libraries to record, such as ``"polars"``.
    upstream_run : str, optional
        The run name the input came from, ``""`` when none.
    **extra
        Optional keys, each one of ``OPTIONAL_KEYS``.

    Returns
    -------
    dict
        The envelope. Libraries that are not installed are left out.

    Raises
    ------
    ValueError
        If an extra key is not in ``OPTIONAL_KEYS``.
    """
    unknown = sorted(set(extra) - OPTIONAL_KEYS)
    if unknown:
        msg = f"unknown provenance keys {unknown}; allowed: {sorted(OPTIONAL_KEYS)}"
        raise ValueError(msg)
    created = dt.datetime.now(dt.UTC).replace(microsecond=0)
    return {
        "soundscapy_version": importlib.metadata.version("soundscapy"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "libraries": _library_versions(extra_libraries),
        "hash_scheme": "h1",
        "data_refs": [dict(ref) for ref in data_refs],
        "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operation": operation,
        "input_hash": input_hash,
        "mapping_hash": mapping_hash,
        "settings_hash": hash_json(settings),
        "upstream_run": upstream_run,
        **{key: copy.deepcopy(value) for key, value in extra.items()},
    }


def show_versions() -> str:
    """Return the versions of soundscapy, Python, the platform and the libraries.

    Returns
    -------
    str
        Multi-line ``key: value`` text; nothing is printed.
    """
    lines = [
        f"soundscapy: {importlib.metadata.version('soundscapy')}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
    ]
    lines.extend(f"{k}: {v}" for k, v in _library_versions(()).items())
    return "\n".join(lines)
