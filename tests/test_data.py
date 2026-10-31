"""The data registry: every shipped file is pinned and verified on read."""

import importlib.resources

import pytest

from soundscapy import _data
from soundscapy.errors import DataIntegrityError, DataReferenceError
from soundscapy.provenance import hash_bytes

_REL = "circumplexes/iso-12913-outdoor/1.toml"


def _shipped() -> list[str]:
    root = importlib.resources.files("soundscapy._data")
    with importlib.resources.as_file(root) as base:
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and p.suffix not in {".py", ".pyc"} and "__pycache__" not in p.parts
        )


def test_every_shipped_file_matches_its_pin():
    for rel in _shipped():
        assert hash_bytes(_data.read_bytes(rel)) == _data.pin(rel)


def test_every_shipped_file_is_pinned():
    shipped = _shipped()
    assert set(shipped) == set(_data.PINS)
    assert shipped, "the registry should not be empty"


def test_read_toml_and_as_path():
    assert _data.read_toml(_REL)["name"] == "iso-12913-outdoor"
    with _data.as_path(_REL) as path:
        assert path.name == "1.toml"
        assert hash_bytes(path.read_bytes()) == _data.pin(_REL)


def test_tampered_bytes_raise(monkeypatch):
    monkeypatch.setattr(_data, "_raw_bytes", lambda _relpath: b"altered")
    with pytest.raises(DataIntegrityError) as info:
        _data.read_bytes(_REL)
    assert hash_bytes(b"altered") in str(info.value)
    assert _data.PINS[_REL] in str(info.value)
    with pytest.raises(DataIntegrityError):
        _data.read_toml(_REL)
    with pytest.raises(DataIntegrityError):
        _data.as_path(_REL)


def test_unregistered_path_raises():
    with pytest.raises(DataReferenceError):
        _data.read_bytes("circumplexes/nope/1.toml")
    with pytest.raises(DataReferenceError):
        _data.pin("nope")
