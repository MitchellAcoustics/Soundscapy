"""
Slim-install behavior tests.

Verify that ``import soundscapy`` does not trigger any optional-dep
imports, and that the gates emit actionable ImportErrors when their
deps appear missing.
"""

import importlib
import importlib.util
import sys

import pytest

# Modules that must NOT be imported as a side effect of ``import soundscapy``.
_OPTIONAL_TRACKED = (
    "acoustic_toolbox",
    "maad",
    "mosqito",
    "rpy2",
)


def _drop_soundscapy_from_modules(monkeypatch) -> None:
    for name in [
        m for m in list(sys.modules) if m == "soundscapy" or m.startswith("soundscapy.")
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_import_soundscapy_does_not_pull_optional_deps(monkeypatch):
    """`import soundscapy` must not eagerly import audio/r dependencies."""
    _drop_soundscapy_from_modules(monkeypatch)
    # Snapshot which optional modules were already imported by the test runner.
    pre = {m for m in _OPTIONAL_TRACKED if m in sys.modules}

    importlib.import_module("soundscapy")

    post = {m for m in _OPTIONAL_TRACKED if m in sys.modules}
    newly_imported = post - pre
    assert not newly_imported, (
        f"`import soundscapy` triggered import of optional deps: {newly_imported}"
    )


def _block_modules(monkeypatch, *names: str) -> None:
    """Make ``importlib.util.find_spec`` return None for the given modules."""
    blocked = set(names)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):  # noqa: ANN202
        if name in blocked:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def test_audio_gate_hint_when_audio_missing(monkeypatch):
    _drop_soundscapy_from_modules(monkeypatch)
    _block_modules(monkeypatch, "acoustic_toolbox", "maad", "mosqito", "tqdm")

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("soundscapy.audio")
    msg = str(excinfo.value)
    assert "soundscapy[audio]" in msg
    assert "pip install 'soundscapy[audio]'" in msg


def test_spi_gate_hint_when_rpy2_missing(monkeypatch):
    _drop_soundscapy_from_modules(monkeypatch)
    _block_modules(monkeypatch, "rpy2")

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("soundscapy.spi")
    msg = str(excinfo.value)
    assert "soundscapy[r]" in msg
    assert "'rpy2'" in msg


def test_satp_gate_hint_when_rpy2_missing(monkeypatch):
    _drop_soundscapy_from_modules(monkeypatch)
    _block_modules(monkeypatch, "rpy2")

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("soundscapy.satp")
    assert "soundscapy[r]" in str(excinfo.value)


def test_r_wrapper_gate_hint_when_rpy2_missing(monkeypatch):
    _drop_soundscapy_from_modules(monkeypatch)
    _block_modules(monkeypatch, "rpy2")

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("soundscapy.r_wrapper")
    assert "soundscapy[r]" in str(excinfo.value)
