"""Nightly check that ``aletta-2024/1.toml`` regenerates byte-identically from its source.

The test downloads ``sem-fit-ipsatized.csv`` (about 10 kB) from the pinned commit of
``MitchellAcoustics/SATP-circumplex-validation``, checks its sha256, runs
``scripts/make_aletta_2024.py`` into a temporary directory and compares the bytes with the
shipped file. It carries the ``reference`` marker and runs in the nightly ``reference`` job
(``task test:reference``), not the default run.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from soundscapy import _data

pytestmark = pytest.mark.reference

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "make_aletta_2024.py"
SHIPPED = "angle_sets/aletta-2024/1.toml"
SOURCE_URL = (
    "https://raw.githubusercontent.com/MitchellAcoustics/SATP-circumplex-validation/"
    "3509da3e9191586430582e42f5f86945aa4965c5/outputs/2024-06-13/sem-fit-ipsatized.csv"
)
SOURCE_SHA256 = "d3ad6da994013082090c27e4592ae5bcdf267a91e5e5fe2c7eb5804eef25df05"


@pytest.fixture(scope="module")
def source_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned CSV, downloaded once into the data cache and checked against its sha256."""
    base = os.environ.get("SOUNDSCAPY_DATA_DIR")
    cache = Path(base) / "test-downloads" if base else tmp_path_factory.mktemp("downloads")
    path = cache / "satp-circumplex-validation" / "sem-fit-ipsatized.csv"
    if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256):
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            SOURCE_URL, timeout=120
        ) as response:  # fixed https URL at a pinned commit
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        assert actual == SOURCE_SHA256, (
            f"{SOURCE_URL}: downloaded {actual}, expected {SOURCE_SHA256}"
        )
        path.write_bytes(data)
    return path


def _generate(source: Path, out: Path) -> subprocess.CompletedProcess[str]:
    # The project interpreter has tomlkit, the script's one dependency; the script's
    # directory is not on sys.path (-I), so nothing beside it can shadow an import.
    return subprocess.run(
        [sys.executable, "-I", str(GENERATOR), "--source", str(source), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_regeneration_is_byte_identical(source_csv, tmp_path):
    result = _generate(source_csv, tmp_path)
    assert result.returncode == 0, result.stderr
    regenerated = (tmp_path / "aletta-2024" / "1.toml").read_bytes()
    assert regenerated == _data.read_bytes(SHIPPED)


def test_tampered_source_is_refused(source_csv, tmp_path):
    tampered = tmp_path / "tampered.csv"
    tampered.write_bytes(source_csv.read_bytes().replace(b'"314"', b'"315"', 1))
    result = _generate(tampered, tmp_path / "out")
    assert result.returncode != 0
    assert "refusing" in result.stderr
    assert SOURCE_SHA256 in result.stderr
    assert not (tmp_path / "out").exists()
