"""Science tests of the bundled course dataset against its source (spec 6.20.4, 7.2).

The file is ``examples/data/DemoData.csv`` of soundscapy 0.x (commit 325e97c1), the
renamed ISD extract the IoA course uses, with three columns dropped; the counts were
measured on that file.
"""

from __future__ import annotations

import tomllib
from collections import Counter

import pytest

from soundscapy import _data
from soundscapy.provenance import hash_bytes
from tests._helpers import csv_rows

pytestmark = pytest.mark.science

_CSV = "datasets/course_demo/course_demo.csv"
_SIDECAR = "datasets/course_demo/course_demo.toml"
_SOURCE_SHA256 = "d6d4f937b53f02f4bdaa28e8401fe90061d0d9fde9da89c0ec898baaa4fcdd3a"
_SITES = {"TateModern": 152, "CamdenTown": 105, "MarchmontGarden": 105, "PancrasLock": 94}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return csv_rows(_data.read_bytes(_CSV).decode("utf-8"))


@pytest.fixture(scope="module")
def sidecar() -> dict:
    return tomllib.loads(_data.read_bytes(_SIDECAR).decode("utf-8"))


def test_course_csv_matches_its_pin():
    assert hash_bytes(_data.read_bytes(_CSV)) == _data.pin(_CSV)


def test_course_shape(rows, sidecar):
    assert len(rows) == 456
    assert len(rows[0]) == 34
    assert list(rows[0]) == sidecar["columns"]


def test_course_site_counts(rows):
    assert Counter(row["LocationID"] for row in rows) == _SITES


def test_course_sidecar(sidecar):
    assert sidecar["copy_hash"] == _data.pin(_CSV)
    assert sidecar["source_sha256"] == _SOURCE_SHA256
    assert sidecar["dropped_columns"] == ["free_response", "ISOPleasant", "ISOEventful"]
    assert sidecar["features"]["units_verified"] is False
    assert sidecar["citation"]["licence"] == "CC-BY-4.0"
    assert sidecar["n_rows"] == 456
    assert sidecar["name"] == "course-demo"
    assert sidecar["version"] == "1"
