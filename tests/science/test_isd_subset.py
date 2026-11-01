"""Science tests of the bundled ISD subset against the counts measured on its source.

The subset is ISD 1.0.1-alpha.1 (Zenodo record 10672568, CC BY 4.0) cut to 15 locations
and 40 columns by ``scripts/make_isd_subset.py``; the counts were measured on that file.
"""

from __future__ import annotations

import csv
import io
import tomllib
from collections import Counter

import pytest

from soundscapy import _data
from soundscapy.provenance import hash_bytes

pytestmark = pytest.mark.science

_CSV = "datasets/isd/subset-v1.0.csv"
_SIDECAR = "datasets/isd/subset-v1.0.toml"
_LOCATIONS = {
    "CamdenTown",
    "EustonTap",
    "MarchmontGarden",
    "PancrasLock",
    "RegentsParkFields",
    "RegentsParkJapan",
    "RussellSq",
    "StPaulsCross",
    "StPaulsRow",
    "TateModern",
    "TorringtonSq",
    "CampoPrincipe",
    "SanMarco",
    "Noorderplantsoen",
    "PingshanPark",
}
_COLUMNS = [
    "LocationID", "SessionID", "GroupID", "RecordID", "start_time", "Language",
    "Survey_Version", "ssi01", "ssi02", "ssi03", "ssi04", "pleasant", "chaotic", "vibrant",
    "uneventful", "calm", "annoying", "eventful", "monotonous", "sss01", "sss02", "sss03",
    "sss04", "sss05", "who01", "who02", "who03", "who04", "who05", "age00", "LCeq_L(C)",
    "LAeq_L(A)", "LAeq_L10(A)", "LAeq_L90(A)", "N_N5", "S_S", "R_R", "T_TonalityHMS",
    "FS_F", "I_I",
]  # fmt: skip
_CITATION = {
    "citation": (
        "Mitchell, A., Oberman, T., Aletta, F., Erfanian, M., Kachlicka, M., Lionello, M., "
        "Fang, X., and Kang, J. (2024). The International Soundscape Database: An integrated "
        "multimedia database of urban soundscape surveys -- questionnaires with acoustical "
        "and contextual information (version 1.0.1-alpha.1) [Data set]. Zenodo."
    ),
    "doi_or_url": "https://doi.org/10.5281/zenodo.10672568",
    "licence": "CC-BY-4.0",
    "date": "2024-02-16",
    "notes": (
        "Subset of ISD 1.0.1-alpha.1 (Zenodo record 10672568; all versions: concept DOI "
        "10.5281/zenodo.5578572). 15 locations and 40 columns, cell text unchanged. Made by "
        "scripts/make_isd_subset.py."
    ),
}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    text = _data.read_bytes(_CSV).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def test_subset_matches_its_pin():
    assert hash_bytes(_data.read_bytes(_CSV)) == _data.pin(_CSV)


def test_subset_is_plain_utf8_with_lf_line_endings():
    data = _data.read_bytes(_CSV)
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data


def test_subset_shape(rows):
    assert len(rows) == 1552
    assert list(rows[0]) == _COLUMNS


def test_subset_language_counts(rows):
    counts = Counter(row["Language"] for row in rows)
    assert counts == {"eng": 1213, "spa": 126, "nld": 89, "cmn": 69, "ita": 55}


def test_subset_locations(rows):
    assert {row["LocationID"] for row in rows} == _LOCATIONS


def test_subset_feature_rows(rows):
    assert sum(row["LAeq_L(A)"].strip() != "" for row in rows) == 1350


def test_sidecar_citation_and_hash():
    sidecar = tomllib.loads(_data.read_bytes(_SIDECAR).decode("utf-8"))
    assert sidecar["citation"] == _CITATION
    assert sidecar["subset_hash"] == _data.pin(_CSV)
    assert sidecar["n_rows"] == 1552
    assert sidecar["columns"] == _COLUMNS
    assert set(sidecar["locations"]) == _LOCATIONS
