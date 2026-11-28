"""Nightly checks on the full ISD: the FR-1.5 counts, the registry's feature columns and the
course rows against their ISD rows (spec 6.7, 6.20.3 and 7.5; PRD FR-1.5).

These tests download ``ISD v1.0 Data.csv`` (about 2 MB) for both 1.0.1 versions, so they
carry the ``network`` marker and run in the nightly ``network`` job, not the default run.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from soundscapy import _data
from soundscapy.survey.mapping import read_mapping
from soundscapy.survey.read import read_survey
from tests._downloads import ISD_FILES, isd_csv
from tests._helpers import csv_rows

pytestmark = [pytest.mark.science, pytest.mark.network]

DEFAULT = "1.0.1-alpha.1"
KEY = ("LocationID", "SessionID", "GroupID", "RecordID")
ATTRIBUTES = ("pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous")
ATTRIBUTES += ("uneventful", "calm")
# Course column -> ISD column, for every column both files hold as the same number.
NUMERIC_PAIRS = {
    **{f"PAQ{i}": name for i, name in enumerate(ATTRIBUTES, start=1)},
    "traffic_noise": "ssi01",
    "other_noise": "ssi02",
    "human_sounds": "ssi03",
    "natural_sounds": "ssi04",
    "overall_sound_environment": "sss01",
    "appropriate": "sss02",
    "perceived_loud": "sss03",
    "visit_often": "sss04",
    "like_to_visit": "sss05",
    "age": "age00",
    "LAeq": "LAeq_L(A)",
    "LA10": "LAeq_L10(A)",
    "LA90": "LAeq_L90(A)",
    "LCeq": "LCeq_L(C)",
    "N5": "N_N5",
    "sharpness": "S_S",
    "roughness": "R_R",
    "tonality": "T_TonalityHMS",
}  # fmt: skip
MISSING_FROM_COURSE = {"PancrasLock/PancrasLock2/PL202/410", "TateModern/TateModern1/TM124/663"}


@pytest.fixture(scope="session")
def isd_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = os.environ.get("SOUNDSCAPY_DATA_DIR")
    return Path(base) / "test-downloads" if base else tmp_path_factory.mktemp("downloads")


@pytest.fixture(scope="module")
def isd_rows(isd_cache) -> list[dict[str, str]]:
    return csv_rows(isd_csv(DEFAULT, isd_cache).read_text(encoding="utf-8-sig"))


def _key(row: dict[str, str]) -> str:
    return "/".join(row[column].strip() for column in KEY)


def _number(text: str) -> float | None:
    return float(text) if text.strip() else None


@pytest.mark.parametrize("version", list(ISD_FILES))
def test_fr15_full_isd(isd_cache, version):
    """PRD FR-1.5: the default rules reproduce the 0.x tutorials' 56 exclusions (spec 6.7)."""
    with _data.as_path("datasets/isd/isd_mapping.toml") as mapping_path:
        mapping = read_mapping(mapping_path)
    counts = read_survey(isd_csv(version, isd_cache), mapping).status.counts()
    assert counts["n"] == 3589
    assert counts["reason:incomplete_attributes"] == 37
    assert counts["reason:all_identical"] == 19
    assert counts["flag:all_midpoint"] == 53
    assert counts["reason:excessive_missing"] == 0
    assert counts["flag:under_age"] == 52
    assert counts["excluded"] == 56
    assert counts["included"] == 3533


def test_registry_feature_columns_match_the_header(isd_cache):
    with isd_csv(DEFAULT, isd_cache).open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    features = columns[columns.index("RecordingLength") + 1 :]
    registry = _data.read_toml("datasets/isd/registry.toml")
    assert features == registry["versions"][DEFAULT]["feature_columns"]
    assert len(features) == 96


def test_course_rows_equal_isd_rows(isd_rows):
    """Spec 7.5: every course row is its ISD row on the four-part key, cell for cell."""
    isd_rows = {_key(row): row for row in isd_rows}
    course = csv_rows(_data.read_bytes("datasets/course_demo/course_demo.csv").decode("utf-8"))
    assert len(course) == 456
    sites = {row["LocationID"] for row in course}
    for row in course:
        isd_row = isd_rows[_key(row)]  # KeyError would mean a course row with no ISD row
        for course_column, isd_column in NUMERIC_PAIRS.items():
            # Exact: both files hold the same decimal values, "4" and "4.0" alike.
            assert _number(row[course_column]) == _number(isd_row[isd_column]), (
                _key(row),
                course_column,
            )
        assert row["start_time"] == isd_row["start_time"]
    course_keys = {_key(row) for row in course}
    isd_keys_at_sites = {key for key, row in isd_rows.items() if row["LocationID"] in sites}
    assert isd_keys_at_sites - course_keys == MISSING_FROM_COURSE
