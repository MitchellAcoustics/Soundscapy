"""Science tests of the shipped angle sets against their sources (spec 6.8.2 and 6.8.4).

``aletta-2024@1`` is checked against a transcription of Table 4 of Aletta et al. (2024)
and the tier rule of that paper's Table 2; ``satp@1.5`` against the two Markdown tables in
``design/literature/mitchell-et-al-2026-satp-chapter-4-notes.md`` (Tables 4.7 and 4.8 of
the chapter's author proof).
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from soundscapy import _data
from soundscapy.instrument import angle_set
from soundscapy.provenance import hash_bytes
from tests._helpers import csv_rows

pytestmark = pytest.mark.science

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).parent / "fixtures" / "aletta_2024_table4.csv"
_NOTES = _ROOT / "design/literature/mitchell-et-al-2026-satp-chapter-4-notes.md"
_ALETTA = "angle_sets/aletta-2024/1.toml"
_SATP = "angle_sets/satp/1.5.toml"
_POSITIONS = [f"PAQ{i}" for i in range(1, 9)]
# Spec 6.8.2's table of the six tier-only languages: Table 4's tier is replaced by the
# paper's stated rule for jpn and kor.
_ALETTA_TIER_ONLY = {
    "fra": "medium",
    "por": "medium",
    "jpn": "medium",
    "kor": "medium",
    "vie": "low",
    "zsm": "low",
}


def _markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Return the rows of the first Markdown table after ``heading`` as dicts."""
    after = text.split(heading, 1)[1].splitlines()
    first = next(i for i, line in enumerate(after) if line.startswith("|"))
    lines = list(itertools.takewhile(lambda line: line.startswith("|"), after[first:]))
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip the header and the separator
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


@pytest.fixture(scope="module")
def notes() -> str:
    return _NOTES.read_text(encoding="utf-8")


@pytest.mark.parametrize("relpath", [_ALETTA, _SATP])
def test_shipped_file_matches_its_pin(relpath):
    assert hash_bytes(_data.read_bytes(relpath)) == _data.pin(relpath)


def test_aletta_angles_equal_table_4():
    s = angle_set("aletta-2024", "1")
    fixture = {row["language"]: row for row in csv_rows(_FIXTURE.read_text(encoding="utf-8"))}
    assert len(fixture) == 12
    with_angles = {code: entry for code, entry in s.languages if entry.has_angles}
    assert set(with_angles) == set(fixture)
    for code, entry in with_angles.items():
        # Exact: the paper prints whole degrees and the file holds them as floats.
        assert entry.angles_deg == tuple(float(fixture[code][p]) for p in _POSITIONS), code
        assert entry.tier == "high"
        assert entry.note == ""


def test_aletta_tier_only_languages():
    s = angle_set("aletta-2024", "1")
    tier_only = {code: entry for code, entry in s.languages if not entry.has_angles}
    assert {code: entry.tier for code, entry in tier_only.items()} == _ALETTA_TIER_ONLY
    assert "Table 4" in tier_only["jpn"].note
    assert "Low" in tier_only["jpn"].note
    assert "56" in tier_only["kor"].note
    assert "High" in tier_only["kor"].note


def test_aletta_citation_names_the_generator():
    s = angle_set("aletta-2024", "1")
    assert "10.1016/j.apacoust.2024.110109" in s.citation.doi_or_url
    assert "scripts/make_aletta_2024.py" in s.citation.notes
    assert "3509da3" in s.citation.notes


def test_satp_tiers_equal_table_4_7(notes):
    s = angle_set("satp", "1.5")
    rows = _markdown_table(notes, "Table 4.7)")
    expected = {row["Language"]: row["Tier"].lower() for row in rows}
    assert len(expected) == 23
    assert {code: entry.tier for code, entry in s.languages} == expected
    tiers = [entry.tier for _, entry in s.languages]
    assert (tiers.count("high"), tiers.count("medium"), tiers.count("low")) == (13, 7, 3)


def test_satp_high_rows_equal_table_4_8(notes):
    s = angle_set("satp", "1.5")
    rows = _markdown_table(notes, "Table 4.8)")
    assert len(rows) == 13
    for row in rows:
        entry = s.by_language[row["Language"]]
        assert entry.tier == "high"
        # Exact: the proof prints the angles to one decimal place, all of them whole.
        assert entry.angles_deg == tuple(float(row[p]) for p in _POSITIONS), row["Language"]
    high_with_angles = {code for code, entry in s.languages if entry.tier == "high"}
    assert high_with_angles == {row["Language"] for row in rows}


def test_satp_cmn_keeps_the_aletta_angles():
    satp = angle_set("satp", "1.5").by_language["cmn"]
    aletta = angle_set("aletta-2024", "1").by_language["cmn"]
    assert satp.tier == "medium"
    assert aletta.tier == "high"
    assert satp.angles_deg == aletta.angles_deg
    assert "4.6.1" in satp.note


def test_satp_tier_only_languages_have_no_angles():
    s = angle_set("satp", "1.5")
    without = {code for code, entry in s.languages if not entry.has_angles}
    assert without == {"tha", "por", "bra", "kor", "fra", "jpn", "alb", "vie", "zsm"}


def test_satp_citation_names_the_proof():
    s = angle_set("satp", "1.5")
    assert "Soundscape Across Languages" in s.citation.citation
    assert re.search(r"Table 4\.7.*p\. 19", s.citation.notes)
    assert re.search(r"Table 4\.8.*p\. 20", s.citation.notes)
    assert "2026-08-13" in s.citation.notes


def test_shared_high_languages_agree():
    """The 11 High languages the two sets share carry the same angles (spec 6.8.4)."""
    aletta = angle_set("aletta-2024", "1").by_language
    satp = angle_set("satp", "1.5").by_language
    shared = {code for code, entry in satp.items() if entry.tier == "high" and code in aletta}
    assert len(shared) == 11
    for code in shared:
        assert satp[code].angles_deg == aletta[code].angles_deg, code
