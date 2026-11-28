"""Science tests of the shipped translated labels against their source (spec 6.2.3).

The eighteen translations of ``iso-12913-outdoor@1`` come from Table 4 of Aletta et al.
(2024), which has no text layer, so two agents transcribed the rendered page without
seeing each other's work. The first wrote the TOML translations; the second wrote
``fixtures/aletta_2024_table4_labels.csv``. Every cell is compared here: a mismatch is
a question for the owner, not something to edit away on either side.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from soundscapy.instrument import circumplex
from tests._helpers import csv_rows

pytestmark = pytest.mark.science

_FIXTURE = Path(__file__).parent / "fixtures" / "aletta_2024_table4_labels.csv"
_POSITIONS = [f"PAQ{i}" for i in range(1, 9)]
# Table 4's language blocks in print order.
_LANGUAGES = [
    "arb", "cmn", "hrv", "nld", "fra", "deu", "ell", "ind", "ita",
    "jpn", "kor", "zsm", "por", "spa", "swe", "tur", "vie",
]  # fmt: skip
# Medium and Low-tier languages of satp@1.5 with no published labels (spec 6.2.3).
_GAP_LANGUAGES = ["pol", "yue", "tha", "bra", "alb"]


def _normalise(label: str) -> str:
    """Make whitespace around slashes, runs of whitespace and NFC the same on both sides."""
    text = re.sub(r"\s+", " ", re.sub(r"\s*/\s*", " / ", label)).strip()
    return unicodedata.normalize("NFC", text)


@pytest.fixture(scope="module")
def pass2() -> dict[str, dict[str, str]]:
    rows = csv_rows(_FIXTURE.read_text(encoding="utf-8"))
    return {row["language"]: row for row in rows}


def test_fixture_covers_the_table(pass2):
    assert sorted(pass2) == sorted(["eng", *_LANGUAGES])
    for row in pass2.values():
        assert all(row[position].strip() for position in _POSITIONS)


def test_english_row_is_the_reference_labels(pass2):
    # The table capitalises the English labels; the file keeps them in lowercase.
    c = circumplex()
    for position in _POSITIONS:
        assert pass2["eng"][position].casefold() == c.label(position)


def test_every_source_language_is_translated_with_a_citation():
    c = circumplex()
    assert c.languages == ("eng", *_LANGUAGES)
    by_language = {tr.language: tr for tr in c.translations}
    assert sorted(by_language) == sorted(_LANGUAGES)
    for tr in by_language.values():
        assert tuple(position for position, _ in tr.labels) == tuple(_POSITIONS)
        assert all(label.strip() for _, label in tr.labels)
        assert tr.citation.citation
        assert "10.1016/j.apacoust.2024.110109" in tr.citation.doi_or_url


@pytest.mark.parametrize("language", _LANGUAGES)
def test_pass1_and_pass2_agree_cell_by_cell(pass2, language):
    c = circumplex()
    mismatches = {
        position: (c.label(position, language), pass2[language][position])
        for position in _POSITIONS
        if _normalise(c.label(position, language)) != _normalise(pass2[language][position])
    }
    assert not mismatches, f"{language}: pass 1 vs pass 2 {mismatches}"


@pytest.mark.parametrize("language", _GAP_LANGUAGES)
def test_gap_languages_fall_back_to_english(language):
    c = circumplex()
    assert language not in c.languages
    for position in _POSITIONS:
        assert c.label(position, language) == c.label(position)
