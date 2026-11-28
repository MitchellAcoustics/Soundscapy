"""The shipped outdoor instrument against a fixture laid out like ISO/TS 12913 Table A.1.

The fixture is not a transcription from the standard. Its Part 2 rows are the soundscapy 0.x
wording and its Part 3 rows come from the design spec, so these tests prove the shipped file
matches that fixture; conformance to the standard awaits the owner's check.

The fixture ``fixtures/iso12913_3_table_a1.csv`` has one row per scale step of Part 2
(ISO/TS 12913-2:2018 Table A.1: the agreement wording, coded 1 = strongly disagree to
5 = strongly agree) and one row per axis end of Part 3 (ISO/TS 12913-3:2019 Annex A: the
attribute at each end of the two axes and the direction it points in). The Part 2 wording
is the soundscapy 0.x wording the shipped file carries; the standards are paywalled, so the
wording awaits the owner's check against the standard (the shipped file's ``notes`` says
so too). The Part 3 rows are taken from the spec's description of the shipped instrument
(design spec section 6.2.3), itself written from the standard.
"""

from __future__ import annotations

import csv
from pathlib import Path

from soundscapy.instrument import circumplex
from tests._helpers import csv_rows

TABLE_A1 = Path(__file__).parent / "fixtures" / "iso12913_3_table_a1.csv"
# The sign and axis each Part 3 direction note stands for.
DIRECTIONS = {"+x": (0, "positive"), "-x": (0, "negative"), "+y": (1, "positive")}
DIRECTIONS["-y"] = (1, "negative")


def _rows(part: str) -> list[dict[str, str]]:
    return [row for row in csv_rows(TABLE_A1.read_text(encoding="utf-8")) if row["part"] == part]


def test_fixture_has_the_four_columns():
    with TABLE_A1.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == ["part", "step", "label", "direction"]


def test_shipped_scale_labels_match_the_table_a1_part_2_fixture():
    rows = _rows("2")
    assert [row["step"] for row in rows] == ["1", "2", "3", "4", "5"]
    assert {row["direction"] for row in rows} == {"1 = strongly disagree to 5 = strongly agree"}
    labels = dict(circumplex().scale.labels)["eng"]
    assert list(labels) == [row["label"] for row in rows]


def test_shipped_axes_match_the_table_a1_part_3_fixture():
    rows = _rows("3")
    c = circumplex()
    assert [row["direction"] for row in rows] == ["+x", "-x", "+y", "-y"]
    for row in rows:
        axis_index, end = DIRECTIONS[row["direction"]]
        position = getattr(c.axes[axis_index], end)
        assert row["step"] == position
        assert row["label"] == c.label(position)
