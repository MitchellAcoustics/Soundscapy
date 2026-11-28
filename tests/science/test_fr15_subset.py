"""FR-1.5 on the bundled subset: the default quality rules reproduce the 0.x processing.

The acceptance test of PRD FR-1.5 (spec 6.7): the defaults give the subset counts measured on
ISD 1.0.1-alpha.1, 16 excluded of 1552, all for a missing attribute item, with 10 all-midpoint
and 8 under-age rows flagged and kept. The full-ISD counts run nightly (``network``).
"""

from __future__ import annotations

import pytest

from soundscapy.datasets import isd

pytestmark = pytest.mark.science


def test_fr15_subset_counts():
    survey = isd.load(subset=True)
    counts = survey.status.counts()
    assert counts["n"] == 1552
    assert counts["reason:incomplete_attributes"] == 16
    assert counts["reason:all_identical"] == 0
    assert counts["flag:all_midpoint"] == 10
    assert counts["reason:excessive_missing"] == 0
    assert counts["flag:under_age"] == 8
    assert counts["excluded"] == 16
    assert counts["included"] == 1536
    expected = {
        "n", "included", "excluded", "reason:incomplete_attributes", "flag:all_midpoint",
        "flag:under_age",
    }  # fmt: skip
    assert {key for key, value in counts.items() if value} == expected
    assert survey.provenance["quality_skipped"] == ["no_consent", "test_response", "paper_input"]
    assert list(survey.scores) == [
        "attributes",
        "source_dominance",
        "overall_quality",
        "appropriateness",
        "who5",
    ]
