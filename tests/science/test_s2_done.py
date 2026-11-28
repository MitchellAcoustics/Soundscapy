"""The S2 done-when (spec 9, row S2), end to end in one test.

It repeats assertions made elsewhere on purpose: ``uv run pytest tests/science/test_s2_done.py``
is the one command that shows the slice is done, as ``test_s1_done.py`` does for S1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soundscapy.datasets import course_demo, isd
from soundscapy.projection.project import project
from soundscapy.survey.mapping import SurveyMapping
from soundscapy.survey.read import read_survey
from soundscapy.survey.status import Reason
from tests._helpers import medium_warning, raw_mapping

pytestmark = pytest.mark.science

SCALE_0_4_CSV = Path(__file__).parents[1] / "survey" / "fixtures" / "survey_0_4.csv"


def test_s2_done():
    subset = isd.load(subset=True)
    counts = subset.status.counts()
    assert (counts["n"], counts["reason:incomplete_attributes"]) == (1552, 16)
    assert counts["reason:all_identical"] == 0
    assert counts["flag:all_midpoint"] == 10
    assert counts["reason:excessive_missing"] == 0
    assert counts["flag:under_age"] == 8
    assert counts["excluded"] == 16

    course = course_demo.load()
    assert course.n() == 456
    assert course.status.counts()["excluded"] == 0

    mapping = SurveyMapping.model_validate(raw_mapping({"response": "ID"}))
    excluded = read_survey(SCALE_0_4_CSV, mapping, on_scale_violation="exclude")
    both = (Reason.SCALE_VIOLATION, Reason.INCOMPLETE_ATTRIBUTES)
    assert excluded.status.reasons == (both, both, both)

    with medium_warning():  # the default profile's angle set flags the Mandarin rows
        assert project(subset).n_valid() == 1536
