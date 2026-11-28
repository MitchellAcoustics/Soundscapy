"""A second instrument through the public path: read_survey, project and plot.scatter.

The synthetic indoor circumplex under ``tests/fixtures/indoor/`` has six attributes at 60
degree steps, axes ``Comfort`` and ``Content`` and a 1..7 scale, so it differs from the
shipped outdoor instrument in attribute count, axis ids, labels and scale (PRD FR-2.0). It
reaches ``read_survey`` through the ``indoor_circumplex_registry`` fixture, which overrides
the shipped-data registry for one test; the package code runs unchanged.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from soundscapy import _data, plot
from soundscapy._model import from_toml
from soundscapy.distribution.coordinates import valid_mask
from soundscapy.errors import DataReferenceError, FormulationError
from soundscapy.instrument import Circumplex, Scale, circumplex
from soundscapy.projection.project import project
from soundscapy.survey.read import read_survey
from soundscapy.survey.status import Reason

INDOOR_REL = "circumplexes/indoor-test/1.toml"
LABELS = ("comfortable", "lively", "engaging", "uncomfortable", "oppressive", "dull")
SCALE_1_7 = (1.0, 7.0, "ordinal")  # minimum, maximum, kind


def _bounds(scale: Scale) -> tuple[float, float, str]:
    return (scale.minimum, scale.maximum, scale.kind)


def test_fixture_file_validates_as_a_circumplex(indoor_circumplex_registry):
    text = (indoor_circumplex_registry / "circumplex.toml").read_text(encoding="utf-8")
    c = from_toml(Circumplex, text)
    assert (c.name, c.version) == ("indoor-test", "1")
    assert c.positions == tuple(f"PAQ{i}" for i in range(1, 7))
    # Exact: multiples of 60.0 are exactly representable.
    assert c.angles_deg == tuple(60.0 * i for i in range(6))
    assert tuple(c.label(p) for p in c.positions) == LABELS
    assert tuple(a.id for a in c.axes) == ("Comfort", "Content")
    assert _bounds(c.scale) == SCALE_1_7
    assert len(dict(c.scale.labels)["eng"]) == 7
    assert circumplex("indoor-test", "1") == c


def _read(indoor):
    return read_survey(indoor / "survey.csv", indoor / "mapping.toml")


def test_read_survey_codes_six_attributes_on_1_to_7(indoor_circumplex_registry):
    survey = _read(indoor_circumplex_registry)
    scores = survey.scores["attributes"]
    assert scores.shape == (12, 6)
    assert _bounds(survey.mapping.groups.attributes.scale) == SCALE_1_7
    valid = scores[np.isfinite(scores)]
    assert valid.min() == 1.0
    assert valid.max() == 7.0
    assert survey.status.included.tolist() == [True] * 11 + [False]
    assert survey.status.reasons[11] == (Reason.INCOMPLETE_ATTRIBUTES,)
    circumplex_ref = survey.provenance["data_refs"][0]
    assert circumplex_ref["name"] == "indoor-test"
    assert circumplex_ref["hash"] == _data.PINS[INDOOR_REL]


def test_project_with_equal_angles(indoor_circumplex_registry):
    coords = project(_read(indoor_circumplex_registry), angles="equal")
    xy = coords.xy[valid_mask(coords)]
    assert xy.shape == (11, 2)
    assert (np.abs(xy) <= 1.0).all()
    frame = coords.to_frame()
    assert list(frame.columns[-2:]) == ["ssp_Comfort", "ssp_Content"]
    assert np.isnan(coords.xy[11]).all()
    (segment,) = coords.origin
    assert _bounds(segment.effective_scale) == SCALE_1_7
    assert segment.angle_sets == ("equal",)


def test_iso2019_is_refused_on_1_to_7(indoor_circumplex_registry):
    survey = _read(indoor_circumplex_registry)
    with pytest.raises(FormulationError):
        project(survey, formulation="iso2019", angles="equal")


def test_project_with_the_2024_profile(indoor_circumplex_registry):
    survey = _read(indoor_circumplex_registry)
    by_profile = project(survey, profile="iso-12913-3-2024")
    equal = project(survey, angles="equal")
    # Exact: the 2024 profile is the canonical formulation at equal angles.
    np.testing.assert_array_equal(by_profile.xy, equal.xy)
    assert by_profile.provenance["profile"] == {
        "id": "iso-12913-3-2024",
        "version": "1",
        "hash": _data.pin("profiles/iso-12913-3-2024/1.toml"),
    }


def test_default_profile_names_both_circumplexes(indoor_circumplex_registry):
    survey = _read(indoor_circumplex_registry)
    with pytest.raises(DataReferenceError, match=re.escape("iso-12913-outdoor@1")) as info:
        project(survey)
    assert "indoor-test@1" in str(info.value)
    assert "satp@1.5" in str(info.value)


def test_scatter_draws_the_indoor_labels(indoor_circumplex_registry):
    coords = project(_read(indoor_circumplex_registry), angles="equal")
    ax = plot.scatter(coords)
    assert {t.get_text() for t in ax.texts} == set(LABELS)
    assert ax.get_xlabel() == "Comfort"
    assert ax.get_ylabel() == "Content"
    # No German translation: the labels fall back to the reference language.
    german = plot.scatter(coords, label_language="deu")
    assert {t.get_text() for t in german.texts} == set(LABELS)


@pytest.mark.usefixtures("indoor_circumplex_registry")
def test_override_is_visible_inside_the_fixture():
    assert INDOOR_REL in _data.PINS
    assert circumplex("indoor-test", "1").name == "indoor-test"


def test_registry_is_restored_after_the_fixture():
    """Runs after the test above: the override left nothing behind."""
    assert INDOOR_REL not in _data.PINS
    with pytest.raises(DataReferenceError, match="indoor-test"):
        circumplex("indoor-test", "1")
    outdoor = circumplex("iso-12913-outdoor", "1")
    assert outdoor.name == "iso-12913-outdoor"
    assert len(outdoor.positions) == 8
