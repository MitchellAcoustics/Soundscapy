"""project: a survey projected with equal angles into Coordinates."""

import dataclasses
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from soundscapy._model import COL_ANGLE_SET, COL_INCLUDED, COL_REASONS, COL_RESPONSE_ID
from soundscapy.distribution.coordinates import n, n_valid, valid_mask
from soundscapy.errors import FormulationError, SettingsError
from soundscapy.instrument import circumplex
from soundscapy.instrument.angles import AngleSelection
from soundscapy.projection.project import ProjectionSettings, project
from soundscapy.provenance import hash_json
from soundscapy.survey.read import read_survey

FIXTURES = Path(__file__).parent / "fixtures"
MAPPING = FIXTURES / "mapping.toml"
MAPPING_1_7 = FIXTURES / "mapping_1_7.toml"
HAND_CSV = FIXTURES / "hand_values.csv"
SCALE_1_7_CSV = FIXTURES / "scale_1_7.csv"
ORDER_CSV = FIXTURES / "order.csv"

# The two ISO/TS 12913-3:2019 hand values, derived in closed form (spec section 7.2).
HAND_XY = np.array(
    [
        [math.sqrt(2) - 1, 0.0],
        [1.5 - 1 / math.sqrt(2), 0.5 - 1 / math.sqrt(2)],
    ]
)


@pytest.fixture(scope="module")
def hand_survey():
    return read_survey(HAND_CSV, MAPPING)


def test_hand_values_iso2019(hand_survey):
    coords = project(hand_survey, formulation="iso2019", angles="equal")
    # 1e-12: float64 sums of 8 terms against the closed form, as the kernel's tests.
    np.testing.assert_allclose(coords.xy[:2], HAND_XY, rtol=0, atol=1e-12)


def test_default_is_canonical_with_the_same_values(hand_survey):
    coords = project(hand_survey)
    assert coords.settings == ProjectionSettings()
    assert isinstance(coords.settings, ProjectionSettings)
    assert coords.settings.formulation == "canonical"
    # 1e-12: canonical equals iso2019 at 45 degrees on 1..5 up to float64 rounding.
    np.testing.assert_allclose(coords.xy[:2], HAND_XY, rtol=0, atol=1e-12)


def test_excluded_row_is_nan_and_has_no_angle_set(hand_survey):
    coords = project(hand_survey)
    assert np.isnan(coords.xy[2]).all()
    assert coords.labels[COL_INCLUDED].tolist() == [True, True, False]
    assert coords.labels[COL_REASONS].tolist() == ["", "", "incomplete_attributes"]
    assert coords.labels[COL_ANGLE_SET].tolist() == ["equal", "equal", ""]


def test_labels_in_column_order(hand_survey):
    coords = project(hand_survey)
    assert list(coords.labels) == [
        "ssp_response_id",
        "ssp_included",
        "ssp_reasons",
        "ssp_flags",
        "ssp_language",
        "ssp_angle_set",
    ]
    assert coords.labels[COL_RESPONSE_ID].tolist() == ["h1", "h2", "h3"]
    assert coords.labels["ssp_language"].tolist() == ["eng"] * 3


def test_survey_is_unchanged(hand_survey):
    before = {
        "scores": hand_survey.scores["attributes"].copy(),
        "response_id": hand_survey.response_id.copy(),
        "language": hand_survey.language.copy(),
        "included": hand_survey.status.included.copy(),
        "reasons": hand_survey.status.reasons,
        "provenance": dict(hand_survey.provenance),
    }
    project(hand_survey)
    np.testing.assert_array_equal(hand_survey.scores["attributes"], before["scores"])
    np.testing.assert_array_equal(hand_survey.response_id, before["response_id"])
    np.testing.assert_array_equal(hand_survey.language, before["language"])
    np.testing.assert_array_equal(hand_survey.status.included, before["included"])
    assert hand_survey.status.reasons == before["reasons"]
    assert hand_survey.provenance == before["provenance"]


def test_project_uses_the_declared_scale():
    survey = read_survey(SCALE_1_7_CSV, MAPPING_1_7)
    coords = project(survey)
    # abs=1e-12: closed form x = 2 / (2 + 2*sqrt(2)) = sqrt(2) - 1 on the 1..7 scale; the
    # circumplex's 1..5 scale would give 3 / (2 + 2*sqrt(2)) = 0.6213.
    assert coords.xy[0, 0] == pytest.approx(math.sqrt(2) - 1, abs=1e-12)
    assert coords.xy[0, 1] == pytest.approx(0.0, abs=1e-12)  # closed form
    assert coords.origin[0].effective_scale == survey.mapping.groups.attributes.scale
    with pytest.raises(FormulationError):
        project(survey, formulation="iso2019")


@pytest.mark.parametrize("angles", ["satp@1.5", "aletta-2024@1"])
def test_project_rejects_unknown_angles(hand_survey, angles):
    with pytest.raises(SettingsError) as info:
        project(hand_survey, angles=angles)
    assert type(info.value) is SettingsError
    assert "angles" in str(info.value)
    assert '"equal"' in str(info.value)
    assert "validation error" not in str(info.value)


@pytest.mark.parametrize("formulation", ["aletta2024", "iso", 3])
def test_project_rejects_unknown_formulation(hand_survey, formulation):
    with pytest.raises(SettingsError) as info:
        project(hand_survey, formulation=formulation)
    assert type(info.value) is SettingsError
    assert "formulation" in str(info.value)
    assert '"canonical"' in str(info.value)
    assert '"iso2019"' in str(info.value)
    assert "validation error" not in str(info.value)


def test_project_rejects_angles_of_another_type(hand_survey):
    with pytest.raises(SettingsError, match="angles"):
        project(hand_survey, angles=45)  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime SettingsError


def test_angle_selection_equals_equal(hand_survey):
    by_name = project(hand_survey, angles="equal")
    by_model = project(hand_survey, angles=AngleSelection())
    np.testing.assert_array_equal(by_model.xy, by_name.xy)
    assert by_model.settings == by_name.settings
    assert by_model.provenance["settings_hash"] == hash_json(by_model.settings)


def test_settings_and_provenance(hand_survey):
    coords = project(hand_survey, formulation="iso2019")
    assert coords.settings == ProjectionSettings(formulation="iso2019")
    prov = coords.provenance
    assert prov["operation"] == "project"
    assert prov["settings_hash"] == hash_json(coords.settings)
    for key in (
        "input_hash",
        "mapping_hash",
        "data_refs",
        "response_id_source",
        "read",
        "id_columns",
        "response_id_columns",
        "response_key_blank_ok",
    ):
        assert prov[key] == hand_survey.provenance[key], key
    assert coords.circumplex == circumplex("iso-12913-outdoor", "1")


def test_origin_segment(hand_survey):
    coords = project(hand_survey)
    (segment,) = coords.origin
    assert segment.kind == "survey"
    assert segment.dataset == "hand_values.csv"
    assert segment.n == 3
    assert segment.input_hash == hand_survey.provenance["input_hash"]
    assert segment.mapping_hash == hand_survey.provenance["mapping_hash"]
    assert segment.angle_sets == ("equal",)
    assert segment.settings_hash == hash_json(coords.settings)


def test_all_rows_excluded_projects_nothing(tmp_path):
    path = tmp_path / "blank.csv"
    lines = HAND_CSV.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[3]]) + "\n", encoding="utf-8")
    coords = project(read_survey(path, MAPPING))
    assert coords.xy.shape == (1, 2)
    assert np.isnan(coords.xy).all()
    assert n_valid(coords) == 0


# Each example writes and reads a file: no deadline, and 25 of the 5040 permutations.
@settings(deadline=None, max_examples=25)
@given(st.permutations(range(7)))
def test_project_row_order_invariance(perm):
    lines = ORDER_CSV.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    assert len(rows) == 7
    base = project(read_survey(ORDER_CSV, MAPPING))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "permuted.csv"
        permuted = [rows[i] for i in perm]
        path.write_text("\n".join([header, *permuted]) + "\n", encoding="utf-8")
        coords = project(read_survey(path, MAPPING))
    index = np.asarray(perm)
    for column in (COL_RESPONSE_ID, COL_INCLUDED, COL_REASONS):
        np.testing.assert_array_equal(coords.labels[column], base.labels[column][index])
    np.testing.assert_array_equal(coords.xy, base.xy[index])


# --- the ISD subset ----------------------------------------------------------------------------


def test_subset_counts(isd_coords):
    assert isd_coords.n() == n(isd_coords) == 1552
    assert isd_coords.n_valid() == n_valid(isd_coords) == 1536
    mask = valid_mask(isd_coords)
    assert int(mask.sum()) == 1536
    assert not mask[~isd_coords.labels[COL_INCLUDED]].any()
    np.testing.assert_array_equal(isd_coords.valid_mask(), mask)


def test_subset_square_bounds(isd_coords):
    for v in (isd_coords.xy[:, 0], isd_coords.xy[:, 1]):
        finite = v[np.isfinite(v)]
        assert finite.size == 1536
        # spec §7.1 square-bound tolerance: float rounding of sums of 8 terms can put |x|
        # one ulp above 1, depending on the summation order
        assert (np.abs(finite) <= 1 + 1e-12).all()


def test_subset_labels_carry_the_id_columns(isd_coords):
    assert list(isd_coords.labels)[6:] == ["LocationID", "SessionID", "GroupID", "RecordID"]


def test_coordinates_is_frozen(hand_survey):
    coords = project(hand_survey)
    with pytest.raises(dataclasses.FrozenInstanceError):
        coords.xy = np.zeros((3, 2))  # ty: ignore[invalid-assignment]  # deliberate: exercises frozen=True
