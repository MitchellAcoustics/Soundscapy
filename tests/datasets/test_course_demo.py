"""The course dataset loader: the bundled extract read through its mapping."""

from __future__ import annotations

import pytest

import soundscapy
from soundscapy import _data
from soundscapy.datasets import course_demo
from soundscapy.projection.project import project
from soundscapy.survey.mapping import read_mapping, write_mapping

_MAPPING = "datasets/course_demo/mapping.toml"
_SAL_2026 = {"id": "sal-2026", "version": "1", "hash": _data.pin("profiles/sal-2026/1.toml")}
_CARRIED = [
    "start_time", "end_time", "perceived_loud", "visit_often", "like_to_visit", "gender",
    "tourist_local", "LAeq", "LA10", "LA90", "LCeq", "N5", "sharpness", "roughness", "tonality",
]  # fmt: skip


@pytest.fixture(scope="module")
def course_survey():
    return course_demo.load()


def test_datasets_all():
    assert soundscapy.datasets.__all__ == ["course_demo", "isd"]


def test_load_course_demo(course_survey):
    survey = course_survey
    assert survey.n() == 456
    assert survey.provenance["response_id_source"] == "key"
    assert survey.language.tolist() == ["eng"] * 456
    assert set(survey.scores) == {
        "attributes",
        "source_dominance",
        "overall_quality",
        "appropriateness",
    }
    data_refs = survey.provenance["data_refs"]
    assert isinstance(data_refs, list)
    assert {
        "kind": "dataset",
        "name": "course-demo",
        "version": "1",
        "hash": _data.pin("datasets/course_demo/course_demo.csv"),
    } in data_refs


def test_course_quality_counts(course_survey):
    counts = course_survey.status.counts()
    assert counts["excluded"] == 0
    assert counts["flag:all_midpoint"] == 2
    assert counts["flag:under_age"] == 0
    skipped = course_survey.provenance["quality_skipped"]
    assert skipped == ["no_consent", "test_response", "paper_input"]


def test_course_labels_carry_the_mapped_columns(course_survey):
    labels = project(course_survey).labels
    assert list(labels)[10:] == _CARRIED
    assert labels["LAeq"].dtype == object


def test_course_projects_under_the_default_profile_without_warning(course_survey):
    coords = project(course_survey)  # filterwarnings = error: English is High in satp@1.5
    assert coords.provenance["profile"] == _SAL_2026
    assert coords.origin[0].angle_sets == ("satp@1.5",)
    assert coords.n_valid() == 456


def test_load_passes_profile_and_settings_file_through(tmp_path):
    survey = course_demo.load(profile="iso-12913-3-2019")
    assert survey.provenance["profile"] == {
        "id": "iso-12913-3-2019",
        "version": "1",
        "hash": _data.pin("profiles/iso-12913-3-2019/1.toml"),
    }
    path = tmp_path / "settings.toml"
    path.write_text('extends = "sal-2026"\n[import]\non_scale_violation = "exclude"\n')
    survey = course_demo.load(settings=path)
    assert survey.settings.on_scale_violation == "exclude"
    assert survey.provenance["overrides"] == {"import.on_scale_violation": "exclude"}


def test_course_mapping_byte_round_trip(tmp_path):
    out = tmp_path / "mapping.toml"
    with _data.as_path(_MAPPING) as path:
        write_mapping(read_mapping(path), out)
        assert out.read_bytes() == path.read_bytes()


def test_course_attributes_by_position(course_survey):
    mapping = course_survey.mapping
    assert mapping.groups.attributes.columns == tuple((f"PAQ{i}", f"PAQ{i}") for i in range(1, 9))


def test_load_passes_settings_through():
    assert (
        course_demo.load(quality={"all_midpoint": "off"}).status.counts()["flag:all_midpoint"] == 0
    )
