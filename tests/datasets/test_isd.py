"""The ISD loader: the bundled subset read through its registry entry."""

from __future__ import annotations

import pytest

from soundscapy import _data
from soundscapy.datasets import isd
from soundscapy.errors import DataReferenceError
from soundscapy.survey.mapping import read_mapping, write_mapping
from soundscapy.survey.read import Survey
from tests._helpers import ISD_FEATURES, ISD_MAPPING_CARRY

_MAPPING = "datasets/isd/isd_mapping.toml"
# Spec 6.4.4's normative example, which the shipped file must equal byte for byte.
_NORMATIVE = """\
schema = "soundscapy.mapping"
schema_version = 1
instrument = "iso-12913-outdoor"
instrument_version = "1"
age = "age00"
carry = ["start_time", "Survey_Version", "sss03", "sss04", "sss05"]

[ids]
response_key = ["LocationID", "SessionID", "GroupID", "RecordID"]
location = "LocationID"
session = "SessionID"
group = "GroupID"
record = "RecordID"

[language]
column = "Language"

[groups.attributes.columns]
pleasant = "pleasant"
vibrant = "vibrant"
eventful = "eventful"
chaotic = "chaotic"
annoying = "annoying"
monotonous = "monotonous"
uneventful = "uneventful"
calm = "calm"

[groups.source_dominance.columns]
traffic_noise = "ssi01"
other_noise = "ssi02"
human_sounds = "ssi03"
natural_sounds = "ssi04"

[groups.overall_quality.columns]
overall_quality = "sss01"

[groups.appropriateness.columns]
appropriateness = "sss02"

[groups.who5.columns]
who5_1 = "who01"
who5_2 = "who02"
who5_3 = "who03"
who5_4 = "who04"
who5_5 = "who05"
"""


def test_load_subset(isd_survey):
    assert isinstance(isd_survey, Survey)
    assert isd_survey.n() == 1552
    assert isd_survey.provenance["response_id_source"] == "key"
    assert len(set(isd_survey.response_id)) == isd_survey.n()


def test_load_subset_records_the_dataset(isd_survey):
    assert {
        "kind": "dataset",
        "name": "isd-subset",
        "version": "1.0.1-alpha.1",
        "hash": _data.pin("datasets/isd/subset-v1.0.csv"),
    } in isd_survey.provenance["data_refs"]


def test_load_subset_incomplete_attributes(isd_survey):
    assert isd_survey.status.counts()["reason:incomplete_attributes"] == 16


def test_load_subset_explicit_default_version():
    assert isd.load(subset=True, version="1.0.1-alpha.1").n() == 1552


def test_load_subset_rejects_other_versions():
    with pytest.raises(DataReferenceError, match=r"1\.0\.1-alpha\.1"):
        isd.load(subset=True, version="1.0.1-alpha")


def test_load_full_is_not_implemented():
    with pytest.raises(NotImplementedError, match="subset=True"):
        isd.load()


def test_load_uses_the_registry_mapping(monkeypatch):
    registry = _data.read_toml("datasets/isd/registry.toml")
    registry["versions"]["1.0.1-alpha.1"]["mapping"] = "missing.toml"
    monkeypatch.setattr(isd, "_registry", lambda: registry)
    with pytest.raises(DataReferenceError, match=r"datasets/isd/missing\.toml"):
        isd.load(subset=True)


def test_isd_mapping_is_the_normative_example():
    assert _data.read_bytes(_MAPPING).decode("utf-8") == _NORMATIVE


def test_load_subset_carries_features(isd_survey, isd_coords):
    assert isd_survey.mapping.carry == (*ISD_MAPPING_CARRY, *ISD_FEATURES)
    assert isd_coords.labels["LAeq_L(A)"].dtype == object
    assert isd_coords.labels["LAeq_L(A)"][0] == isd_survey.data["LAeq_L(A)"].iloc[0]


def test_load_passes_settings_through(isd_survey):
    off = isd.load(subset=True, quality={"all_midpoint": "off"})
    assert off.status.counts()["flag:all_midpoint"] == 0
    assert off.settings.quality.all_midpoint == "off"
    strict = isd.load(subset=True, settings={"quality": {"under_age": "exclude"}})
    counts = strict.status.counts()
    assert counts["reason:under_age"] == 8  # the 8 under-age rows are all complete
    assert counts["excluded"] == 24
    same = isd.load(subset=True, on_scale_violation="exclude")
    assert same.status.counts() == isd_survey.status.counts()  # no violations


def test_registry_feature_columns():
    columns = _data.read_toml("datasets/isd/registry.toml")["versions"]["1.0.1-alpha.1"]
    columns = columns["feature_columns"]
    assert len(columns) == len(set(columns)) == 96
    assert columns[0] == "LZeq_L"
    assert set(ISD_FEATURES) <= set(columns)


def test_isd_mapping_byte_round_trip(tmp_path):
    out = tmp_path / "isd_mapping.toml"
    with _data.as_path(_MAPPING) as path:
        write_mapping(read_mapping(path), out)
        assert out.read_bytes() == path.read_bytes()
