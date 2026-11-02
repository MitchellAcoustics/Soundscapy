"""The ISD loader: the bundled subset read through its registry entry."""

from __future__ import annotations

import pytest

from soundscapy import _data
from soundscapy.datasets import isd
from soundscapy.errors import DataReferenceError
from soundscapy.survey.mapping import read_mapping, write_mapping
from soundscapy.survey.read import Survey

_MAPPING = "datasets/isd/isd_mapping.toml"


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


def test_isd_mapping_byte_round_trip(tmp_path):
    out = tmp_path / "isd_mapping.toml"
    with _data.as_path(_MAPPING) as path:
        write_mapping(read_mapping(path), out)
        assert out.read_bytes() == path.read_bytes()
