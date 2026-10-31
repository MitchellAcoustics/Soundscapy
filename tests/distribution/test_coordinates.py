"""Coordinates: read-only arrays, valid rows, select, split and to_frame."""

import dataclasses

import numpy as np
import pandas as pd
import polars as pl
import pytest

from soundscapy._model import COL_INCLUDED, COL_LANGUAGE, COL_RESPONSE_ID
from soundscapy.datasets import isd
from soundscapy.distribution.coordinates import Coordinates, OriginSegment, n_valid
from soundscapy.errors import MappingError
from soundscapy.instrument import Scale
from soundscapy.projection.project import project

LOCATIONS = sorted(
    [
        "CamdenTown",
        "EustonTap",
        "MarchmontGarden",
        "PancrasLock",
        "RegentsParkFields",
        "RegentsParkJapan",
        "RussellSq",
        "StPaulsCross",
        "StPaulsRow",
        "TateModern",
        "TorringtonSq",
        "CampoPrincipe",
        "SanMarco",
        "Noorderplantsoen",
        "PingshanPark",
    ]
)


@pytest.fixture(scope="module")
def coords() -> Coordinates:
    return project(isd.load(subset=True))


def _rebuild(coords: Coordinates, **changes) -> Coordinates:
    return dataclasses.replace(coords, **changes)


def test_arrays_are_read_only(coords):
    assert not coords.xy.flags.writeable
    for values in coords.labels.values():
        assert not values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        coords.xy[0, 0] = 0.0


def test_constructor_copies_the_arrays(coords):
    xy = np.array(coords.xy)
    labels = {name: np.array(values) for name, values in coords.labels.items()}
    rebuilt = _rebuild(coords, xy=xy, labels=labels)
    xy[0] = 9.0
    labels[COL_RESPONSE_ID][0] = "changed"
    assert rebuilt.xy[0, 0] != 9.0  # exact: an untouched copy
    assert rebuilt.labels[COL_RESPONSE_ID][0] != "changed"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"xy": np.zeros((1552, 3))}, "shape"),
        ({"xy": np.zeros((1552, 2), dtype=np.float32)}, "float64"),
        ({"labels": {COL_INCLUDED: np.ones(3, dtype=bool)}}, "length"),
        ({"labels": {"x": np.ones((1552, 2), dtype=bool)}}, "length"),
    ],
)
def test_constructor_rejects_bad_arrays(coords, change, match):
    labels = {**coords.labels, **change.get("labels", {})}
    with pytest.raises((ValueError, TypeError), match=match):
        _rebuild(coords, **{**change, "labels": labels})


def test_constructor_rejects_origin_counts_that_disagree(coords):
    segment = OriginSegment.model_validate({**coords.origin[0].model_dump(), "n": 5})
    with pytest.raises(ValueError, match="origin"):
        _rebuild(coords, origin=(segment,))


def test_split_location(coords):
    groups = coords.split("location")
    assert list(groups) == LOCATIONS
    assert sum(group.n() for group in groups.values()) == 1552
    for name, group in groups.items():
        assert set(group.labels["LocationID"].tolist()) == {name}
        assert group.origin[0].n == group.n()


def test_split_by_label_column_matches_role(coords):
    by_role = coords.split("location")
    by_column = coords.split("LocationID")
    assert list(by_role) == list(by_column)
    for key in by_role:
        np.testing.assert_array_equal(by_role[key].xy, by_column[key].xy)


def test_split_language(coords):
    groups = coords.split("language")
    assert list(groups) == ["cmn", "eng", "ita", "nld", "spa"]
    for name, group in groups.items():
        assert set(group.labels[COL_LANGUAGE].tolist()) == {name}


def test_split_unknown_raises_listing_choices(coords):
    with pytest.raises(MappingError, match="location") as info:
        coords.split("nonsense")
    assert "language" in str(info.value)
    assert "nonsense" in str(info.value)


def test_select_updates_origin_n(coords):
    mask = coords.labels["LocationID"] == "CamdenTown"
    chosen = coords.select(mask)
    assert chosen.n() == int(mask.sum())
    assert chosen.origin[0].n == chosen.n()
    assert coords.origin[0].n == 1552
    np.testing.assert_array_equal(chosen.xy, coords.xy[mask])
    assert chosen.settings == coords.settings
    assert n_valid(chosen) <= chosen.n()


def test_select_rejects_a_bad_mask(coords):
    with pytest.raises(ValueError, match="mask"):
        coords.select(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="mask"):
        coords.select(np.ones(1552, dtype=np.int64))


def test_select_refuses_pooled_coordinates(coords):
    half = OriginSegment.model_validate({**coords.origin[0].model_dump(), "n": 776})
    pooled = _rebuild(coords, origin=(half, half))
    with pytest.raises(NotImplementedError, match="segment"):
        pooled.select(np.ones(1552, dtype=bool))


def test_to_frame(coords):
    frame = coords.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [*coords.labels, "ssp_ISOPleasant", "ssp_ISOEventful"]
    assert len(frame) == 1552
    # exact: the frame holds copies of the same float64 values
    np.testing.assert_array_equal(frame["ssp_ISOPleasant"].to_numpy(dtype=float), coords.xy[:, 0])


def test_to_frame_like_polars(coords):
    frame = coords.to_frame(like=pl.DataFrame({"a": [1]}))
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns[-2:] == ["ssp_ISOPleasant", "ssp_ISOEventful"]


def test_origin_segment_rejects_non_int_n():
    with pytest.raises(ValueError, match="n"):
        OriginSegment.model_validate(
            {
                "kind": "survey",
                "dataset": "x.csv",
                "n": 1.0,
                "input_hash": "",
                "mapping_hash": "",
                "effective_scale": Scale(minimum=1.0, maximum=5.0),
                "angle_sets": ["equal"],
                "settings_hash": "",
            }
        )


def test_split_rejects_malformed_id_columns(coords):
    broken = _rebuild(coords, provenance={**coords.provenance, "id_columns": ["LocationID"]})
    with pytest.raises(TypeError, match="id_columns"):
        broken.split("location")


def _id_columns(provenance: dict[str, object]) -> dict:
    ids = provenance["id_columns"]
    assert isinstance(ids, dict)
    return ids


def test_provenance_is_not_shared_with_the_survey_or_parent():
    survey = isd.load(subset=True)
    coords = project(survey)
    location = _id_columns(survey.provenance)["location"]
    _id_columns(coords.provenance)["location"] = "X"
    assert _id_columns(survey.provenance)["location"] == location
    fresh = project(survey)
    group = next(iter(fresh.split("location").values()))
    _id_columns(group.provenance)["location"] = "X"
    assert _id_columns(fresh.provenance)["location"] == location


def test_constructor_deep_copies_provenance(coords):
    provenance = {**coords.provenance, "id_columns": {"location": "LocationID"}}
    rebuilt = _rebuild(coords, provenance=provenance)
    provenance["id_columns"]["location"] = "X"
    assert rebuilt.provenance["id_columns"] == {"location": "LocationID"}
