"""FrozenModel, key-to-value pair types, TOML round trip and namespace constants."""

import math
from typing import Literal

import pytest
from pydantic import StrictFloat, ValidationError

from soundscapy import _model
from soundscapy._model import (
    _UNSET,
    FloatPairs,
    FrozenModel,
    StrPairs,
    StrTuplePairs,
    _Unset,
    axis_column,
    from_toml,
    to_toml,
)
from soundscapy.errors import SettingsError


class _Box(FrozenModel):
    x: StrictFloat = 0.0


class _Rec(FrozenModel):
    names: StrPairs = ()
    labels: StrTuplePairs = ()
    values: FloatPairs = ()
    bounds: tuple[StrictFloat, ...] = ()
    clip: Literal["none"] | tuple[StrictFloat, StrictFloat] = "none"


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_floats_rejected(bad):
    with pytest.raises(ValidationError):
        _Box.model_validate({"x": bad})


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        _Box.model_validate({"x": 1.0, "y": 2.0})


def test_frozen():
    box = _Box(x=1.0)
    with pytest.raises(ValidationError):
        box.x = 2.0  # ty: ignore[invalid-assignment]  # deliberate: exercises frozen=True


def test_integer_accepted_as_float():
    assert _Box.model_validate({"x": 1}).x == 1.0


def test_strpairs_from_table_and_pairs_are_equal():
    a = _Rec.model_validate({"names": {"a": "x"}})
    b = _Rec.model_validate({"names": (("a", "x"),)})
    assert a == b
    assert a.names == (("a", "x"),)
    assert a.model_dump(mode="json")["names"] == {"a": "x"}


def test_pairs_keep_order_and_it_matters_for_equality():
    a = _Rec.model_validate({"names": {"a": "1", "b": "2"}})
    b = _Rec.model_validate({"names": {"b": "2", "a": "1"}})
    assert a != b
    assert list(a.model_dump(mode="json")["names"]) == ["a", "b"]


def test_strpairs_duplicate_key_from_pairs_rejected():
    with pytest.raises(ValidationError, match="'a'"):
        _Rec.model_validate({"names": (("a", "x"), ("a", "y"))})


def test_float_and_tuple_pairs_duplicate_key_rejected():
    with pytest.raises(ValidationError, match="'k'"):
        _Rec.model_validate({"values": (("k", 1.0), ("k", 2.0))})
    with pytest.raises(ValidationError, match="'k'"):
        _Rec.model_validate({"labels": (("k", ("a",)), ("k", ("b",)))})


def test_float_pairs_reject_non_finite():
    with pytest.raises(ValidationError):
        _Rec.model_validate({"values": {"k": math.nan}})


def test_str_tuple_pairs_json_table():
    rec = _Rec.model_validate({"labels": {"eng": ["a", "b"]}})
    assert rec.labels == (("eng", ("a", "b")),)
    assert rec.model_dump(mode="json")["labels"] == {"eng": ["a", "b"]}


def test_python_mode_dump_keeps_pairs():
    rec = _Rec.model_validate({"names": {"a": "x"}})
    assert rec.model_dump()["names"] == (("a", "x"),)


def test_toml_round_trip():
    rec = _Rec.model_validate(
        {
            "names": {"a": "x", "b": "y"},
            "labels": {"eng": ["lo", "hi"]},
            "values": {"yes": 1.0, "no": 0.0},
            "bounds": [-1.0, 1.0],
            "clip": "none",
        },
    )
    text = to_toml(rec)
    assert from_toml(_Rec, text) == rec
    assert from_toml(_Rec, to_toml(from_toml(_Rec, text))) == rec
    assert to_toml(rec) == text  # deterministic bytes


def test_toml_round_trip_with_clip_box():
    rec = _Rec.model_validate({"clip": [-1.0, 1.0]})
    assert from_toml(_Rec, to_toml(rec)) == rec


def test_from_toml_bad_value_raises_settings_error():
    with pytest.raises(SettingsError, match="x"):
        from_toml(_Box, 'x = "nope"\n')


def test_from_toml_unknown_key_raises_settings_error():
    with pytest.raises(SettingsError, match="extra"):
        from_toml(_Box, "extra = 1.0\n")


def test_citation_defaults_and_strictness():
    cit = _model.Citation()
    assert (cit.citation, cit.doi_or_url, cit.licence, cit.date, cit.notes) == ("",) * 5
    with pytest.raises(ValidationError):
        _model.Citation.model_validate({"date": 2026})


def test_axis_column_and_constants():
    assert axis_column("ISOPleasant") == "ssp_ISOPleasant"
    assert _model.SSP_PREFIX == "ssp_"
    cols = [
        _model.COL_RESPONSE_ID,
        _model.COL_INCLUDED,
        _model.COL_REASONS,
        _model.COL_FLAGS,
        _model.COL_LANGUAGE,
        _model.COL_ANGLE_SET,
    ]
    assert all(c.startswith("ssp_") for c in cols)
    assert _model.COL_RESPONSE_ID == "ssp_response_id"
    assert _model.COL_ANGLE_SET == "ssp_angle_set"


def test_unset_sentinel():
    assert _UNSET is _Unset.UNSET
    assert len(_Unset) == 1


@pytest.mark.parametrize("bad", [5, "ab", (("a", "x", "y"),), ("a",)])
def test_pairs_reject_malformed_input(bad):
    with pytest.raises(ValidationError):
        _Rec.model_validate({"names": bad})
