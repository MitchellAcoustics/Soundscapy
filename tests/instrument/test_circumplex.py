"""Circumplex: the shipped outdoor instrument, validators and round trips."""

import importlib
import tomllib

import pytest
from pydantic import ValidationError

from soundscapy import _data
from soundscapy._model import from_toml, to_toml
from soundscapy.errors import DataReferenceError, MappingError
from soundscapy.instrument import Circumplex, Scale, circumplex
from soundscapy.provenance import hash_json

# The package attribute `circumplex` is the function; reach the submodule explicitly.
circumplex_module = importlib.import_module("soundscapy.instrument.circumplex")


def test_package_reexports():
    assert circumplex_module.circumplex is circumplex
    assert circumplex_module.Circumplex is Circumplex
    assert Scale is importlib.import_module("soundscapy.instrument.scale").Scale


def test_shipped_outdoor_instrument():
    c = circumplex()
    assert c.positions == tuple(f"PAQ{i}" for i in range(1, 9))
    # Exact: multiples of 45.0 are exactly representable.
    assert c.angles_deg == tuple(45.0 * i for i in range(8))
    assert tuple(a.id for a in c.axes) == ("ISOPleasant", "ISOEventful")
    assert (c.scale.minimum, c.scale.maximum, c.scale.kind) == (1.0, 5.0, "ordinal")
    assert c.scale.labels[0][0] == "eng"
    assert c.axes[0].labels == ()
    assert c.translations == ()


def test_label_and_fallback():
    c = circumplex()
    assert c.label("PAQ6") == "monotonous"
    assert c.label("PAQ1", "deu") == "pleasant"  # no translation: reference language
    assert c.label("PAQ1", "eng") == "pleasant"
    with pytest.raises(MappingError):
        c.label("PAQ9")


def test_position_for():
    c = circumplex()
    assert c.position_for("Calm") == "PAQ8"
    assert c.position_for("PAQ3") == "PAQ3"
    assert c.position_for("paq3") == "PAQ3"
    with pytest.raises(MappingError):
        c.position_for("loud")


def _changed(**changes):
    c = circumplex()
    return Circumplex.model_validate({**c.model_dump(), **changes})


def _attrs(**per_index):
    """Shipped attributes as dicts with the given per-index field overrides."""
    out = [a.model_dump() for a in circumplex().attributes]
    for index, fields in per_index.items():
        out[int(index.removeprefix("i"))].update(fields)
    return out


def test_identity_derived_record_idiom():
    assert _changed(version="2").version == "2"


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"attributes": _attrs()[:1] + _attrs()[2:]}, "PAQ1..PAQn"),
        ({"attributes": _attrs(i3={"angle_deg": 45.0})}, "strictly increasing"),
        ({"attributes": _attrs(i7={"angle_deg": 360.0})}, "outside"),
        ({"quadrants": (("++", "PAQ2"), ("-+", "PAQ4"), ("--", "PAQ6"))}, "exactly the keys"),
        ({"quadrants": (("++", "PAQ2"), ("-+", "PAQ4"), ("--", "PAQ6"), ("+-", "PAQ9"))}, "PAQ9"),
        (
            {
                "axes": (
                    {"id": "ISOPleasant", "positive": "PAQ1", "negative": "PAQ9"},
                    {"id": "ISOEventful", "positive": "PAQ3", "negative": "PAQ7"},
                )
            },
            "unknown position",
        ),
        (
            {
                "translations": (
                    {"language": "eng", "labels": tuple((f"PAQ{i}", "x") for i in range(1, 9))},
                )
            },
            "reference language",
        ),
        ({"translations": ({"language": "deu", "labels": (("PAQ1", "x"),)},)}, "every position"),
    ],
)
def test_validators_reject(changes, fragment):
    with pytest.raises(ValidationError, match=fragment):
        _changed(**changes)


def test_attribute_labels_reference_language_only():
    attrs = _attrs(i0={"labels": (("eng", "pleasant"), ("deu", "angenehm"))})
    with pytest.raises(ValidationError, match="reference language"):
        _changed(attributes=attrs)


def test_missing_schema_rejected():
    data = circumplex().model_dump(by_alias=True)
    del data["schema"]
    with pytest.raises(ValidationError):
        Circumplex.model_validate(data)
    del data["schema_version"]
    with pytest.raises(ValidationError):
        Circumplex.model_validate(data)


def test_alias_round_trips():
    c = circumplex()
    assert Circumplex.model_validate(c.model_dump()) == c
    assert Circumplex.model_validate(c.model_dump(by_alias=True)) == c


def test_circumplex_toml_round_trip():
    c = circumplex()
    text = to_toml(c)
    assert next(iter(tomllib.loads(text))) == "schema"  # by_alias=True is honoured
    assert "schema_name" not in text
    back = from_toml(Circumplex, text)
    assert back == c
    assert hash_json(back) == hash_json(c)


def test_unknown_circumplex_lists_shipped():
    with pytest.raises(DataReferenceError, match=r"iso-12913-outdoor@1"):
        circumplex(version="9")
    with pytest.raises(DataReferenceError):
        circumplex(name="nope")


def test_shipped_file_bytes_are_what_to_toml_writes():
    text = _data.read_bytes("circumplexes/iso-12913-outdoor/1.toml").decode("utf-8")
    assert to_toml(circumplex()) == text
