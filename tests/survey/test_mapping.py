"""SurveyMapping, Coding and the mapping TOML: normalisation, validators, coding order."""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomlkit
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from soundscapy._model import validate
from soundscapy.errors import (
    CodingError,
    DataIntegrityError,
    DataReferenceError,
    MappingError,
    SettingsError,
)
from soundscapy.instrument import Scale, circumplex
from soundscapy.provenance import hash_json
from soundscapy.survey import mapping as mapping_module
from soundscapy.survey.mapping import (
    Coding,
    Groups,
    IdSpec,
    ItemGroup,
    LanguageSpec,
    SurveyMapping,
    apply_coding,
    read_mapping,
    write_mapping,
)
from tests._helpers import LABELS, raw_mapping

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL = FIXTURES / "minimal_mapping.toml"
NONDEFAULT = FIXTURES / "nondefault_mapping.toml"
POSITIONS = tuple(f"PAQ{i}" for i in range(1, 9))


def _raw(**attributes: object) -> dict:
    """A minimal raw mapping dict whose attribute group holds the given fields."""
    raw = raw_mapping({"response": "ID"})
    raw["groups"]["attributes"].update(attributes)
    return raw


def _write_raw(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "mapping.toml"
    path.write_text(tomlkit.dumps(raw), encoding="utf-8")
    return path


def _codes(texts, coding=None, scale=None, keys=POSITIONS[:1]):
    """Code texts given row by row (k cells per row) for the first k positions."""
    arr = np.asarray(texts, dtype=object).reshape(-1, len(keys))
    ids = np.asarray([f"r{i}" for i in range(arr.shape[0])], dtype=object)
    return apply_coding(
        arr,
        coding or Coding(),
        scale or Scale(minimum=1.0, maximum=5.0),
        item_keys=keys,
        response_ids=ids,
    )


# --- normalisation ---------------------------------------------------------------------------


def test_label_and_position_keys_hash_alike():
    by_label = SurveyMapping.model_validate(_raw())
    by_position = _raw()
    by_position["groups"]["attributes"]["columns"] = dict(zip(POSITIONS, LABELS, strict=True))
    by_position = SurveyMapping.model_validate(by_position)
    assert by_label == by_position
    assert hash_json(by_label) == hash_json(by_position)
    assert tuple(k for k, _ in by_label.groups.attributes.columns) == POSITIONS
    # The default scale is filled in from the instrument.
    assert by_label.groups.attributes.scale == circumplex().scale


def test_columns_out_of_order_are_put_in_position_order():
    raw = _raw()
    raw["groups"]["attributes"]["columns"] = {lab: lab for lab in reversed(LABELS)}
    mapping = SurveyMapping.model_validate(raw)
    assert mapping.groups.attributes.columns == tuple(zip(POSITIONS, LABELS, strict=True))


def test_label_and_position_same_item_raises(tmp_path):
    raw = _raw()
    raw["groups"]["attributes"]["columns"]["PAQ1"] = "other"
    with pytest.raises(
        MappingError, match=re.escape("groups.attributes.columns: attribute keys 'pleasant'")
    ) as info:
        read_mapping(_write_raw(tmp_path, raw))
    assert "PAQ1" in str(info.value)


def test_unmapped_attribute_names_the_field_path():
    raw = _raw()
    del raw["groups"]["attributes"]["columns"]["calm"]
    with pytest.raises(
        ValidationError,
        match=re.escape("groups.attributes.columns: attribute PAQ8 ('calm') is not mapped"),
    ):
        SurveyMapping.model_validate(raw)


def test_reverse_label_and_position_alike():
    by_label = SurveyMapping.model_validate(_raw(coding={"reverse": ["calm"]}))
    by_position = SurveyMapping.model_validate(_raw(coding={"reverse": ["PAQ8"]}))
    assert by_label == by_position
    assert hash_json(by_label) == hash_json(by_position)
    group = by_label.groups.attributes
    assert group.coding.reverse == ("PAQ8",)
    coded = _codes(["2"] * 8, group.coding, group.scale, keys=POSITIONS)
    # Exact: small integers; calm (PAQ8) is reversed on 1..5, 1 + 5 - 2 = 4.
    assert coded.tolist() == [[2.0] * 7 + [4.0]]


def test_reverse_is_put_in_position_order():
    mapping = SurveyMapping.model_validate(_raw(coding={"reverse": ["calm", "PAQ2"]}))
    assert mapping.groups.attributes.coding.reverse == ("PAQ2", "PAQ8")


@pytest.mark.parametrize(
    ("reverse", "fragment"),
    [
        (["loud"], "loud"),
        (
            ["calm", "PAQ8"],
            re.escape(
                "groups.attributes.coding.reverse: reverse keys 'calm' and 'PAQ8' repeat PAQ8"
            ),
        ),
    ],
)
def test_reverse_unknown_key_raises(reverse, fragment):
    with pytest.raises(ValidationError, match=fragment):
        SurveyMapping.model_validate(_raw(coding={"reverse": reverse}))


def test_missing_values_may_hold_blank(tmp_path):
    path = _write_raw(tmp_path, _raw(coding={"missing_values": ["", "-99"]}))
    assert read_mapping(path).groups.attributes.coding.missing_values == ("", "-99")


def test_defaults_validate():
    assert IdSpec().response == ""
    assert IdSpec(response="ID").response_key == ()
    assert LanguageSpec().coding == ()
    assert Coding().missing_values == ("", "NA", "N/A", "NaN", "nan", "#N/A")


def test_built_from_model_instances():
    columns = tuple(zip(LABELS, LABELS, strict=True))
    group = ItemGroup(columns=columns, scale=circumplex().scale)
    built = SurveyMapping.model_validate(
        {
            "schema": "soundscapy.mapping",
            "schema_version": 1,
            "ids": IdSpec(
                response_key=("LocationID", "SessionID", "GroupID", "RecordID"),
                location="LocationID",
                session="SessionID",
                group="GroupID",
                record="RecordID",
            ),
            "language": LanguageSpec(column="Language"),
            "groups": Groups(attributes=group),
        }
    )
    assert built == read_mapping(MINIMAL)
    # A built model validates again unchanged (the derived-record idiom).
    assert SurveyMapping.model_validate(built.model_dump()) == built
    assert SurveyMapping.model_validate(built) is built


# --- validation errors -----------------------------------------------------------------------


def _with(path: tuple[str, ...], value: object) -> dict:
    raw = _raw()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return raw


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("not a table", "dictionary"),
        ({**_raw(), "instrument": 1}, "instrument"),
        (_with(("groups",), "x"), "groups"),
        (_with(("groups", "attributes"), "x"), "attributes"),
        (_with(("groups", "attributes", "columns"), "x"), "columns"),
        (_with(("groups", "attributes", "columns"), [("pleasant",)]), "columns"),
        (_with(("groups", "attributes", "coding"), {"reverse": "calm"}), "reverse"),
        (_with(("groups", "attributes", "coding"), {"reverse": [1]}), "reverse"),
    ],
)
def test_malformed_input_reaches_pydantic(raw, fragment):
    with pytest.raises(ValidationError, match=fragment):
        SurveyMapping.model_validate(raw)


def test_unmapped_position_raises(tmp_path):
    raw = _raw()
    del raw["groups"]["attributes"]["columns"]["calm"]
    with pytest.raises(MappingError, match="PAQ8") as info:
        read_mapping(_write_raw(tmp_path, raw))
    assert "calm" in str(info.value)


def test_unknown_attribute_key_raises(tmp_path):
    raw = _raw()
    raw["groups"]["attributes"]["columns"]["loud"] = "x"
    with pytest.raises(MappingError, match="loud"):
        read_mapping(_write_raw(tmp_path, raw))


def test_unknown_instrument_raises():
    with pytest.raises(DataReferenceError, match="indoor"):
        SurveyMapping.model_validate({**_raw(), "instrument": "indoor"})


@pytest.mark.parametrize("raised", [DataIntegrityError, MappingError])
def test_read_mapping_reraises_instrument_errors_unchanged(tmp_path, monkeypatch, raised):
    # Both subclass ValueError, so pydantic would wrap them inside the before-validator.
    original = raised("hash mismatch: expected h1-sha256:aa, got h1-sha256:bb")

    def broken(*_args: object, **_kwargs: object) -> None:
        raise original

    monkeypatch.setattr(mapping_module, "circumplex", broken)
    path = _write_raw(tmp_path, _raw())
    with pytest.raises(raised) as info:
        read_mapping(path)
    assert info.value is original
    with pytest.raises(raised) as info:
        validate(SurveyMapping, _raw(), error=SettingsError)
    assert info.value is original


def test_bad_language_code_raises(tmp_path):
    raw = {**_raw(), "language": {"column": "Language", "coding": {"English": "ENG"}}}
    with pytest.raises(MappingError, match="ENG"):
        read_mapping(_write_raw(tmp_path, raw))


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        ({"column": "Language", "value": "eng"}, "not both"),
        ({"value": "English"}, "English"),
    ],
)
def test_language_spec_rules(fields, fragment):
    with pytest.raises(ValidationError, match=fragment):
        LanguageSpec.model_validate(fields)


def test_response_and_key_together_raise():
    with pytest.raises(ValidationError, match="not both"):
        IdSpec.model_validate({"response": "ID", "response_key": ["A", "B"]})


@pytest.mark.parametrize(
    "blank_ok",
    [["A", "B"], ["C"]],
    ids=["all-of-key", "not-in-key"],
)
def test_blank_ok_must_be_proper_subset(blank_ok):
    with pytest.raises(ValidationError, match="response_key_blank_ok"):
        IdSpec.model_validate({"response_key": ["A", "B"], "response_key_blank_ok": blank_ok})


def test_blank_ok_proper_subset_validates():
    spec = IdSpec.model_validate({"response_key": ["A", "B"], "response_key_blank_ok": ["B"]})
    assert spec.response_key_blank_ok == ("B",)


@pytest.mark.parametrize(
    ("change", "key"),
    [
        (lambda raw: raw["ids"].update(response=""), "ids.response"),
        (
            lambda raw: raw["groups"]["attributes"]["columns"].update(calm=""),
            "groups.attributes.columns.calm",
        ),
        (
            lambda raw: raw.update(language={"coding": {"English": " "}}),
            "language.coding.English",
        ),
    ],
)
def test_blank_value_raises_naming_key(tmp_path, change, key):
    raw = _raw()
    change(raw)
    with pytest.raises(MappingError, match=re.escape(key)):
        read_mapping(_write_raw(tmp_path, raw))


def test_write_blank_value_raises(tmp_path):
    mapping = SurveyMapping.model_validate(_raw(coding={"value_map": {"agree": 4.0}}))
    raw = mapping.model_dump()
    raw["groups"]["attributes"]["columns"] = (
        ("PAQ1", ""),
        *raw["groups"]["attributes"]["columns"][1:],
    )
    with pytest.raises(MappingError, match=re.escape("groups.attributes.columns.pleasant")):
        write_mapping(SurveyMapping.model_validate(raw), tmp_path / "out.toml")


def test_read_mapping_bad_toml_raises(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("schema = ", encoding="utf-8")
    with pytest.raises(MappingError, match=re.escape("bad.toml")):
        read_mapping(path)


def test_read_mapping_keeps_field_path(tmp_path):
    raw = {**_raw(), "ids": {"response": "ID", "extra": "x"}}
    with pytest.raises(MappingError, match=re.escape("ids.extra")):
        read_mapping(_write_raw(tmp_path, raw))


# --- TOML writer -----------------------------------------------------------------------------


def test_write_mapping_byte_round_trip(tmp_path):
    out = tmp_path / "out.toml"
    write_mapping(read_mapping(MINIMAL), out)
    text = out.read_bytes()
    assert text == MINIMAL.read_bytes()
    assert b"PAQ1" not in text
    assert b"scale" not in text
    assert b"coding" not in text


@pytest.mark.parametrize("fixture", [MINIMAL, NONDEFAULT], ids=["minimal", "nondefault"])
def test_mapping_model_round_trip(tmp_path, fixture):
    original = read_mapping(fixture)
    out = tmp_path / "out.toml"
    write_mapping(original, out)
    again = read_mapping(out)
    assert again == original
    assert hash_json(again) == hash_json(original)
    assert out.read_bytes() == fixture.read_bytes()


def test_nondefault_fixture_values():
    group = read_mapping(NONDEFAULT).groups.attributes
    assert group.scale == Scale(minimum=1.0, maximum=7.0, kind="interval")
    assert group.coding.reverse == ("PAQ8",)
    assert group.coding.decimal == ","
    assert read_mapping(NONDEFAULT).language.coding == (("English", "eng"), ("Deutsch", "deu"))


def test_write_mapping_identity_keys_always_written(tmp_path):
    out = tmp_path / "out.toml"
    write_mapping(SurveyMapping.model_validate(_raw()), out)
    data = tomllib.loads(out.read_text(encoding="utf-8"))
    assert data["instrument"] == "iso-12913-outdoor"
    assert data["instrument_version"] == "1"
    assert "language" not in data


# --- apply_coding ----------------------------------------------------------------------------


def test_apply_coding_isd_forms():
    coded = _codes(["4", "4.0", " 4 ", "", "NA"])
    # Exact: 4 is exactly representable; missing tokens are NaN.
    assert coded[:3, 0].tolist() == [4.0, 4.0, 4.0]
    assert all(math.isnan(v) for v in coded[3:, 0])
    assert coded.dtype == np.float64


def test_apply_coding_comma_with_dot_decimal_raises():
    texts = ["3,5", "1", "3,5", "3,5", "3,5", "3,5", "3,5"]
    with pytest.raises(CodingError, match=re.escape("'3,5'")) as info:
        _codes(texts)
    message = str(info.value)
    assert "'3,5' x6" in message  # six offending cells
    for rid in ("r0", "r2", "r3", "r4", "r5"):
        assert rid in message
    assert "r6" not in message  # at most five response ids
    assert "value_map" in info.value.hint


def test_apply_coding_comma_decimal():
    coded = _codes(["3,5", "4"], Coding(decimal=","))
    # Exact: 3.5 is a dyadic rational.
    assert coded[:, 0].tolist() == [3.5, 4.0]
    with pytest.raises(CodingError, match=re.escape("'3.5'")):
        _codes(["3.5"], Coding(decimal=","))


@pytest.mark.parametrize("text", ["inf", "1e400", "abc", "1_0"])
def test_apply_coding_rejects_non_numbers(text):
    with pytest.raises(CodingError):
        _codes([text])


def test_apply_coding_value_map_before_parse():
    coding = Coding.model_validate({"value_map": {"agree": 4.0, "5": 1.0}})
    coded = _codes(["agree", "5", " agree "], coding)
    # Exact: small integers.
    assert coded[:, 0].tolist() == [4.0, 1.0, 4.0]


def test_apply_coding_shape_and_key_errors():
    texts = np.asarray([["1", "2"]], dtype=object)
    ids = np.asarray(["r0"], dtype=object)
    scale = Scale(minimum=1.0, maximum=5.0)
    with pytest.raises(ValueError, match="item_keys"):
        apply_coding(texts, Coding(), scale, item_keys=("PAQ1",), response_ids=ids)
    with pytest.raises(ValueError, match="response_ids"):
        apply_coding(texts, Coding(), scale, item_keys=POSITIONS[:2], response_ids=ids[:0])
    with pytest.raises(MappingError, match="PAQ8"):
        apply_coding(
            texts, Coding(reverse=("PAQ8",)), scale, item_keys=POSITIONS[:2], response_ids=ids
        )
    with pytest.raises(TypeError, match="str"):
        apply_coding(
            np.asarray([[1, "2"]], dtype=object),
            Coding(),
            scale,
            item_keys=POSITIONS[:2],
            response_ids=ids,
        )


# Half-integers in [-10, 10]: dyadic rationals, so every sum and difference below is exact
# in float64. A non-dyadic draw would break exactness: 6 - (6 - 0.3) is 0.2999999999999998.
_HALVES = st.integers(-20, 20).map(lambda k: k / 2)


@st.composite
def _coding_cases(draw):
    minimum = float(draw(st.integers(-5, 5)))
    maximum = minimum + float(draw(st.integers(1, 10)))
    tokens = draw(st.lists(st.sampled_from("abcdefgh"), min_size=1, max_size=5, unique=True))
    value_map = {token: draw(_HALVES) for token in tokens}
    offset = draw(_HALVES)
    k = draw(st.integers(1, 4))
    keys = POSITIONS[:k]
    reverse = tuple(key for key in keys if draw(st.booleans()))
    n = draw(st.integers(1, 6))
    texts = [[draw(st.sampled_from(tokens)) for _ in range(k)] for _ in range(n)]
    return minimum, maximum, value_map, offset, keys, reverse, texts


@given(_coding_cases())
def test_coding_order(case):
    minimum, maximum, value_map, offset, keys, reverse, texts = case
    scale = Scale(minimum=minimum, maximum=maximum, kind="interval")
    coding = Coding.model_validate({"value_map": value_map, "offset": offset, "reverse": reverse})
    arr = np.asarray(texts, dtype=object)
    ids = np.asarray([f"r{i}" for i in range(len(texts))], dtype=object)
    coded = apply_coding(arr, coding, scale, item_keys=keys, response_ids=ids)
    for i, row in enumerate(texts):
        for j, text in enumerate(row):
            shifted = value_map[text] + offset
            expected = minimum + maximum - shifted if keys[j] in reverse else shifted
            assert coded[i, j] == expected  # exact: dyadic values, see _HALVES
            # Reverse is an involution.
            assert minimum + maximum - (minimum + maximum - shifted) == shifted
