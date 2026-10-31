"""Scale: bounds, derived properties, membership and TOML round trip."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from soundscapy._model import from_toml, to_toml
from soundscapy.instrument.scale import Scale
from soundscapy.provenance import hash_json

_PROBE = [1, 5, 3.5, 0, 6, math.nan, math.inf]


def test_derived_properties_exact():
    s = Scale(minimum=1.0, maximum=5.0)
    # Exact: these are small dyadic floats, so the arithmetic is exact.
    assert (s.midpoint, s.half_range, s.range) == (3.0, 2.0, 4.0)


@pytest.mark.parametrize("bounds", [(5.0, 5.0), (5.0, 1.0), (1.5, 5.0), (1.0, 4.5)])
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(ValidationError):
        Scale(minimum=bounds[0], maximum=bounds[1])


def test_interval_allows_fractional_bounds():
    assert Scale(minimum=0.5, maximum=1.5, kind="interval").range == 1.0  # exact: dyadic


def test_labels_must_match_steps():
    with pytest.raises(ValidationError, match="integer steps"):
        Scale(minimum=1.0, maximum=5.0, labels=(("eng", ("a", "b")),))


def test_contains_ordinal_and_interval():
    ordinal = Scale(minimum=1.0, maximum=5.0)
    interval = Scale(minimum=1.0, maximum=5.0, kind="interval")
    assert ordinal.contains(_PROBE).tolist() == [True, True, False, False, False, False, False]
    assert interval.contains(_PROBE).tolist() == [True, True, True, False, False, False, False]
    assert ordinal.contains(np.array([[1, 2], [3, 9]])).shape == (2, 2)


def test_integer_bounds_coerce_to_float():
    from_int = Scale.model_validate({"minimum": 1, "maximum": 5})
    from_float = Scale(minimum=1.0, maximum=5.0)
    assert from_int == from_float
    assert isinstance(from_int.minimum, float)
    assert hash_json(from_int) == hash_json(from_float)


@pytest.mark.parametrize("bad", [True, "1"])
def test_bool_and_str_bounds_rejected(bad):
    with pytest.raises(ValidationError):
        Scale.model_validate({"minimum": bad, "maximum": 5})


@given(
    lo=st.integers(-1000, 1000),
    width=st.integers(1, 1000),
    kind=st.sampled_from(["ordinal", "interval"]),
)
def test_scale_toml_round_trip(lo, width, kind):
    s = Scale(minimum=float(lo), maximum=float(lo + width), kind=kind)
    back = from_toml(Scale, to_toml(s))
    assert back == s
    assert hash_json(back) == hash_json(s)
