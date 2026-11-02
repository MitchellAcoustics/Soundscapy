"""The error and warning hierarchy of spec section 6.3.10."""

from __future__ import annotations

import builtins

import pytest

from soundscapy import errors
from soundscapy.errors import SoundscapyError, SoundscapyWarning

# (class name, built-in) pairs copied from spec 6.3.10.
_ERRORS = [
    ("MappingError", ValueError),
    ("CodingError", ValueError),
    ("EncodingError", ValueError),
    ("FileFormatError", ValueError),
    ("ColumnCollisionError", ValueError),
    ("AxesError", ValueError),
    ("AlignmentError", ValueError),
    ("ScaleViolationError", ValueError),
    ("MissingResponseIdError", ValueError),
    ("DuplicateResponseIdError", ValueError),
    ("UncoveredLanguageError", ValueError),
    ("LowConfidenceLanguageError", ValueError),
    ("FormulationError", ValueError),
    ("ProfileConstraintError", ValueError),
    ("SettingsError", ValueError),
    ("DataReferenceError", LookupError),
    ("SmallSampleError", ValueError),
    ("DensityEstimationError", ValueError),
    ("DataIntegrityError", ValueError),
    ("HashSchemeError", ValueError),
    ("DownloadError", OSError),
    ("ProjectFormatError", ValueError),
    ("RunExistsError", FileExistsError),
    ("OutputFileLockedError", PermissionError),
]

_WARNINGS = [
    "MediumConfidenceLanguageWarning",
    "EqualAnglesFallbackWarning",
    "SmallSampleWarning",
    "PointsOutsideGridWarning",
]


@pytest.mark.parametrize(("name", "builtin"), _ERRORS)
def test_error_subclasses_root_and_builtin(name, builtin):
    cls = getattr(errors, name)
    assert issubclass(cls, SoundscapyError)
    assert issubclass(cls, builtin)
    err = cls("m", hint="h")
    assert str(err) == "m"
    assert err.hint == "h"
    assert cls("m").hint == ""


@pytest.mark.parametrize("name", _WARNINGS)
def test_warning_subclasses_root_and_user_warning(name):
    cls = getattr(errors, name)
    assert issubclass(cls, SoundscapyWarning)
    assert issubclass(cls, UserWarning)


def test_roots():
    assert issubclass(SoundscapyError, Exception)
    assert issubclass(SoundscapyWarning, UserWarning)


def test_all_lists_exactly_the_table_classes():
    expected = {name for name, _ in _ERRORS} | set(_WARNINGS)
    expected |= {"SoundscapyError", "SoundscapyWarning"}
    assert set(errors.__all__) == expected


def test_no_error_name_shadows_a_builtin():
    assert not set(errors.__all__) & set(dir(builtins))


def test_integrity_error_is_named_data_integrity_error():
    assert hasattr(errors, "DataIntegrityError")
    assert not hasattr(errors, "IntegrityError")
