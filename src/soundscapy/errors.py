"""Errors and warnings raised by soundscapy (spec section 6.3.10).

Every error subclasses :class:`SoundscapyError` and the built-in named in the spec table, so
callers can catch either. Every error takes ``(message, *, hint="")``; ``str(err)`` is the
message and :attr:`SoundscapyError.hint` carries an optional second line for the CLI. No name
shadows a standard-library one.
"""

__all__ = [
    "AlignmentError",
    "AxesError",
    "CodingError",
    "ColumnCollisionError",
    "DataIntegrityError",
    "DataReferenceError",
    "DensityEstimationError",
    "DownloadError",
    "DuplicateResponseIdError",
    "EncodingError",
    "EqualAnglesFallbackWarning",
    "FileFormatError",
    "FormulationError",
    "HashSchemeError",
    "LowConfidenceLanguageError",
    "MappingError",
    "MediumConfidenceLanguageWarning",
    "MissingResponseIdError",
    "OutputFileLockedError",
    "PointsOutsideGridWarning",
    "ProfileConstraintError",
    "ProjectFormatError",
    "RunExistsError",
    "ScaleViolationError",
    "SettingsError",
    "SmallSampleError",
    "SmallSampleWarning",
    "SoundscapyError",
    "SoundscapyWarning",
    "UncoveredLanguageError",
]


class SoundscapyError(Exception):
    """Root of every soundscapy error.

    Parameters
    ----------
    message : str
        What went wrong; this is ``str(err)``.
    hint : str, optional
        What to try next; the CLI prints it on a second line.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class SoundscapyWarning(UserWarning):
    """Root of every soundscapy warning."""


class MappingError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``MappingError``)."""


class CodingError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``CodingError``)."""


class EncodingError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``EncodingError``)."""


class FileFormatError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``FileFormatError``)."""


class ColumnCollisionError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``ColumnCollisionError``)."""


class AxesError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``AxesError``)."""


class AlignmentError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``AlignmentError``)."""


class ScaleViolationError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``ScaleViolationError``)."""


class MissingResponseIdError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``MissingResponseIdError``)."""


class DuplicateResponseIdError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``DuplicateResponseIdError``)."""


class UncoveredLanguageError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``UncoveredLanguageError``)."""


class LowConfidenceLanguageError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``LowConfidenceLanguageError``)."""


class FormulationError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``FormulationError``)."""


class ProfileConstraintError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``ProfileConstraintError``)."""


class SettingsError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``SettingsError``)."""


class DataReferenceError(SoundscapyError, LookupError):
    """Raised for the condition named in spec section 6.3.10 (``DataReferenceError``)."""


class SmallSampleError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``SmallSampleError``)."""


class DensityEstimationError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``DensityEstimationError``)."""


class DataIntegrityError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``DataIntegrityError``)."""


class HashSchemeError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``HashSchemeError``)."""


class DownloadError(SoundscapyError, OSError):
    """Raised for the condition named in spec section 6.3.10 (``DownloadError``)."""


class ProjectFormatError(SoundscapyError, ValueError):
    """Raised for the condition named in spec section 6.3.10 (``ProjectFormatError``)."""


class RunExistsError(SoundscapyError, FileExistsError):
    """Raised for the condition named in spec section 6.3.10 (``RunExistsError``)."""


class OutputFileLockedError(SoundscapyError, PermissionError):
    """Raised for the condition named in spec section 6.3.10 (``OutputFileLockedError``)."""


class MediumConfidenceLanguageWarning(SoundscapyWarning):
    """Warning ``MediumConfidenceLanguageWarning`` (spec section 6.3.10)."""


class EqualAnglesFallbackWarning(SoundscapyWarning):
    """Warning ``EqualAnglesFallbackWarning`` (spec section 6.3.10)."""


class SmallSampleWarning(SoundscapyWarning):
    """Warning ``SmallSampleWarning`` (spec section 6.3.10)."""


class PointsOutsideGridWarning(SoundscapyWarning):
    """Warning ``PointsOutsideGridWarning`` (spec section 6.3.10)."""
