"""Errors and warnings raised by soundscapy.

Every error subclasses :class:`SoundscapyError` and the built-in named in its docstring, so
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
    """A survey mapping is invalid or incomplete.

    It also covers an unknown construct key or a bad language code.
    """


class CodingError(SoundscapyError, ValueError):
    """A text value matches no missing token, value-map key or number."""


class EncodingError(SoundscapyError, ValueError):
    """A file cannot be decoded.

    The hint suggests an explicit encoding such as ``encoding="cp1252"``.
    """


class FileFormatError(SoundscapyError, ValueError):
    """A file's format cannot be determined.

    Its delimiter cannot be resolved, its suffix is unknown, or a multi-sheet workbook is read
    without naming a sheet.
    """


class ColumnCollisionError(SoundscapyError, ValueError):
    """An input column collides with a derived column.

    An input column starts with the reserved ``ssp_`` prefix, or attaching or joining
    results would overwrite an existing column.
    """


class AxesError(SoundscapyError, ValueError):
    """A plot function is given an ``ax`` of the wrong kind.

    For example, a radar plot needs polar axes.
    """


class AlignmentError(SoundscapyError, ValueError):
    """A result cannot be aligned with the frame it is attached to."""


class ScaleViolationError(SoundscapyError, ValueError):
    """Coded values fall outside the declared response scale."""


class MissingResponseIdError(SoundscapyError, ValueError):
    """A response has no identifier where one is required."""


class DuplicateResponseIdError(SoundscapyError, ValueError):
    """Two or more responses share the same identifier."""


class UncoveredLanguageError(SoundscapyError, ValueError):
    """A language has no angle set available.

    The message names the tier of a tier-only language.
    """


class LowConfidenceLanguageError(SoundscapyError, ValueError):
    """A language has only low-confidence angles and the settings do not allow them."""


class FormulationError(SoundscapyError, ValueError):
    """A projection formulation cannot be applied.

    ``iso2019`` needs a 1 to 5 scale with nominal angles, and any formulation rejects
    angles whose summed absolute cosines or sines are below 1e-9.
    """


class ProfileConstraintError(SoundscapyError, ValueError):
    """A setting conflicts with a field locked by the chosen profile."""


class SettingsError(SoundscapyError, ValueError):
    """A settings file or section fails validation."""


class DataReferenceError(SoundscapyError, LookupError):
    """A named circumplex, angle set, profile or dataset version does not exist."""


class SmallSampleError(SoundscapyError, ValueError):
    """A group has too few valid responses for the requested operation."""


class DensityEstimationError(SoundscapyError, ValueError):
    """A density cannot be estimated because the points are collinear or identical."""


class DataIntegrityError(SoundscapyError, ValueError):
    """A content hash does not match its expected value. The message names both hashes."""


class HashSchemeError(SoundscapyError, ValueError):
    """A hash string carries an unknown scheme prefix."""


class DownloadError(SoundscapyError, OSError):
    """A download fails because of an HTTP or connection error."""


class ProjectFormatError(SoundscapyError, ValueError):
    """Project files are missing, malformed or of an unknown format version."""


class RunExistsError(SoundscapyError, FileExistsError):
    """A run is saved to a location that already holds one and overwriting was not requested."""


class OutputFileLockedError(SoundscapyError, PermissionError):
    """The target file is open elsewhere and cannot be written.

    The hint suggests closing it, for example in Excel.
    """


class MediumConfidenceLanguageWarning(SoundscapyWarning):
    """Responses use medium-confidence languages.

    Issued once per projection; the message names those languages and their counts.
    """


class EqualAnglesFallbackWarning(SoundscapyWarning):
    """Equal angles replaced a missing angle set.

    Issued once per projection; the message names the languages and counts.
    """


class SmallSampleWarning(SoundscapyWarning):
    """A group has fewer than 30 valid responses.

    Issued once per call when the caller has not opted in.
    """


class PointsOutsideGridWarning(SoundscapyWarning):
    """Valid points fall outside the clip box.

    Issued once per density call; only a user-set clip narrower than the data can cause it.
    """
