"""The course dataset: the renamed ISD extract the IoA soundscape course uses, as a ``Survey``.

The bundled file holds 456 responses from four London locations of ISD 1.0.1-alpha.1
(CC BY 4.0), the ``DemoData.csv`` of soundscapy 0.x with its free-text, projection
columns dropped and every other cell unchanged, with its own mapping (``PAQ1`` to
``PAQ8`` column names, the four source items, overall quality and appropriateness).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from soundscapy import _data
from soundscapy._model import _UNSET, _Unset
from soundscapy.survey.mapping import read_mapping
from soundscapy.survey.read import (
    ImportSettings,
    ReadOptions,
    Survey,
    _read_survey,
    resolve_import_settings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from soundscapy.survey.quality import QualitySettings

__all__ = ["load"]

_DIR: Final = "datasets/course_demo/"
_CSV: Final = _DIR + "course_demo.csv"
_MAPPING: Final = _DIR + "mapping.toml"
_VERSION: Final = "1"


def load(
    *,
    quality: QualitySettings | Mapping[str, object] | _Unset = _UNSET,
    on_scale_violation: Literal["error", "exclude"] | _Unset = _UNSET,
    settings: ImportSettings | Mapping[str, object] | _Unset = _UNSET,
) -> Survey:
    """Load the course dataset as a survey.

    The file is read with its bundled mapping; the result's ``provenance["data_refs"]``
    records the dataset with its pinned hash. Every row is an ISD 1.0.1-alpha.1 row, and
    the mapping sets the language to English for all, since the file has no language
    column.

    Parameters
    ----------
    quality : QualitySettings or mapping, optional
        The quality rules, as ``read_survey()`` takes them.
    on_scale_violation : {"error", "exclude"}, optional
        What a coded value outside its scale does, as ``read_survey()`` takes it.
    settings : ImportSettings or mapping, optional
        Every import setting at once; the two keywords above override its fields.

    Returns
    -------
    Survey
        The 456 responses, read and coded against the course mapping.

    Raises
    ------
    DataIntegrityError
        If a shipped file does not match its pinned hash.
    SettingsError
        If a settings argument has the wrong type or fails validation.

    References
    ----------
    The database the rows come from: [@isd2024].
    """
    with _data.as_path(_MAPPING) as mapping_path:
        mapping = read_mapping(mapping_path)
    import_settings = resolve_import_settings(
        quality=quality, on_scale_violation=on_scale_violation, settings=settings
    )
    dataset_ref = {
        "kind": "dataset",
        "name": "course-demo",
        "version": _VERSION,
        "hash": _data.pin(_CSV),
    }
    with _data.as_path(_CSV) as path:
        return _read_survey(
            path,
            mapping,
            read_options=ReadOptions(),
            settings=import_settings,
            data_refs=[dataset_ref],
        )
