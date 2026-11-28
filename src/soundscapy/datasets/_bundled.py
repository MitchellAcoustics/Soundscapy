"""Reading a bundled dataset: the ritual the ISD and course loaders share."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from soundscapy import _data
from soundscapy._model import _UNSET, _Unset
from soundscapy.survey.read import (
    ImportSettings,
    ReadOptions,
    Survey,
    _read_survey,
    resolve_import_settings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from soundscapy.survey.mapping import SurveyMapping
    from soundscapy.survey.quality import QualitySettings

__all__ = ["load_bundled"]

_DEFAULT_READ_OPTIONS: Final = ReadOptions()


def load_bundled(
    csv: str,
    mapping: SurveyMapping,
    *,
    name: str,
    version: str,
    read_options: ReadOptions = _DEFAULT_READ_OPTIONS,
    quality: QualitySettings | Mapping[str, object] | _Unset = _UNSET,
    on_scale_violation: Literal["error", "exclude"] | _Unset = _UNSET,
    settings: ImportSettings | Mapping[str, object] | _Unset = _UNSET,
) -> Survey:
    """Read a shipped CSV as a survey, recording the dataset with its pinned hash.

    Parameters
    ----------
    csv : str
        The CSV's path relative to ``_data/``.
    mapping : SurveyMapping
        The mapping to read it with.
    name, version : str
        The dataset's name and version for ``provenance["data_refs"]``.
    read_options : ReadOptions, optional
        How the file is read.
    quality, on_scale_violation, settings
        The import settings, as ``read_survey()`` takes them.

    Returns
    -------
    Survey
        The survey, read and coded against ``mapping``.
    """
    import_settings = resolve_import_settings(
        quality=quality, on_scale_violation=on_scale_violation, settings=settings
    )
    dataset_ref = {"kind": "dataset", "name": name, "version": version, "hash": _data.pin(csv)}
    with _data.as_path(csv) as path:
        return _read_survey(
            path,
            mapping,
            read_options=read_options,
            settings=import_settings,
            data_refs=[dataset_ref],
        )
