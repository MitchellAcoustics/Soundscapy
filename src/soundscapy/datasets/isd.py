"""The International Soundscape Database (ISD) as a ``Survey``.

The bundled subset holds 1552 responses from 15 locations of ISD 1.0.1-alpha.1 (Zenodo
record 10672568, CC BY 4.0), 40 columns with cell text unchanged, so tutorials and tests
run offline. Each version's mapping file and read options come from the dataset
registry shipped beside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

from soundscapy import _data
from soundscapy._model import _UNSET, _Unset
from soundscapy.datasets._bundled import load_bundled
from soundscapy.errors import DataReferenceError
from soundscapy.survey.mapping import read_mapping
from soundscapy.survey.read import ImportSettings, ReadOptions, Survey

if TYPE_CHECKING:
    from collections.abc import Mapping

    from soundscapy.survey.quality import QualitySettings

__all__ = ["load"]

_DIR: Final = "datasets/isd/"
_REGISTRY: Final = _DIR + "registry.toml"
_SUBSET: Final = _DIR + "subset-v1.0.csv"
_SIDECAR: Final = _DIR + "subset-v1.0.toml"
_SUBSET_VERSION: Final = "1.0.1-alpha.1"
"""The ISD version the bundled subset was made from."""


def _registry() -> dict[str, Any]:
    return _data.read_toml(_REGISTRY)


def load(
    *,
    subset: bool = False,
    version: str | _Unset = _UNSET,
    quality: QualitySettings | Mapping[str, object] | _Unset = _UNSET,
    on_scale_violation: Literal["error", "exclude"] | _Unset = _UNSET,
    settings: ImportSettings | Mapping[str, object] | _Unset = _UNSET,
) -> Survey:
    """Load the International Soundscape Database as a survey.

    The file is read with the mapping file and read options that the version's registry
    entry names; the result's ``provenance["data_refs"]`` records the dataset with its
    pinned hash. For the subset, the ten acoustic feature columns its sidecar lists are
    added to the mapping's ``carry``, so they travel as labels into every result and
    ``survey.mapping`` describes what was read.

    Parameters
    ----------
    subset : bool, default False
        Read the bundled subset (1552 responses from 15 locations). Only ``True`` is
        supported in this version; the full database arrives with ``fetch``.
    version : str, optional
        The ISD version; left unset, the registry's default. The subset accepts only
        ``"1.0.1-alpha.1"``, the version it was made from.
    quality : QualitySettings or mapping, optional
        The quality rules, as ``read_survey()`` takes them.
    on_scale_violation : {"error", "exclude"}, optional
        What a coded value outside its scale does, as ``read_survey()`` takes it.
    settings : ImportSettings or mapping, optional
        Every import setting at once; the two keywords above override its fields.

    Returns
    -------
    Survey
        The survey, read and coded against the ISD mapping.

    Raises
    ------
    NotImplementedError
        If ``subset`` is false: the full download is not available in this version.
    DataReferenceError
        If ``version`` is not the subset's version, or a file the registry names is not
        shipped.
    DataIntegrityError
        If a shipped file does not match its pinned hash.
    SettingsError
        If a settings argument has the wrong type or fails validation.

    References
    ----------
    The database: [@isd2024].
    """
    if not subset:
        msg = (
            "loading the full ISD needs a download, which arrives with datasets.isd.fetch; "
            "pass subset=True to load the bundled subset"
        )
        raise NotImplementedError(msg)
    registry = _registry()
    resolved = registry["default"] if version is _UNSET else version
    if resolved != _SUBSET_VERSION:
        msg = f"the bundled ISD subset is made from version {_SUBSET_VERSION!r}, not {resolved!r}"
        raise DataReferenceError(msg, hint=f"accepted with subset=True: [{_SUBSET_VERSION!r}]")
    entry = registry["versions"][resolved]
    with _data.as_path(_DIR + entry["mapping"]) as mapping_path:
        mapping = read_mapping(mapping_path)
    features = _data.read_toml(_SIDECAR)["features"]["columns"]
    # A copy without re-validation: the pinned feature names are not blank, repeated or id
    # columns, and reading checks they are in the file.
    mapping = mapping.model_copy(update={"carry": (*mapping.carry, *features)})
    return load_bundled(
        _SUBSET,
        mapping,
        name="isd-subset",
        version=resolved,
        read_options=ReadOptions(**entry["read_options"]),
        quality=quality,
        on_scale_violation=on_scale_violation,
        settings=settings,
    )
