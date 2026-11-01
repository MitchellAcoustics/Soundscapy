"""The International Soundscape Database (ISD) as a ``Survey``.

The bundled subset holds 1552 responses from 15 locations of ISD 1.0.1-alpha.1 (Zenodo
record 10672568, CC BY 4.0), 40 columns with cell text unchanged, so tutorials and tests
run offline. Each version's mapping file and read options come from the dataset
registry shipped beside it.
"""

from typing import Any, Final

from soundscapy import _data
from soundscapy._model import _UNSET, _Unset
from soundscapy.errors import DataReferenceError
from soundscapy.survey.mapping import read_mapping
from soundscapy.survey.read import ReadOptions, Survey, _read_survey

__all__ = ["load"]

_DIR: Final = "datasets/isd/"
_REGISTRY: Final = _DIR + "registry.toml"
_SUBSET: Final = _DIR + "subset-v1.0.csv"
_SUBSET_VERSION: Final = "1.0.1-alpha.1"
"""The ISD version the bundled subset was made from."""


def _registry() -> dict[str, Any]:
    return _data.read_toml(_REGISTRY)


def load(*, subset: bool = False, version: str | _Unset = _UNSET) -> Survey:
    """Load the International Soundscape Database as a survey.

    The file is read with the mapping file and read options that the version's registry
    entry names; the result's ``provenance["data_refs"]`` records the dataset with its
    pinned hash.

    Parameters
    ----------
    subset : bool, default False
        Read the bundled subset (1552 responses from 15 locations). Only ``True`` is
        supported in this version; the full database arrives with ``fetch``.
    version : str, optional
        The ISD version; left unset, the registry's default. The subset accepts only
        ``"1.0.1-alpha.1"``, the version it was made from.

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
    dataset_ref = {
        "kind": "dataset",
        "name": "isd-subset",
        "version": resolved,
        "hash": _data.pin(_SUBSET),
    }
    with _data.as_path(_SUBSET) as path:
        return _read_survey(
            path,
            mapping,
            read_options=ReadOptions(**entry["read_options"]),
            data_refs=[dataset_ref],
        )
