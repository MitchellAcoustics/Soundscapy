# /// script
# requires-python = ">=3.11"
# dependencies = ["tomlkit>=0.13.1"]
# ///
"""Make the bundled ISD subset, its sidecar and the ISD dataset registry.

Reads ``ISD v1.0 Data.csv`` of ISD 1.0.1-alpha.1 (Zenodo record 10672568, CC BY 4.0),
either downloaded or from ``--source PATH``, refuses it unless its sha256 is the pinned
one, and writes into ``src/soundscapy/_data/datasets/isd/``:

- ``subset-v1.0.csv``: the rows of 15 locations and 40 columns, in source order, cell
  text unchanged, UTF-8 without BOM, LF line endings;
- ``subset-v1.0.toml``: the sidecar (source, hashes, locations, columns, citation);
- ``registry.toml``: the dataset registry with the 1.0.1-alpha.1 entry.

Running it twice gives identical bytes. Run with ``uv run scripts/make_isd_subset.py``.
"""

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
from pathlib import Path
from typing import Any

import tomlkit

SOURCE_URL = "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv"
SOURCE_FILE = "ISD v1.0 Data.csv"
SOURCE_SHA256 = "27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735"
SOURCE_VERSION = "1.0.1-alpha.1"
SOURCE_RECORD = "10672568"
OUT_DIR = Path(__file__).resolve().parents[1] / "src/soundscapy/_data/datasets/isd"

LOCATIONS = (
    "CamdenTown",
    "EustonTap",
    "MarchmontGarden",
    "PancrasLock",
    "RegentsParkFields",
    "RegentsParkJapan",
    "RussellSq",
    "StPaulsCross",
    "StPaulsRow",
    "TateModern",
    "TorringtonSq",
    "CampoPrincipe",
    "SanMarco",
    "Noorderplantsoen",
    "PingshanPark",
)
FEATURES = (
    "LAeq_L(A)",
    "LAeq_L10(A)",
    "LAeq_L90(A)",
    "LCeq_L(C)",
    "N_N5",
    "S_S",
    "R_R",
    "T_TonalityHMS",
    "FS_F",
    "I_I",
)
KEPT = frozenset(
    (
        "LocationID",
        "SessionID",
        "GroupID",
        "RecordID",
        "start_time",
        "Language",
        "Survey_Version",
        *(f"ssi0{i}" for i in range(1, 5)),
        "pleasant",
        "chaotic",
        "vibrant",
        "uneventful",
        "calm",
        "annoying",
        "eventful",
        "monotonous",
        *(f"sss0{i}" for i in range(1, 6)),
        *(f"who0{i}" for i in range(1, 6)),
        "age00",
        *FEATURES,
    )
)
LAST_SURVEY_COLUMN = "RecordingLength"  # the acoustic feature columns follow it

CITATION = {
    "citation": (
        "Mitchell, A., Oberman, T., Aletta, F., Erfanian, M., Kachlicka, M., Lionello, M., "
        "Fang, X., and Kang, J. (2024). The International Soundscape Database: An integrated "
        "multimedia database of urban soundscape surveys -- questionnaires with acoustical "
        "and contextual information (version 1.0.1-alpha.1) [Data set]. Zenodo."
    ),
    "doi_or_url": "https://doi.org/10.5281/zenodo.10672568",
    "licence": "CC-BY-4.0",
    "date": "2024-02-16",
    "notes": (
        "Subset of ISD 1.0.1-alpha.1 (Zenodo record 10672568; all versions: concept DOI "
        "10.5281/zenodo.5578572). 15 locations and 40 columns, cell text unchanged. Made by "
        "scripts/make_isd_subset.py."
    ),
}
REGISTRY_ENTRY = {
    "record": SOURCE_RECORD,
    "doi": "10.5281/zenodo.10672568",
    "file": SOURCE_FILE,
    "url": SOURCE_URL,
}
REGISTRY_TAIL = {
    "licence": "CC-BY-4.0",
    "published": "2024-02-16",
    "citation": (
        "Mitchell, A., et al. (2024). The International Soundscape Database (version "
        "1.0.1-alpha.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10672568"
    ),
    "mapping": "isd_mapping.toml",
}


def _lines(values: list[str]) -> Any:  # noqa: ANN401  # a tomlkit array
    array = tomlkit.array()
    array.extend(values)
    return array.multiline(multiline=True)


def subset_csv(source: bytes) -> tuple[bytes, list[str], list[str], int]:
    """Return the subset CSV bytes, the kept columns, the feature columns and the row count."""
    reader = csv.reader(io.StringIO(source.decode("utf-8"), newline=""))
    header = next(reader)
    missing = sorted(KEPT - set(header))
    if missing:
        sys.exit(f"source lacks columns {missing}")
    keep = [i for i, name in enumerate(header) if name in KEPT]
    location = header.index("LocationID")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([header[i] for i in keep])
    n_rows = 0
    seen: set[str] = set()
    for row in reader:
        if row[location] in LOCATIONS:
            writer.writerow([row[i] for i in keep])
            seen.add(row[location])
            n_rows += 1
    if seen != set(LOCATIONS):
        sys.exit(f"source lacks locations {sorted(set(LOCATIONS) - seen)}")
    features = header[header.index(LAST_SURVEY_COLUMN) + 1 :]
    return out.getvalue().encode("utf-8"), [header[i] for i in keep], features, n_rows


def sidecar(columns: list[str], subset: bytes, n_rows: int) -> str:
    """Return the sidecar TOML text."""
    doc = tomlkit.document()
    doc["schema"] = "soundscapy.dataset_sidecar"
    doc["schema_version"] = 1
    doc["name"] = "isd-subset"
    doc["source_version"] = SOURCE_VERSION
    doc["source_record"] = SOURCE_RECORD
    doc["source_file"] = SOURCE_FILE
    doc["source_sha256"] = SOURCE_SHA256
    doc["subset_hash"] = "h1-sha256:" + hashlib.sha256(subset).hexdigest()
    doc["n_rows"] = n_rows
    doc["locations"] = _lines(list(LOCATIONS))
    doc["columns"] = _lines(columns)
    features = tomlkit.table()
    features["columns"] = _lines([c for c in columns if c in FEATURES])
    features["units_verified"] = False
    doc["features"] = features
    citation = tomlkit.table()
    citation.update(CITATION)
    doc["citation"] = citation
    return tomlkit.dumps(doc)


def registry(source: bytes, feature_columns: list[str]) -> str:
    """Return the registry TOML text with the 1.0.1-alpha.1 entry."""
    doc = tomlkit.document()
    doc["schema"] = "soundscapy.dataset_registry"
    doc["schema_version"] = 1
    doc["concept_doi"] = "10.5281/zenodo.5578572"
    doc["default"] = SOURCE_VERSION
    entry = tomlkit.table()
    entry.update(REGISTRY_ENTRY)
    entry["size"] = len(source)
    entry["md5"] = hashlib.md5(source, usedforsecurity=False).hexdigest()
    entry["sha256"] = SOURCE_SHA256
    entry.update(REGISTRY_TAIL)
    entry["read_options"] = tomlkit.inline_table()
    entry["feature_columns"] = _lines(feature_columns)
    entry["notes"] = ""
    versions = tomlkit.table(is_super_table=True)
    versions[SOURCE_VERSION] = entry
    doc["versions"] = versions
    return tomlkit.dumps(doc)


def _download() -> bytes:
    print(f"downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL) as response:  # fixed https URL
        return response.read()


def main() -> None:
    """Check the source, then write the subset, its sidecar and the registry."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help=f"a local copy of {SOURCE_FILE!r}")
    args = parser.parse_args()
    source = args.source.read_bytes() if args.source else _download()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        sys.exit(f"refusing {args.source or SOURCE_URL}: sha256 {digest}, expected {SOURCE_SHA256}")
    subset, columns, features, n_rows = subset_csv(source)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "subset-v1.0.csv": subset,
        "subset-v1.0.toml": sidecar(columns, subset, n_rows).encode("utf-8"),
        "registry.toml": registry(source, features).encode("utf-8"),
    }
    for name, data in outputs.items():
        (OUT_DIR / name).write_bytes(data)
        print(f"wrote {OUT_DIR / name} ({len(data)} bytes)")
    print(f"{n_rows} rows, {len(columns)} columns, {len(features)} feature columns in registry")


if __name__ == "__main__":
    main()
