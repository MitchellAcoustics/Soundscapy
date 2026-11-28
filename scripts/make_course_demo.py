# /// script
# requires-python = ">=3.11"
# dependencies = ["tomlkit>=0.13.1"]
# ///
"""Make the bundled course dataset and its sidecar.

Reads ``examples/data/DemoData.csv`` of soundscapy 0.x, the renamed ISD extract the IoA
soundscape course uses, from ``--source PATH`` or with ``git show`` from ``--git-ref``
(default ``origin/main``), refuses it unless its sha256 is the pinned one, and writes into
``src/soundscapy/_data/datasets/course_demo/``:

- ``course_demo.csv``: every row, with ``free_response``, ``ISOPleasant`` and
  ``ISOEventful`` dropped and every other cell unchanged, UTF-8 without BOM, LF line endings;
- ``course_demo.toml``: the sidecar (source, hashes, columns, features, citation).

The mapping, ``mapping.toml``, is written by hand. Running this twice gives identical bytes.
Run with ``uv run scripts/make_course_demo.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

import tomlkit

SOURCE_PATH = "examples/data/DemoData.csv"
SOURCE_COMMIT = "325e97c1aff124b7d2d426af24592d8a8357b67f"
SOURCE_SHA256 = "d6d4f937b53f02f4bdaa28e8401fe90061d0d9fde9da89c0ec898baaa4fcdd3a"
OUT_DIR = Path(__file__).resolve().parents[1] / "src/soundscapy/_data/datasets/course_demo"
DROPPED = ("free_response", "ISOPleasant", "ISOEventful")
FEATURES = ("LAeq", "LA10", "LA90", "LCeq", "N5", "sharpness", "roughness", "tonality")

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
        "Renamed extract of the ISD used in the IoA soundscape course (examples/data/DemoData.csv "
        "on soundscapy 0.x main): the ISD 1.0.1-alpha.1 rows of CamdenTown, MarchmontGarden, "
        "PancrasLock and TateModern through the four-part key, less "
        "PancrasLock/PancrasLock2/PL202/410 and TateModern/TateModern1/TM124/663. All versions: "
        "concept DOI 10.5281/zenodo.5578572. free_response, ISOPleasant and ISOEventful dropped; "
        "other cells unchanged. The file has no language column: the ISD records 454 rows as eng "
        "and 2 as spa (CamdenTown/CamdenTown3/CT323/10009 and 10010); the mapping sets eng for "
        "all. Made by scripts/make_course_demo.py."
    ),
}


def _lines(values: list[str]) -> tomlkit.items.Array:
    array = tomlkit.array()
    array.extend(values)
    return array.multiline(multiline=True)


def course_csv(source: bytes) -> tuple[bytes, list[str], list[str], int]:
    """Return the CSV bytes, its columns, the locations in first-appearance order and the rows."""
    reader = csv.reader(io.StringIO(source.decode("utf-8"), newline=""))
    header = next(reader)
    missing = sorted(set(DROPPED) - set(header))
    if missing:
        sys.exit(f"source lacks columns {missing}")
    keep = [i for i, name in enumerate(header) if name not in DROPPED]
    location = header.index("LocationID")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([header[i] for i in keep])
    n_rows = 0
    locations: list[str] = []
    for row in reader:
        writer.writerow([row[i] for i in keep])
        if row[location] not in locations:
            locations.append(row[location])
        n_rows += 1
    return out.getvalue().encode("utf-8"), [header[i] for i in keep], locations, n_rows


def sidecar(columns: list[str], locations: list[str], copy: bytes, n_rows: int) -> str:
    """Return the sidecar TOML text."""
    doc = tomlkit.document()
    doc["schema"] = "soundscapy.dataset_sidecar"
    doc["schema_version"] = 1
    doc["name"] = "course-demo"
    doc["version"] = "1"
    doc["source_file"] = SOURCE_PATH
    doc["source_ref"] = f"soundscapy 0.x main, commit {SOURCE_COMMIT}"
    doc["source_sha256"] = SOURCE_SHA256
    doc["copy_hash"] = "h1-sha256:" + hashlib.sha256(copy).hexdigest()
    doc["n_rows"] = n_rows
    doc["dropped_columns"] = list(DROPPED)
    doc["locations"] = _lines(locations)
    doc["columns"] = _lines(columns)
    features = tomlkit.table()
    features["columns"] = _lines([c for c in columns if c in FEATURES])
    features["units_verified"] = False
    doc["features"] = features
    citation = tomlkit.table()
    citation.update(CITATION)
    doc["citation"] = citation
    return tomlkit.dumps(doc)


def _from_git(ref: str) -> bytes:
    git = shutil.which("git") or "git"
    result = subprocess.run(  # noqa: S603  # fixed git command; the ref is the --git-ref argument
        [git, "show", f"{ref}:{SOURCE_PATH}"],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    return result.stdout


def main() -> None:
    """Check the source, then write the dataset and its sidecar."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help=f"a local copy of {SOURCE_PATH!r}")
    parser.add_argument("--git-ref", default="origin/main", help="the 0.x ref to read it from")
    args = parser.parse_args()
    source = args.source.read_bytes() if args.source else _from_git(args.git_ref)
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        where = args.source or f"{args.git_ref}:{SOURCE_PATH}"
        sys.exit(f"refusing {where}: sha256 {digest}, expected {SOURCE_SHA256}")
    copy, columns, locations, n_rows = course_csv(source)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "course_demo.csv": copy,
        "course_demo.toml": sidecar(columns, locations, copy, n_rows).encode("utf-8"),
    }
    for name, data in outputs.items():
        (OUT_DIR / name).write_bytes(data)
        print(f"wrote {OUT_DIR / name} ({len(data)} bytes)")
    print(f"{n_rows} rows, {len(columns)} columns; csv sha256 {hashlib.sha256(copy).hexdigest()}")


if __name__ == "__main__":
    main()
