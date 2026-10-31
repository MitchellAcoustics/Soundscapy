"""Constants and small builders shared by several test modules."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from soundscapy.instrument.scale import Scale

# The eight attribute labels of iso-12913-outdoor, in position order (PAQ1..PAQ8).
LABELS = ("pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful")
LABELS += ("calm",)
# The ISD's CSV order, which is not position order.
CSV_ORDER = ("pleasant", "chaotic", "vibrant", "uneventful", "calm", "annoying", "eventful")
CSV_ORDER += ("monotonous",)
ONES = ",".join(["1"] * 8)
NOMINAL_ANGLES_DEG = np.arange(0.0, 360.0, 45.0)
SCALE_1_5 = Scale(minimum=1.0, maximum=5.0)


def raw_mapping(ids: dict, *, columns: Sequence[str] = LABELS) -> dict:
    """A minimal raw mapping dict: the given ids, each column mapped by its own label."""
    return {
        "schema": "soundscapy.mapping",
        "schema_version": 1,
        "ids": ids,
        "groups": {"attributes": {"columns": {label: label for label in columns}}},
    }


def write_csv(tmp_path: Path, header: str, *rows: str, name: str = "survey.csv") -> Path:
    """Write a header and rows as a UTF-8 file with LF line endings; return its path."""
    path = tmp_path / name
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8")
    return path
