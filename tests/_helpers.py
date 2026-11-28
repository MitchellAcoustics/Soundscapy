"""Constants and small builders shared by several test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
from hypothesis import strategies as st

from soundscapy.instrument.scale import Scale
from soundscapy.survey.quality import QualitySettings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# The eight attribute labels of iso-12913-outdoor, in position order (PAQ1..PAQ8).
LABELS = ("pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful")
LABELS += ("calm",)
# The ISD's CSV order, which is not position order.
CSV_ORDER = ("pleasant", "chaotic", "vibrant", "uneventful", "calm", "annoying", "eventful")
CSV_ORDER += ("monotonous",)
ONES = ",".join(["1"] * 8)
COMPLETE = "1,2,3,4,5,4,3,2"  # complete and not constant
COMPLETE_B = "5,4,3,2,1,2,3,4"
NOMINAL_ANGLES_DEG = np.arange(0.0, 360.0, 45.0)
SCALE_1_5 = Scale(minimum=1.0, maximum=5.0)


def quality_settings() -> st.SearchStrategy[QualitySettings]:
    """Draw a valid ``QualitySettings``: actions from their allowed sets, thresholds in range."""
    actions = {
        name: st.sampled_from(get_args(field.annotation))
        for name, field in QualitySettings.model_fields.items()
        if name not in ("excessive_missing_share", "min_age")
    }
    return st.builds(
        QualitySettings,
        excessive_missing_share=st.floats(0.0, 1.0),
        min_age=st.floats(0.0, 120.0),
        **actions,
    )


def fake_frame(module: str) -> object:
    """An object whose type claims to come from ``module``, without that library installed."""
    return type("DataFrame", (), {"__module__": module})()


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
