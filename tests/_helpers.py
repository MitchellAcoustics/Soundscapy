"""Constants and small builders shared by several test modules."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest
from hypothesis import strategies as st

from soundscapy._model import COL_FLAGS
from soundscapy.errors import MediumConfidenceLanguageWarning
from soundscapy.instrument.scale import Scale
from soundscapy.projection.project import ProjectionSettings
from soundscapy.survey.quality import QualitySettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from soundscapy.distribution.coordinates import Coordinates

# The synthetic three-tier angle set and the small survey that exercises every resolver
# branch: qaa High, qab Medium, qac Low with angles, qad absent, qae Low tier-only.
SYNTHETIC_TIERS = Path(__file__).parent / "fixtures" / "synthetic_tiers"

# The eight attribute labels of iso-12913-outdoor, in position order (PAQ1..PAQ8).
LABELS = ("pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful")
LABELS += ("calm",)
# The ISD's CSV order, which is not position order.
CSV_ORDER = ("pleasant", "chaotic", "vibrant", "uneventful", "calm", "annoying", "eventful")
CSV_ORDER += ("monotonous",)
ONES = ",".join(["1"] * 8)
COMPLETE = "1,2,3,4,5,4,3,2"  # complete and not constant
COMPLETE_B = "5,4,3,2,1,2,3,4"
# The ISD subset's carried columns: the mapping's five, then the sidecar's ten features.
ISD_MAPPING_CARRY = ("start_time", "Survey_Version", "sss03", "sss04", "sss05")
ISD_FEATURES = ("LCeq_L(C)", "LAeq_L(A)", "LAeq_L10(A)", "LAeq_L90(A)", "N_N5", "S_S", "R_R")
ISD_FEATURES += ("T_TonalityHMS", "FS_F", "I_I")
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


def medium_warning() -> pytest.WarningsRecorder:
    """The context a projection with a medium-confidence language runs in.

    The suite turns warnings into errors, so a call that warns once about such a language
    is wrapped in this; the recorder lets a test assert that exactly one warning fired.
    """
    return pytest.warns(MediumConfidenceLanguageWarning)


def csv_rows(text: str) -> list[dict[str, str]]:
    """Read CSV text as one dict per row, every cell as text."""
    return list(csv.DictReader(io.StringIO(text, newline="")))


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


def write_text(tmp_path: Path, text: str, name: str = "settings.toml") -> Path:
    """Write text as a UTF-8 file; return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def projection_settings(coords: Coordinates) -> ProjectionSettings:
    """The settings a projection recorded, narrowed to ``ProjectionSettings``."""
    assert isinstance(coords.settings, ProjectionSettings)
    return coords.settings


def flag_mask(coords: Coordinates, flag: str) -> np.ndarray:
    """Which rows carry ``flag`` in their ``ssp_flags`` label."""
    return np.array([flag in cell.split(";") for cell in coords.labels[COL_FLAGS].tolist()])
