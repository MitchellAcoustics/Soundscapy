"""Shared fixtures."""

# Before anything imports pyplot: tests run headless however they are started (task, plain
# pytest, an IDE), so the backend is fixed here, not by an environment variable.
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd
import pytest

from soundscapy import _data
from soundscapy.datasets import isd
from soundscapy.instrument import circumplex
from soundscapy.projection.project import project
from soundscapy.provenance import hash_bytes
from soundscapy.survey.read import read_survey
from tests._helpers import medium_warning

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from soundscapy.distribution.coordinates import Coordinates
    from soundscapy.survey.read import Survey

MakeFrame: TypeAlias = Callable[[Mapping[str, Sequence[Any]]], Any]

# The synthetic indoor circumplex (PRD FR-2.0) and the survey and mapping that use it.
INDOOR = Path(__file__).parent / "fixtures" / "indoor"
INDOOR_RELPATH = "circumplexes/indoor-test/1.toml"
# The two-response survey behind the ISO/TS 12913-3:2019 hand values (spec section 7.2).
PROJECTION_FIXTURES = Path(__file__).parent / "projection" / "fixtures"


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close the pyplot figures a test made (plot functions create them for ax=None).

    Looked up rather than imported, so test runs that never plot do not import pyplot.
    """
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture(params=["classic", "nullable"])
def frame_lib(request: pytest.FixtureRequest) -> MakeFrame:
    """Return a callable that builds a pandas frame with classic or with nullable dtypes.

    The nullable form is what ``DataFrame.convert_dtypes()`` gives: ``Int64``, ``Float64``,
    ``boolean`` and ``string`` columns with ``pd.NA`` for a missing value.
    """
    if request.param == "classic":
        return lambda data: pd.DataFrame(dict(data))
    return lambda data: pd.DataFrame(dict(data)).convert_dtypes()


@pytest.fixture(scope="session")
def isd_survey() -> Survey:
    """The bundled ISD subset, read once per session; tests must not mutate it."""
    return isd.load(subset=True)


@pytest.fixture(scope="module")
def hand_survey() -> Survey:
    """The hand-values survey, read once per test module; tests must not mutate it."""
    return read_survey(
        PROJECTION_FIXTURES / "hand_values.csv", PROJECTION_FIXTURES / "mapping.toml"
    )


@pytest.fixture(scope="session")
def isd_coords(isd_survey: Survey) -> Coordinates:
    """The ISD subset projected with the default settings; tests must not mutate it.

    Under the default profile the subset's 69 Mandarin rows project with medium-confidence
    angles, so the one warning that fires is expected here.
    """
    with medium_warning():
        return project(isd_survey)


@pytest.fixture
def indoor_circumplex_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Register the synthetic indoor circumplex as if it shipped, for one test.

    ``_data.PINS`` gets a copy with the fixture file's real hash under
    ``circumplexes/indoor-test/1.toml``, and ``_data._resource`` resolves that one path to
    the fixture file, so ``read_survey``, ``project`` and the plot functions run unchanged.
    The ``circumplex()`` cache is cleared on entry and exit, so no test sees a stale
    instrument. Nothing is written to the repository. Yields the fixture directory, which
    holds ``circumplex.toml``, ``survey.csv`` and ``mapping.toml``.
    """
    fixture = INDOOR / "circumplex.toml"
    original_resource = _data._resource  # noqa: SLF001  # the registry seam the tests override

    def resource(relpath: str) -> Traversable:
        return fixture if relpath == INDOOR_RELPATH else original_resource(relpath)

    pins = {**_data.PINS, INDOOR_RELPATH: hash_bytes(fixture.read_bytes())}
    monkeypatch.setattr(_data, "PINS", pins)
    monkeypatch.setattr(_data, "_resource", resource)
    circumplex.cache_clear()
    try:
        yield INDOOR
    finally:
        monkeypatch.undo()
        circumplex.cache_clear()
