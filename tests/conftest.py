"""Shared fixtures."""

# Before anything imports pyplot: tests run headless however they are started (task, plain
# pytest, an IDE), so the backend is fixed here, not by an environment variable.
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd
import polars as pl
import pytest

from soundscapy.datasets import isd
from soundscapy.projection.project import project

if TYPE_CHECKING:
    from soundscapy.distribution.coordinates import Coordinates
    from soundscapy.survey.read import Survey

MakeFrame: TypeAlias = Callable[[Mapping[str, Sequence[Any]]], Any]


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close the pyplot figures a test made (plot functions create them for ax=None).

    Looked up rather than imported, so test runs that never plot do not import pyplot.
    """
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture(params=["pandas", "polars"])
def frame_lib(request: pytest.FixtureRequest) -> MakeFrame:
    """Return a callable that builds a native frame in pandas or in polars."""
    if request.param == "pandas":
        return lambda data: pd.DataFrame(dict(data))
    return lambda data: pl.DataFrame(dict(data))


@pytest.fixture(scope="session")
def isd_survey() -> Survey:
    """The bundled ISD subset, read once per session; tests must not mutate it."""
    return isd.load(subset=True)


@pytest.fixture(scope="session")
def isd_coords(isd_survey: Survey) -> Coordinates:
    """The ISD subset projected with the default settings; tests must not mutate it."""
    return project(isd_survey)
