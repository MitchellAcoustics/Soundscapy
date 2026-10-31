"""Shared fixtures."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import pandas as pd
import polars as pl
import pytest

from soundscapy.datasets import isd
from soundscapy.distribution.coordinates import Coordinates
from soundscapy.projection.project import project
from soundscapy.survey.read import Survey

MakeFrame: TypeAlias = Callable[[Mapping[str, Sequence[Any]]], Any]


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
