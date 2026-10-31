"""Shared fixtures."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import pandas as pd
import polars as pl
import pytest

MakeFrame: TypeAlias = Callable[[Mapping[str, Sequence[Any]]], Any]


@pytest.fixture(params=["pandas", "polars"])
def frame_lib(request: pytest.FixtureRequest) -> MakeFrame:
    """Return a callable that builds a native frame in pandas or in polars."""
    if request.param == "pandas":
        return lambda data: pd.DataFrame(dict(data))
    return lambda data: pl.DataFrame(dict(data))
