"""Frame types reach the caller (review F2): narwhals' IntoDataFrameT in, the same type out.

``typing.assert_type`` does nothing at run time; ``task typecheck`` (ty) checks every call
here, so a signature that falls back to ``Any`` fails the type check.
"""

from typing import assert_type

import pandas as pd
import polars as pl

from soundscapy import attach
from soundscapy.distribution.coordinates import Coordinates
from soundscapy.survey.read import Survey


def test_attach_returns_the_callers_frame_type(isd_survey: Survey, isd_coords: Coordinates):
    pandas_out = attach(isd_survey.data, isd_coords)
    assert_type(pandas_out, pd.DataFrame)
    assert isinstance(pandas_out, pd.DataFrame)
    columns = {str(name): values for name, values in isd_survey.data.to_dict(orient="list").items()}
    polars_out = attach(pl.DataFrame(columns), isd_coords)
    assert_type(polars_out, pl.DataFrame)
    assert isinstance(polars_out, pl.DataFrame)


def test_to_frame_returns_pandas_or_the_library_of_like(isd_coords: Coordinates):
    assert_type(isd_coords.to_frame(), pd.DataFrame)
    like = pl.DataFrame({"z": [0]})
    out = isd_coords.to_frame(like=like)
    assert_type(out, pl.DataFrame)
    assert isinstance(out, pl.DataFrame)


def test_survey_data_and_status_frame_are_pandas(isd_survey: Survey):
    assert_type(isd_survey.data, pd.DataFrame)
    assert_type(isd_survey.status.to_frame(), pd.DataFrame)
