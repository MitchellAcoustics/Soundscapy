import warnings
from enum import StrEnum
from functools import partial
from typing import Annotated

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Field
from pandera.typing.pandas import DataFrame, Series
from pydantic import BeforeValidator, ConfigDict
from pydantic.dataclasses import dataclass
from rpy2 import robjects as ro

import soundscapy.r_wrapper as sspyr
from soundscapy import PAQ_IDS, PAQ_LABELS, get_logger

logger = get_logger()

AllowNan = partial(pa.Field, nullable=True)


class CircModelE(StrEnum):
    """Enumeration of circumplex model types."""

    UNCONSTRAINED = "unconstrained"
    EQUAL_ANG = "equal_ang"
    EQUAL_COM = "equal_com"
    CIRCUMPLEX = "circumplex"


class SATPSchema(pa.DataFrameModel):
    PAQ1: Series[float] = Field(ge=0, le=100)
    PAQ2: Series[float] = Field(ge=0, le=100)
    PAQ3: Series[float] = Field(ge=0, le=100)
    PAQ4: Series[float] = Field(ge=0, le=100)
    PAQ5: Series[float] = Field(ge=0, le=100)
    PAQ6: Series[float] = Field(ge=0, le=100)
    PAQ7: Series[float] = Field(ge=0, le=100)
    PAQ8: Series[float] = Field(ge=0, le=100)

    participant: Series[str]

    class Config:
        drop_invalid_rows = True
        strict = "filter"

    @pa.dataframe_parser
    def column_alias(cls, df: DataFrame) -> DataFrame:
        rename_dict = dict(zip(PAQ_LABELS, PAQ_IDS, strict=False))
        rename_dict.update(
            {
                "Participant": "participant",
            }
        )
        return df.rename(columns=rename_dict)


@dataclass
class ModelType:
    """A data class representing a circumplex model type with its properties."""

    name: CircModelE

    @property
    def equal_ang(self) -> bool:
        """Check if the model uses equal angles constraint."""
        return self.name in {CircModelE.EQUAL_ANG, CircModelE.EQUAL_COM}

    @property
    def equal_com(self) -> bool:
        """Check if the model uses equal communalities constraint."""
        return self.name in {CircModelE.EQUAL_COM, CircModelE.CIRCUMPLEX}


def length_1_array_to_number(v: np.ndarray | float | None) -> float | None:
    """Validate a length-1 numpy array to a float."""
    if v is None or isinstance(v, (float, int)):
        return v
    if isinstance(v, np.ndarray) and v.size == 1:
        return float(v.item())
    msg = "Value must be a numpy array with a single element."
    raise ValueError(msg)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CircE:
    """A data class to hold the results of a CircE model fitting."""

    _raw_bfgs_fit: ro.ListVector
    model_type: ModelType
    datasource: str
    language: str
    n: Annotated[int, BeforeValidator(length_1_array_to_number)]
    m: Annotated[int, BeforeValidator(length_1_array_to_number)]
    chisq: Annotated[float, BeforeValidator(length_1_array_to_number)]
    df: Annotated[int, BeforeValidator(length_1_array_to_number)]
    p: Annotated[float, BeforeValidator(length_1_array_to_number)]
    cfi: Annotated[float, BeforeValidator(length_1_array_to_number)]
    gfi: Annotated[float, BeforeValidator(length_1_array_to_number)]
    agfi: Annotated[float, BeforeValidator(length_1_array_to_number)]
    srmr: Annotated[float, BeforeValidator(length_1_array_to_number)]
    mcsc: Annotated[float, BeforeValidator(length_1_array_to_number)]
    rmsea: Annotated[float, BeforeValidator(length_1_array_to_number)]
    rmsea_l: Annotated[float, BeforeValidator(length_1_array_to_number)]
    rmsea_u: Annotated[float, BeforeValidator(length_1_array_to_number)]
    polar_angles: pd.DataFrame | None = None

    @classmethod
    def from_bfgs(
        cls,
        bfgs_model: ro.ListVector,
        datasource: str,
        language: str,
        model_type: ModelType,
        n: int,
    ) -> "CircE":
        """Create a CircE instance from a fitted BFGS model."""
        fit_stats = sspyr.extract_bfgs_fit(bfgs_model)
        polar_angles = None
        if model_type in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            polar_angles = pd.DataFrame(fit_stats.get("polar_angles", None)).T

        return cls(
            _raw_bfgs_fit=bfgs_model,
            model_type=model_type,
            datasource=datasource,
            language=language,
            n=n,
            m=fit_stats.get("m", None),
            chisq=fit_stats.get("chisq", None),
            df=fit_stats.get("dfnull", None),
            p=fit_stats.get("p", None),
            cfi=fit_stats.get("cfi", None),
            gfi=fit_stats.get("gfi", None),
            agfi=fit_stats.get("agfi", None),
            srmr=fit_stats.get("srmr", None),
            mcsc=fit_stats.get("mcsc", None),
            rmsea=fit_stats.get("rmsea", None),
            rmsea_l=fit_stats.get("rmsea.l", None),
            rmsea_u=fit_stats.get("rmsea.u", None),
            polar_angles=polar_angles,
        )

    @classmethod
    def compute_bfgs_fit(
        cls,
        data_cor: pd.DataFrame,
        datasource: str,
        language: str,
        circ_model: CircModelE,
    ) -> "CircE":
        """
        Compute and return a CircEResult from the given data correlation matrix.

        Examples
        --------
        >>> import soundscapy as sspy
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> circ_model = sspy.satp.CircModelE.CIRCUMPLEX
        >>> circe_res = sspy.satp.CircE.compute_bfgs_fit(
        ... data_cor, "ISD", "EN", circ_model)
        ...

        """
        n = data_cor.shape[0]
        model_type = ModelType(name=circ_model)
        bfgs_model = sspyr.bfgs(
            data_cor=data_cor,
            scales=PAQ_IDS,
            m_val=3,
            equal_ang=model_type.equal_ang,
            equal_com=model_type.equal_com,
        )
        return cls.from_bfgs(bfgs_model, datasource, language, model_type, n)


class SATP:
    def __init__(
        self,
        data: pd.DataFrame,
        language: str | None,
        datasource: str | None,
        *,
        ipsatize_data: bool = True,
    ):
        self._ipsatized = False
        self._raw_data = data
        self.data: DataFrame = SATPSchema.validate(data, lazy=True)

        if ipsatize_data:
            self.ipsatize()

        self.language = language
        self.datasource = datasource
        self.model_results: dict[CircModelE, CircE | None] = {
            CircModelE.UNCONSTRAINED: None,
            CircModelE.EQUAL_COM: None,
            CircModelE.EQUAL_ANG: None,
            CircModelE.CIRCUMPLEX: None,
        }
        self._errors: dict[CircModelE, Exception] = {}

    @property
    def data_corr(self) -> pd.DataFrame:
        return self.data.corr()

    def ipsatize(self) -> None:
        self.data = self._ipsatize_df(self.data, by="participant")
        self._ipsatized = True

    @staticmethod
    def _ipsatize_df(df: DataFrame, by: str = "participant") -> DataFrame:
        return df.groupby(by).transform(lambda x: x - x.mean())

    def run(self, circ_model: CircModelE | None = None) -> None:
        circ_models_to_run = [*CircModelE] if circ_model is None else [circ_model]
        for model in circ_models_to_run:
            try:
                self.model_results[model] = CircE.compute_bfgs_fit(
                    self.data_corr, self.datasource, self.language, model
                )
            except Exception as e:  # noqa: PERF203
                warnings.warn(f"{model.value} raised {e}", stacklevel=2)
                self._errors[model] = e
