from enum import StrEnum
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BeforeValidator, ConfigDict
from pydantic.dataclasses import dataclass
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri

from soundscapy.spi._r_wrapper import get_r_session
from soundscapy.sspylogging import get_logger
from soundscapy.surveys.survey_utils import PAQ_IDS

logger = get_logger()

_, _, _stats_package, _base_package, circe = get_r_session()
logger.debug("R session and packages retrieved successfully.")


class CircModelE(StrEnum):
    """Enumeration of circumplex model types."""

    UNCONSTRAINED = "unconstrained"
    EQUAL_ANG = "equal_ang"
    EQUAL_COM = "equal_com"
    CIRCUMPLEX = "circumplex"


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
class CircEResult:
    """A data class to hold the results of a CircE model fitting."""

    _bfgs_model: ro.ListVector
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
    ) -> "CircEResult":
        """Create a CircEResult instance from a fitted BFGS model."""
        fit_stats = extract_bfgs_fit(bfgs_model)
        polar_angles = None
        if model_type in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            polar_angles = pd.DataFrame(fit_stats.get("polar_angles", None)).T

        return cls(
            _bfgs_model=bfgs_model,
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
        model_type: ModelType,
    ) -> "CircEResult":
        """
        Compute and return a CircEResult from the given data correlation matrix.

        Examples
        --------
        >>> # xdoctest: +SKIP
        >>> import soundscapy as sspy
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> model_type = ModelType(name=CircModelE.CIRCUMPLEX)
        >>> circe_res = sspy.spi.CircEResult.compute_bfgs_fit(
        ... data_cor, "ISD", "EN", model_type)

        """
        n = data_cor.shape[0]
        bfgs_model = bfgs(
            data_cor=data_cor,
            scales=PAQ_IDS,
            m_val=3,
            equal_ang=model_type.equal_ang,
            equal_com=model_type.equal_com,
        )
        return cls.from_bfgs(bfgs_model, datasource, language, model_type, n)


def extract_bfgs_fit(bfgs_model: ro.ListVector) -> dict:
    """
    Extract fit statistics from a fitted BFGS model object.

    Parameters
    ----------
        bfgs_model (RS4): Fitted model object from the circe package.

    Returns
    -------
        dict: Dictionary containing fit statistics.

    Examples
    --------
        >>> # xdoctest: +SKIP
        >>> import soundscapy as sspy
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> n = data_paqs.shape[0]
        >>> model_type = ModelType(name=CircModelE.CIRCUMPLEX)
        >>> circe_res = sspy.spi.bfgs(
        ... data_cor=data_cor,
        ... scales=PAQ_IDS,
        ... m_val=3,
        ... equal_ang=model_type.equal_ang,
        ... equal_com=model_type.equal_com,
        ... )
        >>> fit_stats = sspy.spi.extract_bfgs_fit(circe_res)

    """
    py_res = {}
    with (ro.default_converter + pandas2ri.converter).context():
        for i, name in enumerate(bfgs_model.names):
            val = ro.conversion.get_conversion().rpy2py(bfgs_model[i])
            print(name, val)
            py_res[name.lower()] = val

    py_res["p"] = 1 - _stats_package.pchisq(py_res["chisq"], py_res["dfnull"]).item()

    return py_res


def extract_bfgs_to_table(
    bfgs_model: ro.ListVector,
    datasource: str,
    language: str,
    model_type: ModelType,
    n: int,
    res_table: pd.DataFrame | None = None,
    incl_stats: list[str] = [
        "chisq",
        "df",
        "p",
        "cfi",
        "gfi",
        "agfi",
        "srmr",
        "mcsc",
        "rmsea",
        "rmsea.l",
        "rmsea.u",
    ],
    scales: list[str] = PAQ_IDS,
) -> pd.DataFrame:
    standard_cols = ["dataset", "language", "model_type", "n", "m"]
    cols = standard_cols + incl_stats

    if res_table is None:
        res_table = pd.DataFrame(columns=cols)

    full_stats = extract_bfgs_fit(bfgs_model)

    res_table.loc[model_type.name.value, "dataset"] = datasource
    res_table.loc[model_type.name.value, "language"] = language
    res_table.loc[model_type.name.value, "model_type"] = model_type.name.value
    res_table.loc[model_type.name.value, "n"] = n
    res_table.loc[model_type.name.value, "m"] = full_stats.get("m", None)

    for stat in incl_stats:
        res_table.loc[model_type.name.value, stat] = full_stats.get(stat, None)

    if model_type in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
        polar_angles = full_stats.get("polar_angles", None)
        if polar_angles is not None:
            polar_angles = polar_angles.T
            for i, est in enumerate(["", "_l", "_u"]):
                res_table.loc[
                    model_type.name.value, [scale + est for scale in scales]
                ] = polar_angles.iloc[i].to_numpy()

    return res_table


def bfgs(
    data_cor: pd.DataFrame,
    scales: list[str] = PAQ_IDS,
    m_val: int = 3,
    *,
    equal_ang: bool = True,
    equal_com: bool = True,
) -> ro.ListVector:
    """
    Fit a circumplex model using the BFGS algorithm from the circe package.

    Parameters
    ----------
        data_cor (pd.DataFrame): Correlation matrix of the data.
        scales (list[str], optional): List of scale names. Defaults to PAQ_IDS.
        m_val (int, optional): Number of dimensions. Defaults to 3.
        equal_ang (bool, optional): Whether to enforce equal angles constraint. Defaults to True.
        equal_com (bool, optional): Whether to enforce equal communalities constraint. Defaults to True.

    Returns
    -------
        RS4: Fitted model object from the circe package.

    Examples
    --------
        >>> import soundscapy as sspy
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> n = data_paqs.shape[0]
        >>> model_type = ModelType(name=CircModelE.CIRCUMPLEX)
        >>> circe_res = bfgs(
        ... data_cor=data_cor,
        ... scales=PAQ_IDS,
        ... m_val=3,
        ... equal_ang=model_type.equal_ang,
        ... equal_com=model_type.equal_com,
        ... )

    """
    n = data_cor.shape[0]

    with (ro.default_converter + pandas2ri.converter).context():
        r_data_cor = ro.conversion.get_conversion().py2rpy(data_cor)

    r_cor_mat = _base_package.as_matrix(r_data_cor)

    r_scales = ro.StrVector(scales)

    return circe.CircE_BFGS(
        r_cor_mat,
        v_names=r_scales,
        m=m_val,
        N=n,
        start_values="PFA",
        equal_ang=equal_ang,
        equal_com=equal_com,
        iterlim=1000,
        try_refit_BFGS=True,
    )
