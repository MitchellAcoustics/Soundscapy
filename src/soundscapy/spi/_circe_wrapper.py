from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.methods import RS4

from soundscapy.spi._r_wrapper import get_r_session
from soundscapy.sspylogging import get_logger
from soundscapy.surveys.survey_utils import PAQ_IDS

logger = get_logger()

_, _, _, _base_package, circe = get_r_session()
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


def bfgs(
    data_cor: pd.DataFrame,
    scales: list[str] = PAQ_IDS,
    m_val: int = 3,
    *,
    equal_ang: bool = True,
    equal_com: bool = True,
) -> RS4:
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
