import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri

from soundscapy.sspylogging import get_logger
from soundscapy.surveys.survey_utils import PAQ_IDS

from ._r_wrapper import get_r_session

logger = get_logger()

_, _, _stats_package, _base_package, circe = get_r_session()
logger.debug("R session and packages retrieved successfully.")


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
        >>> # doctest: +SKIP
        >>> import soundscapy as sspy
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> n = data_paqs.shape[0]
        >>> circ_model = ModelType(name=CircModelE.CIRCUMPLEX)
        >>> circe_res = sspy.spi.bfgs(
        ... data_cor=data_cor,
        ... scales=PAQ_IDS,
        ... m_val=3,
        ... equal_ang=circ_model.equal_ang,
        ... equal_com=circ_model.equal_com,
        ... )
        >>> fit_stats = sspy.r_wrapper.extract_bfgs_fit(circe_res)

    """
    with (ro.default_converter + pandas2ri.converter).context():
        py_res = {
            key.lower(): ro.conversion.get_conversion().rpy2py(val)
            for key, val in bfgs_model.items()
        }
    py_res["p"] = 1 - _stats_package.pchisq(py_res["chisq"], py_res["dfnull"]).item()

    return py_res


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
        equal_ang (bool, optional): Whether to enforce equal angles constraint.
            Defaults to True.
        equal_com (bool, optional): Whether to enforce equal communalities constraint.
            Defaults to True.

    Returns
    -------
        RS4: Fitted model object from the circe package.

    Examples
    --------
        >>> import soundscapy as sspy
        >>> from soundscapy.satp import CircModelE, ModelType
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> n = data_paqs.shape[0]
        >>> circ_model = ModelType(name=CircModelE.CIRCUMPLEX)
        >>> circe_res = bfgs(
        ... data_cor=data_cor,
        ... scales=PAQ_IDS,
        ... m_val=3,
        ... equal_ang=circ_model.equal_ang,
        ... equal_com=circ_model.equal_com,
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
