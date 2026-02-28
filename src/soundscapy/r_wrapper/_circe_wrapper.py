import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri

from soundscapy.sspylogging import get_logger
from soundscapy.surveys.survey_utils import PAQ_IDS

from ._r_wrapper import get_r_session

logger = get_logger()


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
        >>> n = len(data_paqs)
        >>> circ_model = ModelType(name=CircModelE.CIRCUMPLEX)
        >>> circe_res = sspy.spi.bfgs(
        ... data_cor=data_cor,
        ... n=n,
        ... scales=PAQ_IDS,
        ... m_val=3,
        ... equal_ang=circ_model.equal_ang,
        ... equal_com=circ_model.equal_com,
        ... )
        >>> fit_stats = sspy.r_wrapper.extract_bfgs_fit(circe_res)

    """
    _, _, stats_package, _, _ = get_r_session()
    with (ro.default_converter + pandas2ri.converter).context():
        py_res = {
            key.lower(): ro.conversion.get_conversion().rpy2py(val)
            for key, val in bfgs_model.items()
        }
    py_res["p"] = 1 - stats_package.pchisq(py_res["chisq"], py_res["dfnull"]).item()

    return py_res


def bfgs(
    data_cor: pd.DataFrame,
    n: int,
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
        n (int): Number of observations (participants) used to compute the correlation
            matrix. Used by CircE_BFGS for chi-square and RMSEA calculations.
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
        ... n=n,
        ... scales=PAQ_IDS,
        ... m_val=3,
        ... equal_ang=circ_model.equal_ang,
        ... equal_com=circ_model.equal_com,
        ... )

    """
    _, _, _, base_package, circe_package = get_r_session()
    with (ro.default_converter + pandas2ri.converter).context():
        r_data_cor = ro.conversion.get_conversion().py2rpy(data_cor)

    r_cor_mat = base_package.as_matrix(r_data_cor)

    r_scales = ro.StrVector(scales)

    return circe_package.CircE_BFGS(
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
