from typing import Any

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from scipy.stats import chi2 as scipy_chi2

from soundscapy.sspylogging import get_logger
from soundscapy.surveys.survey_utils import PAQ_IDS

from ._r_wrapper import get_r_session

logger = get_logger()


def extract_bfgs_fit(bfgs_model: ro.ListVector) -> dict[str, Any]:
    """
    Extract fit statistics from a fitted BFGS model object.

    Parameters
    ----------
    bfgs_model
        Fitted model object from the embedded CircE R scripts.

    Returns
    -------
    :
        Dictionary containing fit statistics.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> import soundscapy as sspy
    >>> from soundscapy.satp import CircModelE
    >>> data = sspy.isd.load()
    >>> data_paqs = data[PAQ_IDS]
    >>> data_paqs = data_paqs.dropna()
    >>> data_cor = data_paqs.corr()
    >>> n = len(data_paqs)
    >>> circ_model = CircModelE.CIRCUMPLEX
    >>> circe_res = sspyr.bfgs(
    ...     data_cor=data_cor,
    ...     n=n,
    ...     scales=PAQ_IDS,
    ...     m_val=3,
    ...     equal_ang=circ_model.equal_ang,
    ...     equal_com=circ_model.equal_com,
    ... )
    >>> fit_stats = sspy.r_wrapper.extract_bfgs_fit(circe_res)

    """
    # Session must already be active (bfgs_model was produced by bfgs()), but
    # calling get_r_session() here ensures a clean error if somehow called in
    # isolation.
    get_r_session()
    with (ro.default_converter + pandas2ri.converter).context():
        py_res = {
            key.lower(): ro.conversion.get_conversion().rpy2py(val)  # type: ignore[missing-attribute]
            for key, val in bfgs_model.items()
        }

    # Normalize all length-1 numpy arrays to Python scalars so callers
    # never need to call .item() themselves.  Vectors/matrices are kept
    # as-is.  This also avoids DeprecationWarning from numpy >= 1.25 when
    # float() or int() is applied to an ndarray with ndim > 0.
    py_res = {
        k: (v.item() if isinstance(v, np.ndarray) and v.shape == (1,) else v)
        for k, v in py_res.items()
    }

    # Guarantee integer types for degree-of-freedom stats.  rpy2 may deliver
    # these as numpy floats if the R object is stored as numeric rather than
    # integer; explicit int() casts here ensures the annotation holds.
    for _int_key in ("m", "d", "dfnull"):
        if _int_key in py_res and py_res[_int_key] is not None:
            py_res[_int_key] = int(py_res[_int_key])

    # Use scipy instead of R's pchisq to avoid py2rpy conversion of pandas
    # Series objects produced by the pandas2ri context above.
    # scipy.chi2.sf(x, df) == 1 - pchisq(x, df) by definition.
    # Use the model's own degrees of freedom ("d"), NOT the null-model df
    # ("dfnull" = k*(k-1)/2).  Using dfnull gives a wildly wrong p-value.
    _chisq = py_res.get("chisq")
    _d = py_res.get("d")
    py_res["p"] = (
        float(scipy_chi2.sf(_chisq, _d))
        if _chisq is not None and _d is not None
        else None
    )

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
    Fit a circumplex model using the embedded CircE BFGS implementation.

    Parameters
    ----------
    data_cor
        Correlation matrix of the data.
    n
        Number of observations (participants) used to compute the correlation
        matrix. Used by CircE_BFGS for chi-square and RMSEA calculations.
    scales
        List of scale names. Defaults to PAQ_IDS.
    m_val
        Number of dimensions. Defaults to 3.
    equal_ang
        Whether to enforce equal angles constraint. Defaults to True.
    equal_com
        Whether to enforce equal communalities constraint. Defaults to True.

    Returns
    -------
    :
        Fitted model object from the embedded CircE scripts.

    Examples
    --------
    >>> import soundscapy as sspy
    >>> from soundscapy.satp import CircModelE
    >>> data = sspy.isd.load()
    >>> data_paqs = data[PAQ_IDS]
    >>> data_paqs = data_paqs.dropna()
    >>> data_cor = data_paqs.corr()
    >>> n = data_paqs.shape[0]
    >>> circ_model = CircModelE.CIRCUMPLEX
    >>> circe_res = bfgs(
    ...     data_cor=data_cor,
    ...     n=n,
    ...     scales=PAQ_IDS,
    ...     m_val=3,
    ...     equal_ang=circ_model.equal_ang,
    ...     equal_com=circ_model.equal_com,
    ... )

    """
    r = get_r_session()
    with (ro.default_converter + pandas2ri.converter).context():
        # Only the Python→R conversion needs the pandas2ri context.
        # Calling as_matrix() inside the context would cause its R-matrix
        # return value to be auto-converted back to numpy by the active
        # converter, producing a numpy array instead of an R matrix.
        r_data_cor = ro.conversion.get_conversion().py2rpy(data_cor)

    r_cor_mat = r.base.as_matrix(r_data_cor)
    r_scales = ro.StrVector(scales)
    circe_bfgs = ro.globalenv["CircE.BFGS"]

    return circe_bfgs(
        r_cor_mat,
        v_names=r_scales,
        m=m_val,
        N=n,
        start_values="PFA",
        equal_ang=equal_ang,
        equal_com=equal_com,
        iterlim=1000,
        try_refit_BFGS=True,
        print_level=0,
        file=ro.NULL,
    )
