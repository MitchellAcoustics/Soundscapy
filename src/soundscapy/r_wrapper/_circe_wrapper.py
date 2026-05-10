from typing import Any

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from scipy.stats import chi2 as scipy_chi2

from soundscapy.surveys.survey_utils import PAQ_IDS

from ._r_wrapper import get_r_session


def bfgs_fit(
    data_cor: pd.DataFrame,
    n: int,
    scales: list[str] = PAQ_IDS,
    m_val: int = 3,
    *,
    equal_ang: bool = True,
    equal_com: bool = True,
) -> dict[str, Any]:
    """
    Fit a circumplex model and return extracted statistics.

    Calls the embedded CircE BFGS implementation and converts the result to a
    Python dict with scalar normalisation and a scipy-computed p-value.

    Parameters
    ----------
    data_cor
        Correlation matrix of the data.
    n
        Number of observations used to compute ``data_cor``.
    scales
        List of scale names. Defaults to PAQ_IDS.
    m_val
        Number of dimensions. Defaults to 3.
    equal_ang
        Whether to enforce equal angles constraint.
    equal_com
        Whether to enforce equal communalities constraint.

    Returns
    -------
    :
        Dictionary of fit statistics.

    """
    r = get_r_session()

    with (ro.default_converter + pandas2ri.converter).context():
        # Only the Python→R conversion needs the pandas2ri context.
        # Calling as_matrix() inside the context would cause its R-matrix
        # return value to be auto-converted back to numpy by the active
        # converter, producing a numpy array instead of an R matrix.
        r_data_cor = ro.conversion.get_conversion().py2rpy(data_cor)

    r_cor_mat = r.base.as_matrix(r_data_cor)
    circe_bfgs = ro.globalenv["CircE.BFGS"]

    bfgs_model = circe_bfgs(
        r_cor_mat,
        v_names=ro.StrVector(scales),
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

    with (ro.default_converter + pandas2ri.converter).context():
        py_res = {
            key.lower(): ro.conversion.get_conversion().rpy2py(val)  # type: ignore[missing-attribute]
            for key, val in bfgs_model.items()
        }

    # Normalise length-1 numpy arrays to Python scalars.
    py_res = {
        k: (v.item() if isinstance(v, np.ndarray) and v.shape == (1,) else v)
        for k, v in py_res.items()
    }

    # rpy2 may deliver degree-of-freedom fields as numpy floats.
    for key in ("m", "d", "dfnull"):
        if key in py_res and py_res[key] is not None:
            py_res[key] = int(py_res[key])

    # Use scipy instead of R's pchisq to avoid py2rpy conversion issues.
    # Use model df ("d"), NOT null-model df ("dfnull") — they give wildly
    # different p-values and only "d" is correct here.
    _chisq, _d = py_res.get("chisq"), py_res.get("d")
    py_res["p"] = (
        float(scipy_chi2.sf(_chisq, _d))
        if _chisq is not None and _d is not None
        else None
    )

    return py_res
