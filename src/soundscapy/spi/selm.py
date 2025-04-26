# %%

from abc import ABC
from typing import Literal
import pandas as pd
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from soundscapy.spi._r_wrapper import (
    R2numpy,
    R2pandas,
    boolvector2npbool,
    pydata2R,
    conversion_rules,
)
from soundscapy.spi._r_selm_wrapper import Rselm
from soundscapy.spi.msn_params import CentredParams, DirectParams

from rpy2.robjects import Formula, ListVector, BoolVector
from rpy2.robjects.conversion import localconverter


MSNParams = CentredParams | DirectParams


@dataclass
class SELM:
    _rselm: Rselm
    _call: str
    _family: Literal["SN", "ST", "SC", "ESN"]
    _method: Literal["MLE", "MPLE"]
    _param_var: np.ndarray | pd.DataFrame | None = None
    _start: MSNParams | None = None

    def __repr__(self) -> str:
        return f"SELM(family={self._family}, method={self._method})"

    @property
    def _param(self) -> dict:
        """Returns the parameters of the model."""
        if self._rselm is None:
            raise ValueError("The model has not been fitted yet.")
        with localconverter(conversion_rules):
            _param = {}
            for k, v in self._rselm.param.items():
                if isinstance(v, BoolVector):
                    # Registering context converter for BoolVector isn't working
                    # Convert manually
                    v = boolvector2npbool(v)
                _param[str(k)] = v

        return _param

    @property
    def cp(self) -> CentredParams:
        """Returns the centred parameters of the model."""
        if self._rselm is None:
            raise ValueError("The model has not been fitted yet.")
        return

    @property
    def coef(self, param_type: Literal["CP", "DP"] = "CP") -> np.ndarray:
        """Returns the coefficients of the model."""
        if self._rselm is None:
            raise ValueError("The model has not been fitted yet.")

        if param_type not in ["CP", "DP"]:
            raise ValueError("param_type must be 'CP' or 'DP'.")
        return R2numpy(self._rselm.coef(param_type=param_type))

    @property
    def vcov(self, param_type: Literal["CP", "DP"] = "CP") -> np.ndarray:
        """
        Variance-covariance matrix of the fitted model.
        """
        if self._rselm is None:
            raise ValueError("The model has not been fitted yet.")

        if param_type not in ["CP", "DP"]:
            raise ValueError("param_type must be 'CP' or 'DP'.")
        return R2numpy(self._rselm.vcov(param_type))

    @classmethod
    def selm(
        cls,
        formula: str,
        family: Literal["SN", "ST", "SC", "ESN"] = "SN",
        data: pd.DataFrame | np.ndarray | None = None,
        weights: pd.Series | npt.NDArray[np.int_] | None = None,
        subset: pd.Index | np.ndarray | None = None,
        na_action: Literal["na.omit", "na.fail", "na.exclude", "na.pass"] = "na.omit",
        start: MSNParams | None = None,
        fixed_param: dict | None = None,
        method: Literal["MLE", "MPLE"] = "MLE",
        penalty: str | None = None,
        model: bool = True,
        x: bool = False,
        y: bool = False,
        contrasts: dict | None = None,
        offset: np.ndarray | None = None,
        *args,
        **kwargs,
    ):
        # Required R conversions
        r_formula = Formula(formula)
        r_data = pydata2R(data) if data is not None else None
        r_weights = pydata2R(weights) if weights is not None else None
        r_subset = pydata2R(subset) if subset is not None else None
        r_start = start.to_r() if start is not None else None
        r_fixed_param = ListVector(fixed_param) if fixed_param is not None else None
        r_contrasts = ListVector(contrasts) if contrasts is not None else None
        r_offset = pydata2R(offset) if offset is not None else None

        # Create the Rselm object
        with localconverter(conversion_rules):
            rselm = Rselm.selm(
                formula=r_formula,
                family=family,
                data=r_data,
                weights=r_weights,
                subset=r_subset,
                na_action=na_action,
                start=r_start,
                fixed_param=r_fixed_param,
                method=method,
                penalty=penalty,
                model=model,
                x=x,
                y=y,
                contrasts=r_contrasts,
                offset=r_offset,
            )

        return cls(
            _rselm=rselm,
            _call=rselm.slots["call"].r_repr(),
            _family=family,
            _method=method,
            _param_var=rselm.slots["param.var"],
        )


# %%

from soundscapy.spi._r_wrapper import get_r_session
from rpy2.robjects.packages import PackageData

_, sn, stats, _ = get_r_session()

ais = PackageData("sn").fetch("ais")["ais"]
f = "log(Fe) ~ BMI + LBM"
s = Rselm.selm(f, family="SN", data=ais)

selm = SELM.selm(f, family="SN", data=R2pandas(ais))
selm._param

# %%
