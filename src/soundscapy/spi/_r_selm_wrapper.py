import warnings
from typing import Literal

import rpy2.rinterface as ri
import rpy2.robjects as ro
from rpy2.robjects.methods import RS4
from rpy2.robjects.packages import importr

from soundscapy.spi._r_wrapper import get_r_session

_, sn, stats, _ = get_r_session()
utils = importr("utils")


class Rselm(RS4):
    """
    Reflection of the S4 class 'selm'

    Following: https://rpy2.github.io/doc/v3.5.x/html/robjects_oop.html
    """

    _coef = utils.getS3method("coef", "selm")
    _plot = utils.getS3method("plot", "selm")
    _residuals = utils.getS3method("residuals", "selm")
    _fitted = utils.getS3method("fitted", "selm")
    _weights = utils.getS3method("weights", "selm")
    _profile = utils.getS3method("profile", "selm")
    _confint = utils.getS3method("confint", "selm")
    _predict = utils.getS3method("predict", "selm")

    def _call_get(self):
        return self.do_slot("call")

    def _call_set(self, value):
        return self.do_slot("call", value)  # type: ignore

    call = property(_call_get, _call_set, None, "Get or set the RS4 slot 'call'.")

    @property
    def param(self):
        """
        Get the parameter of the fitted model.
        """
        return self.do_slot("param")

    def coef(self, param_type: Literal["CP", "DP"] = "CP") -> ro.FloatVector:
        """
        Coefficients of the fitted model.
        """
        if param_type not in ["CP", "DP"]:
            raise ValueError("param_type must be 'CP' or 'DP'.")
        return self._coef(self, param_type)

    def loglik(self) -> ro.FloatVector:
        """
        Log-likelihood of the fitted model.
        """
        return ro.r["logLik"](self)  # type: ignore

    def vcov(self, param_type: Literal["CP", "DP"] = "CP") -> ro.vectors.FloatMatrix:
        """
        Variance-covariance matrix of the fitted model.
        """
        if param_type not in ["CP", "DP"]:
            raise ValueError("param_type must be 'CP' or 'DP'.")
        return ro.r["vcov"](self, param_type)  # type: ignore

    def show(self):
        """
        Show the fitted model.
        """
        return ro.r["show"](self)  # type: ignore

    def summary(self):
        """
        Summary of the fitted model.
        """
        return ro.r["summary"](self)  # type: ignore

    def plot(self, *args, **kwargs):
        """
        Plot the fitted model.
        """
        return self._plot(self, *args, **kwargs)

    def residuals(self) -> ro.FloatVector:
        """
        Residuals of the fitted model.
        """
        return self._residuals(self)

    def fitted(self) -> ro.FloatVector:
        """
        Fitted values of the model.
        """
        return self._fitted(self)

    def weights(self) -> ro.FloatVector | ri.NULLType:
        """
        Weights of the fitted model.
        """
        return self._weights(self)

    def profile(
        self,
        param_type: Literal["CP", "DP"],
        param_name,
        param_values,
        npt,
        opt_control,
        plot_it,
        log,
        levels,
        trace,
    ):
        """
        Profile likelihood of the fitted model.

        Parameters
        ----------
        param_type : Literal["CP", "DP"]
            A character string with the required parameterization; it must be either 'CP' or 'DP'.
        param_name : str or tuple
            Either a single character string or a tuple of two such terms with the name(s)
            of the parameter(s) for which the profile log-likelihood is required.
        param_values
            In the one-parameter case, a numeric vector with the values where the log-likelihood
            must be evaluated; in the two-parameter case, a matrix with two columns
        """

        warnings.warn(
            "The profile method has not been completed or tested. "
            "Use at your own risk.",
            FutureWarning,
        )

        return self._profile(
            self,
            param_type,
            param_name,
            param_values,
            npt,
            opt_control,
            plot_it,
            log,
            levels,
            trace,
        )

    def confint(
        self, parm, level=0.95, param_type=ri.MissingArg, tol=1e-3
    ) -> ro.vectors.FloatMatrix:
        """
        Confidence intervals for the fitted model.
        """

        return self._confint(self, parm, level, param_type, tol)

    def predict(self, newdata=ri.MissingArg, param_type="CP") -> ro.FloatVector:
        """
        Predict values from the fitted model.
        """
        return self._predict(self, newdata=newdata, param_type=param_type)

    @classmethod
    def selm(
        cls,
        formula,
        family="SN",
        data=ri.MissingArg,
        weights=ri.MissingArg,
        subset=ri.MissingArg,
        na_action=ri.MissingArg,
        start=ro.NULL,
        fixed_param=ro.ListVector({}),
        method="MLE",
        penalty=ro.NULL,
        model: bool = True,
        x: bool = False,
        y: bool = False,
        contrasts=ro.NULL,
        offset=ri.MissingArg,
        **kwargs,
    ) -> "Rselm":
        """
        Create a Rselm object.

        Parameters
        ----------
        formula : str
            The formula for the model.
        family : str, optional
            The family of the model. Default is "SN".
        weights : optional
            The weights to be used for the model.
        method : str, optional
            The optimization method. Default is "BFGS".
        kwargs : keyword arguments
            Additional arguments to be passed to the model.

        Returns
        -------
        Rselm
            A Rselm object.
        """
        from rpy2.robjects.conversion import get_conversion

        converter = get_conversion()

        if family not in ["SN", "ST", "SC", "ESN"]:
            raise ValueError("family must be 'SN', 'ST', 'SC', or 'ESN'.")

        with converter.rclass_map_context(ri.SexpS4, {"selm": cls}):
            return sn.selm(
                formula,
                family=family,
                data=data,
                weights=weights,
                subset=subset,
                na_action=na_action,
                start=start,
                fixed_param=fixed_param,
                method=method,
                penalty=penalty,
                model=model,
                x=x,
                y=y,
                contrasts=contrasts,
                offset=offset,
                **kwargs,
            )
