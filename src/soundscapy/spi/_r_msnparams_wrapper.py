# %%

import rpy2.rlike.container as rlc
from rpy2.robjects.conversion import localconverter
from rpy2 import robjects

from soundscapy.spi._r_wrapper import get_r_session, conversion_rules

_, sn, stats, _ = get_r_session()


class RmsnParams(object):
    """
    R wrapper for the msnParams class in the sn package.
    """

    __rname__ = "msnParams"
    __rpackage__ = "sn"

    def __init__(self, _pyrobj):
        # super().__init__(_pyrobj)
        self._pyrobj = _pyrobj

        if all(param in self._pyrobj for param in ["beta", "Omega", "alpha"]):
            self._param_type = "dp"
        elif all(param in self._pyrobj for param in ["mean", "sigma", "skew"]):
            self._param_type = "cp"
        else:
            raise ValueError(f"Invalid parameters. Contains: {self._pyrobj.keys()}")

        self._dim = len(self._pyrobj["beta"])

    def rpy2py(self):
        """
        Convert the RmsnParams object to a Python dictionary.
        """
        return {key: self._pyrobj[key] for key in self._pyrobj.keys()}

    def py2rpy(self):
        """
        Convert the RmsnParams object to an R-compatible format.
        """
        with localconverter(conversion_rules):
            return robjects.ListVector(self.rpy2py())

    def cp2dp(self):
        """
        Convert the RmsnParams object from centered parameters to direct parameters.
        """
        if self._param_type == "dp":
            raise ValueError("Parameters are already in direct form.")
        elif self._param_type == "cp":
            _probj = robjects.r(sn.cp2dp(self._pyrobj))
            return RmsnParams(_probj)

        else:
            raise ValueError("Invalid parameter type.")

    def dp2cp(self):
        """
        Convert the RmsnParams object from direct parameters to centered parameters.
        """
        if self._param_type == "cp":
            raise ValueError("Parameters are already in centered form.")
        elif self._param_type == "dp":
            _probj = robjects.r(sn.dp2cp(self._pyrobj))
            return RmsnParams(_probj)

        else:
            raise ValueError("Invalid parameter type.")
