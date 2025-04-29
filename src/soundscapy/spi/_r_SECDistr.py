# %%
from importlib import metadata
import warnings
from typing import Literal

import rpy2.rinterface as ri
import rpy2.robjects as ro
from rpy2.robjects.methods import RS4, RS4Auto_Type
from rpy2.robjects.packages import importr

from soundscapy.spi._r_wrapper import get_r_session
from soundscapy.spi.msn_params import DirectParams

import numpy as np

_, sn, stats, _ = get_r_session()
utils = importr("utils")


class RSECdistrUv(RS4):
    __rname__ = "SECdistrUv"
    __rpackage__ = "sn"

    def __init__(self, family: str, dp: , name: str):


# %%

dp = DirectParams(xi=3, omega=5, alpha=-np.pi)
dp_r = dp.to_r()

sec = RSECdistrUv(family="SN", dp=dp_r)

# %%
