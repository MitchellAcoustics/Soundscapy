"""Module for wrapping R functionality with rpy2."""

# ruff: noqa: E402
from soundscapy._optional import require_deps

# Gates only the rpy2 *Python* package; R runtime issues (missing R, version
# mismatch) still surface lazily on first call into r_wrapper internals.
require_deps(["rpy2"], extra="r")

from ._r_wrapper import (  # noqa: F401
    bfgs_fit,
    cp2dp,
    dp2cp,
    extract_cp,
    extract_dp,
    sample_msn,
    sample_mtsn,
    selm,
)

# r_wrapper is an internal implementation package.  All user-facing names are
# re-exported from soundscapy.spi and soundscapy.satp.  Nothing is in __all__
# so that ``from soundscapy.r_wrapper import *`` imports nothing.
__all__: list[str] = []
