"""Module for wrapping R functionality with rpy2."""

# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import rpy2  # noqa: F401

except ImportError as e:
    msg = (
        "R functionality requires additional dependencies. "
        "Install with: pip install soundscapy[satp]"
    )
    raise ImportError(msg) from e

# Now we can import our modules that depend on the optional packages
from ._circe_wrapper import bfgs, extract_bfgs_fit
from ._r_wrapper import PKG_SRC, get_r_session
from ._rsn_wrapper import (
    cp2dp,
    dp2cp,
    extract_cp,
    extract_dp,
    sample_msn,
    sample_mtsn,
    selm,
)

__all__ = [
    "PKG_SRC",
    "bfgs",
    "cp2dp",
    "dp2cp",
    "extract_bfgs_fit",
    "extract_cp",
    "extract_dp",
    "get_r_session",
    "sample_msn",
    "sample_mtsn",
    "selm",
]
