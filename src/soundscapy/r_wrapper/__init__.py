"""Module for wrapping R functionality with rpy2."""

# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import rpy2  # noqa: F401

except ImportError as e:
    msg = (
        "R functionality requires additional dependencies. "
        "Install with: pip install soundscapy[r]"
    )
    raise ImportError(msg) from e

# Now we can import our modules that depend on the optional packages
from ._circe_wrapper import bfgs, extract_bfgs_fit  # noqa: F401
from ._r_wrapper import get_r_session  # noqa: F401
from ._rsn_wrapper import (  # noqa: F401
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
