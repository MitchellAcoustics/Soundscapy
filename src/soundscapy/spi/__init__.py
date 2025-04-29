# ruff: noqa: E402
# ignore module level import order because we need to check dependencies first

# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import rpy2  # noqa: F401

except ImportError as e:
    raise ImportError(
        "SPI functionality requires additional dependencies. "
        "Install with: pip install soundscapy[spi]"
    ) from e

# Now we can import our modules that depend on the optional packages
from . import MSN
from .MSN import MultiSkewNorm, DirectParams, CentredParams, cp2dp, dp2cp

__all__ = [
    "MSN",
    "MultiSkewNorm",
    "DirectParams",
    "CentredParams",
    "cp2dp",
    "dp2cp",
]
