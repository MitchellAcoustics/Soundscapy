"""
R integration for skew-normal distribution calculations.

This module provides functions for:
1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

It is not intended to be used directly by end users.
"""

from typing import Any
import sys
import contextlib
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri, default_converter
from rpy2.robjects.vectors import BoolVector
from rpy2.rinterface_lib.sexp import NULLType
import numpy as np
import numpy.typing as npt
import pandas as pd

# These are used in the docstring examples but not in the code
# They will be used by code that imports and uses this module
from soundscapy.logging import get_logger

logger = get_logger()

# Cached values to avoid repeated checks
_r_checked = False
_sn_checked = False

# Session state
_r_session = None
_sn_package = None
_stats_package = None
_base_package = None
_session_active = False


def check_r_availability() -> None:
    """
    Check if R is installed and accessible through rpy2.

    Raises:
        ImportError: If R is not installed or cannot be accessed.
    """
    global _r_checked

    if _r_checked:
        return

    try:
        import rpy2.robjects as robjects

        # Basic check to ensure R is running by getting R version
        r_version = robjects.r("R.version.string")[0]  # type: ignore
        logger.debug(f"R version: {r_version}")

        # Check if minimum R version requirements are met
        # The 'sn' package requires R >= 3.6.0
        r_version_num = robjects.r(
            "as.numeric(R.version$major) + as.numeric(R.version$minor)/10"
        )[0]  # type: ignore
        if r_version_num < 3.6:
            raise ImportError(
                f"R version {r_version_num} is too old. The 'sn' package requires R >= 3.6.0. "
                "Please upgrade your R installation."
            )

        _r_checked = True
    except ImportError:
        raise ImportError(
            "rpy2 is installed but it cannot find an R installation. "
            "Please ensure R is installed and correctly configured. "
            "On Linux: Install R with your package manager (e.g., apt-get install r-base). "
            "On macOS: Install R from CRAN (https://cran.r-project.org/bin/macosx/). "
            "On Windows: Install R from CRAN (https://cran.r-project.org/bin/windows/base/)."
        )
    except Exception as e:
        raise ImportError(
            f"Error accessing R installation: {str(e)}. "
            "Please ensure R is installed and correctly configured."
        )


def check_sn_package() -> None:
    """
    Check if the R 'sn' package is installed.

    Raises:
        ImportError: If the 'sn' package is not installed.
    """
    global _sn_checked

    if _sn_checked:
        return

    # First ensure R is available
    check_r_availability()

    try:
        import rpy2.robjects.packages as rpackages

        # Check if 'sn' package is installed
        try:
            # Just importing to verify it exists
            _ = rpackages.importr("sn")

            # Get package version using R to verify compatibility
            import rpy2.robjects as robjects

            # Use R code to get the package version
            version = robjects.r('as.character(packageVersion("sn"))')[0]  # type: ignore
            logger.debug(f"R 'sn' package version: {version}")

            # Check if package version meets requirements
            # The SPI implementation requires 'sn' >= 2.0.0
            if version < "2.0.0":
                raise ImportError(
                    f"R 'sn' package version {version} is too old. "
                    "The SPI feature requires 'sn' >= 2.0.0. "
                    "Please upgrade the package by running in R: install.packages('sn')"
                )

            _sn_checked = True
        except rpackages.PackageNotInstalledError:
            raise ImportError(
                "R package 'sn' is not installed. "
                "Please install it by running in R: install.packages('sn')"
            )
    except Exception as e:
        if "sn" in str(e):
            # Already a more specific error about the sn package
            raise
        else:
            raise ImportError(
                f"Error checking for R 'sn' package: {str(e)}. "
                "Please ensure the package is installed by running in R: install.packages('sn')"
            )


def check_dependencies() -> dict[str, Any]:
    """
    Check all required R dependencies for the SPI module.

    This function checks:
    1. R installation accessibility
    2. R version compatibility
    3. 'sn' package availability
    4. 'sn' package version compatibility

    Returns:
        dict[str, Any]: Dictionary with dependency information.

    Raises:
        ImportError: If any dependency check fails.
    """
    # Check R availability first
    check_r_availability()

    # Then check for the sn package
    check_sn_package()

    # If we get here, all dependencies are available
    import rpy2.robjects as robjects

    # Return information about the dependencies
    return {
        "rpy2_version": sys.modules["rpy2"].__version__,
        "r_version": robjects.r("R.version.string")[0],  # type: ignore
        "sn_version": robjects.r('as.character(packageVersion("sn"))')[0],  # type: ignore
    }


# === SESSION MANAGEMENT ===


def initialize_r_session() -> dict[str, Any]:
    """
    Initialize an R session for skew-normal distribution calculations.

    This function:
    1. Checks for R and package dependencies
    2. Imports required R packages
    3. Sets up the R environment
    4. Updates global session state

    Returns:
        dict[str, Any]: Session information including R and package versions

    Raises:
        ImportError: If dependencies are missing
        RuntimeError: If session initialization fails
    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active

    # If session is already active, just return the state
    if _session_active:
        logger.debug("R session already initialized")
        return {
            "r_session": "active",
            "sn_package": "loaded",
            "stats_package": "loaded",
            "base_package": "loaded",
        }

    # First check all dependencies
    dep_info = check_dependencies()
    logger.debug(f"Dependencies verified: {dep_info}")

    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages

        # Import required packages
        _sn_package = rpackages.importr("sn")
        _stats_package = rpackages.importr("stats")
        _base_package = rpackages.importr("base")
        logger.debug("Imported R packages: sn, stats, base")

        # Set R random seed for reproducibility
        robjects.r("set.seed(42)")

        # Store R session
        _r_session = robjects

        # Update session state
        _session_active = True
        logger.info("R session successfully initialized")

        return {
            "r_session": "active",
            "sn_package": str(_sn_package),
            "stats_package": str(_stats_package),
            "base_package": str(_base_package),
            **dep_info,
        }

    except Exception as e:
        logger.error(f"Failed to initialize R session: {str(e)}")
        _session_active = False
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None
        raise RuntimeError(f"Failed to initialize R session: {str(e)}")


def shutdown_r_session() -> bool:
    """
    Shutdown the R session and clean up resources.

    This function:
    1. Deactivates numpy conversion
    2. Resets global session state
    3. Performs garbage collection

    Returns:
        bool: True if successful, False otherwise
    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active

    if not _session_active:
        logger.debug("No active R session to shutdown")
        return True

    try:
        import gc

        # Clear references to R objects
        _r_session = None
        _sn_package = None
        _stats_package = None
        _base_package = None

        # Update session state
        _session_active = False

        # Force garbage collection to release R resources
        gc.collect()
        logger.info("R session successfully shutdown")
        return True

    except Exception as e:
        logger.error(f"Error during R session shutdown: {str(e)}")
        return False


def get_r_session() -> tuple[Any, Any, Any, Any]:
    """
    Get the current R session and package objects.

    This function:
    1. Initializes the session if not already active
    2. Returns the session and package references

    Returns:
        tuple[Any, Any, Any, Any]: (r_session, sn_package, stats_package, base_package)

    Raises:
        RuntimeError: If session initialization fails
    """
    global _r_session, _sn_package, _stats_package, _base_package, _session_active

    if not _session_active:
        logger.debug("R session not active, initializing")
        initialize_r_session()

    if (
        not _session_active
        or not _r_session
        or not _sn_package
        or not _stats_package
        or not _base_package
    ):
        raise RuntimeError("Failed to initialize R session")

    return _r_session, _sn_package, _stats_package, _base_package


def is_session_active() -> bool:
    """
    Check if the R session is currently active.

    Returns:
        bool: True if the session is active, False otherwise
    """
    global _session_active
    return _session_active


@contextlib.contextmanager
def r_session_context():
    """
    Context manager for R session management.

    This context manager:
    1. Initializes the R session on entry
    2. Yields the session and package objects
    3. Does NOT shutdown the session on exit to allow reuse

    Yields:
        tuple[Any, Any, Any, Any, Any]: (r_session, sn_package, stats_package, base_package, converter)

    Raises:
        RuntimeError: If session initialization fails
    """
    try:
        if not is_session_active():
            initialize_r_session()

        r_session, sn_package, stats_package, base_package = get_r_session()

        logger.debug("Entering R session context")

        yield r_session, sn_package, stats_package, base_package
    except Exception as e:
        logger.error(f"Error in R session context: {str(e)}")
        raise


# === CONVERSION PATTERNS ===

null_converter = robjects.conversion.Converter("null_converter")
boolvector_converter = robjects.conversion.Converter("boolvector_converter")


@null_converter.py2rpy.register(type(None))
def none2null(none_obj):
    """
    Convert None to R NULL.

    This function is registered with the rpy2 converter to handle
    conversion of None values to R NULL.

    Parameters:
        none_obj (NoneType): The None object to convert.

    Returns:
        robjects.NULL: The converted R NULL object.
    """
    return robjects.r("NULL")


@null_converter.rpy2py.register(NULLType)
def null2none(r_null_obj):
    """
    Convert R NULL to Python None.

    This function is registered with the rpy2 converter to handle
    conversion of R NULL values to Python None.

    Parameters:
        r_obj (robjects.NULL): The R NULL object to convert.

    Returns:
        NoneType: The converted Python None object.
    """
    return None


def npbool2boolvector(py_bool_array: npt.NDArray[np.bool_]):
    """
    Convert numpy array of Python bool to R BoolVector.

    This function is registered with the rpy2 converter to handle
    conversion of numpy arrays of Python bool to R BoolVector.

    Parameters:
        py_bool_array (np.ndarray): The numpy array of Python bool to convert.

    Returns:
        BoolVector: The converted R BoolVector object.
    """
    res = BoolVector(py_bool_array)
    return res


@boolvector_converter.rpy2py.register(BoolVector)
def boolvector2npbool(r_bool_vector):
    """
    Convert R BoolVector to numpy array of Python bool.

    This function is registered with the rpy2 converter to handle
    conversion of R BoolVector objects to numpy array of Python bool.

    Parameters:
        r_bool_vector (BoolVector): The R BoolVector object to convert.

    Returns:
        np.ndarray: The converted numpy array of Python bool values.
    """
    res = np.array(r_bool_vector, dtype=bool)
    return res


conversion_rules = (
    default_converter
    + null_converter
    # + boolvector_converter # These aren't working registered yet. Use the functions directly
    + numpy2ri.converter
    + pandas2ri.converter
)


def numpy2R(arr):
    """Local conversion of R array to numpy as recommended by rpy2"""
    with localconverter(robjects.default_converter + numpy2ri.converter):
        data = robjects.conversion.get_conversion().rpy2py(arr)
    return data


def R2numpy(rarr):
    """Local conversion of R array to numpy as recommended by rpy2"""
    return np.asarray(rarr)


def pandas2R(df):
    """Local conversion of pandas dataframe to R dataframe as recommended by rpy2"""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        data = robjects.conversion.get_conversion().py2rpy(df)
    return data


def R2pandas(rdf):
    """Local conversion of R dataframe to pandas as recommended by rpy2"""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        data = robjects.conversion.get_conversion().rpy2py(rdf)
    return data


def pydata2R(data):
    """
    Convert Python data to R object.

    This function handles both pandas DataFrames and numpy arrays.

    Parameters:
        data (pd.DataFrame | np.ndarray): The data to convert.

    Returns:
        robjects.DataFrame | robjects.NumpyArray: The converted R object.
    """
    if isinstance(data, pd.DataFrame):
        return pandas2R(data)
    elif isinstance(data, np.ndarray):
        return numpy2R(data)
    else:
        raise ValueError("data must be a pandas DataFrame or numpy array.")


"""
Direct Conversion Patterns for R/Python Data Exchange
----------------------------------------------------

This module uses rpy2's built-in conversion mechanism rather than implementing
custom converters. Below are the recommended patterns to follow when working
with R/Python data conversion in the SPI module:

1. Basic automatic conversion:
   - Use the context manager pattern to enable automatic conversion within a scope:
   
   ```python
   from rpy2.robjects import numpy2ri, pandas2ri, default_converter
   
   # Create a converter with support for both numpy and pandas
   converter = default_converter + numpy2ri.converter + pandas2ri.converter
   
   # Use the converter within a context (preferred over global activation)
   with converter.context():
       # Python → R conversion
       r_matrix = r_session.r.matrix(numpy_array, nrow=rows, ncol=cols)
       
       # R → Python conversion 
       numpy_array = np.array(r_matrix)
       pandas_df = pd.DataFrame(r_dataframe)
   ```
   
2. Explicit conversion with get_conversion():
   ```python
   # Current recommended approach (as of rpy2 3.5+)
   from rpy2.robjects.conversion import get_conversion
   
   with converter.context():
       conversion = get_conversion()
       r_obj = conversion.py2rpy(py_obj)  # Python to R
       py_obj = conversion.rpy2py(r_obj)  # R to Python
   ```
   
   Note: The older approach using direct `py2rpy` and `rpy2py` functions is deprecated:
   ```python
   # Deprecated approach (still works but generates warnings)
   from rpy2.robjects.conversion import py2rpy, rpy2py
   
   with converter.context():
       r_obj = py2rpy(py_obj)  # Python to R
       py_obj = rpy2py(r_obj)  # R to Python
   ```

3. Working with R lists and extracting components:
   ```python
   # Extract a component from an R list result
   dp_result = r_sn.msn_mle(...)  # Returns a complex R list
   
   # Extract the 'dp' element:
   dp = dp_result.rx2('dp')
   
   # Or use the helper function:
   dp = extract_r_list_element(dp_result, 'dp')
   ```

4. Setting matrix column names:
   ```python
   with converter.context():
       r_matrix = r_session.r.matrix(data_array, nrow=rows, ncol=cols)
       # Using R code directly with string formatting
       r_session.r('colnames({}) <- c({})'.format(
           r_session.rinterface.deparse_str(r_matrix),
           ', '.join([f'"{name}"' for name in colnames])))
   ```
"""
