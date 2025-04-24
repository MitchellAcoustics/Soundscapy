"""
R integration for skew-normal distribution calculations.

This module provides functions for:
1. Checking R and R package dependencies
2. Initializing and managing R sessions
3. Converting data between R and Python
4. Executing R functions for skew-normal calculations

It is not intended to be used directly by end users.
"""

from typing import Dict, Any, Tuple
import sys
import contextlib
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


def check_dependencies() -> Dict[str, Any]:
    """
    Check all required R dependencies for the SPI module.

    This function checks:
    1. R installation accessibility
    2. R version compatibility
    3. 'sn' package availability
    4. 'sn' package version compatibility

    Returns:
        Dict[str, Any]: Dictionary with dependency information.

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


def initialize_r_session() -> Dict[str, Any]:
    """
    Initialize an R session for skew-normal distribution calculations.

    This function:
    1. Checks for R and package dependencies
    2. Imports required R packages
    3. Sets up the R environment
    4. Updates global session state

    Returns:
        Dict[str, Any]: Session information including R and package versions

    Raises:
        ImportError: If dependencies are missing
        RuntimeError: If session initialization fails
    """
    global _r_session, _sn_package, _stats_package, _session_active

    # If session is already active, just return the state
    if _session_active:
        logger.debug("R session already initialized")
        return {
            "r_session": "active",
            "sn_package": "loaded",
            "stats_package": "loaded",
        }

    # First check all dependencies
    dep_info = check_dependencies()
    logger.debug(f"Dependencies verified: {dep_info}")

    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import numpy2ri

        # Activate numpy conversion
        numpy2ri.activate()
        logger.debug("Activated numpy-to-R conversion")

        # Import required packages
        _sn_package = rpackages.importr("sn")
        _stats_package = rpackages.importr("stats")
        logger.debug("Imported required R packages")

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
            **dep_info,
        }

    except Exception as e:
        logger.error(f"Failed to initialize R session: {str(e)}")
        _session_active = False
        _r_session = None
        _sn_package = None
        _stats_package = None
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
    global _r_session, _sn_package, _stats_package, _session_active

    if not _session_active:
        logger.debug("No active R session to shutdown")
        return True

    try:
        import rpy2.robjects.numpy2ri as numpy2ri
        import gc

        # Deactivate numpy conversion
        numpy2ri.deactivate()
        logger.debug("Deactivated numpy-to-R conversion")

        # Clear references to R objects
        _r_session = None
        _sn_package = None
        _stats_package = None

        # Update session state
        _session_active = False

        # Force garbage collection to release R resources
        gc.collect()
        logger.info("R session successfully shutdown")
        return True

    except Exception as e:
        logger.error(f"Error during R session shutdown: {str(e)}")
        return False


def get_r_session() -> Tuple[Any, Any, Any]:
    """
    Get the current R session and package objects.

    This function:
    1. Initializes the session if not already active
    2. Returns the session and package references

    Returns:
        Tuple[Any, Any, Any]: (r_session, sn_package, stats_package)

    Raises:
        RuntimeError: If session initialization fails
    """
    global _r_session, _sn_package, _stats_package, _session_active

    if not _session_active:
        logger.debug("R session not active, initializing")
        initialize_r_session()

    if not _session_active or not _r_session or not _sn_package or not _stats_package:
        raise RuntimeError("Failed to initialize R session")

    return _r_session, _sn_package, _stats_package


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
        Tuple[Any, Any, Any]: (r_session, sn_package, stats_package)

    Raises:
        RuntimeError: If session initialization fails
    """
    try:
        if not is_session_active():
            initialize_r_session()
        r_session, sn_package, stats_package = get_r_session()
        yield r_session, sn_package, stats_package
    except Exception as e:
        logger.error(f"Error in R session context: {str(e)}")
        raise


# === DATA CONVERSION ===

def extract_r_list_element(r_list: Any, name: str) -> Any:
    """
    Extract a named element from an R list or environment.
    
    This is a minimal helper function to simplify extracting elements from R lists
    when working with the skew-normal distribution functions that return complex
    result objects.
    
    Args:
        r_list: An R list or environment
        name: The name of the element to extract
        
    Returns:
        The extracted element, still as an R object
        
    Raises:
        KeyError: If the named element doesn't exist
        TypeError: If the object is not an R list or environment
    """
    r_session, _, _ = get_r_session()
    
    # Check if it's a list or environment
    is_list = r_session.r("is.list")(r_list)[0]
    is_env = r_session.r("is.environment")(r_list)[0]
    
    if not (is_list or is_env):
        raise TypeError(
            "Input must be an R list or environment, got "
            f"{r_session.r('class')(r_list)[0]}"
        )
    
    # Check if element exists
    names = list(r_session.r("names")(r_list))
    if name not in names:
        raise KeyError(f"Element '{name}' not found in R object. Available names: {names}")
    
    # Extract the element using the r_list.rx2() extraction method
    element = r_list.rx2(name)
    return element


# === CONVERSION PATTERNS ===

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
