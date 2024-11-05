"""
Optional dependency handling for soundscapy.

This module provides utilities for managing optional dependencies across the package.
It allows graceful handling of missing dependencies and provides helpful feedback
about which dependencies are missing and how to install them.

Examples
--------
Basic usage - importing optional dependencies:

>>> from soundscapy._optionals import require_dependencies
>>> # xdoctest: +SKIP
>>> required = require_dependencies("audio")  # Returns dict of imported modules
>>> mosqito = required["mosqito"]  # Access specific module

Error handling for missing dependencies:

>>> # xdoctest: +REQUIRES(env:AUDIO_DEPS=0)
>>> from soundscapy._optionals import require_dependencies
>>> try:
...     required = require_dependencies("audio")
... except ImportError as e:
...     print(str(e))  # doctest: +ELLIPSIS
audio analysis functionality requires additional dependencies. Install with: pip install soundscapy[audio]

Successful dependency loading:

>>> # xdoctest: +REQUIRES(env:AUDIO_DEPS==1)
>>> from soundscapy._optionals import require_dependencies
>>> required = require_dependencies("audio")
>>> isinstance(required, dict)
True
>>> 'mosqito' in required
True

Typical usage in a module:

>>> # xdoctest: +SKIP
>>> from soundscapy._optionals import require_dependencies
>>> # This will raise ImportError with helpful message if dependencies missing
>>> required = require_dependencies("audio")
>>> # Now import feature code that depends on the optional packages
>>> from .binaural import Binaural
>>> from .analysis_settings import AnalysisSettings

Notes
-----
The `require_dependencies()` function is the main interface for managing optional
dependencies. It performs these key functions:

1. Checks if all required packages for a feature group are available
2. Returns a dictionary of imported modules if successful
3. Raises an ImportError with installation instructions if dependencies are missing

The module uses OPTIONAL_DEPENDENCIES to define feature groups and their requirements:
    - packages: Tuple of required package names
    - install: pip install command/target
    - description: Human-readable feature description
"""

from typing import Dict, Any
import importlib

# Map module groups to their pip install targets
OPTIONAL_DEPENDENCIES = {
    "audio": {
        "packages": ("mosqito", "maad", "tqdm", "acoustics"),
        "install": "soundscapy[audio]",
        "description": "audio analysis functionality",
    },
    # Add other groups as needed
}
"""Dict[str, Dict]: Mapping of feature groups to their required dependencies.

Each group contains:
    modules (Tuple[str]): Required module names
    install (str): pip install command/target
    description (str): Human-readable feature description
"""


def format_import_error(group: str) -> str:
    """Create a helpful error message for missing dependencies

    Parameters
    ----------
    group : str
        Name of the dependency group

    Returns
    -------
    str
        Formatted error message with installation instructions
    """
    info = OPTIONAL_DEPENDENCIES[group]
    return (
        f"{info['description'].capitalize()} requires additional dependencies.\n"
        f" If desired, install with: pip install {info['install']}"
    )


def require_dependencies(group: str) -> Dict[str, Any]:
    """Import and return all packages required for a dependency group.

    Parameters
    ----------
    group : str
        The name ofthe dependency group to import

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping package names to imported modules

    Raises
    ------
    ImportError
        If any required package is not available
    KeyError
        If the group name is not recognized
    """
    if group not in OPTIONAL_DEPENDENCIES:
        raise KeyError(f"Unknown dependency group: {group}")

    packages = {}
    try:
        for package in OPTIONAL_DEPENDENCIES[group]["packages"]:
            packages[package] = importlib.import_module(package)
        return packages
    except ImportError as e:
        raise ImportError(format_import_error(group)) from e
