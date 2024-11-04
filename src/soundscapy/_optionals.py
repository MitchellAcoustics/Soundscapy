"""
Optional dependency handling for soundscapy.

This module provides utilities for managing optional dependencies across the package.
It allows graceful handling of missing dependencies and provides helpful feedback
about which dependencies are missing and how to install them.

Example
-------
>>> from soundscapy._optionals import OptionalDependencyManager
>>> manager = OptionalDependencyManager.get_instance()
>>> has_audio = manager.check_module_group("audio")
>>> if has_audio:
...     from soundscapy.audio import Binaural

Notes
-----
This module is intended for internal use only.

The `module_exists` method attempts to import a specified module and handles the outcome in three
different ways based on the error parameter:

 * "ignore": Silently returns `None` if the module isn't found
 * "warn": Logs a warning message if the module is missing
 * "raise": Raises an `ImportError` if the module cannot be imported

The manager uses a dictionary called `MODULE_GROUPS` to organize related dependencies into logical groups.
For example, the "audio" group includes dependencies like "mosqito", "maad", "tqdm", and "acoustics".
Each group specifies the required modules, installation instructions, and a human-readable description.

The `check_module_group` method verifies if all modules in a specified group are available.
It returns a boolean indicating success or failure and can provide helpful feedback about missing dependencies,
including the exact pip command needed to install them.

The test file `test__optionals.py` contains comprehensive tests for both methods.

For end users, this system provides a clean way to handle optional features, as shown in the example
where audio functionality is only accessed if the required dependencies are present.
This prevents crashes due to missing dependencies and instead provides helpful feedback about
what's missing and how to install it.
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
        f"{info['description'].capitalize()} requires additional dependencies."
        f" Install with: pip install {info['install']}"
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

