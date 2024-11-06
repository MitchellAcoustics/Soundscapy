"""
Optional dependency handling for soundscapy.

This module provides utilities for managing optional dependencies across the package.
It allows graceful handling of missing dependencies and provides helpful feedback
about which dependencies are missing and how to install them.
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

OPTIONAL_IMPORTS = {
    "Binaural": ("soundscapy.audio", "Binaural"),
    "AudioAnalysis": ("soundscapy.audio", "AudioAnalysis"),
    "AnalysisSettings": ("soundscapy.audio", "AnalysisSettings"),
    "ConfigManager": ("soundscapy.audio", "ConfigManager"),
    "process_all_metrics": ("soundscapy.audio", "process_all_metrics"),
    "prep_multiindex_df": ("soundscapy.audio", "prep_multiindex_df"),
    "add_results": ("soundscapy.audio", "add_results"),
    "parallel_process": ("soundscapy.audio", "parallel_process"),
}


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
        raise ImportError(
            f"{OPTIONAL_DEPENDENCIES[group]['description']} requires additional dependencies. "
            f"Install with: pip install {OPTIONAL_DEPENDENCIES[group]['install']}"
        ) from e


def import_optional(name: str) -> Any:
    """Import an optional component by name."""
    if name not in OPTIONAL_IMPORTS:
        raise AttributeError(f"module 'soundscapy' has no attribute '{name}'")

    module_name, attr_name = OPTIONAL_IMPORTS[name]
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except ImportError as e:
        group = "audio"  # Can be made dynamic if we add more groups
        raise ImportError(
            f"The {name} component requires {OPTIONAL_DEPENDENCIES[group]['description']}. "
            f"Install with: pip install {OPTIONAL_DEPENDENCIES[group]['install']}"
        ) from e
