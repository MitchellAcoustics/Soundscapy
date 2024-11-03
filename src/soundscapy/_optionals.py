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

The `OptionalDependencyManager` class serves as the core component for handling optional package dependencies.
It maintains a cache of checked modules to avoid repeated import attempts and provides two main methods:
`module_exists` and `check_module_group`.

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

from typing import Optional
import importlib
import types
from loguru import logger

# Map module groups to their pip install targets
MODULE_GROUPS = {
    "audio": {
        "modules": ("mosqito", "maad", "tqdm", "acoustics"),
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


class OptionalDependencyManager:
    """Singleton manager for optional dependencies.

    Uses singleton pattern to ensure consistent caching across the application,
    preventing redundant import attempts and maintaining a single source of truth
    for dependency availability.
    """

    _instance = None

    def __new__(cls):
        # Singleton ensures consistent caching across all package modules
        if cls._instance is None:
            logger.debug("Initializing new OptionalDependencyManager singleton")
            cls._instance = super().__new__(cls)
            # Initialize empty cache - stores both successful and failed imports
            # to avoid repeated import attempts during runtime
            cls._instance._checked_modules = {}
        else:
            logger.debug("Reusing existing OptionalDependencyManager instance")
        return cls._instance

    @classmethod
    def get_instance(cls):
        # Factory method pattern provides cleaner API than direct instantiation
        return cls() if cls._instance is None else cls._instance

    def module_exists(
        self, name: str, error: str = "ignore"
    ) -> Optional[types.ModuleType]:
        """Try to import an optional dependency.

        Uses caching to avoid repeated import attempts, which is especially
        important for missing modules as import attempts are expensive.
        The three error modes support different use cases:
        - ignore: For quiet runtime checks (e.g. feature detection)
        - warn: For informing users about missing optional features
        - raise: For hard dependencies within optional feature groups
        """
        assert error in {"raise", "warn", "ignore"}
        logger.debug(f"Checking for module {name} (error={error})")

        # Check cache first to avoid repeated import attempts
        if name in self._checked_modules:
            logger.debug(f"Using cached result for {name}")
            module = self._checked_modules[name]
            if module is None:
                # Handle cached negative result according to error mode
                logger.debug(f"Cache indicates {name} was not available")
                if error == "raise":
                    raise ImportError(f"Required dependency {name} not found")
                elif error == "warn":
                    logger.warning(f"Missing optional dependency: {name}")
            return module

        try:
            # First-time import attempt
            logger.debug(f"Attempting first-time import of {name}")
            module = importlib.import_module(name)
            # Cache successful import
            self._checked_modules[name] = module
            logger.debug(f"Successfully imported and cached {name}")
            return module
        except ImportError:
            # Cache failed import to avoid future attempts
            logger.debug(f"Import failed for {name}, caching negative result")
            self._checked_modules[name] = None
            if error == "warn":
                logger.warning(f"Missing optional dependency: {name}")
            elif error == "raise":
                logger.error(f"Required dependency {name} not found")
                raise ImportError(f"Required dependency {name} not found")
            return None

    def check_module_group(self, group: str, error: str = "warn") -> bool:
        """Check if all modules in a group are available.

        Groups dependencies logically to support feature-based dependency checking.
        Uses 'ignore' for individual checks to accumulate all missing dependencies
        before reporting, providing better UX than failing on first missing dep.
        """
        logger.debug(f"Checking module group '{group}' with error mode '{error}'")

        # Validate group exists before attempting any imports
        if group not in MODULE_GROUPS:
            logger.error(f"Attempted to check invalid module group: {group}")
            raise ValueError(f"Unknown module group: {group}")

        # Track missing modules to report all missing deps at once
        missing = []
        for name in MODULE_GROUPS[group]["modules"]:
            logger.debug(f"Checking dependency {name} for group {group}")
            if self.module_exists(name, error="ignore") is None:
                missing.append(name)
                logger.debug(f"Module {name} is missing from group {group}")

        if missing:
            # Construct helpful message with installation instructions
            msg = (
                f"Missing optional dependencies for {MODULE_GROUPS[group]['description']}: "
                f"{', '.join(missing)}. Install with: pip install {MODULE_GROUPS[group]['install']}"
            )
            if error == "warn":
                logger.warning(msg)
            elif error == "raise":
                logger.error(msg)
                raise ImportError(msg)
        else:
            logger.debug(f"All dependencies present for group {group}")

        return len(missing) == 0
