# Detailed Implementation Plan for Simplifying Optional Dependencies

## Background

Soundscapy currently uses a complex system for handling optional dependencies that involves:
- Centralized dependency definitions in `_optionals.py`
- Lazy imports via `__getattr__` in the main package
- Module-level dependency checks with `require_dependencies`

This plan outlines the steps to simplify this approach with standard try/except patterns while maintaining the same functionality.

## 1. Core Files Changes

### A. `src/soundscapy/__init__.py`
```python
# REPLACE
from soundscapy._optionals import import_optional
...
def __getattr__(name: str) -> Any:
    """Lazy import handling for optional components."""
    return import_optional(name)

# WITH
# Try to import optional audio module
try:
    from soundscapy import audio
    from soundscapy.audio import (
        Binaural, AudioAnalysis, AnalysisSettings, ConfigManager,
        process_all_metrics, prep_multiindex_df, add_results, parallel_process,
    )
    __all__.extend([
        "audio", "Binaural", "AudioAnalysis", "AnalysisSettings", 
        "ConfigManager", "process_all_metrics", "prep_multiindex_df",
        "add_results", "parallel_process",
    ])
except ImportError:
    # Audio module not available - this is expected if dependencies aren't installed
    pass

# Try to import optional SPI module
try:
    from soundscapy import spi
    from soundscapy.spi import (
        SkewNormalDistribution, fit_skew_normal, calculate_spi, calculate_spi_from_data,
    )
    __all__.extend([
        "spi", "SkewNormalDistribution", "fit_skew_normal", 
        "calculate_spi", "calculate_spi_from_data",
    ])
except ImportError:
    # SPI module not available
    pass
```

### B. Remove `src/soundscapy/_optionals.py` completely
This file will be completely eliminated as its functionality is being replaced with direct imports.

## 2. Module Level Changes

### A. `src/soundscapy/audio/__init__.py`
```python
# REPLACE
from soundscapy._optionals import require_dependencies
# This will raise an ImportError if the required dependencies are not installed
required = require_dependencies("audio")

# WITH
# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import mosqito
    import maad
    import tqdm
    import acoustic_toolbox
except ImportError as e:
    raise ImportError(
        "Audio analysis functionality requires additional dependencies. "
        "Install with: pip install soundscapy[audio]"
    ) from e
```

### B. `src/soundscapy/spi/__init__.py`
```python
# REPLACE
from soundscapy._optionals import require_dependencies
# This will raise an ImportError if dependencies are missing
required = require_dependencies("spi")

# WITH
# Check for Python dependencies directly
try:
    import rpy2.robjects as robjects
except ImportError as e:
    raise ImportError(
        "Soundscape perception indices calculation requires additional dependencies. "
        "Install with: pip install soundscapy[spi]"
    ) from e

# Check for R dependencies
try:
    # Code to check R and sn package availability
    r_version = robjects.r('R.version.string')[0]
    robjects.r('library(sn)')
except Exception as e:
    raise ImportError(
        f"Error with R dependencies: {str(e)}. "
        "Please ensure R and the 'sn' package are installed."
    ) from e
```

## 3. Test Files Updates

### A. `test/test_basic.py`
```python
# Update test_soundscapy_audio_module to test for module instead of class
@pytest.mark.optional_deps("audio")
def test_soundscapy_audio_module():
    assert hasattr(soundscapy, "audio"), "Soundscapy should have an audio module"
    # Test that the key classes are available
    assert hasattr(soundscapy, "Binaural")
    assert hasattr(soundscapy, "AudioAnalysis")

# Update test_soundscapy_spi_module similarly
@pytest.mark.optional_deps("spi")
def test_soundscapy_spi_module():
    assert hasattr(soundscapy, "spi"), "Soundscapy should have an spi module"
    # Test that the key classes are available
    assert hasattr(soundscapy, "SkewNormalDistribution")
```

### B. `test/test__optionals.py`
This test file will need to be completely removed or rewritten to test the new approach:
- Tests for `require_dependencies` will need to be removed
- Tests for `import_optional` will need to be removed
- New tests for the direct import behavior will be needed

### C. `conftest.py`
```python
# REPLACE
def _check_dependencies(group: str) -> bool:
    """Check for dependencies of a group, caching the result."""
    if group not in _dependency_cache:
        try:
            from soundscapy._optionals import require_dependencies

            required = require_dependencies(group)
            logger.debug(f"{group} dependencies found: {list(required.keys())}")
            _dependency_cache[group] = True
        except ImportError as e:
            logger.debug(f"Missing {group} dependencies: {e}")
            _dependency_cache[group] = False
    return _dependency_cache[group]

# WITH
def _check_dependencies(group: str) -> bool:
    """Check for dependencies of a group, caching the result."""
    if group not in _dependency_cache:
        try:
            if group == "audio":
                # Try importing audio-related modules
                import mosqito
                import maad
                import tqdm
                import acoustic_toolbox
                _dependency_cache[group] = True
                logger.debug(f"{group} dependencies found: ['mosqito', 'maad', 'tqdm', 'acoustic_toolbox']")
            elif group == "spi":
                # Try importing SPI-related modules
                import rpy2
                _dependency_cache[group] = True
                logger.debug(f"{group} dependencies found: ['rpy2']")
            else:
                logger.debug(f"Unknown dependency group: {group}")
                _dependency_cache[group] = False
        except ImportError as e:
            logger.debug(f"Missing {group} dependencies: {e}")
            _dependency_cache[group] = False
    return _dependency_cache[group]
```

### D. Update `pytest_configure` in `conftest.py`
```python
# REPLACE
# Set environment variables for each dependency group
from soundscapy._optionals import OPTIONAL_DEPENDENCIES

for group in OPTIONAL_DEPENDENCIES:
    env_var = f"{group.upper()}_DEPS"
    os.environ[env_var] = "1" if _check_dependencies(group) else "0"
    logger.debug(f"Set {env_var}={os.environ[env_var]}")

# WITH
# Define known dependency groups
dependency_groups = ["audio", "spi"]
for group in dependency_groups:
    env_var = f"{group.upper()}_DEPS"
    os.environ[env_var] = "1" if _check_dependencies(group) else "0"
    logger.debug(f"Set {env_var}={os.environ[env_var]}")
```

## 4. Impact on tox.ini

The tox.ini file should work with the new approach without modifications, as it tests for the presence of modules and classes rather than specific implementation details.

## 5. Implementation Order

1. Create a new git branch for these changes: `git checkout -b simplify-optional-deps`
2. Implement changes in the following order:
   a. Update `audio/__init__.py` and `spi/__init__.py`
   b. Update `src/soundscapy/__init__.py`
   c. Update `conftest.py`
   d. Update or remove `test/test__optionals.py`
   e. Update `test/test_basic.py`
3. Run tests to verify the changes: `uv run pytest -m not audio`
4. Run full test suite with optional dependencies: `uv run pytest`
5. Fix any issues that arise during testing
6. Once all tests pass, commit the changes

## 6. Key Considerations

### Error Handling
- Maintain helpful error messages when dependencies are missing
- Make sure error messages include installation instructions

### Testing
- Ensure tests still properly skip when dependencies are missing
- Make sure the `optional_deps` marker still works correctly

### Compatibility
- Ensure correct handling of Python and R dependencies for SPI module
- Make sure classes are still available from the top-level package

## 7. Expected Benefits

1. **Simplicity**: More straightforward code using standard Python patterns
2. **Maintainability**: Easier to understand and modify for new contributors
3. **Consistency**: More consistent with typical Python package design
4. **Performance**: Potential for better import performance by removing indirection
5. **Debugging**: Simpler debugging experience with more direct import paths

## 8. Potential Challenges

1. **IDE Support**: Need to ensure IDE autocompletion still works correctly
2. **Error Messages**: Must maintain helpful error messages and installation instructions
3. **Test Compatibility**: Update tests to work with the new system without breaking existing test markers
4. **Edge Cases**: Careful handling of import edge cases during the transition