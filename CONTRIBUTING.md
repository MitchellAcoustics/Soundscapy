# Contributing to Soundscapy

## Optional Dependencies System

Soundscapy uses a simple but robust system for handling optional features and their dependencies. The system is designed to:

- Provide clear error messages when dependencies are missing
- Allow testing with and without optional dependencies
- Support easy addition of new optional features
- Handle doctest examples appropriately

### Core Components

1. **Dependency Definitions** (`_optionals.py`):

    ```python
    OPTIONAL_DEPENDENCIES = {
        "audio": {
            "packages": ("mosqito", "maad", "acoustics"),
            "install": "soundscapy[audio]",
            "description": "audio analysis functionality",
        },
    }

    def require_dependencies(group: str) -> Dict[str, Any]:
        """Import and return all packages required for a dependency group."""
    ```

2. **Package Configuration** (`pyproject.toml`):

    ```toml
    [project.optional-dependencies]
    audio = [
        "mosqito>=1.2.1",
        "scikit-maad>=1.4.3",
        "acoustics>=0.2.5",
    ]
    ```

3. **Test Configuration** (`conftest.py`):
   The root conftest.py handles:
   - Collection control (skipping modules when dependencies are missing)
   - Environment variables for doctests
   - Test markers for optional dependencies

## Adding Optional Features

### 1. Add New Dependencies

Add new packages to an existing group or create a new group:

```bash
# For existing groups:
uv add new-package --optional audio

# For new groups:
uv add package1 package2 --optional new_group
```

### 2. Update Dependency Definitions

In `_optionals.py`:

```python
OPTIONAL_DEPENDENCIES = {
    "audio": {
        "packages": ("mosqito", "maad", "acoustics", "new-package"),  # Add to existing
        "install": "soundscapy[audio]",
        "description": "audio analysis functionality",
    },
    "new_group": {  # Or create new group
        "packages": ("package1", "package2"),
        "install": "soundscapy[new_group]",
        "description": "description of functionality",
    },
}
```

### 3. Update Test Collection

If adding a new dependency group that requires its own module (like audio/), update the collection check in conftest.py:

```python
def pytest_ignore_collect(collection_path):
    """Control test collection for optional dependency modules."""
    path_str = str(collection_path)
    
    # Add new module paths here
    if "soundscapy/audio/" in path_str:
        return not _check_audio_deps()
    elif "soundscapy/new_group/" in path_str:
        return not _check_new_group_deps()
    
    return False
```

### 4. Implement Feature Code

In your module's docstrings, you only need to skip examples that would fail even with dependencies (like missing files or settings):

```python
"""Module docstring with examples.

Examples
--------
>>> from soundscapy.audio import Feature
>>> feature = Feature.from_file("audio.wav")  # doctest: +SKIP
>>> feature.sampling_rate
44100
"""
from soundscapy._optionals import require_dependencies

# This will raise an ImportError if dependencies are missing
required = require_dependencies("group_name")

# Now import your feature code
from .feature import FeatureClass
```

The module's doctests don't need special handling for dependencies because:

1. If dependencies are missing, the entire module is skipped during collection
2. If dependencies are present, all doctests in the module will run
3. Only use `doctest: +SKIP` for examples that need external files or resources

Note: REQUIRES directives are only needed for doctests outside the optional module that need to handle both success and failure cases.

### 5. Add Tests

For tests outside the optional module, mark them with the optional_deps decorator:

```python
import pytest

@pytest.mark.optional_deps('audio')
def test_feature():
    from soundscapy.audio import Feature
    ...

# Tests within the optional module don't need markers - they're handled by collection
```

### Where to Use xdoctest REQUIRES Directives

The `xdoctest: +REQUIRES(env:AUDIO_DEPS=='1')` directive is only needed in specific cases:

1. **DO NOT USE** in optional module files (e.g., `audio/*.py`):

   ```python
   # In audio/feature.py
   """
   >>> from soundscapy.audio import Feature  # No REQUIRES needed
   >>> feature = Feature()  # doctest: +SKIP (only if needs external resources)
   """
   ```

1. **DO USE** in files outside optional modules that import them:

   ```python
   # In soundscapy/core.py
   """
   >>> # xdoctest: +REQUIRES(env:AUDIO_DEPS=='1')
   >>> from soundscapy.audio import Feature
   >>> feature = Feature()
   """
   ```

1. **DO USE** when demonstrating dependency error handling:

   ```python
   # In _optionals.py or other core files
   """
   >>> # xdoctest: +REQUIRES(env:AUDIO_DEPS=='0')
   >>> from soundscapy._optionals import require_dependencies
   >>> try:
   ...     require_dependencies("audio")
   ... except ImportError as e:
   ...     print(str(e))
   audio analysis functionality requires additional dependencies...
   """
   ```

This is because:

- Optional module files are completely skipped during collection if dependencies are missing
- Files outside optional modules are always collected, so they need explicit control over which examples run
- Error handling examples specifically need to run when dependencies are missing

## How Testing Works

The testing system uses several mechanisms to handle optional dependencies:

1. **Module Collection**:
   - The root conftest.py checks dependencies during collection
   - Modules (and their tests) are skipped entirely if dependencies are missing
   - This prevents import errors during test collection

2. **Environment Variables**:
   - conftest.py sets environment variables (e.g., AUDIO_DEPS)
   - These control which doctests/examples run
   - Allows showing both success and failure cases

3. **Test Markers**:
   - Used for tests outside optional modules
   - Allow granular control over which tests run
   - Helpful for integration tests

## Running Tests

- Run all tests: `pytest`
- Run specific group: `pytest -m "optional_deps('audio')"`
- Skip optional tests: `pytest -m "not optional_deps"`

The test system will automatically:

- Skip collecting modules with missing dependencies
- Run appropriate doctests based on available dependencies
- Skip marked tests when dependencies are missing

## Best Practices

1. **Dependency Management**:
   - Keep dependencies minimal and logical
   - Group related dependencies together
   - Document dependencies in pyproject.toml and _optionals.py

2. **Error Messages**:
   - Use require_dependencies() for consistent error messages
   - Include installation instructions in error messages

3. **Testing**:
   - Add both positive and negative doctest examples
   - Use markers only for tests outside optional modules
   - Test with and without dependencies installed

4. **Documentation**:
   - Document which features require which dependencies
   - Show examples for both success and failure cases
   - Keep doctests up to date with actual functionality

5. **Code Organization**:
   - Keep optional features in their own modules
   - Handle dependencies at module boundaries
   - Follow existing code style (Ruff formatter)

## Additional Development Guidelines

- Run tests with dependencies installed and without
- Update docstrings when changing dependencies
- Follow existing code style
- Keep the dependency system documentation updated