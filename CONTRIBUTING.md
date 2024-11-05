# Contributing to Soundscapy

## General Principles

- Use the `uv` tool for managing dependencies and other project tasks. `uv add` and `uv remove` should be used to add or remove dependencies. `uv add --optional <group>` should be used to add an optional dependency. `uv sync # add --all-extras, etc as needed` should be used to install dependencies and sync with lock file. `uv build` should be used to build the package.
- Try to keep all necessary configurations to `pyproject.toml` where possible. This includes versioning, optional dependencies, tool settings (e.g. `bumpver`) and other project settings.
- Wherever possible, centralise operations and metadata. For instance, version is defined in `pyproject.toml` and automatically brought into `soundscapy` metadata in `__init__.py`; optional dependency groups are defined in `_optionals.py` and checked once at the `<module>.__init__.py` level, rather than for each individual function or at the `soundscapy.__init__.py` level.

## Linting and Formatting

Soundscapy uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting. This will be checked in the CI pipeline, so make sure to run it before committing.

## Releases and Versioning

Soundscapy uses [Semantic Versioning](https://semver.org/). The version number is stored in `soundscapy/pyproject.toml` and updated for each release.

Releases are instantiated by pushing a tag to `dev` or `main`. The tag should be in the format `vX.Y.Z` for stable releases and `vX.Y.ZrcN` for release candidates. Pre-release tags should be used for testing and development purposes only. `dev` tags will trigger a workflow that builds the package and publishes it to the test PyPI server. `rc` or no pre-release tags will trigger a workflow that builds the package and publishes it to PyPI.

Developers should use [`bumpver`](https://github.com/mbarkhau/bumpver) to update the version number. This tool automatically increments the version number where needed and can also apply git tags and push release tags. Pre-releases should be incremented with:

```bash
bumpver update --tag-num
```

For stable releases, use:

```bash
bumpver update --patch # or --minor or --major
```

I recommend testing this with `--dry` first to see what changes will be made. Additional options are available for refraining from committing, pushing, or tagging.

The settings for `bumpver` are stored in `pyproject.toml`.

1. **Major Version**:

   - Incremented for incompatible API changes
   - Currently on zero version pre-stable release. Therefore, breaking changes should be expected and noted using a minor version bump.

2. **Minor Version**:

    - Incremented for new features or significant changes
    - Reset to 0 for major versions

3. **Patch Version**:

    - Incremented for bug fixes or minor changes
    - Reset to 0 for new minor versions

4. **Pre-release Versions**:

    - Use `rc` for release candidates
    - Use `dev` for development versions

### Commit messages

Try to use the [Angular commit message format](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines) for commit messages. Mostly this means starting the commit message with a type, followed by a colon and a short description. For example:

``` txt
feat: add new feature
fix: correct bug in feature
docs: update documentation
```

The avilable types are:

- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing or correcting existing tests
- chore: Changes to the build process or auxiliary tools and libraries such as documentation generation

## Optional Dependencies System

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

### Adding Optional Features

#### 1. Add New Dependencies

Add new packages to an existing group or create a new group:

```bash
# For existing groups:
uv add new-package --optional audio

# For new groups:
uv add package1 package2 --optional new_group
```

#### 2. Update Dependency Definitions

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

#### 3. Update Test Collection

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

#### 4. Implement Feature Code

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

#### Where to Use xdoctest REQUIRES Directives

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

### How Testing Works

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

### Running Tests

The test system **should** automatically:

- Skip collecting modules with missing dependencies
- Run appropriate doctests based on available dependencies
- Skip marked tests when dependencies are missing

### Best Practices

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

### Additional Development Guidelines

- Run tests with dependencies installed and without
- Update docstrings when changing dependencies
- Follow existing code style
- Keep the dependency system documentation updated

## Github Actions

Soundscapy has three primary workflows: `test.yml`, `test-tutorials.yml` and `tag-release.yml`. `test.yml` runs the test suite on all pushes and pull requests. `tag-release.yml` is triggered by a tag push to `main` or `dev` and creates a release on Github and publishes to PyPI.

In all cases, python and dependencies are managed and installed with `uv`.

`test.yml` will test on multiple python versions, defined by the `python-version` matrix. First, it will install the core dependencies with `uv sync`, then run the test suite (which **should** ignore the tests requiring optional dependencies). Then, it will install all optional dependencies `uv sync --all-extras` and run the tests again. This ensures that the tests run with and without optional dependencies.

`test-tutorials.yml` uses `--nbmake` to convert the notebooks to python files and run them. This is useful for testing the tutorials and ensuring they are up to date. It does not test the veracity of the outputs, just whether the notebooks run without errors.

When `tag-release.yml` runs, it will also run the tests by `uses` and `needs` calling the test workflows. This ensures that the release is only created if the tests pass. Then, it will use `uv build` to build the package, the PyPI publish action to publish to PyPI, and the Github release action to create a release on Github. The release is created with the tag name and the release notes are taken from the tag message.
