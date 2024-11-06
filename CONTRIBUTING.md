# Contributing to Soundscapy

## General Principles

- Use the `uv` tool for managing dependencies and other project tasks. `uv add` and `uv remove` should be used to add or remove dependencies. `uv add --optional <group>` should be used to add an optional dependency. `uv sync # add --all-extras, etc as needed` should be used to install dependencies and sync with lock file. `uv build` should be used to build the package.
- Try to keep all necessary configurations to `pyproject.toml` where possible. This includes versioning, optional dependencies, tool settings (e.g. `bumpver`) and other project settings.
- Wherever possible, centralise operations and metadata. For instance, version is defined in `pyproject.toml` and automatically brought into `soundscapy` metadata in `__init__.py`; optional dependency groups are defined in `_optionals.py` and checked once at the `<module>.__init__.py` level, rather than for each individual function or at the `soundscapy.__init__.py` level.

Changes should be made in a feature branch and submitted to `dev` via a pull request. The pull request should be reviewed by at least one other developer before being merged. The `main` branch should only contain stable releases. Docs can be updated directly on `dev` or `main` as needed.

## Linting and Formatting

Soundscapy uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting. This will be checked in the CI pipeline, so make sure to run it before committing.

## Releases and Versioning

Soundscapy uses [Semantic Versioning](https://semver.org/). The version number is stored in `soundscapy/pyproject.toml` and updated for each release.

Releases are instantiated by pushing a tag to `dev` or `main`. The tag should be in the format `vX.Y.Z` for stable releases and `vX.Y.ZrcN` for release candidates. Pre-release tags should be used for testing and development purposes only. `dev` tags will trigger a workflow that builds the package and publishes it to the test PyPI server. This shouldn't need to happen often - at the moment I'm using it mostly for testing the CI tools. `rc` or no pre-release tags will trigger a workflow that builds the package and publishes it to PyPI.

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
   # Package dependencies
   OPTIONAL_DEPENDENCIES = {
       "audio": {
           "packages": ("mosqito", "maad", "acoustics"),
           "install": "soundscapy[audio]",
           "description": "audio analysis functionality",
       },
   }

   # Top-level imports available when dependencies are installed
   OPTIONAL_IMPORTS = {
       'Binaural': ('soundscapy.audio', 'Binaural'),
       'AudioAnalysis': ('soundscapy.audio', 'AudioAnalysis'),
       # ... other optional components
   }
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

3. **Module-Level Dependency Check** (`audio/__init__.py`):

   ```python
   from soundscapy._optionals import require_dependencies

   # This will raise an ImportError if dependencies are missing
   required = require_dependencies("audio")

   # Now import module components
   from .binaural import Binaural
   ```

### Adding New Optional Features

#### 1. Add New Dependencies

Add new packages to an existing group or create a new group:

```bash
# For existing groups:
uv add new-package --optional audio

# For new groups:
uv add package1 package2 --optional new_group
```

#### 2. Update Dependency Definitions

In `_optionals.py`, update both dependency mappings:

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

OPTIONAL_IMPORTS = {
    # Existing imports...
    'NewFeature': ('soundscapy.new_group', 'NewFeature'),  # Add new top-level imports
}
```

#### 3. Implement Feature Code

Create a new module directory if needed (e.g., `new_group/`) with an `__init__.py`:

```python
"""Module docstring describing the new functionality."""
from soundscapy._optionals import require_dependencies

# This will raise an ImportError if dependencies are missing
required = require_dependencies("new_group")

# Now import your feature code
from .feature import NewFeature

__all__ = ["NewFeature"]
```

#### 4. Add to Top-Level Exports

If you want the new feature to be available at the top level, add it to `__all__` in `soundscapy/__init__.py`:

```python
__all__ = [
    # Core modules...
    # Optional modules
    "NewFeature",  # Add new feature here
]
```

### How It Works

The system provides three levels of dependency handling:

1. **Module Level**: The `require_dependencies()` check in the optional module's `__init__.py` ensures dependencies are available before the module is imported.

2. **Top Level Imports**: `__getattr__` in the main `__init__.py` enables importing optional components directly from `soundscapy` with proper error handling:

   ```python
   from soundscapy import Binaural  # Works with deps, helpful error without
   ```

3. **IDE Support**: The explicit `__all__` list in `__init__.py` provides IDE autocompletion while maintaining proper runtime behavior.

Benefits:

- Clear error messages when dependencies are missing
- Optional components available at both module and package level
- Good IDE support through explicit exports
- Centralized dependency configuration
- No runtime overhead for unused optional features

### Testing Optional Dependencies

Soundscapy uses a flexible system for testing optional dependencies that allows both local development testing and full integration testing in CI.

#### Test Structure

Optional dependency tests exist at three levels:

1. **Optional Module Tests**: Tests within optional modules (e.g., `audio/`)
   - Only collected when dependencies are available
   - Test actual functionality
   - No need for special markers or mocking

2. **Integration Tests**: Tests that use optional features from other modules
   - Use `@pytest.mark.optional_deps('group')` marker
   - Skip when dependencies unavailable
   - Test actual integration between components


#### When to Use Each Testing Approach

1. **Use `@pytest.mark.optional_deps` when**:
   - Testing actual functionality that requires dependencies
   - Writing integration tests
   - Testing with real package interactions

2. **No special handling needed when**:
   - Writing tests within an optional module
   - Testing core functionality that doesn't use optional features

### Adding Tests for New Optional Features

When adding new optional features:

1. **Inside Optional Module**:
   - Put tests in the module's test directory
   - No special handling needed
   - Tests will only run when dependencies are available

   ```python
   # soundscapy/new_group/tests/test_feature.py
   def test_new_feature():
       """Regular test, no special handling needed."""
       from soundscapy.new_group import NewFeature
       assert NewFeature.method() == expected
   ```

2. **Integration Tests**:
   - Use the optional_deps marker
   - Put in main test directory

   ```python
   # test/test_integration.py
   @pytest.mark.optional_deps('new_group')
   def test_new_feature_integration():
       """Will skip if dependencies missing."""
       from soundscapy import NewFeature
       assert NewFeature.integrate() == expected
   ```

## Github Actions

Soundscapy has three primary workflows: `test.yml`, `test-tutorials.yml` and `tag-release.yml`. `test.yml` runs the test suite on all pushes and pull requests. `tag-release.yml` is triggered by a tag push to `main` or `dev` and creates a release on Github and publishes to PyPI.

In all cases, python and dependencies are managed and installed with `uv`.

`test.yml` will test on multiple python versions, defined by the `python-version` matrix. First, it will install the core dependencies with `uv sync`, then run the test suite (which **should** ignore the tests requiring optional dependencies). Then, it will install all optional dependencies `uv sync --all-extras` and run the tests again. This ensures that the tests run with and without optional dependencies.

`test-tutorials.yml` uses `--nbmake` to convert the notebooks to python files and run them. This is useful for testing the tutorials and ensuring they are up to date. It does not test the veracity of the outputs, just whether the notebooks run without errors.

When `tag-release.yml` runs, it will also run the tests by `uses` and `needs` calling the test workflows. This ensures that the release is only created if the tests pass. Then, it will use `uv build` to build the package, the PyPI publish action to publish to PyPI, and the Github release action to create a release on Github. The release is created with the tag name and the release notes are taken from the tag message.
