# Contributing to Soundscapy

## General Principles

- Use the `uv` tool for managing dependencies and other project tasks. `uv add` and `uv remove` should be used to add or remove dependencies. `uv add --optional <group>` should be used to add an optional dependency. `uv sync # add --all-extras, etc as needed` should be used to install dependencies and sync with lock file. `uv build` should be used to build the package.
- Try to keep all necessary configurations to `pyproject.toml` where possible. This includes versioning, optional dependencies, tool settings (e.g. `bumpver`) and other project settings.
- Wherever possible, centralise operations and metadata. For instance, version is defined in `pyproject.toml` and automatically brought into `soundscapy` metadata in `__init__.py`; optional dependency checks are performed at the `<module>.__init__.py` level, rather than for each individual function.

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

```txt
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

Soundscapy uses a simple and standard approach to handle optional dependencies.

### Core Components

1. **Package Configuration** (`pyproject.toml`):

   ```toml
   [project.optional-dependencies]
   audio = [
       "mosqito>=1.2.1",
       "scikit-maad>=1.4.3",
       "acoustic-toolbox>=0.1.2",
   ]
   ```

2. **Module-Level Dependency Check** (`audio/__init__.py`):

   ```python
   # Check for required dependencies directly
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

   # Now import module components
   from .binaural import Binaural
   ```

3. **Top-Level Imports** (`soundscapy/__init__.py`):

   ```python
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

#### 2. Implement Feature Code

Create a new module directory if needed (e.g., `new_group/`) with an `__init__.py`:

```python
"""Module docstring describing the new functionality."""
# Check dependencies directly
try:
    import package1
    import package2
except ImportError as e:
    raise ImportError(
        "This functionality requires additional dependencies. "
        "Install with: pip install soundscapy[new_group]"
    ) from e

# Now import your feature code
from .feature import NewFeature

__all__ = ["NewFeature"]
```

#### 3. Add to Top-Level Exports

Update the main `__init__.py` to import and expose the new module:

```python
# Try to import optional new_group module
try:
    from soundscapy import new_group
    from soundscapy.new_group import NewFeature
    __all__.extend(["new_group", "NewFeature"])
except ImportError:
    # new_group module not available - expected if dependencies aren't installed
    pass
```

### How It Works

The system uses standard Python try/except patterns at two levels:

1. **Module Level**: Each optional module checks for its dependencies on import and raises a helpful error if they're missing.

2. **Top Level**: The main package tries to import optional modules and their components, extending **all** only when available.

Benefits:

- Clear error messages when dependencies are missing
- Standard Python import patterns that are easy to understand
- Good IDE support through explicit exports
- No runtime overhead for unused optional features
- Simpler to maintain and extend

### Testing Optional Dependencies

Soundscapy uses a flexible system for testing optional dependencies that allows both local development testing and full integration testing in CI.

#### Test Structure

Optional dependency tests exist at several levels:

1. **Optional Module Tests**: Tests within optional modules (e.g., `audio/`)

   - Only collected when dependencies are available using pytest_ignore_collect
   - Test actual functionality
   - No need for special markers

2. **Integration Tests**: Tests that use optional features from other modules

   - Use `@pytest.mark.optional_deps('group')` marker
   - Expected to fail when dependencies are unavailable
   - Test actual integration between components

3. **In-Development Module Tests**: For modules under active development (e.g., `spi/`)
   - Use module-level `pytestmark = pytest.mark.skip(reason="...")` to skip all tests
   - Prevent pytest collection errors by using `--ignore=src/soundscapy/module/` flag
   - Simplifies testing during development when module imports might fail

#### Testing with Tox

Soundscapy's tox configuration provides separate environments for different dependency groups:

```bash
# Run core tests (no optional dependencies)
tox -e py310-core

# Run with audio dependencies
tox -e py310-audio

# Run with SPI dependencies
tox -e py310-spi

# Run with all dependencies
tox -e py310-all
```

Each environment is configured to run the appropriate tests:

- Core: Run only tests with no optional dependency requirements
- Audio: Run core tests and audio-specific tests, skipping SPI tests
- SPI: Run core tests and SPI-specific tests
- All: Run all tests

Test selection is implemented using pytest's keyword-based filtering to precisely target the right tests:

```python
# Core tests only
pytest -k "not optional_deps"

# Core + audio tests
pytest -k "not optional_deps or optional_deps and audio"

# Core + SPI tests
pytest -k "not optional_deps or optional_deps and spi"
```

#### When to Use Each Testing Approach

1. **Use `@pytest.mark.optional_deps` when**:

   - Testing actual functionality that requires dependencies
   - Writing integration tests
   - Testing with real package interactions

2. **No special handling needed when**:

   - Writing tests within an optional module directory
   - Testing core functionality that doesn't use optional features

3. **Use module-level skip markers when**:
   - Working on a module that is not yet ready for testing
   - Dependencies might cause import errors during collection
   - You want to include tests in the codebase but skip their execution

### Adding Tests for New Optional Features

When adding new optional features:

1. **Inside Optional Module**:

   - Put tests in the module's test directory (e.g., `test/new_group/`)
   - Tests will only be collected when dependencies are available
   - No markers needed for tests within the module's directory

   ```python
   # test/new_group/test_feature.py
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
       """Will be marked as expected to fail if dependencies missing."""
       from soundscapy import NewFeature
       assert NewFeature.integrate() == expected
   ```

## Github Actions

Soundscapy has three primary workflows: `test.yml`, `test-tutorials.yml` and `tag-release.yml`. `test.yml` runs the test suite on all pushes and pull requests. `tag-release.yml` is triggered by a tag push to `main` or `dev` and creates a release on Github and publishes to PyPI.

In all cases, python and dependencies are managed and installed with `uv`.

`test.yml` uses tox to test across multiple Python versions and dependency combinations. The workflow has a two-stage approach:

1. **Linting**: First, it runs ruff checks for code quality and formatting.

2. **Testing**: Then, it runs tox with different environments:
   - **Core**: Tests with just core dependencies using `py{310,311,312}-core`
   - **Audio**: Tests with audio dependencies using `py{310,311,312}-audio`
   - **All**: Tests with all dependencies using `py{310,311,312}-all`

This approach ensures consistent testing between local development (using tox locally) and CI environments, while verifying that the package works correctly with different Python versions and dependency combinations.

`test-tutorials.yml` uses `--nbmake` to convert the notebooks to python files and run them. This is useful for testing the tutorials and ensuring they are up to date. It does not test the veracity of the outputs, just whether the notebooks run without errors.

When `tag-release.yml` runs, it will also run the tests by `uses` and `needs` calling the test workflows. This ensures that the release is only created if the tests pass. Then, it will use `uv build` to build the package, the PyPI publish action to publish to PyPI, and the Github release action to create a release on Github. The release is created with the tag name and the release notes are taken from the tag message.
