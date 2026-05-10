# Contributing to Soundscapy

## General Principles

- Use **Pixi** for local environment management and command execution. In a `pyproject.toml` project, `pixi add --pypi` writes normal Python requirements into native `[project.dependencies]`, `[project.optional-dependencies]`, or `[dependency-groups]`. Use plain `pixi add` for the matching conda dependency when conda-forge provides a suitable package. Do not delete dependencies from the native Python tables just to move them into Pixi; matching entries under `[tool.pixi.dependencies]` or `[tool.pixi.feature.<name>.dependencies]` intentionally override those PyPI requirements inside Pixi environments.
- Try to keep all necessary configurations to `pyproject.toml` where possible. This includes versioning, optional dependencies, tool settings, and other project settings.
- Wherever possible, centralise operations and metadata. For instance, version is defined in `pyproject.toml` and automatically brought into `soundscapy` metadata in `__init__.py`; optional dependency checks are performed at the `<module>.__init__.py` level, rather than for each individual function.

Changes should be made in a feature branch and submitted to `dev` via a pull request. The pull request should be reviewed by at least one other developer before being merged. The `main` branch should only contain stable releases. Docs can be updated directly on `dev` or `main` as needed.

## Linting and Formatting

Soundscapy uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting and [Pyrefly](https://github.com/Pyrefly/pyrefly) for type checking. This will be checked in the CI pipeline, so make sure to run it before committing.

We use [Prek](https://prek.j178.dev) for pre-commit hooks, configured in `.pre-commit-config.yaml` and `pyproject.toml`.

## Documentation

Documentation is built with **Zensical**. Tutorial pages are rendered from the source notebooks in `docs/tutorials/*.ipynb` using **Quarto**, and the API reference pages under `docs/reference/` are maintained as regular Markdown files with `mkdocstrings`.

Use the Pixi docs tasks for local docs work:

```bash
pixi run -e docs docs-render   # Render tutorial markdown pages
pixi run -e docs docs-build    # Render and build the docs site
pixi run -e docs docs-serve    # Render and serve the docs locally
```

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

Soundscapy uses a uniform, pandas-style gate for optional dependencies, built on two
complementary tools: a small `_optional.py` helper (for the gate itself) and
`lazy_loader` / [Scientific Python SPEC 1](https://scientific-python.org/specs/spec-0001/) (for deferred top-level access and type stubs).

### Design principle

> There are two distinct problems. `require_deps` solves the *gate* problem (any import
> path — direct, submodule, or internal — must produce an actionable error message).
> `lazy_loader` solves the *deferral* problem (`import soundscapy` must pay nothing, and
> `soundscapy.Binaural` must still work). Neither can replace the other.

### Core components

#### 1. The gate helper (`src/soundscapy/_optional.py`)

`require_deps(modules, *, extra)` uses `importlib.util.find_spec` (no import side
effects) and raises a uniform `ImportError` listing the PyPI install command:

```python
from soundscapy._optional import require_deps
require_deps(["mosqito", "maad", "acoustic_toolbox", "tqdm"], extra="audio")
# ImportError: 'mosqito', 'scikit-maad', 'acoustic-toolbox', 'tqdm' required for
# soundscapy[audio], not installed. Install with:  pip install 'soundscapy[audio]'
```

Call `require_deps` as the **first statement** of every optional subpackage's
`__init__.py`, before any heavy import. This ensures any import path — `import
soundscapy.audio`, `from soundscapy.audio import Binaural`, or an internal
`from soundscapy.spi.msn import MultiSkewNorm` inside plotting code — all surface the
same message.

If you add a package whose import name differs from its PyPI name, add an entry to
`_DIST_NAME` in `_optional.py` (e.g. `"acoustic_toolbox": "acoustic-toolbox"`).

#### 2. Lazy loading and type stubs ([SPEC 1](https://scientific-python.org/specs/spec-0001/))

Each optional subpackage and the top-level package use `lazy_loader.attach_stub` to
defer imports until first use. The stub file (`.pyi`) adjacent to each `__init__.py`
is the single source of truth: it drives both `lazy_loader` at runtime and static type
checkers (mypy, pyright) at analysis time.

```text
src/soundscapy/
├── __init__.py          # attach_stub(__name__, __file__)
├── __init__.pyi         # lists audio/spi/satp attrs for lazy_loader + type checkers
├── audio/
│   ├── __init__.py      # require_deps(...) then attach_stub(__name__, __file__)
│   └── __init__.pyi     # lists Binaural, AudioAnalysis, etc.
├── spi/
│   ├── __init__.py
│   └── __init__.pyi
└── satp/
    ├── __init__.py
    └── __init__.pyi
```

`import soundscapy` pays nothing — no optional imports are attempted. The gate fires
only when the subpackage is first accessed.

#### 3. Package configuration (`pyproject.toml`)

```toml
[project.optional-dependencies]
audio = ["acoustic-toolbox>=0.1.2", "mosqito>=1.2.1", "scikit-maad>=1.4.3", "tqdm>=4.66.5"]
r     = ["rpy2>=3.5.0"]
all   = ["soundscapy[audio]", "soundscapy[r]"]
```

### Adding a new optional subpackage

#### 1. Add dependencies

```bash
# Add the PyPI dependency to the new extras group:
pixi add --pypi package1 --feature new_group

# If conda-forge has the package, add a conda override too:
pixi add package1 --feature new_group
```

Also add a pixi environment and test task for the new group — follow the `audio` and
`r` patterns in `pixi.toml`.

#### 2. Implement the gate in `new_group/__init__.py`

```python
"""Module docstring describing the new functionality."""
# ruff: noqa: E402
from soundscapy._optional import require_deps

require_deps(["package1", "package2"], extra="new_group")

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
```

#### 3. Write `new_group/__init__.pyi`

```python
from .feature import NewFeature as NewFeature
from .utils import helper_fn as helper_fn
```

The `.pyi` stub is parsed by `lazy_loader` to set up the lazy `__getattr__`, and read
by type checkers for type information. Use relative imports only (`from .X import Y`);
the `as Y` alias marks names as public re-exports per PEP 484.

#### 4. Expose at the top level

Add the new submodule and its public names to `src/soundscapy/__init__.pyi`:

```python
from . import new_group as new_group
from .new_group import NewFeature as NewFeature
```

`__init__.py` itself does not need touching — `attach_stub` reads the updated stub
automatically.

### Testing optional dependencies

#### Per-directory skip (preferred)

Create a `test/new_group/conftest.py` that calls `pytest.importorskip` for each
required package. pytest skips the entire directory if any call fails:

```python
# test/new_group/conftest.py
import pytest
pytest.importorskip("package1")
pytest.importorskip("package2")
```

Tests inside `test/new_group/` need no markers — they are collected only when their
dependencies are present.

#### Inline skip for mixed-directory tests

For tests in a shared file (e.g., `test/test_basic.py`) that verify top-level
access to an optional module, call `pytest.importorskip` at the start of the test
function:

```python
def test_new_group_available():
    pytest.importorskip("package1")
    import soundscapy
    assert hasattr(soundscapy, "NewFeature")
```

#### Gate-failure tests

Tests asserting that the gate fires with the correct error message belong in
`test/test_slim_install.py`. Use `monkeypatch` on `importlib.util.find_spec` to
simulate a missing dependency without actually uninstalling anything:

```python
def test_new_group_gate_hint(monkeypatch):
    # ... block package1 via monkeypatched find_spec ...
    with pytest.raises(ImportError, match=r"soundscapy\[new_group\]"):
        importlib.import_module("soundscapy.new_group")
```

#### Doctest collection from source

The root `conftest.py` maintains a `collect_ignore_glob` list that prevents xdoctest
from collecting inside optional source directories when their extras are not installed.
Add a new entry there when adding a new optional subpackage:

```python
if importlib.util.find_spec("package1") is None:
    collect_ignore_glob.append("src/soundscapy/new_group/*")
```

#### `errors="raise"` vs `"warn"` at call sites

`require_deps` always raises — it is for module-level gates only. If a future function
needs an optional enrichment that should degrade gracefully instead of failing hard,
use `import_optional` from `_optional.py` (not yet added; add it when the first
warn-and-degrade call site appears). The rule of thumb:

- User passed a kwarg whose **only purpose** is the optional path → `errors="raise"`
  (silent degradation would be surprising)
- User passed a kwarg that **enriches** an otherwise-complete result → `errors="warn"`
  (the function still returns something useful without the enrichment)

## Github Actions

Soundscapy has two primary workflows: `test.yml` and `linting.yml`. Dependencies and
environments are managed by **Pixi** across all workflows.

`test.yml` runs a matrix of pixi tasks across Ubuntu, macOS, and Windows:

| Task | Environment | What runs |
|---|---|---|
| `test-import-tripwire` | `test` (slim) | `import soundscapy` guard — fails if any optional dep is eagerly imported |
| `test-base` | `test` (slim) | Core tests + gate-failure tests; optional dirs skipped by `importorskip` |
| `test-audio` | `test-audio` | All of the above plus `test/audio/` |
| `test-r` | `test-r` | All of the above plus `test/spi/` and `test/satp/` |
| `test-all` | `test-all` | Full suite |

To run these locally:

```bash
pixi run test-import-tripwire   # slim-install guard
pixi run test-base              # no optional deps
pixi run test-audio             # audio extras
pixi run test-r                 # R extras
pixi run test-all               # everything
pixi run tests                  # all of the above in sequence
```
