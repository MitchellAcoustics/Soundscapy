# Copilot Instructions for Soundscapy

Soundscapy is a Python library for analysing and visualising soundscape assessments. It processes survey data (Perceptual Assessment Questionnaires — PAQs) and optionally analyses audio files for psychoacoustic metrics.

## Build, Test, and Lint

**Package manager:** [Pixi](https://pixi.sh) — use `pixi run <task>` for all commands. Do not use plain `pip install` or `conda`.

### Linting and formatting

```bash
pixi run lint          # ruff check
pixi run format        # ruff check --fix && ruff format
```

Type checking uses **Pyrefly** (not mypy/pyright for CI, though mypy config exists).

### Testing

Tests live in `test/` and also as doctests inside `src/soundscapy/` (collected by xdoctest via pytest). Run the appropriate suite for what's installed:

```bash
pixi run test-import-tripwire   # guard: import soundscapy must not pull in optional deps
pixi run test-base              # core tests, no optional extras
pixi run test-audio             # includes audio extras
pixi run test-r                 # includes R extras
pixi run test-all               # full suite
pixi run tests                  # all of the above in sequence
```

To run a single test file or test function:

```bash
pixi run -e test pytest test/test_basic.py
pixi run -e test pytest test/test_basic.py::test_function_name
pixi run -e test pytest -k "keyword"
```

### Documentation

```bash
pixi run -e docs docs-build     # full build → docs/_site/
pixi run -e docs docs-preview   # local preview at http://localhost:3000
```

The docs site is built with **great-docs** (`docs/great-docs.yml`; all docs sources live under `docs/`). The `docs/_quarto/` build directory is ephemeral and gitignored — never edit files inside it. Narrative user-guide pages (including the news and migration pages, grouped under the "Project" guide-section) live in `docs/user_guide/`, tutorials and papers in `docs/tutorials/`, and the Gallery (a blog-style Quarto listing of short plot examples) in `docs/gallery/`; pages are auto-discovered and ordered by numeric prefix. Keep user-guide pages explanatory and put plot-variety examples in the Gallery.

The site is versioned via `versions:` in `docs/great-docs.yml`: the latest release is served at the root (API reference from its git tag via `git_ref`) and `main` under `/v/dev/`. Narrative pages are shared across versions, so when documenting an unreleased feature, scope it with front matter `versions: ">=X.Y"` or a `::: {.version-only versions=">=X.Y"}` fence so it stays out of the stable docs. `.github/workflows/docs.yml` deploys to GitHub Pages on every push to `main`; `release.yml` calls it with `deploy: false` as a build check only.

## Architecture

```text
src/soundscapy/
├── __init__.py / __init__.pyi   # top-level API + lazy-loader stub
├── _optional.py                 # optional-dependency gate
├── surveys/                     # core PAQ processing (always available)
├── databases/                   # dataset loaders: ISD, SATP, ARAUS
├── plotting/                    # ISO circumplex plots, Likert charts
├── audio/                       # psychoacoustic metrics (optional: soundscapy[audio])
├── spi/                         # Soundscape Perception Index (optional: soundscapy[r])
├── satp/                        # SATP analysis via R bridge (optional: soundscapy[r])
└── r_wrapper/                   # rpy2-based R interop
```

**Core (always available):** `surveys`, `databases`, `plotting`
**Optional extras:** `audio` (acoustic-toolbox, mosqito, scikit-maad), `r` (rpy2 + R packages)

### Optional dependency pattern

Two complementary mechanisms keep `import soundscapy` lightweight:

1. **Gate (`_optional.py`):** Every optional subpackage's `__init__.py` calls `_require_deps(["pkg1", "pkg2"], extra="audio")` as its **first statement** before any heavy imports. This raises a uniform `ImportError` with the install hint if deps are missing.

2. **Lazy loading (SPEC 1):** `lazy_loader.attach_stub(__name__, __file__)` is used at both the top-level and inside each optional subpackage. The adjacent `.pyi` stub file is the **source of truth** for what names are exposed — it drives both the lazy `__getattr__` at runtime and static type checkers, and its literal `__all__` is what great-docs reads. At the top level, `__getattr__` is `moved_to(..., fallback=<lazy getattr>)` so deprecated names warn before falling back to lazy loading.

When adding a new optional subpackage, create both `__init__.py` (gate + attach_stub) and `__init__.pyi` (public re-exports), then expose it in `src/soundscapy/__init__.pyi`.

### Survey data conventions

Survey data uses a standardised DataFrame layout:

- **PAQ columns:** `PAQ1`–`PAQ8` (Perceptual Assessment Questionnaire items, 5-point Likert scale)
- **ISO coordinates:** `ISOPleasant` and `ISOEventful` derived via `add_iso_coords()`
- Index columns: `LocationID`, `RecordID`, `GroupID`, `SessionID`

The `PAQ_IDS` and `PAQ_LABELS` constants in `soundscapy/constants.py` are the canonical references for PAQ column names and labels.

### Plotting

The plotting module is built around the **ISO circumplex** — a 2D space with ISOPleasant (x) and ISOEventful (y) axes. The main entry points are `iso_plot()`, `scatter()`, `density()`, `jointplot()`, and `create_iso_subplots()`. Plot parameters are plain dataclasses (`param_models.py`). The layered composition system in `layers.py` assembles plots from discrete visual layers.

## Key Conventions

- **All project configuration lives in `pyproject.toml`** — versioning, optional deps, tool settings. Use `pixi add --pypi` (not manual edits) to add Python deps; use plain `pixi add` for matching conda-forge packages.

- **Version management:** The version is pinned manually in `pyproject.toml`; bump it with the pixi task, which wraps `uv version`.

  ```bash
  pixi run version --bump patch   # or minor / major
  ```

  Merging a clean `X.Y.Z` version to `main` triggers `release.yml` (tests, docs build, PyPI + prefix.dev publish, then the `vX.Y.Z` tag). Pre-release versions (`.dev`, `a`, `b`, `rc`) are skipped. In the same release PR, update `versions:` in `docs/great-docs.yml` (see the comment there).

- **Branch strategy:** Feature branches → PR into `dev`. `main` is stable releases only. Docs can be committed directly to `dev` or `main`.

- **Commit message format:** Angular style — `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`.

- **Optional dep guard placement:** `_require_deps(...)` must be the **first statement** in an optional subpackage's `__init__.py`, before any other imports.

- **`.pyi` stubs are not optional:** Every subpackage that uses `lazy_loader.attach_stub` must have a corresponding `.pyi` file. Use `from .module import Symbol as Symbol` (with the `as` alias) to mark names as public re-exports per PEP 484, and add a literal `__all__` listing the same names (great-docs and type checkers cannot evaluate a computed one).

- **Import name ↔ PyPI name mismatches:** Register them in `_DIST_NAME` inside `_optional.py` (e.g. `"acoustic_toolbox": "acoustic-toolbox"`).

- **Testing optional deps:**

  - Missing-dep gate tests → `test/test_slim_install.py` (use `monkeypatch` on `importlib.util.find_spec`)
  - Optional directory tests → `test/<subpackage>/conftest.py` with `pytest.importorskip`
  - When adding a new optional subpackage, also add a `collect_ignore_glob` entry in the root `conftest.py`

- **Logging:** Soundscapy uses loguru. The library disables its own logger by default (`logger.disable("soundscapy")`). Use `soundscapy.enable_debug()` / `setup_logging()` to re-enable. Do not use `print` or stdlib `logging` inside library code.
