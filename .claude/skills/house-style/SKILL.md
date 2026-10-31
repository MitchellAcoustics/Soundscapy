---
name: house-style
description: >
  Project-specific rules for soundscapy's tooling, packaging, tests, docs and
  tutorials. These rules override the vendor skills (modern-python, astral,
  prek, great-docs, quarto). Use this skill whenever you create or change
  pyproject.toml, uv.lock, ruff/ty/pytest/coverage settings, Taskfile.yml,
  prek.toml, CI workflows, great-docs config, the docs site, a
  test file or a tutorial .qmd page. Also use it when you set the toolchain up
  for the first time, or when a vendor skill's default seems to disagree with
  this repo.
---

# soundscapy house style

The vendor skills are general. This file records what is specific to this repo, and
it wins wherever the two disagree. Edit this file, never the vendored skills under
`.claude/skills/` (see `VENDORED.md`), so that updating upstream stays a clean copy.

For the great-docs and tutorial details, read `reference.md` beside this file. Read it
when you configure the docs site, or write a tutorial or user-guide page.

## Decisions

| Setting | Value | Why |
| --- | --- | --- |
| Package / import name | `soundscapy` | Keeps the existing PyPI name and citations. |
| Layout | `src/soundscapy/`, tests in `tests/` | Stops tests importing the working tree by accident. |
| Build backend | `uv_build` | One toolchain. No hatchling, setuptools or VCS versioning. |
| Python floor | `requires-python = ">=3.11"`; ruff `target-version = "py311"`; ty `python-version = "3.11"` | Product-owner decision. It is deliberately wider than the SPEC 0 window, so do not raise it to match SPEC 0 or a vendor default. Test every supported version in CI. |
| Type checker | `ty` | Astral toolchain. Never add mypy, pyright, basedpyright or pyrefly, nor their LSP plugins. |
| Lint / format | `ruff`, `select = ["ALL"]` with the commented ignores below | Maximum signal; every exemption is visible and justified. |
| Docstrings | numpy convention; every public function that implements a procedure carries a `References` section citing the standard clause or paper (PRD FR-12.4), using the `[@key]` bibliography great-docs supports | Matches the scientific Python ecosystem and great-docs rendering. |
| Git hooks | `prek`, configured in its native `prek.toml` | The `builtin` hooks already make the setup prek-only, so a pre-commit-compatible YAML file buys nothing. Never install `pre-commit` itself. |
| Task runner | `Taskfile.yml` (go-task, `version: '3'`), run as `uv run task <name>` | One set of entry points for agents, humans and CI. go-task comes from `go-task-bin` in the `dev` group, so contributors install only uv and git. Bare `task` works only with the venv activated. |
| Coverage gate | `fail_under = 80`, branch coverage on | Matches the Trail of Bits default; raise it, never lower it. |
| CI | GitHub Actions running `task` targets; the nightly workflow runs `test:reference`, `test:network`, `docs:papers` and the quick-start scripts on pushes to `soundscapy-rewrite` touching those paths, and on a schedule once the rewrite is the default branch; releases publish on tag by PyPI trusted publishing | CI runs exactly what developers run. Scheduled runs start only from the default branch, still `main`. |
| Line length | 100 | Room for unit-suffixed names without wrapping every call. |

Where `modern-python` differs: it turns docstring (`D`) rules off by default. Here they
are on with the numpy convention. Everything else in its defaults agrees with the table.

## pyproject.toml

Keep the whole configuration in `pyproject.toml`; no `ruff.toml`, `pytest.ini`,
`setup.cfg` or `.coveragerc`. One file means one place to review.

- Licence metadata follows PEP 639: `license = "BSD-3-Clause"` and
  `license-files = ["LICENSE.md"]`. Add no `License ::` classifiers.
- Dev tooling goes in `[dependency-groups]`, never in `[project.optional-dependencies]`.
  Optional extras are for users (for example an `audio` extra), not for developers.
  - `dev`: ruff, ty, prek, import-linter (layer contracts from the architecture plan
    §4, run by `task lint`), go-task-bin (the `task` command)
  - `test`: pytest, pytest-cov, pytest-xdist, hypothesis
  - `docs`: great-docs, a Jupyter kernel (`ipykernel`) for executing `.qmd` cells, and
    `quarto-cli` (pinned exactly; it fetches the Quarto binary great-docs runs)
  - `[tool.uv] default-groups = ["dev", "test"]` so `uv sync` gives a working dev
    environment. `task setup` uses `--all-groups`.
- Add every dependency with `uv add` or `uv add --group <g>` and commit `uv.lock` in
  the same commit. Say why in the commit message.

### Ruff ignores

Start from `select = ["ALL"]`. Each ignore below is a proposal with its reason; add a
new one only with a comment, and prefer a per-file ignore to a global one.

```toml
[tool.ruff.lint]
select = ["ALL"]
ignore = [
  "COM812", "ISC001",  # conflict with the ruff formatter
  "CPY001",            # no per-file copyright headers; LICENSE.md covers the package
  "D105",              # magic-method docstrings add nothing
  "D107",              # numpy style documents __init__ parameters on the class
  "N803", "N806",      # mathematical names such as X, Pa, ISOPl match the papers
  "PLR0913",           # scientific functions legitimately take many keyword arguments
  "TD002", "TD003",    # TODOs need no author or issue link while the rewrite is young
  "FIX002",            # allow TODO comments; review them before each release
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "PLR2004", "D", "ANN", "INP001"]  # asserts, literal expected values
# Standalone PEP 723 hook scripts, kept short and dependency-free.
".claude/hooks/**" = [
  "INP001", "T201", "S603", "S607", "D103", "PLW1510", "PTH109", "C901", "PLR0911",
]
"scripts/**" = ["INP001", "T201"]  # standalone PEP 723 generators that report
"docs/**/*.py" = ["INP001", "T201"]  # standalone user-facing scripts shown in the docs
```

Think before keeping `PLR2004` (magic numbers) on in package code. Physical constants
belong in named module constants with units in the name, so the rule stays on there.

The hook scripts read `subprocess.run` return codes (no `check=True`), keep `os.getcwd()`
and carry no docstrings, so they stay standalone. The `docs/**/*.py` scripts (such as
`docs/user_guide/_quickstart.py`) are what a user copies and runs; they print by design
and belong to no package.

`design/` is excluded from ruff (`[tool.ruff] extend-exclude`): it holds prose documents
whose code blocks are illustrative or normative text, not package code. The vendored
`.claude/skills/` and `.claude/agents/` are excluded too, because they are copied verbatim
(see `VENDORED.md`). `.claude/hooks/` stays linted.

### Other tool sections

- `[tool.ty]`: keep it minimal. Set the Python version and source roots only; verify
  key names against the ty docs (`/astral:ty`) because ty is pre-1.0.
- `[tool.pytest.ini_options]`: `minversion = "9.2"`, `testpaths = ["tests"]`,
  `addopts = ["--strict-markers", "--strict-config", "--import-mode=importlib", "-ra",
  "-m", "not reference and not network"]`, `xfail_strict = true`,
  `log_level = "INFO"` (captured logs shown on failure; repo-review PP304),
  `filterwarnings = ["error"]`. A later `-m` on the command line overrides the one in
  `addopts`, which is how `test:reference` and `test:network` select their markers
  (`tests/test_markers.py` proves it).
- A third-party warning that would break CI on an upstream release may get a scoped
  `ignore` entry with a comment; never a blanket ignore. Scope it by the soundscapy module
  that calls the library, or by matching the message. Never scope it by the third-party
  module: a warning raised with `stacklevel` 2 or more is attributed to the caller, so a
  filter naming the library's own module does not match. Every expected
  `SoundscapyWarning` is asserted with `pytest.warns`.
- `[tool.ruff.lint.flake8-type-checking]`: `runtime-evaluated-base-classes =
  ["pydantic.BaseModel", "soundscapy._model.FrozenModel"]` from the first commit, so
  ruff's `TC` rules never move an import a pydantic model (or a `FrozenModel` record)
  needs at runtime.
- `[tool.coverage.run]`: `branch = true`, `source = ["soundscapy"]`.
  `[tool.coverage.report]`: `fail_under = 80`, `show_missing = true`.

## Taskfile task names

Every task has a `desc`, every command runs through `uv run`, and `env: {MPLBACKEND: Agg}`
is set globally. These names are referenced by AGENTS.md, the Stop hook and CI, so do not
rename them. Document and invoke them as `uv run task <name>` (the Stop hook uses that form,
and bare `task` only when `uv` is not on PATH).

| Task | Runs |
| --- | --- |
| `setup` | `uv sync --all-groups` then `uv run prek install` |
| `fmt` | `ruff format` then `ruff check --fix` |
| `lint` | `ruff format --check`, `ruff check` (no writes), then `uv run lint-imports` |
| `typecheck` | `uv run ty check` |
| `test` | `uv run pytest {{.CLI_ARGS}}` |
| `test:cov` | pytest with coverage and the 80% gate |
| `check:fast` | `lint` + `typecheck`; must finish in a few seconds (Stop hook) |
| `check` | `lint` + `typecheck` + `test:cov`: the definition of done |
| `test:reference` | pytest `-m reference`: cross-implementation checks (R `sn` fixtures, ISO 532-1 signals, ArtemiS); nightly in CI and on demand |
| `test:network` | pytest `-m network`: tests that download a dataset; nightly in CI and on demand |
| `docs` | `uv run --group docs great-docs build`, which also executes the tutorials; offline, so the paper pages stay frozen |
| `docs:papers` | `uv run --group docs great-docs freeze docs/papers/2022-how-to-analyse.qmd`: re-executes the frozen paper pages (needs the network; nightly in CI). See `reference.md` §4 for why `freeze` suffices |
| `docs:preview` | great-docs local preview |
| `build` | `uv build` |
| `repo-review` | `uvx 'sp-repo-review[cli]' .` |

## prek configuration

- Hooks: ruff (`ruff-check --fix`, `ruff-format`) from `astral-sh/ruff-pre-commit`;
  the standard hygiene hooks (trailing whitespace, end of file, check-toml, check-yaml,
  large files, merge conflicts) from prek's `repo: builtin`, which needs no download;
  `uv lock` as a `repo = "local"`, `language = "system"` hook; `actionlint` for the CI workflows from
  `Mateusz-Grzelinski/actionlint-py`, a Python package whose build fetches the
  actionlint binary, so no Go toolchain or Docker is needed. prek caches it after the
  first run.
- The hygiene hooks that rewrite files exclude the vendored `.claude/skills/` and
  `.claude/agents/` (copied verbatim) and `docs/_freeze/` (written by Quarto).
- Write `prek.toml` as plain TOML 1.0 (`[[repos.hooks]]` tables, not the multiline inline
  tables `prek util yaml-to-toml` emits), so Python's `tomllib` can read it.
- `uv lock` runs the project's own uv, never `astral-sh/uv-pre-commit`. That hook installs
  `uv==<rev>` in its own environment, a second uv version beside `[tool.uv]
  required-version`, and nothing keeps the two in step. `required-version` in
  `pyproject.toml` is the one place the uv version lives.
- `sp-repo-review` runs at `stages = ["manual"]`. It is advisory, not a commit blocker.
- ty is **not** a prek hook. It needs the project environment and is slower than a
  commit hook should be; it runs in `task check:fast` (the Stop hook) and in CI.
- Pin every `rev` to a released tag and update with `prek auto-update`.

## Docs layout

- `docs/` holds the great-docs site. The API reference is generated from docstrings,
  so a public object gets its documentation by writing its docstring well. Use the
  `revise-docstrings` skill for docstring work.
- Tutorials are plain-text Quarto `.qmd` documents with executable `{python}` cells in
  `docs/tutorials/`, a great-docs custom section; paper reproductions (PRD FR-9.3) are
  `.qmd` documents in `docs/papers/`, a second custom section. How-to and explanation
  pages live in `docs/user_guide/`. No notebooks (`.ipynb` or marimo): `.qmd` diffs and reviews like
  code. Cells execute during `task docs`; Quarto's freeze cache (`docs/_freeze/`,
  committed) skips unchanged pages. `reference.md` has the layout and cell options.
  Paper reproductions need the full ISD download, so they stay frozen during `task docs`
  (which must run offline) and are re-executed nightly in CI or on demand.
- `design/` holds the PRD, the architecture plan, specs (`design/specs/`), plans
  (`design/plans/`) and ADRs (`design/adr/`). It is never part of the docs build.
  Superpowers defaults to `docs/superpowers/`; always write its output to `design/`
  instead so planning artefacts cannot leak into the published site.

## Tests

- Mirror the package layout under `tests/` for unit tests. Three further directories
  carry registered markers (`--strict-markers` requires them in `pyproject.toml`):
  `tests/science/` (literature fixtures, default run), `tests/reference/` (marker
  `reference`, cross-implementation checks, nightly) and `tests/regression/` (marker
  `regression`, golden pipelines against committed baselines, default run). A fourth
  marker, `network`, marks any test that needs a download (a dataset fetch); it is
  deselected by default and runs nightly and in `task test:network`.
  `tests/templates/` holds the test templates the extension recipes copy. It is created
  at M2 with the first extension recipe (FR-13.1), not before. Use `hypothesis` for
  numerical invariants (bounds of circumplex coordinates, symmetry, monotonicity).
- Give every float comparison an explicit tolerance set by the significant figures
  of its reference value, with a comment citing the source
  (`pytest.approx(x, rel=5e-4)  # reference given to 4 s.f., ISO 532-1 Annex B`).
  Drift within that precision, from libraries, platforms or reasonable algorithm
  changes, is not a regression (PRD NFR-4).
- Pin published reference values from the cited papers and standards as regression
  tests, citing the source in the test docstring.

## Scientific Python conformance

`task repo-review` runs sp-repo-review against the Scientific Python development
guide. Treat its findings as suggestions to evaluate against this file. Apply a finding
when it fits; when it does not, add its check id to `[tool.repo-review] ignore` in
`pyproject.toml` with a trailing comment giving the reason, and record the same reason
below so the next agent does not re-litigate it. `task repo-review` must exit 0.

Recorded ignores (check id: reason):

- PY007: the task runner is `Taskfile.yml` (go-task), which repo-review does not recognise.
- GH200: no Dependabot. SHA-pinned Actions are bumped by hand in reviewed commits (resolve the new SHA with `git ls-remote`).
- PY006: hooks are configured in `prek.toml`, which repo-review does not read (it looks for
  `.pre-commit-config.yaml`). Its `PC` checks are skipped for the same reason, so they need
  no ignores.
- RF002: ruff `target-version = "py311"` is set explicitly beside `requires-python` (Python floor row).
- SEC001: workflows are linted with actionlint; zizmor is not adopted in Phase 0.
