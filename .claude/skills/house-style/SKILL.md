---
name: house-style
description: >
  Project-specific rules for soundscapy's tooling, packaging, tests, docs and
  notebooks. These rules override the vendor skills (modern-python, astral,
  prek, great-docs, marimo). Use this skill whenever you create or change
  pyproject.toml, uv.lock, ruff/ty/pytest/coverage settings, Taskfile.yml,
  .pre-commit-config.yaml, CI workflows, great-docs config, the docs site, a
  test file or a marimo notebook. Also use it when you set the toolchain up
  for the first time, or when a vendor skill's default seems to disagree with
  this repo.
---

# soundscapy house style

The vendor skills are general. This file records what is specific to this repo, and
it wins wherever the two disagree. Edit this file, never the vendored skills under
`.claude/skills/` (see `VENDORED.md`), so that updating upstream stays a clean copy.

For the great-docs and marimo details, read `reference.md` beside this file. Read it
when you configure the docs site, write a user guide, or touch a notebook.

## Decisions

| Setting | Value | Why |
| --- | --- | --- |
| Package / import name | `soundscapy` | Keeps the existing PyPI name and citations. |
| Layout | `src/soundscapy/`, tests in `tests/` | Stops tests importing the working tree by accident. |
| Build backend | `uv_build` | One toolchain. No hatchling, setuptools or VCS versioning. |
| Python floor | `requires-python = ">=3.11"`; ruff `target-version = "py311"`; ty `python-version = "3.11"` | Product-owner decision. It is deliberately wider than the SPEC 0 window, so do not raise it to match SPEC 0 or a vendor default. Test every supported version in CI. |
| Type checker | `ty` | Astral toolchain. Never add mypy, pyright, basedpyright or pyrefly, nor their LSP plugins. |
| Lint / format | `ruff`, `select = ["ALL"]` with the commented ignores below | Maximum signal; every exemption is visible and justified. |
| Docstrings | numpy convention | Matches the scientific Python ecosystem and great-docs rendering. |
| Git hooks | `prek`, reading `.pre-commit-config.yaml` | Fast drop-in for pre-commit. Never install `pre-commit` itself. |
| Task runner | `Taskfile.yml` (go-task, `version: '3'`) | One set of entry points for agents, humans and CI. |
| Coverage gate | `fail_under = 80`, branch coverage on | Matches the Trail of Bits default; raise it, never lower it. |
| CI | GitHub Actions running `task` targets | CI runs exactly what developers run. |
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
  - `dev`: ruff, ty, prek
  - `test`: pytest, pytest-cov, hypothesis
  - `docs`: great-docs, marimo
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
"docs/notebooks/**" = ["D", "INP001", "ANN", "B018", "PLR2004", "T201"]  # marimo cell style
".claude/hooks/**" = ["INP001", "T201", "S603", "S607"]  # standalone PEP 723 scripts
```

Think before keeping `PLR2004` (magic numbers) on in package code. Physical constants
belong in named module constants with units in the name, so the rule stays on there.

### Other tool sections

- `[tool.ty]`: keep it minimal. Set the Python version and source roots only; verify
  key names against the ty docs (`/astral:ty`) because ty is pre-1.0.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = ["--strict-markers",
  "--strict-config", "--import-mode=importlib", "-ra"]`, `xfail_strict = true`,
  `filterwarnings = ["error"]`.
- `[tool.coverage.run]`: `branch = true`, `source = ["soundscapy"]`.
  `[tool.coverage.report]`: `fail_under = 80`, `show_missing = true`.

## Taskfile task names

Every task has a `desc`. These names are referenced by AGENTS.md, the Stop hook and
CI, so do not rename them.

| Task | Runs |
| --- | --- |
| `setup` | `uv sync --all-groups` then `uv run prek install` |
| `fmt` | `ruff format` then `ruff check --fix` |
| `lint` | `ruff format --check` then `ruff check` (no writes) |
| `typecheck` | `uv run ty check` |
| `test` | `uv run pytest {{.CLI_ARGS}}` |
| `test:cov` | pytest with coverage and the 80% gate |
| `notebooks` | `uv run marimo check docs/notebooks/` |
| `check:fast` | `lint` + `typecheck`; must finish in a few seconds (Stop hook) |
| `check` | `lint` + `typecheck` + `test:cov` + `notebooks`: the definition of done |
| `docs` | `uv run great-docs build` |
| `docs:preview` | great-docs local preview |
| `build` | `uv build` |
| `repo-review` | `uvx sp-repo-review[cli] .` |

## prek configuration

- Hooks: ruff (`ruff-check --fix`, `ruff-format`) from `astral-sh/ruff-pre-commit`;
  the standard hygiene hooks (trailing whitespace, end of file, check-toml, check-yaml,
  large files, merge conflicts); `uv-lock` from `astral-sh/uv-pre-commit`.
- `sp-repo-review` runs at `stages: [manual]`. It is advisory, not a commit blocker.
- ty is **not** a prek hook. It needs the project environment and is slower than a
  commit hook should be; it runs in `task check:fast` (the Stop hook) and in CI.
- Pin every `rev` to a released tag and update with `prek auto-update`.

## Docs layout

- `docs/` holds the great-docs site. The API reference is generated from docstrings,
  so a public object gets its documentation by writing its docstring well. Use the
  `revise-docstrings` skill for docstring work.
- Tutorials and user guides are marimo notebooks in `docs/notebooks/*.py`, embedded into
  `.qmd` pages with great-docs' native marimo shortcode (`marimo: true` in
  `docs/great-docs.yml`). They run live in the reader's browser via Pyodide, so check each
  one with the `wasm-compatibility` skill. There is no export step. `reference.md` has the
  config, shortcode options and the Pyodide constraints.
- `design/` holds the PRD, the architecture plan, specs (`design/specs/`), plans
  (`design/plans/`) and ADRs (`design/adr/`). It is never part of the docs build.
  Superpowers defaults to `docs/superpowers/`; always write its output to `design/`
  instead so planning artefacts cannot leak into the published site.

## Tests

- Mirror the package layout under `tests/`. Use `hypothesis` for numerical invariants
  (bounds of circumplex coordinates, symmetry, monotonicity).
- Give every float comparison an explicit tolerance with a comment saying where it
  comes from (`pytest.approx(x, rel=1e-6)  # ISO 532-1 reference table precision`).
- Pin published reference values from the cited papers and standards as regression
  tests, citing the source in the test docstring.

## Scientific Python conformance

`task repo-review` runs sp-repo-review against the Scientific Python development
guide. Treat its findings as suggestions to evaluate against this file. Apply a finding
when it fits; when it does not, record the reason in this skill so the next agent does
not re-litigate it.
