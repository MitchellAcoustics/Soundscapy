# AGENTS.md

Instructions for any coding agent working in this repository. CLAUDE.md imports this
file. Keep it short; detailed procedures live in skills under `.claude/skills/`.

## Project

`soundscapy` is being rewritten from scratch on the `soundscapy-rewrite` branch. It is an
open-source scientific Python package for soundscape engineering: ISO 12913 survey
analysis, psychoacoustic analysis of calibrated recordings, predictive modelling of
soundscape perception, and scoring and design with the Soundscape Perception Index
(SPI). It must work as a library, in teaching material, and as the engine of a GUI
for acoustic engineers, architects and urban designers.

`design/PRD.md` (draft 0.6) states the requirements; read it before designing
anything, and check every design decision against it.
`design/greenfield_architecture_plan.md` (draft 0.2) is the architecture: it was
written from the PRD and its §15 registers the owner's decisions. Specs and plans are
derived from both. Nothing from the previous soundscapy codebase carries over.

**Status:** only the agent setup exists. There is no `pyproject.toml`, `Taskfile.yml`,
package, test suite or docs site yet, so the `task` commands below do not exist. The
first build step is the toolchain, set up as the `house-style` skill describes.

## Repo map (target layout)

- `design/`: PRD, architecture plan, specs (`design/specs/`), implementation plans
  (`design/plans/`), ADRs (`design/adr/`) and document reviews (`design/reviews/`).
  Never part of the published docs.
- `src/soundscapy/`: the package (src layout).
- `tests/`: the pytest suite.
- `docs/`: the great-docs site (`docs/great-docs.yml`). API reference comes from
  docstrings; tutorials are `.qmd` documents in `docs/tutorials/`.
- `.claude/`: agent settings, hooks and skills. Vendored skills are listed in
  `.claude/skills/VENDORED.md`.

## Commands

Use `task` for everything; run `task --list` for the entry points. Key tasks: `setup`,
`fmt`, `lint`, `typecheck`, `test`, `test:cov`, `check:fast` (lint and
typecheck, run by the Stop hook), `check`, `docs`, `docs:preview`, `build`,
`repo-review`.

**Done means `task check` passes.**

## Environment

- uv only. Never run bare `pip` or `python`; use `uv run` and `uvx`.
- Runtime dependency: `uv add <pkg>`. Developer-only dependency: `uv add --group
  <group> <pkg>`, where the group is `dev` (tooling), `test` or `docs`.
- Commit `uv.lock` with every dependency change, and give the reason in the commit
  message.
- Type checker: `ty`. Linter and formatter: `ruff`. Git hooks: `prek`.

## Code conventions

- Public APIs are fully typed and pass `ty`.
- Every public object has a numpy-style docstring.
- No hidden global state.
- Units and physical quantities are explicit in names, signatures or docstrings.
- No silent dtype or unit coercion.
- Fail loudly with specific exceptions.

## Testing

- Tests accompany every behaviour change.
- Use property-based tests (`hypothesis`) for numerical invariants.
- Numerical tolerances are explicit and justified in a comment.

## Docs and tutorials

- API docs are generated from docstrings by great-docs.
- A tutorial is a plain-text Quarto document, `docs/tutorials/NN-<topic>.qmd`, with
  executable `{python}` cells. No `.ipynb` or marimo notebooks.
- `task docs` builds the site and executes every tutorial; a tutorial is done when it
  renders with every cell run and no errors. Commit `docs/_freeze/` with it.
- Tutorials use the bundled ISD subset so they run offline and in CI.

## Design workflow

- Any feature that spans more than one module or changes a public API goes through
  the superpowers plugin skills: `brainstorming`, then a spec, then `writing-plans`,
  then `executing-plans` or `subagent-driven-development`.
- Write specs to `design/specs/YYYY-MM-DD-<topic>.md`, plans to
  `design/plans/YYYY-MM-DD-<topic>.md` and decisions to `design/adr/NNNN-<title>.md`. This overrides superpowers' default of `docs/superpowers/`, which
  would leak into the published site.

## Review

Finish each unit of work (a feature, a fix, or one milestone of a plan) in this order:

1. Make it work, with tests; `task check` passes.
2. Run a simplify pass (`/simplify` in Claude Code) on the change, then run
   `task check` again. Simplify is the refactor step, so it needs passing tests
   first, and it is not worth running on every commit while code is still moving.
3. Run `/code-review` and address the findings, so review sees the final code. If
   that leads to substantial changes, run one more simplify pass on just those.

Before declaring a PR ready, both passes must have run on its final changes.
PRs target `soundscapy-rewrite`.

## Don't

- Weaken lint or type rules to make errors go away. Fix the code, or add a narrowly
  scoped ignore with a comment giving the reason.
- Reformat files you did not otherwise change.
- Add a dependency without saying why.
