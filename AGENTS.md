# AGENTS.md

Instructions for any coding agent working in this repository. CLAUDE.md imports this
file. Keep it short; detailed procedures live in skills under `.claude/skills/`.

## Project

`soundscapy` is being rewritten from scratch on the `soundscapy-rewrite` branch. It is an
open-source scientific Python package for soundscape engineering: ISO 12913 survey
analysis, psychoacoustic analysis of calibrated recordings, predictive modelling of
soundscape perception, and scoring and design with the Soundscape Perception Index
(SPI). It must work as a library, in teaching notebooks, and as the engine of a GUI
for acoustic engineers, architects and urban designers.

`design/PRD.md` states the requirements. `design/greenfield_architecture_plan.md` is the
draft architecture. Read both before designing anything, and check every design
decision against the PRD. Nothing from the previous soundscapy codebase carries over.

**Status:** only the agent setup exists. There is no `pyproject.toml`, `Taskfile.yml`,
package, test suite or docs site yet, so the `task` commands below do not exist. The
first build step is the toolchain, set up as the `house-style` skill describes.

## Repo map (target layout)

- `design/`: PRD, architecture plan, specs (`design/specs/`), implementation plans
  (`design/plans/`) and ADRs (`design/adr/`). Never part of the published docs.
- `src/soundscapy/`: the package (src layout).
- `tests/`: the pytest suite.
- `docs/`: the great-docs site (`docs/great-docs.yml`). API reference comes from
  docstrings; tutorials are marimo notebooks in `docs/notebooks/`.
- `.claude/`: agent settings, hooks and skills. Vendored skills are listed in
  `.claude/skills/VENDORED.md`.

## Commands

Use `task` for everything; run `task --list` for the entry points. Key tasks: `setup`,
`fmt`, `lint`, `typecheck`, `test`, `test:cov`, `notebooks`, `check:fast` (lint and
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

## Docs and notebooks

- API docs are generated from docstrings by great-docs.
- A tutorial is a marimo notebook, `docs/notebooks/<topic>.py`, embedded in a `.qmd`
  page with great-docs' native shortcode `{{< marimo file="notebooks/<topic>.py" >}}`.
  It runs in the reader's browser, so it must be WASM-compatible.
- Follow the `marimo-notebook` and `wasm-compatibility` skills.
- `marimo-pair` lets an agent work inside a running marimo kernel. While such a
  session is open on a notebook, never edit that notebook's file; change it through
  the live kernel instead.
- Run `task notebooks` after editing a notebook.

## Design workflow

- Any feature that spans more than one module or changes a public API goes through
  the superpowers plugin skills: `brainstorming`, then a spec, then `writing-plans`,
  then `executing-plans` or `subagent-driven-development`.
- Write specs to `design/specs/YYYY-MM-DD-<topic>.md`, plans to
  `design/plans/YYYY-MM-DD-<topic>.md` and decisions to `design/adr/NNNN-<title>.md`. This overrides superpowers' default of `docs/superpowers/`, which
  would leak into the published site.

## Review

- Run the code-review plugin (`/code-review`) before declaring a PR ready.
- PRs target `soundscapy-rewrite`.

## Don't

- Weaken lint or type rules to make errors go away. Fix the code, or add a narrowly
  scoped ignore with a comment giving the reason.
- Reformat files you did not otherwise change.
- Add a dependency without saying why.
