# Contributing to soundscapy

soundscapy is being rewritten on the `soundscapy-rewrite` branch. Pull requests target that
branch.

## Set up

Install two tools: [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[git](https://git-scm.com/downloads). Nothing else is needed.

```bash
git clone https://github.com/MitchellAcoustics/Soundscapy.git
cd Soundscapy
git switch soundscapy-rewrite
uv run task setup
```

`uv run task setup` installs every dependency group and the git hooks (prek). The
`dev` group brings `go-task-bin`, which provides the `task` command, pinned by
`uv.lock`; the `docs` group brings `quarto-cli`, which fetches Quarto 1.10.20. You do not
install go-task or Quarto yourself.

Run every task as `uv run task <name>`. A bare `task <name>` works only while the project's
virtual environment is activated. `uv run task --list` shows every task.

## Definition of done

**`uv run task check` passes.** It checks formatting, lint rules and import contracts,
type-checks with ty, and runs the tests with the coverage gate. CI runs the same command
on Windows, macOS and Linux with Python 3.11 to 3.14.

`uv run task fmt` formats the code and applies the automatic lint fixes.

## Tests

| What | Command |
| --- | --- |
| Default suite: unit, science (literature fixtures) and regression (golden pipelines) tests | `uv run task test` |
| The same with a coverage report | `uv run task test:cov` |
| Only the science tests | `uv run task test -- -m science` |
| Only the regression tests | `uv run task test -- -m regression` |
| Reference tests: slow or external cross-implementation runs | `uv run task test:reference` |
| Network tests: tests that download a dataset | `uv run task test:network` |

The default run deselects the `reference` and `network` markers. The nightly workflow runs
them on pushes to `soundscapy-rewrite` that touch the nightly paths (the workflow itself, the
quick-start scripts, `docs/papers/`, and network or reference test files), and on a schedule
once the rewrite is the default branch.

## Docs

`uv run task docs` builds the site offline into `docs/_site/` and executes every tutorial;
`uv run task docs:preview` serves it locally. The paper reproductions in `docs/papers/`
stay frozen during `task docs`; `uv run task docs:papers` re-executes them and needs the
network. Commit `docs/_freeze/` with any page whose output changed.

If the docs build fails on Windows or macOS because the Quarto that `quarto-cli` fetches
does not run, install [Quarto 1.10.20](https://quarto.org/docs/download/) for your system
and try again. CI builds the docs on Linux only.

## Design workflow

A feature that spans more than one module or changes a public API starts with a spec and
a plan under `design/`. `AGENTS.md` describes the workflow, the code conventions and the
review steps; they apply to human contributors as well as coding agents.

## Changes that need the owner's review

Changes under `src/soundscapy/_data/`, and changes that move a regression baseline, change a
science fixture or bump a method version, affect scientific results. They need Andrew
Mitchell's approval and carry the label `science`. See `GOVERNANCE.md`.
