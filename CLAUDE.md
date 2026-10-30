@AGENTS.md

## Claude-specific notes

- Hooks enforce the house rules: after every `Edit`/`Write`, `ruff format` and
  `ruff check --fix` run on that file (`marimo check` for a marimo notebook). On
  Stop, `task check:fast` must pass. Both do nothing until the toolchain exists.
- Reach for these skills and plugins when the task fits:
  - `house-style`: any change to config, packaging, tests, docs or notebooks. It
    overrides the vendor defaults below.
  - `modern-python` (Trail of Bits) and `/astral:uv`, `/astral:ruff`, `/astral:ty`:
    Python tooling questions. The ty LSP is provided by the `astral` plugin.
  - `prek`: git hooks. `great-docs`, `write-user-guide`, `revise-docstrings`: docs.
    The great-docs skills cover its marimo shortcode; `wasm-compatibility` checks
    that a tutorial will run in the browser.
  - `marimo-notebook`, `jupyter-to-marimo`, `anywidget-generator`,
    `wasm-compatibility`: notebooks. `marimo-pair`: live-kernel editing.
  - `quarto-authoring`: `.qmd` pages. `describe-design`: architecture write-ups.
  - `superpowers`: brainstorming, specs, plans, TDD, systematic debugging.
  - `/code-review`, `/review-pr`, `critical-code-reviewer`, `review-testing`: review.
  - `pr-create`, `pr-threads-address`: pull requests (confirm before pushing).
  - `skill-creator`: creating or improving project skills.
- `git push` is denied by the project permissions. The user pushes, or approves a push.
