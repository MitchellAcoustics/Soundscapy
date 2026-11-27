@AGENTS.md

## Claude-specific notes

- Hooks enforce the house rules: after every `Edit`/`Write`, `ruff check --fix` and then
  `ruff format` run on that file. On Stop, `task check:fast` must pass. Both
  do nothing until the toolchain exists. Every hook starts with
  `uv run --no-config --script`, because the scripts are standalone and take nothing from
  the project's uv settings; without the flag, a uv older than `required-version` refused
  every command. The push guard skips uv entirely in cloud sessions.
- Reach for these skills and plugins when the task fits:
  - `house-style`: any change to config, packaging, tests, docs or tutorials. It
    overrides the vendor defaults below.
  - `modern-python` (Trail of Bits) and `/astral:uv`, `/astral:ruff`, `/astral:ty`:
    Python tooling questions. The ty LSP is provided by the `astral` plugin.
  - `prek`: git hooks. `great-docs`, `write-user-guide`, `revise-docstrings`: docs.
  - `quarto-authoring`: tutorial and guide `.qmd` pages. `describe-design`:
    architecture write-ups.
  - `superpowers`: brainstorming, specs, plans, TDD, systematic debugging.
  - `/simplify` then `/code-review` at the end of each unit of work (see AGENTS.md).
    Both are built into Claude Code. For deeper review: `review-testing` (test
    quality), and the `silent-failure-hunter` and `type-design-analyzer` agents, which
    run only when asked.
  - `skill-creator` (from the claude.ai account): creating or improving project
    skills. If a local session lacks it, enable
    `skill-creator@claude-plugins-official` in `.claude/settings.local.json`.
- Pushing: in cloud sessions agents push their own branches. In local sessions a
  hook blocks `git push`, and the user pushes or approves the push.
- Cloud sessions do not install the plugins above on their own; the environment's
  setup script must contain `.claude/cloud/setup.sh`.
