# Agent setup report

Date: 2026-10-30. Branch: `soundscapy-rewrite`.

This records how the repository was configured for coding agents, measured against the
agent-setup prompt. The scope was limited by the product owner to the Claude Code
skills, plugins and instructions. The Python toolchain, docs site, editor settings and
CI were deliberately left for the first build step.

## Changes since the first setup

| Change | Reason |
| --- | --- |
| marimo skills and the marimo-pair plugin removed | Tutorials are executable `.qmd` documents, not notebooks. |
| Blanket `git push` deny replaced by a local-only hook | Cloud agents push their own branches; local sessions leave pushing to the user. |
| Cloud setup script added | Cloud sessions do not install repo-declared plugins. |
| code-review, pr-review-toolkit and skill-creator plugins switched off | The built-in `/code-review`, `/simplify` and the account's skill-creator replace them; two pr-review-toolkit agents are vendored. |
| posit-dev, github and quarto plugins removed | Only quarto-authoring, review-testing and describe-design were wanted; they are vendored. |

## What is installed

### Plugins (`.claude/settings.json`)

| Plugin | Marketplace | Source repo | Version / commit | Skills | Always-on tokens |
| --- | --- | --- | --- | --- | --- |
| modern-python | trailofbits | trailofbits/skills | 1.6.2, `2be6433` | 1 | ~77 |
| astral (uv, ruff, ty skills, ty LSP) | astral-sh | astral-sh/claude-code-plugins | 0.1.0, `f3ce88a` | 3 | ~154 |
| superpowers | claude-plugins-official | obra/superpowers (pinned by the official catalogue) | 7.0.0 | 15 | ~848, plus ~1,000 injected by its SessionStart hook |

Explicitly switched off: `pyright-lsp`, `code-review`, `pr-review-toolkit` and
`skill-creator` from claude-plugins-official, and the account-synced `data` and
`cowork-plugin-management`. A project `false` was confirmed to switch off an
account-synced plugin. For the plugins pre-installed in the cloud image it is
expected to work the same way; confirm in the next fresh cloud session that
`/code-review` resolves to the built-in command. The only Python LSP is astral's ty.

All plugins in the original set were installed at project scope in a scratch copy of
the repository with an isolated home directory, to prove that the marketplace names and
plugin ids resolve.

### Vendored skills and agents (committed)

| Item | Source | Why vendored |
| --- | --- | --- |
| prek | j178/prek | `gh skill` needs gh 2.90 or later; this environment has 2.89. |
| great-docs, write-user-guide, revise-docstrings | posit-dev/great-docs (0.17.0) | `great-docs skill install` needs great-docs installed, which is part of the deferred toolchain. |
| quarto-authoring, review-testing, describe-design | posit-dev/skills | Taken from plugins whose other skills were not wanted. |
| silent-failure-hunter, type-design-analyzer (agents) | claude-plugins-official, pr-review-toolkit | The two distinctive pr-review-toolkit agents, with shortened, on-request descriptions. |
| house-style | written for this project | Project overrides of the vendor skills, with `reference.md`. |

Upstream paths, commits and licences are recorded in `.claude/skills/VENDORED.md`.

### Built-in commands the workflow relies on

`/simplify` and `/code-review` are built into Claude Code and add nothing to the skill
list. The account's `skill-creator` (from claude.ai) is used for project skills. If a
local session does not show it, enable `skill-creator@claude-plugins-official` in
`.claude/settings.local.json`.

### Security review

Every SKILL.md, agent, hook and bundled script was read before installation. Nothing
writes outside the project except as noted, and nothing changes git config, shell
startup files or system settings.

- **modern-python** adds a SessionStart hook that prepends shims to PATH for the session
  only. The shims refuse bare `python <script>`, `pip`, `pipx` and `uv pip`, and print
  the uv equivalent. Any future script that shells out to bare `python` will fail inside
  agent sessions, which is the intent.
- **superpowers** injects about 1,000 tokens of strongly worded skill-use instructions at
  every start, `/clear` and compaction. It defers to CLAUDE.md and AGENTS.md, which is
  how the `design/` location overrides its default of `docs/superpowers/`.
- **astral**'s ty LSP runs `uvx ty@latest server`, so the editor-side ty version floats
  and may differ from the project's pinned ty.
- The vendored skills and agents contain instructions only, with no scripts or hooks.

## Hooks and permissions

- **PostToolUse** on `Edit|Write|MultiEdit` runs `.claude/hooks/post_edit.py`. On a `.py`
  file it runs `ruff format` then `ruff check --fix` on that file only. Unfixable
  problems exit 2 so the agent sees them.
- **Stop** runs `.claude/hooks/stop_check.py`, which runs `task check:fast`, exits 2 on
  failure, and exits 0 at once when `stop_hook_active` is set.
- **PreToolUse** on `Bash` runs `.claude/hooks/guard_push.py`, which denies `git push`
  in local sessions and allows it in cloud sessions (`CLAUDE_CODE_REMOTE=true`).
  Permission rules cannot be scoped by session type, so a hook does it.
- The edit and Stop hooks do nothing until `pyproject.toml` and `Taskfile.yml` exist.
  All three were tested: an unfixable error blocked with exit 2, auto-fixable errors were
  fixed, the loop guard worked, and every `git push` form tried was denied locally and
  allowed remotely.
- The hooks call only `uv run`, which the modern-python shims pass through, so the two
  do not conflict.
- Permissions allow `task`, `uv run/sync/add/remove/lock/build/tree`, `uvx`, `prek`,
  `great-docs`, `quarto` and read-only git. They deny reading `.env*`, `secrets/`,
  `*.pem` and `*.key`, and bare `pip`.

## Verification

| Check | Result |
| --- | --- |
| Plugins resolve and install | All installed and enabled in an isolated scratch copy, and again in a cloud container by the setup script. |
| No stray Python LSPs | Only ty, from astral. |
| Hooks behave as specified | Passed all cases above. |
| A fresh agent given only AGENTS.md answers the four questions | All four answered correctly. Its list of ambiguities led to one follow-up commit. |
| `task setup`, `task check`, `task docs`, `task build`, `prek run --all-files` | Not run: the toolchain is out of scope for this pass. |
| `task repo-review` | Not run, for the same reason. There are no repo-review follow-ups yet. |

### Skill count

The prompt targets 8–12 active skills. The count matters less than overlap: only
descriptions are always in context, and the real cost of many skills is Claude choosing
between near-duplicates. The trim on 2026-10-30 removed every duplicate.

| | Skills | Agents | Always-on tokens |
| --- | --- | --- | --- |
| Before the trim | 39 | 6 | ~6,000, including the superpowers hook |
| After the trim | 27: 19 from plugins, 8 vendored | 2 | ~2,000 plus the ~1,000-token hook |

Superpowers (15 skills) is kept whole by decision. Copying in only the seven skills
AGENTS.md relies on, without its hook, remains an option if it over-triggers.

## Deviations from the prompt

- **Scope.** Phases 1 and 2 (toolchain and docs initialisation) and the editor and CI
  parts of Phase 5 were not done, at the product owner's request. house-style records
  the decisions they will follow.
- **Approval gate.** The Phase 0 plan was not presented for approval before work began.
  This report and the follow-up review stand in for it.
- **Branching.** The prompt asked for `chore/agent-setup`. It was used for the setup and
  then fast-forwarded into `soundscapy-rewrite`, the root branch the owner asked for.
- **Docs integration.** Tutorials are executable Quarto `.qmd` documents in
  `docs/tutorials/`. marimo was tried first and dropped on 2026-10-30.
- **Python floor.** Set to `>=3.11` by the product owner, wider than the SPEC 0 window
  the prompt suggested.
- **house-style evals.** Drafted with skill-creator's guidance, but its test-and-review
  loop was skipped because a draft was requested.
- **great-docs skills.** Three of its five were copied. `configure-site` (theming) and
  `author-skills` (publishing skills from a docs site) were left out to save context.
- **Plugin set.** The prompt's list (posit plugins, code-review, pr-review-toolkit,
  skill-creator) was trimmed to remove overlap, as recorded above.

## Open decisions

1. **Optional extras.** Add the Taskfile skills from `sunsided/taskfile-skills`, or
   Honnibal's `pre-mortem` command? Neither was installed.

## Updating the vendor plugins and skills

- Plugins: `claude plugin marketplace update <marketplace>`, then
  `claude plugin update <plugin@marketplace>`, or use `/plugin`. Third-party marketplaces
  do not auto-update by default. Re-audit hooks and scripts after an update.
- Vendored skills and agents: re-copy each from the upstream path listed in
  `.claude/skills/VENDORED.md` at a newer commit, review the diff, and update the commit
  in that table. Never edit vendored skills in place; put overrides in `house-style`.

## Cloud sessions

Claude Code cloud sessions do not install the plugins that `.claude/settings.json`
enables, by design (see the cloud environments page of the Claude Code docs). The cloud
image pre-installs code-review, pr-review-toolkit and skill-creator, which the project
settings switch off. `.claude/cloud/setup.sh` installs modern-python, astral and
superpowers at user scope. Paste it into the cloud environment's Setup script field
(environment menu in the session title bar, then Edit), and keep its plugin list in
step with `enabledPlugins`.

The account-synced `github` plugin's MCP server (`api.githubcopilot.com`) is blocked by
the default cloud network policy. Cloud sessions already have a working built-in GitHub
connection, so it is redundant there.
