# Agent setup report

Date: 2026-10-30. Branch: `chore/agent-setup`, based on `soundscapy-rewrite`.

This records how the repository was configured for coding agents, measured against the
agent-setup prompt. The scope was limited by the product owner to the Claude Code
skills, plugins and instructions. The Python toolchain, docs site, editor settings and
CI were deliberately left for the first build step.

## Branches

| Branch | Contents |
| --- | --- |
| `soundscapy-rewrite` | Root of the rewrite and target of every PR. One commit on top of `main`'s planning history clears the old package, keeping the licence, the PRD and the architecture plan (moved to `design/`). |
| `chore/agent-setup` | The agent setup, one commit per phase, ready to merge into `soundscapy-rewrite`. |

## What was installed

### Plugins (project scope, `.claude/settings.json`)

All the plugins (ten at the time, nine since marimo-pair was removed) were installed at project scope in a scratch copy of the repository with an
isolated home directory, to prove that the marketplace names and plugin ids resolve.

| Plugin | Marketplace (`name` from its `marketplace.json`) | Source repo | Version / commit | Skills | Always-on tokens |
| --- | --- | --- | --- | --- | --- |
| modern-python | trailofbits | trailofbits/skills | 1.6.2, `2be6433` | 1 | ~77 |
| astral (uv, ruff, ty skills, ty LSP) | astral-sh | astral-sh/claude-code-plugins | 0.1.0, `f3ce88a` | 3 | ~154 |
| posit-dev | posit-dev-skills | posit-dev/skills | `d614ac4` | 6 | ~641 |
| github | posit-dev-skills | posit-dev/skills | `d614ac4` | 3 | ~315 |
| quarto | posit-dev-skills | posit-dev/skills | `d614ac4` | 3 | ~449 |
| code-review | claude-plugins-official | anthropics/claude-plugins-official | 1.0.1, `f8c5e7a` | 1 | ~22 |
| pr-review-toolkit | claude-plugins-official | anthropics/claude-plugins-official | 1.0.1 | 1 + 6 agents | ~2,035 |
| superpowers | claude-plugins-official | obra/superpowers (pinned by the official catalogue) | 7.0.0 | 15 | ~848, plus ~1,000 injected by its SessionStart hook |
| skill-creator | claude-plugins-official | anthropics/claude-plugins-official | — | 1 | ~114 |

`pyright-lsp@claude-plugins-official` is set to `false`. No pyright, pyrefly or
basedpyright language server is enabled; the only Python LSP is astral's ty.

### Standalone skills (`.claude/skills/`, committed)

| Skill | Source | Why copied rather than installed by its CLI |
| --- | --- | --- |
| prek | j178/prek | `gh skill` needs gh 2.90 or later; this environment has 2.89. |
| great-docs, write-user-guide, revise-docstrings | posit-dev/great-docs (0.17.0) | `great-docs skill install` needs great-docs installed, which is part of the deferred toolchain. |
| house-style | written for this project | Project overrides of the vendor skills, with `reference.md`. |

Upstream commits are recorded in `.claude/skills/VENDORED.md`.

### Security review of everything installed

Every SKILL.md, hook and bundled script was read before installation. Nothing writes
outside the project except as noted, and nothing changes git config, shell startup files
or system settings.

- **modern-python** adds a SessionStart hook that prepends shims to PATH for the session
  only. The shims refuse bare `python <script>`, `pip`, `pipx` and `uv pip`, and print the
  uv equivalent. Kept, as the prompt asked. Any future script that shells out to bare
  `python` will fail inside agent sessions, which is the intent.
- **superpowers** injects about 1,000 tokens of strongly worded skill-use instructions at
  every start, `/clear` and compaction. It defers to CLAUDE.md and AGENTS.md, which is how
  the `design/` location overrides its default of `docs/superpowers/`.
- **astral**'s ty LSP runs `uvx ty@latest server`, so the editor-side ty version floats
  and may differ from the project's pinned ty.
- **posit github** helpers install the third-party gh extension `agynio/gh-pr-review` the
  first time they are used. `pr-create` pushes branches after confirmation; in local
  sessions the push guard hook blocks that until the user pushes.
- **skill-creator**'s eval scripts spawn `claude -p`, briefly write a command file into
  `.claude/commands/`, and the review server kills whatever holds port 3117.

## Hooks and permissions

- **PostToolUse** on `Edit|Write|MultiEdit` runs `.claude/hooks/post_edit.py`. On a `.py`
  file it runs `ruff format` then `ruff check --fix` on that file only. Unfixable
  problems exit 2 so the agent sees them.
- **Stop** runs `.claude/hooks/stop_check.py`, which runs `task check:fast`, exits 2 on
  failure, and exits 0 at once when `stop_hook_active` is set.
- Both hooks do nothing until `pyproject.toml` and `Taskfile.yml` exist, so they are safe
  today. Both were tested against a scratch project: an unfixable error blocked with exit
  2, auto-fixable errors were fixed, and the loop guard worked.
- The hooks call only `uv run`, which the modern-python shims pass through, so the two do
  not conflict.
- Permissions allow `task`, `uv run/sync/add/remove/lock/build/tree`, `uvx`, `prek`,
  `great-docs`, `quarto` and read-only git. They deny reading `.env*`, `secrets/`,
  `*.pem` and `*.key`, and bare `pip`.
- **PreToolUse** on `Bash` runs `.claude/hooks/guard_push.py`, which denies `git push`
  in local sessions and allows it in cloud sessions (`CLAUDE_CODE_REMOTE=true`).
  Permission rules cannot be scoped by session type, so a hook does it. This replaced
  a blanket `Bash(git push:*)` deny on 2026-10-30.

## Verification

| Check | Result |
| --- | --- |
| Plugins resolve and install at project scope | All installed and enabled in an isolated scratch copy. |
| No stray Python LSPs | Only ty, from astral. |
| Hooks behave as specified | Passed all cases above. |
| A fresh agent given only AGENTS.md answers the four questions | All four answered correctly. Its list of ambiguities led to one follow-up commit. |
| `task setup`, `task check`, `task docs`, `task build`, `prek run --all-files` | Not run: the toolchain is out of scope for this pass. |
| `task repo-review` | Not run, for the same reason. There are no repo-review follow-ups yet. |

### Skill count

The prompt targets 8–12 active skills. This setup has **39**: 34 from plugins and 5 in
`.claude/skills/`. The always-on cost is about 4,900 tokens of descriptions plus the
superpowers injection. The bulk comes from three sources, so the trims below are
proposals for the product owner, not applied.

1. **pr-review-toolkit** costs the most (~2,000 tokens, mostly its six agent
   descriptions) and overlaps `code-review` and posit's `critical-code-reviewer`. Keeping
   one review stack would save the most.
2. **superpowers** has 15 skills. `diagnosing-superpowers` is for reporting bugs to its
   maintainers, and `writing-skills` overlaps skill-creator. Plugin skills cannot be
   switched off one at a time; the documented options are a `Skill(...)` deny rule, which
   the docs do not confirm works for plugin skills, or dropping the plugin.
3. **quarto** brings `brand-yml` and `alt-text` along with `quarto-authoring`. **posit-dev**
   brings `implement`, `new-work` and `working-on`, but those three are user-invoked only
   and cost almost nothing until called.

## Deviations from the prompt

- **Scope.** Phases 1 and 2 (toolchain and docs initialisation) and the editor and CI
  parts of Phase 5 were not done, at the product owner's request. house-style records the
  decisions they will follow.
- **Approval gate.** The Phase 0 plan was not presented for approval before work began.
  This report and the follow-up review stand in for it.
- **Branching.** The prompt asked for `chore/agent-setup`. The product owner also asked
  for a `soundscapy-rewrite` root branch, so the setup branch is based on it.
- **Docs integration.** Tutorials are executable Quarto `.qmd` documents in
  `docs/tutorials/`. marimo was tried first (exported, then native great-docs
  integration) and dropped on 2026-10-30, with the marimo skills and the marimo-pair
  plugin removed, so the in-browser (Pyodide) constraints no longer apply.
- **Python floor.** Set to `>=3.11` by the product owner, wider than the SPEC 0 window
  the prompt suggested.
- **house-style evals.** Drafted with skill-creator's guidance, but its test-and-review
  loop was skipped because a draft was requested.
- **great-docs skills.** Three of its five were copied. `configure-site` (theming) and
  `author-skills` (publishing skills from a docs site) were left out to save context.
- **Account-synced plugins.** `data@synced` and `cowork-plugin-management@synced` are set
  to `false` in the project settings. They come from the claude.ai account, and whether a
  project setting overrides them is untested. Turning them off on claude.ai is the
  reliable fix.

## Open decisions

1. **Optional extras.** Add the Taskfile skills from `sunsided/taskfile-skills`, or
   Honnibal's `pre-mortem` command? Neither was installed.
2. **Skill trim.** Which, if any, of the trims above to apply.
3. **Push rule.** Resolved: agents push in cloud sessions, and not in local ones.

## Updating the vendor plugins and skills

- Plugins: `claude plugin marketplace update <marketplace>`, then
  `claude plugin update <plugin@marketplace>`, or use `/plugin`. Third-party marketplaces
  do not auto-update by default. Re-audit hooks and scripts after an update.
- Vendored skills: re-copy each folder from the upstream path listed in
  `.claude/skills/VENDORED.md` at a newer commit, review the diff, and update the commit
  in that table. Never edit vendored skills in place; put overrides in `house-style`.

## Cloud sessions

Claude Code cloud sessions do not install the plugins that `.claude/settings.json`
enables, by design (see the cloud environments page of the Claude Code docs). Only
plugins pre-installed in the cloud image load: code-review, pr-review-toolkit and
skill-creator. `.claude/cloud/setup.sh` installs the full set at user scope. Paste it
into the cloud environment's Setup script field (environment menu in the session title
bar, then Edit). It took about 23 seconds in testing, inside the five-minute limit for
caching. Keep its plugin list in step with `enabledPlugins`.

The account-synced `github` plugin's MCP server (`api.githubcopilot.com`) is blocked by
the default cloud network policy. Cloud sessions already have a working built-in GitHub
connection, so it is redundant there.
