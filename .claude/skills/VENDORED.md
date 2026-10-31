# Vendored skills

These skill directories are copied verbatim from upstream. Do not edit them;
project-specific overrides live in `house-style/`. To update, re-copy from the
upstream path at a newer commit and bump the row here.

| Skill directory | Upstream | Path in upstream | Commit |
| --- | --- | --- | --- |
| prek | https://github.com/j178/prek | skills/prek | 48c8a3ec465a8e98bb8b5b88c8bfacaca84cf84b |
| great-docs | https://github.com/posit-dev/great-docs | docs/skills/great-docs (main, merge of PR #487; the docs group's pin) | af17c76d6ab2c775e438c6804c3b1b6feceef57e |
| write-user-guide | https://github.com/posit-dev/great-docs | docs/skills/write-user-guide | af17c76d6ab2c775e438c6804c3b1b6feceef57e |
| revise-docstrings | https://github.com/posit-dev/great-docs | docs/skills/revise-docstrings | af17c76d6ab2c775e438c6804c3b1b6feceef57e |
| quarto-authoring | https://github.com/posit-dev/skills | quarto/quarto-authoring | d614ac4fc7f8862bba5e041ab6237113e02aa86a |
| review-testing | https://github.com/posit-dev/skills | posit-dev/review-testing | d614ac4fc7f8862bba5e041ab6237113e02aa86a |
| describe-design | https://github.com/posit-dev/skills | posit-dev/describe-design | d614ac4fc7f8862bba5e041ab6237113e02aa86a |

The prek and posit-dev/skills rows were copied from shallow clones on 2026-10-30. The
three great-docs rows were re-copied on 2026-10-31 at the commit the `docs` group pins in
`pyproject.toml` (`[tool.uv.sources]`), so the skills describe the great-docs that `uv.lock`
installs; keep them at that pin when it moves. Their files did not change between
`4232670` (the previous copy) and `af17c76`.

## Vendored agents (`.claude/agents/`)

Copied from https://github.com/anthropics/claude-plugins-official, path
`plugins/pr-review-toolkit/agents/`, commit f8c5e7a1dcb71b10cb74586224b5401f3de5db28
(Apache-2.0). The instructions are verbatim. The `description` field was shortened
and made on-request: the upstream text runs to ~460 tokens of examples and asks to be
used proactively, which overlaps the built-in `/code-review`.

| Agent | Upstream file |
| --- | --- |
| silent-failure-hunter | agents/silent-failure-hunter.md |
| type-design-analyzer | agents/type-design-analyzer.md |

Licences: posit-dev/skills and great-docs are MIT; claude-plugins-official is
Apache-2.0; j178/prek is MIT.

Upstream install commands, for reference:
`gh skill install j178/prek prek` (needs gh >= 2.90), `great-docs skill install great-docs --agent claude`.
All three write to `.claude/skills/<name>/`, which is what the copies above reproduce.
