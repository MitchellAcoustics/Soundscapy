# Vendored skills

These skill directories are copied verbatim from upstream. Do not edit them;
project-specific overrides live in `house-style/`. To update, re-copy from the
upstream path at a newer commit and bump the row here.

| Skill directory | Upstream | Path in upstream | Commit (shallow clone, 2026-10-30) |
| --- | --- | --- | --- |
| marimo-notebook | https://github.com/marimo-team/skills | skills/marimo-notebook | 6454470960d3cd57151aaeffb1176dd55f598b18 |
| jupyter-to-marimo | https://github.com/marimo-team/skills | skills/jupyter-to-marimo | 6454470960d3cd57151aaeffb1176dd55f598b18 |
| anywidget-generator | https://github.com/marimo-team/skills | skills/anywidget (frontmatter name is anywidget-generator) | 6454470960d3cd57151aaeffb1176dd55f598b18 |
| wasm-compatibility | https://github.com/marimo-team/skills | skills/wasm-compatibility | 6454470960d3cd57151aaeffb1176dd55f598b18 |
| prek | https://github.com/j178/prek | skills/prek | 48c8a3ec465a8e98bb8b5b88c8bfacaca84cf84b |
| great-docs | https://github.com/posit-dev/great-docs | docs/skills/great-docs (great-docs 0.17.0) | 4232670877842d7c1489dbef72a076459e767229 |
| write-user-guide | https://github.com/posit-dev/great-docs | docs/skills/write-user-guide | 4232670877842d7c1489dbef72a076459e767229 |
| revise-docstrings | https://github.com/posit-dev/great-docs | docs/skills/revise-docstrings | 4232670877842d7c1489dbef72a076459e767229 |

Upstream install commands, for reference: `npx skills add marimo-team/skills --agent claude-code`,
`gh skill install j178/prek prek` (needs gh >= 2.90), `great-docs skill install great-docs --agent claude`.
All three write to `.claude/skills/<name>/`, which is what the copies above reproduce.
