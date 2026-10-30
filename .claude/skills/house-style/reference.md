# Reference: great-docs and executable .qmd tutorials

Condensed from the great-docs sources listed under "Sources". The vendored skills
`great-docs`, `write-user-guide` and `revise-docstrings` cover more; defer to them for
anything not here. Items marked "(verify)" were inferred, not read in the sources.

## 1. great-docs

Requirements: Python 3.11+ and a separate Quarto install. Install great-docs into the
`docs` dependency group (`uv add --group docs great-docs`).

### Init, build, preview

| Command | Effect |
| --- | --- |
| `great-docs init` | Run once. Detects package (pyproject/setup.cfg/src layout), docstring style, public API; writes `docs/great-docs.yml`; adds `docs/_quarto/` and `docs/_site/` to `.gitignore`. Refuses to overwrite; `--force` resets, `--project-path DIR`, `--config PATH` for a non-`docs/` layout. |
| `great-docs build` | The only day-to-day/CI command. Generates a Quarto project in `docs/_quarto/default/`, renders, assembles the deployable site in `docs/_site/`. Flags: `--watch`, `--no-refresh` (skip API re-discovery), `--jobs N|auto` (parallel Quarto, experimental), `--versions a,b`, `--latest-only`, `--from-repo URL [--branch] [--shallow] [--output-dir] [--preview]`, `--config PATH`. |
| `great-docs preview` | Local server with live reload, port 3000 (`--port N`); `--site-dir DIR` serves a prebuilt site; `--pr N` / `--run ID` / `--branch NAME` fetch a CI artifact (needs `GITHUB_TOKEN` or `--use-gh`). |
| `great-docs scan [--verbose]` | Lists discovered exports and marks `[x]` those present in the `reference` config. |
| `great-docs config` | Writes a fully commented config template (every key at its default). |
| `great-docs lint` / `proofread` / `check-examples` / `check-links` / `seo` | Quality checks (missing docstrings + broken xrefs; spelling; run docstring/page examples; links in built site; sitemap/meta). |
| `great-docs freeze [pages]` | Execute selected pages and persist their `_freeze/` cache. |
| `great-docs setup-github-pages` | Writes a `.github/workflows/` file that builds and deploys on push to `main`. |

### Resulting layout (commit vs ignore)

```
docs/
  great-docs.yml      # config (commit)
  user_guide/         # narrative .qmd sources (commit)
  assets/             # docs-only images, logo (commit)
  tutorials/          # tutorial .qmd pages, a custom section (commit)
  _freeze/            # cached execution results (commit)
  _quarto/default/    # generated Quarto project: _quarto.yml, index.qmd, reference/, user-guide/ (ignore)
  _site/              # deployable HTML + llms.txt, llms-full.txt, skill.md (ignore)
README.md, pyproject.toml, <package>/   # stay at project root; README becomes index.qmd
```

Paths inside `great-docs.yml` (assets, bibliography, scripts) resolve relative to `docs/`; package
metadata and sources are relative to the project root. Never edit files under `docs/_quarto/`.

### Config keys that matter (`docs/great-docs.yml`)

| Key | Default | Notes |
| --- | --- | --- |
| `module` | `null` (auto) | Importable name when it differs from the project name. |
| `display_name` | `null` | Navbar/title name. |
| `parser` | `numpy` | `numpy` / `google` / `sphinx`. Auto-detected by `init`. |
| `dynamic` | `true` | Runtime introspection; `false` = static only. |
| `jupyter` | `python3` | Kernel for executable `{python}` cells, incl. docstring examples. |
| `exclude` / `auto_include` / `no_auto_exclude` | `[]` / `[]` / `false` | Names dropped from discovery (`init`, `scan`) and never documented. Built-in auto-excludes: `main, cli, version, VERSION, core, utils, helpers, logger, log`. |
| `reference` | `[]` (auto by type) | Explicit API sections: list of `{title, desc, contents: [name | {name, members: false}]}`. A listed submodule expands to all its public members. Names must match exports exactly; a typo silently yields an empty page. |
| `inline_methods` | `5` | Methods inline on the class page up to N, split to own pages above; `true`/`false` force. |
| `ref_section_order` | `[]` | Order of `python`, `cli`, `mcp` tabs in Reference. |
| `user_guide` | `null` (= `docs/user_guide/`) | A dir name, or an explicit list of `{section, contents: [file.qmd | {text, href}]}` (paths relative to the guide dir). |
| `sections` | `[]` | Extra navbar sections: `{title, dir, index: true, index_columns, navbar_after, type: blog}`. Accepts `.qmd` or `.md` pages. |
| `homepage` | `index` | `user_guide` makes the first guide page the landing page. |
| `site` | `theme: flatly, toc: true, toc-depth: 2, html-math-method: katex, lang: en` | Passed to Quarto. |
| `navbar_style`, `content_style`, `accent_color`, `dark_mode_toggle`, `logo: {light, dark, show_title}`, `favicon` | theming | See `configure-site` skill. |
| `interlinks.sources` | `{}` | `numpy: {url: https://numpy.org/doc/stable/, aliases: [np]}`; reads `<url>/objects.inv`. `interlinks.add_function_parentheses: true`. |
| `source: {enabled, branch, path}` | enabled | "View source" links to GitHub. |
| `repo`, `site_url`, `pypi`, `github_style` | auto | Repo URL override, canonical site URL, PyPI link, `widget`/`icon`. |
| `freeze: {mode: auto, pre_render: []}`, `pre_render` | auto | Quarto freeze; commit `docs/_freeze/`. |
| `bibliography`, `csl` | `[]` | Project-wide `.bib`; cite with `[@key]` on any page or docstring. |
| `markdown_pages: {enabled, widget}` | true | `.md` companion per HTML page. |
| `skill: {enabled, file, skills, gotchas, best_practices, ...}` | enabled | Emits `skill.md` / SKILL.md for agents. |
| `cli: {enabled, module, name}`, `mcp: {...}`, `versions`, `changelog`, `authors`, `announcement`, `page_status`, `tags`, `seo` | — | Not needed for a first site. |

Output dir is not a config key: it is always `<config dir>/_site/` (`docs/_site/`, or
`website/_site/` with `--config website/great-docs.yml`).

### Adding a user-guide page

1. Create `docs/user_guide/NN-slug.qmd` (numeric prefix orders the sidebar and is stripped from URL/title).
2. Frontmatter: `title: "..."` and `guide-section: "Getting Started"`; pages sharing a `guide-section` are grouped; section order follows the first file in each.
3. Alternative: subdirectories, each with an `index.qmd` whose title names the section; root `index.qmd` lands the "User Guide" navbar link.
4. Subdirs without `.qmd` (e.g. `images/`, `data/`) are copied as assets.
5. Links: other guide pages `[x](installation.qmd)`, API pages `[X](../reference/X.qmd)`, both validated at build. Executable cells: ```` ```{python} ```` with `#| echo: false`, `#| eval: false`, `#| output: false`, `#| fig-cap:`.
6. Quarto callouts `::: {.callout-note}`, tabsets `::: {.panel-tabset}` work as-is.

### How API reference pages are generated

- Discovery uses griffe static analysis (no import) for `init`/`scan`; `llms.txt` and dynamic mode import the package, so install it in CI before `build`.
- Public = no leading underscore; `__all__` is used when present. 13 object types (class, dataclass, ABC, Protocol, Enum, exception, NamedTuple, TypedDict, function, async function, constant, type alias, other); each gets a badge and, by default, a section.
- Output: `reference/index.qmd` plus one `reference/<Name>.qmd` per item; large classes get per-method pages (see `inline_methods`).
- Exclude an object: (a) `exclude:` list in config, (b) `%nodoc` on its own line in the docstring (per-item, lives with the code; still importable), (c) omit it from an explicit `reference:` config, (d) `members: false` on a class entry to drop its methods.

### NumPy docstring conventions great-docs renders

- Sections recognised: summary line (first line, shown in index tables), extended description,
  `Parameters`, `Returns`, `Raises`, `Notes`, `Examples`, `See Also`, `Warnings`, `References`.
  Any custom underlined heading (e.g. `Thresholds\n----------`) becomes its own heading.
- Parameters: `name` on its own line (type comes from annotations; do not repeat it), description indented 4 spaces. `Returns`: type line then indented description.
- Markdown works inside docstrings: tables, callouts `::: {.callout-warning}`, lists, links. RST roles (`:py:exc:`) render as literal text under `parser: numpy`; `**Examples**::` pseudo-headings render as plain text (use real headings).
- `Examples`: executable ```` ```{python} ```` fences run at build and embed output; use a hidden setup cell (`#| echo: false`, `#| output: false`) for shared imports; keep examples self-contained. Non-executed: plain ```` ```python ````.
- Cross-references:
  - `See Also` section: `other_func : Description.` (NumPy form) — merged and de-duplicated with `%seealso`.
  - `%seealso a, b` or `%seealso a : desc, b : desc` on its own line; multiple lines merge; dotted `Class.method` allowed; write bare names (parentheses auto-added for callables).
  - Inline interlinks: `` [](`~pkg.Obj`) `` shows `Obj`; `` [](`pkg.Obj`) `` shows the full path; `` [text](`pkg.Obj`) `` custom text; targets without `()`. Short names (`` [](`Obj`) ``) work if unambiguous. External: `` [](`numpy.ndarray`) `` / `` [](`np.ndarray`) `` via `interlinks.sources`.
  - Code autolinks: bare `` `my_func` `` / `` `my_func()` `` linking to a documented symbol automatically; suppress with `` `name`{.gd-no-link} ``.
- Other `%` directives (stripped from output): `%nodoc`; `%versionadded 2.0`, `%versionchanged 3.1`, `%deprecated 2.6 Use x instead.`; callouts `%note`, `%warning`, `%caution`, `%danger`, `%important`, `%tip`, `%hint` with inline or indented body.
- Citations: `.. [1]` in `References` + `[1]_` in prose render as linked numbered citations; `[@key]` with `bibliography:`.
- Document constants, module vars, type aliases, TypeVars, properties and modules the same way (summary + sections); lead with intent, don't restate the annotation.

### llms.txt

Generated automatically in build step 3 (after reference config refresh, before `skill.md`), no config key. `llms.txt` = compact index of API items with one-line descriptions and links; `llms-full.txt` = full signatures, complete docstrings, CLI help, user-guide text with frontmatter stripped. Both land in `docs/_site/`. Requires the API reference block in the config (the guide page says `api-reference`; the current key is `reference` — verify) and an importable package at build time. Check with `head -30 docs/_site/llms.txt`.

## 2. Tutorials as executable .qmd pages

Decision for this repo: tutorials are plain-text Quarto `.qmd` documents with
executable `{python}` cells, not notebooks (`.ipynb` or marimo). They diff cleanly,
review like code, and run wherever the package is installed.

- Tutorials live in `docs/tutorials/NN-slug.qmd`, registered in `docs/great-docs.yml` as
  a custom section: `sections: [{title: Tutorials, dir: tutorials, index: true}]`.
  How-to guides and explanation pages live in `docs/user_guide/` (FR-12.3's four-part
  structure: tutorials, how-to, reference from docstrings, explanation).
- Code cells are ```` ```{python} ```` fences. They run in the `jupyter` kernel named in
  `great-docs.yml` (default `python3`) during `great-docs build`, so the package and the
  tutorial's dependencies must be installed in the docs environment (`uv sync --group
  docs`).
- Cell options go on `#|` lines at the top of a cell: `#| echo: false`,
  `#| eval: false`, `#| output: false`, `#| fig-cap: "..."`, `#| label: fig-name`.
  Use one hidden setup cell for shared imports.
- Freeze: keep `freeze: {mode: auto}` and commit `docs/_freeze/`, so unchanged pages are
  not re-executed on every build and CI does not need audio data or model weights for
  pages whose source has not changed. Re-run an expensive page with `great-docs freeze
  <page>`.
- Data: tutorials use the bundled ISD subset (PRD FR-10.1), so they run offline and in
  CI. Put tutorial-only files beside the page in a non-`.qmd` subdirectory, which is
  copied as assets.
- A tutorial is done when `great-docs build` renders it with every cell executed,
  without errors and without warnings that the page does not explain.
- Quarto authoring details (callouts, tabsets, cross-references, figures) are in the
  vendored `quarto-authoring` skill.

## 3. Commands cheat-sheet

| Task | Command |
| --- | --- |
| Create docs config (once) | `great-docs init` |
| Build site → `docs/_site/` | `great-docs build` (`--no-refresh`, `--watch`, `--jobs auto`) |
| Live preview | `great-docs preview [--port 3000]` |
| What will be documented | `great-docs scan --verbose` |
| Docstring/xref/link checks | `great-docs lint`, `great-docs check-examples`, `great-docs check-links`, `great-docs proofread` |
| Full commented config | `great-docs config` |
| GitHub Pages workflow | `great-docs setup-github-pages` |
| Cache expensive pages | `great-docs freeze <pages>`; commit `docs/_freeze/` |
| Render one page while writing it | `quarto preview docs/tutorials/<page>.qmd` (verify inside a great-docs project) |


## Sources

From https://github.com/posit-dev/great-docs (great-docs 0.17.0, commit recorded in
`.claude/skills/VENDORED.md`):

- `README.md` (quickstart, config overview, CLI table, feature list)
- `great_docs/assets/great-docs.default.yml` (all config keys and defaults)
- `great_docs/cli.py` (`init`, `build`, `preview` options and help)
- `great_docs/config.py` (`reference` key) and `great_docs/core.py` (render globs, freeze, llms.txt)
- `docs/user_guide/02-quickstart.qmd`, `04-writing-docstrings.qmd`, `06-api-documentation.qmd`,
  `08-user-guides.qmd`, `09-custom-sections.qmd`, `12-cross-referencing.qmd`, `13-building.qmd`,
  `39-llms-txt.qmd`
