# Reference: great-docs + marimo for a scientific Python package

Condensed from the vendor sources listed under "Sources". The full vendor skills are installed
alongside this one; defer to them for anything not covered here:
`great-docs`, `configure-site`, `revise-docstrings`, `write-user-guide`, `author-skills`
(posit-dev) and `marimo-notebook`, `wasm-compatibility`, `jupyter-to-marimo`, `anywidget-generator`
(marimo-team). Items marked "(verify)" are inferred, not read in the sources.

## 1. great-docs

Requirements: Python 3.11+ and a separate Quarto install. `pip install great-docs`
(`pip install "great-docs[marimo]"` adds marimo + uv for notebook embedding).

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
  notebooks/          # marimo .py notebooks, only when marimo: true (commit)
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
| `marimo: {enabled, version}` | `false` | See section 2. `marimo: true` is shorthand. |
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

## 2. marimo notebooks in a great-docs site

### (a) Native support — yes, verified

Evidence: `great_docs/_marimo.py`, `great_docs/assets/_extensions/marimo/marimo.lua` (Quarto
shortcode), `assets/marimo-islands.{js,css}`, `config.py` `marimo_enabled`/`marimo_version`,
`core.py` lines ~510-605, user guide `45-marimo-notebooks.qmd` (status: experimental), README
feature "Interactive Marimo notebooks". Neither repo mentions `quarto-marimo`.

How it works:
- `marimo: true` (or `marimo: {enabled: true, version: "0.23.7"}`) in `great-docs.yml`; `pip install "great-docs[marimo]"`.
- Notebooks must live in `docs/notebooks/*.py` (the build copies `<config dir>/notebooks` into the Quarto project). Build warns and skips if `marimo` is not installed.
- Build runs every notebook: `MarimoIslandGenerator` writes `_marimo_islands/<stem>.html` and `<stem>-nocode.html`; it also runs `python -m marimo export html-wasm <nb> -o notebooks/<stem>/ --mode edit` per notebook (needs `uv`) and injects an auto-run script.
- Embed in any `.qmd`: `{{< marimo file="notebooks/my-demo.py" >}}`. Options: `mode="island"|"iframe"`, `show-code="false"`, `show-copy="false"`, `theme="auto|light|dark"`, `height="600px"` (iframe), `chrome="trimmed|full"` (iframe).
- Island mode: cells inline, outputs/`mo.ui` live via Pyodide in the reader's browser, code read-only, one shared kernel per page, so at most one island notebook per page. Iframe mode: fully editable notebook, own kernel per embed.
- Constraints: runs in Pyodide (pure-Python or Pyodide-bundled packages only, 2 GB, no threads, ~15 MB first download); `import marimo as mo` in its own cell; data via `mo.notebook_location() / "public" / "file.csv"`; author with `marimo edit --sandbox`. Heavy scientific stacks that are not in Pyodide will fail in the browser.

### (b) quarto-marimo Quarto extension

Not present in either source tree and not referenced by great-docs. From general knowledge
(verify before relying on it): a marimo-team Quarto filter that executes ` ```{python} ` / marimo
cells inside `.qmd` during `quarto render`, with options to run cells in the browser via WASM.
Unknown here: current install command, option names, interaction with great-docs' generated
`_quarto.yml` (great-docs owns that file, so an extra filter would have to be wired through a
supported hook or `include_in_header`, which is not documented for filters). Treat as unsupported.

### (c) Export as a build step (marimo CLI)

Subcommands (`marimo export --help`): `html` (runs notebook, HTML file), `html-wasm`, `ipynb`,
`md` ("code fenced markdown file"), `pdf`, `script`, `session`, `thumbnail`. Common flags: `-o/--output`,
`--sandbox` (isolated uv env from PEP 723 header), `--watch`, `-f/--force`, `-y` (auto-yes; goes
before the subcommand: `marimo -y export ...`), `--` to pass args to the notebook. PDF-only flags
read in the sources: `--no-include-inputs`, `--no-include-outputs`, `--as=slides`. Whether
`md`/`ipynb` accept `--include-outputs` is not stated in the skill (verify with `--help`).

Pipeline that needs nothing beyond Quarto's defaults:

```bash
uvx marimo -y export md --sandbox notebooks/demo.py -o docs/examples/demo.md   # (verify md flags)
# or: uvx marimo -y export ipynb --sandbox notebooks/demo.py -o docs/examples/demo.ipynb
great-docs build
```

Custom `sections:` accept `.qmd` or `.md` pages, so an exported `.md` with a `title:` frontmatter
drops straight into e.g. `sections: [{title: Examples, dir: examples, index: true}]`. Quarto's
render globs include `*.ipynb`, so an exported notebook placed in a section dir should also render
(verify that great-docs copies `.ipynb` from section dirs; the docs only promise `.qmd`/`.md`).

### Decision for this repo: native great-docs integration

Use (a), great-docs' own marimo support. It is the integration great-docs and its Quarto
extension (`marimo.lua` shortcode) are built around, it needs no export step in `task docs`,
and readers get live, reactive tutorials instead of static pages.

- `docs/great-docs.yml` sets `marimo: {enabled: true, version: "<pinned marimo>"}`. Pin the
  version to the one in `uv.lock` so local and browser runtimes agree.
- Tutorials live in `docs/notebooks/*.py` (great-docs copies `<config dir>/notebooks`). Data
  files go in `docs/notebooks/public/` and are read with `mo.notebook_location() / "public"`.
- Each tutorial or guide page is a `.qmd` (user guide or a custom section) that embeds its
  notebook with the shortcode: island mode by default, `mode="iframe"` when readers should
  edit code or when a page shows more than one notebook.
- Author with `marimo edit --sandbox`, so PEP 723 metadata lists the packages the browser
  kernel installs. Run the `wasm-compatibility` skill on every tutorial.
- Constraint to design around: notebooks run in Pyodide. NumPy, SciPy, pandas, Polars,
  scikit-learn and Matplotlib work; compiled packages without a Pyodide or WASM wheel do not.
  If a tutorial needs one (likely candidates: audio I/O and psychoacoustic libraries), keep
  that notebook runnable locally, and record in an ADR how its page is handled.
- Do not use the quarto-marimo extension: great-docs owns the generated `_quarto.yml`, and
  the native shortcode is already the Quarto integration.

## 3. marimo notebook conventions (the dozen that matter)

See the `marimo-notebook` skill and its `references/*.md` for detail.

1. File skeleton: PEP 723 header, `import marimo`, `__generated_with = "<ver>"`, `app = marimo.App(width="medium")`, cells as `@app.cell def _(deps...): ...; return (defs,)`, and `if __name__ == "__main__": app.run()`.
2. Dependencies: `# /// script` block with `requires-python` and `dependencies = ["marimo", ...]`; `marimo edit --sandbox nb.py` writes it; `--sandbox` on `edit`/`run`/`export` uses it in an isolated uv env.
3. Every global is defined by exactly one cell (DAG built from cell parameters = reads, return tuple = writes). Reusing a name in two cells is an error.
4. Underscore-prefixed names (`_tmp`) are cell-local; use them for one or two temporaries only, not whole cells; prefer a new name.
5. Only the last top-level expression of a cell is displayed; indented/conditional expressions are not. Assign then put the name on the last line.
6. Don't mutate across cells (`.append()`, in-place DataFrame ops are invisible to the DAG); create new objects. Mutations inside the defining cell are fine.
7. Don't guard cells with `if x:` or wrap in try/except for control flow; cells only run when deps exist.
8. `mo.md("...")` for prose (f-strings for values), `mo.ui.*` for widgets (`slider`, `dropdown`, `table`, `form`, `tabs`, ...); read `.value` in a different cell than the one that creates the widget. `import marimo as mo` in its own cell.
9. Script-mode: `mo.app_meta().mode == "script"`; keep widgets always displayed and only swap the data source. `mo.stop(cond)` halts a cell; `mo.cache`/`mo.persistent_cache` for expensive steps.
10. Reusable code: `with app.setup:` (max one) for shared imports, `@app.function` cells holding a single function/class become importable from other modules.
11. Lint before handing back: `uvx marimo check nb.py` (verify-the-notebook-can-run linter for common mistakes; exact rule list not in the sources). Run headless with `uv run nb.py` or `python nb.py`.
12. Tests: add `pytest` to deps; cells containing only `test_*` functions / `Test*` classes / `@pytest.fixture` run via `pytest nb.py` (and reactively in the editor; disable with `runtime.reactive_test`). Fixtures must sit in the same cell as the tests or in `app.setup`; `conftest.py` works.
13. Project-level settings in `pyproject.toml` under `[tool.marimo.*]` (e.g. `[tool.marimo.formatting] line_length = 120`); precedence: PEP 723 block > pyproject > `~/.config/marimo/marimo.toml`.

## 4. Commands cheat-sheet

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
| New notebook | `uvx marimo edit --sandbox docs/notebooks/demo.py` |
| Run notebook headless | `uv run docs/notebooks/demo.py` |
| Lint notebook | `uvx marimo check docs/notebooks/demo.py` |
| Test notebook | `pytest docs/notebooks/demo.py` |
| Export for the site | `uvx marimo -y export md --sandbox nb.py -o docs/examples/nb.md` (or `ipynb`, `html`) |
| WASM self-check | `marimo export html-wasm nb.py -o test/` then `python -m http.server -d test/` |
| marimo API help | `uv --with marimo run python -c "import marimo as mo; help(mo.ui.form)"` |
| Show marimo config | `marimo config show` / `marimo config describe` |

## Sources

great-docs clone root: `/tmp/claude-0/-home-user-Soundscapy/b104c50b-28af-56d5-81b3-f48a1baeb90a/scratchpad/src/posit-dev_great-docs/`
- `README.md` (quickstart, config overview, CLI table, feature list)
- `great_docs/assets/great-docs.default.yml` (all config keys and defaults)
- `great_docs/cli.py` (`init`, `build`, `preview` options and help)
- `great_docs/config.py` (`marimo_enabled`, `marimo_version`, `reference` key)
- `great_docs/core.py` (notebook copy, island + html-wasm generation, render globs, llms.txt)
- `great_docs/_marimo.py`, `great_docs/assets/_extensions/marimo/marimo.lua`, `_extension.yml`
- `docs/user_guide/02-quickstart.qmd`, `04-writing-docstrings.qmd`, `06-api-documentation.qmd`, `08-user-guides.qmd`, `09-custom-sections.qmd`, `12-cross-referencing.qmd`, `13-building.qmd`, `39-llms-txt.qmd`, `45-marimo-notebooks.qmd`
- `docs/skills/{great-docs,configure-site,revise-docstrings,write-user-guide,author-skills}/SKILL.md` (names only)

marimo skills clone: `/tmp/claude-0/-home-user-Soundscapy/b104c50b-28af-56d5-81b3-f48a1baeb90a/scratchpad/src/marimo-team_skills/skills/marimo-notebook/`
- `SKILL.md`; `references/EXPORTS.md`, `PYTEST.md`, `CONFIGURATION.md`, `DEPLOYMENT.md`, `REACTIVITY.md`, `TOP-LEVEL-IMPORTS.md`, `UI.md`, `EXPENSIVE.md`, `STATE.md` (headings)
