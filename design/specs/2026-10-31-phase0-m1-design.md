# Phase 0 and M1 design specification

## 1. Status

**Spec draft 0.5, 2026-10-31.** Derived from PRD draft 0.9 (`design/PRD.md`) and architecture
plan draft 0.5 (`design/greenfield_architecture_plan.md`). It covers Phase 0 (the toolchain, CI,
docs skeleton and governance files) and M1 (survey analysis: journey A, and journey I for the 2022
paper). Draft 0.2 applied the consistency check and the builder check of draft 0.1, the owner's
great-docs pin, and the ISD record facts read from the Zenodo API on 2026-10-31. Draft 0.3 applies
the owner's answers of 2026-10-31 (§4.3), the SATP v1.5 tables of the 2026 chapter, and the builder
and consistency checks of draft 0.2. Draft 0.4 rebases the spec on PRD 0.9 (its §10.5b, FR-1.5,
FR-2.3 and FR-3.1), applies the owner's second round of answers of 2026-10-31, and the third
consistency and builder checks. After it, no owner input blocks M1 (§10). Draft 0.5 applies the
S1 plan review (2026-10-31), marked "Amended at S1 plan review" where it lands: §6.2.1, §6.2.3,
§6.3.5, §6.3.6, §6.3.8, §6.3.10, §6.4.1, §6.9.1, §6.19, §6.23, §7.1, §9 (S3) and §11.2.

The sources to cite are in `design/literature/` (its `README.md` lists them):
Mitchell, Aletta and Kang 2022 (JASA Express Letters), Aletta et al. 2024 (Applied Acoustics) and
the notes on chapter 4 of *Soundscape Across Languages* (Mitchell et al. 2026), which transcribe
its Tables 4.7 and 4.8 from the author proof. The chapter itself is cited by reference; its PDF
is not committed (owner decision, §4.3).

Where the plan leaves a choice to the spec (plan §0, "Left to the spec" under each §4 subsection,
and the M1 row of §7), this document is authoritative for Phase 0 and M1. Where this spec and the
plan disagree, the plan is updated in the same change (plan §0). The plan edits this spec carries
are listed in §11.1. Nothing here specifies M2 to M5, except constraints M1 code must not preclude.

Code blocks marked **Normative** are exact for M1: names, keys and values must match. Unmarked
blocks are illustrative. "The builder" is whoever writes the implementation plan from this spec.

## 2. Intent

M1 delivers the first release of the rewritten `soundscapy`: survey analysis that a soundscape
researcher, a course participant or a consultant can run from a survey export to Part 3 statistics
and the three Part 3 figures, with the correct angles per language, every number traceable to a
saved configuration. It is published as `1.0.0.dev1`, a pre-release that pip skips by default, so
0.x users are unaffected and the course pilot can `uv add` it.

**Why.** Journey A's hardest step today is getting an arbitrary export into shape (PRD §4 A). The
0.x package centred each response at its own midpoint, dropped rows by position, and scored WHO-5
with missing items as zero (§6.24 lists these bugs). M1 fixes these at the root: a mapping as the
schema, raw data kept, one status per response, and pure functions that record their settings.

**Success** is three things together:

1. Journey A runs end to end on the bundled ISD subset in the library, the CLI and tutorial A, and
   reopening a project and reading its saved runs gives identical numbers (PRD journey A "done
   when"; §6.3.4 defines re-running).
2. The 2022 reproduction document renders from ISD v0.2.3, the version the paper cites, with every
   cell run (PRD journey I, FR-9.3).
3. A participant reaches a first circumplex plot from nothing within an hour by the documented
   local path (FR-12.1), shown by a CI job and one owner-led dry run.

## 3. Goals, non-goals and anti-goals

### 3.1 Goals

| Goal | PRD | Plan |
|---|---|---|
| Toolchain, CI on 3 OS and Python 3.11 to 3.14, docs skeleton, governance files | NFR-1, NFR-2, NFR-8, NFR-12 | §7 Phase 0 |
| A generic circumplex instrument with the outdoor ISO instance shipped, labelled in every language Aletta et al. 2024 covers | FR-2.0, NFR-10 | §4.2 |
| Mapping, intake from CSV and Excel, suggestion from known schemas (SSID, REDCap, Qualtrics), templates | FR-1.1, FR-1.2, FR-1.3, FR-1.3a | §4.4 |
| One per-response status with quality rules reproducing the 0.x tutorials' processing of ISD v1.0 | FR-1.5, FR-2.5a | §4.5 |
| Projection: canonical, 2019 and Aletta 2024 formulations; equal, Aletta 2024 and SATP v1.5 angle sets with tiers | FR-2.1, FR-2.3, FR-2.4, FR-2.4a | §4.2 |
| Profiles and recorded settings, shareable as TOML | FR-2.9, FR-11.1, FR-11.3 | §4.6 |
| Distribution geometry: describe, regions, KDE, fraction contours, N < 30 rule | FR-2.5, FR-3.1, FR-3.2, FR-8.4 | §4.9 |
| Part 3 statistics: item median, IQR and range; correlations; group comparisons; WHO-5; source profiles | FR-2.2, FR-2.6, FR-2.6a, FR-2.6c | §4.4 |
| Plots: scatter, density, fraction contour, joint, Likert, radar | FR-8.1 (M1 part), FR-8.2, FR-8.3 | §4.9 |
| Export of tables to CSV, Excel and parquet; figures to PNG, SVG, PDF | FR-9.1, FR-14.2 | §4.11 |
| A minimal project directory, journey A's CLI | FR-1.7 (M1 part), FR-14.1, FR-14.4, FR-14.5 (A) | §4.11, §6 |
| ISD loader with bundled subset, integrity check, licence display, cache; the bundled course dataset | FR-10.1 | §5 |
| Provenance on every result, cross-platform reproducibility | FR-11.1, FR-11.2, NFR-4 | §4.7, §4.8 |
| Quick-start, tutorial A, the 2022 reproduction | FR-12.1, FR-12.3, FR-12.4, FR-9.3 | §6, §7 |

### 3.2 Non-goals

Nothing for M2 or later is built early (plan rule 9, §2.1). Explicitly excluded from M1:

| Excluded | Arrives | Why it is excluded now |
|---|---|---|
| Skew-normal, KS distance, SPI, targets, shift vector (FR-3.3, FR-3.4, FR-6) | M2 | Journeys C and E |
| Shift arrows, target overlays and SPI annotation in plots (rest of FR-8.1) | M2 | Need targets |
| `design`, briefs, scenarios, `evaluate` | M2 | Journey E |
| Feature-name vocabulary and the ISD feature table (plan §4.13) | M2 | Commitment 13 binds from M2. M1 carries ISD feature columns verbatim with units marked unverified |
| Plugin discovery, entry points, recipes (FR-13.1, FR-13.3) | M2 | PRD §9 |
| Versioning and deprecation policy (FR-13.4) | M2 | PRD §9 |
| Axis-space fingerprint, `Coordinates` concat, cross-axis override | M2 | Plan §8 |
| Renormalisation over present items, ipsatisation, clustering (FR-2.5a Should, FR-2.4b, FR-2.7) | When scheduled | Shoulds |
| Audio, batch runner, progress and cancellation protocols | M3 | Plan §2.1. tqdm itself arrives at M1 for pooch's download bar (§6.19) |
| Predictors and prediction labels beyond the `origin.kind` discriminator | M4 | Plan §4.3 |
| Seeds and `rng` arguments | First random function (M2) | Commitment 15 binds, but M1 has no random function |
| A stateful `Project` class, JSON schemas, `load()` on every record, a `df.ssp` accessor | GUI or later | Plan §2.1 |
| A location-to-country table for course case 2 | M2 (course) | Not in the ISD; needs provenance |
| `validate`, `demo` and `export` CLI commands | Not planned | §6.17 |
| A translated interface (messages, docs) | Not planned for this release | NFR-10: the interface is English; attribute labels are translated (§6.2.3) |
| Test templates (`tests/templates/`) | M2 | FR-13.1 recipes copy them; M1 has no recipe |

### 3.3 Anti-goals

M1 fails, even if every test passes, if any of these is true:

- A user's dataframe is mutated, reordered or renamed, or a derived column overwrites a user column.
- A row disappears from any result without a reason in its status.
- A number in a table or figure cannot be traced to its settings, data hash and package version.
- A wrong-scale export (0..4 declared as 1..5) produces coordinates without an error.
- The default run of `task check` or `task docs` needs the network.
- Importing `soundscapy` prints, logs or warns.
- The core imports matplotlib, or any module outside `soundscapy._frames` imports pandas, polars or
  pyarrow.
- A course participant on Windows cannot install with uv and plot within the hour.
- Code is written for a milestone after M1 "while we are here".
- A shipped data file is edited in place after release instead of gaining a new version.

## 4. Constraints

### 4.1 PRD Musts in M1

FR-1.1, FR-1.2, FR-1.3, FR-1.3a, FR-1.4 (survey levels only; recordings join at M3), FR-1.5, FR-1.7
(minimal directory), FR-2.0, FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-2.4a, FR-2.5, FR-2.5a (Must part),
FR-2.6, FR-2.6a, FR-2.6c, FR-2.9, FR-3.1, FR-3.2, FR-8.1 (scatter, KDE with decile contours, 50%
contour, group overlays, marginals), FR-8.2, FR-8.3, FR-8.4, FR-9.1, FR-9.3 (2022 paper), FR-10.1,
FR-11.1, FR-11.2, FR-11.3, FR-12.1, FR-12.3, FR-12.4, FR-14.1, FR-14.2, FR-14.3 (download progress
only), FR-14.4, FR-14.5 (journey A commands), FR-14.6 (vacuous at M1), FR-14.7. NFR-1, NFR-2,
NFR-4, NFR-5, NFR-6, NFR-7, NFR-8, NFR-9, NFR-10, NFR-11 (in part: a valid `CITATION.cff`; DOIs
come with the first stable release), NFR-12, NFR-13.

NFR-10 is met at M1. Its colour clause and its English-interface clause are met by the plot
defaults and the English messages. Its label clause ("attribute labels available in the supported
survey languages") is met for the sourced languages: the circumplex file carries the translated
labels of every language Aletta et al. 2024 covers, each with its own citation block (§6.2.3).
Extracting them is M1 work. pol, yue and the Medium and Low-tier languages that no source labels
fall back to English, a documented gap the owner can fill later (§10). The gap does not block the
release.

### 4.2 Plan commitments

Commitments 1 to 11, 14, 15 and 16 bind M1 (plan §3). Commitment 12 binds in shape only: no plugin
code. Commitment 13 binds from M2, so M1 must not invent feature names. Owner register rows that M1
code implements: #1 (licences recorded per dependency, not a filter; §6.19), #2, #3 (click CLI
documented by great-docs), #4 (pooch's bar), #5, #6, #7, #9, #10, #11, #12, #15, #16, #17
(identity by content and versioned angle sets keyed by circumplex; the fingerprint waits for M2),
#18, #19, #20, #21, #25, #26, #27, #28, #35 (the scale belongs to the projection, §6.10), #38
(downloads report progress and are not cancellable), #39 (its test is PRD 0.9's FR-1.5
acceptance: the 0.x tutorials' 56 exclusions on ISD v1.0, §6.7), #40. Row #36 (the Aletta 2024
formulation a Must only with a worked example) is superseded by PRD 0.9 FR-2.3 and §10.5b: the
formulation is a full M1 Must, tested against hand values from its own equations (§6.9.1).

### 4.3 Owner decisions applied

Each row is a decision of the owner, not a call made here. Calls made here are in §11.2; the
owner's remaining open questions are in §10 with fallbacks.

| Decision | Applied in |
|---|---|
| great-docs pinned as a git source at `af17c76` [owner 2026-10-31] | §5.2 |
| "The contour enclosing a fraction of responses" (FR-3.1) is the 50th-percentile (median) contour of Mitchell, Aletta and Kang 2022, the simplified view whose shaded area contains 50% of the responses. 50 is the default and any other percentile is selectable. It is computed as the 2022 figures were, with seaborn's KDE levels (the density-mass fraction on the grid), and `enclosed_share` is reported beside it [owner 2026-10-31] | §6.12 |
| The 2022 reproduction reads ISD v0.2.3, the version the paper's §6 cites, registered as its own loader entry (Zenodo record 5914762). Panel (a) is Russell Square. Density uses seaborn's KDE defaults with `bw_adjust=1.2`. The page stays frozen and re-runs nightly [owner 2026-10-31] | §6.20.3, §6.23 |
| The IoA course tutorial data (`examples/data/DemoData.csv` on 0.x `origin/main`) ships as the bundled course dataset `datasets.course_demo` under CC BY 4.0 with ISD attribution, without `free_response` and the precomputed ISOPleasant and ISOEventful. The course's "Broadwalk" is not in the ISD; course case 1 uses RegentsParkFields against RegentsParkJapan [owner 2026-10-31] | §6.20.4, §6.22 |
| The Aletta 2024 angles are the published values (English monotonous 231°, Italian chaotic 142°), which equal the machine-readable output [owner 2026-10-31] | §6.8.2 |
| M1 is published to PyPI as `1.0.0.dev1` [owner 2026-10-31] | §2, §9 S11 |
| Translated attribute labels are extracted at M1 from Aletta et al. 2024, the ISO/TS 12913-3 revision or the 2026 chapter, with a citation block per language; NFR-10's label clause is met at M1 [owner 2026-10-31]. The chapter prints no label tables (checked), so the sourced languages are Aletta et al. 2024's (§6.2.3) | §4.1, §6.2.3 |
| The loader's default `version` is the latest registered version, today ISD 1.0.1-alpha.1. "Latest" is resolved at release time by updating the registry, never at run time, so a result always records a pinned record and hash. The bundled subset is regenerated from 1.0.1-alpha.1; 1.0.1-alpha stays registered [owner 2026-10-31] | §6.20 |
| SATP v1.5 tiers and High-tier angles come from Tables 4.7 and 4.8 of the 2026 chapter and ship at M1 as `satp@1.5`; `sal-2026` uses it from the first release, with no provisional fallback [owner 2026-10-31] | §6.3.4, §6.8.4 |
| The Aletta et al. 2024 formulation is a full M1 Must, specified by the paper's Eqs. 1 to 3 and tested against values computed by hand from them; no further worked example is needed [owner 2026-10-31; PRD 0.9 FR-2.3, §10.5b] | §6.9.1, §7.2 |
| The FR-1.5 acceptance test is the 0.x tutorials' processing of ISD v1.0: 56 responses excluded, 37 with a missing attribute item and 19 all-identical non-midpoint. The ISD publishes no counts of its own; the "more than half missing" rule of the 2021 paper stays configurable [owner 2026-10-31; PRD 0.9 FR-1.5] | §6.7 |
| The 2022 page's 63 dB split is a split over the whole ISD v0.2.3 dataset the page uses, not per location [owner 2026-10-31; PRD 0.9 §10.5b] | §6.23 |
| The owner's ORCID is 0000-0003-0978-5046 (Andrew Mitchell) and GitHub login `MitchellAcoustics`, for CITATION.cff and CODEOWNERS [owner 2026-10-31] | §5.1, §5.8 |
| The chapter 4 author proof is not committed; the chapter is cited by reference, with its tables transcribed in the chapter notes file [owner 2026-10-31; PRD 0.9 §10.5b] | §1, §4.4 |

### 4.4 Environment constraints

| Constraint | Consequence |
|---|---|
| zenodo.org was unreachable for draft 0.1 and reachable since draft 0.2 | The ISD records were read through the Zenodo API on 2026-10-31 (§6.20.3): licence, DOIs, file names, sizes and md5 sums. The sha256 values were computed from downloads of the two v1.0 CSVs and the v0.2.3 workbook. A nightly network test re-checks them |
| readthedocs.io was unreachable | No library documentation hosted there was read; library facts come from source and PyPI |
| The 2026 chapter is held as an author proof with an exclusive-licence notice | Its Tables 4.7 and 4.8 are transcribed in `design/literature/mitchell-et-al-2026-satp-chapter-4-notes.md`, with proof page numbers. The proof is never committed (owner decision, §4.3); the chapter is cited by reference and the notes file points to the transcribed tables |
| Aletta et al. 2024 Table 4 (p. 12) has no text layer: it is drawn as vector paths | Its labels and angles are transcribed by hand from the rendered page. The second reader is an independent second agent pass at build time that re-reads the rendered page and compares every cell; differences go to the owner (§6.2.3, §6.8.2) |
| Quarto and go-task are not system tools on contributor machines | Both come from PyPI wheels pinned by `uv.lock` (§5.4) |
| Python 3.11 floor | numpy, scipy and contourpy lock to different versions on 3.11 and 3.12+ (the lock forks); every tolerance must hold on both |
| No code from 0.x | 0.x is read for data facts only. Angle tables are regenerated from machine-readable sources |

## 5. Part A: Phase 0

Phase 0 builds the toolchain on an empty package. It ends when `uv run task check` passes on
Windows, macOS and Linux with Python 3.11 to 3.14, `uv run task docs` passes on ubuntu with the
newest Python (the canonical platform for frozen output, §5.6.1), and a wheel installs with no
compiler and imports on all three (plan §7).

### 5.1 Files

| Path | Contents |
|---|---|
| `pyproject.toml` | §5.2, written by hand, not taken from `uv init` (its `uv_build` bound depends on the local uv version) |
| `uv.lock` | Committed with every dependency change |
| `Taskfile.yml` | §5.3 |
| `.pre-commit-config.yaml` | §5.5 |
| `.gitignore` | Existing entries plus `.import_linter_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `/docs/_quarto/`, `/docs/_site/`, `/docs/.cache/`, `/.great-docs-build/`, `/.great-docs/`, `.hypothesis/` |
| `.gitattributes` | §5.1.1 |
| `LICENSE.md` | Exists (BSD-3-Clause). Unchanged |
| `README.md` | Rewritten: one paragraph on what the package does, the install line, a link to the quick-start, citation pointer, licence. great-docs turns it into the home page |
| `GOVERNANCE.md` | §5.8 |
| `CITATION.cff` | §5.8 |
| `CONTRIBUTING.md` | §5.8 |
| `.github/CODEOWNERS` | `@MitchellAcoustics` owns `src/soundscapy/_data/`, `tests/science/`, `tests/regression/` and `design/` (§5.8) |
| `.github/workflows/ci.yml`, `nightly.yml`, `release.yml` | §5.6 |
| `src/soundscapy/__init__.py` | Module docstring; `__version__` from `importlib.metadata.version("soundscapy")`; `__all__ = ["__version__"]` |
| `src/soundscapy/py.typed` | Empty |
| `tests/test_package.py` | §5.7 |
| `tests/test_markers.py` | The marker-override meta-test (§5.7) |
| `docs/great-docs.yml` and the pages of §5.9 | Docs skeleton |
| `.claude/hooks/*.py` | §5.4.1 edit to `stop_check.py`; `ruff format` run once on all three hook scripts (§5.2), as its own commit with no other change, recorded in §11.1 |
| `.claude/skills/house-style/SKILL.md`, `reference.md` | §11.1 edits |
| `.claude/skills/great-docs/`, `write-user-guide/`, `revise-docstrings/`, `VENDORED.md` | Re-copied verbatim from great-docs `docs/skills/` at the pinned commit `af17c76` (§5.2), and the three `VENDORED.md` rows bumped, so the skills describe the tool the lock installs |
| `AGENTS.md` | The **Status** paragraph is rewritten: the toolchain, CI and docs skeleton exist, and `task check` is the definition of done. Plus one line: "If `task` is not on PATH, run `uv run task <name>`; go-task comes with the dev group." |

#### 5.1.1 `.gitattributes`

**Normative.** Shipped data and fixtures are hashed by bytes, so a Windows checkout must not
rewrite their line endings.

```gitattributes
* text=auto
*.qmd text eol=lf
*.toml text eol=lf
src/soundscapy/_data/** -text
tests/**/fixtures/** -text
tests/regression/baselines/** -text
```

Order matters: a later matching line wins, so the `-text` lines come last and also cover the
`.toml` files under `_data/`. Verified with `git check-attr`: `_data/profiles/sal-2026/1.toml`
reads `text: unset` (its `eol: lf` then has no effect), and `pyproject.toml` reads `text: set`,
`eol: lf`.
In draft 0.1's order the shipped TOML files were text with LF endings.

### 5.2 `pyproject.toml`

**Normative.** Verified with uv 0.13.6, ruff 0.17.3, ty 0.0.95, pytest 9.2.0 and import-linter
2.15 on a throwaway package (research, toolchain area). Lower bounds are the current releases on
2026-10-31. No runtime dependency is added at Phase 0; M1 adds them per §6.19.

```toml
[project]
name = "soundscapy"
version = "1.0.0.dev0"
description = "Soundscape survey analysis and assessment following ISO 12913"
readme = "README.md"
requires-python = ">=3.11"
license = "BSD-3-Clause"
license-files = ["LICENSE.md"]
authors = [{ name = "Andrew Mitchell" }]
classifiers = [
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",
  "Topic :: Scientific/Engineering",
  "Typing :: Typed",
]
dependencies = []

[project.urls]
Source = "https://github.com/MitchellAcoustics/Soundscapy"

[build-system]
requires = ["uv_build>=0.13.6,<0.14"]
build-backend = "uv_build"

[dependency-groups]
dev = ["go-task-bin>=3.54.0", "import-linter>=2.15", "prek>=0.5.9", "ruff>=0.17.3", "ty>=0.0.95"]
test = ["hypothesis>=6.170.14", "pytest>=9.2", "pytest-cov>=7.1.0", "pytest-xdist>=3.8.0"]
docs = ["great-docs", "ipykernel>=7.4.0", "quarto-cli==1.10.20"]

[tool.uv]
required-version = ">=0.13.6"
default-groups = ["dev", "test"]

[tool.uv.sources]
# great-docs main at the merge of PR #487 (owner decision, 2026-10-31); the vendored skills in
# .claude/skills/VENDORED.md are copied at the same commit. 0.17.0 on PyPI lacks the docs/
# layout house-style uses. Move to the first PyPI release after 0.17.0 when it exists.
great-docs = { git = "https://github.com/posit-dev/great-docs", rev = "af17c76d6ab2c775e438c6804c3b1b6feceef57e" }

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["ALL"]
ignore = [
  "COM812", "ISC001",  # conflict with the ruff formatter
  "CPY001",            # no per-file copyright headers; LICENSE.md covers the package
  "D105",              # magic-method docstrings add nothing
  "D107",              # numpy style documents __init__ parameters on the class
  "N803", "N806",      # mathematical names such as X, Pa, ISOPl match the papers
  "PLR0913",           # scientific functions legitimately take many keyword arguments
  "TD002", "TD003",    # TODOs need no author or issue link while the rewrite is young
  "FIX002",            # allow TODO comments; review them before each release
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.flake8-type-checking]
runtime-evaluated-base-classes = ["pydantic.BaseModel", "soundscapy._model.FrozenModel"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "PLR2004", "D", "ANN", "INP001"]  # asserts, literal expected values
# Standalone PEP 723 hook scripts, kept short and dependency-free.
".claude/hooks/**" = [
  "INP001", "T201", "S603", "S607", "D103", "PLW1510", "PTH109", "C901", "PLR0911",
]
"scripts/**" = ["INP001", "T201"]  # standalone PEP 723 generators that report

[tool.ty.environment]
python-version = "3.11"
root = ["./src"]

[tool.pytest.ini_options]
minversion = "9.2"
testpaths = ["tests"]
addopts = [
  "--strict-markers", "--strict-config", "--import-mode=importlib", "-ra",
  "-m", "not reference and not network",
]
xfail_strict = true
filterwarnings = ["error"]
markers = [
  "science: literature fixtures and committed cross-implementation fixtures (default run)",
  "regression: golden pipelines against committed baselines (default run)",
  "reference: slow or external cross-implementation runs (nightly, task test:reference)",
  "network: needs a download (nightly, task test:network)",
]

[tool.coverage.run]
branch = true
source = ["soundscapy"]

[tool.coverage.report]
fail_under = 80
show_missing = true

[tool.repo-review]
ignore = []  # each entry added with a trailing comment giving the reason (§5.6.4)
```

**The great-docs pin** (owner decision). `af17c76` is the merge of PR #487 ("refactor/freeze-steps",
2026-10-31T00:22:25Z), the last PR merge on main. `uv lock` with uv 0.13.6 (the `required-version`
floor, run through `uvx --from uv==0.13.6`) resolves it as `great-docs 0.17.1.dev1042+gaf17c76d6`
(verified; uv 0.11.32 gave the same). PR #487 reworked freezing, so the Phase 0 freeze check (§5.9)
runs against this pin. If that check and its `quarto render` fallback, or `task docs`, fail because
of #487, the fallback pin is `095db9b307c76d20576d5bdda9bc8145e238543a` (the merge of #486), with
the vendored skills re-copied at the same commit. The pin moves to PyPI at the first release after
0.17.0.

**The hook scripts.** Phase 0 runs `ruff format` once on `.claude/hooks/*.py` (it edits
`stop_check.py` anyway) and adds the per-file ignore above. The reformat is its own commit with no
other change, so review sees the `stop_check.py` edit separately; §11.1 records it. On ruff 0.17.3
with this config the three scripts raise D103, PLW1510, PTH109, C901 and PLR0911, plus E501 on three
lines that the formatter then wraps; after the format run the ignore list leaves the check clean
(verified). E501 is therefore not ignored. The scripts stay standalone: no docstrings, no
`check=True` on `subprocess.run` (they read the return code), `os.getcwd()` kept.

`ty` does not accept `python-version` directly under `[tool.ty]` (verified); keep it under
`[tool.ty.environment]`. Without `runtime-evaluated-base-classes`, ruff's TC003 fix moves imports a
pydantic model needs at runtime under `TYPE_CHECKING` and construction fails (verified).

#### 5.2.1 Import contracts

**Normative.** Verified with import-linter 2.15 on a scratch package: optional layers (in
parentheses) may be absent; `:` joins sibling layers that may import each other, `|` joins
independent siblings (so `cli` sits above `workspace`, which it calls); `TYPE_CHECKING` imports are
excluded; ignore lines for absent modules are silent.

```toml
[tool.importlinter]
root_package = "soundscapy"
include_external_packages = true
exclude_type_checking_imports = true

[[tool.importlinter.contracts]]
name = "Layers (plan section 5)"
type = "layers"
containers = ["soundscapy"]
layers = [
  "(cli)",
  "(workspace)",
  "(plot)",
  "(datasets)",
  "(projection)",
  "(survey) : (distribution)",
  "(instrument)",
  "(errors) : (provenance) : (_frames) : (_data) : (_model) : (_log) : (tables)",
]

[[tool.importlinter.contracts]]
name = "Core never imports matplotlib"
type = "forbidden"
source_modules = ["soundscapy"]
forbidden_modules = ["matplotlib"]
ignore_imports = ["soundscapy.plot -> matplotlib", "soundscapy.plot.** -> matplotlib"]
allow_indirect_imports = true
unmatched_ignore_imports_alerting = "none"

[[tool.importlinter.contracts]]
name = "Dataframe libraries only in soundscapy._frames"
type = "forbidden"
source_modules = ["soundscapy"]
forbidden_modules = ["pandas", "polars", "pyarrow"]
ignore_imports = [
  "soundscapy._frames -> pandas",
  "soundscapy._frames -> pyarrow",
  "soundscapy._frames -> polars",
]
allow_indirect_imports = true
unmatched_ignore_imports_alerting = "none"

[[tool.importlinter.contracts]]
name = "No GUI toolkit (FR-14.7)"
type = "forbidden"
source_modules = ["soundscapy"]
forbidden_modules = ["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "gi", "kivy", "toga"]
allow_indirect_imports = true

[[tool.importlinter.contracts]]
name = "distribution never imports survey"
type = "forbidden"
source_modules = ["soundscapy.distribution"]
forbidden_modules = ["soundscapy.survey"]
```

The GUI contract keeps the core free of any GUI technology (FR-14.7). Verified with import-linter
2.15: forbidden modules that are not installed are ignored silently, and `import tkinter` (standard
library) breaks the contract.

The last contract is added at slice S1 (its source module must exist). It keeps the sideways import
one-way: `survey.stats` may import `distribution.coordinates`, never the reverse. The layers
contract already stops anything below `plot` importing `plot`, `cli` or `workspace`. M2 adds its
layers (`spi`, `design`) when their modules exist; they are **not** pre-listed, because listing
`(spi) | (design)` as independent would forbid the `design` to `spi` import that plan §5 allows.

### 5.3 `Taskfile.yml`

**Normative** task names (AGENTS.md, the Stop hook and CI depend on them). Every task has a `desc`;
every command runs through `uv run`; `env: {MPLBACKEND: Agg}` is set globally.

| Task | Command |
|---|---|
| `setup` | `uv sync --all-groups`, then `uv run prek install` |
| `fmt` | `uv run ruff format`, then `uv run ruff check --fix` |
| `lint` | `uv run ruff format --check`, `uv run ruff check`, `uv run lint-imports` |
| `typecheck` | `uv run ty check` |
| `test` | `uv run pytest {{.CLI_ARGS}}` |
| `test:cov` | `uv run pytest --cov --cov-report=term-missing {{.CLI_ARGS}}` |
| `test:reference` | `uv run pytest -m reference {{.CLI_ARGS}}` |
| `test:network` | `uv run pytest -m network {{.CLI_ARGS}}` |
| `check:fast` | `task: lint`, `task: typecheck` (offline, a few seconds) |
| `check` | `task: lint`, `task: typecheck`, `task: test:cov` |
| `docs` | `uv run --group docs great-docs build` (offline; papers stay frozen) |
| `docs:papers` | `uv run --group docs great-docs freeze docs/papers/2022-how-to-analyse.qmd` (network). If Phase 0 finds that this does not re-execute a `freeze: true` page, the task runs the §5.9 fallback instead |
| `docs:preview` | `uv run --group docs great-docs preview` |
| `build` | `uv build` |
| `repo-review` | `uvx 'sp-repo-review[cli]' .` |

A later `-m` on the command line overrides the `-m` in `addopts`, so `test:reference` and
`test:network` select their markers; the meta-test of §5.7 proves this. `repo-review` exits 3
while any check fails, so Phase 0 is not done until the ignore list of §5.6.4 makes it exit 0.

### 5.4 How contributors get go-task and Quarto

Contributors install only **uv and git**. `uv sync` installs `go-task-bin` (a community-maintained
wheel listed under go-task's "Community packages", pinned by `uv.lock`; win_amd64, win_arm64,
macOS arm64 and x86_64, manylinux and musllinux wheels exist). Quarto comes from `quarto-cli` in
the docs group (an sdist that fetches the Quarto 1.10.20 binary; great-docs finds it with
`shutil.which` inside `uv run`). The documented command form is `uv run task <name>`; bare `task`
works only with the venv activated. CONTRIBUTING.md says exactly this.

A full great-docs build through `quarto-cli` was not run on Windows or macOS, and CI builds the docs
on ubuntu only (§5.6.1). **Fallback:** if a contributor's build fails on Windows or macOS,
CONTRIBUTING.md documents a system Quarto 1.10.20 install for that platform. Quarto 1.10 is the
stable line (PyPI `quarto-cli` is 1.10.20; v1.11 tags are prereleases).

#### 5.4.1 Stop hook edit

`.claude/hooks/stop_check.py` today returns 1 when `task` is not on PATH. Phase 0 changes the
command selection to:

```python
# Amended at Phase 0 code review: uv first, so the go-task pinned in uv.lock runs rather
# than whatever `task` is on PATH (Taskwarrior installs one).
if shutil.which("uv"):
    cmd = ["uv", "run", "task", "check:fast"]
elif shutil.which("task"):
    cmd = ["task", "check:fast"]
else:
    print("stop_check hook: neither `task` nor `uv` is on PATH; skipped check:fast", file=sys.stderr)
    return 1
```

### 5.5 prek

**Normative** shape. The hygiene hooks use prek's `repo: builtin`, which needs no download
(verified offline). `ruff-pre-commit` v0.17.3, `uv-pre-commit` 0.13.6 and `actionlint-py`
v1.7.12.25 need one fetch on first use; CI caches `~/.cache/prek`.

```yaml
repos:
  - repo: builtin
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-toml
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=600]
      - id: check-merge-conflict
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.3
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/astral-sh/uv-pre-commit
    rev: 0.13.6
    hooks:
      - id: uv-lock
  - repo: https://github.com/Mateusz-Grzelinski/actionlint-py
    rev: v1.7.12.25
    hooks:
      - id: actionlint
  - repo: local
    hooks:
      - id: repo-review
        name: sp-repo-review
        entry: uvx 'sp-repo-review[cli]' .
        language: system
        pass_filenames: false
        stages: [manual]
```

`--maxkb=600` admits the bundled subset (about 315 KiB) and the course dataset (about 73 KiB) with
margin. actionlint comes from the `actionlint-py` PyPI wrapper, a `language: python` hook, so no Go
toolchain is needed. Its package is an sdist whose build fetches the pre-built actionlint binary
from GitHub, so the first use needs the network (checked on 2026-10-31: PyPI's latest is
`1.7.12.25`, sdist only, and the repository's `.pre-commit-hooks.yaml` defines hook id `actionlint`
on `^.github/workflows/`). actionlint is a hook, not part of `task lint`, so `check:fast` stays
offline. ty is not a hook (house-style).

### 5.6 CI

Actions are pinned by commit SHA with the tag in a comment: `actions/checkout` v7.0.1 and
`astral-sh/setup-uv` v10.3.3 (`68559e99cc807c1eb7e9238fd43a15bba460eec2`). setup-uv publishes no
moving `v10` tag. The builder resolves checkout's SHA with `git ls-remote`. Every job sets
`permissions: contents: read` unless stated, uses `enable-cache: true`, and runs `uv sync --locked`
(except `lowest`, which re-resolves).

#### 5.6.1 `ci.yml` (push to `soundscapy-rewrite`, pull requests, manual)

| Job | Matrix | Steps |
|---|---|---|
| `check` | ubuntu, windows, macos × 3.11, 3.12, 3.13, 3.14; `fail-fast: false` | `uv run task check` |
| `lowest` | ubuntu × 3.11; job-level `env: {UV_RESOLUTION: lowest-direct}` | `uv sync`, `uv tree --depth 1` (prints the installed direct versions into the log), `uv run task test`. Proves the runtime floors of §6.19 (pandas 2.2 in particular) |
| `docs` | ubuntu × 3.14 only, the canonical platform for frozen output | `uv sync --locked --group docs`, `uv run task docs`, then `git diff --exit-code docs/_freeze` to catch a page whose freeze is stale |
| `citation` | ubuntu × 3.14 | `uvx cffconvert@2.0.0 --validate` (validates `CITATION.cff` against schema 1.2.0) |
| `install` | ubuntu, windows, macos × 3.11, 3.14 | `uv build`; in a fresh venv `uv pip install --only-binary :all: dist/*.whl` in a step with `shell: bash`, so the `*.whl` glob expands on Windows too (PowerShell passes it literally); run `python -c "import soundscapy"`, `docs/user_guide/_quickstart.py` and `soundscapy --version` (the latter from M1 slice S8). This job, not `docs`, is what covers Windows and macOS: installation, the quick-start script and importing |
| `install-win-arm` | `windows-11-arm` × 3.13 with x64 Python; `continue-on-error: true` | The `install` steps. x64 Python runs emulated on the Arm runner and installs win_amd64 wheels, which is the route a course participant on a Windows Arm laptop takes. Native arm64 Python is not tested, because pyarrow ships no win_arm64 wheel. The job records the state rather than gating, because emulated runs on a new runner image are not yet a stable signal |

The environment variable, not a `--resolution` flag on `uv sync`, is what makes `lowest` test the
floors: a later `uv run` re-resolves with the default `highest` mode and reinstalls the newest
versions unless `UV_RESOLUTION` is set for the whole job (verified with uv 0.13.6, run through
`uvx --from uv==0.13.6`: after `uv sync --resolution lowest-direct`, `uv run` reported "Ignoring
existing lockfile due to change in resolution mode" and upgraded tomlkit 0.13.0 to 0.15.2; with the
variable set it kept 0.13.0, and `uv tree --depth 1` printed `tomlkit v0.13.0`).

#### 5.6.2 `nightly.yml` (cron, manual)

`uv run task test:reference`, `uv run task test:network`, `uv run task docs:papers`, then a
freeze-diff check that opens an issue when `docs/_freeze` changed; these run on ubuntu × 3.14.
Caches the pooch directory (`SOUNDSCAPY_DATA_DIR` set to a cache path). The freeze-diff job is the
one job that writes: it sets `permissions: {contents: read, issues: write}`.

A `quickstart` job runs the user-facing install scripts end to end on a clean runner with only uv:
`_quickstart.sh` on ubuntu and macOS, `_quickstart.ps1` on Windows, each in a fresh temporary
directory. It also runs the one-cell hosted-notebook install (NFR-1) on ubuntu: in a fresh
`uv venv --seed`, `python -m pip install 'soundscapy==1.0.0.dev1'`, the command a hosted notebook's
`%pip install 'soundscapy==1.0.0.dev1'` cell runs, then `_quickstart.py`. Before the first release,
when `soundscapy==1.0.0.dev1` is not on PyPI, the job replaces that requirement with the git URL of
the newest `v*` tag, or of the `soundscapy-rewrite` head while no tag exists
(`soundscapy @ git+https://github.com/MitchellAcoustics/Soundscapy@<ref>`); after it, the scripts
and the pip cell run unchanged against PyPI. A TestPyPI upload is not needed.

#### 5.6.3 `release.yml` (tags `v*`)

Environment `pypi`, `permissions: id-token: write`; `uv build`, then
`uv publish --trusted-publishing always`. The owner sets up the PyPI trusted publisher when a dev
release is ready. `release.yml` exists but is not triggered until then.

#### 5.6.4 repo-review

Phase 0 runs `task repo-review` and, for each failing check that does not fit house-style, adds the
check id to `[tool.repo-review] ignore` with a trailing comment giving the reason, and records the
same reason in the house-style skill. The job is green when the command exits 0.

### 5.7 Phase 0 tests

| Test | Asserts |
|---|---|
| `tests/test_package.py::test_version_matches_metadata` | `soundscapy.__version__ == importlib.metadata.version("soundscapy")` |
| `tests/test_package.py::test_import_is_silent` | `subprocess.run([sys.executable, "-c", "import soundscapy"])` writes nothing to stdout or stderr (NFR-13). Run with `-W error` |
| `tests/test_markers.py::test_later_m_overrides_addopts` | In a temporary directory with a copy of the marker config and two marked tests, `pytest -m network` runs the network test and `pytest` alone deselects it |

No stub modules are created and no file is omitted from coverage: one importing test keeps
coverage at 100% of the few statements in `__init__.py` (verified; a test that never imports the
package gives 0% and fails the gate).

### 5.8 Governance files

| File | Essential contents |
|---|---|
| `GOVERNANCE.md` | Andrew Mitchell cuts releases and reviews every change that affects scientific results: any change that moves a regression baseline, edits a file under `_data/`, changes a science fixture, or bumps a method version (NFR-12). Such PRs carry the label `science` and need the owner's approval. Releases follow PRD §9. CODEOWNERS (`@MitchellAcoustics` on the owned paths) enforces the owner's review of those paths |
| `CITATION.cff` | `cff-version: 1.2.0`, `message: "If you use this software, please cite it using the metadata from this file."`, title, `type: software`, author Andrew Mitchell with `orcid: "https://orcid.org/0000-0003-0978-5046"`, `license: BSD-3-Clause`, `repository-code`. `preferred-citation` is left out until the owner names one. `message` is required by the schema: `cffconvert --validate` rejects the file without it. The file with the ORCID validates (verified with cffconvert 2.0.0; the ORCID check digit is valid). CI validates it (§5.6.1) |
| `CONTRIBUTING.md` | Install uv and git; `uv run task setup`; `uv run task check` is the definition of done; how to run science, regression, reference and network tests; the design workflow pointer to AGENTS.md; that `_data/` and baselines need owner review |

### 5.9 Docs skeleton

**Normative** `docs/great-docs.yml` at Phase 0 (keys from the great-docs main layout at the pinned
rev `af17c76`; `freeze.mode` is still read as `"auto"` there, checked in its `config.py`; paths
resolve relative to `docs/`):

```yaml
parser: numpy
dynamic: true
jupyter: python3
user_guide: user_guide
sections:
  - title: Tutorials
    dir: tutorials
    index: true
  - title: Papers
    dir: papers
    index: true
bibliography: references.bib
freeze:
  mode: auto
cli:
  enabled: false
```

Pages at Phase 0:

| Page | Contents |
|---|---|
| `docs/user_guide/01-install.qmd` | Install uv (PowerShell and macOS/Linux tabs), create a project, add soundscapy and jupyterlab with the exact pre-release pin (`uv add 'soundscapy==1.0.0.dev1' jupyterlab`; uv accepts an exact pre-release pin without `--prerelease`, verified with uv 0.13.6), open JupyterLab, run the first-plot code. The commands are not typed into the page: they live in `docs/user_guide/_quickstart.ps1`, `_quickstart.sh` and `_quickstart.py` and are included with `{{< include >}}` shortcodes on bare lines, which great-docs expands into fenced code blocks itself (the `.ps1` include passes `lang="powershell"`; verified with great-docs `af17c76` and Quarto 1.10.20; a `#| file:` cell option does not work). CI runs the same files (§5.6.1). A "Hosted notebook" section gives the one install cell, `%pip install 'soundscapy==1.0.0.dev1'`, then the same first-plot code (NFR-1); the nightly `quickstart` job runs its pip command (§5.6.2) |
| `docs/tutorials/00-check-install.qmd` | One executed cell printing `soundscapy.__version__` |
| `docs/papers/index.qmd` | States that each reproduction downloads the ISD version its paper used (v0.2.3 for 2022), is frozen, and re-runs nightly |
| `docs/references.bib` | Empty at Phase 0; grows with each cited procedure |

At Phase 0 `_quickstart.py` prints the version; slice S1 replaces it with the three-line first plot
(§6.21). Interlinks to numpy, scipy and matplotlib are added only after a build with the network off
passes with them (inventories may be fetched at build time; `task docs` must stay offline). Pydantic
models document their fields under an `Attributes` section, not `Parameters`, which great-docs
otherwise flags as unknown parameters. great-docs strips numeric filename prefixes from URLs.

Every user-guide page sets `guide-section` in its front matter to one of `How-to`, `Explanation`
or `Reference` (`01-install.qmd` is `How-to`). great-docs reads this key for sidebar grouping (it
avoids Quarto's own `section`; checked at `af17c76`), so FR-12.3's four-part structure is
checkable: a docs test reads every page under `docs/user_guide/` and fails on a missing or unknown
value. Tutorials are the fourth part by directory.

Phase 0 also checks whether `great-docs freeze <page>` re-executes a page that sets `freeze: true`,
and records the result in house-style `reference.md`. The 2022 page depends on it. The check runs
against the pin `af17c76`, because PR #487 changed how freezing works; the result names the
commit it was run on.

**Freeze fallback**, tried in order until one re-executes the page:

| Step | `task docs:papers` runs |
|---|---|
| 1 | `great-docs freeze docs/papers/2022-how-to-analyse.qmd` at `af17c76` |
| 2 | `quarto render` on the page great-docs generates under `docs/_quarto/`, with execution forced. The exact flag (`--execute`, or `--cache-refresh` with `execute: freeze: false` in the call's metadata) is verified at Phase 0 and pinned in the Taskfile |
| 3 | Move the great-docs pin to `095db9b` (§5.2) and repeat step 1 |

The Phase 0 exit records which step worked.

### 5.10 Phase 0 exit criteria

| Exit | Shown by |
|---|---|
| `uv run task check` green on 3 OS × 3.11 to 3.14 | `ci.yml` `check` |
| `uv run task docs` green, offline, on ubuntu × 3.14 | `ci.yml` `docs` |
| Clean no-compiler install, import and quick-start script on 3 OS (NFR-1) | `ci.yml` `install` |
| Dependency floors provable | `ci.yml` `lowest` (vacuous until M1 adds dependencies) |
| `task repo-review` exits 0 | Local run recorded in the PR |
| Markers behave as specified | `tests/test_markers.py` |
| Import is silent | `tests/test_package.py` |
| Freeze behaviour of `freeze: true` pages recorded | house-style `reference.md` entry, naming commit `af17c76` and the fallback step of §5.9 that worked |
| Stop hook works without `task` on PATH | Manual run recorded in the PR |
| `CITATION.cff` valid, with the owner's ORCID | `ci.yml` `citation` |
| CODEOWNERS names `@MitchellAcoustics` on the owned paths | The Phase 0 PR diff |
| Vendored great-docs skills match the pin | `VENDORED.md` rows at `af17c76` |
| Hook scripts lint clean | `ruff check .claude/hooks` inside `task lint`; the reformat is its own commit |
| AGENTS.md Status paragraph rewritten | The Phase 0 PR diff |

## 6. Part B: M1

### 6.1 Module and file layout

Each module is created with its first code. Layers follow §5.2.1; within a layer, files may import
each other freely, and `survey` may import `distribution` but not the reverse.

| Layer | Path | Holds |
|---|---|---|
| Bottom | `_model.py` | `FrozenModel`, the pydantic base for every value, settings and mapping type |
| | `errors.py` | Every exception and warning class (§6.3.10) |
| | `provenance.py` | The hashing helper, the envelope builder, `show_versions()` |
| | `_log.py` | The `soundscapy` logger, its default handler, `set_log_level()` |
| | `_frames.py` | The only importer of pandas, polars and pyarrow: conversions, readers, writers, `frame_hash` |
| | `tables.py` | `Table` (used by `survey`, `distribution` and `workspace`, so it cannot live higher) |
| | `_data/__init__.py` | Resource access (`importlib.resources`), the registry of shipped files with hash pins, the raw profile reader |
| | `_data/circumplexes/iso-12913-outdoor/1.toml` | The default instrument, with translated labels (§6.2.3) |
| | `_data/angle_sets/<name>/<version>.toml` | `aletta-2024/1.toml`, `satp/1.5.toml` |
| | `_data/profiles/<id>/<version>.toml` | The three profiles, version `1` each (§6.3.4) |
| | `_data/datasets/isd/` | `subset-v1.0.csv`, `subset-v1.0.toml` (sidecar), `registry.toml` |
| | `_data/datasets/course_demo/` | `course_demo.csv`, `course_demo.toml` (sidecar), `mapping.toml` (§6.20.4) |
| | `_data/templates/` | `canonical-survey.csv`, `canonical-survey.xlsx`, `canonical-mapping.toml` |
| | `_data/schemas/` | `ssid.toml`, `redcap.toml`, `qualtrics.toml` alias tables for `suggest_mapping` |
| `instrument` | `scale.py`, `circumplex.py`, `angles.py` | `Scale`, `Attribute`, `Axis`, `Circumplex`, `circumplex()`; `AngleSet`, `AngleSetRef`, `AngleSelection`, the resolver |
| `survey` | `mapping.py` | `SurveyMapping`, `ItemGroup`, `Coding`, ids, language and flag specs; TOML read and write (tomlkit) |
| | `suggest.py` | `suggest_mapping`, `MappingDraft` |
| | `read.py` | `read_survey`, `ReadOptions`, `Survey`, `ImportSettings` |
| | `status.py` | `Reason`, `Flag`, `Status` |
| | `quality.py` | `QualitySettings`, `Action`, the rules |
| | `items.py` | `item_statistics`, `ItemStatisticsSettings`, `response_shares` |
| | `context.py` | `join_context` (location and session tables, §6.5.1) |
| | `stats.py` | `correlate`, `compare_groups`, their settings, effect-size formulas |
| | `scoring.py` | `who5`, `source_profile` |
| | `template.py` | `write_template` |
| `distribution` | `coordinates.py` | `Coordinates`, `OriginSegment`, `n`, `n_valid` |
| | `regions.py` | `Region` (a FrozenModel), default regions, the boundary rule |
| | `describe.py` | `describe`, `DescribeSettings` |
| | `kde.py` | `density`, `contours`, `marginals`, `Density`, `Densities`, `ContourSet`, `DensitySettings`, `ContourSettings` |
| `projection` | `formulas.py` | `project_scores` (pure numpy kernel) |
| | `project.py` | `project`, `ProjectionSettings` |
| `datasets` | `isd.py`, `isd_mapping.toml`, `isd_v0.2.3_mapping.toml`, `course_demo.py` | `isd.load`, `isd.fetch`; `course_demo.load`. Each ISD version's registry entry names its mapping file (§6.20.3) |
| `plot` | `_common.py`, `circumplex.py`, `items.py` | Shared helpers; scatter, density, fraction_contour, joint; likert, radar. matplotlib is imported inside each function, never at module top (§6.15) |
| `workspace` | `__init__.py`, `_layout.py`, `_io.py` | Project functions, including `export_run` and `add_table`; the path helper that alone knows file names; `export` and `attach` (top-level names re-exported from here) |
| `cli` | `cli.py`, `__main__.py` | The click app; `python -m soundscapy`. The top layer, above `workspace` (§5.2.1) |
| repo | `scripts/make_isd_subset.py`, `scripts/make_course_demo.py`, `scripts/make_aletta_2024.py`, `scripts/make_seaborn_fixtures.py` | PEP 723 generators, each idempotent and checking its inputs' hashes |


### 6.2 Value types

#### 6.2.1 `FrozenModel` and records

**Normative.**

```python
class FrozenModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
```

Every value, settings and mapping type subclasses it. `allow_inf_nan=False` makes every float
field reject ±inf and NaN at validation, so canonical JSON (`allow_nan=False`, §6.3.5) can always
hash a model. "No limit" is never a ±inf sentinel: it is the literal `"none"` in a field typed
`Literal["none"] | tuple[...]` (`DensitySettings.clip`, `Region.box`). This replaces draft 0.1's
±inf sentinels, which canonical JSON could not hash. Verified with pydantic and `tomllib`: TOML
`clip = "none"` and `clip = [[-1.0, 1.0], [-1.0, 1.0]]` validate, while `-inf` inside the array
and `bw_adjust = nan` or `inf` are rejected. Fields use `StrictFloat`, `StrictInt`,
`StrictStr` and `StrictBool` per field, never model-level `strict` (which rejects TOML arrays for
tuple fields). Collections are tuples; mappings that must hash are `tuple[tuple[K, V], ...]`
exposed through a property returning a `dict`. A changed settings record is derived with
`Model.model_validate({**old.model_dump(), **changes})`, never `model_copy(update=)` (which skips
validation). Every optional field has a non-`None` default, because TOML cannot store `None`
(commitment 6). Serialisation to TOML is `model_dump(mode="json", by_alias=True)` written by
tomlkit; reading is `tomllib` then `model_validate`. (Amended at S1 plan review, 2026-10-31:
`by_alias=True`, because every `schema` key is a field aliased `schema`, §6.4.1.)

Records that hold arrays or frames are `@dataclass(frozen=True, eq=False, slots=True)`: `Survey`,
`Status`, `Coordinates`, `Density`, `ContourSet`, `Table`, `MappingDraft`. Their constructors copy
numeric input arrays to `float64`/`bool` with `copy=True` and call `setflags(write=False)`; text
arrays (`str` or `object` dtype) are copied and made read-only the same way. Frames are held only
by `Survey.data` and `Table.frame`, and are copied. No other record holds a frame: inside the core
everything is numpy (plan rule 4 keeps backend-specific code in the boundary helper), and
`_frames.new_frame` builds the user-facing frame at the return boundary.

#### 6.2.2 `Scale`

| Field | Type | Default | Invariant |
|---|---|---|---|
| `minimum` | StrictFloat | required | `minimum < maximum` |
| `maximum` | StrictFloat | required | |
| `kind` | `Literal["ordinal", "interval"]` | `"ordinal"` | ordinal scales need integer-valued bounds |
| `labels` | `tuple[tuple[str, tuple[str, ...]], ...]` | `()` | per language, one label per integer step when present |

Derived properties: `midpoint = (minimum + maximum) / 2`, `half_range = (maximum - minimum) / 2`,
`range = maximum - minimum`. `contains(values) -> ndarray[bool]` is true for finite values in
`[minimum, maximum]` that are integer-valued when `kind="ordinal"`.

#### 6.2.3 `Attribute`, `Axis`, `Circumplex`

| Type | Fields and invariants |
|---|---|
| `Attribute` | `position: StrictStr` matching `^PAQ[1-9][0-9]*$`; `angle_deg: StrictFloat` in `[0, 360)`; `labels: tuple[tuple[str, str], ...]` keyed by ISO 639-3 code |
| `Axis` | `id: StrictStr` (an identifier, used in column names); `positive: str` and `negative: str` (positions at the axis ends); `labels` by language |
| `Circumplex` | `schema: Literal["soundscapy.circumplex"]` and `schema_version: Literal[1]`, required, the identity keys of the file, with the field alias of §6.4.1 (amended at S1 plan review, 2026-10-31); `name`, `version` (StrictStr); `attributes: tuple[Attribute, ...]` with positions PAQ1..PAQn contiguous in order and angles strictly increasing; `axes: tuple[Axis, Axis]` (x then y); `scale: Scale` (the default scale); `quadrants: tuple[tuple[str, str], ...]` mapping a sign key (`"++"`, `"-+"`, `"--"`, `"+-"`, x sign first) to a position whose label names the quadrant; `reference_language: str = "eng"`; `citation` block (§6.3.7) |

`Circumplex` methods: `positions`, `angles_deg` (the nominal angles), `label(position, language)`
(falls back to the reference language), `position_for(key)` (accepts a position or a reference
label, case-insensitive; raises `MappingError` otherwise). "Equal angles" always means the
circumplex's own nominal angles, which is what makes an indoor circumplex (FR-2.0) work without
special cases.

The shipped instrument, `iso-12913-outdoor` version `1`: positions PAQ1..PAQ8 at 0, 45, ..., 315
degrees with English labels pleasant, vibrant, eventful, chaotic, annoying, monotonous, uneventful,
calm; axes `ISOPleasant` (positive PAQ1, negative PAQ5) and `ISOEventful` (positive PAQ3,
negative PAQ7); scale 1..5 ordinal with the Table A.1 agreement labels (strongly disagree to
strongly agree; taken from 0.x until the owner checks the wording, §11.2); quadrants `++` PAQ2,
`-+` PAQ4, `--` PAQ6, `+-` PAQ8. `circumplex(name="iso-12913-outdoor", version="1")` returns it.

**Translated labels** (NFR-10). Version 1 carries a translation for every language Aletta et al.
2024 covers, the only source that prints the labels. `Circumplex.translations` is a tuple of
`Translation` (FrozenModel): `language` (the code the angle sets use), `labels` (one per position,
`tuple[tuple[str, str], ...]`) and a `citation` block (§6.3.7) of its own. In the TOML file each is
a `[translations.<code>]` table with `PAQ1` to `PAQ8` keys and a `[translations.<code>.citation]`
table. Labels are stored as published, in the native script, including multi-term labels such as
`"aangenaam / prettig"`. Axis labels are the labels of the attributes at the axis ends.

| Languages | Labels |
|---|---|
| eng, arb, cmn, deu, ell, fra, hrv, ind, ita, jpn, kor, nld, por, spa, swe, tur, vie, zsm (18) | Aletta et al. 2024, Table 4, p. 12, with that paper's citation block. The table has no text layer, so it is transcribed by hand from the rendered page. The second reader is an independent second agent pass at build time: it re-reads the rendered page and compares every cell with the transcription, and any difference goes to the owner |
| pol, yue (new in SATP v1.5), tha, bra, alb (Medium and Low-tier languages of `satp@1.5` with no source) | None shipped. `label()` falls back to English. The 2026 chapter prints no label tables (checked), so no source exists in hand. A documented known gap the owner can fill later (§10); a filled language is a new circumplex version |

`bra` and `alb` are the chapter's codes, kept as published so a later translation joins the angle
set: `bra` reads as Brazilian Portuguese beside `por` (in ISO 639-3 it is Braj), and `alb` is the
ISO 639-2/B code for Albanian (639-3 `sqi`). The notes of `satp/1.5.toml` say so. A test asserts
that every sourced language has a translation with all eight positions and a non-empty citation,
and that each language of the second row has none and gets the English labels from `label()`.
`label(position, language)` falls back to the reference language for any language with no
translation, a gap language or a language of a user's own angle set; its docstring says so and
lists the gap languages. Labels in a non-Latin script need a font with those glyphs; the plotting
how-to says so. A label correction after release goes into version 2, never an edit (plan rule 15).

### 6.3 Cross-cutting contracts

#### 6.3.1 Status (commitment 5)

`Status` is a frozen dataclass of three aligned sequences of length n: `included: ndarray[bool]`,
`reasons: tuple[tuple[Reason, ...], ...]`, `flags: tuple[tuple[Flag, ...], ...]`. Invariant:
`included[i]` is true exactly when `reasons[i]` is empty. Codes inside a tuple are unique and in
enum order. Methods return new objects: `with_reason(mask, reason)`, `with_flag(mask, flag)`,
`counts() -> dict[str, int]`, `to_frame()`.

**Normative** codes, in this order:

| Kind | Codes |
|---|---|
| `Reason` (excludes) | `scale_violation`, `all_identical`, `excessive_missing`, `no_consent`, `test_response`, `under_age`, `paper_input`, `incomplete_attributes`, `uncovered_language`, `low_confidence_language` |
| `Flag` (kept, noted) | `all_midpoint`, `under_age`, `paper_input`, `medium_confidence_language`, `equal_angles_fallback` |

As table columns: `ssp_included` (bool), `ssp_reasons` and `ssp_flags` (codes joined with `;`,
empty string when none). No row is ever dropped (plan rule 11). `n` is the number of rows; `n_valid`
is the number of rows with `included` true and finite x and y. Both are defined once, in
`distribution.coordinates`, and every consumer calls them.

#### 6.3.2 `response_id` (commitment 5)

The first rule that applies wins, and the rule used is recorded as
`provenance["response_id_source"]` (`"column"`, `"key"` or `"row"`):

1. The mapping names a `response` column: its text, stripped.
2. The mapping names `response_key` columns: their texts joined with `/`.
3. The source is a file: `row-<n>`, where n is the record number in the stored raw file, the
   header being row 1, so the first record is `row-2`, matching what Excel shows. For a sheet, n is
   the worksheet row number. Skipped blank rows keep their numbers out of use, so ids do not shift.
4. The source is an in-memory frame and neither column is mapped: `MissingResponseIdError`, with
   the hint to map a `response` column or a `response_key`.

A blank id raises `MissingResponseIdError`; a duplicate raises `DuplicateResponseIdError`. Both
list up to five offenders with their row numbers. A key part containing `/` is a `MappingError`.
A blank key part raises `MissingResponseIdError` unless the mapping lists that column in
`IdSpec.response_key_blank_ok` (§6.4.1); a listed part may be blank, the joined id then keeps an
empty segment (`<location>/<session>/<group>/`), and the uniqueness check runs on the joined ids as
always. For the ISD the key is `(LocationID, SessionID, GroupID, RecordID)`: unique across all 3589
rows, no part contains `/`, and readable, unlike `RecordID` alone (2664 unique values) or `GroupID`
(`RES` spans two locations). In v0.2.3, 607 of the 1909 rows are recording-only rows with a blank
`RecordID`, a blank `Language` and no survey items, so its mapping lists `RecordID` in
`response_key_blank_ok`; the joined ids are unique across all 1909 rows (checked on the downloaded
workbook).

#### 6.3.3 Derived-column namespace (commitment 5)

Every derived column in a frame that also holds user columns starts with `ssp_`: `ssp_response_id`,
`ssp_<axis id>` (so `ssp_ISOPleasant`, `ssp_ISOEventful`), `ssp_language`, `ssp_angle_set`, the
three status columns, and `ssp_who5_*` when WHO-5 results are attached. Tables the package builds
with no user columns (describe output, statistics) use plain names. `read_survey` raises
`ColumnCollisionError` if any input column starts with `ssp_`, before anything else runs, so a
collision cannot surface mid-analysis. `attach` (§6.16) never overwrites.

#### 6.3.4 Settings and profiles (commitment 6)

Each operation owns one settings model, defined beside it:

| Operation | Model | Profile section |
|---|---|---|
| `read_survey` | `ImportSettings(quality: QualitySettings, on_scale_violation: Literal["error", "exclude"] = "error")` | `[import]` |
| `project` | `ProjectionSettings(formulation: Literal["canonical", "iso2019", "aletta2024"] = "canonical", angles: AngleSelection)` | `[projection]` |
| `describe` | `DescribeSettings` (§6.11) | `[describe]` |
| `density`, `marginals` | `DensitySettings` (§6.12) | `[density]` |
| `contours` | `ContourSettings` (§6.12) | `[contours]` |
| `item_statistics`, `response_shares` | `ItemStatisticsSettings` (§6.13.1) | `[item_statistics]` |
| `correlate` | `CorrelateSettings` (§6.13.2) | `[correlate]` |
| `compare_groups` | `CompareSettings` (§6.13.3) | `[compare_groups]` |
| `who5`, `source_profile` | `Who5Settings`, `SourceProfileSettings` (§6.14) | `[who5]`, `[source_profile]` |

Every argument that changes a number is a field of its operation's settings model and is recorded
with the result, as `describe`'s `by` and `regions` are: grouping (`by`), item groups, variables,
features, tests, rules and thresholds alike. Data passed in (a frame of features, a `Coordinates`)
is an input, recorded by its hash (§6.3.6), not a setting. Every optional keyword defaults to the
`_UNSET` sentinel, never `None`, and `by = ""` means no grouping.

**Profiles** are versioned data, like angle sets: `_data/profiles/<id>/<version>.toml`. The ids
(no colons, safe in Windows file names) are `sal-2026` (the default), `iso-12913-3-2019` and
`iso-12913-3-2024`; each ships version `1`. A profile file holds `schema = "soundscapy.profile"`,
`schema_version = 1`, `id`, `version`, a `[citation]` block (§6.3.7) and one section per
operation; an omitted key means the model default. A released version is never edited: a changed
default is a new version (plan rule 15). `profile=` takes `"<id>@<version>"` or a bare id, which
means the latest shipped version; the result always records the version used.

| Profile | `projection.formulation` | `projection.angles.default` | `item_statistics.dispersion` | Locked fields |
|---|---|---|---|---|
| `iso-12913-3-2019@1` | `iso2019` | `"equal"` | `range` | formulation, angles (default and per_language), dispersion |
| `iso-12913-3-2024@1` | `canonical`, labelled "interpretation pending the owner's check of the text" in the profile's `notes` (PRD §10.6) | `"equal"` | `iqr` | formulation, angles, dispersion |
| `sal-2026@1` | `canonical` | `satp@1.5` (§6.8.4) | `iqr` | none |

`sal-2026@1` uses `satp@1.5` from the first release, so no shipped profile is provisional. All
three take the `DensitySettings` defaults (§6.12). No profile uses `aletta2024` by default. It is
selectable under `sal-2026` (recorded as an override) for reproducing Aletta et al. 2024, and the
two ISO profiles refuse it with `ProfileConstraintError`, because their formulation is locked.

**Precedence** (the one rule; §6.17 and §6.18 refer here). First the base profile is chosen:

1. an explicit `profile=` argument, or the CLI's `--profile`;
2. else the profile recorded in the input's provenance (carry-over, below);
3. else the profile named by `extends` in a `settings=` file or in the project's `settings.toml`;
4. else `sal-2026`.

Then each field takes the first value set in this order, highest first, and one `model_validate`
runs on the result:

| Rank | Source |
|---|---|
| 1 | explicit keyword arguments, or CLI options |
| 2 | `settings=` (a model, a mapping of fields, or a settings file; the CLI's `--settings`) |
| 3 | the project's `settings.toml` (inside `workspace.run` and the CLI only) |
| 4 | the base profile's section |

A keyword argument left at its sentinel default (`_UNSET`) does not override. An ISO profile raises
`ProfileConstraintError` when a resolved locked field differs from the profile's value, naming the
field and suggesting `profile="sal-2026"`.

**Profile carry-over.** Every operation takes `profile: str | _Unset = _UNSET`. Left unset, it
inherits the profile recorded in its input's provenance (a `Survey`, `Coordinates`, `Density` or
`Densities`; for `contours`, the density). Only an input that records no profile (an in-memory
frame passed to `read_survey`, or a user-built record) falls through to step 3 or 4. So
`describe(project(read_survey(..., profile="iso-12913-3-2019")))` runs every step under the 2019
profile without repeating it. An explicit profile (step 1) replaces the base profile for that
operation and is recorded as the profile, not as an override; when it differs from the input's,
`provenance["profile_inherited"]` records the input's `{id, version}` beside it. A test runs the
2019 pipeline (read, project, describe, item statistics) with the profile named only on
`read_survey`, and asserts the `iso2019` formulation, the `range` dispersion, the recorded profile
on every result and an empty `overrides`; a second run passes `profile="sal-2026"` to `project`
and asserts that `profile` records `sal-2026@1`, `profile_inherited` records the 2019 profile, and
`overrides` stays empty.

The result records `provenance["profile"] = {"id": ..., "version": ..., "hash": ...}` and
`provenance["overrides"]`, a dict from dotted key to value for every field that differs from the
base profile (for example `{"projection.angles.on_uncovered": "exclude"}`). `list_profiles()`
returns the `id@version` labels. A user settings file has a top-level `extends = "<id>@<version>"`
(a bare id means the latest shipped) plus sections; `save_settings(path, *, extends, overrides)`
writes only the overrides; `settings=` and `--settings` accept such a file.

**Re-running.** Running an operation again, in the library, through `workspace.run` or from the
CLI, uses the current inputs and settings. An existing run's sidecar is a record, never an input.
"Reopening gives identical numbers" means that reopening a project and reading a saved run returns
identical numbers; recomputing after a settings edit is a new run.

#### 6.3.5 Hashing (commitment 7)

**Normative.** One helper in `provenance` produces every hash: `h1-sha256:<64 lowercase hex>`.

| Input | Bytes hashed |
|---|---|
| A file | its raw bytes |
| A pydantic model or plain JSON value | `json.dumps(model.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")` (`by_alias=True` amended at S1 plan review, 2026-10-31, §6.2.1) |
| An in-memory frame (`_frames.frame_hash`) | Normalise first, then hash. Every numeric column (integer or float, nullable or not) becomes `float64` with null as NaN; every boolean column without nulls stays `bool`; every other column, including a boolean column with nulls, becomes text with nulls kept as nulls. Nulls (None, NaN, NaT, `pd.NA`) are found first; text comes from each value's Python form (the series' `to_list()`, never a backend cast to string): `str` as is, datetime, date and time values truncated to whole microseconds and then `.isoformat()` (backends differ below a microsecond), a categorical value by these same rules applied to its category's value, any other value `str()` (so booleans are `True`/`False` in every backend). Then hash a header, the canonical JSON of `{"columns": [...], "dtypes": [...], "n_rows": n}` with dtype names `float64`, `bool` or `string`, then a 0x00 byte, then per column: `float64` as little-endian bytes with every NaN written as the quiet NaN `0x7FF8000000000000`; `bool` as `uint8`; text value by value, a non-null value as its UTF-8 byte length (little-endian `uint64`) followed by its UTF-8 bytes, a null as the `uint64` `0xFFFFFFFFFFFFFFFF` alone. (Amended at S1 plan review, 2026-10-31: the earlier "UTF-8 values joined by U+001F, null as U+0000" let different frames hash alike, across columns, on values holding U+001F, and between null and `"\x00"`.) |
| A numeric array record (`Density`) | The canonical JSON of `{"arrays": ["x", "y", "z", "z_points", "covariance"], "shapes": [...]}`, a 0x00 byte, then each array in that order as C-order little-endian `float64` with NaN canonicalised as above |
| A set of densities (`Densities`) | The canonical JSON of `{"groups": [keys in group order], "arrays": ["x", "y", "z", "z_points", "covariance"], "shapes": [[...] per group]}`, a 0x00 byte, then for each group in group order its five arrays in the `Density` order. Used as the upstream hash of a `ContourSet` (§6.12) |

"Same content" means equal after this normalisation, so a pandas `int64` column, a pandas float
column holding the same integers, and a polars `Int64` column with the same nulls all hash alike.
Integer and float columns are not told apart: that is deliberate, because pandas turns an integer
column with a missing value into float. A string column whose values look numeric stays text.

Non-finite floats never reach canonical JSON: `FrozenModel` rejects them (§6.2.1), and the helper
raises `ValueError` naming the key if a plain JSON value holds one.

A hash string with any other prefix raises `HashSchemeError`. A test greps the package and fails if
`hashlib` is imported anywhere but `provenance.py`. The pooch registry uses pooch's bare `sha256:`
form derived from the same digest.

Golden vectors, pinned in `tests/test_provenance.py`:

| Input | Hash |
|---|---|
| empty bytes | `h1-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| JSON `{}` | `h1-sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| JSON `{"b": [1.5, "é"], "a": 1}` | `h1-sha256:2bb202b35963ad9e57f87e5828e072cc53f4720ecd10a7d625f850411a31c4c6` |
| JSON `{"clip": "none"}` (the no-limit literal) | `h1-sha256:7c457b553a03102b7220154eb95650e02d2840d02f397a2d226afba1fa85a1dd` |
| JSON `{"a": inf}` | raises `ValueError` (no hash) |
| frame `{"i": [1, None, 3], "f": [0.5, 1.5, -2.0], "t": ["a", None, "é"]}` (pandas or polars) | `h1-sha256:0ef558dd8a05e2aa709ff3e49550a1d91eb4aeab621fde47153bfd3be5028ee8` |

(The third value is the hash of the bytes `{"a":1,"b":[1.5,"é"]}`, which shows key sorting and
`ensure_ascii=False`. All four values recomputed for draft 0.2.) The frame vector was pinned in S1
Task 3 under the amended text rule, in `tests/test_frames.py`, where the hand-computed value and
`frame_hash` agree; the builder adds the `Density` and `Densities` vectors in S5 and pins them
(amended at S1 plan review, 2026-10-31). The vectors are: one content (an
integer column with one missing value, a float column and a text column with a null) held as a
pandas frame, where the integer column becomes `float64`, and as a polars frame, where it stays
`Int64`, with one hash; one `Density` vector (its arrays include `z_points`); and one `Densities`
vector of two groups, which also shows that swapping the group order changes the hash.

#### 6.3.6 Provenance envelope (commitment 8)

Every result carries `settings` (its operation's model) and `provenance`, a `dict[str, object]`
built by `provenance.envelope(...)`. **Normative** keys:

| Key | Value |
|---|---|
| `soundscapy_version`, `python_version`, `platform` | strings |
| `libraries` | `{name: version}` from `importlib.metadata` for numpy, scipy, pandas, pyarrow, narwhals, pydantic, contourpy; plus polars when it is the input library, openpyxl when an xlsx was read, and matplotlib in figure sidecars (§6.18) |
| `hash_scheme` | `"h1"` |
| `data_refs` | list of `{kind, name, version, hash}` for shipped or registered data used (circumplex, angle sets, profile, dataset). An angle set passed as an object through `angle_sets=` is recorded with its full angle table as well (§6.9.2) |
| `created_utc` | ISO 8601 with `Z` |
| `operation` | the public function name, e.g. `"project"` |
| `input_hashes` | on results that take more than one data input (`correlate`): `{argument name: h1 hash}` for each extra input, such as a features frame or `coords` |
| `input_hash`, `mapping_hash`, `settings_hash` | `h1` hashes (`mapping_hash` is `""` where no mapping is involved). For a survey read from a file, `input_hash` is the raw file's hash (`SourceRef.hash`); for an in-memory source, its `frame_hash`; for a derived result, the upstream result's `input_hash`, except `contours`, whose `input_hash` is the `Densities` array hash |
| `upstream_run` | the run name the input came from, `""` when none |
| `upstream` | `{settings_hash, input_hash}` copied from the input result, on results derived from another result's record rather than from its data (`contours`, from its density, §6.12) |
| `quality_skipped` | on a `Survey` and results derived from it: the list of quality rules skipped because their columns are not mapped (§6.7), `[]` when none |
| `profile`, `profile_inherited`, `overrides` | §6.3.4; `profile` is `{id, version, hash}`, `profile_inherited` is present only when an explicit profile replaced the input's |
| `response_id_source`, `read` | on survey-derived results: §6.3.2, and the resolved `ReadOptions` |
| `id_columns`, `response_id_columns`, `response_key_blank_ok` | on survey-derived results: `{role: user column}` for each mapped id role, `{}` when none (how `split(by=<role>)` finds its column, §6.10); the columns the response id was built from (`[response]` for `column`, the `response_key` columns in key order for `key`, `[]` for `row`); the key columns that may be blank (`[]` unless the source is `key`). `attach` rebuilds ids from exactly these (§6.16). (Amended at S1 plan review, 2026-10-31: §6.10 and §6.16 named no place for them.) |

No `seed` key exists at M1 (no random function). The mapping can only gain keys. `show_versions()`
returns a multi-line string of the version keys and never prints.

#### 6.3.7 References and citation blocks (commitment 7)

Persisted references are `{name, version}` plus the content hash, never a path. Every shipped data
file has a `[citation]` table with `citation`, `doi_or_url`, `licence`, `date` and `notes` (all
StrictStr; empty string when unknown). Shipped files are registered in `_data/__init__.py` with
their `h1` hashes, and a test fails if a shipped file's bytes differ from its pin, so a released
file cannot be edited silently (plan rule 15). A corrected table ships as a new version.

#### 6.3.8 Frame boundary (commitment 10)

`_frames` uses `narwhals.stable.v2` and is the only module that imports pandas, polars or pyarrow
(import contract, §5.2.1). Its functions:

| Function | Behaviour |
|---|---|
| `to_eager(obj)` | `nw.from_native(obj, eager_only=True)`, never `pass_through`. A lazy frame raises `TypeError("lazy frames are not accepted; call .collect() first")`; a dict, Series or other object raises `TypeError` |
| `columns_float(frame, names)` | Cast to `Float64`, `to_numpy`, `np.array(..., dtype=float64, copy=True)`, `setflags(write=False)`. NaN and null are both missing. Boolean, object and text columns raise `CodingError` naming the column (text must go through `Coding`) |
| `column_text(frame, name)` | Values as Python `str` with nulls as `""` |
| `to_native(frame)` | The user's library out |
| `with_columns(frame, columns)` | Adds columns; raises `ColumnCollisionError` on an existing name |
| `new_frame(columns, like=None)` | pandas, or the library of `like` |
| `read_text_table(path, *, encoding, delimiter)`, `read_xlsx_text(path, options)`, `write_csv`, `write_xlsx`, `write_parquet`, `read_parquet` | §6.5, §6.16. (Amended at S1 plan review, 2026-10-31: `read_text_table` takes keywords, because `ReadOptions` lives in `survey`, above `_frames`; S7 adds `header_row` and `skip_rows` keywords to it when `ReadOptions` gains them, and `read_xlsx_text`'s signature is provisional until S7 gives it the same keyword form plus `sheet`.) |
| `frame_hash(frame)` | §6.3.5 |

narwhals errors (`InvalidOperationError`, `ColumnNotFoundError`, `DuplicateError`) are re-raised as
soundscapy errors with backend-neutral messages, because their text differs by backend. Copying on
every conversion covers pandas 2.x, where `to_numpy` returns a writable view of the user's frame.
Frames created from paths or from nothing are pandas (plan §4.10).

#### 6.3.9 Logging (NFR-13, owner row #25)

**Normative** mechanism:

```python
logger = logging.getLogger("soundscapy")
logger.setLevel(logging.INFO)

class _DefaultHandler(logging.StreamHandler):
    def emit(self, record):          # stream looked up at emit time, so notebooks capture it
        if logging.getLogger().handlers:
            return                   # the application configured logging: defer to it
        self.stream = sys.stderr
        super().emit(record)

logger.addHandler(_DefaultHandler())  # propagate stays True, so caplog and GUIs receive records
```

(Amended 2026-11-01, review F8, `design/reviews/2026-10-31-s1-tool-idiom-review.md`: the INFO
default is set only when the logger has no level, so an application's earlier
`logging.getLogger("soundscapy").setLevel(...)` is kept. A `logging.Filter` on the logger applies
the root logger's level once the root has handlers, unless `set_log_level` was called or the
`soundscapy` level is not the default. Without it, a logger's own level alone decides
propagation, so `logging.basicConfig(level=logging.WARNING)` did not silence soundscapy's INFO
lines, contrary to NFR-13's "or with standard logging configuration".)

`set_log_level(level)` sets the `soundscapy` logger's level and is the one silencing call; there is
no environment variable. The package never prints. Importing `soundscapy` touches no third-party
logger. Pooch's INFO lines ("Downloading file ...") are hidden inside `fetch` only: `fetch` adds a
filter that drops pooch records below WARNING to `pooch.get_logger()` before its pooch call and
removes it in `finally`, so pooch's warnings still show and pooch's logger is unchanged afterwards.
The `fetch` docstring says so. A filter, not `setLevel`, because pooch's logger is a bare
`logging.Logger("pooch")`, not registered with `logging`, so it keeps its level cache: after its
first message a later `setLevel(WARNING)` is ignored (verified with pooch 1.9.0: an INFO line, then
`setLevel(WARNING)`, then a second INFO line, prints both; with the filter the second is dropped
and a WARNING still prints). M1 log lines
are: one INFO summary per `read_survey` (rows read, blank rows skipped, included, excluded by
reason), and one INFO line per download (§6.20). Conditions the user must act on are warnings or
errors, never log lines.

#### 6.3.10 Errors and warnings (commitment 14)

Roots: `SoundscapyError(Exception)` and `SoundscapyWarning(UserWarning)`. Every error also
subclasses the built-in named below and takes `(message, *, hint: str = "")`; `str()` gives the
message, and the CLI prints the hint on a second line. No name shadows a standard-library one,
which is why the integrity error is `DataIntegrityError` (not `IntegrityError`, which
`sqlite3` defines).

| Class | Built-in | Raised when |
|---|---|---|
| `MappingError` | ValueError | mapping invalid or incomplete; unknown construct key; bad language code |
| `CodingError` | ValueError | text matches no missing token, value-map key or number |
| `EncodingError` | ValueError | decode failure (hint: `encoding="cp1252"`) |
| `FileFormatError` | ValueError | delimiter cannot be resolved; unknown suffix; multi-sheet workbook without `ReadOptions.sheet` |
| `ColumnCollisionError` | ValueError | an input column starts with `ssp_`; `attach` or `join_context` would overwrite |
| `AxesError` | ValueError | a plot function is given an `ax` of the wrong kind (`radar` needs polar axes, §6.15) |
| `AlignmentError` | ValueError | `attach` cannot align a result with the frame (§6.16) |
| `ScaleViolationError` | ValueError | coded values outside the declared scale (§6.4.3) |
| `MissingResponseIdError`, `DuplicateResponseIdError` | ValueError | §6.3.2 |
| `UncoveredLanguageError`, `LowConfidenceLanguageError` | ValueError | §6.8.3; the uncovered message names a tier-only language's tier |
| `FormulationError` | ValueError | `iso2019` on a scale other than 1..5 or with non-nominal angles (`canonical` and `aletta2024` accept any scale and any non-degenerate angles); any formulation given angles whose Σ\|cos θ_i\| or Σ\|sin θ_i\| is below 1e-9 (§6.9.1; amended at S1 plan review, 2026-10-31) |
| `ProfileConstraintError` | ValueError | §6.3.4 |
| `SettingsError` | ValueError | a settings file or section fails validation |
| `DataReferenceError` | LookupError | unknown circumplex, angle set, profile or dataset version |
| `SmallSampleError` | ValueError | §6.12, §6.15 |
| `DensityEstimationError` | ValueError | collinear or identical points |
| `DataIntegrityError` | ValueError | a hash does not match (names both hashes) |
| `HashSchemeError` | ValueError | unknown hash prefix |
| `DownloadError` | OSError | HTTP or connection failure |
| `ProjectFormatError` | ValueError | project files missing, malformed or of an unknown format version |
| `RunExistsError` | FileExistsError | `save_run` without `overwrite=True` |
| `OutputFileLockedError` | PermissionError | the target file is open elsewhere (hint: close it in Excel) |

| Warning | Fires |
|---|---|
| `MediumConfidenceLanguageWarning` | once per `project`, naming Medium-tier languages and counts |
| `EqualAnglesFallbackWarning` | once per `project` when `equal` fallback was applied, naming languages and counts |
| `SmallSampleWarning` | once per `describe` or `density` call when a group has `n_valid < 30` and the user has not opted in |
| `PointsOutsideGridWarning` | once per `density` call when valid points fall outside the clip box; M1 coordinates stay in the square, so only a user-set clip narrower than the data triggers it |

`FileFormatError`, `RunExistsError`, `AlignmentError` and `AxesError` are additions to the decision
list; each has one raising condition at M1.

### 6.4 Survey mapping and coding (commitment 4, FR-1.3)

#### 6.4.1 Fields

| Model | Fields |
|---|---|
| `SurveyMapping` | `schema: Literal["soundscapy.mapping"]` (amended at S1 plan review, 2026-10-31: the Python field is `schema_name` with `Field(alias="schema")` and `populate_by_name=True`, because a field named `schema` warns "shadows an attribute in parent BaseModel", fatal under `filterwarnings = ["error"]`; the TOML and hashed key stays `schema`, and every model with a `schema` key, circumplex (§6.2.3), angle set, profile and dataset registry, does the same), `schema_version: Literal[1]`, `instrument: str = "iso-12913-outdoor"`, `instrument_version: str = "1"`, `ids: IdSpec`, `language: LanguageSpec`, `groups: Groups`, `age: str = ""`, `flags: FlagSpecs`, `carry: tuple[str, ...] = ()` |
| `IdSpec` | `response: str = ""`, `response_key: tuple[str, ...] = ()` (not both), `response_key_blank_ok: tuple[str, ...] = ()` (key columns that may be blank, a proper subset of `response_key`, §6.3.2), `location`, `session`, `group`, `record: str = ""` |
| `LanguageSpec` | `column: str = ""` or `value: str = ""` (not both), `coding: tuple[tuple[str, str], ...] = ()` mapping raw text to ISO 639-3. Every resulting code must match `^[a-z]{3}$`, else `MappingError`. A blank cell is unmapped and gives `""`; it matters only for a row that is projected with adjusted angles (§6.8.3) |
| `Groups` | `attributes: ItemGroup` (required), `source_dominance`, `overall_quality`, `appropriateness`, `who5: ItemGroup` (each optional: an empty group with no columns means not mapped) |
| `ItemGroup` | `columns: tuple[tuple[str, str], ...]` (item key to column name), `scale: Scale` (defaults per group, below), `coding: Coding = Coding()` |
| `Coding` | `missing_values: tuple[str, ...] = ("", "NA", "N/A", "NaN", "nan", "#N/A")`, `value_map: tuple[tuple[str, StrictFloat], ...] = ()`, `decimal: Literal[".", ","] = "."`, `offset: StrictFloat = 0.0`, `reverse: tuple[str, ...] = ()` (item keys) |
| `FlagSpecs` | `consent`, `test`, `paper_input: FlagSpec` with `column: str = ""`, `true_values: tuple[str, ...] = ()` |

Item keys per group: `attributes` takes a position (`PAQ1`) or a reference-language label
(`pleasant`), normalised to positions on validation; every position of the circumplex must be
mapped. `source_dominance`: `traffic_noise`, `other_noise`, `human_sounds`, `natural_sounds`.
`overall_quality`: `overall_quality`. `appropriateness`: `appropriateness`. `who5`: `who5_1` to
`who5_5`. Default scales: the circumplex scale for attributes; 1..5 ordinal for source dominance,
overall quality and appropriateness; 0..5 ordinal for WHO-5. The mapping hash covers the normalised
model, so `pleasant = "x"` and `PAQ1 = "x"` hash equally.

#### 6.4.2 Coding order

**Normative**, per cell of a mapped item column (plan §4.4; the missing-token and parse steps only
come before the plan's value map, offset, reverse):

1. Strip whitespace. A value in `missing_values` is missing (NaN).
2. A key of `value_map` is replaced by its number.
3. Otherwise the text is parsed as a number, with `decimal` as the decimal mark. Failure raises
   `CodingError` listing the distinct offending texts, counts and up to five response ids, with the
   hint to add a `value_map` for labels or a missing code.
4. `offset` is added.
5. For item keys in `reverse`: `score = minimum + maximum - score`, against the group's scale.

For an in-memory frame whose column is already numeric, steps 1 to 3 reduce to taking the number
(NaN and null are missing); a boolean column raises `CodingError`. The scale check (§6.4.3) runs
after step 5.

#### 6.4.3 Scale violations (FR-2.1)

A coded value fails when `Scale.contains` is false. By default `read_survey` raises
`ScaleViolationError`; the message gives, per column, the offending values with counts and up to
five response ids, and the hint lists the fixes: add a missing code, set `offset`, or pass
`on_scale_violation="exclude"`. With `exclude`, each violating cell becomes missing and the response
gets reason `scale_violation`; no warning fires, because the choice was explicit (plan rule 10).
The ISD has no violations.

#### 6.4.4 TOML

`read_mapping(path)` and `write_mapping(mapping, path)` use tomlkit. A blank value (`""`) where a
draft left a TODO raises `MappingError` naming the key.

**Normative** TOML form. Every key-to-name field (`ItemGroup.columns`, `LanguageSpec.coding`,
`Coding.value_map`) is written as a TOML table (`pleasant = "pleasant"`), not as an array of
pairs. The model keeps `tuple[tuple[K, V], ...]` so it hashes (§6.2.1); a `mode="before"` field
validator turns a table into pairs, and a field serializer turns pairs back into a table for
`model_dump`. The writer's rules make the output deterministic:

| Rule | Effect |
|---|---|
| Identity keys always written | `schema`, `schema_version`, `instrument`, `instrument_version`, even at their defaults |
| Other fields at their default are omitted | no empty `[flags]`, `coding` or `scale` tables |
| Top-level scalars and arrays first, then tables | each in field order (TOML requires plain keys before the first table) |
| Attribute keys written as the instrument's reference-language labels | `PAQ1` is written `pleasant`; the model holds positions, so the hash is unchanged |

A test reads the shipped `isd_mapping.toml`, validates it, writes it back and asserts the bytes
are identical. A prototype of these rules (pydantic and tomlkit, the models of §6.4.1) reproduces
the example below byte for byte and reads back to an equal model (verified).

**Normative** example (the ISD mapping, shipped as `datasets/isd_mapping.toml`):

```toml
schema = "soundscapy.mapping"
schema_version = 1
instrument = "iso-12913-outdoor"
instrument_version = "1"
age = "age00"
carry = ["start_time", "Survey_Version", "sss03", "sss04", "sss05"]

[ids]
response_key = ["LocationID", "SessionID", "GroupID", "RecordID"]
location = "LocationID"
session = "SessionID"
group = "GroupID"
record = "RecordID"

[language]
column = "Language"

[groups.attributes.columns]
pleasant = "pleasant"
vibrant = "vibrant"
eventful = "eventful"
chaotic = "chaotic"
annoying = "annoying"
monotonous = "monotonous"
uneventful = "uneventful"
calm = "calm"

[groups.source_dominance.columns]
traffic_noise = "ssi01"
other_noise = "ssi02"
human_sounds = "ssi03"
natural_sounds = "ssi04"

[groups.overall_quality.columns]
overall_quality = "sss01"

[groups.appropriateness.columns]
appropriateness = "sss02"

[groups.who5.columns]
who5_1 = "who01"
who5_2 = "who02"
who5_3 = "who03"
who5_4 = "who04"
who5_5 = "who05"
```

The ten acoustic columns of the subset are added to `carry` by `datasets.isd` (the full table's
96 are carried when `load(subset=False)`); their units are marked unverified in the sidecar. The
feature column names differ by ISD version (1.0.1-alpha.1 drops 1.0.1-alpha's `_Max` suffix, and
v0.2.3 uses names such as `LAeq_L(A)(dB(SPL))`, §6.20.3), so the registry, not the mapping, lists
each version's feature columns. v0.2.3 has its own mapping file, `isd_v0.2.3_mapping.toml`
(§6.20.3).
Attributes map by name, never by column order: the CSV order (pleasant, chaotic, vibrant, ...)
differs from the circumplex order.

### 6.5 Reading a survey (FR-1.1)

**Normative** signature:

```python
def read_survey(
    source: str | os.PathLike[str] | IntoDataFrame,
    mapping: SurveyMapping | str | os.PathLike[str],
    *,
    read_options: ReadOptions | Mapping[str, object] | _Unset = _UNSET,
    quality: QualitySettings | Mapping[str, object] | _Unset = _UNSET,
    on_scale_violation: Literal["error", "exclude"] | _Unset = _UNSET,
    settings: ImportSettings | Mapping[str, object] | _Unset = _UNSET,
    profile: str | _Unset = _UNSET,
) -> Survey: ...
```

Every optional argument uses the `_UNSET` sentinel (§6.3.4), never `None`. The sheet of a workbook
is `ReadOptions.sheet`; there is no separate `sheet` argument, here or in `suggest_mapping`. Left
unset, `profile` follows steps 3 and 4 of the §6.3.4 rule (a settings file's `extends`, else
`sal-2026`), because a source records no profile.

`ReadOptions` (a FrozenModel): `format: Literal["auto", "csv", "tsv", "xlsx"] = "auto"` (by
suffix); `sheet: str = ""`; `encoding: str = "utf-8-sig"`; `delimiter: str = "auto"`;
`decimal: Literal[".", ","] = "."`; `header_row: StrictInt = 1`; `skip_rows: tuple[int, ...] = ()`
(physical row numbers to skip, e.g. Qualtrics' label rows). The resolved options (the sniffed
delimiter, the sheet actually read) are what is persisted and recorded in `provenance["read"]`.

Behaviour:

| Step | Rule |
|---|---|
| Collision check | any input column starting with `ssp_` raises `ColumnCollisionError` |
| CSV and TSV | read with the standard `csv` module so record numbers are known; every cell is text. Decode errors raise `EncodingError` (hint: try `encoding="cp1252"`). `delimiter="auto"` sniffs the first 64 KiB among `,`, `;`, tab and `|`; failure raises `FileFormatError` asking for `delimiter=` |
| xlsx | `openpyxl.load_workbook(read_only=True, data_only=True)`. A workbook with more than one sheet and an empty `ReadOptions.sheet` raises `FileFormatError` listing the sheet names. Cell to text: int as `str(int)`; float as `repr(float)`; datetime as ISO 8601; bool as `TRUE`/`FALSE`; empty as `""`. Ids with leading zeros survive only if stored as text in Excel; the docs say so |
| Blank rows | rows whose every cell is empty after stripping are skipped and counted. Trailing blank rows in xlsx are dropped by every reader; the docs say so |
| Parsing | only mapped construct columns are coded (§6.4.2); ids, language and carried columns stay text, so no id or code is coerced |
| Ids, language | §6.3.2; language coded via `LanguageSpec` |
| Quality | §6.7, inside `read_survey`, writing `Status` |
| Log | one INFO summary line (§6.3.9) |

`Survey` fields: `data` (a copy of the input as read: text columns from a file, the user's frame
copied for an in-memory source), `mapping` (normalised), `scores: dict[str, ndarray]` (per group, a
read-only `(n, k)` float64 array in item-key order; attributes in position order), `response_id`
(read-only array of str), `ids: dict[str, ndarray]` (id roles), `language` (array of 639-3 codes,
or `""` when not mapped), `status: Status`, `settings: ImportSettings`, `source:
SourceRef(name, hash, read_options)`, `provenance`. Methods: `n`, `to_frame()` (the data with
`ssp_response_id` and status columns, in the input's library), `select(mask)`.

#### 6.5.1 Location and session tables (FR-1.4)

Plan §4.4's minimal hierarchy: a dataset may carry optional location and session tables, keyed by
the mapping's id columns and joined only on request. Any level may be missing.

| Function | Behaviour |
|---|---|
| `survey.join_context(data, table, level)` | `data` is a `Survey` or a `Coordinates`, the two records that carry id roles; anything else raises `TypeError`. `level` is `"location"` or `"session"`. `table` is a frame (or a path, read as text like §6.5) whose key column has the name the mapping gives that id role. Returns a `Table`, one row per response in `data`'s row order: `response_id`, the role's key, and the table's other columns, left-joined. A role the mapping does not name raises `MappingError`; a duplicate key in `table` raises `DuplicateResponseIdError` naming the key; a key in `data` absent from `table` gives nulls, and the `Table.notes` count the unmatched keys and rows; a column name already present raises `ColumnCollisionError`. One INFO log line gives the level, the matched and unmatched counts, as `attach` does (§6.16) |
| `workspace.add_table(project, dataset, level, path)` | Copies the file byte for byte to `datasets/<name>/context/<level>.<suffix>`, records its hash in `dataset.toml` and the manifest, and validates the key column against the dataset's mapping |

`correlate(..., features=<table>, on="location")` takes the joined `Table`, and `attach` adds its
columns to the user's frame, so course case 2's site-level variables need no special code. Recording
tables join at M3. A test joins a location table to the subset, joins a session table to a survey
with only location ids (the session level is then missing and the call raises `MappingError` naming
the role), and checks that the row count never changes.

### 6.6 Mapping suggestion and templates (FR-1.2, FR-1.3a)

#### 6.6.1 `suggest_mapping`

```python
def suggest_mapping(
    source: str | os.PathLike[str] | IntoDataFrame,
    *,
    schema: Literal["auto", "ssid", "redcap", "qualtrics", "canonical"] = "auto",
    instrument: str = "iso-12913-outdoor",
    read_options: ReadOptions | Mapping[str, object] | _Unset = _UNSET,
) -> MappingDraft: ...
```

Matching, in order, stopping at the first unambiguous hit per construct: (1) normalised exact names
(lowercase, spaces, hyphens and dots to `_`); (2) alias tables in `_data/schemas/<schema>.toml`; (3)
reference-language labels of the circumplex. `auto` tries canonical, then ssid, then redcap, then
qualtrics, and records which won. Two or more candidates for one construct are listed as ambiguous
and never picked. The draft also detects value labels: a column whose distinct texts are the Table
A.1 labels (any case) gets a proposed `value_map`; a numeric column whose observed range is the
scale shifted by a constant gets a proposed `offset`.

`MappingDraft` (dataclass): `proposed` (construct to column), `ambiguous` (construct to candidate
columns), `unmatched_columns`, `value_maps`, `offsets`, `schema_used`. `to_mapping()` returns a
`SurveyMapping` or raises `MappingError` listing what is missing. `to_toml()` writes a draft file in
which every unresolved key has the value `""` and a `# TODO: choose one of: ...` comment (tomlkit
can write comments; tomli-w cannot). Notebook editing, TOML editing and the CLI's interactive mode
all end in the same `mapping.toml`; a test shows the CLI and notebook paths produce byte-identical
files (FR-1.3a).

#### 6.6.2 Schemas

`ssid.toml` holds the ISD column names. `redcap.toml` holds the SSID REDCap dictionary names when
the owner supplies them; until then it holds the ISD names and the file is marked
`provisional = true` (§11.2). `canonical` is the template's names (§6.6.3).

`qualtrics.toml` (PRD FR-1.3a names Qualtrics) is a minimal alias table, marked
`provisional = true` like the REDCap one: Qualtrics' fixed metadata columns (`ResponseId` to the
`response` id, `UserLanguage` to the language column, `StartDate` carried), and a proposed
`LanguageSpec.coding` from Qualtrics' language codes to ISO 639-3 for the languages of the shipped
angle sets (for example `EN` to `eng`, `ES-ES` to `spa`, `ZH-S` to `cmn`). Question columns in a
Qualtrics export are named by the survey author, so attributes match by step (3) or stay
unmatched. When row 3 of the file starts with Qualtrics' `{"ImportId":` marker, the draft proposes
`ReadOptions.skip_rows=(2, 3)` for the two label rows. A how-to page walks through a Qualtrics
export with this draft.

#### 6.6.3 Canonical template

`_data/templates/canonical-survey.csv` (header only, UTF-8 with BOM, so Excel opens it cleanly),
`canonical-survey.xlsx` (the header, data-validation dropdowns for every coded column, and a
`codebook` sheet listing each column, its construct, scale and labels) and `canonical-mapping.toml`.
Columns, in order: `response_id`, `location_id`, `session_id`, `group_id`, `language`, `pleasant`,
`vibrant`, `eventful`, `chaotic`, `annoying`, `monotonous`, `uneventful`, `calm`, `traffic_noise`,
`other_noise`, `human_sounds`, `natural_sounds`, `overall_quality`, `appropriateness`, `who5_1` to
`who5_5`, `age`, `consent`. `write_template(path, *, format="csv")` copies them out; the CLI
`template` command wraps it. The names are a call made here (§11.2), changeable until the course
material is written.

### 6.7 Quality rules (FR-1.5)

`QualitySettings` holds one `Action` (`"exclude"`, `"flag"` or `"off"`) per rule plus thresholds.
**Normative** defaults. They reproduce the processing the 0.x tutorials applied to ISD v1.0, which
is FR-1.5's acceptance test (PRD 0.9), and follow the 2021 JASA paper's rules where the 2022 code
got them wrong (§6.24):

| Rule | Default | Definition | Writes |
|---|---|---|---|
| `all_identical` | exclude | all attribute items present and equal, and the value is not the scale midpoint | reason `all_identical` |
| `all_midpoint` | flag | all attribute items present and equal to the midpoint (kept, per the 2021 rule) | flag `all_midpoint` |
| `excessive_missing` | exclude | missing share over mapped attribute and source-dominance items `> excessive_missing_share` (default 0.5; 12 items when both groups are mapped). From the 2021 paper's text; it fires on no ISD row | reason `excessive_missing` |
| `incomplete_attributes` | always | any attribute item missing (FR-2.5a; not configurable at M1) | reason `incomplete_attributes` |
| `no_consent` | exclude | consent column mapped and value not in `true_values` | reason `no_consent` |
| `test_response` | exclude | test column mapped and value in `true_values` | reason `test_response` |
| `paper_input` | flag | paper column mapped and value in `true_values` | flag `paper_input` |
| `under_age` | flag | age mapped, parsed as a number, and `< min_age` (default 18) | flag `under_age` |

A rule whose columns are not mapped is skipped, and the skip is recorded in the survey's
provenance as `quality_skipped: [rule, ...]`. The code written follows the action: a flag code
under `flag`, a reason code under `exclude`. So `under_age` and `paper_input` exist in both enums,
and the default can change without a schema change. The default for `under_age` is `flag` (a call
made here, §11.2): it keeps the rows, as the 0.x tutorials did (their 56 exclusions leave the 52
under-age rows in).

**Where the rules come from.** The ISD publishes no exclusion counts: its Zenodo `Scripts.zip`
compiles and recodes the data only. The 0.x code never implemented a "more than half missing" rule
(the early 2022 code filled missing PAQ items with 3 and left a TODO); the rule appears only in the
text of the 2021 paper. So `excessive_missing` keeps its definition and default, documented as from
the 2021 paper, and stays configurable. Its denominator is the 2021 paper's wording, the attribute
plus source-dominance items, 12 when both groups are mapped (a call made here, §11.2).

**Pinned counts** (verified on ISD 1.0.1-alpha, Zenodo record 10639661, byte-identical to the CSV
bundled on 0.x `origin/main`; they hold unchanged for the default 1.0.1-alpha.1, record 10672568,
whose survey columns are identical cell for cell on the same 3589 keys, §6.20.3):

| Data | rows | `incomplete_attributes` | `all_identical` | `all_midpoint` | `excessive_missing` | `under_age` | excluded | Run |
|---|---|---|---|---|---|---|---|---|
| Full ISD 1.0.1-alpha.1 (and 1.0.1-alpha) | 3589 | 37 | 19 | 53 | 0 | 52 | 56 | nightly, `network` |
| Subset (Option C, from 1.0.1-alpha.1) | 1552 | 16 | 0 | 10 | 0 | 8 | 16 | default run |

**The FR-1.5 acceptance test** (PRD 0.9) is that the default rules reproduce the 0.x tutorials'
processing of ISD v1.0: 56 excluded, which is 37 with any attribute item missing
(`incomplete_attributes`) plus 19 all-identical non-midpoint responses (`all_identical`), and 53
all-midpoint rows flagged, not excluded; `excessive_missing` fires 0 times. It runs in two parts:
a default-run science test on the committed subset, with the subset counts above, and a nightly
`network` test on the full ISD, with the full counts. Neither waits for an owner input, and neither
is a release blocker beyond passing like any other test. All counts were re-checked on the
downloaded 1.0.1-alpha.1 CSV for draft 0.4.

### 6.8 Angle sets and their resolution (FR-2.4, FR-2.4a, commitment 2)

#### 6.8.1 File format

**Normative.** `_data/angle_sets/<name>/<version>.toml`; a user set uses the same format.

```toml
schema = "soundscapy.angle_set"
schema_version = 1
name = "aletta-2024"
version = "1"
circumplex = "iso-12913-outdoor"
circumplex_version = "1"

[citation]
citation = "Aletta, F., Mitchell, A., Oberman, T., Kang, J., et al. (2024). Soundscape descriptors in eighteen languages: Translation and validation through listening experiments. Applied Acoustics 224, 110109."
doi_or_url = "https://doi.org/10.1016/j.apacoust.2024.110109"
licence = "CC-BY-4.0 (the paper); BSD-3-Clause (the source repository MitchellAcoustics/SATP-circumplex-validation)"
date = "2024"
notes = "Angles: Table 4, p. 12, generated by scripts/make_aletta_2024.py from SATP-circumplex-validation@3509da3 outputs/2024-06-13/sem-fit-ipsatized.csv ('Equal comm.' rows). Tiers: ..."

[languages.eng]
tier = "high"
angles_deg = [0, 46, 94, 138, 177, 231, 275, 340]

[languages.fra]
tier = "medium"
note = "Did not pass Step 2 (Table 2, p. 10); no adjusted angles."
```

A language may be listed with a tier and no `angles_deg` (a **tier-only** language). It records
that the source assessed the language and what it concluded; the resolver treats it as uncovered
(§6.8.3), and the error or warning names its tier.

`AngleSet` (FrozenModel) validates: `angles_deg` is empty (tier-only) or has one value per
circumplex position, each in `[0, 360)`, strictly increasing; a `high` language has angles; `tier`
in `high`, `medium`, `low`; language keys match `^[a-z]{3}$`. Identity is `(name, version)` plus the
circumplex it applies to; `AngleSetRef(name, version)` is what settings hold, and its label is
`name@version`. A released version is never edited; a corrected table becomes version 2 (plan rule
15).

#### 6.8.2 `aletta-2024` version 1

`scripts/make_aletta_2024.py` reads `outputs/2024-06-13/sem-fit-ipsatized.csv` from
`MitchellAcoustics/SATP-circumplex-validation` at commit
`3509da3e9191586430582e42f5f86945aa4965c5` (sha256
`d3ad6da994013082090c27e4592ae5bcdf267a91e5e5fe2c7eb5804eef25df05`, checked), keeps the rows with
`Model Type == "Equal comm."` for the 12 languages that pass the paper's fit rule (CFI ≥ 0.92,
GFI ≥ 0.90, SRMR < 0.08; it asserts the rule selects exactly these), and reflects a language
(θ to (360 − θ) mod 360) when its raw PAQ2 angle exceeds its raw PAQ3 angle. Expected output,
PAQ1..PAQ8 in degrees, pinned by a science test:

| Lang | Angles | Lang | Angles |
|---|---|---|---|
| arb | 0, 36, 45, 135, 167, 201, 242, 308 | ind | 0, 53, 104, 123, 139, 202, 284, 308 |
| cmn | 0, 18, 38, 154, 171, 196, 217, 318 | ita | 0, 57, 104, 142, 170, 274, 285, 336 |
| deu | 0, 64, 97, 132, 182, 254, 282, 336 | nld | 0, 43, 111, 125, 174, 257, 307, 341 |
| ell | 0, 72, 86, 133, 161, 233, 267, 328 | spa | 0, 41, 103, 147, 174, 238, 279, 332 |
| eng | 0, 46, 94, 138, 177, 231, 275, 340 | swe | 0, 66, 87, 146, 175, 249, 275, 335 |
| hrv | 0, 84, 93, 160, 173, 243, 273, 354 | tur | 0, 55, 97, 106, 157, 254, 289, 313 |

These are the published values (owner decision, §4.3): English monotonous is 231° and Italian
chaotic 142°, as in the machine-readable output.

**Checked against the paper.** Table 4 of the paper (p. 12, in `design/literature/`) was read from
the rendered page on 2026-10-31. Its adjusted angles for the 12 languages above equal the generated
table in every cell. Table 4 also assigns a confidence tier to six more languages, and two of its
rows disagree with the paper's own rule:

| Language | Table 2 (p. 10): Step 1 CI, carried to Step 3 | Table 4 tier | Tier by the stated rule | Shipped in `aletta-2024@1` |
|---|---|---|---|---|
| fra | 0.931, No | Medium | Medium | `medium`, tier-only |
| por | 0.764, No | Medium | Medium | `medium`, tier-only |
| jpn | 0.833, No | Low | Medium (passed Step 1) | `medium`, tier-only; note cites both |
| kor | 0.819, No | High, with angles 0, 56, 90, 124, 151, 251, 275, 288 | Medium | `medium`, tier-only; note cites both and keeps Table 4's angles as text |
| vie | 0.694, n/a | Low | Low | `low`, tier-only |
| zsm | 0.674, n/a | Low | Low | `low`, tier-only |

The stated rule is Table 4's caption (Low: not passing Step 1; Medium: not passing Step 2; High:
the rest) with Table 2's thresholds (Step 1: CI > 0.7; Step 2: CFI > 0.92, GFI > 0.90 and
SRMR < 0.08). The paper's §5.1 (p. 13) also lists Korean among the Medium or Low languages, and
SATP v1.5 lists both jpn and kor as Medium (§6.8.4), so the rule wins (a call made here, §11.2).
The generator holds a transcription of Table 2 and assigns the six tiers from it, asserting the
result above. The CFI threshold is applied as ≥ 0.92 because Table 2 carries spa (CFI 0.920) to
Step 3. The 12 High languages carry no note. The name is `aletta-2024@1` (a call made here, §11.2).

Only the generated `aletta-2024/1.toml` is committed. The default run checks the committed file
against its pinned hash in the `_data` registry and its angles against the table above. The
regeneration test is marked `reference` (nightly): it downloads the CSV at the pinned commit,
checks its sha256, regenerates the file and asserts byte equality. The generator also accepts a
local path to the CSV (`--source PATH`), checking the same hash, so it runs offline.

#### 6.8.3 `AngleSelection` and the resolver

| Field | Type | Default |
|---|---|---|
| `default` | `AngleSetRef \| Literal["equal"]` | from the profile |
| `per_language` | `tuple[tuple[str, AngleSetRef], ...]` | `()` |
| `on_uncovered` | `Literal["error", "exclude", "equal"]` | `"error"` |
| `on_low_confidence` | `Literal["error", "exclude", "equal"]` | `"error"` |

Per response, the set is `per_language[lang]` if present, else `default`. If the chosen set is
`"equal"`, the circumplex's nominal angles are used and language is not consulted. Otherwise:

| Language in chosen set | Result |
|---|---|
| tier high | projected; `angle_set` label `name@version` |
| tier medium | projected; flag `medium_confidence_language`; one `MediumConfidenceLanguageWarning` |
| tier low | per `on_low_confidence`: `error` raises `LowConfidenceLanguageError`; `exclude` writes reason `low_confidence_language`; `equal` projects with nominal angles, flag `equal_angles_fallback`, label `equal`, one `EqualAnglesFallbackWarning` |
| absent, or tier-only (listed with a tier and no angles) | per `on_uncovered`, the same three branches with `UncoveredLanguageError` and reason `uncovered_language`. For a tier-only language the message names its tier, for example "kor (Medium in satp@1.5, no adjusted angles): 12 responses" |
| language not mapped | `MappingError`: adjusted angles need a `[language]` column or value |
| language cell blank (`""`) on a row still included | `MappingError` naming the rows. A row already excluded (for example by `incomplete_attributes`, as v0.2.3's recording-only rows are) is not projected, so its blank language is never consulted |

Errors list each language with its count and name the overrides. Sets resolve from the shipped
`_data` registry, from the project's `angle_sets/` (§6.18), or from `angle_sets=` (a tuple of
`AngleSet` objects passed to `project`); there is no global registry. A set whose circumplex
differs from the survey's raises `DataReferenceError`.

#### 6.8.4 `satp` version 1.5

`_data/angle_sets/satp/1.5.toml` ships at M1, written by hand from the chapter notes
(`design/literature/mitchell-et-al-2026-satp-chapter-4-notes.md`), because no machine-readable
source exists. Its citation block names Mitchell, A., Oberman, T., Aletta, F. and Kang, J. (2026),
"Soundscape Translation Data: Analysis, Validation, and Application", chapter 4 of *Soundscape
Across Languages* (Springer Nature), and the proof pages: Table 4.7 (tiers) p. 19, Table 4.8
(angles) p. 20, §4.6.1 (cmn) p. 24. Its notes say the values come from the author proof dated
2026-08-13 and are to be checked against the published chapter.

| Tier | Languages | Angles |
|---|---|---|
| High (13) | arb, deu, ell, eng, hrv, ind, ita, nld, pol, spa, swe, tur, yue | Table 4.8. For the 11 High languages it shares with `aletta-2024@1` (eng among them; pol and yue are new), every angle is equal to that set's (checked) |
| Medium with angles (1) | cmn | Aletta et al. 2024's cmn angles (0, 18, 38, 154, 171, 196, 217, 318), citing §4.6.1: cmn passed Step 2 and its SEM results and angles are identical to Aletta 2024's; it moved to Medium because of Step 3 (a call made here, §11.2) |
| Medium, tier-only (6) | tha, por, bra, kor, fra, jpn | none: Table 4.8 lists only High languages |
| Low, tier-only (3) | alb, vie, zsm | none: the chapter advises against projecting them |

The chapter says Medium-tier "adjusted angles from Step 2 may still improve cross-language
comparability, though results should be interpreted with greater caution"; that is why cmn keeps
its angles and projects with a `MediumConfidenceLanguageWarning`. A science test parses the two
Markdown tables of the notes file and asserts that `satp/1.5.toml` holds the same 23 tiers and the
same 13 High rows, and that its cmn angles equal `aletta-2024@1`'s.

**Consequences.** Every ISD language is covered by `satp@1.5`: eng, spa, nld and ita are High, and
cmn (1891 of 3589 rows, 52.7%; 69 subset rows, all at PingshanPark) is Medium. So under the default
profile the subset projects every included row, flags the cmn rows `medium_confidence_language`, and
`project` raises one `MediumConfidenceLanguageWarning`. Low-tier and uncovered behaviour is shown
with the synthetic test set (§6.22).

### 6.9 Projection (FR-2.3)

#### 6.9.1 Kernel

**Normative.** `projection.formulas.project_scores(scores, angles_deg, scale, formulation) ->
ndarray` takes `scores` of shape (n, k), `angles_deg` of shape (n, k) (per-response angles), a
`Scale` and a formulation, and returns (n, 2) float64. Rows with any NaN return NaN.

Canonical (Mitchell 2025; chapter 4, 2026). With μ = `scale.midpoint` and h = `scale.half_range`
taken from the declared scale, never from the data:

```
x = Σ_i cos θ_i (s_i − μ) / (h · Σ_i |cos θ_i|)
y = Σ_i sin θ_i (s_i − μ) / (h · Σ_i |sin θ_i|)
```

ISO/TS 12913-3:2019 (`iso2019`), with c = cos 45°:

```
x = [(PAQ1 − PAQ5) + c(PAQ8 − PAQ4) + c(PAQ2 − PAQ6)] / (4 + √32)
y = [(PAQ3 − PAQ7) + c(PAQ4 − PAQ8) + c(PAQ2 − PAQ6)] / (4 + √32)
```

Aletta et al. 2024 (`aletta2024`), the paper's Eqs. 1 to 3 (p. 14, in `design/literature/`), with
σ_i the raw, uncentred score and ρ = `scale.range` (maximum − minimum) from the declared scale:

```
x = Σ_i cos θ_i σ_i / λ_P,   λ_P = (ρ/2) Σ_i |cos θ_i|
y = Σ_i sin θ_i σ_i / λ_E,   λ_E = (ρ/2) Σ_i |sin θ_i|
```

`iso2019` raises `FormulationError` unless the scale is 1..5 and every response uses the nominal
45° angles (on 1..7 it would reach 1.5). canonical and iso2019 bound each axis separately,
`|x| ≤ 1` and `|y| ≤ 1`; the radius can exceed 1 (1.0824 at 45° on the 1..5 grid), so nothing clips
at the unit circle and nothing clamps. At 45° on 1..5 the two are algebraically identical
(Σ cos θ_i = Σ sin θ_i = 0 and Σ|cos θ_i| = 2 + 2√2); the exhaustive test of §7.2 checks all
390,625 responses. Sums are taken in position order with `numpy` float64 so results match across
platforms to 1e-12. Every formulation raises `FormulationError` naming the axis when, in any row,
Σ|cos θ_i| or Σ|sin θ_i| is below 1e-9 (a degenerate angle set; cos and sin of multiples of 90° leave
float64 residues up to 1.8e-16, so no exact zero test is used). Angle-set and circumplex validation
do not bound these sums, so the check is the kernel's (amended at S1 plan review, 2026-10-31).

`aletta2024` does not centre the scores, so the canonical formulation's guarantees do not hold for
it. When Σ cos θ_i ≠ 0 a neutral response does not map to the origin: under `aletta-2024@1`'s eng
angles, all items at 3 on 1..5 give (0.08155, 0.10640). That offset is the defect the canonical
formulation fixes. For the same reason its coordinates can leave the [−1, 1] square: with the eng
angles x reaches 1.0815 and y 1.1064 on the 1..5 grid. At 45° on 1..5 Σ cos θ_i = Σ sin θ_i = 0 and
λ_P = λ_E = 4 + √32, so it reduces to the 2019 formula exactly. It accepts any declared scale and
any non-degenerate angle set (above), and it emits no warning: choosing it is deliberate (a call made here, §11.2). No
profile uses it by default; the ISO profiles refuse it (§6.3.4). It exists for reproducing work
done with Aletta et al. 2024's equations.

#### 6.9.2 `project`

```python
def project(
    survey: Survey,
    *,
    formulation: Literal["canonical", "iso2019", "aletta2024"] | _Unset = _UNSET,
    angles: AngleSetRef | Literal["equal"] | AngleSelection | _Unset = _UNSET,
    on_uncovered: Literal["error", "exclude", "equal"] | _Unset = _UNSET,
    on_low_confidence: Literal["error", "exclude", "equal"] | _Unset = _UNSET,
    angle_sets: Sequence[AngleSet] | _Unset = _UNSET,
    carry: Sequence[str] | _Unset = _UNSET,
    settings: ProjectionSettings | Mapping[str, object] | _Unset = _UNSET,
    profile: str | _Unset = _UNSET,
) -> Coordinates: ...
```

`angles` accepts `"equal"`, an `AngleSetRef`, a string `"name@version"`, or a full `AngleSelection`.
It resolves settings (§6.3.4), resolves angles per response (§6.8.3), builds the Coordinates'
status from the survey's status plus projection reasons and flags (the Survey is unchanged),
computes xy for included responses only, and returns `Coordinates` whose rows align with the
survey's rows. `carry` adds survey columns to the labels on top of the mapping's `carry`; it
changes no number, so it is not a settings field.

`angle_sets=` stays for library use without a project. A set passed this way is not looked up by
path or name later: provenance records its full angle table (every language, tier and angle) and
its content hash in `data_refs`, so the result never depends on a file on one machine. This is how
this spec reads owner decision #40 (sets chosen from registered sets by name and version) for use
without a project; it is recorded in §11.2 for the owner to check.

### 6.10 `Coordinates` (commitment 3)

There is no `ProjectionResult`: `Coordinates` carries the producing operation's settings and
provenance itself (plan Appendix A is updated, §11.1).

| Field | Type | Contents |
|---|---|---|
| `xy` | read-only (n, 2) float64 | NaN rows where not included |
| `circumplex` | `Circumplex` | the instrument |
| `labels` | `dict[str, ndarray]` of read-only length-n arrays (`bool`, `float64`, or `str`/`object` for text), in column order | required for kind `survey`: `ssp_response_id`, `ssp_included`, `ssp_reasons`, `ssp_flags`, `ssp_language`, `ssp_angle_set`, each mapped id role under the user's column name, carried columns. No frame is held; `to_frame` builds one with `_frames.new_frame` |
| `origin` | `tuple[OriginSegment, ...]` | one segment at M1 |
| `settings` | `FrozenModel` (a `ProjectionSettings` at M1) | typed as the base class, so `distribution` does not import `projection`, which sits above it (§5.2.1). `workspace.read_coordinates` validates the sidecar's settings into `ProjectionSettings` |
| `provenance` | dict | §6.3.6 |

`OriginSegment` (FrozenModel): `kind: Literal["survey"]` (M2 and M4 widen it), `dataset: str`
(project dataset name, file name or `"frame"`), `n: int`, `input_hash`, `mapping_hash`,
`effective_scale: Scale`, `angle_sets: tuple[str, ...]` (labels used), `settings_hash`. Each segment
keeps its own scale, angle sets and settings hash, so pooling at M2 changes no stored shape.

Methods: `n`, `n_valid` (module functions with the same names exist and are the one definition),
`to_frame(like=None)` (labels plus `ssp_<axis id>` columns, in the source's library),
`select(mask)` (a new `Coordinates`, origin `n` updated), `split(by) -> dict[str, Coordinates]`
where `by` is an id role (`"location"`, `"session"`, `"group"`), `"language"`, or a carried column.
Groups are in sorted key order.

### 6.11 Regions and `describe` (FR-2.5, commitment 9)

`Region` (FrozenModel): `name: StrictStr`, `x_sign` and `y_sign` in {-1, 0, 1} (0 meaning
unconstrained), `box: Literal["none"] | tuple[tuple[StrictFloat, StrictFloat], tuple[StrictFloat,
StrictFloat]] = "none"`. A box bounds a user-defined rectangle with finite edges, x interval then
y interval. Default regions come from the circumplex: four quadrants (signs from the quadrant
keys, named by the reference label of the quadrant's position) and four half-planes (named by the
axis end labels).

**Boundary rule** (normative; a call made here, §11.2): with `atol = 1e-9`, a valid point is
inside a region when `sign · c > atol` on each constrained axis and it lies more than `atol`
inside each box edge. Membership is strict, so no point is counted fractionally. Per group, once:
`n_on_boundary` is the number of valid points within `atol` of either quadrant-defining axis
(`|x| ≤ atol` or `|y| ≤ atol`), and `boundary_share = n_on_boundary / n_valid`. Every valid point
not on the boundary lies in exactly one quadrant, so **Σ quadrant `proportion` + `boundary_share`
= 1** for every group (a property test, §7.1). The rule and `atol` are fields of
`DescribeSettings`, recorded with the result. Why 1e-9: it absorbs float rounding (about 1e-15)
yet sits far below the smallest non-zero distance from an axis that integer Likert data can
produce (about 0.03). At 45° on the 1..5 grid, 2.72% of possible responses lie exactly on the x
axis in exact arithmetic, so the rule moves every quadrant proportion and is fixed here, before
the first baseline (plan §4.9).

```python
def describe(coords, *, by=_UNSET, regions=_UNSET, quantiles=_UNSET, allow_small_n=_UNSET,
             settings=_UNSET, profile=_UNSET) -> Table
```

`DescribeSettings`: `by: str = ""` (no grouping), `regions: Literal["default"] | tuple[Region,
...] = "default"`, `quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)`, `atol = 1e-9`,
`boundary_rule = "strict-with-boundary-share"`, `min_n = 30`, `allow_small_n: StrictBool = False`.
`by` and `regions` are settings fields, so they are recorded and a saved settings file reproduces
the table. The frame is tidy, one row per group × axis: `group`, `axis`, `n`, `n_valid`, `mean`,
`sd` (ddof 1), `q05`, `q25`, `q50`, `q75`, `q95` (numpy `linear`). `extra["regions"]`: `group`,
`region`, `n_valid`, `n_inside`, `proportion`. `extra["boundary"]`: `group`, `n_valid`,
`n_on_boundary`, `boundary_share`. `extra["angle_sets"]`: `group`, `angle_set`, `count` (the mix
report, FR-2.4). One `SmallSampleWarning` names every group with `n_valid < min_n`, unless
`allow_small_n`.

### 6.12 KDE and contours (FR-3.1, FR-3.2, commitment 9, plan §8 G4-07)

One estimator, `scipy.stats.gaussian_kde`, with seaborn's semantics (owner row #10).

**What "the 50% contour" means** (owner decision, §4.3). FR-3.1's "contour enclosing a stated
fraction of responses" is the 50th-percentile (median) contour of Mitchell, Aletta and Kang 2022
[@mitchell2022], the simplified view whose shaded area contains 50% of the responses. 50% is the
default and any other fraction can be chosen. It is computed the way the 2022 figures were made:
seaborn's KDE levels, that is the fraction of the density mass on the grid (the 2022 code drew it
with seaborn `kdeplot(thresh=0.5, levels=2)`). Every `ContourSet` also reports `enclosed_share`,
the observed share of the group's valid points inside the contour, beside it. The docstrings of
`density` and `contours` cite the paper in their References sections.

`DensitySettings` (normative defaults):

| Field | Default | Meaning |
|---|---|---|
| `by` | `""` | grouping: an id role, `"language"` or a carried column, as in `Coordinates.split`; `""` means one group named `"all"` |
| `bw_method` | `"scott"` | scipy `bw_method`: `"scott"`, `"silverman"` or a StrictFloat factor |
| `bw_adjust` | `1.2` | multiplies the factor, as seaborn does; 1.2 is the value of the 2022 figures |
| `gridsize` | `200` | points per axis |
| `extent` | `"data"` | `"data"`: per axis `linspace(max(min − cut·bw, clip_lo), min(max + cut·bw, clip_hi), gridsize)`, bw = sqrt of the diagonal of `kde.covariance` after `bw_adjust`; `"square"`: `linspace(clip_lo, clip_hi, gridsize)` |
| `cut` | `3.0` | bandwidths beyond the data |
| `clip` | `((-1.0, 1.0), (-1.0, 1.0))` | `Literal["none"] \| tuple[interval, interval]`; `"none"` means no clip (the 2022 reproduction's settings), and `extent="square"` then raises `SettingsError` |
| `min_n` | `30` | FR-3.2 |
| `allow_small_n` | `False` | the recorded override |

These are seaborn's `kdeplot` defaults (Scott's rule, `gridsize=200`, `cut=3`) with
`bw_adjust=1.2`, the 2022 paper's method, so new work and the paper agree; every profile, the
chapter's `sal-2026` included, uses them. New work clips to the [−1, 1] square, which the
projection never leaves; the 2022 reproduction does not clip (§6.23). Both are calls made here
(§11.2).

**Normalisation** (normative): Z is evaluated on the grid (`meshgrid` "xy", Z[row = y, col = x]).
The level holding fraction f of the grid mass is

```
z_sorted = sort(Z.ravel())[::-1];  cum = cumsum(z_sorted) / Z.sum()
level(f) = z_sorted[min(searchsorted(cum, f), len − 1)]
```

which equals seaborn's `_quantile_to_level(Z, 1 − f)`. Each group is normalised on its own grid
(seaborn `common_norm=False`).

```python
def density(coords, *, by=_UNSET, bw_method=_UNSET, bw_adjust=_UNSET, gridsize=_UNSET,
            extent=_UNSET, cut=_UNSET, clip=_UNSET, allow_small_n=_UNSET, settings=_UNSET,
            profile=_UNSET) -> Densities
def contours(density: Densities | Density, *, fractions=_UNSET, settings=_UNSET,
             profile=_UNSET) -> ContourSet
def marginals(coords, *, by=_UNSET, settings=_UNSET, profile=_UNSET, **density_kwargs) -> Table
```

`ContourSettings` (FrozenModel, profile section `[contours]`): `fractions: tuple[StrictFloat, ...]
= (0.5,)`, each in (0, 1), strictly increasing. Decile contours are fractions 0.1 to 0.9.

**One return type.** `density` always returns `Densities`: a frozen container of named `Density`
objects, one per group in group order (sorted keys, as `Coordinates.split`), with a single entry
`"all"` when `by` is `""`. It holds `settings` (`DensitySettings`) and `provenance` for the set,
and offers `names`, `items()`, `len()` and lookup by name. A `Density` (frozen dataclass) holds
`group`, the read-only arrays `x`, `y`, `z` (the grid) and `z_points` (the density at each of the
group's valid points, from the same `gaussian_kde`, in row order), `factor`, `covariance` and
`n_valid`. `z_points` is part of the array hash (§6.3.5) and of the saved form.

`density` fits each group's valid points only. `n_valid < 3` raises `SmallSampleError`; groups with
`n_valid < min_n` raise one `SmallSampleWarning` naming them all, unless `allow_small_n`;
collinear or identical points (`LinAlgError`) raise `DensityEstimationError` naming the group.

`contours` takes a `Densities`, or a single `Density`, which it treats as a set of one. It works per
group and returns a `ContourSet` keyed by group: per group and fraction the `level`, the polygons
(a tuple of read-only (k, 2) arrays) and `enclosed_share`, the share of the group's valid points
with `z_points >= level`. It uses contourpy
`contour_generator(x, y, z, fill_type="OuterCode").filled(level, inf)` so a region that meets the
grid edge is still a closed polygon. `ContourSet` carries `settings` (`ContourSettings`) and
`provenance`, whose `input_hash` is the `Densities` array hash (§6.3.5) and whose `upstream` key
(§6.3.6) copies the density's `settings_hash` and `input_hash`. contourpy imports no matplotlib (verified), so
this stays in the core.

`marginals` fits a 1-D `gaussian_kde` per axis and group to the same valid points, with the same
`by`, `bw_method`, `bw_adjust`, `gridsize`, `cut`, `clip`, `min_n` and `allow_small_n` rules as
`density` (taken from `DensitySettings`, recorded with the result). It returns a tidy `Table`:
`group`, `axis`, `value`, `density`, with the fitted `factor` per group and axis in
`extra["bandwidth"]`. `plot.joint` draws its marginal curves from it.

**Serialisation** (FR-14.2). `Densities` and `ContourSet` save through `export` and `save_run`. A
`Densities` run stores one parquet per group, `<name>-<k>.parquet` with `k` the zero-padded group
index (group keys need not be safe file names), plus one JSON sidecar `<name>.json` for the set
(§6.18). Each group's parquet holds the grid in long form (`kind = "grid"`: `x`, `y`, `z`) followed
by its points (`kind = "point"`: `response_id`, `x`, `y`, `z` from `z_points`). The sidecar holds
the settings and provenance, the group keys in order with their file names, and per group
`factor`, `covariance` and `n_valid`. `ContourSet` writes one parquet with one row per vertex
(`group`, `fraction`, `level`, `polygon`, `vertex`, `x`, `y`) and `enclosed_share` per group and
fraction in its sidecar. `read_run` returns a parquet as a `Table`; `read_density(project, name)`
returns an equal `Densities`, and `read_contours(project, name)` an equal `ContourSet`.

### 6.13 Part 3 statistics (FR-2.2, FR-2.6, FR-2.6a)

All statistics are computed by scipy; the package adds only documented formulas over its outputs.
Every result is a tidy `Table` whose notes name the test and the effect-size formula with citations.

**One inclusion rule** for every analysis of a `Survey` (`item_statistics`, `correlate`,
`compare_groups`, `source_profile`, the `who5` summaries and `response_shares`). Each settings
model has `include: Literal["quality_passed", "all"] = "quality_passed"`, recorded with the result.
`quality_passed` keeps a response whose status holds no reason, or only `incomplete_attributes`
(FR-2.5a: a response missing an attribute item still counts for the items it has). `all` keeps
every row. A `Survey`'s status holds only import and quality reasons; language reasons are added
by `project` to the `Coordinates`, so they never remove a response from an item analysis. For a
`Coordinates` input (`compare_groups`, the `coords` of `correlate`) the rows used are the
`n_valid` rows. Missing values are then dropped per item or per pair, and `n` counts what was
used.

#### 6.13.1 `item_statistics`

```python
def item_statistics(survey, *, by=_UNSET, groups=_UNSET, dispersion=_UNSET,
                    quantile_method=_UNSET, include=_UNSET, settings=_UNSET,
                    profile=_UNSET) -> Table
```

`ItemStatisticsSettings`: `by: str = ""`, `groups: tuple[str, ...] = ("attributes",)` (item
groups), `dispersion: Literal["iqr", "range"] = "iqr"`, `quantile_method = "linear"` (numpy method
name, type 7; `linear` equals Excel `QUARTILE.INC`; a call made here, §11.2, recorded with every
result), `include` (the shared rule above). Columns: `group`, `item_group`,
`item`, `label`, `n`, `n_missing`, `median` (never rounded; half-integers occur), `q25`, `q75`,
`iqr`, or `minimum`, `maximum`, `range`. Responses count wherever the item is present, including
responses excluded only for `incomplete_attributes` (FR-2.5a). The notes say which rule applied.

`response_shares(survey, *, item_group=_UNSET, by=_UNSET, include=_UNSET, settings=_UNSET,
profile=_UNSET) -> Table` records its choices in `ItemStatisticsSettings` (`groups` holds the one
item group; `item_group` left unset means `"attributes"`, the field default) and is the geometry of the Likert plot (FR-8.4): tidy per group × item × scale step, `group`, `item`,
`label`, `step`, `step_label`, `count`, `share` (shares over the item's present responses sum to 1
per group and item). `plot.likert` draws from it and computes nothing.

#### 6.13.2 `correlate`

```python
def correlate(survey, *, features=_UNSET, on=_UNSET, key=_UNSET, items=_UNSET, coords=_UNSET,
              method=_UNSET, include=_UNSET, settings=_UNSET, profile=_UNSET) -> Table
```

`features` is a list of survey column names (carried) or a frame (or `Table`) keyed by `key` and
joined on the id role `on` (`"response"`, `"group"`, `"session"`, `"location"`); per-recording ISD
features join on `group`. `coords` adds the axes as variables. `CorrelateSettings` records every
choice: `features: tuple[str, ...] = ()` (the feature column names used, whether carried or from a
frame), `on: str = "response"`, `key: str = ""` (empty means the mapped column of role `on`),
`items: tuple[str, ...] = ("attributes",)`, `use_coords: StrictBool = False` (true when `coords`
was given), `method: Literal["spearman", "pearson"] = "spearman"` and `include`. A features frame
and the `coords` are inputs: provenance records their hashes (`input_hashes`, a dict by
argument name). Spearman is `scipy.stats.spearmanr` (average ranks, equal to the
tie-corrected formula; the untied formula can differ substantially and is never used); Pearson is
`pearsonr`. Pairs with a missing value are dropped per pair. Columns: `x`, `y`, `method`, `r`, `n`,
`p`, `p_method` (`"t approximation, n-2 df"` for both).

#### 6.13.3 `compare_groups`

```python
def compare_groups(data: Survey | Coordinates, *, by=_UNSET, variables=_UNSET, test=_UNSET,
                   include=_UNSET, settings=_UNSET, profile=_UNSET) -> Table
```

`CompareSettings`: `by: str = ""` (required in effect: `""` raises `SettingsError`, because one
group has nothing to compare), `variables: tuple[str, ...] = ()` (empty means every attribute item
for a `Survey`, both axes for `Coordinates`), `test: Literal["auto", "mannwhitney", "kruskal",
"welch", "anova"] = "auto"`, `include`.
`auto` picks by measurement level and group count k, and records the choice per row:

| Data | k = 2 | k > 2 |
|---|---|---|
| ordinal items (Survey) | Mann-Whitney U, two-sided | Kruskal-Wallis H |
| coordinates (interval) | Welch t-test | one-way ANOVA |

Columns: `variable`, `test`, `statistic`, `df1`, `df2` (Welch: its df and NaN; Kruskal: `k − 1`
and NaN; ANOVA: `k − 1` and `N − k`; Mann-Whitney: NaN and NaN), `n_per_group` (text such as
`"RegentsParkFields=118;RegentsParkJapan=93"`), `n`, `p`, `effect_size`, `effect_size_name`. Effect
sizes:

| Test | Effect size | Formula | Source to cite |
|---|---|---|---|
| Mann-Whitney | rank-biserial r | `2·U1/(n1·n2) − 1`, positive when group 1 ranks higher | Kerby 2014 |
| Kruskal-Wallis | ε² | `H / (N − 1)`, i.e. `H / ((N² − 1)/(N + 1))` | Tomczak & Tomczak 2014 |
| Welch t | Cohen's d | `(m1 − m2) / s_pooled`, pooled with `n − 1` weights | Cohen 1988 |
| ANOVA | η² | `SS_between / SS_total` | Cohen 1988 |

Regression, mixed models, ANOVA tables and Holm correction (FR-2.6) are a how-to page using
statsmodels from the docs group, importing it inside the page. statsmodels is not a runtime
dependency: importing it calls `warnings.simplefilter("always", ...)` on several classes, which
changes global state (plan §8 G8-05).

### 6.14 WHO-5 and source-dominance profiles (FR-2.6c)

`who5(survey, *, by=_UNSET, rule=_UNSET, include=_UNSET, settings=_UNSET, profile=_UNSET) -> Table`,
one row per response, every row of the survey: `response_id`, `included`, `reasons`, `flags` (the
status, plain names as in §6.3.3), `who5_items_present`, `who5_raw` (0..25), `who5_percent` (`raw ×
4`), `who5_poor_wellbeing` (`raw ≤ 12`), `who5_depression_screen` (`raw ≤ 7`). A score exists only
when all five items are present (`Who5Settings.rule = "complete_items"`); otherwise the scores are
NaN and the booleans null. The WHO-5 guidance gives no prorating rule, so none is applied (a call
made here, §11.2). `extra["summary"]` gives per group `n`, `n_scored`, mean raw score and the two
cut-off shares, over the rows that pass the shared inclusion rule only. It never reads a stored sum
such as the ISD's `WHO_Sum`, which treats missing items as zero and scores nine non-responders as 0.
The cut-offs are named module constants citing Topp et al. 2015 (Psychother Psychosom 84:167-176);
the citation is checked against the primary source before release (a task in S11, not an owner
question).

`source_profile(survey, *, by=_UNSET, threshold=_UNSET, include=_UNSET, settings=_UNSET,
profile=_UNSET) -> Table`, tidy per group × source: `n`, `mean`, `median`, `q25`, `q75`, `iqr`,
`share_at_least` (the share of present responses at or above `threshold`). No dominant-source
label at M1 (a call made here, §11.2).

| Model | Fields |
|---|---|
| `Who5Settings` | `include` (the shared rule), `rule: Literal["complete_items"] = "complete_items"`, `by: str = ""` |
| `SourceProfileSettings` | `include` (the shared rule), `by: str = "location"`, `threshold: StrictInt = 4` (on the 1..5 dominance scale; must lie on the group's scale) |

### 6.15 Plotting (FR-8.1 to FR-8.4, commitment 9)

`soundscapy.plot` functions, all keyword-only after the data argument. Arguments that reach a
geometry function (`by`, `settings`, `allow_small_n`, `fractions`, `fraction`, `item_group`)
default to `_UNSET` and are passed to the settings fields, so the operation records them; drawing
arguments (`ax`, `fill`, `scatter`, `label_language`, and `radar`'s `statistic`, which picks a
column of `item_statistics` and changes no number) default to `None` or a value:

| Function | Draws |
|---|---|
| `scatter(coords, *, by=_UNSET, ax=None, label_language=None, **kwargs)` | responses on the circumplex; one series per group |
| `density(coords, *, by=_UNSET, fractions=_UNSET, fill=True, settings=_UNSET, allow_small_n=_UNSET, ax=None, **kwargs)` | KDE with decile contours, filled with `cividis`; one `Densities` from `distribution.density`, one layer per group. `fractions` left unset means the deciles 0.1 to 0.9, passed to `ContourSettings.fractions` and recorded |
| `fraction_contour(coords, *, by=_UNSET, fraction=_UNSET, scatter=True, ...)`; `fraction` left unset means the `ContourSettings` default, 0.5 | the simplified 50th-percentile contour (Part 3's third view) |
| `joint(coords, *, by=_UNSET, ax=None, ...)` | density or scatter plus marginal KDEs (from `distribution.marginals`) in `ax.inset_axes`, so it returns one Axes |
| `likert(survey, *, by=_UNSET, item_group=_UNSET, ax=None, ...)` | stacked bars of response shares per item (from `survey.response_shares`) |
| `radar(survey, *, by=_UNSET, statistic="median", ax=None, ...)` | per-attribute medians. With `ax=None` it creates `fig.add_subplot(projection="polar")`; a non-polar `ax` raises `AxesError` (§6.3.10) |

The `ax` contract: `ax=None` creates a new pyplot figure with `plt.subplots()` (polar for `radar`);
the function returns the Axes and never calls `show()`; `**kwargs` go to the main artist.
(Amended 2026-11-01, owner decision on review F5, `design/reviews/2026-10-31-s1-tool-idiom-review.md`:
the earlier pyplot-free `Figure()` departed from matplotlib's documented pattern for libraries and
made every notebook cell end in `ax.figure`; a GUI or web service passes its own `ax` and never takes
the `ax=None` path.) Shared helpers in `plot/_common.py`: `circumplex_axes(ax, circumplex,
label_language)` (the ±1 square, axis lines, axis and quadrant labels in the chosen language,
reference language by default); `check_n(groups, min_n, allow_small_n)`. No origin-label hook exists
at M1, because no M1 function calls it; M2 adds it with its first caller. Colour: groups take the colours of the active `axes.prop_cycle`, so a style sheet,
`rc_context` or `style.context` the user applies sets them; while that cycle is matplotlib's
default, the `petroff10` style is used, and `ax=None` creates its Axes under it; `cividis` for
fills (FR-8.3, NFR-10). (Amended 2026-11-01, review F6: hard-coded `petroff10` colours ignored
matplotlib's own styling routes, which FR-8.3 names as the way to customise.) A single-group density with
`n_valid < 30` raises `SmallSampleError` unless `allow_small_n=True`; a grouped density draws small
groups as scatter with "N = 12 (< 30), scatter shown" in the legend and one `SmallSampleWarning`
(FR-3.2, owner row
#19). Geometry comes from `distribution` and `survey` (`density`, `contours`, `marginals`,
`response_shares`, `item_statistics`); plot code computes nothing itself (FR-8.4). Figures save to
PNG, SVG and PDF with `ax.figure.savefig` (FR-9.1).

**Lazy matplotlib.** `soundscapy.plot` imports matplotlib inside each function, never at module
top; type hints use `TYPE_CHECKING` imports. `import soundscapy` therefore never imports
matplotlib, so matplotlib's first-run font-cache message cannot appear on import. The silent-import
test stays, and a second test asserts `"matplotlib" not in sys.modules` after `import soundscapy`
in a fresh interpreter.

**Notebook display.** With `ax=None` the figure is a pyplot figure, so a notebook shows it
when the cell ends and a script saves it with `plt.savefig()`; no cell ends in `ax.figure`, which
would now show the plot twice. (Amended 2026-11-01, review F5; this replaces the `ax.figure`
display convention.) A docs test reads every executed `{python}` cell of `docs/tutorials/*.qmd` that
calls `sspy.plot`, and asserts that its output in `docs/_freeze/` holds an image.

### 6.16 Tables, export and attach (FR-9.1, FR-14.2)

`Table` (dataclass): `frame` (the input's library), `name`, `settings`, `provenance`,
`notes: tuple[str, ...]`, `extra: dict[str, Table]`. `export(result, path)` chooses by suffix:

| Suffix | Writes |
|---|---|
| `.csv` | UTF-8 with BOM (Excel opens it without mojibake) |
| `.xlsx` | openpyxl; sheets `data`, each `extra` table, `settings` (dotted key and value rows), `provenance` |
| `.parquet` | pyarrow, plus `<stem>.json` with settings, provenance and notes. A `Densities` writes one parquet per group as in §6.12 |

Both live in `workspace._io` and are re-exported at the top level; the core never calls them.
`export` accepts a `Table`, `Coordinates` (via `to_frame`), `Survey` (its status frame), `Densities`
or `ContourSet` (§6.12). A target held open elsewhere raises `OutputFileLockedError`.

`attach(frame, result)` appends `ssp_` columns for `Coordinates` (`ssp_<axis>`, status, language,
angle set) or a per-response `Table` (each column except `response_id` and the status columns,
prefixed `ssp_`, so `who5` adds `ssp_who5_*`) to the user's frame, in the user's library, returning
a new frame. It aligns on the response id, by the rule the result's
`provenance["response_id_source"]` records (§6.3.2):

| Id source | Alignment | Requirement, else `AlignmentError` |
|---|---|---|
| `column` or `key` | join by id, keeping the frame's row order | the frame holds the mapped id column or every key column, and the ids built from them as in §6.3.2 are unique. Frame rows whose id is not in the result get nulls |
| `row` (`row-<n>`) | by position | the frame has exactly the survey's row count |

`attach` records the alignment it used (`"id"` or `"position"`) and the number of matched rows in
one INFO log line, since a plain frame cannot carry it. There is no hash fallback. The docs
recommend attaching to `survey.data` (returned in the user's library), which always aligns.
`attach` never overwrites (`ColumnCollisionError`).

### 6.17 CLI (FR-14.1, FR-14.5)

`soundscapy` (click; `[project.scripts] soundscapy = "soundscapy.cli:main"`; `python -m
soundscapy`). Global options: `--dir/-C PATH` (project directory, default `.`), `--profile
ID[@VERSION]`, `--settings FILE`, `-q/--quiet`, `-v/--verbose`, `--version` (from
`importlib.metadata`, never `click.__version__`). Each command makes one library call. Settings
follow the one precedence rule of §6.3.4: command options rank with keyword arguments, `--settings`
with `settings=`, and `--profile` replaces the base profile for that operation and is recorded as
the profile, not as an override.

| Command | Library call | Writes |
|---|---|---|
| `init [DIR]` | `workspace.init_project` | the project skeleton |
| `template OUT [--format csv\|xlsx]` | `write_template` | the template |
| `suggest-mapping EXPORT [--schema S] [-o FILE]` | `suggest_mapping` | a draft TOML |
| `import EXPORT --name NAME [--mapping FILE] [--accept-suggestion] [--interactive] [--sheet S] [--encoding E] [--delimiter D]` | `workspace.add_dataset` | dataset files and the `<name>-status` run. With no mapping and no `--accept-suggestion`: writes `mapping.draft.toml`, prints its path, exits 1. `--interactive` needs a TTY, else exit 2 |
| `coordinates DATASET [--run NAME] [--formulation F] [--angles A] [--on-uncovered X] [--on-low-confidence X]` | `project` via `workspace.run` | the coordinates run |
| `describe RUN [--by ROLE]` | `describe` | a table run |
| `items DATASET [--by ROLE] [--dispersion D]` | `item_statistics` | a table run |
| `compare SOURCE --by ROLE [--variables V...]` | `compare_groups` | a table run |
| `correlate DATASET --features FILE --on ROLE [--key COL]` | `correlate` | a table run |
| `who5 DATASET [--by ROLE]` | `who5` | a table run |
| `sources DATASET [--by ROLE]` | `source_profile` | a table run |
| `density RUN [--by ROLE] [--bw-adjust F] [--no-clip] [--allow-small-n]` | `density` via `workspace.run` | a density run: one parquet per group and one sidecar (§6.12) |
| `contours RUN [--fractions F...]` | `contours` via `workspace.run` | a contour run |
| `angle-set add FILE` | `workspace.register_angle_set` | `angle_sets/<name>/<version>.toml` |
| `context add DATASET --level location\|session FILE` | `workspace.add_table` | `datasets/<name>/context/<level>.<suffix>` |
| `plot RUN --kind K [--by ROLE] --out FILE` | `workspace.save_figure` (one `plot` function, then `savefig` and the sidecar) | the figure and `<figure>.json` |
| `versions` | `show_versions` | stdout |

Every writing command takes `--xlsx` (also write an Excel copy) and `--overwrite`. Written file
paths go to stdout, one per line; logs and errors go to stderr. Exit codes: 0 success, 1 a
`SoundscapyError` (message and hint on stderr), 2 a usage error. There is no `validate` (import
already validates), no `demo` (several calls) and no `export` (CSV copies are written by default
and `--xlsx` gives the Excel path; owner row #28). CLI help text is the CLI reference
(`cli: {enabled: true, module: soundscapy.cli, name: main}` in great-docs at slice S8, naming the
click group that `[project.scripts]` points at; great-docs' auto-discovery is the fallback if the
explicit name fails). Tests use
`CliRunner` with `tmp_path`, never `isolated_filesystem` (deprecated in click 8.5).

### 6.18 Project directory and workspace (commitment 11, FR-14.4)

**Normative** layout; `workspace/_layout.py` alone knows these names.

```
<project>/
  soundscapy-project.toml        # format = 1, created_utc, soundscapy_version
  settings.toml                  # extends = "sal-2026", plus overrides
  manifest.json                  # datasets, runs, angle sets: names, hashes, versions
  angle_sets/<name>/<version>.toml
  datasets/<name>/
    raw/<original file name>     # byte-for-byte copy
    context/<level>.<suffix>     # optional location or session table, byte-for-byte (§6.5.1)
    dataset.toml                 # resolved read options, raw and context hashes, n_rows, source name
    mapping.toml
    mapping.draft.toml           # only while a draft is pending
  runs/<run>/
    <table>.parquet              # also ContourSet vertices (§6.12)
    <table>-<k>.parquet          # a Densities run: one per group, k the zero-padded group index
    <table>.json                 # authoritative sidecar: kind, settings, provenance, notes
    <table>.csv                  # default copy (UTF-8 with BOM); <table>.xlsx with --xlsx
  figures/
    <figure>.<png|svg|pdf>
    <figure>.json                # the data drawn: run name, kind, settings, provenance; matplotlib version
```

Functions: `init_project(path)`; `add_dataset(project, name, source, mapping=_UNSET, *,
accept_suggestion=False, read_options=_UNSET)` (`mapping` unset means the suggestion
with `accept_suggestion`, else a `mapping.draft.toml` and a `MappingError`, as the CLI's `import`
does; `read_options` reach every parsed number, so they follow the `_UNSET` rule) (copies raw, writes `dataset.toml` and `mapping.toml`,
runs `read_survey`, saves the status run); `open_dataset(project, name) -> Survey` (re-reads raw,
checks the hash, raising `DataIntegrityError` on mismatch, re-applies the mapping);
`register_angle_set(project, path)` (validates and copies); `run(project, operation, ...)` (resolves
project settings, calls the operation, saves); `save_run(project, name, result, *,
overwrite=False)`; `read_run(project, name) -> Table`; `read_coordinates(project, name) ->
Coordinates` (validates the sidecar's settings into `ProjectionSettings`, §6.10); `read_density ->
Densities`, `read_contours` (§6.12); `add_table(project, dataset, level, path)` (§6.5.1);
`save_figure(project, name, ax, *, source_run)` (saves the figure and writes `<figure>.json` with
the source run's name, kind, settings and provenance, so a figure is traceable like a table);
`export_run(project, run, path)` (copies a saved run to a CSV, xlsx or parquet path, by suffix). The
workspace function is named `export_run` so it cannot be confused with `sspy.export(result, path)`,
which writes an in-memory result. Run names default to `<dataset>-<operation>`. Settings follow the
precedence rule of §6.3.4, with the project's `settings.toml` at rank 3. Re-running an operation
uses the current inputs and settings: an existing run's sidecar is a record, never an input
(§6.3.4). A format other than 1 raises `ProjectFormatError`; alphas promise no compatibility.
Coordinates sidecars store the circumplex and origin, so `read_coordinates` rebuilds an equal
object.

### 6.19 Runtime dependencies at M1

Each is added with `uv add` and its reason in the commit message; the `lowest` CI job proves the
floors.

| Package | Floor | Licence | Why |
|---|---|---|---|
| numpy | 1.26 | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | arrays inside the core |
| scipy | 1.15 | BSD-3-Clause | `gaussian_kde`, `stats` tests |
| contourpy | 1.2 | BSD-3-Clause | contour polygons without matplotlib (the core imports it, so it is declared) |
| pandas | 2.2.1 | BSD-3-Clause | frames from paths; Anaconda course laptops have 2.2. (Amended at S1 plan review, 2026-10-31: 2.2.0 warns on import when pyarrow is absent, which it is in the `lowest` job until slice S4 adds pyarrow, and the warning is fatal under `filterwarnings = ["error"]`; 2.2.1 does not warn.) |
| pyarrow | 15 | Apache-2.0 | parquet (owner row #6) |
| narwhals | 2.0 | MIT | the boundary (`stable.v2`) |
| pydantic | 2.7 | MIT | settings and value types |
| matplotlib | 3.10.7 | Matplotlib licence (PSF-based) | plotting; `petroff10`. (Amended at S1 plan review, 2026-10-31: 3.10.0 to 3.10.6 emit `PyparsingDeprecationWarning` on import against pyparsing 3.3.) |
| pooch | 1.8 | BSD-3-Clause | ISD download and cache |
| tqdm | 4.66 | MPL-2.0 AND MIT | pooch's progress bar needs it, and owner decision #4 makes `tqdm.auto` the default reporter; so core at M1, not M3 (a call made here, §11.2; a plan §5 edit, §11.1) |
| click | 8.2 | BSD-3-Clause | CLI; 8.2 is the first with `CliRunner` stderr always separate |
| tomlkit | 0.13.1 | MIT | writing mapping drafts with TODO comments. (Amended at S1 Task 12, 2026-10-31: 0.13.0 writes different bytes for a mapping with non-default fields, so `write_mapping` does not round-trip byte for byte; 0.13.1 to 0.14.0 match.) |
| openpyxl | 3.1 | MIT | Excel read and write (one library; called directly, so no Pandas4Warning) |

The licence column records each runtime dependency's licence (NFR-7), read from PyPI metadata on
2026-10-31: `License-Expression` where the package sets it, else its `License` field or classifier
(scipy declares only the classifier "BSD License"; its `LICENSE.txt` is BSD-3-Clause). All are
permissive and compatible with BSD-3-Clause; tqdm's MPL-2.0 part is file-level copyleft and is
used unmodified. The table is kept as `tests/licences.toml`, and a test fails if a runtime
dependency in `pyproject.toml` has no entry there or its installed metadata declares no licence.

Dev group adds `pandas-stubs`; test group adds `polars`. seaborn and statsmodels are not runtime
dependencies (seaborn appears only in `scripts/make_seaborn_fixtures.py` via `uv run --with`;
statsmodels only in the docs group).

### 6.20 The ISD loader and bundled data (FR-10.1)

#### 6.20.1 Bundled subset

`_data/datasets/isd/subset-v1.0.csv`: Option C, 1552 rows (eng 1213, spa 126, nld 89, cmn 69, ita
55), from the 11 London locations (CamdenTown, EustonTap, MarchmontGarden, PancrasLock,
RegentsParkFields, RegentsParkJapan, RussellSq, StPaulsCross, StPaulsRow, TateModern, TorringtonSq)
plus CampoPrincipe, SanMarco, Noorderplantsoen and PingshanPark. It is made from ISD 1.0.1-alpha.1,
the default version (§6.20.2). 40 columns, cell text copied verbatim, plain CSV with LF line
endings, about 315 KiB:

| Group | Columns |
|---|---|
| ids and meta (7) | LocationID, SessionID, GroupID, RecordID, start_time, Language, Survey_Version |
| source dominance (4) | ssi01 to ssi04 |
| attributes (8) | pleasant, chaotic, vibrant, uneventful, calm, annoying, eventful, monotonous (CSV order) |
| ISO items (5) | sss01 to sss05 |
| WHO-5 (5) | who01 to who05 |
| age (1) | age00 |
| features (10) | `LAeq_L(A)`, `LAeq_L10(A)`, `LAeq_L90(A)`, `LCeq_L(C)`, `N_N5`, `S_S`, `R_R`, `T_TonalityHMS`, `FS_F`, `I_I` |

1350 subset rows carry feature values. That is 53 more than a subset of 1.0.1-alpha would carry
(1297), all at PingshanPark, because 1.0.1-alpha.1 adds acoustic values for 468 cmn rows
(§6.20.3). The survey columns, and so every pinned count of §6.7, are unchanged.

Left out: gen00 (mixes `1`/`2` codes with `Male`/`Female` in cmn rows; unknown meaning), misc03,
free text (mojibake in misc01) and `WHO_Sum`. `scripts/make_isd_subset.py` reads
`ISD v1.0 Data.csv` of ISD 1.0.1-alpha.1, either downloaded from Zenodo record 10672568 or from a
local path (`--source PATH`), checks its sha256
`27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735`, and writes the CSV and the
sidecar `subset-v1.0.toml` (citation block, source version and record, source and subset hashes,
locations, columns, feature units marked unverified). The subset's own sha256 is pinned in the
`_data` registry once the generator has written it (§11.3).

The licence permits it: every ISD version on Zenodo is CC BY 4.0, open access, so a subset may be
redistributed with attribution. The sidecar's citation block is normative:

```toml
[citation]
citation = "Mitchell, A., Oberman, T., Aletta, F., Erfanian, M., Kachlicka, M., Lionello, M., Fang, X., and Kang, J. (2024). The International Soundscape Database: An integrated multimedia database of urban soundscape surveys -- questionnaires with acoustical and contextual information (version 1.0.1-alpha.1) [Data set]. Zenodo."
doi_or_url = "https://doi.org/10.5281/zenodo.10672568"
licence = "CC-BY-4.0"
date = "2024-02-16"
notes = "Subset of ISD 1.0.1-alpha.1 (Zenodo record 10672568; all versions: concept DOI 10.5281/zenodo.5578572). 15 locations and 40 columns, cell text unchanged. Made by scripts/make_isd_subset.py."
```

The author list matches record 10672568's creators in the Zenodo API (eight, Fang included). The
default run checks the committed subset against its pinned sha256 and the subset counts of §6.7,
offline. The check of each subset row against its source row (through the key) and the full-ISD
counts are `network` tests (nightly): they fetch record 10672568 with its pinned hash.

#### 6.20.2 Loader

`datasets.isd.load(*, subset=False, version=_UNSET, locations=_UNSET, quality=_UNSET,
on_scale_violation=_UNSET, settings=_UNSET, profile=_UNSET) -> Survey` (top-level `load_isd`)
reads with the mapping file and read options its registry entry names, and passes `quality`,
`on_scale_violation`, `settings` and `profile` through to `read_survey` unchanged. `version` left
unset means the registry's `default`. `subset=True` reads the bundled file and accepts only the
version it was made from, `1.0.1-alpha.1`. `fetch(*, version=_UNSET) -> Path` downloads via pooch
into `pooch.os_cache("soundscapy")`, overridable with `SOUNDSCAPY_DATA_DIR`, with
`retry_if_failed=2` and the progress bar on. HTTP errors raise `DownloadError`; a hash mismatch
raises `DataIntegrityError` naming both hashes; an unknown version raises `DataReferenceError`
listing the registered ones. One INFO line per download names URL, size, version, licence and
citation. Each version has its own URL (0.x mapped two versions to one URL).

**The default version is the latest registered version** (owner decision, §4.3), today
1.0.1-alpha.1. "Latest" is resolved at release time, by updating `default` in `registry.toml`,
never at run time: the loader never asks Zenodo which version is newest, so every result records a
pinned record and hash (`data_refs`, §6.3.6). On 2026-10-31 the Zenodo API's "latest version"
link for the concept record resolves to record 10672568, which is 1.0.1-alpha.1. Moving the default
to a newer version is a release change: the subset is regenerated from it and its counts are
re-pinned.

#### 6.20.3 ISD records and registry

Read from the Zenodo API on 2026-10-31. The sha256 values were computed from downloads made the
same day; size and md5 match the API.

| Version | Record (DOI 10.5281/zenodo.<id>) | Published | File, size | md5 | sha256 |
|---|---|---|---|---|---|
| 1.0.1-alpha.1 (default) | 10672568 | 2024-02-16 | `ISD v1.0 Data.csv`, 2,077,871 bytes | `4d3ecb80fbf5e14063ee6f16bba63d41` | `27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735` |
| 1.0.1-alpha | 10639661 | 2024-02-09 | `ISD v1.0 Data.csv`, 1,860,308 bytes | `d47299560679c565bbb7448f7170029c` | `1e54e41c694b7054e137210dc632a7d0c0aa6eca91e4b5e6a59605ea6411e3cf` |
| 0.2.3 (the 2022 paper) | 5914762 | 2021-11-09 | `SSID Lockdown Database VL0.2.2.xlsx`, 594,585 bytes | `5f6178e57e0c4b5a064f292fd78743bc` | `d56890aab4676417d5c1a6e82f48b3b118f4d1c3505da58cca78f0a25e4501a6` |

Facts the registry and its tests depend on:

| Fact | Consequence |
|---|---|
| Concept DOI 10.5281/zenodo.5578572 covers every version; all are CC BY 4.0, open access | Each registry entry cites its own record DOI; the concept DOI goes in the notes |
| 1.0.1-alpha's CSV is byte-identical to the CSV bundled on 0.x `origin/main` | The pinned counts of §6.7 were first computed on it |
| 1.0.1-alpha.1's CSV has the same 3589 rows, in the same order, with the same four-part keys and 142 columns. Its 46 survey and id columns are identical cell for cell except `RecordingLength` (468 rows differ). The 96 acoustic columns drop the `_Max` suffix (`LAeq_L(A)_Max` becomes `LAeq_L(A)`), with identical values wherever 1.0.1-alpha had one. 468 more rows, all cmn, carry acoustic values: 1943 rows against 1475 | Every survey count pinned in this spec holds for both versions. The registry lists each version's feature columns. Both versions stay registered |
| Each v1.0 record also holds `ISD v1.0 Metadata.xlsx`, `Survey_Data.zip`, `Scripts.zip`, and WAV_* and SLM_* audio zips | Redistributable CC BY 4.0 audio exists (PRD §10.3 asks); noted for M3 planning, when the owner confirms it |
| Three Zenodo records are labelled 0.2.3: 5914762, 5914721 and 5914715. All three hold the same `SSID Lockdown Database VL0.2.2.xlsx` (same size and md5) and differ in the accompanying notebook and environment files | The registry uses 5914762, the record the 0.x loader used; its notes name the other two |
| The v0.2.3 workbook has two sheets, `SSID Lockdown Database VL0.2.1` (1909 rows, 78 columns) and `Acoustics Settings`. Its column names differ from v1.0 (for example `Traffic`, `Overall`, `Age`, `LAeq_L(A)(dB(SPL))`), its `Language` holds names (`English`, `Italian`, `Spanish`), and a `Lockdown` column marks the 2020 lockdown surveys (571 rows) | It needs its own mapping file, `datasets/isd_v0.2.3_mapping.toml`, and its registry entry names the sheet, because a multi-sheet workbook without a sheet raises `FileFormatError` (§6.5). 607 rows are recording-only: blank `RecordID`, blank `Language`, no survey items. With `RecordID` allowed blank, the joined four-part key is unique across all 1909 rows (checked) |
| Direct downloads work at `https://zenodo.org/records/<id>/files/<url-encoded name>` (checked for all three files) | The registry URL form |

**Normative** `_data/datasets/isd/registry.toml` keys per version: `record`, `doi`, `file`, `url`,
`size`, `md5`, `sha256`, `licence`, `published`, `citation`, `mapping`, `read_options` (a table of
`ReadOptions` fields, empty for the CSVs), `feature_columns` and `notes`. The three entries:

```toml
schema = "soundscapy.dataset_registry"
schema_version = 1
concept_doi = "10.5281/zenodo.5578572"
default = "1.0.1-alpha.1"

[versions."1.0.1-alpha.1"]
record = "10672568"
doi = "10.5281/zenodo.10672568"
file = "ISD v1.0 Data.csv"
url = "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv"
size = 2077871
md5 = "4d3ecb80fbf5e14063ee6f16bba63d41"
sha256 = "27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735"
licence = "CC-BY-4.0"
published = "2024-02-16"
citation = "Mitchell, A., et al. (2024). The International Soundscape Database (version 1.0.1-alpha.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10672568"
mapping = "isd_mapping.toml"
read_options = {}
feature_columns = ["LAeq_L(A)", "..."]  # the 96 acoustic columns, written by the generator
notes = ""

[versions."1.0.1-alpha"]
record = "10639661"
doi = "10.5281/zenodo.10639661"
file = "ISD v1.0 Data.csv"
url = "https://zenodo.org/records/10639661/files/ISD%20v1.0%20Data.csv"
size = 1860308
md5 = "d47299560679c565bbb7448f7170029c"
sha256 = "1e54e41c694b7054e137210dc632a7d0c0aa6eca91e4b5e6a59605ea6411e3cf"
licence = "CC-BY-4.0"
published = "2024-02-09"
citation = "Mitchell, A., et al. (2024). The International Soundscape Database (version 1.0.1-alpha) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10639661"
mapping = "isd_mapping.toml"
read_options = {}
feature_columns = ["LAeq_L(A)_Max", "..."]
notes = ""

[versions."0.2.3"]
record = "5914762"
doi = "10.5281/zenodo.5914762"
file = "SSID Lockdown Database VL0.2.2.xlsx"
url = "https://zenodo.org/records/5914762/files/SSID%20Lockdown%20Database%20VL0.2.2.xlsx"
size = 594585
md5 = "5f6178e57e0c4b5a064f292fd78743bc"
sha256 = "d56890aab4676417d5c1a6e82f48b3b118f4d1c3505da58cca78f0a25e4501a6"
licence = "CC-BY-4.0"
published = "2021-11-09"
citation = "Mitchell, A., et al. (2021). The International Soundscape Database (version 0.2.3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5914762"
mapping = "isd_v0.2.3_mapping.toml"
read_options = { sheet = "SSID Lockdown Database VL0.2.1" }
feature_columns = ["Loudness_N5(soneGF)", "..."]  # the 15 acoustic columns
notes = "Records 5914721 and 5914715 are also labelled 0.2.3 and hold the same workbook (same md5)."
```

`"..."` stands for the rest of each list, which the builder writes from the file's header. The
`published` dates are the API's `publication_date`. A `network` test downloads each version, checks
size, md5 and sha256, and checks that `feature_columns` equals the header's acoustic columns.

**The v0.2.3 mapping** (`isd_v0.2.3_mapping.toml`, normative content, written in the §6.4.4 form):
`response_key` LocationID, SessionID, GroupID, RecordID, with the matching id roles and
`response_key_blank_ok = ["RecordID"]` (§6.3.2); `language` column `Language` with the coding
`English` to `eng`, `Italian` to `ita`, `Spanish` to `spa`, a blank cell staying unmapped (`""`);
attributes by their English names (the same eight names as v1.0); source dominance `Traffic`,
`Other`, `Human`, `Natural`; `overall_quality` `Overall`; `appropriateness` `Appropriate`; WHO-5
`who01` to `who05`; `age` `Age`; carried `start_time`, `Lockdown`, `sss04`, `sss05`, `loud` and the
acoustic columns. A test reads the workbook's header from a committed one-row fixture of it and
validates the mapping against it offline. The 607 recording-only rows read with their blank
`RecordID` and language, are excluded by `incomplete_attributes` (and `excessive_missing`), and are
reported in the status table, never dropped; their blank language is never consulted, because an
excluded row is not projected (§6.8.3).

#### 6.20.4 Course dataset

The IoA course tutorial data ships as a bundled course dataset (owner decision, §4.3), so journey
K's course material and the tests have a realistic export with renamed columns.

| Item | Value |
|---|---|
| Source | `examples/data/DemoData.csv` on 0.x `origin/main` (`git show origin/main:examples/data/DemoData.csv`), sha256 `d6d4f937b53f02f4bdaa28e8401fe90061d0d9fde9da89c0ec898baaa4fcdd3a`, 93,202 bytes, LF line endings |
| Rows | 456: CamdenTown 105, MarchmontGarden 105, PancrasLock 94, TateModern 152 |
| Relation to the ISD | Every row matches an ISD row on the four-part key, and the 456 are the ISD rows for those four sites minus two (`PancrasLock/PancrasLock2/PL202/410` and `TateModern/TateModern1/TM124/663`). Every compared value equals the ISD's: the eight attributes, the source, ISO and age items, and the eight acoustic columns against 1.0.1-alpha.1's names |
| Columns (37 in the source) | LocationID, SessionID, GroupID, RecordID, start_time, end_time; traffic_noise, other_noise, human_sounds, natural_sounds; PAQ1 to PAQ8 in the 0.x order (pleasant, vibrant, eventful, chaotic, annoying, monotonous, uneventful, calm); overall_sound_environment, appropriate, perceived_loud, visit_often, like_to_visit; age, gender, free_response, tourist_local; LAeq, LA10, LA90, LCeq, N5, sharpness, roughness, tonality; ISOPleasant, ISOEventful |
| Bundled copy | `_data/datasets/course_demo/course_demo.csv`: the source without `free_response`, `ISOPleasant` and `ISOEventful`, other cells unchanged, 34 columns, about 73 KiB. The precomputed coordinates are derived values the package recomputes; the free text is not needed. The sidecar `course_demo.toml` records the drop, the source hash and the copy's hash |
| Licence and citation | CC BY 4.0, citing the ISD (concept DOI 10.5281/zenodo.5578572 and record 10672568) in the sidecar's citation block, with a note that the file is the IoA course's renamed extract |
| Mapping | `_data/datasets/course_demo/mapping.toml`: `response_key` the four ids; attributes `pleasant = "PAQ1"` to `calm = "PAQ8"`; source dominance by the same names; `overall_quality = "overall_sound_environment"`, `appropriateness = "appropriate"`; `age = "age"`; `language.value = "eng"`; carried `start_time`, `end_time`, `perceived_loud`, `visit_often`, `like_to_visit`, `gender`, `tourist_local` and the eight acoustic columns. There is no WHO-5 |
| Loader | `datasets.course_demo.load(*, quality=_UNSET, on_scale_violation=_UNSET, settings=_UNSET, profile=_UNSET) -> Survey`, offline |
| Generator | `scripts/make_course_demo.py` reads the source from `origin/main` or `--source PATH`, checks its sha256, drops the three columns and writes the copy and sidecar |

The file has no language column. The ISD records 454 of its rows as eng and 2 as spa, so
`language.value = "eng"` is right for all but two rows; the sidecar notes say so (a call made here,
§11.2). The mapping doubles as the realistic "renamed columns" fixture of FR-1.3 and FR-1.3a:
`suggest_mapping` on it must propose the attribute positions from `PAQ1` to `PAQ8` and leave nothing
ambiguous.

### 6.21 Top-level names and the quick-start

`import soundscapy as sspy`. **Normative** `__all__` (about 40 names; they settle over the M2
course and freeze at the first non-alpha release):

| Tier | Names |
|---|---|
| In | `read_survey`, `read_mapping`, `write_mapping`, `suggest_mapping`, `write_template`, `load_isd` |
| Types | `Survey`, `SurveyMapping`, `Coding`, `ReadOptions`, `QualitySettings`, `Circumplex`, `Scale`, `AngleSelection`, `AngleSetRef`, `ProjectionSettings`, `Coordinates`, `DensitySettings`, `Density`, `Densities`, `ContourSet`, `Table` |
| Analysis | `circumplex`, `project`, `describe`, `density`, `contours`, `marginals`, `item_statistics`, `response_shares`, `correlate`, `compare_groups`, `who5`, `source_profile`, `join_context` |
| Out and admin | `attach`, `export`, `list_profiles`, `save_settings`, `show_versions`, `set_log_level`, `__version__` |
| Modules | `plot`, `datasets` (with `datasets.isd` and `datasets.course_demo`), `workspace`, `errors` |

Plot functions stay in `plot`, so `sspy.density` (geometry) and `sspy.plot.density` (drawing) do
not collide. The top level imports these modules eagerly. The quick-start first plot (slice S1
replaces `_quickstart.py` with it):

```python
import soundscapy as sspy
coords = sspy.project(sspy.load_isd(subset=True), angles="equal")
sspy.plot.scatter(coords, by="location").figure.savefig("first-plot.png")
```

The install page shows the notebook form of the same cell, which ends with the display convention
of §6.15 instead of saving: `sspy.plot.scatter(coords, by="location").figure`.

### 6.22 Tutorial A (`docs/tutorials/01-survey-to-findings.qmd`)

Runs offline on the subset. Every plot cell ends with `ax.figure` (§6.15). Sections: load the
subset and show the status table; project with the default profile; the language section;
`describe` by location with region proportions and the boundary share; item statistics and the
Likert and radar plots; density by location and the 50% contour, printing `enclosed_share`;
compare and correlate RegentsParkFields with RegentsParkJapan (course case 1), correlating the
features joined on `group`, and a location table joined with `join_context`; WHO-5 and source
profiles; a synthetic messy REDCap export (`;`-separated, cp1252, label-coded) taken through
`suggest_mapping`, the `CodingError` and its fix; workspace save and reopen; the CLI as
non-executed bash; xlsx and svg export. User-guide pages added at M1: mapping how-to (with the
course dataset as its renamed-columns example, §6.20.4), settings and profiles explanation,
projection explanation (formulations, angle sets, tiers), distributions explanation (the 2022
50th-percentile contour, KDE mass fraction and `enclosed_share`, boundary rule), statistics how-to
with statsmodels, the Qualtrics how-to, and the canonical layout reference (FR-1.2, FR-12.3). Each
sets `guide-section` (§5.9).

**Course case 1.** The course brief compares Regent's Park "Broadwalk" with the Japanese Garden.
No ISD site is the Broadwalk, and the owner could not recall which site the brief meant, so the
case uses RegentsParkFields against RegentsParkJapan, from the full ISD in the course and from the
subset here. A note in the cell text says so.

**The language section** shows the real behaviour under the default profile first: the subset
projects with `satp@1.5`, PingshanPark's 69 cmn rows project with Medium-tier angles, carry the flag
`medium_confidence_language`, and one `MediumConfidenceLanguageWarning` names cmn and its count
(the cell shows the warning). It then shows the Low and uncovered paths, which no ISD language
reaches under `satp@1.5`, on the synthetic three-tier test set of slice S3: a small generated
survey whose languages are the ISO 639-3 private-use codes `qaa` (high), `qab` (medium), `qac`
(low) and `qad` (uncovered), with an angle set named `synthetic-tiers@1` whose notes say it is
synthetic. The cell text says so too. It demonstrates the three `on_uncovered` and
`on_low_confidence` paths: `error` in a `#| error: true` cell, then `exclude`, then `equal`.

### 6.23 The 2022 reproduction (`docs/papers/2022-how-to-analyse.qmd`)

Frozen, and re-run nightly (owner decision, §4.3). It reads ISD v0.2.3, the version the paper's
§6 cites, with `load_isd(version="0.2.3")`, which applies the registry's v0.2.3 mapping and sheet
(§6.20.3). **The filter**, stated in the page: the whole page uses the paper's "entire dataset of
1338 responses at all 13 locations" (its §3), the pre-lockdown surveys, `Lockdown == 0`.
`Lockdown == 0` selects exactly 1338 rows of the workbook, at 13 locations (checked on the
downloaded file; the 571 other rows are the 2020 lockdown surveys). This reading of the paper is a
call made here (§11.2). The page computes and prints its counts beside the paper's 1338, rather
than stating them.

**The counts** the page prints, as checked on the downloaded workbook for draft 0.4:

| Count | Rows |
|---|---|
| `Lockdown == 0` | 1338 |
| Excluded by the normal rules | 46: the 36 recording-only rows with no attribute answers (`incomplete_attributes`, and `excessive_missing`, since all 12 items are missing), and 10 more with an attribute item missing (`incomplete_attributes`) |
| Included | 1292 (English 1213, Italian 69, Spanish 10) |
| Panel (d), included rows, `(0, 63]` | 512 |
| Panel (d), included rows, `(63, 150]` | 615 |
| Panel (d), included rows with no `LAeq_L(A)(dB(SPL))` value | 165, reported, in neither bin |

The 36 rows without attribute answers are never dropped: they stay in the status table with their
reasons, and the page reports them.

It uses a settings file committed beside it: profile `iso-12913-3-2019` (45°, the 2019 formula,
the paper's Eqs. 1 and 2; amended at S1 plan review, 2026-10-31), the 2021 quality rules, and density with seaborn's KDE defaults and
`bw_adjust=1.2` (`bw_method="scott"`, `gridsize=200`, `extent="data"`, `cut=3`), `clip="none"` and
per-group normalisation (owner decision, §4.3; the 0.x code at commit ece4e0c, which drew the
figures, sets `default_bw_adjust = 1.2`). Panels, as drawn by the ece4e0c notebook:

| Panel | Content |
|---|---|
| (a) | RussellSq joint plot with marginals (owner decision) |
| (b) | RegentsParkFields 50% contour with scatter |
| (c) | CamdenTown, RussellSq and PancrasLock 50% contours overlaid |
| (d) | the 1338 rows split at LAeq 63 dB over all of them, not per location, column `LAeq_L(A)(dB(SPL))`, as `pd.cut` bins `(0, 63]` and `(63, 150]` |

The 63 dB split over the whole v0.2.3 dataset is confirmed by the owner (§4.3), and is what the
ece4e0c code drew; the paper's text and caption say 65 dB. The page computes the split on the 1338
rows and states the difference from the caption.

A Deviations section lists what differs from the paper: recomputed coordinates, the 2021 quality
rules instead of the 2022 code's (which excluded all-neutral responses and filled missing items
with 3), and the 63 versus 65 dB wording. It prints `enclosed_share` per panel and group. No
numeric baseline is kept (owner row #10). `task docs` uses the freeze; `task docs:papers`
re-executes it (§5.9).

### 6.24 0.x bugs this design avoids

| 0.x behaviour | Avoided by |
|---|---|
| Centring each response at its own (max+min)/2 under adjusted angles (30% of responses moved, all-5 mapped to the origin) | §6.9.1: centre from the declared scale; the constant-shift property test |
| `validate()` dropping rows with index labels used as positions | No row is ever dropped; ids never depend on order |
| All-neutral responses excluded by a sum-versus-3 comparison; missing items filled with 3 | §6.7 rules; FR-2.5a exclusion |
| `WHO_Sum` with missing items as 0 | §6.14 |
| Two ISD versions sharing one URL; a version pointing at another's file | §6.20.2 per-version registry |
| Hand-typed angle tables with two values off | §6.8.2 generated from the machine-readable output |
| Attributes assumed in column order | §6.4.4 name-based mapping |

## 7. Testing

Tests mirror the package under `tests/`, plus `tests/science/`, `tests/regression/` and
`tests/reference/` (house-style). `tests/templates/` is created at M2 with the first FR-13.1
recipe, not at M1. A `frame_lib` fixture parametrises
frame-taking tests over pandas and polars. Every float comparison states its tolerance and why.

### 7.1 Unit and property tests

Property tests use hypothesis. **Named invariants**:

| Invariant | Statement | Tolerance |
|---|---|---|
| square bounds | for any scale, angle set and in-scale scores, `|x| ≤ 1` and `|y| ≤ 1` under canonical | `1e-12` (float rounding of sums of 8 terms) |
| neutral to origin | under canonical and iso2019 only: all items at the midpoint map to (0, 0) for any angle set | `1e-12` |
| constant shift | under canonical and iso2019 only: adding c to every item (staying in scale) moves x by `c·Σcos θ_i / (h·Σ|cos θ_i|)` and y likewise; this is the 0.x bug's test | `1e-12` |
| neutral off the origin (aletta2024) | the opposite property: under aletta2024 a neutral response maps off the origin whenever Σ cos θ_i ≠ 0 (x = μ·Σcos θ_i / λ_P, so x ≠ 0), and likewise for y with Σ sin θ_i | `1e-12` on the value; the generated angle sets keep \|Σcos θ_i\| above 1e-6 so the sign is unambiguous |
| row-order invariance | permuting rows permutes xy and status identically for every id source, and ids identically for `column` and `key` ids (`row-<n>` ids follow physical position, §6.3.2 rule 3; amended at S1 plan review, 2026-10-31); describe output is unchanged | exact for ids, `1e-12` for statistics |
| coding order | value map, offset and reverse applied in the §6.4.2 order; reverse is an involution | exact |
| TOML round-trip | every FrozenModel survives TOML write and read with an equal value and equal hash; the shipped `isd_mapping.toml` round-trips byte for byte (§6.4.4) | exact |
| finite floats | every FrozenModel float field rejects ±inf and NaN; `"none"` is the only no-limit value | exact |
| status invariants | `included == (len(reasons) == 0)` for every row; codes unique and ordered; row count equals input row count | exact |
| region sums | per group, the four quadrant `proportion` values plus `boundary_share` sum to 1 (§6.11); every proportion lies in [0, 1] | `1e-12` |
| hashing | the same content in pandas and polars, and as int or float columns, has the same `frame_hash` after the §6.3.5 normalisation; golden vectors hold | exact |
| profile carry-over | an operation with `profile` unset records its input's profile; an explicit different one is recorded as `profile`, the input's in `profile_inherited`, and `overrides` stays empty (§6.3.4) | exact |
| immutability | arrays on every record are read-only; mutating the user's frame after `read_survey` changes nothing | exact |

Unit tests cover every error and warning path of §6.3.10 (each warning asserted with
`pytest.warns`, and asserted to fire once), every quality rule × action, every angle-tier branch
including tier-only languages, lazy-frame rejection, the silent-import test, the multi-sheet
refusal, delimiter sniffing for `,`, `;`, tab and `|`, cp1252 decoding failure and success,
`join_context` on a `Survey` and on `Coordinates` (and its `TypeError` on a plain frame), the
Qualtrics label-row detection, the settings precedence of §6.3.4 rank by rank, and the licence
test of §6.19. Every operation's settings model is checked to hold each argument of §6.13 and §6.14
that changes a number, and an operation run twice with the recorded settings gives equal results.

**API documentation test** (FR-12.4). A test walks `soundscapy.__all__` and the public names of
each public module (`plot`, `datasets`, `workspace`, `errors`, and each module's `__all__`), and
asserts that every public function or class has a numpy docstring, and that every one implementing
a procedure (a name listed in `tests/procedures.txt`, kept with the code; for example `project`,
`describe`, `density`, `correlate`, `who5`) has a `References` section with at least one
`[@key]` found in `docs/references.bib`.

### 7.2 Science tests (`tests/science/`, marker `science`, default run)

| Test | Source | Tolerance |
|---|---|---|
| 2019 hand values: (p, v, e, ch, a, m, u, ca) = (5,3,3,3,1,3,3,3) gives x = √2 − 1 = 0.41421356, y = 0; (4,5,2,1,2,1,4,5) gives x = 0.79289322, y = −0.20710678 | derived by hand from the ISO/TS 12913-3:2019 formula | `1e-12` |
| canonical equals iso2019 at 45° on 1..5, exhaustively over all 5^8 = 390,625 responses | plan §4.2; FR-2.3 | `atol 1e-12` (measured max 2.8e-16) |
| aletta2024 hand values, `aletta-2024@1` eng angles (0, 46, 94, 138, 177, 231, 275, 340) on 1..5, items in position order PAQ1..PAQ8: neutral (3,3,3,3,3,3,3,3) gives x = 0.081548638586, y = 0.106399432136, off the origin because Σ cos θ_i = 0.280655509 ≠ 0 (canonical gives (0, 0)); (5,3,3,3,1,3,3,3) gives x = 0.468703026294, y = 0.094906448962; (4,5,2,1,2,1,4,5) gives x = 0.872773717478, y = −0.011682271564 | computed by hand from Aletta et al. 2024 Eqs. 1 to 3 (λ = (ρ/2) Σ\|cos θ_i\|, ρ = 4), checked at 30 digits with mpmath for draft 0.4 | `1e-12` (float64 sums of 8 terms) |
| aletta2024 equals iso2019 at 45° on 1..5: the two 2019 hand values above, and exhaustively over all 390,625 responses | Aletta et al. 2024 Eq. 3 (λ = 4 + √32 at 45°); FR-2.3 | `atol 1e-12` |
| committed `aletta-2024/1.toml` matches its pinned hash, its angles equal §6.8.2, and its 12 High rows equal a transcription of the paper's Table 4 committed as `tests/science/fixtures/aletta_2024_table4.csv`; its six tier-only languages have the tiers of §6.8.2 (regeneration is `reference`, §7.5) | SATP-circumplex-validation@3509da3; Aletta et al. 2024, Tables 2 and 4 | exact |
| committed `satp/1.5.toml` holds the 23 tiers and 13 High angle rows of the chapter notes' Tables 4.7 and 4.8 (parsed from `design/literature/mitchell-et-al-2026-satp-chapter-4-notes.md`), and its cmn angles equal `aletta-2024@1`'s (§6.8.4) | the 2026 chapter, pp. 19, 20, 24 | exact |
| every language Aletta et al. 2024 covers (18) has a translation of all eight positions with that paper's citation block, and its labels equal a transcription of Table 4; each gap language (pol, yue, tha, bra, alb) has no translation and `label()` returns the English label (§6.2.3) | Aletta et al. 2024 Table 4 | exact |
| Spearman equals the tie-corrected formula on tied Likert samples | Part 3 "Spearman with ties" | `1e-12` |
| effect sizes on hand-computed cases | §6.13.3 citations | `1e-12` |
| committed subset matches its pinned sha256; the FR-1.5 acceptance test on the subset counts of §6.7, 1552 rows, 16 excluded (16 `incomplete_attributes`, 0 `all_identical`), 10 `all_midpoint` flagged, 0 `excessive_missing` (row checks against the source are `network`, §7.5) | the bundled subset; the 0.x tutorials' processing (PRD 0.9 FR-1.5) | exact |
| committed course dataset matches its pinned sha256, has 456 rows with the site counts of §6.20.4, and reads through its mapping with no `CodingError` | the bundled course dataset | exact |
| KDE matches committed seaborn fixtures (0.13.2 and 0.11.2) for two locations, data extent, no clip and clip ±1, `bw_adjust` 1.0 and 1.2: the density grid values and the contour **levels**; and line contours computed with contourpy `lines` on the same grid and levels equal matplotlib's line contour vertices stored in the fixtures. Filled polygons are not compared with seaborn, whose fills come from matplotlib's own filled contouring | `scripts/make_seaborn_fixtures.py` run once per seaborn version (below) | `rtol 1e-9` (measured agreement 2.3e-15; margin for the 3.11 scipy fork) |

The seaborn check runs on every PR because it guards owner row #10 and needs no tool at test time.
The fixture generator stores matplotlib's line contour vertices (`ax.contour` at the seaborn
levels) beside the grid and levels. Seaborn 0.11.2 does not run on today's numpy and pandas, so
its run pins the environment in the script's PEP 723 header, and the fixture records the versions
it was made with:

| seaborn | Run | Pins (checked on 2026-10-31: installs and draws a `kdeplot` with `bw_adjust=1.2`) |
|---|---|---|
| 0.13.2 | `uv run scripts/make_seaborn_fixtures.py --seaborn 0.13.2` | current releases |
| 0.11.2 | `uv run --python 3.11 scripts/make_seaborn_fixtures.py --seaborn 0.11.2` | `numpy<2` (1.26.4), `matplotlib==3.7.5`, `pandas<2` (1.5.3), `scipy<1.14` (1.13.1) |

A PEP 723 header holds one environment, so the script keeps one header per seaborn version: the
builder splits it into two scripts, or one script that re-invokes itself, whichever is shorter.

### 7.3 Fixtures

| Fixture | State | Test |
|---|---|---|
| ISD 1.0.1-alpha.1, 1.0.1-alpha and 0.2.3 files, hashes, counts | available (Zenodo, §6.20.3) | subset checks active offline; full-file checks nightly |
| Course dataset | available (0.x `origin/main`, §6.20.4) | active |
| Aletta 2024 machine-readable angles and published Table 4 | available (`design/literature/`) | active |
| SATP v1.5 tiers and High-tier angles | available (chapter notes, `design/literature/`) | active |
| Translated attribute labels for the 18 languages of Aletta et al. 2024 | available (Aletta 2024 Table 4; the chapter prints none) | active |
| seaborn reference outputs | generated | active |
| FR-1.5 acceptance counts (the 0.x tutorials' 56 on ISD v1.0) | available (computed from the ISD, §6.7) | subset counts active offline; full counts nightly |
| Aletta 2024 formulation hand values | available (computed from the paper's Eqs. 1 to 3, §6.9.1) | active |
| Translated labels for pol, yue, tha, bra, alb | no source in hand (§6.2.3) | the fallback-to-English test is active; the gap is documented |
| WHO-5 cut-offs from Topp 2015 | primary source to be read (S11) | constants active, citation check a release task |

No M1 test awaits an owner input. If one arises later, it uses `pytest.skip`, never `xfail`
(with `xfail_strict`, an unexpected pass fails), and its skip message starts "awaits owner input:"
so S11 can grep for it.

### 7.4 Regression baselines (`tests/regression/`, marker `regression`)

Golden pipelines on the subset: read → project (equal; `aletta-2024@1`, where `on_uncovered` is
unused because the set covers all five languages; and the default `satp@1.5`, where the 69 cmn rows
carry the Medium flag) → describe → item statistics → density by location and 50% contours → WHO-5;
and the course dataset read through its mapping. Baselines are committed CSV and JSON under
`tests/regression/baselines/` with the library versions that produced them; `created_utc` and
version keys are excluded from comparison. Tolerances: `rtol 1e-12` for the package's own
arithmetic, `rtol 1e-9` for KDE outputs (scipy differs across the 3.11 fork). Changing a baseline
needs owner review (GOVERNANCE.md).

### 7.5 Network and reference tests

Neither marker runs in the default `task check` or `task docs`, so the default run needs no
network (§3.3).

| Marker | Test |
|---|---|
| `network` | each registered ISD version (1.0.1-alpha.1, 1.0.1-alpha, 0.2.3) fetches and matches size, md5 and sha256; `feature_columns` match the header; v0.2.3 reads through its mapping and sheet (§6.20.3) |
| `network` | the FR-1.5 acceptance test on the full ISD: quality counts on 1.0.1-alpha.1 and on 1.0.1-alpha, 37/19/53/0/52 and 56 excluded on each (§6.7) |
| `network` | v0.2.3 reads with its 607 recording-only rows kept and excluded with reasons, and the 2022 filter gives 1338 rows, 46 excluded, 1292 included (§6.20.3, §6.23) |
| `network` | each subset row equals its source row in record 10672568, through the key |
| `network` | each course-dataset row equals its ISD row on the compared columns, through the key (§6.20.4) |
| `network` | `who5` differs from `WHO_Sum` on exactly the 32 rows with a missing item |
| `reference` | `aletta-2024/1.toml` regenerates byte-identically from the downloaded SATP CSV (§6.8.2) |
| `reference` | the 2022 paper page re-renders (`task docs:papers`) and prints its row count |

### 7.6 Plot tests

Plot tests assert on artist data, never on pixels: scatter offsets equal `coords.xy` of the valid
rows; contour paths equal `ContourSet` polygons; the default colours are `petroff10`; axis labels
follow `label_language`; `plt.get_fignums()` is empty after a call with `ax=None`; the passed `ax`
is returned; a kwarg (`color`) reaches the artist; the N rule's refuse, override and grouped
fallback cases; saving to PNG, SVG and PDF succeeds; Likert bar widths equal `response_shares`;
joint marginal curves equal `marginals`; `radar` with `ax=None` returns polar axes and with a
Cartesian `ax` raises `AxesError`; a figure saved through `save_figure` has a sidecar whose
provenance equals the source run's. A synthetic indoor circumplex (FR-2.0) runs every analysis and
plot. `"matplotlib" not in sys.modules` after a bare `import soundscapy` (§6.15).

**Tutorial display check.** After `task docs`, a docs test reads `docs/_freeze/` for each
`docs/tutorials/*.qmd` and asserts that every cell calling `sspy.plot` has an image output
(§6.15), so a cell that forgot `ax.figure` fails.

### 7.7 Coverage and platforms

Branch coverage at 80% or more, never lowered. CI runs 3 OS × Python 3.11 to 3.14, the `lowest`
job and the install jobs (§5.6). Golden numbers must pass on both sides of the numpy, scipy and
contourpy lock fork.

## 8. Acceptance

| Requirement or exit | Shown done by |
|---|---|
| Phase 0 exits | §5.10 |
| M1 release gate | §9 S11: every owner-blocked M1 Must is a release blocker unless the owner waives it in writing in the release PR; today none is owner-blocked |
| FR-1.1 | tests reading CSV (`,` and `;`), TSV, xlsx with one and several sheets, utf-8-sig and cp1252, on pandas and polars inputs |
| FR-1.2 | templates ship; `write_template` and `template` tested; the canonical layout reference page renders |
| FR-1.3 | ISD, v0.2.3, course and canonical mappings round-trip; a non-English, relabelled export maps and reads; the course dataset (renamed columns) reads through its mapping |
| FR-1.3a | `suggest_mapping` on ISD, canonical, the course dataset, a Qualtrics-shaped fixture and the messy REDCap fixture; ambiguous matches never picked; CLI and notebook paths give byte-identical `mapping.toml` |
| FR-1.4 | `split` and `by` over location, session and group roles; a survey with only location ids works; `join_context` (on a `Survey` or `Coordinates`) and `add_table` join location and session tables without changing the row count, and the notes count unmatched keys (§6.5.1) |
| FR-1.5 | rules × actions tested; the acceptance test of PRD 0.9: the default rules reproduce the 0.x tutorials' processing, as subset counts (default run, science) and the full-ISD 56 = 37 `incomplete_attributes` + 19 `all_identical`, with 53 `all_midpoint` flagged and 0 `excessive_missing` (nightly, `network`) (§6.7) |
| FR-1.7, FR-14.4 | init, add, project, save, then reopening and reading the saved run gives identical coordinates; re-running after a settings edit makes a new run and never reads the old sidecar as input (§6.3.4); tampered raw raises `DataIntegrityError` |
| FR-2.0 | the synthetic indoor circumplex runs every analysis and plot |
| FR-2.1 | `ScaleViolationError` on a 0..4 export; `exclude` path records reasons |
| FR-2.2 | median, IQR and range by group, with the quantile method recorded |
| FR-2.3 | science tests of §7.2: canonical equals iso2019 exhaustively at 45°; the 2019 hand values; the aletta2024 hand values from Eqs. 1 to 3, including the neutral response off the origin, and its reduction to the 2019 formula at 45° on 1..5; `FormulationError` for iso2019 on 1..7 |
| FR-2.3 (aletta2024) | selectable as `formulation="aletta2024"` under `sal-2026`, recorded as an override; refused by both ISO profiles with `ProfileConstraintError`; no warning; the §7.1 neutral-off-origin property test |
| FR-2.4, FR-2.4a | resolver branches, tier-only languages included, on the synthetic three-tier set and the mixed-language subset sites; the mix report; `satp@1.5` and `aletta-2024@1` science tests (§7.2); under the default profile the cmn rows project with one `MediumConfidenceLanguageWarning` |
| FR-2.5 | describe statistics; quadrant, half-plane and user-box regions; Σ quadrant proportion + `boundary_share` = 1 per group |
| FR-2.5a | incomplete responses have NaN xy, reason `incomplete_attributes`, and still count in item statistics |
| FR-2.6, FR-2.6a | correlate and compare tables with the FR-2.6a fields; statsmodels how-to renders |
| FR-2.6c | WHO-5 complete-items rule; source profile table |
| FR-2.9 | three profiles; ISO lock refusals; overrides recorded |
| FR-3.1, FR-3.2 | seaborn fixtures; the 50th-percentile contour of the 2022 paper by default and other fractions on request, each with `enclosed_share` computed from `z_points`; `density` grouped and ungrouped returns `Densities`; N rule cases |
| FR-8.1 (M1), FR-8.2, FR-8.3, FR-8.4 | plot tests of §7.6; geometry usable without matplotlib (import contract): `density`, `contours`, `marginals`, `response_shares` and `describe` return every number a plot draws, and each plot test compares the artists with them |
| FR-9.1 | export to CSV, xlsx, parquet; figures to PNG, SVG, PDF |
| FR-9.3, journey I (2022) | the 2022 page renders nightly from ISD v0.2.3 with every cell run, printing its row count beside the paper's 1338 |
| FR-10.1 | subset and course dataset load offline with their pinned hashes; the default version is the registry's `default`, 1.0.1-alpha.1; nightly fetch of the three registered versions with size, md5 and sha256 checks; CC BY 4.0 licence and citation in the log line and the sidecars |
| FR-11.1, FR-11.3 | provenance keys present on every result; settings files round-trip and reproduce results |
| FR-11.2, NFR-4 | regression baselines pass on 3 OS and both lock forks |
| FR-12.1 | the `install` job green; the nightly `quickstart` job runs `_quickstart.sh` and `_quickstart.ps1` end to end; an owner-led one-hour dry run on Windows |
| FR-12.3, FR-12.4 | every user-guide page has a valid `guide-section` (§5.9); great-docs lint clean; the API documentation test of §7.1 (every public object documented, every procedure with a `References` section) |
| FR-14.1, FR-14.5 (A) | `CliRunner` runs journey A end to end without prompts, including `who5`, `sources`, `density`, `contours`, `angle-set add` and `context add`; exit codes |
| FR-14.2 | parquet plus JSON sidecar, CSV and Excel export for every run, `Densities` and `ContourSet` included; figures saved with a provenance sidecar |
| FR-14.3 | pooch progress bar on download; no cancel hook (owner row #38) |
| FR-14.7 | the "No GUI toolkit" import contract (§5.2.1) passes in `task lint` |
| NFR-1 | the `install` job, `--only-binary :all:` |
| NFR-2 | the `check` job on ubuntu, windows and macos × Python 3.11 to 3.14 (§5.6.1), with golden numbers passing on both sides of the lock fork |
| NFR-5 | the default `task check` and `task docs` run with no network (network tests are deselected by marker, §5.2); the only download is `fetch`, which logs one INFO line naming URL, size, version, licence and citation (§6.20.2) |
| NFR-6 | the API documentation test of §7.1 (every procedure has a References section with a key in `references.bib`); every result records its profile `{id, version, hash}` and every edition-dependent choice (formulation, angle set) in its settings |
| NFR-7 | the licence column of §6.19 and the test that every runtime dependency has an entry and declared licence |
| NFR-8 | CI runs the tests on all supported platforms (NFR-2 row), the science fixtures by default, and builds and executes the docs (`docs` job) |
| NFR-9 | the layers and import contracts of §5.2.1 pass; the module layout of §6.1 holds at review, no module growing past one responsibility; ADRs written in S1 |
| NFR-10 | `cividis` and `petroff10` defaults; English interface; translated attribute labels for the sourced languages (the 18 of Aletta et al. 2024), each with a citation, and the documented English fallback for pol, yue, tha, bra and alb (§6.2.3, §7.2) |
| NFR-11 (in part) | `CITATION.cff` validates with `cffconvert --validate` in the `citation` job |
| NFR-12 | GOVERNANCE.md and CODEOWNERS, which names `@MitchellAcoustics` on `src/soundscapy/_data/`, `tests/science/`, `tests/regression/` and `design/` |
| NFR-13 | silent import, with matplotlib not imported; one INFO line per read and per download; pooch's own INFO lines hidden inside `fetch` only; `set_log_level("WARNING")` silences all |
| Journey A | tutorial A renders offline with every cell run, and every plot cell has an image output |

## 9. Build order

Slice 1 is a thin end-to-end path, but its load-bearing contracts are built and tested first. Every
slice ends with the AGENTS.md sequence: `task check`, then `/simplify`, then `/code-review`.

| Slice | Delivers | Done when |
|---|---|---|
| S0 Phase 0 | §5 | §5.10 |
| S1 Thin slice, contracts first | (a) errors, `_model`, hashing with golden vectors, provenance envelope, `_log`, `_frames`, `Status`, namespace constants, `_data` registry with hash pins; (b) `Scale`, `Axis`, `Circumplex`, `iso-12913-outdoor/1.toml`; (c) the projection kernel on bare numpy; (d) the subset generator and CSV, a minimal `SurveyMapping` (attributes, ids, language), `read_survey` from a path, `response_id`, `Coding`; (e) `project` with equal angles, `Coordinates`, `OriginSegment`, `attach`; (f) `plot.scatter`, `load_isd(subset=True)`, the new `_quickstart.py`. Then ADRs: Coordinates carries settings, response_id, the `ssp_` namespace, h1 hashing | the science tests for exhaustive equality, bounds, neutral, constant shift and 2019 hand values pass; hash vectors, TOML round-trips, pandas 2.2, pandas 3.x and polars frame tests and the lazy-frame `TypeError` pass; `sspy.plot.scatter(sspy.project(sspy.load_isd(subset=True), angles="equal"), by="location")` returns Axes whose offsets equal xy; import is silent; `task check` passes |
| S2 Status, quality, full mapping | all construct groups, flags, carry; missing codes, value maps, reverse; `ScaleViolationError` and `exclude`; `QualitySettings` with every rule; the course dataset, its generator, mapping and loader (§6.20.4) | subset counts 1552/16/0/10/8 pinned in the default run; full-ISD counts pinned in the nightly `network` test; the row-count property holds; every rule × action tested; the FR-1.5 acceptance test passes on the subset (default run) and is in place nightly on the full ISD; the course dataset reads through its mapping |
| S3 Angle sets, profiles, settings, labels | `AngleSet` with tier-only languages, `AngleSelection`, the resolver, `aletta-2024@1` and its generator (with the Table 2 tiers), `satp@1.5` from the chapter notes, the three profiles as versioned files (`<id>@1`, `sal-2026@1` on `satp@1.5`), the precedence rule, overrides and profile carry-over, `save_settings`, `register_angle_set`, `angle_sets=` recorded in full, the synthetic three-tier set (`qaa` to `qad`); the `aletta2024` formulation in the kernel and the `Literal` (§6.9.1); the translated labels of §6.2.3 transcribed into `iso-12913-outdoor/1.toml`, then checked cell by cell by the second agent pass; the §11.4 Table A.1 fixture `tests/science/fixtures/iso12913_3_table_a1.csv` (scale labels and direction per part, the 0.x wording until the owner's check) and a test comparing the shipped `[scale.labels]` with it (amended at S1 plan review, 2026-10-31: §11.4's row was in no slice) | the committed aletta and satp files match their pins and their sources offline (§7.2), and the aletta file regenerates byte-identically in the nightly `reference` run; error, exclude and equal paths tested on the synthetic three-tier set and the mixed-language subset sites; tier-only languages resolve as uncovered and name their tier; the 2019 profile carry-over test passes; the Medium warning fires once for cmn; `iso2019` refused on 1..7; the aletta2024 formulation passes its hand values, its 45° reduction and its neutral-off-origin property, and the ISO profiles refuse it; the 18 sourced languages have labels with a citation and the gap languages fall back to English; every model round-trips through TOML with an equal hash |
| S4 Workspace | §6.18, `export`, `export_run`, `add_table` and `join_context` on `Survey` and `Coordinates` (§6.5.1), the precedence rule with `settings.toml` at rank 3 | init, add, project, save, then reopening and reading the run gives identical coordinates; re-running uses current settings and never the old sidecar; tampered raw raises; each sidecar validates back into its settings model (`read_coordinates` into `ProjectionSettings`); a location table joins without changing the row count |
| S5 Distribution and density plots | `Region`, `describe`, `density` returning `Densities` (grouped by the `DensitySettings.by` field, with `z_points`), `contours` per group, `ContourSettings`, `marginals`, the per-group parquet and set sidecar of §6.12, the `Density` and `Densities` hash vectors, `save_figure`, the N rule, `plot.density`, `plot.fraction_contour`, `plot.joint` | seaborn grid, levels and line contours match to 1e-9; `enclosed_share` equals the share of points with `z_points >= level`; Σ quadrant proportion + `boundary_share` = 1; `read_density` and `read_contours` rebuild equal records; N < 30 warn, refuse and fallback tested; import contracts kept |
| S6 Part 3 statistics and item plots | `item_statistics`, `response_shares`, `correlate`, `compare_groups`, `who5`, `source_profile`, each with every number-changing argument a settings field (§6.13, §6.14), the shared inclusion rule, `plot.likert`, `plot.radar` with `AxesError`, the statsmodels how-to | Spearman identity; `include` and every other choice recorded on every result; effect-size hand cases; FR-2.6a fields present; the WHO-5 network test passes nightly |
| S7 Intake breadth | xlsx, delimiter sniffing, encodings, `suggest_mapping` with ssid, redcap, qualtrics (provisional) and canonical, Qualtrics label-row detection, label detection, TOML drafts with TODOs, templates, the messy-export fixture | FR-1.1, FR-1.2 and FR-1.3a acceptance tests pass on pandas and polars; multi-sheet without `ReadOptions.sheet` raises; ambiguous matches never picked |
| S8 CLI | §6.17, including `who5`, `sources`, `density` (a per-group run), `contours`, `angle-set add` and `context add`; `--profile` and `--settings` under the §6.3.4 rule; great-docs CLI pages (`name: main`) | `CliRunner` runs journey A without prompts; exit codes 0/1/2; the draft and rerun flow; stdout and stderr kept apart |
| S9 Full ISD loader | fetch and load via pooch, the three-version registry of §6.20.3 with default 1.0.1-alpha.1, the v0.2.3 mapping and sheet, pass-through arguments, download line, progress bar, the pooch log filter, error mapping | the nightly fetch verifies size, md5 and sha256 of the three versions; v0.2.3 reads through its mapping; the subset row checks pass nightly; a fake downloader covers the error paths; `set_log_level("WARNING")` silences everything |
| S10 Docs | quick-start (with the exact pre-release pin), tutorial A with the cmn Medium-tier language section and grouped `Densities` (density by location), the 2022 page (frozen, from v0.2.3), explanation pages with `guide-section`, References sections, the nightly `quickstart` job | `task docs` renders offline with every cell run; every tutorial plot cell has an image; the API documentation test passes; `task docs:papers` renders, by whichever §5.9 step Phase 0 recorded; great-docs lint clean; the owner's one-hour dry run passes |
| S11 Release gate | owner fixtures that have arrived switched on, the Topp 2015 citation checked against the primary source, the version set to `1.0.0.dev1`, the PyPI release (owner decision, §4.3) | **The release gate**, one rule (also in §8): every owner-blocked M1 Must is a release blocker unless the owner waives it in writing in the release PR; today none is owner-blocked. No skip remains for an input that has arrived; owner review under GOVERNANCE.md; tagged `v1.0.0.dev1` and published to PyPI by `release.yml` once the trusted publisher is configured |

## 10. Owner inputs

Nothing blocks M1. The owner's second round of answers of 2026-10-31, recorded in PRD 0.9 §10.5b,
resolved every open question of draft 0.3 (§4.3). What remains is one action at release time, one
report for the chapter, and one known gap to fill later:

| Item | Kind | Blocks | Meanwhile |
|---|---|---|---|
| PyPI trusted publisher | owner action, at release time | the PyPI upload only (§5.6.3) | `release.yml` exists and is not triggered |
| The proof inconsistencies listed at the end of the chapter notes: "ten" High-confidence languages in the abstract and concluding remarks against 13 listed; Table 4.7's caption says 22 languages against 23 rows; Table 4.8's caption ("each language passing Step 2") against its 13 High-tier rows | report to the owner, for the correction of the chapter | nothing | none needed |
| Translated labels for pol, yue, tha, bra and alb | known gap, for the owner to fill later | nothing (NFR-10's label clause is met for the sourced languages, §4.1) | English labels, documented (§6.2.3); a filled language ships as a new circumplex version |

Resolved, citing PRD 0.9 §10.5b (row numbers are draft 0.1's, so older cross-references hold):

| Draft 0.3 row | Resolution |
|---|---|
| 4: ISD published exclusion counts; the missingness denominator | The ISD publishes no counts. FR-1.5's acceptance test is the 0.x tutorials' 56 exclusions on ISD v1.0 (§6.7). The denominator is the 2021 paper's wording, 12 items (a call made here, §11.2) |
| 6: Aletta 2024 formulation worked example | Not needed: the formulation is a full M1 Must, tested against hand values from its Eqs. 1 to 3 (§6.9.1, §7.2) |
| 13: the 63 dB split of the 2022 page's panel (d) | 63 dB over the whole v0.2.3 dataset the page uses, the 1338 pre-lockdown rows, confirmed by the owner (§6.23) |
| 21: committing the chapter 4 author proof | Not committed; cited by reference, with the transcribed tables in the chapter notes file (§1, §4.4) |
| Actions: ORCID and GitHub login | Supplied: CITATION.cff carries ORCID 0000-0003-0978-5046 and CODEOWNERS names `@MitchellAcoustics` (§5.1, §5.8). No preferred citation was named, so `preferred-citation` stays out of CITATION.cff |

Rows 1, 5, 8, 15, 16 and 20 of draft 0.1 were answered in the first round (§4.3); the others are
calls made in this spec (§11.2).

## 11. Calls made in this spec, and what is left to the builder

### 11.1 Plan and house-style edits carried by this spec

| Document | Edit |
|---|---|
| Plan §5 | tqdm is a core dependency from M1 (pooch's bar; owner decision #4), not M3; `_log`, `_model` and `tables` join the bottom layer; `cli` is a layer above `workspace` (it calls it), not its independent sibling; the core never imports a GUI toolkit (an import contract, FR-14.7) |
| Plan §6 (Docs convention) | the docs group pins great-docs as a git source at `af17c76` (owner decision, 2026-10-31) until the first PyPI release after 0.17.0; fallback `095db9b` |
| Plan Appendix A (`Coordinates`) | `labels` is a dict of read-only numpy arrays, not a frame; `to_frame` converts at the return (plan rule 4) |
| Plan §4.4 (hierarchy) | names the M1 functions: `survey.join_context` and `workspace.add_table` |
| Plan §4.9 | `describe` reports a per-group `boundary_share` with strict membership, and Σ quadrant proportion + `boundary_share` = 1; `ContourSet` carries settings and provenance; `marginals` and `response_shares` are geometry; `density` always returns `Densities` (one `Density` per group, `by` a settings field) and each `Density` holds `z_points`; the 50% contour is the 2022 paper's 50th-percentile contour (owner decision, 2026-10-31) |
| Plan §4.6 (profiles) | profiles are versioned data (`_data/profiles/<id>/<version>.toml`, `<id>@<version>`), recorded as `{id, version, hash}`; one settings precedence rule (§6.3.4); a run's sidecar is a record, never an input |
| Plan §5 (`datasets` layer) | the ISD registry holds 1.0.1-alpha.1 (default), 1.0.1-alpha and 0.2.3, each with its own mapping; "latest" is resolved at release time; the course dataset ships as `datasets.course_demo` (owner decisions, 2026-10-31) |
| Plan Appendix A | `ProjectionResult` dropped; `Coordinates` carries settings and provenance |
| Plan §4.2 | M1 formulations are canonical, iso2019 and aletta2024 (PRD 0.9 FR-2.3: the Aletta 2024 Eqs. 1 to 3, no centring; selectable, refused by the ISO profiles, used by no profile by default); `satp@1.5` ships at M1 and `sal-2026@1` uses it; an angle set may list tier-only languages, which resolve as uncovered |
| Plan §4.11 | replace "per-run settings in a sidecar take precedence over project defaults" with: a run's sidecar records the settings it ran with; re-running an operation uses the current inputs and settings, and a sidecar is never an input (§6.3.4) |
| Plan §9.1 register | annotate #36 as superseded by PRD 0.9 FR-2.3 and §10.5b (the Aletta 2024 formulation is a full M1 Must); annotate #39 with its PRD 0.9 test, the 0.x tutorials' 56 exclusions on ISD v1.0 (FR-1.5) |
| Plan §4.4 (quality) | the default rules reproduce the 0.x tutorials' processing of ISD v1.0 (56 excluded); `excessive_missing` stays configurable, from the 2021 paper's text |
| house-style SKILL.md | add `soundscapy._model.FrozenModel` beside `pydantic.BaseModel` in `runtime-evaluated-base-classes`; add `minversion = "9.2"` and the `"-m", "not reference and not network"` entry to the pytest `addopts` (§5.2); add `CPY001` to the ignores with its reason; replace the `.claude/hooks/**` per-file ignore with the §5.2 list (INP001, T201, S603, S607, D103, PLW1510, PTH109, C901, PLR0911) and add the `scripts/**` ignore (INP001, T201) for the PEP 723 generators; add the `network` marker and `test:network`, `docs:papers` tasks; say that `tests/templates/` is created at M2 with the first FR-13.1 recipe, not at M1; replace the per-module filterwarnings advice: a scoped ignore names the soundscapy module that calls the library, or matches the message, never a third-party module whose warning uses `stacklevel ≥ 2`; record repo-review ignores; `uv run task` form; actionlint through `actionlint-py` |
| house-style reference.md | its source label says great-docs 0.17.0 but describes main at 4232670; relabel it as main at `af17c76` after the re-copy, and re-check its statements against that commit; record the freeze behaviour check on `af17c76` and which §5.9 fallback step worked; note `{{< include >}}` for shared code; note `guide-section` and the `ax.figure` display convention |
| `.claude/skills/VENDORED.md` and the three great-docs skills | re-copied at `af17c76`; rows bumped (§5.1) |
| `.claude/hooks/*.py` | §5.4.1; the per-file ignore of §5.2; one `ruff format` run on the three scripts, committed on its own with no other change (commit message: "style: ruff format hook scripts (one-off, no behaviour change)") |
| AGENTS.md | the Status paragraph rewritten for the state after Phase 0 (toolchain, CI and docs skeleton exist; `task check` is the definition of done); the `uv run task` line (§5.1) |

### 11.2 Calls made in this spec (a reader should check them)

| Call | Section | Why it matters |
|---|---|---|
| A scale violation raises by default | §6.4.3 | a wrong-scale export otherwise skews every row silently |
| `response_id` from a column, a key, or the physical row number; never memory order | §6.3.2 | ids survive re-sorting and are findable in Excel |
| The `ssp_` prefix, checked at import | §6.3.3 | collisions fail before analysis |
| No `ProjectionResult`; `Coordinates` carries settings and segment-level origin | §6.10 | one point type; pooling needs no shape change |
| One hash scheme, `h1-sha256:` | §6.3.5 | detectable scheme changes |
| Strict region membership plus a per-group `boundary_share` (was owner's call 7) | §6.11 | no fractional respondents; the invariant Σ proportion + `boundary_share` = 1 is checkable; plan §4.9 requires the rule before the first baseline, and it stands on its own merits |
| KDE defaults: seaborn's `kdeplot` defaults with `bw_adjust=1.2` (the 2022 paper's method) for every profile, the chapter's `sal-2026` included; clip to the [−1, 1] square for new work, no clip in the 2022 reproduction's settings (the meaning of the 50% contour is the owner's decision, §4.3) | §6.12 | new work and the 2022 paper agree; owner rows #10 and G4-07 |
| `enclosed_share` computed from `z_points`, the density at each valid point from the same `gaussian_kde`, stored in each `Density`, its array hash and its parquet form (builder check Q5) | §6.12, §6.3.5 | the share is reproducible from the saved record alone |
| `density` always returns `Densities`, with `by` a `DensitySettings` field; `contours` works per group; one parquet per group plus one sidecar for the set; the set's array-hash layout pinned (Q1) | §6.12, §6.3.5 | one return type whatever the grouping, and the grouping is recorded |
| 2021 quality rules (all-midpoint kept with a flag) | §6.7 | the 2022 code's rule was a bug |
| `under_age` and `paper_input` codes in both enums | §6.7 | the default can change without a schema change |
| `under_age` flags by default (was owner's call 18) | §6.7 | keeps rows, as the 0.x tutorials' 56 exclusions did; switchable without a schema change |
| One inclusion rule, `include="quality_passed"`, for every Survey analysis; only `incomplete_attributes` is tolerated | §6.13 | FR-2.5a, without letting straightliners into item medians; one rule instead of four (builder check C5) |
| `who5` returns every row with its status; summaries use included rows | §6.14 | no row disappears; summaries follow the shared rule |
| WHO-5 scored on complete items only; cut-offs raw ≤ 12 and ≤ 7 citing Topp et al. 2015 (was owner's call 11) | §6.14 | the WHO-5 guidance gives no prorating; the citation is checked against the primary source in S11 |
| Quantile method `linear` (type 7, Excel `QUARTILE.INC`), recorded with every result (was owner's call 12) | §6.13.1 | matches the spreadsheets Part 3 users check against; recorded, so a different method is a visible override |
| No dominant-source label at M1 (was owner's call 19) | §6.14 | no published rule to implement; the profile table gives the numbers |
| Classic one-way ANOVA (not Welch) for k > 2 coordinates | §6.13.3 | scipy's Welch ANOVA is newer than the scipy floor |
| `click>=8.2` rather than 8.1 | §6.19 | separated stderr in `CliRunner` |
| `FileFormatError`, `RunExistsError`, `AlignmentError` and `AxesError` added to the error list; `radar` creates polar axes when `ax=None` (K9) | §6.3.10, §6.15 | one class per condition, all under `SoundscapyError` |
| Excel read with openpyxl directly in `_frames`, not via `pandas.read_excel` | §6.5 | row numbers for ids, controlled text conversion |
| The 2022 page is not part of offline `task docs`; the whole page uses the paper's "entire dataset", v0.2.3 filtered on `Lockdown == 0`, which selects exactly the paper's stated 1338 responses; panel (d) splits all 1338 at 63 dB (the owner confirmed the split, §4.3); the page computes and prints its counts, and the 36 rows without attribute answers are excluded by the normal rules and reported (R2) | §6.23 | FR-9.3 asks for the public dataset; the filter is the builder's reading of the paper, checked against its own count |
| v0.2.3's 607 recording-only rows: the mapping's `response_key_blank_ok = ["RecordID"]` lets that key part be blank (the joined ids stay unique, checked); a blank language cell is unmapped (`""`) and matters only for a projected row; the rows are excluded by `incomplete_attributes` and reported, never dropped (R3) | §6.3.2, §6.4.1, §6.20.3 | the workbook's real shape reads without editing the source, and no row disappears |
| Freeze fallback for the 2022 page: `great-docs freeze`, then `quarto render` with execution forced, then the `095db9b` pin; the Phase 0 exit records which worked (Q4) | §5.9 | the page must re-execute nightly whichever way #487 left freezing |
| Docs CI and the freeze-diff check on ubuntu × the newest Python only; the 3-OS `install` job covers installation, the quick-start script and importing (Q6) | §5.6.1, §5.10 | frozen output needs one canonical platform |
| Profiles locking fields under ISO names | §6.3.4 | a number labelled ISO never uses non-ISO options |
| Bundled subset Option C, 40 columns | §6.20.1 | serves tutorial A and course cases 1 to 3 |
| The subset ships under CC BY 4.0 with the record's citation; no synthetic fallback, no licence release gate (was owner's calls 2 and 3) | §6.20.1 | the Zenodo API shows every ISD version as CC BY 4.0, open access |
| `aletta-2024@1` lists the paper's six non-High languages as tier-only, with tiers from Table 2 by the paper's stated rule: jpn Medium (Table 4 says Low) and kor Medium (Table 4 says High, with angles) | §6.8.2 | Table 2, the text of §5.1 and SATP v1.5 agree with the rule; tier-only languages project nothing either way, so only the message's tier differs |
| `satp@1.5` lists cmn as Medium with the Aletta 2024 cmn angles, citing the chapter's §4.6.1; the other Medium languages and the Low ones are tier-only (for the owner to check) | §6.8.4 | the chapter gives angles only for High languages, but says Medium angles "may still improve cross-language comparability" and that cmn's angles are unchanged |
| A tier-only language resolves as uncovered (`on_uncovered` applies) and the message names its tier | §6.8.3 | a language without angles cannot be projected with adjusted angles; the tier tells the user why |
| `angle_sets=` stays on `project` for use without a project; provenance records the full angle table and its content hash, so no result depends on a machine path. This is the spec's reading of owner decision #40 (for the owner to check) | §6.9.2 | the intent of #40 is that results reference registered data, never a path |
| Profiles are versioned data (`<id>@<version>`, first version `1`), recorded as `{id, version, hash}`; `extends` names `<id>@<version>` or a bare id for the latest shipped (Q2) | §6.3.4 | a profile can change without editing a released file (plan rule 15) |
| One release gate: every owner-blocked M1 Must blocks the release unless the owner waives it in writing; after PRD 0.9 none is owner-blocked (Q2, K2) | §9 S11, §8 | one rule instead of per-test exceptions |
| One settings precedence rule, stated once: keyword arguments or CLI options, `settings=`, the project's `settings.toml`, the input's profile, the named or default profile. An explicit `profile=` or `--profile` replaces the base profile and is recorded as the profile, not as an override. A run's sidecar is a record, never an input (Q3) | §6.3.4 | one rule for the library, the workspace and the CLI; reopening returns saved numbers, recomputing is a new run |
| Every argument that changes a number is a settings field and is recorded; `_UNSET` for every optional keyword; `""` means no grouping (K4) | §6.3.4, §6.13, §6.14 | FR-11.1 for every operation, not only `describe` |
| `join_context` accepts `Survey` or `Coordinates` and returns a `Table` whose notes count unmatched keys, with one INFO line (K1) | §6.5.1 | only those records carry id roles; a `Table` can carry the unmatched count |
| `Coordinates.settings` typed as `FrozenModel`; `read_coordinates` validates it into `ProjectionSettings` (K8) | §6.10 | `distribution` must not import `projection` |
| A minimal Qualtrics alias table ships at M1, marked provisional like the REDCap one (K6) | §6.6.2 | PRD FR-1.3a names Qualtrics |
| `redcap.toml` holds the ISD names, marked provisional, until the SSID REDCap dictionary is in hand; the Table A.1 agreement labels are 0.x's until the owner checks the wording | §6.6.2, §6.2.3 | both have working fallbacks and need no decision |
| The install scripts use the exact pre-release pin, which uv accepts without `--prerelease`; a nightly job runs them end to end against the git tag before the first release and PyPI after (K7) | §5.6.2, §5.9 | `1.0.0.dev1` is a pre-release that a bare `uv add soundscapy` skips |
| The course dataset's mapping sets `language.value = "eng"`; the ISD records 2 of its 456 rows as spa, and the sidecar says so | §6.20.4 | the file has no language column, and adjusted angles need a language |
| Translated labels stored as published, with the chapter's codes `bra` and `alb` kept so the angle set and a later translation join | §6.2.3 | matching the source beats a silent recoding; the notes give the ISO 639-3 reading |
| Labels ship for the 18 languages of Aletta et al. 2024, from that paper with its citation; pol, yue and the Medium and Low-tier languages with no source (tha, bra, alb) fall back to English, a documented gap for the owner to fill later, not a release blocker; the §7.2 label test covers the sourced languages and asserts the fallback for the others (R4) | §6.2.3, §4.1 | the 2026 chapter prints no label tables (checked), so no other source is in hand |
| The "second reader" of the hand transcription of Aletta 2024 Table 4 is an independent second agent pass at build time that re-reads the rendered page and compares every cell; differences go to the owner (R5) | §4.4, §6.2.3 | the table has no text layer, so no machine check exists |
| The `aletta2024` formulation accepts any declared scale and angle set and emits nothing: choosing it is deliberate. No profile uses it by default; it is selectable under `sal-2026` for reproducing Aletta et al. 2024, and the ISO profiles refuse it (P1, R1) | §6.9.1, §6.3.4 | a warning on an explicit choice would be noise (plan rule 10); the docstring and the projection explanation page state its neutral offset |
| The one-cell hosted-notebook install (`%pip install 'soundscapy==1.0.0.dev1'`) is on the install page, and the nightly `quickstart` job runs its pip command in a fresh venv (R8) | §5.6.2, §5.9 | NFR-1 asks for a single install cell in hosted notebooks |
| tqdm a core dependency at M1 (was owner's call 9) | §6.19 | owner decision #4 makes `tqdm.auto` the default reporter, and pooch's bar needs tqdm |
| ISO 12913-3:2024 profile uses the canonical formulation, labelled "interpretation pending the owner's check of the text" (was owner's call 10) | §6.3.4 | already PRD §10.6; not a new question |
| Canonical template column names as in §6.6.3 (was owner's call 14) | §6.6.3 | changeable until the course material is written |
| Angle set named `aletta-2024@1` (was owner's call 17) | §6.8.2 | names the paper the angles come from |
| Profile carry-over: `profile` unset inherits the input's profile; an explicit different profile is recorded as `profile`, with the input's in `profile_inherited`, and `overrides` stays empty (§6.3.4 wins; R7) | §6.3.4 | a 2019 pipeline named once stays 2019; a change of profile is visible in provenance without being mistaken for a field override (consistency check B6) |
| No ±inf sentinels: `allow_inf_nan=False` on `FrozenModel`, and `"none"` for no clip or no box | §6.2.1 | canonical JSON cannot hash ±inf; a literal is simpler than a string serialiser for infinities (consistency check B3) |
| Mapping TOML table form is normative, with a before-validator and serializer; the shipped mapping round-trips byte for byte | §6.4.4 | the file users edit is the readable form. The mapping hash covers `model_dump(mode="json", by_alias=True)` (amended at S1 plan review, 2026-10-31), in which the serializer writes the key-to-name fields as tables; canonical JSON sorts their keys, and attribute keys stay positions, so `pleasant = "x"` and `PAQ1 = "x"` hash alike (B4, K9) |
| `frame_hash` normalises numerics to float64 with null as NaN before hashing | §6.3.5 | pandas turns an int column with a missing value into float; the same content must hash alike (C6) |
| `attach` aligns by id for mapped ids and by position only for `row-<n>` ids; no hash fallback | §6.16 | the alignment rule follows from §6.3.2; a hash comparison could not align a re-sorted frame (C3) |
| Default run offline: only the generated aletta file and the subset are committed and checked by hash; regeneration and source-row checks run nightly | §6.8.2, §6.20.1, §7.5 | §3.3's anti-goal on network use (B5) |
| Seaborn agreement checked on grid values, levels and line contours, not filled polygons | §7.2 | seaborn's fills come from matplotlib; contourpy's filled output is not the same object (C4) |
| `marginals` and `response_shares` as geometry functions | §6.12, §6.13.1 | FR-8.4: plots compute nothing (B9) |
| `ContourSet` carries settings and provenance; `regions` and `by` are `DescribeSettings` fields; figures get a JSON sidecar | §6.11, §6.12, §6.18 | FR-11.1: every number traceable, figures included (B10) |
| `Densities` and `ContourSet` serialise as parquet plus JSON sidecar | §6.12 | FR-14.2 (B11) |
| CLI gains `who5`, `sources`, `density`, `contours`, `angle-set add`, `context add` | §6.17 | FR-14.5 journey A coverage, one library call each (B12) |
| Location and session tables through `join_context` and `add_table` | §6.5.1 | FR-1.4's minimal hierarchy, plan §4.4 (B16) |
| `Coordinates.labels` as read-only numpy arrays | §6.10 | no frame inside the core; plan rule 4 (B20) |
| Workspace export named `export_run`; `sspy.export` unchanged | §6.18 | two functions with one name confused the API (B8) |
| ~~Tutorial plot cells end with `ax.figure`~~ (superseded 2026-11-01: `ax=None` uses pyplot, review F5); a docs test checks each tutorial plot cell renders an image | §6.15 | notebooks show plot output |
| Tutorial A's language section shows cmn projecting with the Medium warning under `satp@1.5`, and the Low and uncovered paths on the synthetic `qaa` to `qad` set | §6.22 | no ISD language is Low or uncovered under `satp@1.5` (C2, S4) |
| The FR-1.5 acceptance test is PRD 0.9's 56 exclusions, run on the subset counts by default and on the full ISD nightly; it no longer awaits an owner input. `excessive_missing` keeps its 2021-paper definition and default (`exclude`, missing share > 0.5), and its denominator is the 2021 paper's wording, the 12 attribute and source-dominance items (B19; R6) | §6.7 | the acceptance criterion is testable now, and the ISD publishes no counts to wait for |
| matplotlib imported inside `plot` functions only | §6.15 | `import soundscapy` cannot trigger matplotlib's font-cache message (NFR-13) |
| Pooch's INFO lines hidden by a filter inside `fetch`, not by `setLevel` at import | §6.3.9 | no third-party state changed on import; `setLevel` is ignored after pooch's first message |
| `lowest` CI job sets `UV_RESOLUTION=lowest-direct` for the whole job | §5.6.1 | otherwise `uv run` re-resolves to the newest versions (B2) |
| Hook scripts: one `ruff format` run and a narrowly scoped per-file ignore | §5.2 | standalone PEP 723 hook scripts, kept short and dependency-free (B1) |

### 11.3 Left to the builder

- Private helper names and internal module splits within a listed file.
- The exact wording of error messages and hints, within the content rules of §6.3.10.
- The `frame_hash`, `Density` and `Densities` vectors, computed after the §6.3.5 normalisation is
  written and then pinned.
- The sha256 pins of the regenerated subset and of the course copy, once their generators have
  written them (§6.20).
- The checkout action SHA and the repo-review ignore list (§5.6).
- The forced-execution flag of the §5.9 fallback step 2, verified at Phase 0.
- Whether the seaborn fixture generator is two scripts or one that re-invokes itself (§7.2).
- How `suggest_mapping` normalises names beyond the stated rule, and the alias tables' contents
  from the ISD names.
- The layout and wording of tutorial A and the explanation pages, within §6.22.
- Whether CONTRIBUTING.md needs the system Quarto note for Windows or macOS (§5.4 fallback).
- Whether great-docs interlinks can be enabled offline (§5.9).

### 11.4 Resolutions from the final consistency check (2026-10-31)

The last consistency check on draft 0.4 found no blocking problem. Its findings are resolved here
rather than by another rewrite. Each row is binding for the plan and overrides any earlier sentence
in this spec that it contradicts.

| Finding | Resolution |
| --- | --- |
| ty 0.0.95 checks `scripts/` and does not read PEP 723 headers, so generator imports such as seaborn fail | Add `[tool.ty.src] exclude = ["scripts"]` in Phase 0 and verify it with the pinned ty. The generators are run with `uv run`, never imported by the package. |
| The `lowest` job resolves dev and test dependencies without a floor to ancient releases (polars 0.0.1, pandas-stubs) | Every dev and test dependency gets a floor when added (the current release at that time), as §5.2 does for the others. A test in Phase 0 asserts every `pyproject.toml` dependency has a lower bound. |
| `angle_sets=` on `project` against FR-2.4 and owner decision #40 | Kept for library use without a project; the full table and its hash go in `data_refs`, so no result depends on a machine path. Listed in §10 below as a non-blocking owner check. |
| `contours(Density)` has no settings or provenance to inherit | `contours` accepts only `Densities`; one density is a single-group `Densities`. `upstream` copies the set's `settings_hash` and `input_hash`; the profile is inherited from the set. |
| A grouped density with a group of fewer than 3 valid points raises, against FR-3.2 | `density` skips groups with `n_valid < min_n` (never raises for them), records each skipped group and its n in `Densities`, and emits one `SmallSampleWarning` listing them. The grouped plot draws annotated scatter for skipped groups. A single-group call with `n_valid < 3` still raises `SmallSampleError`. |
| `excessive_missing` "fires on no ISD row" | Fires on no ISD v1.0 row (both 1.0.1 versions); on v0.2.3 it fires on the recording-only rows. |
| `who5` against `WHO_Sum` | `WHO_Sum` is the 0..100 percentage with missing items counted as 0. The network test compares `who5_percent` with `WHO_Sum`: equal on every complete row, and different (NaN against a number) on exactly the 32 rows with a missing item, 9 of them all-missing with `WHO_Sum` 0. |
| 2019 hand values quoted to 8 decimals but compared at 1e-12 | The fixture uses the closed forms x = (1 + 2√2)/(2 + 2√2) and y = −1/(2 + 2√2), compared at 1e-12. |
| §10 claims an exhaustive owner list while other sections defer checks to the owner | §10 gains a "non-blocking owner checks" list: the Table A.1 label wording (0.x wording until checked), the SSID REDCap dictionary (ISD names until supplied), the cmn Medium-with-angles reading (§6.8.4), the `angle_sets=` reading of #40, and the 63 dB split already confirmed. None blocks M1. |
| Plan statements contradicted without an edit | §11.1 also edits plan §8 G7-03 (profiles are versioned data), plan §8 G3-06 (location and session tables join at M1 through `join_context`), and plan §4.6 (provenance key names as §6.3.6). `DensitySettings` gains `normalisation: Literal["per_group"] = "per_group"`, recorded. |
| Two decimal-mark settings | `ReadOptions.decimal` is dropped; `Coding.decimal` (per group, default ".") is the only one. `ReadOptions` is part of the input description, recorded in `provenance["read"]` and `dataset.toml`, not a settings model. |
| The translated-labels test is circular | The second agent pass writes `tests/science/fixtures/aletta_2024_table4_labels.csv` independently from the rendered paper; the shipped TOML is compared with it. |
| Panel (d) wording | "The included rows of the 1338, split at 63 dB over the whole set, not per location; rows without LAeq are in neither bin and are reported." |
| ISD mappings sit outside `_data/` | All shipped mappings live under `src/soundscapy/_data/` (`datasets/isd/`, `datasets/isd_v0.2.3/`, `datasets/course_demo/`), registered with h1 pins and covered by CODEOWNERS. The loader modules read them from there. |
| An explicit `settings=` file whose `extends` disagrees with the inherited profile | An explicit `settings=` file's `extends` ranks with an explicit `profile=` (step 1 of §6.3.4). A precedence test covers it. |
| Labels stored twice | `Attribute.labels` holds the reference language only; `Circumplex.translations` holds every other language. `label()` reads the reference language from the attribute and the rest from translations; a validator rejects a translation for the reference language. |
| The quick-start scripts would launch JupyterLab in CI | The scripts stop after the first plot; launching JupyterLab is a separate line on the page, not in the scripts. |
| FR-2.1's Table A.1 clause has no acceptance test | A science fixture transcribes Table A.1 (scale labels and direction per part); a test compares the shipped scales with it. The wording is checked by the owner (non-blocking list). |
