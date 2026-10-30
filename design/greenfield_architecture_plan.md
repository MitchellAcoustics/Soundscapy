# Architecture plan for the soundscapy rewrite

Status: **draft 0.5** (2026-10-30), in step with PRD draft 0.7. This is the document AGENTS.md means
by "the architecture plan". Draft 0.5 separates the load-bearing architecture from deferred detail.
Draft 0.4 had the right shape, but it specified field lists, signatures and flags for five
milestones in the same voice as the M1 contracts, and it missed several decisions M1 needs. A review
of 0.4 (`design/reviews/plan-0.4-overengineering-review.md`) found it over-engineered in depth, not
in shape. Draft 0.5 keeps the shape, cuts the depth and closes the M1 gaps. Bracketed G-numbers
refer to that review's decision register. Research records (the skew-normal search, the 0.x
branches, the LISQM backend evaluation) are in `design/research/2026-10-30-research-records.md`.
The owner-decision register, §15 in drafts 0.2 to 0.4, is now §9. Draft 0.4 stays in git history and
is not normative. The next step is the Phase 0 and M1 spec.

## 0. How to read this plan

| Tier | Meaning | Where |
|---|---|---|
| Commitment | Fixed now. Changing one needs an owner decision and an update to this plan. | §3, §4 |
| Rule | How to make a later call. The milestone spec applies it. | §2, §5, §6 |
| Deferred | Open until a named trigger. The constraints that already bind it are listed. | §7, §8 |

Content belongs in this plan only if at least one of these holds:

- it crosses a module or milestone boundary;
- reversing it after the first persisted file or alpha release would force a migration or a rewrite
  across modules;
- it constrains how future decisions are made (a rule or a trigger).

Everything else goes in the spec of the milestone that first needs it: field lists, defaults,
signatures, flags, thresholds, library choices and versions, file layouts beyond the M1 subset, and
recipe text. A default fixed by the PRD or the cited science may appear here with its source. Code
in this plan is always illustrative. When the plan and a spec disagree, the spec wins for its
milestone, and the plan is updated in the same change.

## 1. Forces

| Force | What it means for the design |
|---|---|
| Capacity | One part-time, unfunded maintainer plus coding agents (NFR-9). Small modules, no bespoke frameworks, nothing built before its first user. An abstraction a contributor cannot explain in one sentence is suspect. |
| Faithfulness | Every number is defensible against a standard or a paper. Settings, data versions, seeds and methods are recorded with each result (FR-11) and shareable as text (FR-14.2). |
| Four front ends | A library, teaching material, a CLI and a later GUI drive one core (FR-12, FR-14). The core is headless and serialisable, and exchanges work through a project directory. |
| Generality | The circumplex is not eight English words (FR-2.0). Survey data arrive in any eager dataframe library with the user's own column names. |
| Units of work | A response is the unit of analysis; a distribution of points is the unit of interpretation (FR-2.5, FR-3, FR-6). One point type serves every downstream feature. |
| Item data survive | Item responses stay separate from derived coordinates, with missingness explicit (FR-1.8). The package never rewrites the user's table. |
| Big unknowns | The prediction model (M4), the M3 backends, the GUI (FR-14.7), the SPI paper's exact method and several owner fixtures (PRD §10.3). The plan must not guess them. |
| Alpha freedom | Releases from M2 are 1.0 alphas and nothing is frozen at M2 (PRD §9). Breaking changes are cheap until the first non-alpha release. |
| Install simplicity | One install, no compiler, no R, on a course laptop (NFR-1). Audio and torch stay optional and are never imported until asked for. |

M1 and M2 (survey analysis and SPI, no audio) are the 1.0 candidate. The architecture must be
complete enough at M1 that M3 and M4 add modules rather than reshape them, and must build nothing
for M3 to M5 early.

## 2. Design rules

Each rule says what it rules out, because that is what makes it testable.

1. **Functional core, imperative edges.** Pure functions over arrays and frames; no mutation, global
   state, file writes, printing or figures. IO, the project directory and the CLI sit at the edges.
   Rules out stateful "analyser" and "manager" classes.
2. **Results are immutable and carry their settings and provenance.** Rules out results that cannot
   be re-derived.
3. **Objects for values and contracts, not for workflow.** Rules out inheritance from third-party
   classes, deep hierarchies and pipelines disguised as classes.
4. **We do not own the user's dataframe.** Read it through a mapping; never rename or reorder its
   columns; return derived columns beside them. Rules out row objects and backend-specific code
   outside the boundary helper.
5. **Positions, not words.** Attributes are keyed by position (PAQ1 to PAQn, counter-clockwise from
   +x); labels are display data. Rules out code that assumes eight English words or a column order,
   the bug class the SSID codebook order caused before.
6. **Geometry is computed; drawing is thin.** Rules out plotting code that computes and computing
   code that imports matplotlib.
7. **No second implementation of anything a maintained library provides.** The known places where
   the package writes its own code are those with nothing to wrap or that the PRD assigns to it:
   projection, the skew-normal, the 2-D KS distance, SPI, regions, quality rules and item scoring
   (WHO-5), elementary level metrics and IACC (FR-4.4), calibration, channel rules and statistics
   over metric series (FR-4.0), batch orchestration and harmonisation. Rules out a second KDE,
   mixed-model fitter, statistical test or plotting system.
8. **An abstraction comes with its second implementation.** A protocol, registry, selector or plugin
   group arrives when a second concrete implementation exists. Two owner decisions are the
   exceptions: the predictor interface, whose outer shape is fixed in advance (§4.12), and plugin
   discovery at M2 with the first alpha (PRD §9), built as the smallest helper that can list a group.
9. **Each milestone ships alone.** A design call is acceptable only if its milestone is releasable
   and useful without the later ones (PRD §9). Rules out M1 code that serves M3.
10. **Fail loudly, with the override in the signature.** Specific exceptions and warning classes;
    the override is an explicit, recorded argument. A warning fires once per operation, never per
    row, never for a deliberate choice. Rules out silent drops and coercion.
11. **Row count never shrinks.** Every excluded or flagged response keeps its row and states why
    (§4.5). Rules out filtering inside analysis functions.
12. **Hierarchy is id columns and joins, not object graphs.** Groups, sessions, locations and
    recordings are tables keyed by the mapping's id columns.
13. **Headless, but not mute.** The library reports through `logging` (NFR-13); the default output
    is visible and suppressible with one call. Rules out `print`, `input`, `show()` and implicit
    file writes.
14. **Dependency direction is enforced** by a minimal import-linter contract in CI (§5).
15. **Published data are versioned and never edited** (§4.7).

### 2.1 Where heavier implementations should be resisted

Do not build these until §8 names their trigger.

| Resist | Until | Why |
|---|---|---|
| Progress and cancellation protocols | M3 batch runner | No M1 or M2 operation needs them; downloads use pooch's progress bar. |
| Executor-pluggable batch runner | M3, with measured runtimes | The per-file cache may be the whole resume story. |
| Metric registry with a backend selector | Two backends collide at M3 | Rule 8. |
| Entry-point groups beyond the one discovery needs at M2 | A group has a real member | Discovery itself is an owner decision at M2; the groups for metrics and predictors are created with their first member. |
| A `DatasetLoader` protocol, two KS variants, two WAV readers | A second implementation | Rule 8. |
| Multi-seed SPI inside the core result | M2 spec | An agreed Should; a separate helper is likely. |
| Renormalisation machinery (FR-2.5a, a Should) | When the Should is scheduled | The Must (exclude with a reason) needs none of it; when built, FR-2.5a binds in full. |
| `load()` on every record; published JSON schemas | Something reads back; GUI work | One generic serialisation path; pydantic generates schemas on demand. |
| A `METHOD_VERSION` on every procedure | A baselined procedure drifts | Only baselined procedures need one. |
| A stateful `Project` class | GUI design or M3 | Functions and one path-helper module. |
| A persisted circumplex fingerprint | Target files ship (M2) | Value equality is enough at M1. |
| CLI flags for M2 to M4 | Each milestone spec | They wrap signatures not yet fixed. |
| `PlotSpec`, a report engine, a theme system | GUI design, if ever | Geometry plus settings suffice; matplotlib styles cover theming. |
| pandera or another schema DSL | Validation outgrows the mapping | The mapping is the schema (§4.4). |
| A dataframe accessor (`df.ssp`) | Not for 1.0 | Magic, pandas-specific, hard to type, hides the mapping. |
| A uv workspace of several packages | A seam must split | Release and CI cost; subpackage seams keep the option. |

## 3. Commitments

The only list of things that are not open. "Binds from" is the milestone whose code must honour it.

| # | Commitment | Binds from | § |
|---|---|---|---|
| 1 | Functional core; frozen results; arrays read-only, frames copied on construction | M1 | 4.1 |
| 2 | The circumplex is an instrument defined by its content; instrument and effective scale recorded separately | M1 | 4.2 |
| 3 | `Coordinates` is the one point type; rows aligned to responses; `origin` a sequence of segments | M1 | 4.3 |
| 4 | `SurveyMapping` is the schema; stored responses are raw; coding applied once, in a fixed order | M1 | 4.4 |
| 5 | One per-response status (`included`, `reasons`, `flags`); a `response_id` rule; a derived-column namespace; one `n` and `n_valid` | M1 | 4.5 |
| 6 | Settings recorded per operation as frozen pydantic models; profiles are bundles; omitted TOML key means default | M1 | 4.6 |
| 7 | Published data versioned and referenced by `{name, version}` plus hash; no machine paths; one hashing helper | M1 | 4.7 |
| 8 | Every result carries `settings` plus one extensible `provenance` mapping; data files carry a citation block | M1 | 4.8 |
| 9 | Geometry in `distribution`; plot `ax` contract; one KDE with recorded bandwidth; one region-boundary rule | M1 | 4.9 |
| 10 | narwhals at the boundary only; lazy frames rejected; numpy inside | M1 | 4.10 |
| 11 | The project directory is the unit of exchange; named datasets; parquet plus authoritative JSON sidecar | M1 | 4.11 |
| 12 | Extension contracts fixed in shape, grown when needed; framework owns metric statistics; predictor outer shape fixed | Shape now; discovery M2; code M3, M4 | 4.12 |
| 13 | One feature-name vocabulary naming the quantity, not the backend, shared by the ISD feature table, audio results and predictor inputs | M2 | 4.13 |
| 14 | Minimal enforced layering; small typed error vocabulary; one package logger | M1 | 4.14 |
| 15 | `rng` keyword on every random function, as SPEC 7 states | M1 | 6 |
| 16 | No `ssp.project` versus `soundscapy.project` collision: the project-directory subpackage is `workspace` | M1 | 5 |

## 4. Load-bearing decisions

Each subsection opens with what is fixed, then says why (with the alternative it beat) and what is
left to the spec.

### 4.1 Functional core with immutable results

Computation is pure functions over numpy arrays and frames. Result records are frozen; arrays inside
them are set `writeable=False`, and frames are copied on construction. IO, the project directory and
the CLI sit at the edges. No stateful `Project` class exists until a GUI or M3 needs one; until then
one path-helper module in `workspace` owns the directory layout.

**Why.** A notebook, a CLI, a GUI and a test can all call pure functions the same way. Retrofitting
immutability means rewriting every module. Draft 0.1 and the 0.x package grew container classes that
accumulated methods; this beat them.

**Left to the spec.** Which records are pydantic models (no arrays) and which are frozen dataclasses
(arrays inside).

### 4.2 The circumplex is an instrument defined by its content

- `Scale`, `Attribute` (position id, angle, labels by ISO 639-3 code) and `Circumplex` (attributes,
  two named axes, default scale) are small frozen value objects.
- Every analysis works on generic positions. The circumplex owns its axis names, so nothing
  hard-codes `pleasant`. Labels resolve to positions only at entry points: mapping construction,
  region selection and plot labels.
- Records keep the instrument and the effective scale used as separate fields. That keeps both
  readings of FR-6.3 open: whether a 1..7 survey is the same instrument as a 1..5 one for scoring
  is a PRD clarification (§9.2 #1) due before targets ship.
- Identity is by content. At M1 circumplexes compare by value and results embed the full circumplex;
  the persisted fingerprint arrives with target files at M2.
- A circumplex names a reference language, whose labels are the default for plots (NFR-10) and the
  aliases accepted at entry points.

**Angle sets** are versioned data (FR-2.4, FR-2.4a). Identity is `(name, version)` plus the
circumplex they apply to; each language entry has an angle per position and a confidence tier.
Angles resolve per response: an optional language-to-set map over registered sets (by name and
version, no paths), then the default set, then the tier and coverage rules. High-tier languages
project silently; Medium-tier languages project with one warning naming them (FR-2.4a); Low-tier
and uncovered languages follow `on_low_confidence` and `on_uncovered`, each offering `error`,
`exclude` (recommended; reason recorded) or `equal` (equal angles, one warning). Both knobs stay,
because FR-2.4 and the owner's §10.5 decision require them. Every
response records its set in an `angle_set` label, and the mix report is a count of those labels. A
user set uses the same file format and is registered first. Deriving angle sets is out of scope.

**Formulations** are canonical, ISO 12913-3:2019 and Aletta 2024 (FR-2.3). Projection reads the
midpoint and half-range from the effective scale, which is what makes the canonical formulation
reduce to the 2019 one. The Aletta 2024 kernel needs an owner worked example before the M1 spec
(§9.2). The adjusted angle set stays an M1 Must either way.

**Why.** Draft 0.1 keyed attributes by English name. The generalised circumplex (FR-2.0) and the
translation work make positions the stable thing and labels the variable thing.

**Left to the spec.** Field names; how quadrant names are declared (the ISO instrument names each
quadrant after its diagonal attribute).

### 4.3 `Coordinates` is the one type for survey, predicted and sampled points

- Projected responses, pooled single-response predictions (FR-5.9) and samples from a target are all
  `Coordinates`. Statistics, regions, KDE, shift, SPI, scenarios and plots take it.
- `xy` has shape (n, 2), with NaN rows aligned to the input responses, so a user can always join
  back.
- Labels are not stored in the user's native library; they are converted to it only at
  user-facing returns. Required label columns depend on `origin.kind`.
- `origin` is a sequence of segments, one at M1, so pooling later does not change stored shapes. Any
  future concat refuses a circumplex mismatch; no concat code is written yet.
- Every prediction shows `origin.kind = prediction` in every table and export and is never mixed
  silently with measured responses (FR-5.6). Points pooled from single-response predictions also
  carry one warning and a "predicted (single-response)" label in every table, plot and SPI result,
  because they are conditional means with less spread than real responses; the explanation docs
  state the bias (FR-5.9).
- The N < 30 rule (FR-3.2) applies to survey points and to predicted points of either kind; it does
  not apply to samples drawn from a target.
- No response weights; summaries come from `describe(coords, by=...)`, not from methods.

**Why.** This makes journeys A, C, D and E share one path instead of four. Storing coordinates as
columns of the response table was the alternative; it fails for sampled and pooled points and causes
the `pleasant` column collision. Storing labels in the native library would make every consumer
dual-backend code.

**Left to the spec.** Required labels per kind; origin payloads (sample at M2, prediction at M4; the
`kind` discriminator exists from M1).

### 4.4 `SurveyMapping` is the schema, and stored responses are raw

- A `SurveyMapping` says which user column is which construct and on what scale (FR-1.3) and is
  persisted as TOML (FR-1.3a). It is never called `Mapping`, which would shadow
  `collections.abc.Mapping`. It covers the FR-1.3 constructs at M1 (attributes, source dominance,
  overall quality, appropriateness, WHO-5) plus language and id columns, with construct groups and
  positions keyed unambiguously. Language values map to ISO 639-3 through an explicit coding, not by
  guessing at free-text names.
- `Coding` converts export values to scores in a fixed order: value map, offset, then reverse
  against the resolved scale. The order is part of the file's meaning. Coding lives in the mapping
  (owner-accepted call).
- **Stored responses are raw.** Coding is applied once, inside the import boundary helper, never to
  stored data. The project keeps the export byte-for-byte with its hash, plus the read options
  needed to re-read it identically (sheet, encoding, dtype overrides such as ids with leading
  zeros), and re-applies the deterministic mapping on open. A workbook with several sheets needs
  the sheet named.
- Quality rules are named and configurable and write to the per-response status (§4.5). Settings
  persist enabled rule names and thresholds, never callables.
- Mapping suggestion is exact and alias matching against known export schemas, returning a draft
  mapping plus unmatched columns. The user confirms or corrects it (FR-1.3a) by editing the draft
  in a notebook, by editing the TOML file, or through the CLI's interactive mode; all three end in
  the same saved file.
- Id roles are open beyond the SSID hierarchy (owner-accepted call) and are added when a loader
  first needs them. Hierarchy (FR-1.4) is id columns and joins: at M1 a dataset may carry optional
  location and session tables keyed by the mapping's id columns, joined on request, and tolerates
  any missing level; recordings join at M3.
- Part 3 statistics live in `survey`: item medians and dispersion (FR-2.2), correlations and
  group comparisons with effect sizes (FR-2.6, FR-2.6a), WHO-5 scoring and source-dominance
  profiles (FR-2.6c). Correlations against acoustic features take the features as a plain table
  joined by id, so `survey` never imports `audio`.

**Why.** Draft 0.1 proposed pandera. The checks are a few dozen lines against the mapping, and
FR-1.5 needs a per-response decision and reason, not a raised error. Storing a coded copy was draft
0.4's alternative; it makes two sources of truth and lets coding be applied twice.

**Left to the spec.** The field set; key normalisation; rule thresholds, checked against the ISD's
published exclusions; which export schemas ship at M1 (SSID and REDCap at least; Qualtrics shipped
or explicitly deferred).

### 4.5 One per-response status contract

- One status per response explains every absence: quality rules, missing items, language refusals,
  scale violations. It is `included`, `reasons` and `flags`, so "kept but flagged" is representable.
  Quality and projection write the same status.
- `response_id` has one stated origin: require a mapped column, generate at import and record, or
  derive from the stored raw export. The M1 spec chooses. An id never depends on the order of an
  in-memory frame, because such ids break on re-sort.
- Derived columns live in a namespace that cannot collide with canonical attribute names or user
  columns (`pleasant` is both an attribute and an axis). A collision raises.
- `n` and `n_valid` are defined once and used by `describe`, the N < 30 rule (FR-3.2) and SPI. N
  counts valid, included points.
- Missing attribute items exclude the response with a reason (FR-2.5a, owner decision). The
  renormalisation option is a Should; when it is built, every FR-2.5a constraint binds: canonical
  formulation only and never in an ISO profile; at least one present item with a non-zero
  normaliser on each axis, else exclude; each such response labelled with its number of present
  items; renormalised points left out of group statistics, densities and SPI unless the user opts
  in. The reason is scientific: an unbalanced subset of items breaks Σcos θ = 0 and changes the
  projected variance, and no source defines the rule.

**Why.** Draft 0.4's two columns could not express "kept but flagged", and quality and projection
status lived in two places. Fixing this after the first persisted results renames public columns.

**Left to the spec.** Column names; whether a scale violation raises or flags; the id mechanism; the
namespace convention.

### 4.6 Settings are recorded per operation

- Each operation has its own frozen pydantic settings model, living with the operation. Entry points
  take keyword arguments plus an optional `settings=` or `profile=` and resolve them into that
  model, which the result records. A result records only its own operation's settings.
- A profile (FR-2.9) is a named bundle of per-operation records saved as one TOML file, one section
  per operation. A low-level loader reads a profile into raw sections, and each operation validates
  its own section with its own model, so `profile=` works in the core without the core importing
  anything above it. A result always records the profile it started from (`derived_from`) and the
  fields that differ from it, if any, so editing a bandwidth never misstates what a projection used.
- Profiles are named faithfully to their sources, with the chapter profile as default, and the ISO
  profiles never use features their standard lacks (FR-2.9):

  | Profile | Formulation | Angles | Dispersion |
  |---|---|---|---|
  | ISO/TS 12913-3:2019 | 2019 fixed formula | equal | range |
  | ISO/TS 12913-3:2024 | the revision's general-angle formulas (A.1 to A.4) | equal | IQR |
  | Soundscape Across Languages (2026, default) | canonical | SATP v1.5 with tiers | IQR |

  Reading the 2024 revision's A.1 to A.4 as the canonical formulation is an interpretation, pending
  the owner's check of the text (PRD §10.6). Profile identifiers are the M1 spec's choice.
- Dispersion (range or IQR) belongs to item statistics (FR-2.2), not projection.
- **TOML rule.** An omitted key means the default. No field has a `None` distinct from its default,
  because TOML has no null and FR-11.3 round-trips would fail.

**Why.** Draft 0.4's single `Settings` root in the bottom layer would become a hub every milestone
edits, or break the layer contract. It made a projection record density and SPI settings, and turned
it into `profile=None` when an unrelated field changed. Stdlib dataclasses with hand validation were
the other alternative; pydantic is needed from the first result for FR-11.1, FR-11.3 and FR-14.2,
and switching later touches every module.

**Left to the spec.** Field lists; the profile identifiers; how the differing fields are reported.

### 4.7 Published parameters are versioned data, referenced structurally

- Angle sets (with tiers), targets, profiles and circumplexes ship as TOML under `_data/`. A
  released version is never edited or removed, so old results stay reproducible (FR-6.3).
- Persisted references are `{name, version}` plus the content hash in the result, never machine
  paths. User files are copied into the project and referenced project-relatively.
- One canonical hashing helper produces every hash, and the scheme carries a version, so a scheme
  change is detectable instead of silently refusing every shipped target or missing every cache.

**Why.** Draft 0.4's `name@version`-or-path strings let machine paths into shared results. A
universal data-reference type was the heavier alternative and is gold-plating.

**Left to the spec.** File layout per data kind; the hash format. Angle sets use the reference at
M1, targets at M2.

### 4.8 Every result carries provenance from the first result

The envelope is the operation's `settings` plus one extensible `provenance` mapping, built by one
helper and carried by every record, `Prediction` included. Provenance holds at least: package,
Python and wrapped-library versions; versions and hashes of shipped data used; created time (UTC);
the input-data hash, mapping hash and upstream run id (lineage); and the seed used where randomness
was involved. Data files carry a separate citation block: citation, DOI or URL, licence, date,
notes. Both can only gain keys. `show_versions()` comes free with the helper.

**Why.** FR-11 needs provenance from the first result. Lineage keys cost almost nothing now and
avoid a migration when GUI staleness logic arrives. Draft 0.4's separate `versions`, `methods` and
`created` fields could not grow without schema changes.

**Left to the spec.** Key names. Method versions exist only for baselined procedures (§8).

### 4.9 Geometry is computed in `distribution`; drawing is thin

- Density grids, contours, region proportions, shifts and Likert proportions are returned as data,
  usable without matplotlib (FR-8.4). Regions are value objects addressed by axis signs, so they
  work for any instrument.
- Plot functions take `ax=None`, return the `Axes`, create a figure only when `ax` is `None`, never
  call `show()`, and forward keyword arguments to the artist (FR-8.3). Breaking this later breaks
  every tutorial.
- Shared helpers apply prediction labels, survey-language labels and the N check, so plot authors
  cannot forget them; one test template checks them.
- N < 30 is decided at the drawing layer (owner): geometry warns, a single density plot refuses
  unless overridden, and the grouped view draws annotated scatter for small groups.
- One estimator, `scipy.stats.gaussian_kde`, with scipy `bw_method` semantics; the factor and the
  absolute bandwidth matrix are recorded, because the papers state no bandwidth and it moves the
  50% contour. Grid extent and normalisation are recorded settings, because the normalisation
  domain defines what "the contour enclosing 50%" means; matching a seaborn-era figure needs the
  same clip.
- One recorded rule assigns points exactly on a quadrant axis. Integer Likert scores at 45° put
  many points on the axes, so it moves every quadrant proportion. The M1 spec fixes it on its own
  merits before the first baseline; at M2 the KS tie rule is reconciled with it, or the
  procedure's method version is bumped.
- Styling uses matplotlib's own mechanisms with a colour-blind-safe default (FR-8.3).

**Why.** The 0.x `ISOPlot` hierarchy mixed KDE and SPI computation with drawing. seaborn's KDE wraps
the same scipy estimator, so the 2022 figures reproduce without seaborn.

**Left to the spec.** Plot and class names; the contour library; how the grouped view is
parameterised; the `bw_adjust` conversion (reproduction spec).

### 4.10 narwhals stays at the boundary

The package accepts any eager frame narwhals supports and rejects lazy frames with `TypeError`. One
helper converts inbound and outbound; numpy is used inside. Frames the package creates, and frames
read from paths, are pandas. polars is tested for frames passed in. The helper follows narwhals' own
conventions, reviewed while building (owner).

**Why.** Dataframe-agnosticism is an owner decision. Its cost stays low only if narwhals appears
where a frame enters or leaves. A feature that needs more converts to numpy.

**Left to the spec.** The pandas floor, settled once; blank-cell handling across backends. A
`backend=` argument waits for a user request (§8).

### 4.11 The project directory is the unit of exchange

A plain folder that the library, CLI and GUI all read (FR-1.7, FR-14.4). The minimal M1 directory
holds, by role:

- a project file with an integer format version;
- named datasets, each with its raw export, hash, read options and mapping; a project holds
  several surveys, because E1 at M2 needs two;
- default settings, as the user's input;
- result tables as parquet, each with an authoritative JSON sidecar, written by one save routine;
- a manifest of input hashes and versions.

Per-run settings in a sidecar take precedence over project defaults. Runs are keyed by name. Alphas
carry no format compatibility promise; the policy is set by the M2 alpha. Hand-made tables are CSV
or TOML; computed tables are parquet with a one-call export to CSV and Excel (FR-14.2; pyarrow
accepted as core by the owner).

**Why.** The contract for any GUI, in-process or out-of-process, is the library plus the project
directory. Draft 0.4's single-survey layout broke E1, and its coded copy broke the raw invariant
(§4.4).

**Left to the spec.** File names; default CSV copies from the CLI or an `export` command (choose
one, keep an Excel path). The full layout beyond M1 is direction of travel only.

### 4.12 Extension contracts are fixed in shape and grown when needed

- An entry point resolves to one light object per group: metadata readable without heavy imports,
  lazy loading (as xarray's `BackendEntrypoint` does), and a versioned contract. This is the only
  contract that crosses package boundaries.
- Plugin discovery ships at M2 with the first alpha (owner decision, PRD §9) as the smallest helper
  that lists a group, with the dataset-loader recipe. The groups for metrics and predictors are
  created with their first member (M3, M4). The names are reserved now.
- **Metrics.** The framework owns statistics, channel rules, naming and units. Implementations
  return per-channel series, cross-channel values or standard-defined scalars, and state the
  standard or method, its definition and citation, and the wrapped library with that library's own
  version and licence (FR-4.0, NFR-7). This boundary decides whether a metric recipe is 20 lines or
  200.
- **Predictors** (owner: interface now, model at M4). The outer shape is a spec readable without
  loading, a declared output kind that allows pleasantness-only models, and a `deployable` flag.
  Only deployable predictors are barred from perceptual inputs, because the 2021 "known location"
  model (FR-5.3a) uses survey data by design. Absent optional inputs are recorded, not warned about.
- Deep models are separate packages. Fetched weights are hash-pinned and never loaded in a way
  that executes code (NFR-5).

**Why.** Rule 8. Draft 0.4's hard perceptual-input validator contradicts FR-5.3a.

**Left to the spec.** The entry-object fields and how in-tree and discovered members merge (M2);
predictor payload types (M4, against two real predictors); metric attributes (M3, after two
wrappers); the collision rule for two backends computing one metric (M3).

### 4.13 One feature-name vocabulary from M2 onward

One vocabulary names feature columns. A name identifies the quantity as its standard or method
defines it, plus the statistic and the channel where they apply; the library that computed it goes
in provenance. A different method is a different quantity with a different name, so a column never
changes meaning when the backend changes. The M2 ISD loader maps the ISD's column names onto it,
M3 emits it, and M4 input contracts bind to it.

**Why.** Otherwise M4 binds to two vocabularies, or published course material is renamed.

**Left to the spec.** The syntax and name lists (M2 ISD loader spec, revisited after the first M3
wrappers); the units table.

### 4.14 Layering, errors and logging

- One ordered import-linter layers list plus a few protections (§5).
- `SoundscapyError` and `SoundscapyWarning` are the roots; leaves also subclass the matching
  built-in (`ValueError`, `ImportError`, `UserWarning`). One class per condition, added when code
  first raises it. No name shadows a standard library one.
- One package logger, `soundscapy`, with visible default output and a single silencing call
  (NFR-13). A download says what it fetches and under which licence, once (FR-10.1).

**Why.** Tests and a GUI catch a small typed vocabulary. Draft 0.4 listed about 19 classes ahead of
the code that raises them.

**Left to the spec.** Leaf names; the logging mechanism, settled at the first log line (the M1 ISD
loader).

## 5. Module map and layering

Single distribution, `src/` layout. Each subpackage is created with its first code.

```
workspace, cli            project directory, project files, click app     M1
plot                      drawing on matplotlib Axes                      M1
spi, design | audio | predict | datasets                       M2 | M3 | M4 | M1
survey, projection, distribution    mapping, import, quality, Part 3 statistics |
                          formulations | points, regions, describe, KDE, skew-normal, KS
instrument                Scale, Attribute, Circumplex, angle sets        M1
errors, provenance, _frames, _data, plugins    roots | envelope, hashing, citation |
                          narwhals helper | shipped data and profile loader | discovery (M2)
```

Read top to bottom: each layer may import only the layers below it. Within the middle layers,
`projection` imports `survey` and `distribution`. The allowed sideways imports are `design` to
`spi` (M2), `design` to `predict` (M4) and `predict` to `audio` (M4); `audio` and `predict` are
always imported lazily, inside the functions that need them. `datasets` keeps each loader's mapping
beside its module. `_data` holds the shipped TOML data, the bundled ISD subset and the low-level
profile loader of §4.6; each settings model lives with its operation, and `workspace` only reads
and writes project files.

**Phase 0 protections**, extended per milestone:

1. one ordered `layers` contract matching the diagram;
2. the core (everything below `plot`) never imports `plot`, `cli`, `workspace` or matplotlib;
3. pandas and polars are imported only in named IO boundary modules.

**Dependencies.** Core follows from the commitments and owner decisions: numpy, scipy, pydantic,
narwhals, pandas, pyarrow, matplotlib, pooch, click, a TOML writer and an Excel reader and writer
(FR-1.1, FR-9.1). The list lives in `pyproject.toml`. Each extra is created in the milestone of its
first module; the `phonometry` extra is an owner decision. tqdm arrives with the M3 batch runner
(owner decision #4). scipy.stats covers the correlation and group tests of FR-2.6 and FR-2.6a;
FR-2.6's regression and ANOVA are met through statsmodels, as a core dependency or a documented
how-to, pending PRD §10.6. seaborn is not a dependency. Tooling detail lives in the `house-style`
skill and `pyproject.toml`. Phase 0 exits with a clean install on Windows, macOS and Linux with no
compiler, followed by the quick-start (NFR-1).

## 6. Conventions

| Topic | Convention |
|---|---|
| API tiers | A small, curated narrative top level, plus complete module APIs as the reference. The M1 spec lists the names; they settle over the M2 course and freeze at the first non-alpha release. The top level imports M1 and M2 modules eagerly with an explicit `__all__`; `audio` and `predict` are imported explicitly. |
| Inputs | Keyword arguments first; optional `settings=` or `profile=`; resolved into the operation's recorded model (§4.6). |
| Randomness | Every random function takes `rng` exactly as SPEC 7 states; no global numpy state. If `None` is accepted, the seed used is recorded. SPI requires a seed (FR-6.1). |
| Frames | Any eager frame in, same library out; pandas where a path or nothing went in (§4.10). |
| Columns | Derived columns use the reserved namespace and never overwrite (§4.5); features use the shared vocabulary (§4.13). |
| Statistics tables | Tidy tables with the FR-2.6a fields (statistic, df or N, p, effect size), computed by scipy or a wrapped library, never re-implemented. |
| Errors and warnings | Roots plus one class per condition (§4.14). A warning fires once per operation, never per row, never for a deliberate choice: a legacy profile is recorded, not warned about. |
| Overrides | An explicit keyword argument, recorded in the result. |
| Logging | Through the `soundscapy` logger only; the package never prints. |
| Plots | The `ax` contract (§4.9); a label language defaulting to the instrument's reference language (NFR-10). |
| Optional dependencies | Imported inside the function that needs them; a missing extra raises one error naming it at call time, never at import. |
| CLI | One library call per command, no logic in the CLI, a non-interactive mode always (FR-14.1). Commands may read inputs from outside the project (an export) and write into it. |
| Deprecation | Alphas may break anything; the changelog says what. The FR-13.4 policy, written by M2, governs releases from 1.0 final. A baselined procedure that drifts beyond tolerance bumps its method version with a "Scientific changes" entry. |
| Tests | Unit and property tests (hypothesis for invariants), literature fixtures, cross-implementation references and golden pipelines, per house style. Tolerances justified by the reference's significant figures (NFR-4). Plot tests assert on geometry and artist data. |
| Docs | numpy docstrings with a `References` section on every procedure (FR-12.4); great-docs; Quarto tutorials, per house style. Paper reproductions need the full ISD, so they live apart from the offline tutorials, are frozen, and are re-executed nightly or on demand rather than by every `task docs`. |
| ADRs | Written after the first M1 vertical slice, from code that exercised the decision. |

## 7. Milestone direction

Each milestone adds modules. Nothing here fixes fields, flags or library versions.

| Milestone | Adds | Must decide in its spec | Owner inputs | Exit |
|---|---|---|---|---|
| Phase 0. Toolchain | The toolchain per `house-style`; the §5 protections; test markers; `GOVERNANCE.md`, `CITATION.cff`, `CONTRIBUTING.md`; the docs build with tutorials and paper reproductions separated (§6) | – | – | `task check` and `task docs` green on an empty package; a clean no-compiler install on three OS (NFR-1) |
| M1. Survey analysis (A; I for the 2022 paper) | `errors`, `provenance`, `_frames`, `_data`, `instrument`, `survey` (mapping, import, quality, Part 3 statistics), `projection`, `distribution` (points, regions, describe, KDE geometry), `plot` basics, the ISD loader and bundled subset, a minimal `workspace`, journey A's CLI, the canonical layout as a CSV template (FR-1.2), the install quick-start, tutorial A, the 2022 reproduction | *Mechanisms for what §4 fixes:* the `response_id` mechanism, status and namespace column names, the hashing format, the profile identifiers, reference and file names. *Open for the spec:* whether a scale violation raises or flags, the region-boundary rule, KDE grid extent and normalisation, the top-level names, required survey labels, quality defaults, the ISD record version, the default CSV copy versus an `export` command | The Aletta 2024 worked example; ISD languages outside SATP v1.5 and Aletta 2024, and the Medium-tier list; ISD record versions; the 2022 KDE settings and angle choice; ISD exclusion counts | Journey A done; the 2022 `.qmd` renders with every cell run, following the paper's methodology; no numeric baseline for that paper (owner decision) |
| M2. SPI and design, first alpha (C, E1, K; I for the 2024 paper) | The skew-normal and the KS distance; `spi` with a versioned targets library; `design` with briefs, scenario records and evaluation; the ISD feature table on the shared vocabulary; tutorials C and E; course acceptance cases; the 2024 reproduction with the SPI acceptance test last; the versioning policy; plugin discovery with the plot-type and dataset recipes; the C and E1 CLI (FR-14.5; narrowing proposed in §9.2) | Circumplex identity for scoring and the fingerprint; skew-normal and fit types; SPI settings and the multi-seed helper; ranking table shape; target fields from fixtures; format compatibility; feature syntax and name lists; scenario and KPI fields | Example 6 data and target; the 2024 targets and sample size; the KS variant, direction combination, tie rule and truncation; one R `sn` run | The 2024 tables reproduce within tolerance; Example 6 runs; course exercises run unchanged |
| M3. Audio (B) | `audio` with windowed reads, calibration applied once and recorded, built-in level metrics and IACC; MoSQITo as the default psychoacoustic backend, scikit-maad for ecoacoustic indices, phonometry as an opt-in extra, no backend that depends on PyQt6; a batch runner whose per-file cache is the resume mechanism; the metrics plugin group; journey B's CLI; the metric recipe; tutorial B; the ArtemiS comparison page | Signal and calibration types from real recorder files (at most two channels); the audio reader; metric attributes after two wrappers; the backend collision rule (MoSQITo and phonometry both offer ISO 532-1 and DIN 45692); the runner and its executor; progress, cancellation and whether partial results are needed; the backend roster, re-running the research note; Windows cache paths; audio formats; equalisation filter forms | Redistributable audio; which tonality each Table D.1 edition names | Journey B done; ISO 532-1 reference signals pass; a large batch resumes |
| M4. Prediction (D, E2; I for the 2021 paper) | `predict` with the predictor contract as code; the predictors plugin group; the default model chosen at M4 planning; a linear predictor for the 2021 reproduction; the SoundAQnet plugin package; E2 and its CLI; the predictor recipe | Payload types against two real predictors; scenario inputs; the default model; weight fetching | The features the deployable 2021 models use | Journeys D and E2 done |
| M5. Ecosystem (L, F, G) | SATP, ARAUS and DeLTA loaders; MOSAIQ import and export; target search behind an `optim` extra, returning a target with provenance; the contributor trial of each recipe | Participant and stimulus roles; the MOSAIQ boundary | – | Journeys F, G and L done |
| GUI | To be designed (FR-14.7) | Its planning triggers the stateful project class, staleness logic, published schemas, and the in-process or out-of-process question | – | – |

## 8. Deferred decisions

| Decision | Trigger | Already binding |
|---|---|---|
| Persisted circumplex fingerprint [G2-03] | M2 spec, with target files | Identity by content; canonical hash |
| Scale in circumplex identity [G2-04] | FR-6.3 clarification, before M2 | Instrument and effective scale recorded separately |
| KDE grid extent and normalisation [G4-07] | M1 spec at the latest | Recorded settings; the normalisation domain defines the meaning of the fraction contour; a seaborn match needs the same clip; 2022 settings if supplied |
| Participant and stimulus roles [G3-05] | FR-2.4b or M5 loaders | Id roles are open |
| Group, session, recordings tables [G3-06] | Groups at M2; others at M3 | Keyed by id column names |
| Renormalise, ipsatise, cluster [G3-17] | When each Should is scheduled; FR-2.4b at M5 | Renormalisation: every FR-2.5a constraint (§4.5) |
| Loader `describe()` with MOSAIQ fields [G3-19] | MOSAIQ stable, or M5 | Plain citation metadata at M1 under another name |
| `backend=` argument, polars extra [G3-09] | A user request or the M2 course | pandas from paths |
| Origin payloads [G4-04] | Sample M2; prediction M4 | `kind` discriminator from M1 |
| `Coordinates` concat [G4-03] | Journey G, pooled E1 or the course | `origin` a sequence; refuse mismatch |
| Skew-normal and fit layout [G4-10] | M2 spec, after `sn` fixtures | Direct parameters in `sn` terms; a frozen value object; positive-definiteness validated; inadmissible centred parameters refused with a specific error; fitting done in centred parameters, because direct-parameter MLE runs to the alpha boundary (FR-3.4). Validated against the owner's fresh R `sn` run; the 0.x `dp2cp` port is at most a starting point, recorded by an ADR because AGENTS.md says nothing carries over (research note §2) |
| SPI settings variants [G4-12] | Owner confirms the paper's method | Implement the paper's combination only. The result records the KS variant, how the two directions combine, the tie rule, whether target samples are truncated to the square, the seed, the sample size (paper default) and the test-sample N, because each moves D (FR-6.1). Working hypothesis to test first: Fasano–Franceschini with D = (D1 + D2)/2 |
| Multi-seed SPI [G4-13] | M2 spec | An agreed Should; outside core result fields |
| Target search [G5-02] | M5 journey F | Returns a target with provenance |
| Entry-object fields and merge rule [G5-06] | M2 spec | §4.12 shape; discovery at M2 (owner) |
| Metrics and predictors groups [G5-07] | First member: M3, M4 | Names reserved |
| Signal, read, calibration [G6-01] | M3 spec, real recorder files | Calibrate once; two channels; windowed reads |
| Metric registry by backend [G6-03] | M3 spec | A collision rule; no selector before it |
| Batch runner [G6-05] | M3 spec, measured runtimes | Cache is the resume; top-level work functions |
| Time-series cache [G6-06] | FR-4.11 scheduled | Series in memory |
| Progress reporting [G6-09] | M3 batch runner | `tqdm.auto` default (owner); work-unit label, not a path |
| Cancellation, partial results [G6-10] | M3 runner spec | FR-14.3; downloads per §9.2 #4 |
| Method versions [G7-03] | First beyond-tolerance drift of a baselined procedure | Baselined procedures only; profiles unversioned |
| Readers per record [G7-06] | Results read back (M3 or GUI) | One generic serialisation path |
| Staleness over lineage [G7-07] | GUI planning | Lineage keys from M1 |
| Stateful project class [G7-15] | GUI design or M3 | Path-helper module |
| Published JSON schemas [G7-17] | GUI work, fields stable past M3 | pydantic generates them |
| Out-of-process GUI [G7-18] | GUI planning | Library plus project directory |
| M2 to M4 CLI names and flags [G7-20] | Each milestone spec | One library call per command |
| Extras [G8-03] | Each extra's first module | `phonometry` fixed |
| statsmodels [G8-05] | First feature needing it | Docs group otherwise |
| Recipe texts [G9-08] | Plot, dataset M2; metric M3; predictor M4 | Appendix B; concept and file counts per recipe |
| Freezing top-level names [G1-03] | First non-alpha release | A small curated set |
| PRD §10.6 items for this plan: mixed-language plot labels; timestamps and time zones; audio formats; Windows cache paths | First multi-language figure; M3 sessions, recordings and spec | Labels by ISO 639-3; windowed reads |

## 9. Owner decisions

### 9.1 Register

Draft 0.4's §15 (numbers kept) plus PRD §10. This is the one register; PRD §10 should point here,
and rows become ADR links as ADRs are written. Rows marked *changed* or *later* are implemented
differently or later in 0.5, and need the owner's confirmation.

| # | Decision | In 0.5 |
|---|---|---|
| 1 | Copyleft acceptable; licence recorded per backend, not a filter (NFR-7) | §4.12; research note |
| 2 | Dataframe-agnostic via narwhals at the boundary; pandas and polars tested | §4.10 |
| 3 | click CLI, documented by great-docs | §5, §6 |
| 4 | Progress visible in notebooks via a default `tqdm.auto` reporter | *Later*: built with the M3 runner; pooch's bar at M1 |
| 5 | Python 3.11 floor; phonometry its own 3.13+ extra, never an environment marker | §5, §7 |
| 6 | pyarrow accepted; CSV hand-made, parquet computed, CSV and Excel export | §4.11 |
| 7 | TOML for settings and mappings; JSON for machine records | §4.6, §4.11 |
| 8 | No SPI uncertainty machinery; paper's sample size default; seed required | §6, §7 |
| 9 | Missing items exclude by default; renormalisation canonical-only | §4.5 |
| 10 | 2022 paper by methodology, no numeric baseline; same KDE as seaborn | §4.9, §7; 0.4's golden table dropped |
| 11 | Governance: Andrew Mitchell cuts releases and reviews scientific changes (NFR-12) | §7 Phase 0 |
| 12 | Profiles named faithfully; chapter profile default | §4.6 |
| 13 | SPI acceptance from 2024 tables; Example 6 a tutorial; method recorded; optional multi-seed | §7, §8 |
| 14 | Renormalisation a constrained Should, refused in ISO profiles | §4.5, §8 |
| 15 | Three-way override for uncovered and Low-tier languages; one status for every exclusion | §4.2, §4.5 |
| 16 | `SurveyMapping`; scales per construct group; coding order; string keys | §4.4 |
| 17 | Circumplex identity by content; versioned targets and angle sets, never removed; keyed by circumplex | §4.2, §4.7; fingerprint *later* (M2) |
| 18 | `Coordinates` rows aligned, mandatory labels, one origin, `describe(by=)`, no weights | §4.3; origin a sequence (*changed*, same intent) |
| 19 | N < 30 at the drawing layer, scatter fallback | §4.9 |
| 20 | A `Settings` model; a profile is a preset | §4.6; per-operation records and bundles replace the root (*changed*) |
| 21 | narwhals boundary per narwhals' conventions | §4.10 |
| 22 | M1 re-scoped; skew-normal, KS, recipes, most CLI and plugins at M2 | §7, §4.12 |
| 23 | Table D.1 not a gate; MoSQITo the single `audio` backend; no PyQt6; phonometry extra; substitutions named | §7; research note |
| 24 | Predicted single-response distributions warned and labelled | §4.3, §4.9 |
| 25 | Logging with a visible default; never prints; downloads announced | §4.14 |
| 26 | Warnings once per operation; scoped filterwarnings; a cancellation error | §4.14, house style; the cancellation error *later* (M3) |
| 27 | Optional imports at call time; eager top level; `audio`, `predict` explicit | §6 |
| 28 | Raw export byte-for-byte; schema sidecar; runs by name; CLI CSV by default | §4.4, §4.11; the read options persisted beside the raw export replace the coded copy and sidecar (*changed*, same intent); CSV default or `export` chosen in M1 spec |
| 29 | `Signal` is a window; windowed I/O; hashes in the batch run | §7, §8 |
| 30 | `evaluate` takes scenario records; specs discoverable unloaded; FR-5.5 validator; `MetricImplementation`; fit result; centred fitting | §4.12, §7, §8; scenario records carry the Annex A level, description and `produced_by` into the evaluation tables; a `deployable` flag replaces the validator (*changed*) |
| 31 | No overlap statistic; shift to a target by closed-form mean | §7 |
| 32 | Predictor interface now; model at M4 | §4.12 |
| 33 | Releases from M2 are 1.0 alphas; nothing frozen at M2 | §1, §6 |
| 34 | Skew-normal fitting a Must at M2 | §7 |
| a | Builder call: coding lives in the mapping | §4.4 |
| b | Builder call: id roles open beyond the SSID hierarchy | §4.4 |

PRD §10 decisions that bind the architecture, not already above:

- **§10 (0.1 to 0.2):** canonical robust projection; generalised circumplex; two shipped angle-set
  versions, SATP v1.5 default, tiers enforced; Methods B and C out; FR-2.8 and FR-5.7 Later; local
  notebooks only; English-only interface; user-supplied equalisation filters; design quality over
  performance (NFR-3); MOSAIQ normative once stable.
- **§10.1:** Quarto tutorials; standards by public edition; prediction stands alone and single
  responses pool into distributions; E split into E1 and E2; minimal project directory at M1;
  per-response angles with an overridable error; tolerances by significant figures; ArtemiS
  comparisons indicative; metric definitions from source libraries; `sn` parameterisation and
  terminology; no GUI design fixed; no reference to the old package's layouts; 2021 paper from the
  ISD features. (The linear models as fallback default predictor was superseded by #32.)
- **§10.5:** tutorials A, C, E at M2 and B at M3; non-interactive CLI always; recipes per milestone;
  default backends never depend on the Python version; targets and angle sets versioned with old
  versions kept.

PRD §10.3 inputs are listed per milestone in §7. PRD §10.6 findings are applied as answered; those
touching this plan are in §8.

### 9.2 Proposed PRD changes awaiting the owner

Until the owner accepts one, the PRD binds and the plan follows it.

| # | PRD item | Proposed change | Why |
|---|---|---|---|
| 1 | FR-6.3 | State whether the scale is part of circumplex identity for scoring, or identity is positions, angles and axes with the scale recorded. Before M2. | Otherwise a 1..7 survey can never be scored against a 1..5 target (FR-6.4). |
| 2 | FR-2.3 | Aletta 2024 formulation conditional on a worked example added to §10.3; Should if none by the M1 spec. | Untestable without one. The angle set stays an M1 Must. |
| 3 | FR-14.5 | M2 CLI narrowed to E1 `evaluate` with a brief file, plus scoring against shipped or bespoke targets. Target fitting and editing wait for GUI planning. | The course runs in notebooks; C's fitting commands have no M2 user. |
| 4 | FR-14.3 | Downloads report progress through pooch and are not cancellable; recorded as an exception. | pooch's `downloader=` hook makes this a later fix if a GUI needs it. |
| 5 | FR-1.5 | Untestable thresholds become "defaults that reproduce the ISD's published processing"; ISD exclusion counts added to §10.3. | Testable against published numbers. |
| 6 | FR-2.4 | Per-language version choice scoped to registered sets by name and version, no paths. | Portable references (§4.7). |

Considered and not changed: FR-14.2's columnar format, FR-6.1's multi-seed Should, the FR-1.3
construct list. The review also argued for plugin discovery at M4 and renormalisation as Later;
both contradict decisions the owner took on 2026-10-30 (PRD §9, FR-2.5a), so the plan follows
those decisions and does not re-propose them.

## Appendix A. M1 type sketches (non-normative)

*Illustrative, the M1 spec decides.* Names, fields and defaults may all change, except where a
source is cited.

```python
class Scale(BaseModel, frozen=True):          # e.g. 1..5, ISO/TS 12913-2 Table A.1
    minimum: float; maximum: float; kind: Literal["ordinal", "interval"]

class Attribute(BaseModel, frozen=True):
    position: str; angle_deg: float; labels: Mapping[str, str]   # {"eng": "pleasant"}

class Circumplex(BaseModel, frozen=True):
    name: str; attributes: tuple[Attribute, ...]; axes: tuple[str, str]; scale: Scale
    reference_language: str = "eng"           # default plot labels and entry-point aliases (NFR-10)

class AngleSetRef(BaseModel, frozen=True):    # persisted; the result also records the hash
    name: str; version: str                   # ("satp", "1.5")

class AngleSelection(BaseModel, frozen=True):
    default: AngleSetRef | Literal["equal"]
    per_language: Mapping[str, AngleSetRef] = {}                       # ISO 639-3 keys
    on_uncovered: Literal["error", "exclude", "equal"] = "error"       # FR-2.4
    on_low_confidence: Literal["error", "exclude", "equal"] = "error"  # FR-2.4a

@dataclass(frozen=True, eq=False)
class Coordinates:
    xy: np.ndarray                       # (n, 2), read-only; NaN rows for excluded responses
    circumplex: Circumplex
    labels: nw.DataFrame                 # response_id, language, angle_set, status, ids
    origin: tuple[OriginSegment, ...]    # one segment at M1, kind="survey"

class ProjectionSettings(BaseModel, frozen=True):
    formulation: Literal["canonical", "iso2019", "aletta2024"]         # FR-2.3
    angles: AngleSelection

class DensitySettings(BaseModel, frozen=True):
    bandwidth: float | Literal["scott", "silverman"] = "scott"         # scipy bw_method
    fraction: float = 0.5                # the 50th-percentile contour, FR-3.1
    min_n: int = 30                      # FR-3.2

@dataclass(frozen=True)
class ProjectionResult:
    coordinates: Coordinates
    settings: ProjectionSettings
    provenance: Mapping[str, Any]        # versions, data refs and hashes, created,
                                         # input_hash, mapping_hash, upstream_run, seed
```

Status sketch: `included` (bool), `reasons` and `flags` (tuples of codes). Candidate reasons from
draft 0.4: `scale_violation`, `all_identical`, `excessive_missing`, `under_age`, `no_consent`,
`test_response`, `incomplete_attributes`, `uncovered_language`, `low_confidence_language`. Candidate
flag: `paper_input`. Thresholds come from the ISD's published processing.

M1 CLI sketch: `import`, `validate`, `project`, `describe`, `plot`, and `export` if the spec chooses
an export command over default CSV copies.

## Appendix B. Extension recipe outlines (non-normative)

*Illustrative, the milestone spec decides.* Each recipe records its concept and file counts, so the
M5 contributor trial can test the one-day goal.

| Recipe | Milestone | Outline |
|---|---|---|
| Plot type | M2 | A `plot` function over `distribution` geometry, honouring the `ax` contract and the shared label helpers. Test on artist data; docs from the docstring. |
| Dataset loader | M2 | A `datasets` module with its mapping beside it, fetched with pooch from the public record at a pinned version, with hash and citation block. Test on a bundled fixture. |
| Metric | M3 | A `MetricImplementation` returning series, cross-channel values or standard scalars; the framework adds statistics, channel rules, names and units. In-tree first; an entry point once the group exists. Test on a reference signal with a cited tolerance. |
| Predictor | M4 | A predictor exposing its spec unloaded, its output kind and its `deployable` flag, under the predictors group. Test the input contract and the output label. |

## Appendix C. Glossary (non-normative)

| Term | Meaning here |
|---|---|
| Position | An attribute's stable id in a circumplex (PAQ1 to PAQn, counter-clockwise from +x). |
| Instrument | A circumplex: attributes with angles and labels, two named axes, a default scale. Identity by content. |
| Effective scale | The scale a survey actually used, recorded beside the instrument. |
| Angle set | Versioned per-language angles with a confidence tier per language. |
| Formulation | The projection formula: canonical, ISO 12913-3:2019 or Aletta 2024. |
| Profile | A named bundle of per-operation settings records in one TOML file. |
| Origin | What produced a set of points: segments, each a survey, a prediction or a sample. |
| Status | The per-response `included`, `reasons` and `flags` that explain every absence. |
| Run | One named execution of an operation in a project: a table plus its JSON sidecar. |
| Provenance | The extensible mapping on every result: versions, data references and hashes, time, lineage, seed. |
| Citation block | Citation, DOI or URL, licence, date and notes on a shipped or fetched data file. |
