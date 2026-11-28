# Alternative architecture for soundscapy 1.0 (blind draft)

Date: 2026-11-27. Status: research note, not a decision. Written from `design/PRD.md`
(draft 0.9), `AGENTS.md` and `design/literature/` only. No other design document or code
was read. Where the PRD records an implementation choice "in the plan" (§10.2, §10.5), it
is treated here as one option among several.

---

## 1. Reading of the PRD

### 1.1 The forces that drive the architecture

1. **One part-time, unfunded maintainer, and new contributors as a primary persona (P6,
   NFR-9, journey L).** This outweighs everything else. Each layer of abstraction is a
   cost paid on every contribution. The package should be a set of plain functions over
   small typed value objects, built on libraries contributors already know (numpy,
   pandas, scipy, matplotlib, pydantic). Anything bespoke must justify itself against
   "a volunteer adds a metric in under a day".

2. **Provenance on every number (FR-11.1, FR-2.3, FR-6.3, FR-5.6, NFR-4, NFR-6).** The
   product is not "a projection" but "a projection, plus the formulation, angle-set
   version, scale, language rule and package version that produced it". Results must
   be values that carry a record with them, not bare DataFrames. That record must
   survive saving to disk (FR-14.2) and drive warnings and labels later (FR-5.9).

3. **A circumplex that is data, not code (FR-2.0, FR-2.4, FR-6.3).** Attributes,
   angles, labels, axes and scale are all configuration. Angles vary per response
   (language) and are versioned. Coordinates live in an "axis-pair space" identified
   by content. No module may hard-code eight English words or the column names
   `ISOPleasant`/`ISOEventful`.

4. **Published numbers are the acceptance tests (journeys C and I, FR-1.5, FR-3.4,
   FR-6.1, FR-13.2, NFR-4).** The 56 exclusions, the SPI tables, the `sn` fixtures and
   the 2022 KDE behaviour are what "done" means. The test harness for literature
   fixtures and reference values is load-bearing infrastructure, built in the first
   slice, not added at the end.

5. **Headless now, GUI later (FR-8.4, FR-14.1 to FR-14.7, NFR-13).** All of these must
   hold from the start, because retrofitting any of them is a rewrite: every operation
   is a function of (inputs, settings) with no display or interaction; settings are
   schema'd and serialisable; long operations take a progress callback and a cancel
   token; plot geometry is pure data; errors are structured.

6. **Windows-first, wheels-only installs for non-programmers (NFR-1, NFR-2, FR-12.1,
   journey K).** No R, no compiler, nothing that needs a system library. Multiprocessing
   must work under `spawn`. Heavy features (torch, pymoo) stay out of the core install.
   `import soundscapy` must be fast and must not import matplotlib or audio libraries.

7. **Two kinds of truth: item-level responses vs derived values (FR-1.5, FR-2.5a,
   FR-5.6, FR-1.8).** Raw item data is never mutated or dropped. Exclusions, coordinates
   and predictions are separate layers keyed by `response_id`, and predicted values are
   never mixed in with measured ones.

8. **Audio is a different world (journey B, FR-4.x).** It is batch, file-oriented and
   long-running. It joins the survey side only through identifiers and a tidy table.
   It should share provenance and project conventions but no internals with the survey
   side.

### 1.2 Where the PRD over-constrains

- **FR-14.2 "a text form for every result".** CSV export covers tables. For object
  results (a fitted skew-normal, an SPI score) the text form is the TOML/JSON
  provenance record. That should be stated as sufficient, or every new result type
  becomes a serialisation chore.
- **FR-1.3a for a non-programmer at M1.** The real non-programmer interaction is GUI
  work. At M1, "suggest, confirm in a TOML file, reuse" is enough. Qualtrics schema
  detection is a nice-to-have, and only the SSID/REDCap codebook should be a Must.
- **FR-2.5a renormalisation (a constrained Should)** carries six sub-rules for a
  feature no source defines. It earns a design slot, not M1 effort.
- **FR-3.2's "single density plot refuses"** is a drawing-layer policy described as a
  science rule. Geometry should compute and flag; plot functions decide. The PRD's
  §10.5 already says this. The FR text should match.
- **Journey I's 2022 reproduction "renders with every cell run" in `task docs`.** This
  forces a full ISD v0.2.3 download into every docs build. Reproductions should run on
  a separate nightly or release job (§10.6 already hints at this).
- **FR-6.4 and FR-7.3 rankings, FR-7.5 report content.** These are tables. The PRD
  should say a ranking table is enough, so nobody builds a report model.

### 1.3 Where the PRD under-specifies

- **When the canonical formula equals the 2019 one.** They are identical only when the
  angles are the standard 45° set and the scale is 1..5. 2019's divisor (4 + √32) is
  ρΣ|cos θ_i| with ρ = 2. Centring drops out only because Σcos θ_i = 0. On a 1..7 scale
  the 2019 formula is not bounded by [−1, 1]. The test suite must state this condition
  (property test: equal angles and 1..5 scale gives equality to 1e-12).
- **Column naming of coordinates** for non-default circumplexes, and how coordinates
  are joined back to items. Proposed: columns are the circumplex's declared axis ids,
  the key is `response_id`.
- **Identity of an "axis pair by content" (FR-6.3).** What is hashed is not defined.
  Proposed in §2.2.
- **Missingness codes** for non-circumplex items, time zones, and accepted audio
  formats (the §10.6 findings). These need defaults at M1 (proposed: explicit
  `missing_values` in the mapping, timestamps stored timezone-aware with a declared
  zone, audio = whatever libsndfile reads).
- **Precedence between settings layers** (profile vs explicit override vs project
  file). Proposed in §2.4.
- **Structured errors for a GUI.** Not specified. Proposed in §2.10.
- **Settings for the 2-D KS / SPI statistic.** These wait on owner input (§10.3). The
  API must make variant, direction combination, tie rule and truncation explicit
  parameters from day one, so the answer is a default change, not an API change.

---

## 2. Architecture

### 2.1 Package layout and dependency direction

```
src/soundscapy/
  __init__.py          curated public API, lazy attribute loading (PEP 562)
  errors.py            exception and warning hierarchy (no deps)
  provenance.py        Provenance, Result, ResultTable, fingerprint(), method ids
  settings/            pydantic models, profiles, TOML io, JSON-schema export
  instrument/          Scale, Circumplex, AxisPair, AngleSet, LanguageTier, registries
  survey/              read_survey, Mapping (+ suggest), coding (Table A.1),
                       quality rules, StatusLog, Survey
  circumplex/          projection kernels (pure numpy), project(), item and
                       coordinate statistics, regions ("percent likely")
  stats/               correlations, group comparisons, WHO-5, source dominance
  distributions/       Empirical2D, SkewNormal2D (dp/cp, pdf, sample, fit), kde
  spi/                 ks2d, spi(), spi_table(), Target, target library
  design/              DesignBrief, Scenario, evaluate()
  geometry/            DensityGrid, ContourSet, ScatterLayer, RegionTable (pure data)
  plot/                matplotlib drawing of geometry objects only
  audio/               index, read, Calibration, windows, channel rules, run_batch
  metrics/             MetricSpec protocol + adapters: levels, mosqito, maad
  predict/             Predictor protocol, PredictorSpec, Prediction, linear2021
  datasets/            isd (pooch), bundled subset, dataset protocol
  project/             directory layout, save/load results, registries copied in
  plugins.py           entry-point discovery
  cli/                 click commands (thin)
  data/                shipped TOML (circumplexes, angle sets, targets, profiles),
                       ISD teaching subset (parquet)
```

Dependency direction, from lowest to highest. Each layer may import only layers below
it. CI enforces this with `import-linter` contracts.

```
errors → provenance → settings → instrument
  → survey → circumplex → stats
  → distributions → spi → design
  → geometry → plot
  audio → metrics            (imports only errors/provenance/settings)
  predict                    (instrument, audio, metrics, distributions)
  datasets, project, plugins → cli
```

`plot` imports `geometry` but never `spi` internals. `geometry` never imports
matplotlib. `audio` and `survey` never import each other: they meet only in a join on
identifiers.

*Rejected:* a layout by persona or journey (`journeys/a.py`). Journeys share most code,
so it would duplicate it. A single flat module per FR area without a direction rule: the
rule is what keeps contribution local (NFR-9).

### 2.2 Core data model

All value objects are frozen. Configuration objects are pydantic v2 models; data
carriers are frozen dataclasses holding numpy arrays or pandas frames.

**Instrument (FR-2.0, FR-2.1, FR-2.4, FR-2.4a)**

- `Scale(min: int, max: int, direction: Literal["ascending"], labels: dict[int, str])`,
  with derived `midpoint` μ and `half_range` ρ. Table A.1 coding lives in the mapping
  (raw label → code). The scale only validates codes.
- `Circumplex(id, version, positions: tuple[str, ...]` (`PAQ1…PAQn`, counter-clockwise
  per the SATP convention), `nominal_angles: tuple[float, ...]`,
  `axes: AxisPair, labels: dict[lang, dict[position, str]], default_scale: Scale)`.
- `AxisPair(x_id, y_id, x_label, y_label, family)`. `fingerprint()` hashes the
  canonical JSON of (`family`, `x_id`, `y_id`, the sorted position ids). This is
  "identity by content" (FR-6.3). The outdoor ISO circumplex and the Torresin indoor
  one differ by family and positions, so targets refuse across them unless overridden.
- `AngleSet(name, version, circumplex_fingerprint, entries: dict[lang, AngleEntry])`
  with `AngleEntry(angles: tuple[float, ...], tier: Literal["high","medium","low"],
  source)`. These are shipped as `data/angles/satp-1.5.toml` and
  `data/angles/aletta-2024.toml`. Keying by circumplex fingerprint answers the §10.6
  "angle sets keyed by circumplex" finding.
- Registries are tiny functions over `importlib.resources`:
  `angle_sets.get(name, version)`, `angle_sets.list()`, `targets.get(...)`. User sets
  are validated and copied into the project's `registry/` (FR-2.4 registration); a
  result references them by name, version and fingerprint, never by path.

**Survey (FR-1.3 to FR-1.5)**

- `Survey(data: pd.DataFrame, mapping: Mapping, circumplex: Circumplex, scale: Scale,
  language: LanguageDecl, status: StatusLog)`.
  - `data` is item-level, one row per response, canonical column names, never mutated.
    Hierarchy columns are `location_id`, `session_id`, `group_id`, `response_id`, all
    nullable except `response_id` (FR-1.4 tolerates partial hierarchies).
  - `LanguageDecl` is either `fixed: str` or `column: str`.
  - `StatusLog` is an append-only table (`response_id, stage, reason, detail`), where
    the stage is `quality`, `projection`, ... and the reason is drawn from a closed
    set: `all_identical`, `missing_item`, `uncovered_language`, `low_tier`, ...
    `survey.status_table()` derives the single `included / reason` column that FR-1.5
    demands (first exclusion wins, all reasons kept). One column for GUIs, the full
    log for auditing.
- *Rejected:* storing metadata in `DataFrame.attrs`. It is silently dropped by many
  pandas operations, which defeats provenance. Also rejected: a subclassed DataFrame,
  which is fragile and confuses contributors.

**Projection (FR-2.3, FR-2.5, FR-2.5a, FR-2.9)**

- Kernel: `project_array(scores: (n,k), angles: (n,k), scale) -> (n,2)`, pure numpy with
  per-row normalisers. Per-response angles (FR-2.4) are just a gathered `(n,k)` array.
  There are three kernels: `canonical`, `iso2019`, `aletta2024`.
- `Projection(coords: pd.DataFrame, axes: AxisPair, provenance: Provenance)`. The
  columns are `response_id`, `<x_id>`, `<y_id>`, `angle_set`, `language`, `tier` and,
  if renormalised, `n_items`. Excluded responses are absent from `coords` and present
  in the survey's `StatusLog`. `Projection.survey` keeps a reference back for joins
  and grouping.

**Distributions and targets (FR-3.1, FR-3.4, FR-5.9, FR-6.2, FR-6.3)**

- `Distribution` protocol: `axes`, `mean() -> (2,)`, `sample(n, rng) -> (n,2)`, `kind`
  (`observed | predicted | parametric`), `n` (None for parametric).
- `Empirical2D(points, axes, kind, source: Provenance)` comes from
  `projection.distribution(where=...)` or from a set of single-response predictions
  (FR-5.9). If `kind == "predicted"` and the predictor was single-response, a
  `PredictedSpreadWarning` is attached.
- `SkewNormal2D(xi, Omega, alpha, axes)` uses Azzalini's direct parameters. It offers
  `.to_centred()`, `.from_centred()` (raises `InadmissibleCPError`), `.pdf`, `.sample`
  (stochastic representation, seeded), `.mean()` (closed form, for FR-3.3), and
  `SkewNormal2D.fit(points)` (MLE in CP for numerical stability, as Azzalini
  recommends; scipy.optimize).
- `Target(name, version, distribution: SkewNormal2D, status:
  Literal["demonstration","reference","bespoke"], citation)`, shipped as
  `data/targets/*.toml`. Superseded versions stay as separate files and are never
  deleted (FR-6.3).

**Results with provenance (FR-11.1, FR-14.2)**

- `Provenance` (pydantic): `operation`, `method_id`, `method_version` (an integer bumped
  when outputs move beyond tolerance, NFR-4), `settings` (the fully resolved model
  dump), `profile` (name, if one was used), `inputs` (fingerprints), `assets` (angle-set
  or target name, version and fingerprint), `versions` (soundscapy plus each wrapped
  library), `warnings` (class, message, context), `created`.
- `ResultTable(df, provenance)` is any tabular result. `.to_parquet(dir)` writes
  `name.parquet` and `name.json`; `.to_csv()` and `.to_excel()` are one call each
  (FR-9.1, FR-14.2). Object results (an `SPIScore`, a fitted `SkewNormal2D`) are
  pydantic models with `.to_toml()`.
- Every public analysis function returns something with `.provenance`. A small helper
  (`provenance.capture(operation, method, settings, inputs)`), used as a context
  manager, records warnings with `warnings.catch_warnings(record=True)` and re-emits
  them, so warnings travel with the result. That is how FR-5.9's label "in every table
  and plot" happens without per-plot code: `plot.*` reads `provenance.warnings` and
  adds the label.

*Rejected:* xarray for results. Contributors know it less well, and the tables are
naturally relational. A `Result[T]` generic monad wrapping everything was also
rejected: generics plus `ty` plus contributors is friction. Two concrete result kinds
are enough.

### 2.3 Dataframe strategy

**Decision:** pandas (≥2.2) is the only public dataframe type. Numerical kernels are
numpy functions. Polars or Arrow input is accepted at the intake boundary only
(`read_survey(df)` converts through `pandas.api.interchange.from_dataframe` or
pyarrow), and outputs are pandas.

Why: course participants, seaborn, statsmodels, scipy examples and Excel I/O all speak
pandas. 10,000 responses (NFR-3) is tiny. Keeping the maths in numpy kernels means the
dataframe choice touches only intake, grouping and output. It is replaceable later
without touching the science.

*Rejected:* narwhals throughout. It is a real tax for a lone maintainer: every
contributor must learn its subset API, recipes become harder, and it buys only "accept
polars natively", which the boundary conversion already gives. Polars-only was
rejected for familiarity in teaching and for seaborn/statsmodels friction.

Rules: nullable integer dtypes for codes (`Int8`); categorical for ids and status; no
silent coercion (AGENTS.md). If intake would need a coercion, it raises `MappingError`
naming the column.

### 2.4 Settings, profiles and serialisation (FR-2.9, FR-11, FR-14.2)

- Every operation has one pydantic settings model: `ProjectionSettings`,
  `QualitySettings`, `DensitySettings`, `SPISettings`, `CalibrationSettings`,
  `MetricSettings`, `BatchSettings`. `AnalysisSettings` groups the survey-side ones.
- **Profiles** are named, versioned `AnalysisSettings` presets in
  `data/profiles/*.toml`: `iso-12913-3-2019`, `iso-12913-3-2024`,
  `soundscape-across-languages` (the default). A profile may lock fields: the ISO
  profiles lock `formulation` and forbid renormalisation (FR-2.5a). Overriding a locked
  field raises `ProfileViolationError` unless `profile=None`.
- **Resolution order, lowest to highest:** package default, then profile, then project
  `settings/analysis.toml`, then explicit keyword arguments. The *resolved* model is
  what provenance stores, alongside the profile name. A result never depends on what a
  profile name means in a later release.
- **Format:** TOML for anything a human edits or attaches to a paper (FR-11.3): read
  with the standard library's `tomllib`, write with `tomli-w`. JSON for
  machine-written provenance sidecars. `soundscapy.settings.json_schema()` exports
  JSON Schema for every model, which is the GUI's form definition for free (FR-14.7).
- *Rejected:* YAML, which has implicit typing pitfalls and needs a non-stdlib reader.
  Hand-written dataclasses with validation: bespoke, and no schema. attrs plus cattrs:
  no JSON Schema, and fewer contributors know it.

### 2.5 Project directory (FR-1.7, FR-14.4)

```
my-study/
  soundscapy.toml          project manifest: format version, package version, created
  mapping/<name>.toml      FR-1.3 mappings, reusable on the next export
  settings/analysis.toml   overrides on top of a profile
  settings/audio.toml
  registry/angles/*.toml   user angle sets, copied in (FR-2.4)
  registry/targets/*.toml  bespoke targets
  briefs/<name>.toml       design briefs and scenarios (FR-7.1, FR-7.2)
  inputs.toml              input files referenced with sha256 (not copied by default)
  results/<op>-<yyyymmddThhmmss>-<hash8>/
      <table>.parquet  <table>.json  [figure.svg]
  runs/<run-id>/           audio batches: manifest.json, files/<key>.parquet, log
  cache/                   safe to delete; gitignore'd
```

- `<hash8>` is the fingerprint of (operation, resolved settings, input fingerprints).
  Rerunning with identical inputs finds the existing result, which gives free caching
  and resumability.
- `Project.open(path)` / `Project.init(path)` is a thin object. All library functions
  also work without a project. A project is where things are saved, not a context the
  library needs (no hidden global state, per AGENTS.md).
- Packaging and sharing (zip, GUI open) are deferred, as the PRD allows.
- *Rejected:* a SQLite or DuckDB project file. It is opaque to non-programmers and git,
  and fails FR-14.2's "inspectable". A single HDF5 or zarr store: same objection, plus
  a heavy dependency.

### 2.6 Extension and plugin mechanism (FR-5.2, FR-13.1, FR-13.3)

- Discovery uses standard `importlib.metadata.entry_points` with groups
  `soundscapy.predictors`, `soundscapy.metrics`, `soundscapy.datasets` and
  `soundscapy.plots`. **The core registers its own built-ins through the same entry
  points**, so there is one code path and the recipes show exactly what the core does.
- Each extension type is a `typing.Protocol` plus a pydantic `Spec`:
  - `Predictor`: `spec: PredictorSpec` (inputs, required/optional context, design
    parameters, outputs, `output_kind: "single" | "distribution"`, name, version,
    training data, licence) and `predict(inputs: PredictorInputs, *, progress, cancel)
    -> Prediction`. Prediction tables carry `source = "predicted:<name>@<version>"`
    (FR-5.6).
  - `Metric`: `spec: MetricSpec` (name, library, library_version, method, standard,
    unit, kind `per_channel | cross_channel`, signed, framewise, licence) and
    `compute(signal_pa: np.ndarray, fs: int, settings) -> MetricOutput` (series and/or
    scalar per channel).
  - `DatasetLoader`: `spec` (catalogue fields, licence, versions) and
    `load(version, cache_dir) -> Survey`.
  - `PlotType`: a function `(geometry objects..., ax) -> Axes`, registered by name.
- `plugins.list()` returns specs and missing requirements without importing heavy
  modules: entry-point metadata only, then a lazy load. A plugin that fails to import
  becomes a `PluginLoadWarning`, never a crash. `SOUNDSCAPY_PLUGIN_API = 1` is
  checked against the plugin's declared `api_version`.
- SoundAQnet ships as a separate distribution, `soundscapy-soundaqnet`, so torch never
  enters the core lock (NFR-1, FR-5.4). Weights are fetched with pooch on first use,
  with an INFO log of URL and licence and a sha256 check.
- *Rejected:* pluggy, which is built for calling hooks, not for registering typed
  objects, and is another concept to learn. Import-time registration decorators were
  rejected because ordering and side effects make "list without importing" impossible.

### 2.7 Plot geometry vs drawing (FR-3.1, FR-3.2, FR-8.1 to FR-8.4)

- `geometry` returns plain dataclasses of numpy arrays: `ScatterLayer`,
  `DensityGrid(x, y, z, bandwidth, n, estimator)`, `ContourSet(levels, polygons:
  list[list[(m,2) array]], kind: "deciles" | "hdr")`, `RegionTable`,
  `MarginalCurves`, `ShiftArrow`, `TargetLayer` (a grid from the skew-normal pdf).
- **KDE:** `scipy.stats.gaussian_kde` driven exactly as seaborn's bivariate KDE (Scott
  factor × `bw_adjust`, gridsize 200, `cut` 3, clipping to the circumplex), exposed as
  the `seaborn-compatible` estimator settings. The 2022 reproduction uses
  `bw_adjust=1.2` (owner-confirmed, §10.5b). Percentile contours use the
  highest-density level enclosing a fraction of mass, computed here. Contour polygons
  come from **contourpy** (matplotlib's own contouring engine, a standalone wheel), so
  geometry never imports matplotlib, and a GUI gets the same polygons.
- N < 30 (FR-3.2): `density_geometry` raises `SmallSampleError` unless
  `allow_small_n=True`, in which case it records a `SmallSampleWarning`.
  `plot.density(..., by=...)` catches this per group, draws scatter and adds a caption
  note (the drawing-layer policy, §10.5).
- `plot` is matplotlib only for 1.0: `plot.scatter`, `plot.density`, `plot.contour50`,
  `plot.target`, `plot.likert`, `plot.radar`, `plot.compare`. Every function takes and
  returns an `Axes`, accepts `**kwargs` passed straight to matplotlib (FR-8.3), and
  uses a colour-blind-safe default cycle (Okabe–Ito). There is no styling system.
- *Rejected:* seaborn as the drawing layer. It fuses estimation and drawing, which is
  the problem FR-8.4 names, and makes the KDE settings implicit. seaborn stays a test
  dependency, to prove the grid matches. A plotly backend waits for the GUI.

### 2.8 Audio batch orchestration (FR-1.6, FR-4.1 to FR-4.11, FR-11.4, FR-14.3)

- **Indexing:** `audio.index(folder | list, link=table)` uses `soundfile.info` (format,
  channels, rate, depth, duration) without loading. A sha256 is computed lazily at run
  time. It returns a `ResultTable`.
- **Calibration:** a tagged union. `KnownLevel(level_db_per_channel, reference
  segment)`, `CalibratorTone(file | segment, level_db, frequency_hz=1000)`,
  `Sensitivity(pa_per_fs)`. Each resolves to a per-channel gain (FS → Pa), recorded in
  provenance. EQ filter (FR-4.3, a Should) is an optional user FIR/SOS file applied
  after calibration.
- **Work unit:** (file, window), where the window is the whole file or a user window on
  long SLM files, read with `soundfile.blocks` (FR-4.1a). The unit key is
  sha256(file bytes) ⊕ fingerprint(calibration, metric settings, method versions).
- **Runner:** `run_batch(index, calibration, metrics, *, workers, run_dir, progress,
  cancel)` uses the standard library's `concurrent.futures.ProcessPoolExecutor` with an
  explicit `spawn` context, the same on every OS, so Windows is not special. Each unit
  writes `runs/<id>/files/<key>.parquet` on completion. Resuming means skipping keys
  that exist (FR-4.8). Cancel is a `threading.Event` checked between units; queued
  futures are cancelled. `manifest.json` records hashes, versions, timings and failures
  (FR-11.4).
- **Output shape:** one long tidy table (`file, window, channel, metric, statistic,
  value, unit, library, method`). `representative(table, rule="max" | "mean")` applies
  the higher-of-channels rule to final statistics, not frames (FR-4.6). It records the
  chosen channel, skips `cross_channel` metrics (IACC), and refuses signed statistics
  unless told. `to_wide()` gives the survey-joinable table (FR-4.9).
- **Backends at M3:** levels (LAeq, LCeq, LAFn) with A/C weighting built from IEC
  61672 pole/zero definitions, validated against the standard's tolerance tables (an
  "elementary" metric, FR-4.4). MoSQITo (loudness ISO 532-1, sharpness DIN 45692,
  roughness, tonality where available). scikit-maad indices. IACC computed directly.
- *Rejected:* joblib/loky. Fine, but cancellation and per-unit progress need workarounds,
  and the standard library is enough for picklable module-level tasks. dask and
  Hamilton: framework weight far beyond the need (NFR-9). Threads: the metric code is
  GIL-bound Python/numpy.

### 2.9 CLI (FR-14.1, FR-14.5)

`click`, thin over the library. Every command reads and writes a project directory and
has a fully non-interactive mode.

```
soundscapy init DIR
soundscapy map suggest EXPORT [--schema ssid] -o mapping/survey.toml   # then edit or --accept
soundscapy analyse DIR --input EXPORT --mapping mapping/survey.toml [--profile ...]
                       # quality → project → summaries → Part 3 figures
soundscapy score DIR --target spi2024/vibrant@1 | --target-file t.toml [--by location_id]
soundscapy evaluate DIR --brief briefs/park.toml
soundscapy audio DIR --folder rec/ --calibration settings/audio.toml [--resume]   # M3
soundscapy predict DIR --model linear2021-fixed --features ...                   # M4
soundscapy export DIR RESULT --to csv|xlsx
soundscapy datasets list|fetch isd [--version]
soundscapy plugins list
```

*Rejected:* Typer, an extra layer over click whose annotation-driven behaviour changes
between versions. argparse: roughly twice the code for nested commands and help.
Generating the CLI automatically from pydantic models: a bespoke framework, and it
harms help-text quality.

### 2.10 Logging, warnings and errors (NFR-13)

- One logger, `logging.getLogger("soundscapy")`. At import, attach a stderr handler at
  INFO **only if** the root logger has no handlers, with a filter that defers when the
  user later configures logging. `soundscapy.set_log_level("WARNING")` is the one-call
  silencer. A GUI adds its own handler. The library never calls `print`.
- Exceptions: `SoundscapyError(code: str, context: dict)`, subclassed by
  `MappingError`, `ScaleViolationError`, `UncoveredLanguageError`,
  `LowTierLanguageError`, `ProfileViolationError`, `SmallSampleError`,
  `AxisPairMismatchError`, `InadmissibleCPError`, `CalibrationError`,
  `PluginLoadError`, `CancelledError`. `code` is stable and documented, and `context`
  is JSON-serialisable: that is the structured error a GUI needs (§10.6).
- Warnings: `SoundscapyWarning`, subclassed by `MediumTierWarning`,
  `MixedGeometryWarning`, `SmallSampleWarning`, `PredictedSpreadWarning`,
  `MethodSubstitutionWarning`, `PluginLoadWarning`. They are captured into provenance
  (§2.2).
- Progress: `Progress` protocol, `(done: int, total: int | None, message: str) -> None`.
  The default is `tqdm.auto` when stderr is interactive, otherwise INFO log lines every
  10%. Downloads report progress through pooch and are not cancellable (FR-14.3).

---

## 3. Public API sketch

```python
import soundscapy as sspy

# Journey A: export to Part 3 statistics and figures, every number reproducible
mapping = sspy.suggest_mapping("export.xlsx", schema="ssid")   # FR-1.3a; edit or accept
mapping.save("my-study/mapping/survey.toml")
survey  = sspy.read_survey("export.xlsx", mapping=mapping)     # coded per Table A.1 (FR-2.1)
survey  = sspy.check_quality(survey)                           # StatusLog; ISD v1.0 → 56 out
proj    = sspy.project(survey, profile="soundscape-across-languages",
                       on_uncovered="exclude")                 # FR-2.4 override, recorded
items   = sspy.item_summary(survey, by="location_id")          # median, IQR (+ range)
coords  = sspy.coord_summary(proj, by="location_id",
                             regions=["calm", "vibrant"])      # mean, SD, quantiles, %likely
ax      = sspy.plot.density(proj, by="location_id", percentile=50)  # N<30 → scatter
sspy.save(my_study := sspy.Project.open("my-study"), proj, items, coords, ax.figure)
survey.status_table()                                          # one column explains every absence

# Journey C: score a site
target  = sspy.targets.get("spi2024/vibrant", version=1)
bespoke = sspy.SkewNormal2D.from_direct(xi=[0.4, 0.6], Omega=[[0.2, 0.05], [0.05, 0.1]],
                                        alpha=[-1.0, 2.0], axes=proj.axes)
sspy.plot.target(bespoke)                                      # shown before scoring (FR-6.2)
site    = proj.distribution(where={"location_id": "RegentsParkJapan"})
score   = sspy.spi(site, target, settings=sspy.SPISettings(seed=42))  # n, variant recorded
table   = sspy.spi_table(proj, by="location_id", targets=[target, bespoke])  # FR-6.4 ranks
score.value, score.provenance.to_toml()
```

Signatures behind it (abridged):

```python
def project(survey: Survey, *, profile: str | None = "soundscape-across-languages",
            settings: ProjectionSettings | None = None,
            on_uncovered: Literal["error", "exclude", "equal_angles"] = "error",
            on_low_tier: Literal["error", "exclude", "equal_angles"] = "error") -> Projection
def spi(test: Distribution, target: Target | SkewNormal2D, *,
        settings: SPISettings | None = None, allow_axis_mismatch: bool = False) -> SPIScore
class SPISettings(BaseModel):
    n_target: int; seed: int; ks_variant: Literal["fasano_franceschini", "peacock"]
    combine: Literal["max", "mean"]; ties: Literal["inclusive", "exclusive"]
    truncate_target_to_square: bool
```

---

## 4. Delivery plan

Principles: each slice is a vertical path from data to a saved, provenance-carrying
result with its acceptance test. Every slice ends with `uv run task check` green. Paper
reproductions run in a separate nightly `docs:papers` job, not on every PR. Toolchain
and CI are already in place per AGENTS.md.

### M1: journeys A and I (2022). Release `1.0.dev1` (not announced)

| Slice | Delivers | Proves | Acceptance tests |
|---|---|---|---|
| S1 Walking skeleton | `errors`, `provenance`, `instrument` (default circumplex, 1..5 scale), canonical kernel, bundled ISD subset, `project`, `plot.scatter`, `ResultTable` parquet+json | Provenance survives a save/load round trip; geometry/drawing split exists from day one | Hypothesis: neutral → origin; bounds [−1, 1] for random angle sets; equal angles + 1..5 → 2019 formula to 1e-12; hand-computed 2022 Eqs. 1–2 values |
| S2 Intake and quality | CSV/Excel reading, `Mapping` TOML, `suggest_mapping` (stdlib `difflib` plus an alias table for the SSID codebook), Table A.1 coding, `StatusLog`, quality rules | FR-1.3/1.5 without silent drops | **ISD v1.0: 56 excluded = 37 missing + 19 single non-midpoint** (network-marked test); scale-violation and mapping error codes |
| S3 Languages and formulations | `AngleSet` TOMLs (SATP v1.5, Aletta 2024), tiers, per-response angles, the 3-way overrides, `iso2019` and `aletta2024` kernels, the three profiles, locked fields | FR-2.3/2.4/2.4a/2.9 | Angle tables vs chapter notes (transcribed fixture); Aletta Eqs. 1–3 by hand; owner's worked example (§10.3) when supplied; warnings captured in provenance |
| S4 Distributions and Part 3 figures | `Empirical2D`, seaborn-compatible KDE, HDR/decile contours via contourpy, regions, N<30 policy, item stats, Likert, radar | FR-3.1/3.2, FR-8.1/8.2/8.4 | Our grid equals seaborn's to 1e-10 (seaborn as test dep); contour mass fraction ≈ requested ±1/n; region proportions by hand |
| S5 Journey A done | `coord_summary`, `Project`, CLI `init`/`map`/`analyse`/`export`, draft tutorial A, 2022 reproduction `.qmd` (ISD v0.2.3) | "Export to Part 3 statistics in one session, reproducible from saved config" | Rerun from project TOML gives byte-identical parquet; Windows CI; 2022 `.qmd` renders nightly |

Deferred from M1: FR-2.4b ipsatisation (to M5), FR-2.5a renormalisation (post-1.0;
the kernel already supports per-row normalisers, so it is cheap later), FR-2.7
clustering (post-1.0), Qualtrics schema detection (M2 if asked for).

### M2: journeys C, E1, I (SPI) and K. Release `1.0a1`

| Slice | Delivers | Proves | Acceptance tests |
|---|---|---|---|
| S6 Skew-normal | `SkewNormal2D` dp/cp, pdf, seeded sampling, closed-form mean, MLE fit | FR-3.4 validated (NFR-9) | **R `sn` fixtures** (owner's run): dmsn values, cp↔dp, `selm` fit to stated sig figs; PD checks; sample moments vs closed form (hypothesis, loose tolerance justified) |
| S7 SPI | `ks2d` (FF and Peacock), `spi`, `spi_table`, `Target` library (3 demo + Example 6 tranquil), axis-pair refusal | FR-6.1–6.4 | **2024 paper tables at paper seed and n** once §10.3 fixes the variant; ks2d vs published FF example values; identical samples give D = 0; axis mismatch raises |
| S8 Course statistics | correlations (scipy Spearman/Pearson), comparisons with effect sizes, WHO-5, source-dominance profiles, ISD psychoacoustic-table join | Course cases 1–3 | Hand-checked effect sizes; WHO-5 per its published rule (owner confirms, §10.6) |
| S9 Design E1 | `DesignBrief`, `Scenario` (TOML), `evaluate()`: SPI, shift vectors, quadrant proportions, KPI flag, ranking table, overlay plot | FR-7.1–7.3 | Example 6 story tutorial; SPI = 56 only if owner data arrives |
| S10 Course-ready and 1.0a1 | Tutorials A, C, E; install how-to for Windows/macOS/Colab; CLI `score`/`evaluate`; `plugins.py` + plot-type and dataset-loader recipes with test templates; `Predictor` protocol and spec frozen (no model yet); versioning/deprecation policy; GOVERNANCE.md; CITATION.cff | Journey K; plugin path exists before anyone needs it | First plot inside the hour, timed with one non-programmer on Windows; every tutorial renders in CI |

Deferred: FR-6.1 multi-seed summary (a 20-line addition; do it in S7 only if time
allows), FR-6.5 to M5, FR-7.4/7.5 as metadata fields and tables only.

### M3: journey B. Release `1.0a2`

| Slice | Delivers | Acceptance tests |
|---|---|---|
| S11 Index and calibration | `audio.index`, `read`, `Calibration` union, windows | Synthetic 1 kHz tone at 94 dB gives 94.00 dB after `CalibratorTone`; gain round-trip; mono/stereo; >2 channels refused |
| S12 Levels | A/C weighting, LAeq/LCeq/LAFn, the `Metric` protocol, metric recipe | IEC 61672 weighting tolerance table; pink-noise LAeq analytic |
| S13 Library adapters | MoSQITo and scikit-maad adapters with `MetricSpec` (library, method, licence), IACC, channel rules, long → wide | MoSQITo's own validation signals (ISO 532-1 test signals) reproduced through our adapter; channel-rule property tests; signed-statistic refusal |
| S14 Batch | `run_batch` (spawn pool, resume, cancel, manifest), CLI `audio` | Kill mid-run, then resume finishes with no recompute (count units); cancel leaves a consistent run dir; Windows CI |
| S15 Trust | ArtemiS comparison page (FR-4.10), B tutorial on shipped or synthetic clips | Renders; differences tabulated, not gated |

Deferred: FR-4.11 time-series exposure beyond `MetricOutput.series` (already in the
protocol), FR-8.5, FR-4.3 EQ (in S11 if a user filter exists to test with).

### M4: journeys D, E2 and I (2021). Release `1.0a3`

- S16 `linear2021` (fixed-effects and known-location variants, FR-5.3a); the 2021
  reproduction from the ISD feature table (tolerance per published sig figs).
- S17 Prediction to distribution (FR-5.9) with `PredictedSpreadWarning`; E2 through
  `evaluate()` with predicted scenarios; CLI `predict`.
- S18 Default predictor chosen at planning (FR-5.3); predictor recipe; the
  `soundscapy-soundaqnet` plugin package (weights via pooch, sha256, licence log).

### M5: journeys L, F and G

- S19 Contributor trial (FR-13.1: one new contributor, one recipe, under a day); fix
  what it exposes. This is the most valuable M5 item.
- S20 FR-6.5 NSGA-II via `pymoo` as an `[optimise]` extra.
- S21 Journey G: ship the ISD fully and **one** more dataset (SATP, since its languages
  exercise FR-2.4). Defer ARAUS/DeLTA and MOSAIQ import/export until the schema is
  stable (§11 already flags the draft status).

### What I would cut or defer relative to the PRD, and why

- FR-2.5a renormalisation and FR-2.4b ipsatisation: no field-survey user needs them for
  1.0, and the kernel design keeps them cheap later.
- FR-2.7 clustering: a Should with no defined method in the cited sources; users have
  scikit-learn.
- FR-1.8 / FR-10.3 MOSAIQ conformance: the schema is a draft, so implementing it now
  means implementing it twice.
- FR-10.2 beyond one dataset: each loader is mostly a mapping file, and the dataset
  recipe lets contributors do the rest. That is journey L in action.
- FR-7.5 report content: a ranked table plus overlay figure is enough; FR-9.2 already
  rules out reports.
- The 2022 reproduction leaves the per-PR docs build (nightly instead), so CI stays
  offline and fast.

---

## 5. Risks and open questions

### 5.1 Top risks

1. **The SPI acceptance depends on unknown statistic details (§10.3).** If the 2024
   paper's KS variant, combination, tie rule or truncation cannot be recovered, the
   tables will not reproduce. Mitigation: every choice is a setting from S7, and the
   fixture test is parametrised so the answer is a default change. Escalate this owner
   input first, before M2 planning.
2. **Skew-normal MLE numerics.** Bivariate SN likelihoods have boundary MLEs (α → ∞)
   and flat regions. Mitigation: fit in CP as Azzalini advises, report convergence and
   boundary flags in provenance, and validate against `sn` fixtures that include a
   near-boundary case.
3. **seaborn-compatibility drift.** If seaborn changes defaults, the 2022 reproduction
   stays tied to our frozen copy of its settings, which is the point. But the
   equality test must pin the seaborn version it was checked against.
4. **Windows plus spawn plus notebooks.** Worker functions must be importable module
   functions, and plugins' metric callables must be picklable. Mitigation: the runner
   passes metric *names* resolved in the worker via entry points, never callables, and
   Windows CI runs S14's resume test.
5. **MoSQITo coverage, speed and maintenance.** It may be slow or lag its standards.
   Mitigation: FR-4.0 already makes coverage a non-gate; adapters isolate it, and the
   long-table output means a backend swap changes only the `library` and `method`
   columns.
6. **pydantic v2 under `ty`.** ty's support for `dataclass_transform` and pydantic is
   still maturing. Mitigation: settle it in S1, and if it is painful, keep pydantic
   only on settings and specs, not on data carriers (already the design).
7. **Maintainer bandwidth.** M2 holds SPI, skew-normal, course statistics, design E1,
   three tutorials and the plugin path. Mitigation: S8 and S10's recipes are the first
   candidates to slip to `1.0a2`; S6–S7 are not negotiable.
8. **ISD language coverage under the default profile (§10.3).** If ISD languages fall
   outside SATP v1.5, the headline course exercise errors. Mitigation: the bundled
   teaching subset ships with a mapping that declares a language column. The tutorial
   shows `on_uncovered="exclude"` deliberately, as a teaching moment.

### 5.2 Questions only the owner can answer

1. The 2024 SPI paper's KS variant, direction combination, tie rule, target truncation
   and sample size, and the R `sn` fixture run (both block M2).
2. A worked example of the canonical projection with adjusted angles (blocks S3's
   headline fixture).
3. Which ISD languages are outside SATP v1.5, and the intended default for the course:
   exclude, or keep the error?
4. Is pandas-only at the public surface acceptable, given the plan's narwhals choice?
   Is there a known user group that needs native polars output?
5. Should the 2019 profile be refused on non-1..5 scales (where its divisor is
   undefined), or generalised? This plan refuses.
6. The WHO-5 scoring rule to cite (raw ×4 percentage, or raw sum), and the
   missing-item rule for it.
7. Is redistributable audio available (calibrator recording, clips with ArtemiS
   values)? This decides whether S11–S15 use shipped or synthetic fixtures.
8. Copyleft confined to extras (§10.6), or allowed in core? This affects whether any
   GPL metric backend can be a default.
9. Is "a ranked table plus overlay figure" acceptable as the full content of FR-7.3 and
   FR-7.5 for 1.0?
10. Is a nightly job, rather than per-PR `task docs`, acceptable for paper
    reproductions?

### Critical files for implementation
- /Users/mitch/Documents/GitHub/Soundscapy/.claude/worktrees/compound-engineering-brainstorm-519d9f/design/PRD.md
- /Users/mitch/Documents/GitHub/Soundscapy/.claude/worktrees/compound-engineering-brainstorm-519d9f/design/literature/mitchell-et-al-2026-satp-chapter-4-notes.md
- /Users/mitch/Documents/GitHub/Soundscapy/.claude/worktrees/compound-engineering-brainstorm-519d9f/design/literature/aletta-et-al-2024-applied-acoustics.pdf
- /Users/mitch/Documents/GitHub/Soundscapy/.claude/worktrees/compound-engineering-brainstorm-519d9f/design/literature/mitchell-aletta-kang-2022-jasa-el.pdf
- /Users/mitch/Documents/GitHub/Soundscapy/.claude/worktrees/compound-engineering-brainstorm-519d9f/AGENTS.md
