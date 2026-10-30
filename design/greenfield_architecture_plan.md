# Architecture plan for the soundscapy rewrite

Status: **draft 0.2 for review** (2026-10-30). This draft has been rewritten from PRD
draft 0.5 (`design/PRD.md`) and supersedes draft 0.1, which predated the PRD. It is
the document AGENTS.md means by "the architecture plan", and it may now be used for
design and planning. Nothing in it is fixed except the science; every decision below
is a proposal with the alternative it beat and the reason, so that each one can be
challenged on its merits.

How to read it: §1 and §2 state the forces and the principles; §3 is the domain
model, which is where most of the design risk lives; §4 to §12 cover modules, API,
tooling, testing, concurrency, plotting, persistence and GUI readiness; §13 lists the
places where this plan pushes back on the PRD or the literature; §14 is the delivery
sequence; §15 is the list of decisions only the owner can take.

---

## 1. What shapes the design

Five facts from the PRD drive almost every choice below.

1. **The unit of analysis is a response, and the unit of interpretation is a
   distribution.** Everything scientific in the PRD (FR-2.5, FR-3, FR-5.9, FR-6, FR-7)
   works on a set of points in the circumplex: survey responses projected one at a
   time, single-response predictions pooled, a sampled target. The package needs one
   representation of "a set of circumplex coordinates with provenance" and one of "a
   parametric distribution on the circumplex", and every downstream feature must take
   either.
2. **Item-level data must survive untouched.** FR-1.8 and the MOSAIQ schema require
   item responses kept separate from derived coordinates, with missingness explicit.
   So the package never rewrites a user's table; it reads it through a declared
   mapping and writes derived columns and records beside it.
3. **Every number is defensible.** Profiles (FR-2.9), formulations (FR-2.3), angle
   sets with versions and tiers (FR-2.4), seeds and sample sizes (FR-6.1), calibration
   (FR-4.2) and model versions (FR-5.6) must all be recorded with each result and be
   shareable as text (FR-11, FR-14.2). The design therefore separates *settings*
   (declarative, serialisable, validated) from *computation* (pure functions) from
   *results* (immutable, carrying the settings that produced them).
4. **A GUI will drive this core without a rewrite** (FR-14). Headless, serialisable,
   progress and cancellation on long jobs, a project directory as the unit of
   exchange, and a CLI that proves each journey works without Python.
5. **One part-time maintainer and volunteer contributors** (NFR-9). Small modules
   with one job, a strict public boundary, no bespoke frameworks, and recipes with
   test templates for the four extension types (FR-13.1). Any abstraction that a new
   contributor cannot explain in one sentence is suspect.

A sixth fact is a constraint on the plan itself: milestones M1 and M2 (survey analysis
and SPI, no audio) are the 1.0 candidate (§9 of the PRD). The architecture must be
complete enough at M1 that M3 and M4 add modules rather than reshape them, but nothing
for M3 to M5 is built early.

---

## 2. Design principles

Each principle says what it rules out, because that is what makes it testable.

1. **Functional core.** Computation is pure functions over numpy arrays and pandas
   DataFrames. No function mutates its input, reads global state, writes files, prints
   or shows a figure. Rules out "analyser" and "manager" classes with mutable state.
2. **Settings are data; results carry their settings.** Every configurable thing is a
   frozen pydantic model with a stable text form. Every result object holds the
   resolved settings and the package version that produced it. Rules out settings
   passed as loose keyword arguments through several layers, and results that cannot
   be re-derived.
3. **Objects for values and contracts, not for workflow.** Classes are used for value
   objects (a circumplex, a scale, a calibration), for result records, and for
   `Protocol` extension points (a metric backend, a predictor, a dataset loader). Rules
   out inheritance from third-party classes, deep hierarchies, and classes whose
   methods are a pipeline in disguise.
4. **We do not own the user's DataFrame.** Survey data stay in pandas with the user's
   column names. A `Mapping` says which column is which construct and on what scale.
   The package validates through the mapping and adds derived columns with documented
   names. Rules out renaming or reordering user columns and wrapping rows in objects.
5. **Positions, not words, for attributes.** A circumplex names its attributes by
   position identifier (PAQ1…PAQn, counter-clockwise from the positive x-axis) and
   carries labels per language. The mapping binds user columns to positions
   explicitly. Rules out code that assumes eight English words or a particular column
   order, the bug class that the SSID codebook order (paq01 = pleasant, paq02 =
   chaotic, …) has caused before.
6. **Geometry is computed, drawing is thin.** Density grids, contour polygons,
   quadrant proportions and shift vectors are returned as data by the core. Drawing
   functions put that data on a matplotlib `Axes`. Rules out plotting code that
   computes, and computing code that imports matplotlib.
7. **Wrap, never re-implement, unless there is nothing to wrap.** Psychoacoustic
   metrics, statistics, optimisation, file formats and data fetching come from
   maintained libraries. The package writes only what does not exist: circumplex
   projection and its variants, skew-normal parameterisation and fitting, the 2-D KS
   distance, SPI, calibration, batch orchestration and the harmonisation layer. Rules
   out a second KDE, a second mixed-model fitter, a second plotting system.
8. **Fail loudly, with the override in the signature.** Undeclared languages,
   low-confidence angle sets, N < 30, scale violations and missing optional inputs
   raise specific exceptions or emit specific warning classes, and the override is an
   explicit argument whose value is recorded in the result. Rules out silent drops,
   silent coercion and `warnings.warn` with a generic category.
9. **Headless by construction.** No `plt.show()`, no `print`, no `input`, no implicit
   file writes, logging through the standard library with a `NullHandler`. Progress
   through a callback, cancellation through a token. Rules out tqdm or rich inside
   the library (the CLI may use them).
10. **Dependency direction is enforced.** Lower layers never import higher ones; an
    import-linter contract runs in CI. Rules out the "everything imports everything"
    drift that makes a volunteer refactor frightening.

---

## 3. The domain model

This section is the heart of the plan. The names are proposals; the shapes and the
boundaries between them are the point.

### 3.1 Instruments: circumplex, scale, angle sets, profiles

```python
class Scale(BaseModel, frozen=True):
    """An ordinal or interval response scale, e.g. Likert 1..5 (ISO Table A.1)."""
    minimum: float
    maximum: float
    kind: Literal["ordinal", "interval"]
    anchors: tuple[str, ...] = ()           # optional labels, lowest first
    # midpoint and half-range are derived properties (FR-2.3): mu = (min+max)/2, rho = (max-min)/2

class Attribute(BaseModel, frozen=True):
    position: str                           # "PAQ1" … "PAQn"
    angle_deg: float                        # equal-angle default for this circumplex
    labels: Mapping[str, str]               # ISO 639-3 code -> label, e.g. {"eng": "pleasant"}

class Circumplex(BaseModel, frozen=True):
    """An instrument specification (FR-2.0), not a list of eight English words."""
    name: str                               # "iso12913-outdoor", "torresin-indoor"
    attributes: tuple[Attribute, ...]       # counter-clockwise from +x
    axes: tuple[str, str]                   # ("pleasant", "eventful"); output column names
    scale: Scale
    reference_language: str                 # language whose labels are the canonical aliases

class AngleSet(BaseModel, frozen=True):
    """A versioned set of per-language angles with confidence tiers (FR-2.4, FR-2.4a)."""
    name: str                               # "satp-v1.5", "aletta-2024", or user-supplied
    circumplex: str                         # which circumplex these angles belong to
    languages: Mapping[str, LanguageAngles] # ISO 639-3 -> angles per position + tier
    provenance: Provenance

class Profile(BaseModel, frozen=True):
    """A named bundle of edition-specific settings (FR-2.9)."""
    name: str                               # "iso12913-3-2019", "iso12913-3-2024"
    formulation: Literal["canonical", "iso2019", "aletta2024"]
    angles: Literal["equal"] | str          # "equal" or an AngleSet name
    dispersion: Literal["range", "iqr"]
    metric_set: str                         # name of a Table D.1 edition (used from M3)
```

Decisions and the alternatives they beat:

- **Position identifiers are the key, labels are display.** Draft 0.1 wanted canonical
  English names as keys. The PRD's generalised circumplex (FR-2.0) and the
  translation work make positions the stable thing and labels the variable thing. For
  the default circumplex, the reference-language labels (`pleasant`, `vibrant`, …) are
  accepted as aliases wherever a position is expected, so tutorial code reads
  naturally and the indoor circumplex still works unchanged.
- **The circumplex owns its output column names.** Projection writes the two axis
  names the circumplex declares, so the indoor circumplex produces its own axes and no
  analysis code hard-codes `pleasant`/`eventful`.
- **Scale coding is explicit.** The scale lives on the circumplex (the instrument
  defines it) and may be overridden per mapping (a survey used 1..7). Projection
  reads `mu` and `rho` from the scale actually used, which is what makes the canonical
  formulation reduce correctly to the 2019 one.
- **Angle sets and profiles are data files, not code.** They ship under
  `soundscapy/_data/` as TOML with provenance, and a user-supplied set is the same
  file format. Adding the next SATP version is a data change plus a test.
- **Profiles are thin.** A profile resolves to concrete settings and is recorded by
  name *and* by resolved values in every result, so a later change to what a profile
  means cannot silently change old results.

### 3.2 Surveys: mappings, tables, quality

```python
class Mapping(BaseModel, frozen=True):
    """Which user column means which construct (FR-1.3), persisted as a file (FR-1.3a)."""
    instrument: str                         # circumplex name
    scale: Scale | None = None              # override of the instrument's scale
    attributes: Mapping[str, str]           # position -> column name
    language: str | LanguageColumn          # one language, or {"column": name}
    ids: Ids                                # response/group/session/location column names, any subset
    source_dominance: Mapping[str, str] = {}   # construct -> column
    overall_quality: str | None = None
    appropriateness: str | None = None
    who5: tuple[str, ...] = ()
    extra: Mapping[str, ColumnRole] = {}    # demographics, free text, flags, carried through
    provenance: Provenance

class Dataset:                              # frozen dataclass, plain container
    responses: pd.DataFrame
    mapping: Mapping
    groups: pd.DataFrame | None = None      # one row per group; recording path lives here
    sessions: pd.DataFrame | None = None
    locations: pd.DataFrame | None = None
    recordings: RecordingIndex | None = None   # from M3; never loads audio
    provenance: Provenance
```

- **`Mapping` is the schema; there is no separate schema library.** Draft 0.1 proposed
  pandera. The checks the package needs (declared columns present, values within the
  declared scale, ids unique where required, language codes known) are a few dozen
  lines against the mapping, and FR-1.5 needs them to produce a per-response
  *decision and reason*, not a raised error. pandera would add a dependency, a second
  description of the same columns, and a class-based DSL that ty may not type well.
  Decision: a `validate(dataset) -> ValidationReport` function in the package;
  pandera is not used. Revisit only if validation needs grow beyond the mapping.
- **Quality rules are a list of named rule objects** (FR-1.5), each a small callable
  `(responses, mapping) -> pd.Series[bool]` with a name and a reason string. `apply_rules`
  adds `included` and `exclusion_reason` columns and returns a new frame. Rules are
  configurable by listing them in settings; the default list is documented. Nothing is
  dropped.
- **Mapping inference** (FR-1.3a) is `suggest_mapping(df, instrument, known_schemas=...)
  -> MappingSuggestion`, where each suggested binding carries a confidence and the
  evidence (exact name match against the SSID or REDCap codebook, fuzzy match, value
  range). A `MappingSuggestion` becomes a `Mapping` by `accept()` after the user
  confirms or corrects in the CLI prompt, in a notebook, or in a GUI form generated from
  the model's JSON schema. Known export schemas are data files, not code.
- **Hierarchy as separate tables** (FR-1.4). One wide frame cannot carry session logs
  and recordings cleanly. Most analysis functions take only `responses` and the
  mapping; `Dataset` exists so loaders have one return type and the project directory
  has one layout. It has no behaviour beyond validation and joins.
- **Ipsatisation** (FR-2.4b) is a pure transform `ipsatise(responses, mapping) ->
  pd.DataFrame` that returns a new frame and is recorded in provenance when used; it
  is never a default.

### 3.3 Coordinates and distributions: the central abstractions

```python
@dataclass(frozen=True)
class Coordinates:
    """A set of points in one circumplex, with provenance (FR-2.5, FR-5.6, FR-5.9)."""
    xy: np.ndarray                          # (n, 2), float64, in [-1, 1]^2
    circumplex: Circumplex
    source: Literal["survey", "prediction", "sample"]
    labels: pd.DataFrame | None             # optional per-point columns: ids, grouping keys
    settings: ProjectionSettings | PredictorRef | TargetRef   # what produced them
    # methods: subset(mask), group_by(key) -> dict[key, Coordinates], mean(), sd(), quantiles(),
    #          fraction_in(region) ("percent likely to perceive"), to_frame()

@dataclass(frozen=True)
class KernelDensity:
    """A KDE view of Coordinates with an explicit bandwidth (FR-3.1)."""
    coordinates: Coordinates
    bandwidth: float | tuple[float, float] | Literal["scott", "silverman"]
    # methods: grid(resolution) -> DensityGrid, mass_contour(fraction) -> Polygon,
    #          decile_contours() -> list[Polygon], marginals() -> (x, y) densities

class SkewNormal(BaseModel, frozen=True):
    """Bivariate skew-normal in Azzalini's direct parameters (FR-3.4, FR-6.2)."""
    xi: tuple[float, float]
    omega: tuple[tuple[float, float], tuple[float, float]]
    alpha: tuple[float, float]
    # classmethods: from_centred(mean, cov, gamma1) raising NotAdmissibleError;
    #               fit(coordinates, method="mle"|"mpe")
    # methods: to_centred(), sample(n, rng) -> Coordinates(source="sample"), pdf(xy)

class Target(BaseModel, frozen=True):
    name: str
    distribution: SkewNormal
    circumplex: str
    kind: Literal["demonstration", "reference", "bespoke"]
    provenance: Provenance
```

- **`Coordinates` is the object every downstream feature accepts.** Projected
  survey responses, pooled single-response predictions (FR-5.9) and samples from a
  target are all `Coordinates`; `source` and `settings` keep them distinguishable in
  every table and plot (FR-5.6). Descriptive statistics, quadrant proportions, KDE,
  shift vectors, SPI, scenario comparison and every plot take `Coordinates`. This is
  the single most important decision in the plan: it makes journeys A, C, D and E
  share one path instead of four.
- **Regions** (quadrants, half-planes, user polygons) are value objects built on
  `matplotlib.path.Path` for point-in-polygon, no shapely. Quadrant regions are
  defined from the circumplex axes so they work for any instrument.
- **KDE is a view, not a transform.** Bandwidth is explicit and recorded; the default
  is Scott's rule, stated in every result, with a sensitivity helper. The "50th
  percentile contour" is computed from the cumulative mass of the grid so its meaning
  is unambiguous and the fraction is a parameter (FR-3.1). Contour polygons come from
  `contourpy`, which matplotlib already depends on, so geometry needs no figure.
- **SkewNormal is pure numpy/scipy**, following Azzalini's `sn` for parameterisation,
  centred/direct conversion, admissibility, sampling and fitting; validated once
  against `sn` outputs committed as fixtures (NFR-9, FR-3.4). There is no R anywhere.
- **The 2-D KS distance** is a numpy implementation of the published method, chosen to
  match the SPI paper (FR-6.1); the paper's tables are the acceptance test.
- **N < 30** (FR-3.2) is enforced where a density is requested: `KernelDensity` raises
  `SmallSampleError` unless `allow_small_sample=True`, in which case it emits
  `SmallSampleWarning` and records the override.

### 3.4 Scoring and design

```python
def spi(test: Coordinates, target: Target, *, n_samples: int, rng: int | np.random.Generator,
        bootstrap: int | None = None) -> SPIResult
# SPIResult: value, ks_distance, n_samples, seed, sampling_sd, bootstrap_interval, settings, versions

def shift(a: Coordinates, b: Coordinates) -> ShiftVector
def overlap(a: KernelDensity, b: KernelDensity, fraction: float) -> float
def rank_against(target, scenarios: Mapping[str, Coordinates], **spi_kwargs) -> pd.DataFrame
def rank_targets(test, targets: Mapping[str, Target], **spi_kwargs) -> pd.DataFrame

class Brief(BaseModel, frozen=True):        # FR-7.1
    ambition: str; target: Target; kpi: KPI; assessment_method: str; scenarios: tuple[str, ...]

class Scenario(BaseModel, frozen=True):     # FR-7.2
    name: str; inputs: ScenarioInputs; produced_by: PredictorRef | SurveyRef; intervention_level: Level | None

def evaluate(brief: Brief, baseline: Coordinates, scenarios: Mapping[str, Coordinates], **spi_kwargs) -> Evaluation
# Evaluation: per-scenario SPI, shift, quadrant proportions, meets_kpi; tables ready for a report (FR-7.3, FR-7.5)
```

- **SPI is explicitly stochastic**, as published: `n_samples` and `rng` are required
  keyword arguments with no defaults in the core function; the profile supplies the
  paper's defaults when called through settings. Sampling variability is always
  reported; a bootstrap over the test set is an option (the PRD's open item 1).
- **Target search** (FR-6.5, Should) lives behind the `optim` extra with pymoo and
  returns a `Target` whose provenance records the ranking, objectives and optimiser
  settings. It is the only place in the package that needs a long-running optimiser,
  so it uses the same progress and cancellation hooks as audio.
- **Design (FR-7) is thin by design.** A `Brief` and `Scenario` are records; `evaluate`
  composes `spi`, `shift` and `fraction_in` and returns tables. There is no report
  engine (FR-9.2).

### 3.5 Audio: signals, calibration, metrics, batches (M3)

```python
@dataclass(frozen=True)
class Signal:
    samples: np.ndarray                     # (channels, n), float64 Pa after calibration, else "full scale"
    rate_hz: int
    channels: tuple[str, ...]               # ("L", "R") or ("M",)
    calibration: Calibration | None         # None means uncalibrated, and metrics that need Pa refuse

class Calibration(BaseModel, frozen=True):  # FR-4.2
    method: Literal["known_level", "calibrator_signal"]
    per_channel_db: tuple[float, ...]       # resolved gain per channel, dB
    reference: CalibratorReference | None   # file or segment, stated level, IEC 61672 procedure
    provenance: Provenance

class MetricSpec(BaseModel, frozen=True):   # one entry in a metric set
    metric: str                             # semantic name: "loudness_iso532_1_tv", "sharpness_din45692", "laeq"
    backend: str | None = None              # "mosqito", "maad", "builtin"; None = registered default
    params: Mapping[str, Any] = {}
    statistics: tuple[str, ...]             # ("p5", "p50", "p95", "mean", "max", "rmc", ...)  (FR-4.7)
    channel_rule: Literal["each", "higher", "mean"] = "higher"   # FR-4.6
    windows: Window | None = None           # FR-4.1a: fixed-length windows over long files

class MetricBackend(Protocol):              # extension point (FR-13.1, recipe at M3)
    name: str; library: str; version: str; standard: str; units: str; licence: str
    kind: Literal["per_channel", "cross_channel"]
    def compute(self, signal: Signal, **params) -> MetricOutput   # time series and/or scalar, with frame rate
```

- **Metrics are keyed by what they measure, not by which library computes them.**
  Settings name the metric; a registry maps `(metric, backend)` to an implementation.
  Each backend surfaces its library, version, standard and definition (FR-4.0), and
  its licence, so the metric table in the docs is generated, not hand-written.
- **Level metrics and IACC are built in** (FR-4.4 allows it): A/C weighting filters via
  scipy (IEC 61672 pole-zero design), Leq, SEL, percentiles, IACC by normalised
  cross-correlation. Everything else is wrapped.
- **Backends, from an evaluation of every Python entry in the LISQM list** (Appendix
  C has the full table). Against the hard requirements (pip wheels, no compiler, no
  MATLAB or R, Python 3.11 and later, no copyleft in the `soundscapy` distribution)
  the field is small:
  - **MoSQITo 1.2.1** (Apache-2.0; numpy, scipy, pyuff) is the default backend. It
    provides ISO 532-1 stationary and time-varying loudness, DIN 45692 sharpness,
    Daniel–Weber roughness, ECMA-418-2 loudness and roughness (an earlier edition than
    the 2025 fourth), and ECMA-74 prominence and tone-to-noise ratios. Smoke-tested in
    this session on Python 3.11 and 3.13 with numpy 2: all of those import and run,
    and 1 kHz at 74 dB gives 10.9 sone, consistent with ISO 532-1. Two risks: it has
    had no release since April 2024, and its ECMA roughness module imports matplotlib
    without declaring it. Mitigations: pin a tested range (NFR-4); declare matplotlib
    in the `audio` extra; Apache-2.0 permits a maintained fork or vendored patch if
    the project stays dormant.
  - **scikit-maad 1.5.2** (BSD-3; adds scikit-image and pywavelets) for ecoacoustic
    indices (FR-4.5), in the `audio` extra.
  - **phonometry 3.3** (MIT; numpy and scipy only) is the only permissive package that
    covers the whole of Table D.1 and the research set: ISO 532-1/2/3 loudness, all
    four ECMA-418-2 metrics (2025 edition), DIN 45692 sharpness, Osses fluctuation
    strength, psychoacoustic annoyance, ECMA-418-1 tonality. It requires **Python
    3.13 or later**, which is above the PRD's 3.11 floor, it is one person's work
    with self-reported conformance checks, and it is young (released 2026). Proposed
    use: an optional second backend installed with an environment marker
    (`phonometry; python_version >= "3.13"`), so users on 3.13 gain fluctuation
    strength and ECMA tonality while 3.11 and 3.12 users get MoSQITo only. See §15.
  - **Excluded from the distribution, with the reason**: `sottek-hearing-model`
    (AGPL-3 on PyPI, GPL-3 on GitHub; depends on PyQt6; the best-verified ECMA-418-2
    implementation, translated from the SQAT authors' MATLAB); `MetaSona` (fast C
    implementation with wheels, but the combined work is GPL-3 and the authors call
    roughness and tonality experimental); `refmap-psychoacoustics`, `rota-order`,
    `torch_amt` (GPL-3); `pySQAT` (TU Delft port of SQAT, no licence file, no
    release, so unusable by anyone); `ecma418-torch` (MIT but requires torch, one
    commit); `iso532-1-rs` (Apache-2.0 but no wheels, source build); `acoustic-toolbox`
    (BSD-3 and useful, but it pulls numba and soundfile for level metrics that are
    thirty lines of scipy here); `PsychoacousticParametersMeasurer`,
    `NumpyLibforPsychoAcoustic`, `LoudnessModel` (no licence, no release, or dormant).
    A GPL backend can still exist as a separately published plugin that a user
    installs knowingly, if the owner accepts that (§15).
- **Coverage statement for Table D.1 at M3, given the above.** Levels (LAeq, LCeq,
  LAF5, LAF95, LA10 − LA90, LCeq − LAeq, SEL, kurtosis) built in; loudness (N5, Navg,
  Nrmc, N95, N5/N95) and sharpness (S5, Savg, S50, S95) from MoSQITo; roughness (R10,
  R50) from MoSQITo (Daniel–Weber and ECMA); tonality via ECMA-74 PR/TNR from MoSQITo,
  with ECMA-418-2 tonality only through phonometry on 3.13+; fluctuation strength (F10,
  F50) only through phonometry on 3.13+; IACC built in. On Python 3.11 and 3.12,
  fluctuation strength and ECMA-418-2 tonality are documented gaps (FR-4.4), not
  bespoke implementations.
- **Calibration is a value object applied once**, when a `Signal` is loaded; metrics
  never see raw full-scale data without knowing it. Loading with no calibration is
  allowed (for inspection and metrics that are level-independent), and a metric that
  needs Pa raises `UncalibratedSignalError`.
- **Audio I/O.** WAV via `scipy.io.wavfile` in the core (it reads 16-, 24- and 32-bit
  PCM and float, which covers sound-level-meter and binaural recorder exports) and
  `soundfile` behind the `audio` extra for FLAC and other formats, so the LGPL question
  (PRD open item 2) affects only the extra. See §15.
- **Batch runner.** `analyse(recordings: RecordingIndex, metrics: MetricSet, *,
  executor, cache_dir, progress, cancel) -> AudioResults`. Work units are
  `(path, file_hash, calibration, MetricSet)` and are picklable; results are cached per
  file as a small table keyed by (file hash, settings hash, backend versions), so a
  rerun computes only what is missing (FR-4.8) and a manifest records hashes, versions
  and timings (FR-11.4). Output is one tidy table per recording and channel with
  columns `<metric>_<statistic>` and a units table beside it (FR-4.9).

### 3.6 Prediction (M4)

```python
class Predictor(Protocol):                  # FR-5.1
    info: PredictorInfo                     # name, version, training data, setting, licence, citation
    inputs: InputContract                   # required and optional features/context/design params, with units
    outputs: OutputContract                 # PAQ items, coordinates, sources, scene; "single" or "distribution"
    def predict(self, inputs: PredictionInputs) -> Prediction

@dataclass(frozen=True)
class Prediction:
    per_input: pd.DataFrame                 # one row per clip: predicted items/coordinates, labelled by model
    coordinates: Coordinates                # source="prediction"; pooled from single responses (FR-5.9) or sampled
    predictor: PredictorRef
```

- **Plugins by entry points.** Predictors register under the `soundscapy.predictors`
  entry-point group (metric backends under `soundscapy.metrics`, dataset loaders under
  `soundscapy.datasets`). `predictors.available()` discovers them at call time and
  lists requirements; there is no import-time registry.
- **Ship the linear models first** (FR-5.3a): a `LinearPredictor` built from a
  coefficient table shipped as data, with "known location" and "fixed effects only"
  as explicit modes. The probabilistic default (FR-5.3) is chosen after the suitability
  check and implemented against the same protocol. Deep models (SoundAQnet) are a
  separate package `soundscapy-soundaqnet` that depends on torch and fetches weights
  on first use with pooch under their licence (FR-5.4, NFR-5).
- **Graceful degradation is declared, not guessed** (FR-5.5): the input contract marks
  each input required or optional, `predict` raises `MissingInputError` naming the
  missing required inputs, and the `Prediction` records which optional inputs were
  absent.

### 3.7 Datasets

Loaders return `Dataset`s with a shipped `Mapping`, fetched with `pooch` from the public
record with hash checks and licence display on first download (FR-10.1, NFR-5). A small
ISD subset is bundled for tests and tutorials. Each loader is a module with one public
`load()` and a `describe()` returning MOSAIQ catalogue fields (FR-10.3). Harmonising a
new dataset is writing a mapping file and a loader of a few dozen lines, which is the
dataset recipe (FR-13.1).

### 3.8 Provenance and results

Every result record (`ProjectionResult`, `DescriptiveStats`, `SPIResult`,
`AudioResults`, `Prediction`, `Evaluation`) is a frozen dataclass with three common
fields: `settings` (the resolved pydantic model), `versions` (package, Python, and the
wrapped libraries involved) and `created` (UTC). Each has `to_frame()` for its tabular
part and `save(dir)`/`load(dir)` writing tables plus a JSON sidecar, which is how the
project directory stores results (§11).

---

## 4. Module layout

Single distribution, `src/` layout, dependency direction strictly downward within the
list. Each module has one job and a short `__init__` that names its public API.

```
src/soundscapy/
├── __init__.py          # narrative API (lazy, see §5); __version__
├── errors.py            # exception and warning hierarchy
├── provenance.py        # Provenance, Versions, result-record base helpers
├── settings.py          # Settings root model, Profile loading, TOML/JSON read and write
├── instrument/          # Scale, Attribute, Circumplex, AngleSet, tiers, shipped instruments
├── survey/              # Mapping, suggest_mapping, import (csv/xlsx), validate, quality rules,
│                        #   coding, ipsatise, descriptive statistics, who5, source dominance, tests
├── projection/          # the three formulations as pure numpy kernels; project(); ProjectionResult
├── distribution/        # Coordinates, regions, KernelDensity, DensityGrid, SkewNormal, ks2d, shift, overlap
├── spi/                 # Target, targets library, spi(), rankings; search/ behind the optim extra
├── design/              # Brief, Scenario, KPI, evaluate(), intervention levels
├── audio/               # Signal, io, Calibration, weighting, MetricSpec, registry, backends/, batch, manifest
├── predict/             # Predictor protocol, discovery, LinearPredictor, contracts
├── datasets/            # pooch registry, isd/, satp/, araus/, delta/, bundled subset
├── plot/                # circumplex axes, scatter, density, contour, marginals, shift, target, likert, radar, timeseries
├── project/             # Project directory: layout, open/save, manifest, settings and mapping files
├── cli/                 # typer app; one command group per journey
└── _data/               # instruments/*.toml, angles/*.toml, profiles/*.toml, targets/*.toml,
                         #   schemas/*.toml (SSID, REDCap, Qualtrics), models/*.csv, isd_subset/
```

Layer contracts (enforced by import-linter):

| Layer | May import |
|---|---|
| `errors`, `provenance`, `settings`, `instrument` | numpy, pydantic, stdlib |
| `survey`, `projection`, `distribution` | the row above, pandas, scipy, contourpy |
| `spi`, `design`, `predict`, `audio`, `datasets` | the rows above (not each other, except `design` → `spi`, `predict` → `audio` contracts) |
| `plot` | everything above, matplotlib |
| `project`, `cli` | everything |

Draft 0.1 considered a uv workspace of several packages and rejected it for release
and CI cost; that still holds. The seams above are kept so that `audio` or `predict`
could be split into their own distributions later without renaming imports.

Extras (`pyproject.toml` `[project.optional-dependencies]`): `audio` (mosqito,
scikit-maad, soundfile), `optim` (pymoo), `excel` (openpyxl, if pandas' own extra is
not enough), `all`. Optional imports are gated at module import with one
`MissingExtraError("install soundscapy[audio]")`.

---

## 5. API design

Two tiers. The **narrative API** at the top level is about twenty names, enough for
every tutorial and for a GUI's command set. The **module API** is complete and is the
reference documentation. Top-level names are loaded lazily (SPEC 1 style
`__getattr__`) so `import soundscapy` stays fast and the audio stack never imports
unless used.

Journey A, C and E1 in the narrative API, as a tutorial would show them:

```python
import soundscapy as ssp

ds = ssp.read_survey("export.xlsx", mapping="mapping.toml")      # or ssp.suggest_mapping(df, "iso12913-outdoor")
report = ssp.validate(ds)                                         # ValidationReport; nothing dropped
ds = ssp.apply_quality_rules(ds)                                  # adds included, exclusion_reason

proj = ssp.project(ds, profile="iso12913-3-2024")                 # per response; ProjectionResult
coords = proj.coordinates                                         # Coordinates(source="survey")
by_site = coords.group_by("location")                             # dict[str, Coordinates]
ssp.describe(by_site)                                             # tidy table: mean, sd, quantiles, quadrant %
by_site["Camden"].fraction_in(ssp.regions.quadrant("calm"))       # 0.41

ax = ssp.plot.circumplex(coords.circumplex)                       # styled Axes
ssp.plot.density(by_site["Camden"], fraction=0.5, ax=ax, label="Camden")
ssp.plot.scatter(by_site["Camden"], ax=ax)

target = ssp.targets.load("iso12913-4-example6-tranquil")         # or ssp.SkewNormal(xi=..., omega=..., alpha=...)
score = ssp.spi(by_site["Camden"], target, n_samples=1000, rng=0) # SPIResult(value=56, ...)

brief = ssp.Brief(ambition="…", target=target, kpi=ssp.KPI(spi_min=70), ...)
result = ssp.evaluate(brief, baseline=by_site["Camden"], scenarios={"post": post_coords})
result.table                                                      # SPI, shift, % calm, meets_kpi per scenario
```

Conventions:

- Functions take and return DataFrames or records; they never mutate inputs. Derived
  columns are added to a returned copy with documented names.
- `settings=` or `profile=` keyword arguments accept a model, a name, or a path; the
  resolved model is what the result records.
- Randomness always takes `rng: int | np.random.Generator`; no function reads global
  numpy state.
- Plot functions take `ax: Axes | None`, return the `Axes`, never create a figure
  unless `ax is None`, never call `show()`, and pass unknown keyword arguments to the
  matplotlib artist they draw (FR-8.3).
- Every record has `to_frame()` where tabular, `save()`/`load()`, and `settings`.
- Exceptions derive from `SoundscapyError`; warnings from `SoundscapyWarning`. Each
  specific condition has its own class so tests and GUIs can catch one thing.
- A pandas accessor (`df.ssp.project()`) is rejected for 1.0: it is magic, hard to type
  with ty, and hides the mapping argument.
- Deprecations: `DeprecationWarning` for two minor releases, a changelog entry, and a
  "method version" bump when a scientific result changes beyond tolerance (NFR-4,
  FR-13.4).

---

## 6. Tooling

The house-style skill (`.claude/skills/house-style/SKILL.md`) fixes most of this; it is
restated here only where the plan relies on it or adds to it.

| Concern | Choice | Why, and what it beat |
|---|---|---|
| Packaging | uv, `uv_build`, `uv.lock` committed, `src/` layout, PEP 639 licence metadata | House style. |
| Python | 3.11 and later, three OS in CI | Owner decision (NFR-2); Windows first-class. |
| Lint and format | ruff, `select = ["ALL"]`, numpy docstrings | House style. |
| Types | ty; `py.typed` shipped | House style. Pydantic models and plain functions type well; pandas return types are `pd.DataFrame` with documented columns, not typed frames. |
| Settings and specs | pydantic v2 frozen models | JSON schema for free, which is what a GUI form and the settings-file docs are generated from. Beat dataclasses + hand-written loaders. |
| Settings files | TOML written by `tomli-w`, read by `tomllib` | Human-editable with comments, standard library reader, no YAML dependency. JSON for machine-written manifests and result sidecars. See §15. |
| Tables | pandas as the public contract; numpy inside kernels | Researchers, seaborn and statsmodels all speak pandas. polars or narwhals would add an abstraction tax before there is demand; kernels on numpy keep the door open. |
| Numerics | numpy, scipy (stats, signal, optimize), contourpy | Standard. |
| Statistics | scipy.stats and statsmodels; thin `tidy()` helpers returning statistic, df or N, p, effect size | FR-2.6, FR-2.6a. pingouin was considered and rejected as a dependency for a few effect sizes that are ten lines each. |
| Audio | scipy.io.wavfile core; mosqito, scikit-maad, soundfile (and phonometry on 3.13+) in the `audio` extra | §3.5 and Appendix C. |
| Optimisation | pymoo in the `optim` extra | FR-6.5 only. |
| Data fetching | pooch | Standard in scientific Python; hash checks, cache, licence message. |
| Plotting | matplotlib; seaborn only where it saves real code (stacked bars, palettes) | §10. |
| CLI | typer | Well known to volunteers, typed, generates help; beat argparse (verbose for sub-commands) and cyclopts (less known). |
| Tests | pytest, hypothesis, pytest-cov, pytest-xdist; `filterwarnings = error` | House style plus §8. |
| Architecture checks | import-linter contracts in `task lint` | §4. |
| Docs | great-docs; Quarto `.qmd` tutorials; Diátaxis layout | House style; FR-12. |
| CI | GitHub Actions running `task` targets; 3 OS × supported Pythons; docs executed; nightly slow audio and dataset tests; release on tag by trusted publishing | NFR-8. |
| Changelog and versioning | SemVer from 1.0; Keep-a-Changelog with a "Scientific changes" section; `CITATION.cff`; Zenodo DOI per release | FR-13.4, NFR-11. |

---

## 7. Where each programming style is used

This answers the "where OOP, where functional, where concurrency" question directly.

| Style | Used for | Not used for |
|---|---|---|
| Pure functions over arrays and frames | Projection kernels, statistics, KDE geometry, KS distance, SPI, shift and overlap, quality rules, calibration gain, batch work units | Anything that needs to remember state between calls |
| Frozen value objects (pydantic) | Scale, Circumplex, AngleSet, Profile, Mapping, Calibration, MetricSpec, Target, Brief, Scenario, all settings | Carrying numpy arrays or frames (pydantic serialisation would fight them) |
| Frozen result records (dataclasses) | Coordinates, KernelDensity, every `*Result` | Mutable accumulation |
| Protocols | MetricBackend, Predictor, DatasetLoader, ProgressReporter | Internal code that has one implementation |
| One stateful class | `Project` (an open project directory with explicit `save`) | Anything else |
| Process pool | Audio batches (file-level) and target search (candidate evaluation) | Survey analysis, KDE, SPI, which are vectorised and interactive at 10,000 responses |

---

## 8. Validation, tests and regression baselines

Four kinds of test, each with a home and a marker, so a contributor knows what to run
and what a failure means (FR-13.2, NFR-4, NFR-8).

1. **Unit and property tests** (`tests/<module>/`, default run). Hypothesis invariants:
   projected coordinates lie in [−1, 1]² for any angle set and scale; a neutral
   response maps to the origin; the canonical formulation equals the 2019 formula at
   45° angles to rounding; direct → centred → direct round trips when admissible;
   `ks2d(a, a) == 0` and symmetry; `fraction_in` over complementary regions sums to 1;
   calibration gain is invertible.
2. **Literature fixtures** (`tests/science/`, default run, each test docstring cites the
   source and the significant figures that set its tolerance). The ISO worked
   examples supplied by the owner; Part 4 Example 6 (SPI 56); the 2024 paper's tables
   at the stated seed and sample size; the 2022 paper's figures as geometry (contour
   vertices and proportions, not pixels); the 2021 models predicting the published
   location means from the ISD feature table.
3. **Cross-implementation references** (`tests/reference/`, marked `reference`, run in
   CI nightly and on demand). Skew-normal fitting and sampling against committed R
   `sn` outputs (generated once by a script kept beside them); ISO 532-1 reference
   signals against MoSQITo (to catch a wrapped-library regression); selected metrics
   against ArtemiS tables where shareable (FR-4.10).
4. **Golden pipelines** (`tests/regression/`, marked `regression`). ISD subset →
   validate → project → describe → SPI, compared to committed baseline tables with
   per-column tolerances by significant figure. A drift inside tolerance passes; a
   drift outside fails and must be explained in the changelog with a method version
   bump.

Plot tests assert on returned geometry and on artist data (offsets, contour vertices),
not pixels; two or three layout smoke tests may use image comparison. Test templates for
the four extension recipes live under `tests/templates/` and are copied by the recipe.

---

## 9. Concurrency, progress and cancellation

- **Only file-level parallelism, only where it pays.** `audio.analyse` runs work
  units through a `concurrent.futures` executor chosen by the caller (`"processes"`
  default, `"threads"`, `"serial"`, or any executor object, which is how a GUI or
  joblib plugs in). Work functions are importable top-level functions so Windows
  `spawn` works. Nested parallelism inside wrapped libraries is disabled where they
  expose a switch, to avoid oversubscription.
- **Resumable by construction.** The per-file cache is the resume mechanism; there is
  no separate checkpoint format. A manifest lists what was computed with which
  versions (FR-4.8, FR-11.4).
- **Progress and cancellation are protocols**, not libraries: `ProgressReporter`
  with `start(total)`, `advance(path)`, `finish()`; `CancelToken` with `cancelled()`.
  The library checks the token between work units (FR-14.3). The CLI binds a rich or
  tqdm bar; a GUI binds its own; tests bind a recorder.
- **Target search** (FR-6.5) uses the same two protocols around pymoo's evaluation
  loop.
- **Everything else is vectorised numpy.** KDE on a 200 × 200 grid for 10,000 points is
  fast enough; an FFT-based KDE is a possible later optimisation, not a design item
  (NFR-3).

---

## 10. Visualisation without a plotting package

The lesson of the 0.x `ISOPlot` class (a layer hierarchy that re-implements figure
management and mixes KDE and SPI computation with drawing) is applied as three rules.

1. **Geometry lives in `distribution`** and is tested numerically: `DensityGrid`,
   `mass_contour`, `decile_contours`, `marginals`, `fraction_in`, `shift`. A GUI, a
   plotly renderer or a report can consume them without matplotlib (FR-8.4).
2. **Drawing functions are small and matplotlib-native.** `plot.circumplex(circumplex,
   ax=None)` styles the frame (square [−1, 1]² axes, zero lines, quadrant labels and
   axis titles from the instrument, in the chosen language). `plot.scatter`,
   `plot.density`, `plot.contour`, `plot.marginals`, `plot.shift`, `plot.target`,
   `plot.spi_annotation`, `plot.likert`, `plot.radar`, `plot.timeseries`. Each draws on
   an `Axes`, returns it, and forwards `**kwargs` to the artist. Faceting is
   `plt.subplots` or seaborn's `FacetGrid` with the user's own code; the package does
   not own subplot layout. A `plot.by_group(coords_by_group, layers=[...])` convenience
   covers the common "one panel per location" case in a few lines.
3. **Styling is a matplotlib style sheet** shipped as data with a colour-blind-safe
   palette (FR-8.3, NFR-10); users override with matplotlib's own `rcParams` and style
   context managers, not a bespoke theme system.

A declarative `PlotSpec` for the GUI is deliberately *not* designed now. The GUI needs
the geometry and the settings that produced it, both of which exist; how it renders
them is its own concern (FR-14.7). Likert stacked bars and radar plots are written
directly (tens of lines each) rather than depending on lightly maintained packages.

---

## 11. Persistence: the project directory and settings files

The project directory (FR-1.7, FR-14.4) is a plain folder that the library, CLI and
GUI all read:

```
my-study/
├── project.toml            # name, package version, profile, pointers below
├── mapping.toml            # Mapping (FR-1.3), saved and reused across exports
├── settings.toml           # resolved analysis settings (angles, bandwidth, seeds, metric set …)
├── data/
│   ├── responses.csv       # the user's table, untouched
│   ├── groups.csv, sessions.csv, locations.csv      # optional
│   └── recordings.toml     # RecordingIndex (M3): paths, formats, hashes, calibration refs
├── results/
│   ├── projection/         # coordinates.csv + projection.json (settings, versions)
│   ├── audio/              # one table per recording + manifest.json + cache/
│   ├── spi/                # spi.csv + spi.json
│   └── evaluation/
└── manifest.json           # hashes of inputs, versions, timestamps for the whole project
```

- Tables are CSV with a JSON sidecar declaring dtypes and units, so the folder opens
  in Excel, diffs in git and needs no pyarrow. Parquet is an option for large audio
  results behind the `audio` extra. See §15 for the question this raises.
- `Project.open(path)`, `.save()`, `.results["projection"]` is the one stateful class;
  everything it holds is the frozen records of §3.8.
- Settings files (FR-11.3) are the TOML form of the pydantic settings model, with the
  JSON schema published in the docs so a GUI can generate forms and validate edits.

---

## 12. GUI readiness checklist

The core meets FR-14 by construction if the following hold, each of which has a test:

- No function in `src/` calls `show`, `print`, `input` or writes a file except
  `project` and explicit `save`/`export` methods.
- Every settings model and every result record round-trips through its text form.
- `audio.analyse`, `spi.search` and dataset downloads accept a `ProgressReporter`
  and a `CancelToken`.
- Each CLI command for journeys A, C, E1 (M1–M2), B (M3) and E2 (M4) reads and writes
  only the project directory, so "can the CLI do it?" is the GUI-readiness test.
- JSON schemas for `Mapping`, `Settings`, `Target` and `Brief` are published, which is
  what a form-driven GUI needs first.

Whether the GUI runs in-process (Python, driving the library directly) or
out-of-process (driving the CLI and the project directory) is left open, as the PRD
requires; both are served.

---

## 13. Where this plan pushes back

Points where the plan disagrees with draft 0.1, interprets the PRD, or asks the
literature a question it leaves open. None changes the science.

1. **No pandera, no schema DSL.** The mapping is the schema (§3.2). Draft 0.1's
   pandera layer would have duplicated it.
2. **Positions over names** (§3.1) reverses draft 0.1's "names, never positions". The
   PRD's generalised circumplex makes positions the stable key.
3. **No `PlotSpec` yet** (§10). Draft 0.1 proposed one for the GUI; the PRD fixes no GUI
   design, and geometry plus settings already give a GUI what it needs.
4. **CSV, not parquet, as the project table format** (§11), to keep the install light
   and the folder human-readable. This is a real trade-off for thousands of audio
   rows; see §15.
5. **SPI has no default sample size in the core function** (§3.4). The paper's value
   is supplied by the profile. This keeps "the number depends on n and the seed"
   visible at the call site, which the PRD's reproducibility requirements want.
6. **KDE bandwidth is a recorded setting with a documented default**, because the
   papers do not state one and it changes the 50% contour. A sensitivity helper is
   provided so a user can show their contour is not an artefact of the default.
7. **The audio `MetricBackend` surfaces licence and version**, so the generated metric
   table can show a consultant exactly what computed each number and under what
   terms. This is cheap now and hard to retrofit.
8. **Deep models are separate packages, never an extra of the core** (§3.6), to keep
   torch out of the course install (NFR-1) and the weights' licences out of the core's.
9. **Mapping suggestion is data-driven** (§3.2): known export schemas are files, so a
   course tutor can add a new survey platform's layout without a release.
10. **`Dataset` has almost no behaviour.** Draft 0.1 and the 0.x package both grew
    container classes that accumulated methods; this plan keeps the container as a
    record and puts behaviour in functions, so the public surface stays small.

---

## 14. Delivery sequence

Mapped to the PRD's milestones (§9). Each phase ends with `task check` green on three
operating systems, executed tutorials, and regression baselines committed. The first
phase is the toolchain; the house-style skill specifies it. Architecture decision
records are written as the decisions are taken, under `design/adr/`.

| Phase | Deliverables | Exit criterion |
|---|---|---|
| 0. Toolchain | `pyproject.toml`, `Taskfile.yml`, prek, CI, docs skeleton, import-linter contracts, ADR-0001 (functional core and records), ADR-0002 (positions and mappings), ADR-0003 (Coordinates as the central type), ADR-0004 (settings files and project directory) | `task check` and `task docs` green on an empty package |
| M1. Survey analysis | `instrument`, `survey`, `projection`, `distribution` (Coordinates, regions, KDE, ks2d, SkewNormal), `plot` basics, `datasets.isd` with bundled subset, `project` minimal directory, CLI for journey A, recipes for plot type and dataset loader, plugin discovery, tutorial A, reproduction of the 2022 paper | Journey A done-when; 2022 figures regenerate as geometry within tolerance |
| M2. SPI and design (1.0) | `spi` with targets library and bootstrap, `design` with Brief/Scenario/evaluate, CLI for C and E1, tutorials C and E, course acceptance cases 2 and 3, reproduction of the 2024 paper, versioning policy, 1.0 release | Part 4 Example 6 reproduces; course exercises run unchanged |
| M3. Audio | `audio` (Signal, I/O, calibration, built-in level metrics, MoSQITo and scikit-maad backends, batch runner, manifest), CLI for B, metric recipe, tutorial B, ArtemiS comparison page | Journey B done-when; ISO 532-1 reference signals pass; a 1,000-file batch resumes |
| M4. Prediction | `predict` (protocol, discovery, LinearPredictor, default probabilistic predictor after suitability check), `soundscapy-soundaqnet` plugin, E2 in `design`, CLI for E2, predictor recipe, reproduction of the 2021 paper | Journey D and E2 done-when |
| M5. Ecosystem | SATP, ARAUS, DeLTA loaders with mappings; MOSAIQ import/export; target search behind `optim`; contributor trial of each recipe | Journeys F, G, L done-when |

---

## 15. Decisions for the owner

Ordered by how much a wrong guess would cost. Each has the plan's recommendation.

1. **Copyleft, concretely.** The best-verified Python implementation of the
   ECMA-418-2 metrics (`sottek-hearing-model`) is AGPL/GPL-3, as are MetaSona,
   refmap-psychoacoustics and rota-order, and `soundfile` links LGPL libsndfile.
   Recommendation: no GPL or AGPL code in the core or in any extra of the
   `soundscapy` distribution; a GPL backend may exist only as a separately published
   plugin (`soundscapy-shm`, say) that a user installs knowingly; LGPL through dynamic
   linking is acceptable in the `audio` extra. This decides the M3 metric list. (PRD
   open item 2.)
1a. **phonometry and the Python floor.** phonometry is the only permissive package
   that fills the fluctuation-strength and ECMA-418-2 tonality gaps, and it needs
   Python 3.13. Options: (i) keep the 3.11 floor and ship phonometry as a
   3.13-only optional backend (recommended; the gaps are then documented per Python
   version); (ii) raise the floor to 3.13 for the `audio` extra only; (iii) raise the
   package floor to 3.13, reversing the NFR-2 decision; (iv) exclude phonometry
   until it is independently validated. Whichever is chosen, the ArtemiS comparison
   (FR-4.10) is run per backend so users see how each agrees.
2. **Project table format.** CSV with sidecars (recommended, §11) or parquet
   (adds pyarrow, roughly 100 MB on Windows, but scales to large audio results and
   keeps dtypes natively). A middle path is CSV by default with parquet behind the
   `audio` extra for audio results.
3. **Settings file format.** TOML (recommended, standard-library reader, comments) or
   YAML (more familiar to some researchers, needs a dependency, no comment
   round-trip either way). JSON is used for machine-written records regardless.
4. **SPI uncertainty** (PRD open item 1). Recommendation: always report the
   sampling SD from repeated target draws; offer a bootstrap over the test set as an
   option off by default; the 2024 paper's sample size is the profile default.
5. **Public table contract.** pandas only for 1.0 (recommended) or dataframe-agnostic
   via narwhals from the start. The kernels are numpy either way.
6. **CLI framework.** typer (recommended) or argparse to avoid a dependency.
7. **Governance** (PRD open item 3): who cuts a release, who reviews a change that
   moves a regression baseline, and the cadence. The plan assumes a "Scientific
   changes" changelog section and a method version bump need one maintainer's review
   plus a passing reference test, but that is a policy, not a design.

### Inputs the plan still needs from the owner (unchanged from PRD §10.3)

The ISD language coverage against the SATP v1.5 tiers; the ISD record versions used by
each reproduced paper; machine-readable fixtures for Part 4 Example 6 and the 2024
paper's targets; whether redistributable audio exists; and the features the 2021
deployable models use.

---

## Appendix A. Exception and warning hierarchy

```
SoundscapyError
├── MappingError            column missing, duplicate binding, unknown position
├── ScaleError              value outside declared scale, wrong coding
├── LanguageError
│   ├── UncoveredLanguageError      no angle set for a response's language (override: allow_equal_angles)
│   └── LowConfidenceLanguageError  Low-tier language refused (override: allow_low_confidence)
├── SmallSampleError        density requested for N < 30 (override: allow_small_sample)
├── NotAdmissibleError      centred parameters outside the admissible region
├── CircumplexMismatchError target or coordinates on a different circumplex
├── UncalibratedSignalError metric needs Pa, signal has no calibration
├── MissingInputError       predictor's required input absent
├── MissingExtraError       optional dependency not installed, names the extra
└── ProjectError            project directory malformed or from an unsupported version

SoundscapyWarning
├── MediumConfidenceWarning Medium-tier language projected
├── SmallSampleWarning      density computed with N < 30 under override
├── OptionalInputWarning    predictor ran without an optional input
└── LegacyFormulationWarning 2019 or Aletta-2024 formulation selected
```

## Appendix B. Extension recipes (FR-13.1), in one line each

- **Plot type** (M1): a function `plot.<name>(data, *, ax=None, **kwargs) -> Axes` over
  geometry from `distribution`; test asserts artist data; docs page from the docstring.
- **Dataset loader** (M1): a `_data/schemas/<name>.toml` mapping plus a module with
  `load()` and `describe()` using pooch; test against the bundled fixture.
- **Metric backend** (M3): a class satisfying `MetricBackend` registered under the
  `soundscapy.metrics` entry point; test against a reference signal with a cited
  tolerance.
- **Predictor** (M4): a class satisfying `Predictor` registered under
  `soundscapy.predictors`; test that `predict` honours the input contract and labels
  its output.

## Appendix C. Evaluation of the LISQM candidates

Every project in the LISQM list (https://github.com/ZhenxianLi/LISQM) with a Python
interface, checked on 2026-10-30 against the hard requirements for the Python
environment: installable from PyPI as wheels on Windows, macOS and Linux with no
compiler, no MATLAB, no R; Python 3.11 and later; no copyleft licence inside the
`soundscapy` distribution (NFR-1, NFR-2, NFR-7). MATLAB, Julia, Rust-only, C#, Pure
Data and C-only entries (SQAT, AMT, AARAE, ITA-Toolbox, PsySound3, PsychoacousticMetrics.jl,
Kirin Hypha, Mosqito.NET, SoundPalette, zwickerloudness, epnl, and others) fail the
first requirement and are not tabulated. LISQM computes nothing itself; coverage
claims are the projects' own.

| Project | Licence | PyPI, Python, deps | Table D.1 and research coverage | Status (LISQM, this check) | Verdict |
|---|---|---|---|---|---|
| MoSQITo 1.2.1 | Apache-2.0 | yes; `>=3.5`; numpy, scipy, pyuff (matplotlib undeclared for ECMA roughness) | ISO 532-1 ZWST/ZWTV, DIN 45692 sharpness, Daniel–Weber roughness, ECMA-418-2 loudness and roughness (pre-2025 edition), ECMA-74 PR/TNR, SII | established; no release since 2024-04; imports and runs on 3.11 and 3.13 with numpy 2 | **Default backend**, pinned |
| phonometry 3.3.0 | MIT | yes; **`>=3.13`**; numpy, scipy | ISO 532-1/2/3, ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength; DIN 45692; Osses FS; PA; ECMA-418-1; ISO 1996-2; plus levels, weighting, octave bands | newly released 2026; one maintainer; 1,893 self-reported conformance checks in CI; DOI | **Optional backend on 3.13+** (owner decision 1a) |
| scikit-maad 1.5.2 | BSD-3 | yes; `>=3.9`; numpy, scipy, pandas, matplotlib, scikit-image, pywavelets | 50+ ecoacoustic indices, SPL | active, maintained | **Backend for FR-4.5 ecoacoustics**, `audio` extra |
| acoustic-toolbox 0.2.2 | BSD-3 | yes; `>=3.11`; numba, pysoundfile, pyoctaveband, matplotlib, pandas | weighting, Leq, octave bands, ISO 1996-2 tonal audibility | active fork of python-acoustics (archived) | Not used: level metrics are built in without numba; revisit for ISO 1996-2 |
| sottek-hearing-model 0.1.14 | AGPL-3 (PyPI), GPL-3 (GitHub) | yes; `>=3.11`; numpy, scipy, bottleneck, matplotlib, **PyQt6**, tqdm | ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength | active; translated from the SQAT authors' verified MATLAB | Excluded by licence; candidate for a separate plugin |
| MetaSona 0.2.2 | GPL-3 combined (Apache/BSD/MIT parts) | yes, wheels for all three OS; `>=3.9,<3.15`; numpy, scipy | Zwicker loudness, DIN sharpness, D–W roughness, Aures tonality, ECMA-418-2 (2025) loudness, roughness, tonality; 15–70× faster than MoSQITo | new; authors call roughness and tonality experimental | Excluded by licence and maturity |
| pySQAT | **none stated** | no; `requirements.txt` with PySide6 | SQAT's set: ISO 532-1, ECMA-418-2, DIN 45692, D–W, Osses FS, Aures, PA, EPNL | developing; MATLAB-vs-Python validation workflow, no results published | Unusable without a licence; watch |
| ecma418-torch | MIT | no; requires torch | ECMA-418-2 (2025) roughness, fluctuation strength | one commit | Excluded: torch, unproven |
| iso532-1-rs | Apache-2.0 | no wheels; maturin source build | ISO 532-1 ZWST/ZWTV, MoSQITo-parity tests | new; author requests review | Excluded: no wheels; a possible fast backend later |
| refmap-psychoacoustics | GPL-3 | no | aural detectability (Python); SHM metrics via sottek-hearing-model | active | Excluded by licence |
| rota-order 0.7.0 | GPL-3 | yes | ECMA-418-1 TNR and PR | active | Excluded by licence; MoSQITo covers PR/TNR |
| torch_amt 0.2.0 | GPL-3 | yes; torch | Moore–Glasberg time-varying loudness | active | Excluded; not in Table D.1 |
| LoudnessModel | BSD-2 | no release | Moore et al. 2018 time-varying loudness | dormant | Not needed |
| PyDSM 0.15.2 | GPL-3 | yes | ISO 226 contours | dormant | Not needed |
| PsychoacousticParametersMeasurer | none stated | no | Zwicker set from the book | legacy, 2021 | Unusable |
| NumpyLibforPsychoAcoustic | none stated | no | Moore–Glasberg loudness | new | Unusable without a licence |
| MoSQITo-FDP, python-acoustics, SUAVE, deeuu/loudness | Apache / BSD-3 / LGPL-2.1 / GPL-3 | archived or dormant | early ECMA-418-2, ISO 1996-2, EPNL, loudness | legacy | Not used |
| PsyTools | unknown | code not public | unknown | unknown | Not evaluable |

Gaps that no admissible package fills on Python 3.11 and 3.12: fluctuation strength
(Osses or ECMA-418-2) and ECMA-418-2 tonality. Aures tonality exists only in an unmerged
MoSQITo pull request and in pySQAT. These are documented as gaps per FR-4.4, and the
metric-backend registry means a later package, a licence change in pySQAT, or a
MoSQITo release closes a gap by adding one backend class.
