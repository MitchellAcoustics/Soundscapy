# Architecture plan for the soundscapy rewrite

Status: **draft 0.2 for review** (2026-10-30). This draft has been rewritten from PRD
draft 0.5 (`design/PRD.md`) and supersedes draft 0.1, which predated the PRD. A
fresh-eyes builder check was run on it; its answerable questions are folded in and
marked "my call". Owner decisions taken on the first review (2026-10-30): no copyleft
restriction on dependencies; dataframe-agnostic through narwhals; click for the CLI;
progress visible in notebooks as well as the CLI. On the second review, every
remaining question in §15 was decided; the plan is now ready for a spec and an
implementation plan for Phase 0 and M1. It is
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

1. **Functional core.** Computation is pure functions over numpy arrays and
   dataframes. No function mutates its input, reads global state, writes files, prints
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
4. **We do not own the user's DataFrame.** Survey data stay in the user's own
   dataframe library (pandas, polars, pyarrow or any other narwhals-supported eager
   frame) with the user's column names, and come back in the same library. A
   `Mapping` says which column is which construct and on what scale. The package
   validates through the mapping and adds derived columns with documented names.
   Rules out renaming or reordering user columns, wrapping rows in objects, and
   pandas-specific code outside the two boundaries named in §6.
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
   goes through a `ProgressReporter` protocol and cancellation through a token; the
   package ships one default reporter built on `tqdm.auto`, which renders a text bar
   in a terminal and a widget in a notebook, so notebook users see progress without
   wiring anything. Rules out any other output from the library, and rules out rich,
   which the CLI alone may use.
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

class AngleSelection(BaseModel, frozen=True):
    """How angles are resolved per response (FR-2.4): equal angles, or a default set with
    per-language overrides that may name a shipped version or a user-supplied file."""
    default: Literal["equal"] | str         # "equal", or an AngleSet name such as "satp-v1.5"
    per_language: Mapping[str, str] = {}    # ISO 639-3 -> AngleSet name or path to a user set
    allow_low_confidence: bool = False      # FR-2.4a override, recorded in the result
    on_uncovered: Literal["error", "equal"] = "error"   # FR-2.4 override

class Profile(BaseModel, frozen=True):
    """A named bundle of edition-specific settings (FR-2.9)."""
    name: str                               # "iso12913-3-2019", "iso12913-3-2024"
    formulation: Literal["canonical", "iso2019", "aletta2024"]
    angles: AngleSelection
    dispersion: Literal["range", "iqr"]
    missing_items: Literal["exclude", "renormalise"] = "exclude"   # "renormalise" canonical only; §3.2
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
- **Angles resolve per response.** `AngleSelection` is applied to each response's
  language: the per-language override first, then the default set, then the
  `on_uncovered` rule. The projection result records, per response, which set and
  version supplied its angles, so a nine-language dataset using two shipped versions
  and one user file is representable and auditable. Deriving angle sets is out of
  scope (FR-2.4a); a user file is the same TOML format as a shipped set.
- **Angle sets and profiles are data files, not code.** They ship under
  `soundscapy/_data/` as TOML with provenance, and a user-supplied set is the same
  file format. Adding the next SATP version is a data change plus a test.
- **Profiles are thin.** A profile resolves to concrete settings and is recorded by
  name *and* by resolved values in every result, so a later change to what a profile
  means cannot silently change old results.

### 3.2 Surveys: mappings, tables, quality

```python
class Coding(BaseModel, frozen=True):
    """How raw export values become scale scores (FR-2.1): REDCap 0..4, text anchors,
    reversed parts. Identity when the export already holds Table A.1 scores."""
    values: Mapping[str | int, int] | None = None   # raw value -> score; None = numeric as is
    offset: int = 0                         # e.g. +1 for 0-based exports
    reverse: bool = False                   # part coded in the opposite direction

class Mapping(BaseModel, frozen=True):
    """Which user column means which construct (FR-1.3), persisted as a file (FR-1.3a)."""
    instrument: str                         # circumplex name
    scale: Scale | None = None              # override of the instrument's scale
    attributes: Mapping[str, str]           # position -> column name
    coding: Coding | Mapping[str, Coding] = Coding()   # one for all attributes, or per position
    language: str | LanguageColumn          # one language, or {"column": name}
    ids: Ids                                # column names for the roles response, group, session,
                                            #   location (SSID) and participant, stimulus (listening
                                            #   experiments); any subset
    source_dominance: Mapping[str, str] = {}   # construct -> column
    overall_quality: str | None = None
    appropriateness: str | None = None
    who5: tuple[str, ...] = ()
    extra: Mapping[str, ColumnRole] = {}    # demographics, free text, flags, carried through
    provenance: Provenance

class Dataset:                              # frozen dataclass, plain container
    responses: IntoFrame                    # any narwhals-supported eager frame; returned in kind
    mapping: Mapping
    groups: IntoFrame | None = None         # one row per group; recording path lives here
    sessions: IntoFrame | None = None
    locations: IntoFrame | None = None
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
  `(responses, mapping) -> boolean mask` with a name and a reason string. `apply_rules`
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
- **Coding lives in the mapping, not in a separate step.** Exports arrive as 0..4,
  1..5, text anchors or reversed parts; `Coding` on the mapping converts them to
  Table A.1 scores at import, and `suggest_mapping` proposes it from the observed
  values. This keeps the mapping the single persisted description of an instrument's
  export (FR-1.3a). My call; the canonical layout (FR-1.2) uses the identity coding.
- **Id roles are open, not SSID-only.** `Ids` carries the SSID roles and the
  listening-experiment roles (participant, stimulus), so SATP and ARAUS (M5) and
  ipsatisation (below) need no schema change after 1.0. My call.
- **Ipsatisation** (FR-2.4b) is a pure transform `ipsatise(responses, mapping) ->
  frame` grouping by the participant id, returning a new frame, recorded in
  provenance when used; it is never a default.
- **Missing attribute items.** The canonical formulation normalises over all items,
  so a response with a missing item cannot be projected without a rule. The default
  is `missing_items="exclude"`: the coordinate is NaN and the response carries the
  reason `incomplete_attributes`, so it counts in item-level statistics but not in
  coordinate statistics. `"renormalise"` projects over the present items with
  Σ|cos θ_i| and Σ|sin θ_i| recomputed over those items, which keeps the canonical
  formulation's guarantees (coordinates in [−1, 1], neutral at the origin) and is
  recorded in provenance; it is offered only for the canonical formulation, because
  the 2019 fixed denominator and the Aletta 2024 variant define no such rule, so
  under those formulations a missing item always excludes the response. Owner
  decision, 2026-10-30.

### 3.3 Coordinates and distributions: the central abstractions

```python
@dataclass(frozen=True)
class Coordinates:
    """A set of points in one circumplex, with provenance (FR-2.5, FR-5.6, FR-5.9)."""
    xy: np.ndarray                          # (n, 2), float64, in [-1, 1]^2
    circumplex: Circumplex
    source: Literal["survey", "prediction", "sample"]
    labels: IntoFrame | None                # optional per-point columns: ids, grouping keys, in the input's library
    settings: ProjectionSettings | PredictorRef | TargetRef   # what produced them; the *Ref
                                            #   records live in provenance.py so distribution never imports spi
    # methods: subset(mask), group_by(key) -> dict[key, Coordinates], mean(), sd(), quantiles(),
    #          fraction_in(region) ("percent likely to perceive"), to_frame()

@dataclass(frozen=True)
class KernelDensity:
    """A KDE view of Coordinates with an explicit bandwidth (FR-3.1)."""
    coordinates: Coordinates
    bandwidth: float | Literal["scott", "silverman"]   # scipy gaussian_kde bw_method: a scalar factor
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
- **Regions** (quadrants, half-planes, user polygons) are value objects with a numpy
  point-in-polygon test (a few lines of ray casting), so `distribution` imports
  neither matplotlib nor shapely. Quadrant regions are defined from the circumplex
  axes so they work for any instrument.
- **KDE is a view, not a transform.** The estimator is `scipy.stats.gaussian_kde`
  (Gaussian kernel, bandwidth a scalar factor times the data covariance), the same
  family seaborn uses, so there is exactly one KDE definition in the package
  (principle 7). Bandwidth is explicit and recorded; the default is Scott's rule,
  stated in every result, with a sensitivity helper. The "50th
  percentile contour" is computed from the cumulative mass of the grid so its meaning
  is unambiguous and the fraction is a parameter (FR-3.1). Contour polygons come from
  `contourpy`, which matplotlib already depends on, so geometry needs no figure.
- **SkewNormal is implemented here, in numpy and scipy, because nothing suitable
  exists in Python.** A research pass on 2026-10-30 checked the assumption: scipy's
  `skewnorm` is univariate only, and the request for a multivariate version (scipy
  issue 14870) has been open since 2021; no package on PyPI provides a multivariate
  skew-normal (sixteen plausible names probed); `mvskew` (MIT, PyPI) is samplers
  only; the one GitHub implementation (spc90/multivariate_skew_distributions) has a
  density and sampler but no fitting, no centred parameters and no licence; PyMC and
  TensorFlow Probability offer the univariate case only. Azzalini's own site lists
  Python code for the univariate case alone.
- **What the 0.x branches already did, and what is folded in.** Two 0.x branches
  explored this (reviewed with their history on 2026-10-30):
  - `scipy-msn` (April 2025, one WIP commit on top of the rpy2 SPI work): a 1,800-line
    scipy-style `multivariate_skew_normal_gen` with pdf, cdf, rvs, dp2cp, cp2dp and
    MLE, plus a plan to contribute it to scipy. Run in this session against scipy
    1.17: it imports private scipy internals (`_PSD.whiten`, `pinvh(rcond=)`) that
    have since been removed, so 13 of 15 tests fail on import or call; after patching
    those, `dp2cp` matches R `sn` to 2e-9 and `rvs` reproduces the R mean and
    covariance, but `cp2dp` is not the inverse of `dp2cp` (round-trip error 6e-2)
    and `fit` returns the starting values with alpha = 0 on a strongly skewed sample.
    Its author labelled it "failing first passes". **Not adopted as code.** Its test
    design (pdf/logpdf consistency, cdf equal to the multivariate normal at alpha = 0,
    sample moments, dp/cp round trip, MLE recovery) is adopted as the test list, and
    its sampler is the same conditioning construction the plan uses.
  - `157-remove-r-sn-dependency-for-spi` (May–June 2026, two commits): `_mvskew.py`,
    a faithful Python translation of `sn`'s internal `msn.dp2cp` and `mst.dp2cp`
    (with `zeta`, `b(nu)`, cumulants and Mardia measures) wired into `dp2cp` for the
    SN, ESN, ST and SC families, with a committed R-derived baseline. Run in this
    session on scipy 1.17 and numpy 2: all 13 pure tests pass and the SN result
    matches R to 2e-9. It stops there: `cp2dp`, sampling and fitting still go through
    rpy2. **Adopted**: the SN and ESN part of `msn_dp2cp` becomes
    `distribution/skewnormal.py`'s `dp2cp` after cleaning (drop `np.matrix`, drop the
    skew-t and skew-Cauchy paths the PRD does not need, add types), together with its
    R baseline values as the first reference fixture. This fits the planned design
    without change: `SkewNormal` stays a frozen value object and the conversions stay
    pure functions.
  - **Is the planned implementation better than either?** Yes, because it is the only
    one that is complete: `dmsn` (pdf and log-pdf), `rmsn` (sampling by the
    conditioning construction, with an optional rejection step to truncate to the
    circumplex square, which the 0.x R wrapper offered as `sample_mtsn`), `dp2cp`
    (from the 157 port), `cp2dp` by the closed-form inverse for the SN family
    (marginal skewness to delta, then alpha from the scaled Omega; Azzalini and
    Capitanio 1999, `sn` `msn.cp2dp`) with the admissibility check (Arellano-Valle and
    Azzalini 2008), and maximum-likelihood fitting on the centred parameters with
    `scipy.optimize` from method-of-moments starting values, as `selm` does. It is a
    few hundred lines against the 1,800 of the scipy-style draft because it does not
    carry scipy's generic multivariate framework, and it uses no private scipy API.
    Every function is validated against `sn` outputs committed as fixtures (NFR-9,
    FR-3.4), generated once by an R script kept beside them; the 157 branch shows the
    pattern. R is not a dependency. Contributing to scipy later is possible because
    the API mirrors scipy's, but it is not on the path to 1.0.
- **The 2-D KS distance** is a numpy implementation of the published method, chosen to
  match the SPI paper (FR-6.1); the paper's tables are the acceptance test.
- **N < 30** (FR-3.2) is enforced where a density is requested: `KernelDensity` raises
  `SmallSampleError` unless `allow_small_sample=True`, in which case it emits
  `SmallSampleWarning` and records the override.

### 3.4 Scoring and design

```python
def spi(test: Coordinates, target: Target, *, rng: int | np.random.Generator,
        n_samples: int = SPI_PAPER_N_SAMPLES) -> SPIResult
# SPIResult: value, ks_distance, n_samples, seed, settings, versions

def shift(a: Coordinates, b: Coordinates) -> ShiftVector
def overlap(a: KernelDensity, b: KernelDensity, fraction: float) -> float
def rank_against(target, scenarios: Mapping[str, Coordinates], **spi_kwargs) -> Frame
def rank_targets(test, targets: Mapping[str, Target], **spi_kwargs) -> Frame
# Frame: a table the package creates, in the library of the input frame where there is one,
# else the `backend=` argument (default pandas); see §5 conventions

class Brief(BaseModel, frozen=True):        # FR-7.1
    ambition: str; target: Target; kpi: KPI; assessment_method: str; scenarios: tuple[str, ...]

class Scenario(BaseModel, frozen=True):     # FR-7.2
    name: str; inputs: ScenarioInputs; produced_by: PredictorRef | SurveyRef; intervention_level: Level | None

def evaluate(brief: Brief, baseline: Coordinates, scenarios: Mapping[str, Coordinates], **spi_kwargs) -> Evaluation
# Evaluation: per-scenario SPI, shift, quadrant proportions, meets_kpi; tables ready for a report (FR-7.3, FR-7.5)
```

- **SPI is computed as published, and no more.** The target is sampled with an
  explicit seed (`rng` is required) and the paper's sample size as the default, so
  FR-6.1's default is met at the function level and every score is reproducible.
  No sampling-variability or bootstrap machinery is built: the SPI paper and Part 4
  report the index as a number, and the owner ruled (2026-10-30) that uncertainty
  reporting beyond the methodology is a leftover of the dropped APA-style
  requirement, so PRD FR-6.1's [proposed] bootstrap clause and open item 1 are
  withdrawn. A user who wants to see the seed sensitivity calls `spi` with several
  seeds; nothing in the package hides that this is possible.
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
  C has the full table). The hard requirements are pip wheels, no compiler, no MATLAB
  or R, and Python 3.11 and later. Licence is recorded per backend but is no longer a
  filter: the owner decided on 2026-10-30 that copyleft dependencies are acceptable
  (PRD NFR-7 needs amending to match). The field is still small:
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
  - **sottek-hearing-model 0.1.14** (AGPL-3 on PyPI; numpy, scipy, bottleneck, plus
    matplotlib, tqdm and PyQt6 declared as hard dependencies) is the ECMA-418-2
    backend for the 2025 edition: loudness, roughness, tonality and fluctuation
    strength, translated from the SQAT authors' verified MATLAB and maintained by the
    RefMap project. It fills both gaps that MoSQITo leaves on Python 3.11 and 3.12.
    Its PyQt6 dependency is the one cost (a large wheel in the `audio` extra); the
    plan asks upstream to make it optional and, failing that, accepts it.
  - **Not used, with the reason**: `MetaSona` (fast C implementation with wheels on
    all three OS, but the authors call roughness and tonality experimental and
    conformance unestablished; a candidate fast backend once validated); `pySQAT`
    (TU Delft port of SQAT, no licence file and no release, so unusable by anyone
    until a licence is added; the most complete candidate otherwise); `ecma418-torch`
    (requires torch, one commit); `iso532-1-rs` (no wheels, source build);
    `acoustic-toolbox` (useful, but pulls numba and soundfile for level metrics that
    are thirty lines of scipy here); `refmap-psychoacoustics` (its Python part is
    detectability only), `rota-order` (PR/TNR, which MoSQITo has), `torch_amt`
    (torch; not in Table D.1); `PsychoacousticParametersMeasurer`,
    `NumpyLibforPsychoAcoustic`, `LoudnessModel` (no licence, no release, or dormant).
- **Coverage statement for Table D.1 at M3, given the above.** Levels (LAeq, LCeq,
  LAF5, LAF95, LA10 − LA90, LCeq − LAeq, SEL, kurtosis) built in; loudness (N5, Navg,
  Nrmc, N95, N5/N95) and sharpness (S5, Savg, S50, S95) from MoSQITo; roughness (R10,
  R50) from MoSQITo (Daniel–Weber) and sottek-hearing-model (ECMA-418-2:2025);
  tonality from sottek-hearing-model (ECMA-418-2) and MoSQITo (ECMA-74 PR and TNR);
  fluctuation strength (F10, F50) from sottek-hearing-model (ECMA-418-2) and, on
  3.13+, phonometry (Osses); IACC built in. Every Table D.1 metric therefore has a
  backend on Python 3.11. Aures tonality has no admissible backend on 3.11 and is a
  documented gap (FR-4.4); psychoacoustic annoyance (FR-4.5) is phonometry-only.
- **Calibration is a value object applied once**, when a `Signal` is loaded; metrics
  never see raw full-scale data without knowing it. Loading with no calibration is
  allowed (for inspection and metrics that are level-independent), and a metric that
  needs Pa raises `UncalibratedSignalError`.
- **Audio I/O.** WAV via `scipy.io.wavfile` in the core (it reads 16-, 24- and 32-bit
  PCM and float, which covers sound-level-meter and binaural recorder exports) and
  `soundfile` in the `audio` extra for FLAC and other formats.
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
    per_input: Frame                        # one row per clip: predicted items/coordinates, labelled by model
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
├── cli/                 # click app; one command group per journey; documented by great-docs
└── _data/               # instruments/*.toml, angles/*.toml, profiles/*.toml, targets/*.toml,
                         #   schemas/*.toml (SSID, REDCap, Qualtrics), models/*.csv, isd_subset/
```

Layer contracts (enforced by import-linter):

| Layer | May import |
|---|---|
| `errors`, `provenance`, `settings`, `instrument` | numpy, pydantic, stdlib |
| `survey`, `projection`, `distribution` | the row above, narwhals, scipy, contourpy (pandas and polars only inside `survey.io` and `survey.stats`) |
| `spi`, `design`, `predict`, `audio`, `datasets` | the rows above (not each other, except `design` → `spi`, `predict` → `audio` contracts) |
| `plot` | everything above, matplotlib |
| `project`, `cli` | everything |

Draft 0.1 considered a uv workspace of several packages and rejected it for release
and CI cost; that still holds. The seams above are kept so that `audio` or `predict`
could be split into their own distributions later without renaming imports.

Extras (`pyproject.toml` `[project.optional-dependencies]`): `audio` (mosqito,
sottek-hearing-model, scikit-maad, soundfile, matplotlib pinned for MoSQITo,
phonometry on 3.13+), `optim` (pymoo), `polars` (polars and fastexcel, for users who
want polars frames back), `all`. openpyxl is a core dependency because Excel import
and export are Must requirements at M1 (FR-1.1, FR-9.1). Optional imports are gated at module import with one
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

- Functions take and return frames or records; they never mutate inputs. Derived
  columns are added to a returned copy with documented names, in the caller's own
  dataframe library.
- Tables the package creates (`describe`, rankings, audio results) come back in the
  library of the frame that went in; where nothing went in, `backend=` chooses, with
  pandas as the default.
- Long operations take `progress=` (default: the tqdm.auto reporter; `None` for
  silence; any `ProgressReporter` for a GUI) and `cancel=`.
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
- A dataframe accessor (`df.ssp.project()`) is rejected for 1.0: it is magic, hard to
  type with ty, pandas-specific, and hides the mapping argument.
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
| Tables | narwhals (`narwhals.stable.v2`) at the boundary; numpy inside kernels; pandas and polars tested in CI; pyarrow for parquet results | Owner decision. Functions accept any narwhals-supported eager frame and return the same kind. A spike in this session showed the boundary pattern (select mapped columns to numpy, compute, `with_columns` back) is a dozen lines and identical for pandas and polars. Two pandas-only boundaries remain and are documented: statsmodels (convert on the way in) and `read_survey` for Excel (pandas with openpyxl by default, polars with fastexcel via the `polars` extra). No backend-specific code anywhere else; an import-linter contract keeps `pandas` and `polars` imports out of every module except `survey.io` and `survey.stats`. |
| Numerics | numpy, scipy (stats, signal, optimize), contourpy | Standard. |
| Statistics | scipy.stats and statsmodels; thin `tidy()` helpers returning statistic, df or N, p, effect size | FR-2.6, FR-2.6a. pingouin was considered and rejected as a dependency for a few effect sizes that are ten lines each. |
| Audio | scipy.io.wavfile core; mosqito, sottek-hearing-model, scikit-maad, soundfile (and phonometry on 3.13+) in the `audio` extra | §3.5 and Appendix C. Licence recorded per backend, not a filter. |
| Optimisation | pymoo in the `optim` extra | FR-6.5 only. |
| Data fetching | pooch | Standard in scientific Python; hash checks, cache, licence message. |
| Progress | `ProgressReporter` protocol; default reporter on `tqdm.auto` (core dependency, pure Python) | Notebook users see a widget, terminal users a bar, GUIs bind their own, tests bind a recorder. |
| Plotting | matplotlib; seaborn only where it saves real code (stacked bars, palettes) | §10. |
| CLI | click | Owner decision: great-docs documents click commands automatically (`cli: enabled: true` in `great-docs.yml`, discovered from `soundscapy.cli`), so the CLI reference is generated like the API reference. |
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
   at the stated seed and sample size; the 2021 models predicting the published
   location means from the ISD feature table. The 2022 paper has no numeric
   fixture (owner decision, 2026-10-30): it prints figures, so its `.qmd` reproduces
   them by following the methodology, and the test is that it renders with every
   cell run. Its KDE is the same estimator seaborn's `kdeplot` wraps
   (`scipy.stats.gaussian_kde`), and seaborn's `bw_adjust` maps onto
   `KernelDensity.bandwidth` as a multiplier of Scott's factor, which is documented
   so the paper's settings can be stated exactly.
3. **Cross-implementation references** (`tests/reference/`, marked `reference`, run in
   CI nightly and on demand). Skew-normal density, conversions, sampling and fitting
   against committed R `sn` outputs (generated once by a script kept beside them;
   the 0.x branch `157-remove-r-sn-dependency-for-spi` supplies the first baseline
   and the pattern); ISO 532-1 reference
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
- **Progress and cancellation are protocols**: `ProgressReporter` with
  `start(total)`, `advance(path)`, `finish()`; `CancelToken` with `cancelled()`. The
  library checks the token between work units (FR-14.3). The default reporter wraps
  `tqdm.auto`, so a notebook shows a widget and a terminal a text bar with no setup;
  `progress=None` silences it; the CLI may bind rich instead; a GUI binds its own;
  tests bind a recorder. Cancellation in a notebook is the kernel interrupt, which
  the runner catches between work units and turns into a clean partial result.
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
├── data/                   # tables people write, check or edit by hand: CSV
│   ├── responses.csv       # the user's table, untouched
│   ├── groups.csv, sessions.csv, locations.csv      # optional
│   └── recordings.toml     # RecordingIndex (M3): paths, formats, hashes, calibration refs
├── results/                # tables the package computes: parquet, plus a JSON record each
│   ├── projection/         # coordinates.parquet + projection.json (settings, versions)
│   ├── audio/              # results.parquet + units.json + manifest.json + cache/*.parquet
│   ├── spi/                # spi.parquet + spi.json
│   └── evaluation/
├── exports/                # on demand: CSV and Excel copies of any result (FR-9.1)
└── manifest.json           # hashes of inputs, versions, timestamps for the whole project
```

- **One rule decides the format of a table.** Tables that a person writes, checks or
  edits by hand (survey exports, group, session and location tables, the recording
  index, mappings, settings) are CSV or TOML: they open in Excel, diff in git and
  need no tooling. Tables that the package computes (coordinates, descriptive
  statistics, audio metrics and their per-file cache, SPI scores, evaluations) are
  parquet: dtypes and units survive a round trip without a sidecar, a 1,000-file
  audio run reads back in milliseconds, and nobody should be hand-editing a
  computed result. Every computed table can be exported to CSV and Excel with one
  call or one CLI command, into `exports/`, which is how a user checks numbers or
  places them in a report (FR-9.1). pyarrow is therefore a core dependency; the
  owner accepted its size (2026-10-30). pandas and polars both read parquet, so the
  narwhals boundary is unaffected.
- CSV inputs get a small JSON sidecar only when the package has to pin a dtype the
  CSV cannot carry (an id column that must stay a string, for example); otherwise the
  mapping's declared scales and roles are enough.
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
4. **CSV for hand-made tables, parquet for computed ones** (§11), rather than one
   format for everything: the folder stays readable where people look, and fast and
   typed where the package looks.
5. **SPI carries no uncertainty machinery** (§3.4), because the methodology reports
   a number; the seed is required so reproducibility is explicit rather than
   statistical.
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
10. **narwhals is confined to the boundary.** Dataframe-agnosticism is an owner
    decision; the plan keeps its cost low by allowing narwhals calls only where a
    user frame enters or leaves (select to numpy, add columns, group keys) and by
    testing pandas and polars in CI. If a feature needs more than that, it converts
    to numpy rather than growing a narwhals-wide code path.
11. **`Dataset` has almost no behaviour.** Draft 0.1 and the 0.x package both grew
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
| M1. Survey analysis | `instrument`, `survey`, `projection`, `distribution` (Coordinates, regions, KDE, ks2d, SkewNormal), `plot` basics, `datasets.isd` with bundled subset, `project` minimal directory, CLI for journey A, recipes for plot type and dataset loader, plugin discovery, tutorial A, reproduction of the 2022 paper | Journey A done-when; the 2022 paper's `.qmd` renders its figures and tables from the ISD with every cell run, following the paper's methodology and the same KDE estimator seaborn wraps |
| M2. SPI and design (1.0) | `spi` with targets library, `design` with Brief/Scenario/evaluate, CLI for C and E1, tutorials C and E, course acceptance cases 2 and 3, reproduction of the 2024 paper, versioning policy, 1.0 release | Part 4 Example 6 reproduces; course exercises run unchanged |
| M3. Audio | `audio` (Signal, I/O, calibration, built-in level metrics, MoSQITo and scikit-maad backends, batch runner, manifest), CLI for B, metric recipe, tutorial B, ArtemiS comparison page | Journey B done-when; ISO 532-1 reference signals pass; a 1,000-file batch resumes |
| M4. Prediction | `predict` (protocol, discovery, LinearPredictor, default probabilistic predictor after suitability check), `soundscapy-soundaqnet` plugin, E2 in `design`, CLI for E2, predictor recipe, reproduction of the 2021 paper | Journey D and E2 done-when |
| M5. Ecosystem | SATP, ARAUS, DeLTA loaders with mappings; MOSAIQ import/export; target search behind `optim`; contributor trial of each recipe | Journeys F, G, L done-when |

---

## 15. Decisions taken by the owner

All decisions are now taken (Andrew Mitchell, 2026-10-30, two review rounds). They are
recorded here so the spec and plans can cite them, and so the PRD can be amended
where noted.

| # | Decision | Where it lands |
|---|---|---|
| 1 | Copyleft dependencies are acceptable; licence is recorded per backend, not a filter | §3.5, Appendix C; **PRD NFR-7 and open item 2 to amend** |
| 2 | Dataframe-agnostic through narwhals, confined to the boundary; pandas and polars tested | §2, §3.2, §6 |
| 3 | click for the CLI, documented by great-docs | §4, §6 |
| 4 | Progress visible in notebooks through a default `tqdm.auto` reporter | §2, §9 |
| 5 | Python floor stays 3.11; phonometry ships as a 3.13-only optional backend | §3.5, Appendix C |
| 6 | pyarrow accepted; CSV for hand-made tables, parquet for computed tables, CSV and Excel export on demand | §11 |
| 7 | TOML for settings and mappings; JSON for machine-written records | §6, §11 |
| 8 | No SPI uncertainty machinery; the paper's sample size is the function default and the seed is required | §3.4; **PRD FR-6.1 [proposed] clause and open item 1 withdrawn** |
| 9 | Missing attribute items exclude the response by default; renormalisation is an option under the canonical formulation only | §3.1, §3.2 |
| 10 | The 2022 paper is reproduced by methodology, not by a numeric baseline; same KDE estimator seaborn wraps | §8, §14 |
| 11 | Governance: Andrew Mitchell cuts releases and reviews every change that affects scientific results; cadence per milestone (§14) | **PRD NFR-12 and open item 3 to amend**; `GOVERNANCE.md` at Phase 0 |

### Inputs the plan still needs from the owner (from PRD §10.3)

The ISD language coverage against the SATP v1.5 tiers; the ISD record versions used by
each reproduced paper; machine-readable fixtures for Part 4 Example 6 and the 2024
paper's targets and sample size; whether redistributable audio exists; and the
features the 2021 deployable models use. Added by this plan: R `sn` available once
to generate the skew-normal reference fixtures (the 157 branch shows this has been
done before).

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
compiler, no MATLAB, no R; Python 3.11 and later (NFR-1, NFR-2). Licence is recorded
but, by the owner's decision of the same day, is not a filter; verdicts below reflect
that. MATLAB, Julia, Rust-only, C#, Pure
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
| sottek-hearing-model 0.1.14 | AGPL-3 (PyPI), GPL-3 (GitHub) | yes; `>=3.11`; numpy, scipy, bottleneck, matplotlib, **PyQt6**, tqdm | ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength | active; translated from the SQAT authors' verified MATLAB | **ECMA-418-2 backend**, `audio` extra; ask upstream to make PyQt6 optional |
| MetaSona 0.2.2 | GPL-3 combined (Apache/BSD/MIT parts) | yes, wheels for all three OS; `>=3.9,<3.15`; numpy, scipy | Zwicker loudness, DIN sharpness, D–W roughness, Aures tonality, ECMA-418-2 (2025) loudness, roughness, tonality; 15–70× faster than MoSQITo | new; authors call roughness and tonality experimental | Not yet: maturity; a candidate fast backend (and the only Aures tonality) once validated |
| pySQAT | **none stated** | no; `requirements.txt` with PySide6 | SQAT's set: ISO 532-1, ECMA-418-2, DIN 45692, D–W, Osses FS, Aures, PA, EPNL | developing; MATLAB-vs-Python validation workflow, no results published | Unusable without a licence; watch |
| ecma418-torch | MIT | no; requires torch | ECMA-418-2 (2025) roughness, fluctuation strength | one commit | Excluded: torch, unproven |
| iso532-1-rs | Apache-2.0 | no wheels; maturin source build | ISO 532-1 ZWST/ZWTV, MoSQITo-parity tests | new; author requests review | Excluded: no wheels; a possible fast backend later |
| refmap-psychoacoustics | GPL-3 | no | aural detectability (Python); SHM metrics via sottek-hearing-model | active | Not needed; its metrics come through sottek-hearing-model |
| rota-order 0.7.0 | GPL-3 | yes | ECMA-418-1 TNR and PR | active | Not needed; MoSQITo covers PR/TNR |
| torch_amt 0.2.0 | GPL-3 | yes; torch | Moore–Glasberg time-varying loudness | active | Not needed; torch, not in Table D.1 |
| LoudnessModel | BSD-2 | no release | Moore et al. 2018 time-varying loudness | dormant | Not needed |
| PyDSM 0.15.2 | GPL-3 | yes | ISO 226 contours | dormant | Not needed |
| PsychoacousticParametersMeasurer | none stated | no | Zwicker set from the book | legacy, 2021 | Unusable |
| NumpyLibforPsychoAcoustic | none stated | no | Moore–Glasberg loudness | new | Unusable without a licence |
| MoSQITo-FDP, python-acoustics, SUAVE, deeuu/loudness | Apache / BSD-3 / LGPL-2.1 / GPL-3 | archived or dormant | early ECMA-418-2, ISO 1996-2, EPNL, loudness | legacy | Not used |
| PsyTools | unknown | code not public | unknown | unknown | Not evaluable |

Gaps on Python 3.11 and 3.12 after the licence decision: Aures tonality (only in an
unmerged MoSQITo pull request, in unlicensed pySQAT, and in experimental MetaSona),
Osses fluctuation strength and psychoacoustic annoyance (phonometry, 3.13+). Table D.1
itself is fully covered. These are documented per FR-4.4, and the metric-backend
registry means a later package, a licence change in pySQAT, or a MoSQITo release
closes a gap by adding one backend class.
