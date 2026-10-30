# Architecture plan for the soundscapy rewrite

Status: **draft 0.4** (2026-10-30). Draft 0.2 was rewritten from PRD draft 0.5 and
superseded draft 0.1, which predated the PRD. Draft 0.3 applied a cross-document
consistency review (§16). Draft 0.4 applies the owner's responses to the four-read
review of PRD 0.6 and plan 0.3 (`design/reviews/prd-0.6-plan-0.3-review.md`), which
changed several public types before any code exists; §15 lists those decisions and
§17 the findings still open. It is in step with PRD draft 0.7 and is the document
AGENTS.md means by "the architecture plan"; the next step is the Phase 0 and M1 spec.
Nothing in it is fixed except the science; every decision below is a proposal with the
alternative it beat and the reason, so that each one can be challenged on its merits.

How to read it: §1 and §2 state the forces and the principles; §3 is the domain
model, which is where most of the design risk lives; §4 to §12 cover modules, API,
tooling, testing, concurrency, plotting, persistence and GUI readiness; §13 lists the
places where this plan departs from draft 0.1 or settles what the literature leaves
open; §14 is the delivery sequence; §15 is the register of the owner's decisions;
§16 records what the consistency review changed and the three M1 contracts a builder
would otherwise guess (CLI commands, core dependencies, default quality rules); §17
lists the review findings not yet decided.

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
   explicit argument whose value is recorded in the result. A warning is for a
   condition the user should act on, once per operation, never for a choice the user
   made deliberately (a legacy profile is recorded, not warned about). Rules out
   silent drops, silent coercion, `warnings.warn` with a generic category, and
   per-row warnings.
9. **Headless, but not mute.** No `plt.show()`, no `print`, no `input`, no implicit
   file writes. The library reports through the standard `logging` module the way
   scientific-Python packages such as pooch do (PRD NFR-13): a package logger
   `soundscapy` with a default `StreamHandler` at INFO attached on import, so a
   terminal or notebook user sees what a download fetches and under which licence,
   and what a long run is doing; `soundscapy.set_log_level("WARNING")` or standard
   logging configuration silences or redirects it; a GUI attaches its own handler.
   Progress goes through a `ProgressReporter` protocol and cancellation through a
   token; the package ships one default reporter built on `tqdm.auto`, which renders
   a text bar in a terminal and a widget in a notebook and never raises or warns.
   Rules out print, rich (which the CLI alone may use) and any output that cannot be
   captured or silenced through `logging`.
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
    # fingerprint: a derived content hash over positions, angles, axes and scale (FR-6.3);
    # every by-name reference to a circumplex carries (name, fingerprint)

class CircumplexRef(BaseModel, frozen=True):
    name: str
    fingerprint: str

class AngleSet(BaseModel, frozen=True):
    """A versioned set of per-language angles with confidence tiers (FR-2.4, FR-2.4a)."""
    name: str                               # "satp", "aletta", or user-supplied
    version: str                            # "1.5", "2024"; superseded versions are never removed
    circumplex: CircumplexRef
    languages: Mapping[str, LanguageAngles] # ISO 639-3 -> angles per position + tier
    provenance: Provenance

class AngleSelection(BaseModel, frozen=True):
    """How angles are resolved per response (FR-2.4): equal angles, or a default set with
    per-language overrides that may name a shipped version or a user-supplied file."""
    default: Literal["equal"] | str         # "equal", or "satp@1.5"
    per_language: Mapping[str, str] = {}    # ISO 639-3 -> "name@version" or path to a user set
    on_uncovered: Literal["error", "exclude", "equal"] = "error"       # FR-2.4 override
    on_low_confidence: Literal["error", "exclude", "equal"] = "error"  # FR-2.4a override

class ProjectionSettings(BaseModel, frozen=True):
    formulation: Literal["canonical", "iso2019", "aletta2024"]
    angles: AngleSelection
    dispersion: Literal["range", "iqr"]
    missing_items: Literal["exclude", "renormalise"] = "exclude"   # FR-2.5a; canonical only, never in ISO profiles

class DensitySettings(BaseModel, frozen=True):
    bandwidth: float | Literal["scott", "silverman"] = "scott"     # scipy bw_method factor; see §3.3
    fraction: float = 0.5                   # the "50th-percentile contour" (FR-3.1)
    grid: int = 200                         # grid over exactly [-1, 1]^2
    min_n: int = 30                         # FR-3.2
    allow_small_sample: bool = False

class SPISettings(BaseModel, frozen=True):   # M2
    n_samples: int = SPI_PAPER_N_SAMPLES
    statistic: Literal["fasano-franceschini", "peacock"]   # owner confirms the paper's choice (PRD §10.3)
    truncate: bool                          # sample the target truncated to the square; owner confirms
    n_seeds: int = 1                        # > 1 reports min, mean, max (FR-6.1, Should)

class Settings(BaseModel, frozen=True):
    """The resolved analysis settings recorded in every result (FR-11.1)."""
    profile: str | None                     # the preset this started from, or None once edited
    projection: ProjectionSettings
    density: DensitySettings
    quality: QualitySettings                # rule list and thresholds (§16.3)
    spi: SPISettings | None = None          # M2
    # audio: AudioSettings | None, added at M3; nothing M3-specific exists in M1 settings
```

A **profile** (FR-2.9) is a named `Settings` preset shipped as a TOML file:
`iso12913-3-2019` (iso2019 formulation, equal angles, range), `iso12913-3-2024`
(canonical formulation as the revision's general-angle formulas, equal angles, IQR;
no translated angles, because the revision has only a placeholder for them) and
`soundscape-across-languages-2026` (canonical, `satp@1.5` with tiers, IQR), the
default. A profile named after a standard is faithful to its text; `renormalise` is
not available in the two ISO profiles.

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
  `on_uncovered` and `on_low_confidence` rules. `"error"` stops the projection;
  `"exclude"` leaves the coordinate NaN with the reason `uncovered_language` or
  `low_confidence_language` in the shared status column (FR-1.5), which is the
  recommended override because it never mixes geometries; `"equal"` projects those
  responses with equal angles, emits one `MixedAngleSetsWarning` naming the languages,
  and `describe` reports the mix. The projection result records, per response, which
  set and version supplied its angles (`Coordinates.labels["angle_set"]`, §3.3), so a
  nine-language dataset using two shipped versions and one user file is representable
  and auditable. Language values arrive as "en", "English" or ISO 639-3; the mapping's
  language coding normalises them to ISO 639-3 at the boundary. Deriving angle sets is
  out of scope (FR-2.4a); a user file is the same TOML format as a shipped set.
- **Angle sets, targets and profiles are versioned data files, not code.** They ship
  under `soundscapy/_data/` as TOML with provenance and a version, and a user-supplied
  set is the same file format. Adding the next SATP version is a data change plus a
  test; the old version stays shipped, so an old result that names `satp@1.5` is
  reproducible after the update (FR-6.3). Angle sets are keyed by circumplex as well
  as language; a non-ISO circumplex with a declared language falls back to its own
  angles.
- **Profiles are presets, settings are the record.** A result records the resolved
  `Settings`, and `Settings.profile` names the preset only while the values still
  equal it; once a user edits one field the result says `profile = None` and a
  `derived_from` note, computed at record time, not stored. A later change to what a
  profile means therefore cannot silently change old results, and a result never
  claims a profile it does not match.

### 3.2 Surveys: mappings, tables, quality

```python
class Coding(BaseModel, frozen=True):
    """How raw export values become scale scores (FR-2.1): REDCap 0..4, text anchors,
    reversed parts. Identity when the export already holds Table A.1 scores.
    Applied in this order: values, then offset, then reverse as min + max - x against
    the resolved scale. Scores are floats (a slider is a scale too)."""
    values: dict[str, float] | None = None  # raw value (as a string) -> score; None = numeric as is.
                                            #   Keys are strings because TOML and JSON keys are; raw
                                            #   values are normalised by str(int(x)) for integral numbers
    offset: float = 0                       # e.g. +1 for 0-based exports
    reverse: bool = False                   # part coded in the opposite direction

ConstructGroup = Literal["attributes", "source_dominance", "overall_quality",
                         "appropriateness", "who5"]

class SurveyMapping(BaseModel, frozen=True):
    """Which user column means which construct (FR-1.3), persisted as a file (FR-1.3a).
    Named SurveyMapping, never Mapping, which would shadow collections.abc.Mapping."""
    instrument: CircumplexRef
    scales: dict[ConstructGroup, Scale] = {}            # overrides of the ISO defaults: attributes 1..5
                                                        #   (from the instrument), source dominance 1..5,
                                                        #   overall quality 1..5, appropriateness 1..5, WHO-5 0..5
    attributes: dict[str, str]                          # position -> column name
    coding: dict[ConstructGroup | str, Coding] = {}     # per construct group, or per position; identity if absent
    language: str | LanguageColumn                      # one language, or {"column": name, "values": {...}}
                                                        #   with a coding of the column's values to ISO 639-3
    ids: Ids                                            # column names for the roles response, group, session,
                                                        #   location (SSID) and participant, stimulus (listening
                                                        #   experiments); any subset
    source_dominance: dict[str, str] = {}               # construct -> column
    overall_quality: str | None = None
    appropriateness: str | None = None
    who5: tuple[str, ...] = ()
    extra: dict[str, ColumnRole] = {}                   # demographics, free text, flags, carried through
    provenance: Provenance

class Dataset:                              # frozen dataclass, plain container; the frames inside are
                                            #   the user's objects and are never mutated by the package
    responses: IntoDataFrame                # any narwhals-supported *eager* frame; lazy frames are rejected
    mapping: SurveyMapping
    groups: IntoFrame | None = None         # one row per group; recording path lives here
    sessions: IntoFrame | None = None
    locations: IntoFrame | None = None
    recordings: RecordingIndex | None = None   # from M3; never loads audio
    provenance: Provenance
```

- **`SurveyMapping` is the schema; there is no separate schema library.** Draft 0.1
  proposed pandera. The checks the package needs (declared columns present, values
  within the declared scale, ids unique where required, language codes known) are a
  few dozen lines against the mapping, and FR-1.5 needs them to produce a per-response
  *decision and reason*, not a raised error. pandera would add a dependency, a second
  description of the same columns, and a class-based DSL that ty may not type well.
  Decision: a `validate(dataset) -> ValidationReport` function in the package;
  pandera is not used. Revisit only if validation needs grow beyond the mapping.
- **Quality rules are a list of named rule objects** (FR-1.5), each a small callable
  `(responses, mapping) -> boolean mask` with a name and a reason string.
  `apply_quality_rules` adds `included` (bool) and `exclusion_reason` (a `;`-joined
  string, empty when included) and returns a new frame in the caller's library; it is
  the only function that writes those columns onto the responses frame. The projection
  and the language rules write their reasons (`incomplete_attributes`,
  `uncovered_language`, `low_confidence_language`) to the same two column names on
  `Coordinates.labels` (§3.3), never back onto the responses frame, so one column name
  explains every absence and the user's table is written once. Rules are configurable
  by listing them in settings; the default list is in §16.3. Nothing is dropped.
- **The narwhals boundary follows narwhals' own conventions**, reviewed against its
  documentation while building (owner, 2026-10-30). The plan's commitments: accept
  `IntoDataFrame` and reject lazy frames with a `TypeError`; one internal helper
  converts to `nw.DataFrame` and back with `to_native()`; derived columns are added
  with `with_columns` from numpy results; "returned in kind" means the input frame's
  library, and where a path went in (`read_survey`, `Project.open`) a
  `backend="pandas" | "polars"` argument chooses, pandas by default. Blank Likert
  cells arrive as float NaN from pandas and nullable integers from polars; coding and
  scale checks handle both and tests cover both, which is why polars is in the `test`
  dependency group. A `# ty: ignore` is allowed only inside those helpers, with a
  comment.
- **Mapping inference** (FR-1.3a) is `suggest_mapping(df, instrument, known_schemas=...)
  -> MappingSuggestion`, where each suggested binding carries a confidence and the
  evidence (exact name match against the SSID or REDCap codebook, fuzzy match, value
  range). A `MappingSuggestion` becomes a `Mapping` by `accept()` after the user
  confirms or corrects in the CLI prompt, in a notebook, or in a GUI form generated from
  the model's JSON schema. Known export schemas are data files, not code.
- **The canonical survey layout** (FR-1.2) is a documented column set using the ISO
  instrument's reference-language labels and the SSID id names, with the identity
  coding. It ships as CSV and Excel templates under `_data/templates/`, exposed by
  `ssp.templates.save(path)` and the CLI, and is the layout the course teaches (M1).
- **Missingness** (FR-1.8, Should, M5): the domain model represents a missing item as
  null; the MOSAIQ tri-state (present, absent, not reported) is carried only at the
  MOSAIQ import/export boundary (an `survey.mosaiq` module at M5), which maps "not
  reported" to null plus a per-dataset note. Deferred deliberately.
- **Site clustering** (FR-2.7, Should) is a thin `survey.cluster(features, k, ...)` over
  scikit-learn's standard algorithms, returned as a labelled table; scikit-learn is an
  `analysis` extra, not core.
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
  coordinate statistics. `"renormalise"` (PRD 0.7 FR-2.5a, a constrained Should)
  projects over the present items with Σ|cos θ_i| and Σ|sin θ_i| recomputed over those
  items. It keeps coordinates in [−1, 1] and neutral at the origin, but an unbalanced
  subset no longer has Σcos θ = 0 and the projected variance changes, and no source
  defines the rule. So: it is offered only for the canonical formulation and refused in
  the two ISO profiles; it requires at least one present item with a non-zero
  normaliser on each axis, else the response is excluded; `labels` carries
  `n_items_present`; renormalised responses are excluded from `describe`, densities
  and SPI unless `include_renormalised=True`; and the choice is recorded in
  `Settings`. Under the 2019 and Aletta 2024 formulations a missing item always
  excludes the response.

### 3.3 Coordinates and distributions: the central abstractions

```python
@dataclass(frozen=True, eq=False)
class Coordinates:
    """A set of points in one circumplex, with provenance (FR-2.5, FR-5.6, FR-5.9)."""
    xy: np.ndarray                          # (n, 2), float64; valid rows in [-1, 1]^2, excluded rows NaN,
                                            #   one row per input response so labels and xy stay aligned
    circumplex: Circumplex
    labels: IntoDataFrame                   # per-point columns in the input's library, never None; always
                                            #   response_id, language, angle_set ("satp@1.5" | "equal"),
                                            #   included, exclusion_reason, n_items_present, plus the
                                            #   mapping's id and grouping columns
    origin: Origin                          # one record with kind = "survey" | "prediction" | "sample"
                                            #   and the settings that produced the points: the resolved
                                            #   Settings plus mapping hash and angle-set versions for a
                                            #   survey, a PredictorRef for a prediction, a TargetRef for a
                                            #   sample; defined in provenance.py so distribution never
                                            #   imports spi or predict
    # properties: n, n_valid
    # methods: subset(mask), group_by(*keys) -> Iterator[tuple[key_tuple, Coordinates]],
    #          mean(), sd(), quantiles(), fraction_in(region) ("percent likely to perceive"),
    #          to_frame(), equals(other, rtol) (no generated __eq__ on an ndarray field)
    # weights: not a field; the PRD never asks for response weights, and the ADR records that

def describe(coords: Coordinates, *, by: str | Sequence[str] | None = None) -> Frame
# one tidy row per group (crossed keys and time bins are just columns of labels): n, n_valid, mean, sd,
# quantiles, quadrant proportions, the angle-set mix, and the renormalised count

@dataclass(frozen=True)
class KernelDensity:
    """A KDE view of Coordinates with an explicit bandwidth (FR-3.1)."""
    coordinates: Coordinates
    settings: DensitySettings
    # grid over exactly [-1, 1]^2 at settings.grid; density evaluated on the square and mass normalised
    # over it, so a mass contour has one meaning; records the scipy factor AND the resolved absolute
    # bandwidth matrix (factor^2 x data covariance) in axis units, since "bandwidth" to a user means the latter
    # methods: grid() -> DensityGrid, mass_contour(fraction) -> Polygon (plain arrays, no matplotlib),
    #          decile_contours() -> list[Polygon], marginals() -> (x, y) densities
    # N < 30: constructing a KernelDensity on fewer than settings.min_n valid points emits
    # SmallSampleWarning and records n; refusing is the drawing layer's job (§10)

class SkewNormal(BaseModel, frozen=True):
    """Bivariate skew-normal in Azzalini's direct parameters (FR-3.4, FR-6.2)."""
    xi: tuple[float, float]
    omega: tuple[tuple[float, float], ...]  # 2 x 2
    alpha: tuple[float, float]
    # classmethods: from_centred(mean, cov, gamma1) raising NotAdmissibleError (marginal skewness
    #               must be below ~0.9953, Azzalini and Capitanio 1999)
    # methods: to_centred(), mean() (closed form), sample(n, rng, truncate) -> Coordinates(kind="sample"),
    #          pdf(xy)

def fit_skew_normal(coords: Coordinates, *, method: Literal["mle"] = "mle") -> SkewNormalFit
# SkewNormalFit: distribution, log_likelihood, converged, n_iter, method, start (method of moments),
# settings, versions. Fitting is done in centred parameters as sn's selm does, because direct-parameter
# MLE runs to the alpha boundary; a Must at M2 (owner, 2026-10-30)

class Target(BaseModel, frozen=True):      # lives in spi/, not distribution/ (§4)
    name: str
    version: str                            # FR-6.3; superseded versions stay shipped
    distribution: SkewNormal
    circumplex: CircumplexRef               # name plus fingerprint, so an edited instrument is caught
    kind: Literal["demonstration", "reference", "bespoke"]
    provenance: Provenance
```

- **`Coordinates` is the object every downstream feature accepts.** Projected
  survey responses, pooled single-response predictions (FR-5.9) and samples from a
  target are all `Coordinates`; `origin.kind` keeps them distinguishable in every
  table and plot (FR-5.6), and a set pooled from single-response predictions carries
  `PredictedPointsWarning` once and the label "predicted (single-response)" in every
  table and plot, because conditional means have less spread than responses (PRD 0.7
  FR-5.9). Descriptive statistics, quadrant proportions, KDE, shift vectors, SPI,
  scenario comparison and every plot take `Coordinates`. This is the single most
  important decision in the plan: it makes journeys A, C, D and E share one path
  instead of four. Rows stay aligned with the input responses (NaN where excluded), so
  `labels` is never optional and a user can always join back to their table.
- **Regions** (quadrants, half-planes, user polygons) are value objects with a numpy
  point-in-polygon test (a few lines of ray casting), so `distribution` imports
  neither matplotlib nor shapely. Quadrants are addressed by axis signs
  (`quadrant("+", "-")` is positive on the first axis, negative on the second) so
  they work for any instrument; an instrument may name them, and the ISO instrument
  names each quadrant after the attribute on its diagonal (`"calm"`, `"vibrant"`,
  `"chaotic"`, `"monotonous"`), which `quadrant()` accepts as aliases.
- **KDE is a view, not a transform.** The estimator is `scipy.stats.gaussian_kde`
  (Gaussian kernel, bandwidth a scalar factor times the data covariance), the same
  family seaborn uses, so there is exactly one KDE definition in the package
  (principle 7). `bandwidth` has scipy's `bw_method` semantics exactly: `"scott"`,
  `"silverman"`, or a float that *is* the factor. seaborn's `bw_adjust` is a
  multiplier of Scott's factor, so a seaborn setting is reproduced by
  `bandwidth = bw_adjust * scott_factor(n, d)`, and a helper computes that number.
  The resolved factor and the absolute bandwidth matrix are recorded in every result;
  the default is Scott's rule, with a sensitivity helper. The grid is fixed to
  [−1, 1]² with the mass normalised over the square, so the "50th percentile contour"
  computed from the cumulative mass of the grid has one meaning and matches seaborn's
  `levels=[0.5]` only when seaborn is told the same `clip`; the fraction is a parameter
  (FR-3.1). Contour polygons come from `contourpy`, which matplotlib already depends
  on, and are returned as plain arrays, so geometry needs no figure.
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
- **The 2-D KS distance** is a numpy implementation of a named statistic. Peacock
  1983 (all pairwise origins) and Fasano–Franceschini 1987 (data points as origins,
  the two directions averaged) give different D, the tie rule at a quadrant boundary
  matters on a five-point scale, and whether the target is sampled truncated to the
  square changes every score. The builder's belief is that the SPI paper's code used
  the Fasano–Franceschini variant with D = (D1 + D2)/2; the owner confirms (PRD
  §10.3). Whichever it is, `SPISettings` names the statistic, the tie rule (`<=` on
  both axes unless the paper says otherwise) and the truncation, and the paper's
  tables are the acceptance test, written last in M2.
- **N < 30** (FR-3.2) is a warning in the geometry and a decision in the drawing
  layer: `KernelDensity` on fewer than `min_n` valid points emits `SmallSampleWarning`
  and records n; `plot.density` on such a group refuses with `SmallSampleError` unless
  `allow_small_sample=True`; `plot.by_group` and the CLI draw scatter for the small
  groups and annotate them, so one small site never fails a twenty-site figure. N is
  the number of valid (projected, included) points; the rule is not applied to
  sampled distributions.

### 3.4 Scoring and design

```python
def spi(test: Coordinates, target: Target, *, rng: int | np.random.Generator,
        settings: SPISettings | None = None) -> SPIResult
# SPIResult: value (the single-seed published number), ks_distance, n_test, n_samples, seed, statistic,
# tie_rule, truncated, settings, versions; when settings.n_seeds > 1 also seeds, values_min, values_mean,
# values_max (FR-6.1, Should)

def shift(a: Coordinates | Target, b: Coordinates | Target) -> ShiftVector
# difference of means; a Target contributes its closed-form mean, so shift against a target needs no seed
def rank_against(target, scenarios: Mapping[str, Coordinates], **spi_kwargs) -> Frame
def rank_targets(test, targets: Mapping[str, Target], **spi_kwargs) -> Frame
# Frame: a table the package creates, in the library of the input frame where there is one,
# else the `backend=` argument (default pandas); see §5 conventions
# No overlap statistic: none is defined in the cited literature (PRD 0.7 FR-3.3)

class Brief(BaseModel, frozen=True):        # FR-7.1
    ambition: str; target: Target; kpi: KPI; assessment_method: str; scenarios: tuple[str, ...]

class Scenario(BaseModel, frozen=True):     # FR-7.2
    name: str; inputs: ScenarioInputs; produced_by: PredictorRef | SurveyRef; intervention_level: Level | None
    description: str = ""

def evaluate(brief: Brief, baseline: Coordinates, scenarios: Mapping[Scenario, Coordinates], **spi_kwargs) -> Evaluation
# Evaluation: per-scenario SPI, shift, quadrant proportions, meets_kpi, the scenario's Annex A level and
# produced_by; tables ready for a report (FR-7.3, FR-7.5)
```

- **SPI is computed as published, with its inputs on record.** The target is sampled
  with an explicit seed (`rng` is required) and the paper's sample size as the
  default, so FR-6.1's default is met at the function level and every score is
  reproducible; the statistic variant, tie rule, truncation and the test-sample N are
  in the result because they all move the number. The single-seed value is the
  published method. `n_seeds > 1` (a Should, owner 2026-10-30) repeats the draw and
  reports min, mean and max, so a KPI check near its threshold (FR-7.3) can be
  defended; there is no bootstrap over responses.
- **Target search** (FR-6.5, Should) lives behind the `optim` extra with pymoo and
  returns a `Target` whose provenance records the ranking, objectives and optimiser
  settings. It is the only place in the package that needs a long-running optimiser,
  so it uses the same progress and cancellation hooks as audio.
- **Design (FR-7) is thin by design.** A `Brief` and `Scenario` are records; `evaluate`
  takes `Scenario`s so the Annex A level, description and `produced_by` reach the
  tables, composes `spi`, `shift` and `fraction_in`, and returns tables. There is no report
  engine (FR-9.2). The Part 4 evaluation content (FR-7.5, Should) is `Evaluation`'s
  five tables and figures in the order the standard lists them: the brief
  (`Brief.to_frame()`), the baseline (descriptives, quadrant proportions, figure), each
  intervention (its `Scenario` with the Annex A level and free-text description),
  the post-assessment (the same descriptives per scenario) and the check against the
  brief (SPI, shift, `meets_kpi`). Each is exportable like any result.

### 3.5 Audio: signals, calibration, metrics, batches (M3)

```python
@dataclass(frozen=True)
class Signal:
    """One window of one recording in memory: a 30-s clip whole, or one window of an hours-long file."""
    samples: np.ndarray                     # (channels, n), float64 Pa after calibration, else "full scale"
    rate_hz: int
    channels: tuple[str, ...]               # ("L", "R") or ("M",); more than two channels is refused (FR-4.1)
    calibration: Calibration | None         # None means uncalibrated, and metrics that need Pa refuse
    window: Window | None                   # start and length within the recording, None for a whole clip

def read(path, *, calibration=None, window: Window | None = None) -> Iterator[Signal]
# yields whole clips, or fixed-length windows read one at a time (FR-4.1a), so an hour of 48 kHz stereo
# (2.8 GB as float64) is never held in memory; windowing is the batch runner's concern and a backend
# only ever sees one Signal

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
    channel_rule: Literal["each", "higher", "mean"] = "higher"   # FR-4.6: per metric, on the final statistic;
                                            #   the chosen channel is recorded per cell; not applied to signed
                                            #   statistics or cross-channel metrics

class MetricImplementation(Protocol):       # one class per (metric, backend) pair; extension point (FR-13.1, M3)
    metric: str; backend: str; library: str; version: str; method: str; units: str; licence: str
    kind: Literal["per_channel", "cross_channel"]
    def compute(self, signal: Signal, **params) -> MetricOutput   # time series and/or scalar, with frame rate
# Entry points under `soundscapy.metrics` yield classes; the registry instantiates them once. A
# per_channel implementation receives a mono Signal per channel; cross_channel receives both.
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
  (PRD NFR-7), and the same day that complete Table D.1 coverage is not a requirement
  and a missing open-source implementation never blocks development (PRD 0.7 FR-4.4).
  The decision: **MoSQITo is the single backend of the `audio` extra at M3, with
  built-in level metrics and IACC; phonometry is an explicit `phonometry` extra for
  users on Python 3.13 and later; no backend that depends on PyQt6 is adopted.** The
  field is still small:
  - **MoSQITo 1.2.1** (Apache-2.0; numpy, scipy, pyuff) is the default backend. It
    provides ISO 532-1 stationary and time-varying loudness, DIN 45692 sharpness,
    Daniel–Weber roughness, ECMA-418-2 loudness and roughness (an earlier edition than
    the 2025 fourth), and ECMA-74 prominence and tone-to-noise ratios. Smoke-tested in
    this session on Python 3.11 and 3.13 with numpy 2: all of those import and run,
    and 1 kHz at 74 dB gives 10.9 sone, consistent with ISO 532-1. Two risks: it has
    had no release since April 2024, and its ECMA roughness module imports matplotlib
    without declaring it. Mitigations: pin a tested range (NFR-4); matplotlib is a
    core dependency anyway (§16.2); Apache-2.0 permits a maintained fork or vendored
    patch if the project stays dormant.
  - **scikit-maad 1.5.2** (BSD-3; adds scikit-image and pywavelets) for ecoacoustic
    indices (FR-4.5), in the `audio` extra.
  - **phonometry 3.3** (MIT; numpy and scipy only) is the only permissive package that
    covers the whole of Table D.1 and the research set: ISO 532-1/2/3 loudness, all
    four ECMA-418-2 metrics (2025 edition), DIN 45692 sharpness, Osses fluctuation
    strength, psychoacoustic annoyance, ECMA-418-1 tonality. It requires **Python
    3.13 or later**, which is above the PRD's 3.11 floor, it is one person's work
    with self-reported conformance checks, and it is young (released 2026). Use: an
    optional second backend in its own extra, `soundscapy[phonometry]`, never an
    environment marker on `audio`, so `soundscapy[audio]` installs the same metric
    list on every Python version and the metric list a project sees depends on what
    the user chose to install, which the result records (FR-11.2). Its metrics are
    registered under their own method names (Osses fluctuation strength, ECMA-418-2
    tonality), so a column never changes meaning when the backend changes (§15, #5
    and #17).
  - **sottek-hearing-model 0.1.14** (AGPL-3 on PyPI, GPL-3 on GitHub; numpy, scipy,
    bottleneck, plus matplotlib, tqdm and PyQt6 declared as hard dependencies) covers
    ECMA-418-2:2025 loudness, roughness, tonality and fluctuation strength, translated
    from the SQAT authors' verified MATLAB. **Not adopted** (owner, 2026-10-30): a
    0.1.x package whose hard dependency is a GPL GUI toolkit of some 60 MB has no
    place in a headless extra for consultants' laptops, and its licence is itself
    ambiguous. Revisit only if PyQt6 becomes optional upstream; the registry makes
    that one implementation class.
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
- **What M3 offers, given the above.** With `soundscapy[audio]` on any supported
  Python: levels (LAeq, LCeq, LAF5, LAF95, LA10 − LA90, LCeq − LAeq, SEL, kurtosis)
  and IACC built in; loudness (N5, Navg, Nrmc, N95, N5/N95) and sharpness (S5, Savg,
  S50, S95) from MoSQITo; roughness statistics from MoSQITo by the Daniel–Weber method,
  offered under that name; prominence and tone-to-noise ratios (ECMA-74) from MoSQITo;
  ecoacoustic indices from scikit-maad. With `soundscapy[phonometry]` on 3.13 and
  later: ECMA-418-2:2025 loudness, roughness, tonality and fluctuation strength, Osses
  fluctuation strength and psychoacoustic annoyance. Documented gaps on 3.11 and
  3.12: fluctuation strength, ECMA-418-2 tonality and Aures tonality. Where Table D.1
  names a method the backend does not implement, the metric appears under the method
  the backend implements and the substitution is documented, not presented as
  coverage (FR-4.4); which tonality Table D.1 means in each edition is an owner input.
- **Calibration is a value object applied once**, when a `Signal` is loaded; metrics
  never see raw full-scale data without knowing it. Loading with no calibration is
  allowed (for inspection and metrics that are level-independent), and a metric that
  needs Pa raises `UncalibratedSignalError`.
- **Equalisation** (FR-4.3, Should): `Signal.equalise(filter)` applies a user-supplied
  FIR or IIR filter (coefficients in a small TOML or WAV impulse response) and records
  it in the signal's provenance; no filter ships.
- **Time series** (FR-4.11, Should): `MetricOutput` carries the time series and its
  frame rate; the batch runner stores them per file in the cache (`cache/<hash>.parquet`
  holds the series, the summary table holds only statistics), and
  `AudioResults.timeseries(recording, metric)` reads one back for `plot.timeseries`
  and `plot.channels` (FR-8.5, per-channel comparison).
- **Audio I/O.** WAV via `scipy.io.wavfile` in the core (it reads 16-, 24- and 32-bit
  PCM and float, which covers sound-level-meter and binaural recorder exports) and
  `soundfile` in the `audio` extra for FLAC and other formats.
- **Batch runner.** `analyse(recordings: RecordingIndex, metrics: MetricSet, *,
  executor, cache_dir, progress, cancel) -> AudioResults`. Work units are
  `(path, file_hash, calibration, MetricSet, window)` and are picklable; results are
  cached per file as a small table keyed by (file hash, settings hash, backend
  versions), so a rerun computes only what is missing (FR-4.8) and a manifest records
  hashes, versions and timings (FR-11.4). The file hash is computed once by `analyse`,
  never by `index`, so indexing stays a header read (FR-1.6). Per-file calibration is
  written into `AudioResults.settings`, not only onto the `Signal` (FR-4.2). Output is
  one tidy table per recording, channel and window with columns `<metric>_<statistic>`
  and a units table beside it (FR-4.9).

### 3.6 Prediction (M4)

```python
class PredictorSpec(BaseModel, frozen=True): # what the entry point exposes WITHOUT loading the model
    info: PredictorInfo                     # name, version, training data, setting, licence, citation
    inputs: InputContract                   # required and optional features/context/design params, with units;
                                            #   a validator rejects any perceptual (survey-derived) input (FR-5.5)
    outputs: OutputContract                 # PAQ items, coordinates, sources, scene; "single" or "distribution"

class Predictor(Protocol):                  # FR-5.1
    spec: PredictorSpec
    def predict(self, inputs: PredictionInputs, *, rng=None, n_samples=None) -> Prediction

@dataclass(frozen=True)
class Prediction:
    per_input: Frame                        # one row per clip: predicted items/coordinates, labelled by model
    coordinates: Coordinates                # origin.kind="prediction"; pooled from single responses (FR-5.9,
                                            #   with PredictedPointsWarning and label) or sampled
    predictor: PredictorRef                 # name, version, kind, n_samples | None, seed | None, absent optional inputs
```

- **Plugins by entry points.** Predictors register under the `soundscapy.predictors`
  entry-point group (metric implementations under `soundscapy.metrics`, dataset
  loaders under `soundscapy.datasets`). An entry point resolves to a `PredictorSpec`
  plus a loader callable, so `predictors.available()` lists names, requirements and
  contracts (FR-5.2) without importing torch; the model loads only on first `predict`.
  There is no import-time registry. Discovery is one generic
  `plugins.discover(group)` over `importlib.metadata.entry_points`, delivered at M2
  with the first alpha (PRD §9) and unit-tested by monkeypatching the entry points.
- **The interface is fixed now; the model is chosen at M4 planning** (owner,
  2026-10-30; PRD 0.7 FR-5.3). The candidates are the 2021 linear models (FR-5.3a: a
  `LinearPredictor` from a coefficient table shipped as data, with "known location",
  which needs prior survey data at the site, and "fixed effects only" as explicit
  modes; needed anyway to reproduce the 2021 paper), the ARAUS "Probably Pleasant"
  model and the Tong et al. 2026 group-level model. Each is one class behind the
  protocol, which is the point of the module's design. Deep models (SoundAQnet) are a
  separate package `soundscapy-soundaqnet` that depends on torch and fetches
  hash-pinned weights on first use with pooch under their licence (FR-5.4, NFR-5),
  loaded through a format that does not execute code on load (safetensors, or
  `weights_only=True`).
- **A distributional predictor samples its output** into `Coordinates` with an
  explicit `n_samples` and `rng`, both recorded in `PredictorRef` (`None` for a
  single-response predictor), so the N < 30 rule and SPI treat it exactly like pooled
  single responses (FR-5.9).
- **Graceful degradation is declared, not guessed** (FR-5.5): the input contract marks
  each input required or optional, `predict` raises `MissingInputError` naming the
  missing required inputs, and the `Prediction` records which optional inputs were
  absent.

### 3.7 Datasets

Loaders return `Dataset`s with a shipped `Mapping`, fetched with `pooch` from the public
record with hash checks and licence display on first download (FR-10.1, NFR-5). The ISD
loader also returns the ISD's published psychoacoustic feature table as
`Dataset.groups` columns (one row per group, joined by group id), which course
acceptance case 1 (M2) and the 2021 reproduction (M4) need without the audio pipeline.
A small ISD subset, with its feature columns, is bundled for tests and tutorials.
Loaders pin the record version they were validated against and expose it. Each loader is a module with one public
`load()` and a `describe()` returning MOSAIQ catalogue fields (FR-10.3). Harmonising a
new dataset is writing a mapping file and a loader of a few dozen lines, which is the
dataset recipe (FR-13.1).

### 3.8 Provenance and results

Every result record (`ProjectionResult`, `DescriptiveStats`, `SPIResult`,
`SkewNormalFit`, `AudioResults`, `Prediction`, `Evaluation`) is a frozen dataclass
with four common fields: `settings` (the resolved pydantic model), `versions`
(package, Python, the wrapped libraries involved, and the versions of every shipped
data file used: angle sets, targets, profiles), `methods` (the method version of each
procedure involved, a `METHOD_VERSION` constant per procedure that is bumped when a
scientific result changes beyond tolerance, FR-13.4 and NFR-4) and `created` (UTC).
Each has `to_frame()` for its tabular part and `save(dir)`/`load(dir)` writing tables
plus a JSON sidecar, which is how the project directory stores results (§11). The JSON
sidecar is authoritative for a result; `settings.toml` is the user's input.

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
├── distribution/        # Coordinates, regions, describe, KernelDensity, DensityGrid, SkewNormal, fit, ks2d, shift
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
| `errors`, `provenance`, `settings`, `instrument`, `plugins` | numpy, pydantic, stdlib |
| `survey`, `projection`, `distribution` | the row above, narwhals, scipy, contourpy; `survey.io` may import pandas, polars, pyarrow and openpyxl (file reading and Excel export); `survey.stats` may import pandas and statsmodels (conversion on the way in) |
| `spi`, `design`, `predict`, `audio`, `datasets` | the rows above (not each other, except `design` → `spi`, `predict` → `audio` contracts); `datasets` and `audio.batch` may import pyarrow and pandas for table I/O |
| `plot` | everything above, matplotlib; `plot.timeseries` and `plot.channels` import `audio` lazily inside the function |
| `project`, `cli` | everything, including pyarrow and pandas for table I/O; `audio` and `predict` imported lazily inside the functions that need them |

The import-linter configuration lives in `pyproject.toml` as one `layers` contract
plus `forbidden` contracts with `include_external_packages = true` for pandas, polars
and matplotlib, with `ignore_imports` for exactly the boundaries named above.

Draft 0.1 considered a uv workspace of several packages and rejected it for release
and CI cost; that still holds. The seams above are kept so that `audio` or `predict`
could be split into their own distributions later without renaming imports.

Extras (`pyproject.toml` `[project.optional-dependencies]`): `audio` (mosqito,
scikit-maad, soundfile), `phonometry` (phonometry; Python 3.13 and later), `optim`
(pymoo), `polars` (polars and fastexcel, for users who want polars frames back),
`all`. The core dependency set is enumerated in §16.2; matplotlib is core because
`plot` is an M1 module. Optional dependencies are handled the way current Python
practice recommends (owner, 2026-10-30): the import lives inside the function that
needs it and raises one `MissingExtraError("install soundscapy[audio]")` at call
time, never at module import, so `import soundscapy`, `Project.open()` and `plot`
work on a course laptop without the audio extra; `audio` and `predict` are explicit
subpackages a user imports when they want them.

---

## 5. API design

Two tiers. The **narrative API** at the top level is about twenty names, enough for
every tutorial and for a GUI's command set. The **module API** is complete and is the
reference documentation. The top level imports the M1 and M2 modules eagerly with an
explicit `__all__` (numpy, scipy and pydantic are the import cost either way, and
eager names are what ty and great-docs' static discovery can see); `audio` and
`predict` are subpackages imported explicitly, so the audio stack never loads unless
asked for. If import time ever matters, `lazy_loader` with a `.pyi` stub is the
accepted pattern, not a bare `__getattr__`.

Journey A, C and E1 in the narrative API, as a tutorial would show them:

```python
import soundscapy as ssp

ds = ssp.read_survey("export.xlsx", mapping="mapping.toml")      # pandas by default; backend="polars" to choose
                                                                  # or ssp.suggest_mapping(df, "iso12913-outdoor")
report = ssp.validate(ds)                                         # ValidationReport; nothing dropped
ds = ssp.apply_quality_rules(ds)                                  # adds included, exclusion_reason

proj = ssp.project(ds, profile="soundscape-across-languages-2026")   # per response; ProjectionResult
coords = proj.coordinates                                         # Coordinates, origin.kind == "survey"
ssp.describe(coords, by="location")                               # tidy table: n, mean, sd, quantiles, quadrant %
camden = coords.subset(coords.labels["location"] == "Camden")
camden.fraction_in(ssp.regions.quadrant("+", "-"))                # 0.41; "calm" is an alias on the ISO instrument

ax = ssp.plot.circumplex(coords.circumplex)                       # styled Axes
ssp.plot.density(camden, fraction=0.5, ax=ax, label="Camden")     # SmallSampleError below 30 unless overridden
ssp.plot.by_group(coords, by="location", layers=["scatter", "density"])   # scatter only for small groups

target = ssp.targets.load("iso12913-4-example6-tranquil")         # or ssp.SkewNormal(xi=..., omega=..., alpha=...)
score = ssp.spi(camden, target, rng=0)                            # SPIResult(value=56, n_test=..., statistic=...)

brief = ssp.Brief(ambition="…", target=target, kpi=ssp.KPI(spi_min=70), ...)
post = ssp.Scenario(name="post", inputs=..., produced_by=survey_ref, intervention_level="source")
result = ssp.evaluate(brief, baseline=camden, scenarios={post: post_coords})
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
  specific condition has its own class so tests and GUIs can catch one thing; a
  warning fires once per operation, never per row, and never for a choice the user
  made deliberately.
- Informational output goes through the `soundscapy` logger (principle 9); the
  package never prints.
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
| Types | ty; `py.typed` shipped | House style. Pydantic models and plain functions type well; frames are typed as narwhals' `IntoFrame`/`Frame` with documented columns, not as typed frames. |
| Settings and specs | pydantic v2 frozen models | JSON schema for free, which is what a GUI form and the settings-file docs are generated from. Beat dataclasses + hand-written loaders. |
| Settings files | TOML written by `tomli-w`, read by `tomllib` | Human-editable with comments, standard library reader, no YAML dependency. JSON for machine-written manifests and result sidecars (§15, #7). |
| Tables | narwhals (`narwhals.stable.v2`) at the boundary; numpy inside kernels; pandas and polars tested in CI; pyarrow for parquet results | Owner decision. Functions accept any narwhals-supported eager frame and return the same kind. A spike in this session showed the boundary pattern (select mapped columns to numpy, compute, `with_columns` back) is a dozen lines and identical for pandas and polars. Two pandas-only boundaries remain and are documented: statsmodels (convert on the way in) and `read_survey` for Excel (pandas with openpyxl by default, polars with fastexcel via the `polars` extra). No backend-specific code anywhere else; an import-linter contract keeps `pandas` and `polars` imports out of every module except `survey.io` and `survey.stats`. |
| Numerics | numpy, scipy (stats, signal, optimize), contourpy | Standard. |
| Statistics | scipy.stats and statsmodels; thin `tidy()` helpers returning statistic, df or N, p, effect size | FR-2.6, FR-2.6a. pingouin was considered and rejected as a dependency for a few effect sizes that are ten lines each. |
| Audio | scipy.io.wavfile core; mosqito, scikit-maad, soundfile in the `audio` extra; phonometry in its own `phonometry` extra (3.13+) | §3.5 and Appendix C. Licence recorded per backend, not a filter; no PyQt6. |
| Logging | stdlib `logging`, package logger with a default INFO stream handler, `set_log_level()` | PRD NFR-13; principle 9. |
| Optimisation | pymoo in the `optim` extra | FR-6.5 only. |
| Data fetching | pooch | Standard in scientific Python; hash checks, cache, licence message. |
| Progress | `ProgressReporter` protocol with `start`, `advance`, `message`, `finish`; default reporter on `tqdm.auto` (core dependency, pure Python), which degrades silently and never warns | Notebook users see a widget, terminal users a bar, GUIs bind their own, tests bind a recorder. |
| Plotting | matplotlib; seaborn only where it saves real code (stacked bars, palettes) | §10. |
| CLI | click | Owner decision: great-docs documents click commands automatically (`cli: enabled: true` in `great-docs.yml`, discovered from `soundscapy.cli`), so the CLI reference is generated like the API reference. |
| Tests | pytest, hypothesis, pytest-cov, pytest-xdist, polars in the `test` group; `filterwarnings = error` for `SoundscapyWarning` and the package's own code, with narrowly scoped, commented per-module ignores pre-authorised for third-party `DeprecationWarning`s (statsmodels, matplotlib, mosqito) so an upstream release does not break CI; every expected `SoundscapyWarning` is asserted with `pytest.warns` | House style plus §8. |
| ruff with pydantic | `[tool.ruff.lint.flake8-type-checking] runtime-evaluated-base-classes = ["pydantic.BaseModel"]` from day one; expected global ignores, each with a reason: `EM101/EM102/TRY003`, `FBT001/FBT002` (flags are keyword-only instead), the `PD` family (fires on narwhals code), `PLR0912/PLR0915` on CLI commands; `PLR2004` stays on; never `slots=True` with `cached_property` | House style allows commented ignores; this pre-empts the first contributor fighting the Stop hook. |
| CI matrix | PRs: Linux on the floor and ceiling Pythons plus Windows on one; merges and nightly: 3 OS × every supported Python, `test:reference`, pandas 2 and 3; document a tested Python ceiling | A part-time maintainer should not wait on twelve jobs per push. |
| Architecture checks | import-linter contracts in `task lint` | §4. |
| Docs | great-docs; Quarto `.qmd` tutorials; Diátaxis layout | House style; FR-12. |
| CI | GitHub Actions running `task` targets; 3 OS × supported Pythons; docs executed; nightly slow audio and dataset tests; release on tag by trusted publishing | NFR-8. |
| Changelog and versioning | SemVer from 1.0; Keep-a-Changelog with a "Scientific changes" section; `CITATION.cff`; Zenodo DOI per release | FR-13.4, NFR-11. |

---

## 7. Where each programming style is used

This answers the "where OOP, where functional, where concurrency" question directly.

| Style | Used for | Not used for |
|---|---|---|
| Pure functions over arrays and frames | Projection kernels, statistics, KDE geometry, KS distance, SPI, shift, skew-normal fitting, quality rules, calibration gain, batch work units | Anything that needs to remember state between calls |
| Frozen value objects (pydantic) | Scale, Circumplex, AngleSet, Profile, Mapping, Calibration, MetricSpec, Target, Brief, Scenario, all settings | Carrying numpy arrays or frames (pydantic serialisation would fight them) |
| Frozen result records (dataclasses) | Dataset, Signal, Coordinates, KernelDensity, Prediction, every `*Result` | Mutable accumulation |
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
   (`scipy.stats.gaussian_kde`); its `bw_adjust` converts to `KernelDensity.bandwidth`
   as §3.3 states, so the paper's settings can be stated exactly.
3. **Cross-implementation references** (`tests/reference/`, marked `reference`, run in
   CI nightly and on demand). Skew-normal density, conversions, sampling and fitting
   against committed R `sn` outputs (generated once by a script kept beside them;
   the 0.x branch `157-remove-r-sn-dependency-for-spi` supplies the first baseline
   and the pattern); ISO 532-1 reference
   signals against MoSQITo (to catch a wrapped-library regression); selected metrics
   against ArtemiS tables where shareable (FR-4.10).
4. **Golden pipelines** (`tests/regression/`, marked `regression`). ISD subset →
   validate → project → describe → SPI, compared to committed baseline tables with
   per-column tolerances by significant figure; each baseline file records the
   wrapped-library versions it was generated with (NFR-4). A drift inside tolerance
   passes; a drift outside fails and must be explained in the changelog with a method
   version bump. The 2022 reproduction adds a golden table of its quadrant
   percentages here, so a change in that page's numbers is at least noticed even
   though the paper has no numeric fixture.

The paper reproductions (FR-9.3) are `.qmd` documents in `docs/papers/`, a great-docs
custom section beside `docs/tutorials/`. They need the full ISD from Zenodo, so they
are frozen (`docs/_freeze/`) and re-executed nightly and on demand, not by every
`task docs`, which stays offline for tutorials.

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
  `start(total)`, `advance(path)`, `message(text)`, `finish()`; `CancelToken` with
  `cancelled()`. The library checks the token between work units (FR-14.3) and, when
  it fires, raises `CancelledError` carrying the partial result. The default reporter
  wraps `tqdm.auto`, so a notebook shows a widget and a terminal a text bar with no
  setup; `progress=None` silences it; the CLI may bind rich instead; a GUI binds its
  own; tests bind a recorder. Cancellation in a notebook is the kernel interrupt,
  which the runner catches between work units and turns into the same partial result.
- **Dataset downloads** (FR-10.1) announce what is fetched and under which licence
  through the `soundscapy` logger before fetching, and `describe()` returns the
  licence text as data. pooch's downloader has no cancellation hook; downloads are
  therefore documented as not cancellable, and a custom downloader is added only if a
  GUI needs it. At M1 and M2 the only long operations are downloads, SPI rankings
  over many sites and skew-normal fitting; the last two are seconds and take neither
  protocol.
- **Target search** (FR-6.5) uses the two protocols around pymoo's evaluation loop.
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
   `plot.spi_annotation`, `plot.likert`, `plot.radar`, `plot.timeseries`,
   `plot.channels`. Each takes `language=` (default: the instrument's reference
   language) for attribute, axis and quadrant labels (NFR-10), draws on
   an `Axes`, returns it, and forwards `**kwargs` to the artist. Faceting is
   `plt.subplots` or seaborn's `FacetGrid` with the user's own code; the package does
   not own subplot layout. A `plot.by_group(coords, by="location", layers=[...])`
   convenience covers the common "one panel per location" case in a few lines, and
   draws scatter with an annotation for any group below the N < 30 rule (FR-3.2).
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
├── project.toml            # name, package version, project-format version; nothing else
├── mapping.toml            # SurveyMapping (FR-1.3), saved and reused across exports
├── settings.toml           # the Settings model (the user's input; a result's JSON sidecar is authoritative)
├── data/
│   ├── raw/export.xlsx     # the user's export, byte-for-byte, hash in manifest.json
│   ├── responses.csv       # the coded, normalised table the package reads
│   ├── responses.schema.json   # always written: dtypes, sheet, encoding, source hash
│   ├── groups.csv, sessions.csv, locations.csv      # optional
│   └── recordings.toml     # RecordingIndex (M3): paths, formats, calibration refs (no hashes)
├── results/                # tables the package computes: parquet, plus a JSON record each
│   ├── projection/<run>/   # coordinates.parquet + projection.json; one run per profile or settings
│   ├── survey/<run>/       # validation, descriptives
│   ├── audio/<run>/        # results.parquet + units.json + manifest.json + cache/*.parquet
│   ├── spi/<run>/          # spi.parquet + spi.json
│   └── evaluation/<run>/
├── exports/                # CSV written by every CLI command by default, Excel on request (FR-9.1)
│   └── figures/            # png, svg, pdf
└── manifest.json           # hashes of inputs, versions, timestamps for the whole project
```

The M1 directory is `project.toml`, `mapping.toml`, `settings.toml`, `data/raw/`,
`data/responses.*`, `results/survey/`, `results/projection/`, `exports/` and
`manifest.json`; nothing for recordings until M3.

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
- This satisfies FR-14.2 (every result serialisable to a documented text format):
  each computed table has a one-call CSV export and a JSON record of its settings,
  and parquet is a documented, open, inspectable format. Because a consultant shares
  the folder with people who open things in Excel, every CLI command also writes the
  CSV copy to `exports/` by default; the library's `save()` does not.
- **Figures** (FR-9.1): `exports/figures/<name>.{png,svg,pdf}`, written by
  `plot.save(fig, name)` or the CLI; every tutorial and CLI journey writes the three
  Part 3 visualisations there. `describe()` and the `ValidationReport` are results
  like any other and live under `results/survey/<run>/`.
- **The user's export is never rewritten.** `import` copies it byte-for-byte to
  `data/raw/`, records its hash, and writes the coded table beside a schema sidecar
  that is always present (dtypes, the sheet used, encoding, source hash), so an id
  with leading zeros or a date survives the round trip and the format is uniform. A
  workbook with more than one sheet needs `--sheet`; the command lists them.
- **Runs are keyed by name**, so a consultant comparing the 2019 and 2024 profiles
  keeps both projections in one project.
- `Project.open(path, backend="pandas")`, `.save()`, `.results["projection"][run]` is
  the one stateful class; everything it holds is the frozen records of §3.8.
- Settings files (FR-11.3) are the TOML form of the pydantic settings model, with the
  JSON schema published in the docs so a GUI can generate forms and validate edits.

---

## 12. GUI readiness checklist

The core meets FR-14 by construction if the following hold, each of which has a test:

- No function in `src/` calls `show`, `print`, `input` or writes a file except
  `project`, explicit `save`/`export` methods, and the pooch download cache used by
  dataset loaders and model-weight fetches (NFR-5: explicit downloads, announced
  through the logger). Everything a user sees arrives through `logging`, a warning
  class or an exception, so a GUI captures all of it with one handler.
- Every settings model and every result record round-trips through its text form.
- `audio.analyse`, `spi.search` and dataset downloads accept a `ProgressReporter`
  and a `CancelToken`.
- Each CLI command for journeys A, C, E1 (M1–M2), B (M3) and E2 (M4) reads and writes
  only the project directory, so "can the CLI do it?" is the GUI-readiness test, and
  every command has a non-interactive mode (FR-14.1).
- JSON schemas for `Mapping`, `Settings`, `Target` and `Brief` are published, which is
  what a form-driven GUI needs first.

Whether the GUI runs in-process (Python, driving the library directly) or
out-of-process (driving the CLI and the project directory) is left open, as the PRD
requires; both are served.

---

## 13. Where this plan departs from draft 0.1 or settles an open point

Points where the plan disagrees with draft 0.1, interprets the PRD, or settles a
question the literature leaves open. None changes the science.

1. **No pandera, no schema DSL.** The mapping is the schema (§3.2). Draft 0.1's
   pandera layer would have duplicated it.
2. **Positions over names** (§3.1) reverses draft 0.1's "names, never positions". The
   PRD's generalised circumplex makes positions the stable key.
3. **No `PlotSpec` yet** (§10). Draft 0.1 proposed one for the GUI; the PRD fixes no GUI
   design, and geometry plus settings already give a GUI what it needs.
4. **CSV for hand-made tables, parquet for computed ones** (§11), rather than one
   format for everything: the folder stays readable where people look, and fast and
   typed where the package looks.
5. **SPI reports the published single-seed number**, with the statistic variant, tie
   rule, truncation and test-sample N on record and an optional multi-seed summary
   (§3.4); the seed is required so reproducibility is explicit rather than
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
| 0. Toolchain | `pyproject.toml`, `Taskfile.yml`, prek, CI, docs skeleton, import-linter contracts, `GOVERNANCE.md`, `CITATION.cff`, `CONTRIBUTING.md`, ADR-0001 (functional core and records), ADR-0002 (positions, mappings, coding and id roles), ADR-0003 (Coordinates as the central type), ADR-0004 (settings files and project directory) | `task check` and `task docs` green on an empty package |
| M1. Survey analysis | `instrument`, `survey` (mapping suggestion against the SSID and REDCap schemas only), `projection`, `distribution` (Coordinates, regions, describe, KDE geometry), `plot` basics, `datasets.isd` with bundled subset, `project` minimal directory, canonical layout as a CSV template (FR-1.2), CLI `import`, `validate`, `project`, `describe`, `plot`, `export`, install quick-start page (FR-12.1), tutorial A, reproduction of the 2022 paper with its golden table | Journey A done-when; the 2022 paper's `.qmd` renders its figures and tables from the ISD with every cell run, following the paper's methodology and the same KDE estimator seaborn wraps |
| M2. SPI and design (1.0a1) | `distribution.skewnormal` (SkewNormal, fit, `sn` fixtures) and `ks2d`, `spi` with targets library, `design` with Brief/Scenario/evaluate, CLI for C and E1 plus `init`, `template` (Excel too), `map`, `targets`, `plugins`, plugin discovery, recipes for plot type and dataset loader, tutorials C and E, ISD feature table in the loader, course acceptance cases 1, 2 and 3, reproduction of the 2024 paper with the SPI acceptance test last, versioning policy, first alpha release | The 2024 paper's tables reproduce within tolerance; the Example 6 tutorial runs; course exercises run unchanged |
| M3. Audio | `audio` (Signal as a window, windowed I/O, calibration, equalisation, built-in level metrics and IACC, MoSQITo and scikit-maad backends, `phonometry` extra, batch runner with time-series cache, manifest), CLI for B, metric recipe, tutorial B, ArtemiS comparison page | Journey B done-when; ISO 532-1 reference signals pass; a 1,000-file batch resumes; Linux-only full backend matrix |
| M4. Prediction | `predict` (protocol, specs, discovery without loading, the default predictor the owner chooses at M4 planning, LinearPredictor for the 2021 reproduction), `soundscapy-soundaqnet` plugin, E2 in `design`, CLI for E2, predictor recipe, reproduction of the 2021 paper | Journey D and E2 done-when |
| M5. Ecosystem | SATP, ARAUS, DeLTA loaders with mappings; MOSAIQ import/export; target search behind `optim`; contributor trial of each recipe | Journeys F, G, L done-when |

---

## 15. Decisions taken by the owner

All decisions are now taken (Andrew Mitchell, 2026-10-30, two review rounds) and PRD
0.6 reflects them. They are recorded here so the spec and plans can cite them. Two
smaller calls made during the builder check (coding lives in the mapping; id roles are
open beyond the SSID hierarchy, §3.2) are accepted by the owner's approval of this
draft and become ADR-0002 material at Phase 0.

| # | Decision | Where it lands |
|---|---|---|
| 1 | Copyleft dependencies are acceptable; licence is recorded per backend, not a filter | §3.5, Appendix C; PRD 0.6 NFR-7. The open-core rationale behind the old restriction is not abandoned: because every backend records its licence, a future commercially licensed GUI can be shipped without the `audio` extra or with the AGPL backend swapped, and the register makes that cost visible rather than hidden |
| 2 | Dataframe-agnostic through narwhals, confined to the boundary; pandas and polars tested | §2, §3.2, §6 |
| 3 | click for the CLI, documented by great-docs | §4, §6 |
| 4 | Progress visible in notebooks through a default `tqdm.auto` reporter | §2, §9 |
| 5 | Python floor stays 3.11; phonometry ships as a 3.13-only optional backend (since the second review: as its own extra, never an environment marker; #17) | §3.5, Appendix C |
| 6 | pyarrow accepted; CSV for hand-made tables, parquet for computed tables, CSV and Excel export on demand | §11 |
| 7 | TOML for settings and mappings; JSON for machine-written records | §6, §11 |
| 8 | No SPI uncertainty machinery; the paper's sample size is the function default and the seed is required | §3.4; PRD 0.6 FR-6.1 and journey C |
| 9 | Missing attribute items exclude the response by default; renormalisation is an option under the canonical formulation only | §3.1, §3.2; PRD 0.6 FR-2.5a |
| 10 | The 2022 paper is reproduced by methodology, not by a numeric baseline; same KDE estimator seaborn wraps | §8, §14; PRD 0.6 journey I |
| 11 | Governance: Andrew Mitchell cuts releases and reviews every change that affects scientific results; cadence per milestone (§14) | PRD 0.6 NFR-12; `GOVERNANCE.md` at Phase 0 (§14) |

Decisions on the review of PRD 0.6 and plan 0.3 (`design/reviews/prd-0.6-plan-0.3-review.md`,
owner responses of 2026-10-30; the review's finding numbers in brackets):

| # | Decision | Where it lands |
|---|---|---|
| 12 | Profiles named faithfully to their sources; the chapter profile is the default [1] | §3.1; PRD 0.7 §2, FR-2.9 |
| 13 | SPI acceptance from the 2024 paper's tables; Example 6 as a tutorial; statistic, tie rule and truncation named and recorded; optional multi-seed summary [2, 15] | §3.3, §3.4; PRD 0.7 journey C, FR-6.1 |
| 14 | FR-2.5a renormalisation a constrained Should, refused in ISO profiles [3] | §3.1, §3.2; PRD 0.7 FR-2.5a |
| 15 | Three-way override for uncovered and Low-tier languages, "exclude" recommended; one status column for every exclusion [4] | §3.1, §3.2, §16.3; PRD 0.7 FR-1.5, FR-2.4 |
| 16 | `SurveyMapping`; scales per construct group; coding order; string keys [5] | §3.2 |
| 17 | Circumplex fingerprints; versions on targets and angle sets, never removed; angle sets keyed by circumplex [6] | §3.1, §3.3; PRD 0.7 FR-6.3 |
| 18 | `Coordinates` rows aligned with responses, mandatory labels, one origin record, `describe(by=)`, no weights [7] | §3.3 |
| 19 | N < 30 decided at the drawing layer with scatter fallback [8] | §3.3, §10; PRD 0.7 FR-3.2 |
| 20 | `Settings` model defined; `Profile` is a preset; `metric_set` gone from M1 [9] | §3.1 |
| 21 | narwhals boundary follows narwhals' conventions, reviewed while building; the plan's commitments in §3.2 [10] | §3.2 |
| 22 | M1 re-scoped; `SkewNormal`, `ks2d`, plugins, recipes and most CLI commands to M2 [11, 25] | §14, §16.1; PRD 0.7 §9 |
| 23 | Table D.1 coverage not a requirement; MoSQITo the single `audio` backend; no PyQt6 dependency; phonometry an explicit extra; substitutions named [12, 13] | §3.5, §4, Appendix C; PRD 0.7 FR-4.4, FR-4.6 |
| 24 | Predicted single-response distributions warned and labelled [14] | §3.3; PRD 0.7 FR-5.9 |
| 25 | Library logs through `logging` with a visible default handler; never prints; downloads announced that way [16] | §2, §9, §12; PRD 0.7 NFR-13 |
| 26 | Warnings once per operation, none for deliberate choices; `filterwarnings` scoped; `CancelledError` [17] | §2, §6, Appendix A |
| 27 | Optional imports at call time per current practice; eager top-level API; `audio` and `predict` explicit subpackages [18] | §4, §5 |
| 28 | Raw export kept byte-for-byte; schema sidecar always written; runs keyed by name; CLI exports CSV by default [19] | §11 |
| 29 | `Signal` is a window; windowed I/O; hashes in `analyse` [20] | §3.5 |
| 30 | `evaluate` takes `Scenario`s; predictor specs discoverable without loading; FR-5.5 enforced by a validator; `MetricImplementation`; `SkewNormalFit`; fitting in centred parameters [21] | §3.4, §3.5, §3.6, §3.3 |
| 31 | No overlap statistic; shift against a target by closed-form mean [22] | §3.4; PRD 0.7 FR-3.3 |
| 32 | Predictor interface fixed now; the model chosen at M4 planning [23] | §3.6; PRD 0.7 journey D, FR-5.3 |
| 33 | Releases from M2 are 1.0 alphas; no API frozen at M2 [24] | §14; PRD 0.7 §9 |
| 34 | Skew-normal fitting stays a Must, at M2 [29] | §3.3, §14; PRD FR-3.4 |

### Inputs the plan still needs from the owner (from PRD §10.3)

The ISD language coverage against the SATP v1.5 tiers; the ISD record versions used by
each reproduced paper; machine-readable fixtures for the 2024 paper's targets and
sample size, and for Part 4 Example 6 if any exist; the 2-D KS variant, direction
combination, tie rule and truncation the 2024 paper used; the 2022 paper's KDE
settings and angle choice; a worked example of the canonical projection with adjusted
angles; which tonality each Table D.1 edition names; whether redistributable audio
exists; the features the 2021 deployable models use; and R `sn` available once to
generate the skew-normal reference fixtures (the 157 branch shows this has been done
before).

---

## 16. Consistency review and the M1 contracts

A cross-document review on 2026-10-30 (PRD 0.6, this plan, the 0.3 review, the agent
setup report, AGENTS.md, CLAUDE.md and the house-style skill) found no contradiction
in Python floor, licence, tutorial format, file locations, milestone assignments or
tool choices, and a set of stale pointers and gaps, all applied in draft 0.3. Three
of the gaps were contracts a builder would have guessed at M1; they are fixed here.

### 16.1 CLI commands (FR-14.5), by milestone

Every command reads and writes a project directory (`--project`, default the current
directory). Output tables go to `results/`, exports to `exports/`, figures to
`exports/figures/`. Each command maps to one library call so the CLI never contains
logic.

| Milestone | Command | Library call | Writes |
|---|---|---|---|
| M1 | `soundscapy import <export> [--sheet] [--mapping]` | `read_survey` (creates the project directory if absent) | `data/raw/`, `data/responses.csv` and schema sidecar |
| M1 | `soundscapy validate` | `validate`, `apply_quality_rules` | `results/survey/<run>/validation.*` |
| M1 | `soundscapy project [--profile] [--run]` | `project` | `results/projection/<run>/` |
| M1 | `soundscapy describe [--by location]` | `describe` | `results/survey/<run>/descriptives.*` |
| M1 | `soundscapy plot <scatter|density|contour|likert|radar> [--by]` | `plot.*`, `plot.save`; scatter for small groups | `exports/figures/` |
| M1 | `soundscapy export <result> [--format csv|xlsx]` | `Result.export` | `exports/` |
| M2 | `soundscapy init` | `Project.create` | `project.toml`, `settings.toml` from a profile |
| M2 | `soundscapy template [--format csv|xlsx]` | `templates.save` | the canonical layout (CSV at M1 via the docs) |
| M2 | `soundscapy map <export> [--interactive]` | `suggest_mapping`; non-interactive writes the suggestion with confidences, `--interactive` confirms per binding | `mapping.toml`, `mapping.suggestion.json` |
| M2 | `soundscapy spi --target <name|file> [--by] [--seeds N]` | `spi`, `rank_against` | `results/spi/<run>/` |
| M2 | `soundscapy targets <list|show|fit>` | `targets.*`, `fit_skew_normal` | `results/spi/<run>/targets/` |
| M2 | `soundscapy evaluate <brief.toml>` | `evaluate` | `results/evaluation/<run>/` |
| M2 | `soundscapy plugins` | `plugins.discover` for all groups | stdout |
| M3 | `soundscapy audio index <dir>` | `RecordingIndex.scan` (header read, no hash) | `data/recordings.toml` |
| M3 | `soundscapy audio calibrate` | `Calibration.from_*` | `data/recordings.toml` |
| M3 | `soundscapy audio analyse [--metrics <set>] [--jobs N] [--window]` | `audio.analyse` | `results/audio/<run>/` |
| M4 | `soundscapy predict --model <name>` | `predictors.get(...).predict` | `results/prediction/<run>/` |
| M4 | `soundscapy evaluate` gains predicted scenarios | `evaluate` | as M2 |

Every command writes a CSV copy of its result tables to `exports/` as well.

### 16.2 Core dependencies (NFR-1)

Core, all pure-Python or wheels on the three platforms: numpy, scipy, pydantic,
narwhals, pandas (default backend), pyarrow (parquet), openpyxl (Excel), contourpy,
matplotlib, statsmodels (FR-2.6, FR-2.6a), tqdm, pooch, tomli-w, click. Not core:
seaborn (used only if a stacked-bar or palette helper saves real code; otherwise
dropped), scikit-learn (`analysis` extra, FR-2.7), everything in the `audio`,
`optim` and `polars` extras, torch (plugin packages only). The install test at Phase
0 is a clean `pip install soundscapy` on Windows, macOS and Linux with no compiler,
followed by the quick-start path (FR-12.1).

### 16.3 Default quality rules (FR-1.5)

Applied in this order, each adding to `exclusion_reason` when it fires; all
configurable; none drops a row. Column roles come from the mapping's `extra` entries.

| Rule | Fires when | Reason string |
|---|---|---|
| `scale_violation` | any circumplex item outside the declared scale after coding (FR-2.1) | `scale_violation` |
| `all_identical` | all present circumplex items equal (including all-neutral, which is flagged but, as in the 2021 paper, not excluded unless the user opts in) | `all_identical` |
| `excessive_missing` | more than half of the circumplex plus source-dominance items missing (the 2021 paper's rule); threshold configurable | `excessive_missing` |
| `under_age` | an `age` role column below the configured minimum (default 18) | `under_age` |
| `no_consent` | a `consent` role column not affirmative | `no_consent` |
| `test_response` | a `test_flag` role column set | `test_response` |
| `paper_input` | a `paper_input` role column set (kept by default; flagged for review) | `paper_input` |

The projection adds `incomplete_attributes` (FR-2.5a), `uncovered_language` and
`low_confidence_language` (FR-2.4, FR-2.4a) to the same two column names on
`Coordinates.labels`.

## 17. Review findings not yet decided

The minor findings of the 0.6 review (26 to 28 and 30 to 40 in
`design/reviews/prd-0.6-plan-0.3-review.md`) await the owner and are listed in PRD 0.7
§10.6. The ones that touch this plan: statsmodels as core or an `analysis` extra
(§16.2 keeps it core until decided); Windows path rules for the audio cache; the
mixed-language label rule for plots; timestamps and time zones in the session and
recording tables; a supported audio format list. None blocks the Phase 0 / M1 spec.

---

## Appendix A. Exception and warning hierarchy

```
SoundscapyError
├── MappingError            column missing, duplicate binding, unknown position
├── ScaleError              value outside declared scale, wrong coding
├── LanguageError
│   ├── UncoveredLanguageError      no angle set for a response's language (override: on_uncovered="exclude" | "equal")
│   └── LowConfidenceLanguageError  Low-tier language refused (override: on_low_confidence="exclude" | "equal")
├── SmallSampleError        density *plot* requested for N < 30 (override: allow_small_sample)
├── NotAdmissibleError      centred parameters outside the admissible region
├── CircumplexMismatchError target, angle set or coordinates on a different circumplex (by fingerprint)
├── UncalibratedSignalError metric needs Pa, signal has no calibration
├── AudioFormatError        unsupported file, more than two channels, unreadable header
├── MissingInputError       predictor's required input absent
├── MissingExtraError       optional dependency not installed, names the extra (raised at call time)
├── CancelledError          a long operation was cancelled; carries the partial result
└── ProjectError            project directory malformed or from an unsupported version

SoundscapyWarning            each fires once per operation, never per row
├── MediumConfidenceWarning Medium-tier languages projected (lists them)
├── MixedAngleSetsWarning   equal and adjusted angles in one projection under an override
├── SmallSampleWarning      density geometry computed with N < 30
├── RenormalisedWarning     responses projected over a subset of items (FR-2.5a)
├── PredictedPointsWarning  a distribution built from single-response predictions (FR-5.9)
└── OptionalInputWarning    predictor ran without an optional input
(no warning for selecting a legacy formulation: the choice is recorded in the result)
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
| phonometry 3.3.0 | MIT | yes; **`>=3.13`**; numpy, scipy | ISO 532-1/2/3, ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength; DIN 45692; Osses FS; PA; ECMA-418-1; ISO 1996-2; plus levels, weighting, octave bands | newly released 2026; one maintainer; 1,893 self-reported conformance checks in CI; DOI | **Optional backend in its own `phonometry` extra, 3.13+** (§15, #5 and #17) |
| scikit-maad 1.5.2 | BSD-3 | yes; `>=3.9`; numpy, scipy, pandas, matplotlib, scikit-image, pywavelets | 50+ ecoacoustic indices, SPL | active, maintained | **Backend for FR-4.5 ecoacoustics**, `audio` extra |
| acoustic-toolbox 0.2.2 | BSD-3 | yes; `>=3.11`; numba, pysoundfile, pyoctaveband, matplotlib, pandas | weighting, Leq, octave bands, ISO 1996-2 tonal audibility | active fork of python-acoustics (archived) | Not used: level metrics are built in without numba; revisit for ISO 1996-2 |
| sottek-hearing-model 0.1.14 | AGPL-3 (PyPI), GPL-3 (GitHub) | yes; `>=3.11`; numpy, scipy, bottleneck, matplotlib, **PyQt6**, tqdm | ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength | active; translated from the SQAT authors' verified MATLAB | **Not adopted** (owner, 2026-10-30): PyQt6 hard dependency, 0.1.x, licence ambiguous; revisit if PyQt6 becomes optional |
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

Gaps with `soundscapy[audio]` on Python 3.11 and 3.12: fluctuation strength,
ECMA-418-2 roughness and tonality, Aures tonality (only in an unmerged MoSQITo pull
request, in unlicensed pySQAT, and in experimental MetaSona) and psychoacoustic
annoyance; the `phonometry` extra on 3.13+ closes all but Aures tonality. Coverage is
not a requirement (PRD 0.7 FR-4.4). These are documented, and the metric registry
means a later package, a licence change in pySQAT, a MoSQITo release, or
sottek-hearing-model dropping PyQt6 closes a gap by adding one implementation class.
