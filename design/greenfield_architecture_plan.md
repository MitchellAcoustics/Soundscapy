# A greenfield architecture for an open-source soundscape engineering package

Status: **draft for review** (v0.1, 2026-10-30). Nothing below is fixed except the science.

This document plans a scientific Python package for soundscape assessment, analysis,
predictive modelling, scoring and design, built from scratch. It deliberately ignores
the design decisions embodied in the current `soundscapy` codebase and treats only the
scientific concepts in the literature as given. Where the current package is mentioned
it is as a cautionary example, not a constraint.

Source material: the SSID protocol (Mitchell et al. 2020), "How to analyse and
represent quantitative soundscape data" (Mitchell, Aletta & Kang 2022), the lockdown
predictive-modelling paper (Mitchell et al. 2021), the predictive-model framework paper
(Mitchell et al. 2023), the Soundscape Perception Indices paper (Mitchell et al. 2024),
the Soundscapy v0.6 paper (Mitchell 2024), the DCNN-CaF annoyance paper (Hou et al.
2023), the SoundSCaper captioning paper (Hou et al. 2026), and the soundscape datasets
review (Liang et al. 2026).

---

## 1. What the science requires of the software

Reading the nine papers as a requirements document, the package must support five
connected activities. Each has a distinct data shape, and the architecture follows
from getting those shapes right.

| Activity | Input shape | Core computation | Output shape | Source |
|---|---|---|---|---|
| Survey analysis | One row per response, hierarchically keyed (location, session, group) with 8 PAQ items plus source-dominance, overall-quality, WHO-5 and demographic items on documented scales | Validation, circumplex projection with language-specific angles, descriptive and distributional statistics | Per-response coordinates in [-1, 1]²; per-group distributions | SSID 2020; JASA-EL 2022; ISO/TS 12913-3; Aletta et al. 2024 |
| Psychoacoustic analysis | Calibrated binaural recordings (typically 30 s, 44.1 kHz, 24-bit), one per response group | Standardised metrics (ISO 532-1 loudness and sharpness, ECMA-418-2 roughness/impulsiveness/fluctuation strength, tonality, LAeq, LCeq−LAeq, LA10−LA90, ecoacoustic indices) with percentile statistics, max-of-channels rule | One tidy row per recording (or per channel) | JASA 2021; Soundscapy 2024; ISO/TS 12913-3 |
| Predictive modelling | Feature table joined to responses via group id; optional location-level context | Multilevel or flat regression; deep audio models (DCNN-CaF, SoundAQnet) | A *distribution* of perception per location, not a point | JASA 2021; INTER-NOISE 2023; Hou 2023, 2026 |
| Scoring (SPI) | Two distributions in circumplex space: a test (observed or predicted) and a target (bespoke or reference) | Bivariate skew-normal parameterisation and sampling, 2-D Kolmogorov–Smirnov distance, SPI = 100(1 − D); NSGA-II search for reference targets from rankings | Scalar 0–100 per (test, target); target parameter sets | JASA 2024 |
| Design | Existing and predicted distributions plus a target region or distribution | Compare scenarios, "percent likely to perceive", shift vectors | Ranked interventions, plots, reports | JASA-EL 2022; INTER-NOISE 2023 |

Three cross-cutting facts shape everything else:

1. **The unifying object is a distribution of points in the circumplex.** Survey
   responses for a location, a predictive model's output, an SPI target, and a design
   goal are all "a bivariate distribution over (pleasant, eventful)". The package
   should have exactly one abstraction for this, and survey analysis, modelling,
   scoring and design should all consume and produce it.
2. **Data are hierarchical and multimodal, and datasets disagree.** The SSID
   hierarchy is Location → Session → Group → Response, with recordings at group level
   and time-series logs at session level. ARAUS, SATP, DeLTA, HSDD and others use
   different scales, item wordings, participant–stimulus mappings and collection
   settings. A harmonisation layer with explicit codebooks is required, not optional.
3. **Reproducibility is a scientific deliverable.** Angles, scale ranges, KDE
   bandwidths, contour percentiles, target parameters, random seeds, metric settings
   and calibration levels all change the numbers. Every result must carry the
   specification that produced it.

---

## 2. Design principles (and what they rule out)

1. **Functional core, thin object shell.** All computation lives in pure, stateless,
   picklable functions over numpy arrays and DataFrames. Objects exist only as
   immutable *specifications* (what to compute) and *results* (what was computed),
   never as stateful "managers" or "analysers". This gives testability, trivial
   multiprocessing, and a serialisable surface a GUI can drive.
2. **Specs are data.** Every configurable thing (circumplex angles, Likert scales,
   metric settings, targets, plot layouts) is a frozen dataclass or pydantic model
   with a stable JSON/YAML form. If a spec cannot round-trip through JSON it is
   wrongly designed.
3. **Names, never positions.** Perceptual attributes are referred to by canonical
   names (`pleasant`, `vibrant`, ...), never `PAQ1..PAQ8`. The SSID codebook order
   (paq01 = pleasant, paq02 = chaotic, ...) is not the circumplex order, and that
   mismatch has caused real bugs. A codebook maps dataset columns to canonical names.
4. **Compute geometry separately from drawing it.** KDE grids, probability-mass
   contours, quadrant fractions and shift vectors are computed by the core and
   returned as data. Drawing is a thin matplotlib layer. A GUI, a plotly backend or a
   report generator renders the same geometry.
5. **No inheritance from third-party classes, no deep hierarchies.** Composition and
   `typing.Protocol` for extension points. Backends (MoSQITo, scikit-maad, torch
   models) are adapters behind a protocol, never base classes.
6. **Build on the ecosystem.** numpy, scipy, pandas, pandera, pydantic, matplotlib,
   seaborn, statsmodels, soundfile, pooch, MoSQITo, scikit-maad, pymoo. Write only
   what does not exist: the circumplex geometry, the skew-normal machinery, the 2-D
   KS distance, the SPI framework, and the harmonisation layer.
7. **No R.** The R `sn` dependency via rpy2 is the single largest installation and
   maintenance burden in the current package. The skew-normal maths (direct/centred
   parameter conversion, sampling, maximum-likelihood fit) is a few hundred lines of
   scipy and is implemented in Python, validated once against `sn` in a one-off
   script kept under `tests/reference/`.
8. **Headless by construction.** No `plt.show()`, no `print`, no global mutable
   state, no implicit file writes in the core. Progress is reported through
   callbacks. Logging uses the standard library (a library should not impose loguru
   on its users).
9. **Volunteer-maintainable.** Small modules with one responsibility, strict typing,
   architecture decision records, and a public API small enough to document fully.

---

## 3. The domain model

### 3.1 Perceptual response tables and codebooks

Survey data stay as pandas DataFrames. Researchers live in DataFrames, and every
downstream tool (seaborn, statsmodels, scikit-learn) expects them. The package does
not wrap rows in objects.

What the package adds is a **codebook**: a declarative description of what each
column means, validated with pydantic and serialisable to YAML.

```python
@dataclass(frozen=True)
class Scale:
    low: float            # numeric value of the lowest anchor
    high: float           # numeric value of the highest anchor
    kind: Literal["likert", "continuous", "sam", "vas"]
    anchors: tuple[str, ...] = ()
    reversed: bool = False          # True if high value = disagree

@dataclass(frozen=True)
class Item:
    column: str                     # column name in the DataFrame
    construct: str                  # canonical name: "pleasant", "annoying", "source_traffic", ...
    scale: Scale
    instrument: str                 # "ISO12913-2:MethodA", "SAM", "WHO-5", "SSID"
    language: str | None = None     # BCP-47, e.g. "en", "it", "zh"

@dataclass(frozen=True)
class Codebook:
    items: tuple[Item, ...]
    keys: Keys                      # names of id columns: response, group, session, location, recording
    mapping: Literal["one_to_one", "many_to_one", "one_to_many", "crossed"]
    setting: Literal["in_situ", "lab", "online", "mixed"]
    provenance: Provenance          # dataset name, version, DOI, licence
```

A `Codebook` is the contract between a dataset and the rest of the package. The ISD
loader ships one; a user with their own REDCap export writes one (or has it inferred
from column names and confirms it). Harmonising ARAUS, SATP or DeLTA to the canonical
constructs is then a codebook, not code. The datasets review (Liang et al. 2026)
recommends exactly this attribute-mapping discipline.

pandera validates DataFrames against the codebook at module boundaries (loaders,
projection, modelling). Validation is opt-out for speed but on by default.

Open question: pandas-only, or dataframe-agnostic via narwhals so polars users are
first-class? Recommendation: pandas is the public contract for v1; all numerical
kernels take numpy arrays, so the DataFrame layer is thin and could be made agnostic
later at low cost. Do not pay the narwhals abstraction tax before there is demand.

### 3.2 The circumplex specification

```python
@dataclass(frozen=True)
class Circumplex:
    attributes: tuple[str, ...]        # canonical names, any order
    angles_deg: tuple[float, ...]      # one per attribute
    axes: tuple[str, str] = ("pleasant", "eventful")

    @classmethod
    def iso_equal_angles(cls) -> "Circumplex": ...
    @classmethod
    def from_language(cls, lang: str) -> "Circumplex": ...   # Aletta et al. 2024 adjusted angles, packaged as data
```

The projection is the general form from Mitchell & Aletta (2023) and Aletta et al.
(2024):

P = (1/λ_P) Σ cos θ_i σ_i,  E = (1/λ_E) Σ sin θ_i σ_i,  λ_P = (ρ/2) Σ |cos θ_i|,
λ_E = (ρ/2) Σ |sin θ_i|, with ρ the scale range.

`project(values, circumplex, scale) -> ndarray[n, 2]` is a pure numpy kernel. The ISO
equal-angle case must reproduce the ISO/TS 12913-3 formula exactly (unit test). Two
scientific details to confirm before implementation and record in an ADR: (a) with
non-symmetric adjusted angles Σ cos θ_i ≠ 0, so a constant response vector does not
project to the origin unless responses are centred first; (b) the normalisation uses
λ_E for E (the Soundscapy paper prints λ_P in both equations, which is a typo).

### 3.3 Perception distributions: the central abstraction

```python
class PerceptionDistribution(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...   # (n, 2)
    def mean(self) -> np.ndarray: ...
    def fraction_in(self, region: Region) -> float: ...

@dataclass(frozen=True)
class Empirical(PerceptionDistribution):       # survey responses or model outputs
    points: np.ndarray
    weights: np.ndarray | None = None

@dataclass(frozen=True)
class KDE(PerceptionDistribution):             # explicit bandwidth, explicit grid
    points: np.ndarray
    bandwidth: float | Literal["scott", "silverman"]
    def density_grid(self, resolution: int = 200) -> DensityGrid: ...
    def mass_contours(self, levels: Sequence[float]) -> list[Polygon]: ...   # e.g. [0.5] = the "50th-percentile contour"

@dataclass(frozen=True)
class SkewNormal(PerceptionDistribution):      # Azzalini MSN, direct parameters
    xi: np.ndarray; omega: np.ndarray; alpha: np.ndarray
    @classmethod
    def from_centred(cls, mean, cov, skew) -> "SkewNormal": ...   # raises if not admissible
    @classmethod
    def fit(cls, points, *, method="mle") -> "SkewNormal": ...
    def to_centred(self) -> CentredParams: ...
```

`Region` covers half-planes, quadrants and arbitrary polygons (matplotlib.path, no
shapely). "Percent of people likely to perceive the soundscape as calm" is
`dist.fraction_in(Quadrant.CALM)`, whether `dist` is survey data, a model output, or a
target. Shift vectors between distributions, overlap of mass contours, and the SPI all
operate on this protocol.

### 3.4 Distances and the SPI

```python
def ks2d(a: np.ndarray, b: np.ndarray) -> float: ...                     # Fasano–Franceschini, vectorised numpy
def spi(test: PerceptionDistribution, target: PerceptionDistribution, *,
        n_samples: int = 1000, rng: np.random.Generator | int) -> SPIResult
```

SPI is stochastic because the target is sampled. The API therefore requires an
explicit seed or generator, exposes `n_samples`, and `SPIResult` records both plus the
standard error from repeated draws if requested. A deterministic variant (compare
against a fine quantile grid of the target) is worth prototyping and benchmarking
against the paper's Table III, but is not assumed.

Targets are specs:

```python
@dataclass(frozen=True)
class Target:
    name: str
    distribution: SkewNormal
    kind: Literal["bespoke", "reference"]
    provenance: Provenance        # who defined it, from what ranking, which optimiser run
```

A small registry of named reference targets ships as data files, versioned, each with
provenance. Deriving a reference target from an a-priori ranking (NSGA-II via pymoo,
objectives: Spearman rank correlation and rank-weighted SPI sum, selection by
distance to the ideal point) lives in an optional extra because pymoo is heavy and
the use case is rare.

### 3.5 Audio signals and metrics

```python
@dataclass(frozen=True)
class Signal:
    data: np.ndarray               # (channels, samples), float64, Pa after calibration
    fs: int
    channel_names: tuple[str, ...]  # ("L", "R")
    calibration: Calibration | None

@dataclass(frozen=True)
class MetricSpec:
    metric: str                    # semantic name: "loudness_zwicker_tv", "sharpness_din", "roughness_ecma", "laeq"
    backend: str | None = None     # "mosqito", "maad", "builtin"; None = default backend for this metric
    params: Mapping[str, Any] = field(default_factory=dict)
    statistics: tuple[str, ...] = ("p5", "p50", "p95", "mean", "max")
    channels: Literal["each", "max", "mean"] = "each"   # ISO/TS 12913-3 uses max of L/R

class Metric(Protocol):
    name: str; backend: str; units: str
    def __call__(self, signal: Signal, **params) -> MetricOutput: ...   # time series and/or scalar
```

Metrics are keyed by what they measure, not by which library computes them. The
current package keys settings by library (`settings['MoSQITo']['loudness_zwtv']`),
which couples user configuration to an implementation detail and makes swapping or
validating backends awkward. A registry maps `(metric, backend)` to an implementation;
users and configs name the metric.

Batch processing is a pure function over a list of files with an explicit executor,
per-file caching keyed by (file hash, config hash, package version), and a progress
callback. It is resumable by construction: re-running after a crash recomputes only
missing files. Results are one tidy row per (recording, channel) with columns
`<label>_<stat>` and a units table alongside.

### 3.6 Predictors

```python
class Predictor(Protocol):
    inputs: FeatureContract         # which metric columns / context fields it needs, with units
    def predict(self, features: pd.DataFrame, context: pd.DataFrame | None = None) -> PredictionResult: ...

@dataclass(frozen=True)
class PredictionResult:
    per_row: np.ndarray             # (n, 2) point predictions or (n, 8) PAQs
    distribution: PerceptionDistribution   # aggregated, with uncertainty
```

The framework paper's constraints become the contract: a predictor declares its
required inputs (must-have acoustic features, optional context, optional
demographics), must not require perceptual inputs, and must output a distribution.
Three families of implementation:

- `LinearPredictor` from a coefficient table (the published unscaled lockdown models
  are the first instances, with location random effects as an optional table).
- `MixedModel` fitted with statsmodels `MixedLM` (random intercepts and slopes by
  location), with the feature-screening and stepwise workflow from the 2021 paper as
  helper functions rather than hidden inside a class. Bayesian fitting via bambi is a
  candidate optional extra because it yields posterior predictive *distributions*
  natively, which is what the framework asks for.
- Deep audio models (DCNN-CaF, SoundAQnet) as **separate plugin packages**
  (`soundscapy-models-torch`) implementing the protocol, discovered via entry points.
  torch and pretrained weights do not belong in the core distribution.

### 3.7 The dataset container

```python
@dataclass(frozen=True)
class Dataset:
    responses: pd.DataFrame
    codebook: Codebook
    groups: pd.DataFrame | None = None       # one row per group: recording path, timestamps, photo
    sessions: pd.DataFrame | None = None     # session notes, SLM and environmental log paths
    locations: pd.DataFrame | None = None    # static context: typology, GPS, land use
    features: pd.DataFrame | None = None     # computed acoustic features keyed by recording
    audio: AudioIndex | None = None          # lazy index of files, never loaded eagerly
```

Loaders for ISD, SATP, ARAUS, DeLTA (and later HSDD, LCS) fetch from Zenodo or the
NTU dataverse with pooch (hash-verified, cached, licence shown on first download) and
return a `Dataset`. No dataset is bundled except a small ISD subset for tests and
docs. A canonical on-disk format for user projects, a directory of parquet tables plus
`codebook.yaml` and `manifest.json`, is proposed so a GUI and the CLI share a project
layout.

Quality checks (Erfanian et al. 2021; the 2021 JASA paper: drop all-identical PAQ
rows, drop rows with more than half the PAQ and source items missing, keep
all-neutral rows) are a declarative list of rules producing a mask and a reason
column, not a function that silently returns two frames.

---

## 4. Package and repository structure

Single installable distribution, `src/` layout, optional extras. A uv workspace of
several packages was considered and rejected for now: it isolates heavy dependencies
cleanly, but doubles release and CI work for a volunteer team. The module layout
below keeps the seams clean so that `audio` or `models` could be split into their own
distributions later without renaming imports.

```
soundscapy/
├── __init__.py          # narrative top-level API, lazily loaded (SPEC 1)
├── exceptions.py
├── circumplex/          # Circumplex spec, project(), regions, descriptive stats
├── surveys/             # Codebook, Item, Scale, schemas, quality rules, harmonise()
├── distributions/       # PerceptionDistribution, Empirical, KDE, SkewNormal, ks2d, shift vectors
├── spi/                 # Target, spi(), reference-target search (extra: optim)
├── audio/               # Signal, Calibration, metric registry, backends/, batch runner (extra: audio)
├── models/              # Predictor protocol, LinearPredictor, MixedModel, features, evaluation
├── design/              # scenarios, comparison, "percent likely", reporting tables
├── datasets/            # pooch registry, one module per dataset, each returning Dataset + Codebook
├── viz/                 # circumplex axes styling, scatter/density/contour drawers, likert, PlotSpec
├── io/                  # parquet/csv/xlsx readers, project layout, manifests, provenance
├── cli/                 # typer app: validate, project, analyse-audio, score, report
└── _data/               # angle tables per language, dataset codebooks (yaml), reference targets, test subset
```

Dependency direction is strictly downward: `viz`, `design`, `cli` depend on
everything; `spi` and `models` depend on `distributions` and `circumplex`; `audio`
depends only on numpy/scipy and its backends; `circumplex`, `distributions` and
`surveys` depend only on numpy, scipy, pandas and pydantic. An import-linter contract
enforces this in CI.

Extras: `[audio]` (soundfile, mosqito, scikit-maad), `[optim]` (pymoo),
`[bayes]` (bambi), `[all]`. Optional imports are gated at module level with a uniform,
actionable `ImportError`.

Repository:

```
.
├── pyproject.toml           # uv-managed; hatchling or uv_build backend
├── src/soundscapy/
├── tests/
│   ├── unit/                # fast, pure; mirrors src layout
│   ├── regression/          # golden-file numeric baselines (parquet) with tolerances
│   ├── integration/         # dataset loaders (network-marked), end-to-end pipelines
│   ├── reference/           # one-off cross-validation scripts (R sn, ISO 532-1 reference program, ArtemiS tables)
│   └── data/                # tiny wavs, ISD subset, synthetic fixtures
├── docs/                    # mkdocs-material + mkdocstrings; Diátaxis layout; executed notebooks
│   └── adr/                 # architecture decision records
├── benchmarks/              # asv or pytest-benchmark for audio and KDE
├── .github/workflows/       # test matrix, docs, release (trusted publishing), nightly slow tests
├── CITATION.cff, CHANGELOG.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, GOVERNANCE.md
```

---

## 5. Tooling

| Concern | Choice | Why, and what was rejected |
|---|---|---|
| Environment and packaging | **uv** with `pyproject.toml`, `uv.lock` committed, `hatchling` build backend | One tool for envs, locking, building and publishing. pixi was useful only because of R; without R, conda is not needed for development. A conda-forge feedstock is still produced for users. |
| Python versions | SPEC 0 window (currently 3.11–3.13) | Predictable support policy, matches the ecosystem. |
| Lint and format | **ruff** (lint + format), rule set `ALL` with a short, justified ignore list | Replaces black, isort, flake8, pydocstyle. The current repo has lint disabled in pre-commit; it must be enforced from the first commit. |
| Typing | **mypy strict** on `src/`, `py.typed` shipped; pandera and pydantic plugins | ty or pyrefly are promising but their pandas/pandera story is immature; reassess yearly. |
| Validation | **pydantic v2** for specs and configs, **pandera** for DataFrames | Both are standard; both serialise. |
| Tests | **pytest**, **hypothesis** (property tests on projection bounds, DP/CP round trips, KS symmetry), **pytest-xdist**, **pytest-cov**, **pytest-mpl** only for a handful of layout tests | Image comparison is brittle; prefer asserting on returned geometry. |
| Architecture checks | **import-linter** layer contracts | Prevents the "everything imports everything" drift that makes volunteer refactors scary. |
| Docs | **mkdocs-material** + **mkdocstrings-python** (numpy docstrings) + **mkdocs-jupyter** with notebooks executed in CI | One toolchain, pure Python. Quarto is excellent but adds a non-Python binary and a second site; rejected for now. |
| Docs structure | Diátaxis: tutorials, how-to, reference, explanation; plus ADRs | Explanation pages carry the science (ISO 12913, circumplex, SPI). |
| CI | GitHub Actions: 3 OS × SPEC-0 Pythons, uv cache, slow audio tests nightly, docs preview per PR, release on tag via PyPI trusted publishing | |
| Pre-commit | ruff, mypy (changed files), toml-sort, check-jsonschema for workflows, end-of-file | Runs in pre-commit.ci so contributors need no local setup. |
| Changelog and versioning | Keep-a-Changelog, SemVer after 1.0, deprecation window of two minor releases with `DeprecationWarning` | towncrier adds process for little gain at this team size. |
| Data fetching | **pooch** | Hash-verified, cached, standard in scientific Python. |
| Performance | numpy vectorisation first; `concurrent.futures` for file-level parallelism; numba only if profiling demands it, behind an optional extra | |
| Citation and archiving | `CITATION.cff`, Zenodo DOI per release, JOSS paper after 1.0 | |

---

## 6. API design

Two tiers. The **narrative API** is about fifteen verbs at the top level, enough for a
tutorial and a GUI. The **module API** is complete and documented.

```python
import soundscapy as ssp

ds = ssp.datasets.isd.load()                              # Dataset with codebook
ds = ssp.validate(ds, rules=ssp.surveys.DEFAULT_RULES)    # adds 'valid' and 'invalid_reason'
ds = ssp.project(ds, circumplex=ssp.Circumplex.from_language("en"))   # adds 'pleasant_iso', 'eventful_iso'

camden = ds.responses.query("location == 'CamdenTown'")
dist = ssp.KDE.from_frame(camden, bandwidth="scott")
dist.fraction_in(ssp.Quadrant.VIBRANT)                    # 0.41
ssp.describe(ds, by="location")                           # mean, sd, quantiles, quadrant fractions

ax = ssp.viz.circumplex(ax=None)                          # styled axes
ssp.viz.density(dist, levels=[0.5], ax=ax, label="Camden Town")
ssp.viz.scatter(camden, ax=ax)

target = ssp.spi.targets.load("park_v1")                  # or ssp.SkewNormal(xi=..., omega=..., alpha=...)
score = ssp.spi.score(dist, target, n_samples=1000, rng=0)   # SPIResult(value=62, d_ks=0.38, ...)

features = ssp.audio.analyse(ds.audio, config=ssp.audio.Config.iso12913(), executor="processes", cache=".ssp-cache")
model = ssp.models.LinearPredictor.from_published("mitchell2021_eventful")
pred = model.predict(features)                            # PredictionResult with .distribution
ssp.spi.score(pred.distribution, target, rng=0)
```

Conventions:

- Functions take and return DataFrames or spec/result objects; they never mutate
  inputs. Column additions return a new frame. `pipe`-friendly.
- A pandas accessor (`df.ssp.project(...)`) is tempting but magical and hard to type;
  not in v1. Revisit if users ask.
- Every result object has `to_dict()`, `to_frame()` where sensible, and
  `from_dict()`; results carry the spec that produced them and the package version.
- Randomness always takes `rng: np.random.Generator | int`.
- Plot functions take `ax` and return `ax`; they never create figures unless
  `ax=None`, and never call `show()`.
- Exceptions form a small hierarchy rooted at `SoundscapyError` (`SchemaError`,
  `NotAdmissibleError` for CP→DP, `BackendMissingError`).
- Deprecations: `warnings.warn(..., DeprecationWarning)` plus a changelog entry; no
  silent behaviour changes.

---

## 7. Visualisation without writing a plotting package

The current `ISOPlot` class (about 1,800 lines, a layer hierarchy with multiple
inheritance, subplot management, legends, styling) is the clearest example of what
to avoid: it re-implements figure management that matplotlib already provides, and it
mixes computation (KDE, SPI scoring) with drawing.

Proposed shape:

1. **Geometry in the core.** `KDE.density_grid()`, `KDE.mass_contours([0.5])`,
   `Empirical.quadrant_fractions()`, `shift_vector(a, b)`. These are tested
   numerically. The contour at "50% of responses" is computed explicitly from the
   density grid (cumulative mass), so its meaning is unambiguous and the percentile
   is a parameter, as the 2022 paper asks.
2. **Small drawing functions on an Axes.** `viz.circumplex(ax)` styles the frame
   (square [-1, 1]² axes, zero lines, optional diagonals, quadrant labels, axis
   titles from the circumplex spec); `viz.scatter`, `viz.density`, `viz.contour`,
   `viz.marginals`, `viz.shift_arrows`, `viz.likert_stacked`, `viz.radar`. Each is
   tens of lines over seaborn/matplotlib. Faceting uses seaborn `FacetGrid` or plain
   `plt.subplots`; the package does not own subplot layout.
3. **A declarative `PlotSpec`** (pydantic): data reference, layers, hue, levels,
   theme. `viz.render(spec, backend="matplotlib")` today; a plotly or vega renderer
   later for the GUI, consuming the same geometry. The spec is also the thing a GUI
   saves in a project file.
4. **Themes** are matplotlib style sheets shipped as data, plus a colour-blind-safe
   default palette.

Likert plots: build on seaborn or a small stacked-bar helper rather than depending on
`plot-likert`, which is lightly maintained.

---

## 8. Robustness: validation, tests, regression baselines

- **Boundary validation** with pandera schemas generated from the codebook; strict in
  tests and loaders, switchable off for large pipelines.
- **Property-based tests** (hypothesis): projection output always within [-1, 1];
  equal-angle projection equals the ISO formula; DP→CP→DP round trips; `ks2d(a, a) == 0`
  and symmetry; `fraction_in` over complementary regions sums to 1.
- **Literature fixtures**: the ISO worked examples, the lockdown model coefficients
  (predicting the published 2020 location means within rounding), SPI Table III
  scores for the ISD locations within a seeded tolerance, the park reference target
  from Eq. 7 reproducing its own ranking.
- **Cross-implementation reference tests** (run manually or nightly): skew-normal
  fit and sampling against R `sn`; loudness against the ISO 532-1 reference signals
  shipped with the standard; selected metrics against ArtemiS values where the
  authors can share them. Results stored as parquet under `tests/regression/` with
  tolerances and a script to regenerate them.
- **Golden pipelines**: end-to-end ISD subset → validate → project → describe → SPI,
  with outputs compared to stored baselines. Any numeric drift fails CI and must be
  explained in the changelog.
- **Plot tests** assert on the artists' data (contour vertices, scatter offsets),
  not on pixels, except for two or three layout smoke tests via pytest-mpl.
- **Benchmarks** for the audio batch runner and KDE on 10k points, tracked over time.

---

## 9. Concurrency

- File-level parallelism in `audio.analyse` through `concurrent.futures`
  (`ProcessPoolExecutor` default, threads or serial selectable, a `joblib`/`dask`
  adapter possible later). Work units are `(path, MetricSpec list, calibration)`,
  all picklable. The runner never parallelises inside a metric and across files at
  once, which currently causes oversubscription with MoSQITo's own parallel loops.
- Per-file cache in parquet keyed by content hash and config hash: resumable, and a
  second analyst with the same config gets instant results.
- Progress through an `on_progress(done, total, path)` callback; the CLI wires tqdm,
  a GUI wires its own bar.
- SPI reference-target search parallelises candidate evaluation through pymoo's
  starmap runner.
- KDE and projection are vectorised numpy and do not need parallelism.

---

## 10. Reproducibility and provenance

Every pipeline function accepts a spec and every result records it. A `Manifest`
written next to batch outputs captures package version, Python and dependency
versions (from the lock file), config hash, input file hashes, seeds and timestamps.
Project directories (`io.Project`) hold `codebook.yaml`, `config.yaml`, parquet
tables and manifests, so a project can be zipped, shared with a reviewer and
re-run. This is the artefact a GUI opens and saves.

---

## 11. GUI readiness

The GUI is out of scope for the core, but the core must not block it. The checklist:

- No global state, no I/O side effects in computation, no `show()`/`print`.
- Every input and output JSON-serialisable (specs, results, geometry, manifests).
- Long jobs are functions with progress callbacks and a cancellation token.
- A project directory format the GUI can open and save.
- A `PlotSpec` so the GUI can render with its own charting library.
- The **CLI is the first GUI**: `ssp validate`, `ssp project`, `ssp analyse-audio`,
  `ssp score`, `ssp report`. If a workflow cannot be expressed as a CLI command over a
  project directory, it is not ready for a GUI either.
- Later, a thin service layer (FastAPI or a Panel app) wraps the same functions.
  Whether the GUI is desktop (Tauri/web view) or web is a separate decision.

---

## 12. Scope boundaries and things deliberately left out

- **Field data capture** (REDCap forms, tablet apps) is out of scope. The package
  starts at "an exported table plus audio files". An ISD/SSID codebook and an
  importer for REDCap exports are in scope.
- **Soundscape captioning** (SoundSCaper) is a consumer of perception predictions,
  not a core feature. A plugin could wrap it; the LLM dependency never enters core.
- **Spatial audio / ambisonics processing** is out of scope; binaural and mono are in.
- **Ecoacoustic indices** (scikit-maad) are supported through the metric registry
  but are not a focus.
- **Sound-source recognition** models are plugins behind the predictor protocol.

---

## 13. Places where the plan challenges the literature's current practice

These are not disagreements with the science; they are places where faithfully
implementing it requires a decision the papers leave open.

1. **SPI stochasticity.** The published method samples the target (n = 1000) and
   reports integers. Implementation must expose the seed and sample size and should
   report variability; a deterministic alternative should be benchmarked.
2. **The 50% contour is provisional** by the authors' own statement. It is a
   parameter with a documented default, not a constant.
3. **KDE bandwidth** is undocumented in the papers and is dataset-dependent. The
   default is Scott's rule on the projected coordinates, stated explicitly in every
   result, and a bandwidth sensitivity helper is provided.
4. **Location as a random effect** does not generalise to new sites (the framework
   paper says so). The predictor contract makes context requirements explicit and
   distinguishes "known location" from "fixed-effects-only" prediction.
5. **Max-of-channels for psychoacoustic metrics** (ISO/TS 12913-3) is one channel
   rule among several; it is a per-metric setting with that default.
6. **Adjusted angles and centring**: see §3.2. This needs a short numerical check
   against Aletta et al. 2024 before the kernel is frozen.
7. **Deep models trained on ARAUS** predict per-clip Likert means from a lab setting.
   Their outputs enter the package as `PredictionResult`s with the training setting in
   provenance so that SPI scores built on them are labelled as such.

---

## 14. Delivery plan

Each phase ends with a tagged release, executed docs, and regression baselines.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0. Foundations | Repo skeleton, CI, tooling, ADR-001..006 (data model, circumplex, distributions, audio registry, predictor protocol, no-R), codebook and Circumplex specs | `uv run pytest` green on three OS; import-linter contracts in place |
| 1. Vertical slice | `datasets.isd`, `surveys` validation and harmonisation, `circumplex.project`, `distributions` (Empirical, KDE, SkewNormal, ks2d), `spi.score`, `viz` basics, CLI `validate/project/score` | Reproduce the 2022 JASA-EL figures and SPI Table III (seeded tolerance) from the ISD; no R anywhere |
| 2. Audio | `Signal`, calibration, metric registry with MoSQITo and built-in level metrics, batch runner with cache, ISO 12913-3 config preset | Reference tests against ISO 532-1 signals; 1,000-file batch resumable; lockdown feature set reproducible |
| 3. Models | Predictor protocol, `LinearPredictor.from_published`, `MixedModel` workflow helpers, evaluation (MAE, location-mean R²), feature screening | Reproduce the 2021 lockdown predictions within rounding |
| 4. Design and scoring | Scenario comparison, shift vectors, "percent likely", reference-target search (`[optim]`), report generation | A worked design case study in docs |
| 5. Ecosystem | SATP, ARAUS, DeLTA loaders with codebooks; plugin entry points; `soundscapy-models-torch` prototype | Harmonised cross-dataset analysis tutorial |
| 6. GUI prototype | Panel or FastAPI front-end over the project format | A non-programmer completes validate → project → plot → score |

---

## 15. Decisions needed from you

1. Name and lineage: a `soundscapy` 2.0 with a migration guide, or a new name with
   `soundscapy` frozen? This affects whether any compatibility shims are owed.
2. DataFrame contract: pandas-only for v1 (recommended), or dataframe-agnostic from
   day one?
3. Single distribution with extras (recommended) versus a uv workspace of packages.
4. Psychoacoustic backends: commit to MoSQITo for ISO 532-1 and ECMA-418-2 (confirm
   its ECMA coverage for roughness, tonality and fluctuation strength), with built-in
   level metrics, and drop python-acoustics? Which metrics are must-have for 1.0?
5. Skew-normal: pure Python with a one-off R validation (recommended), or keep an
   optional R bridge indefinitely?
6. Mixed models: statsmodels only (recommended for 1.0), bambi as an extra, or
   Bayesian as the primary route?
7. Deep models as separate plugin packages (recommended) or an in-repo `[torch]` extra?
8. Visualisation: matplotlib/seaborn only for 1.0 with `PlotSpec` reserved for the
   GUI (recommended), or ship a plotly renderer early?
9. A canonical project-directory format (recommended) and whether the ISD should be
   published in that format as the reference example.
10. Scope of dataset support at 1.0: ISD and SATP only, or ARAUS and DeLTA as well?
11. Governance: who are the maintainers per module, and what review rules apply?

Reply with objections, counter-proposals or constraints and the plan will be revised
before any code is written.
