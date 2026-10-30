# Review of PRD draft 0.6 and architecture plan draft 0.3

Date: 2026-10-30. Inputs: `design/PRD.md` (0.6), `design/greenfield_architecture_plan.md`
(0.3), AGENTS.md, the house-style skill. Outcome: findings for the owner; no document
was edited except the stale version pointer in AGENTS.md (it said the plan was draft 0.2).

## Method

Four reads, compared afterwards so that none anchored the others. None read the 0.3
review.

- **Lead read.** Both documents, written down before the other three reported.
- **Critical review of the PRD.** A fresh reviewer given only the PRD and AGENTS.md:
  contradictions, testability, scientific fidelity, scope versus capacity, gaps, and
  agreed decisions that look wrong.
- **Builder check.** A fresh reviewer given the PRD, the plan and the house style, asked
  what they would still have to decide before planning Phase 0, M1 and M2.
- **Plan review.** A fresh reviewer asked to test the plan's claim that it is in step
  with the PRD, walking every FR and NFR, and to probe the plan's own design.

Findings raised by two, three or four reads are marked **(×2)**, **(×3)**, **(×4)**.
Claims a reviewer made that did not check out were dropped (one reviewer said AGENTS.md
still called the plan out of date; it does not, it only carries the wrong draft number).

## Verdict

The two documents are consistent with each other on the things the plan's §16 checked
(Python floor, licence, tutorial format, file locations, milestones, tools). They are
not yet ready for a Phase 0 / M1 spec, for three reasons.

1. **The PRD's "nothing is open" overstates the position.** The §10.3 inputs block the
   M1 tutorial (ISD language coverage), the M1 loader (record versions) and all of M2's
   numeric acceptance (Example 6, the 2024 targets, the sample size, the 2-D KS variant).
   Several Must acceptance criteria are not yet acceptance criteria (findings 1, 2, 3).
2. **A handful of plan decisions change public types and must be settled before code
   exists** (findings 5 to 10): the name `Mapping`, circumplex identity, the shape of
   `Coordinates`, the override semantics for uncovered languages, N < 30 as a hard
   error, and the settings model.
3. **M1 is the largest milestone in the plan and carries work no M1 journey needs**
   (finding 11).

Everything below is fixable in the documents; nothing needs the architecture restarted.
The central decisions (one `Coordinates` type for journeys A, C, D and E; positions not
words; settings as data with results carrying them; geometry separate from drawing; the
mapping as the schema; the LISQM evaluation) hold up under all four reads and should be
kept.

## Findings, ranked

### Blocking: settle before the Phase 0 / M1 spec

**1. The "2024 revision" profile is not the 2024 revision (×2).** PRD §2 says the
revised Part 3 has only a placeholder on translated attributes, yet the profile named
`iso12913-3-2024` bundles the canonical projection with SATP v1.5 angles from a 2026
book chapter (FR-2.9, plan §3.1). A consultant stating "this number complies with
ISO/TS 12913-3:2024" would be wrong, which defeats FR-2.9's purpose and NFR-6. The §2
sentence "where a standard and a Mitchell source differ, the Mitchell source is the
intended reading" compounds it for a self-described reference implementation.
*Change:* three honestly named profiles: `iso12913-3-2019`, `iso12913-3-2024` (the
revision's formulas, equal angles) and `satp-v1.5` or similar (canonical projection
plus adjusted angles, citing the chapter) as the default. Profiles named after a
standard are faithful to its text; delete the "Mitchell overrides the standard" rule.

**2. "Example 6 reproduces SPI 56" cannot be the acceptance test for a Monte Carlo
number, and the KS statistic is unnamed (×3).** The 2-D KS distance between finite
samples has a sample-size floor (identical distributions give D of order 1/√n, so SPI
100 is unreachable), varies with the seed, and differs between Peacock 1983 and
Fasano–Franceschini 1987, between one- and two-direction averaging, and with the tie
rule at quadrant boundaries, which matters on a five-point scale. Part 4 Example 6 is a
narrative example that almost certainly carries no data, seed or algorithm; whether the
paper's targets were sampled truncated to the square (plan §3.3 makes truncation
optional) changes every SPI. *Change:* make the numeric acceptance the 2024 paper's
tables at the paper's seed, sample size and a stated tolerance; make Example 6 a
tutorial that reproduces the story, with "SPI ≈ 56" only if the owner supplies the
data. Name the statistic (the builder believes the paper's code used the
Fasano–Franceschini variant with D = (D1 + D2)/2; the owner must confirm), the tie
rule and the truncation default, and record all of them in `SPIResult.settings`.
Sequence M2 so the acceptance test is the last task.

**3. FR-2.5a renormalisation breaks guarantees it claims to keep (×2).** Recomputing
the normalisers over present items keeps coordinates in [−1, 1] and sends a neutral
response to the origin, but: with only items of zero cosine (or sine) present a
normaliser is zero; an unbalanced subset no longer has Σcos θ = 0, so a uniform
non-neutral response is displaced; a response projected on two items gets full-weight
coordinates and a different variance from an eight-item one, so group means and KDEs
mix two instruments. No standard or chapter defines the rule, so results carrying it
cite nothing, including under the default profile. *Change:* demote to Should or drop.
If kept: forbid it inside standards-named profiles, require a minimum number of present
items with a non-zero normaliser on each axis, add a per-response `n_items_present`
column, and exclude renormalised responses from Part 3 statistics by default.

**4. The override for uncovered languages projects with equal angles (×3).** FR-2.4
says the error "can be overridden" without saying to what; plan §3.1 picks
`on_uncovered="equal"`, which projects a Polish response with English geometry and
pools it with adjusted-angle responses in one mean and one KDE. The chapter's advice
for uncovered and Low-tier languages is not to project. *Change:*
`on_uncovered: "error" | "exclude" | "equal"` with `"exclude"` writing NaN and the
reason `uncovered_language` to the shared `exclusion_reason` column (the FR-2.5a
mechanism); `"equal"` allowed but emitting a `MixedAngleSetsWarning` once per
projection, with `describe` reporting the mix. Same three-way choice for Low-tier
languages. Write it into FR-2.4 and an ADR.

**5. `Mapping` shadows `typing.Mapping` and does five jobs (×2).** Plan §3.2 declares
`class Mapping(BaseModel)` and in the same block types fields as `Mapping[str, str]`.
It will not survive ruff and ty, and it is in every signature and file. It also
carries schema, coding, id roles, extras and provenance, while `coding` applies to
attributes only: source dominance, overall quality, appropriateness and WHO-5 arrive
0-based from REDCap too and have no coding or scale (FR-1.3, FR-2.6c, §16.3's
`excessive_missing` counts source-dominance items). *Change:* rename to
`SurveyMapping`; add `scales` per construct group with ISO defaults (PAQ 1..5, source
dominance 1..5, overall quality, appropriateness, WHO-5 0..5); define the order of
`Coding` operations (`values`, then `offset`, then `reverse` as `min + max − x` against
the resolved scale) and make score type float; `Coding.values` keys are strings only,
with a documented stringification rule, because TOML and JSON keys are strings and
Excel delivers `1.0` while REDCap delivers `"1"`.

**6. Circumplex identity is by name string (×2).** `Target.circumplex: str` and
`AngleSet.circumplex: str` against `Coordinates.circumplex: Circumplex`, so
`CircumplexMismatchError` can only compare names: two `iso12913-outdoor` instruments
with a 1..7 scale override or edited angles pass. FR-6.3's "versioned library" has no
version field (`Target` carries provenance only), and nothing says old angle-set or
target versions stay shipped so old results remain reproducible (×3 across the
versioning point). *Change:* give `Circumplex` a content fingerprint (positions,
angles, axes, scale); every by-name reference carries name plus fingerprint; add
`version` to `Target` and `AngleSet`; require results to record the data-file versions;
state that superseded versions are never removed.

**7. `Coordinates` has three unresolved shapes inside it (×3).** `xy` is "(n, 2) in
[−1, 1]²" but an excluded response's coordinate is NaN; per-response angle set and
version are promised but `settings: ProjectionRecord | PredictorRef | TargetRef` is one
record per object and `labels` is optional; `group_by(key) -> dict` loses the key name
and cannot do crossed or time groupings (FR-2.5), and `describe(dict)` follows it; a
frozen dataclass with an `ndarray` field gets an unusable generated `__eq__`.
*Change:* rows aligned with the responses frame, NaN for absent, plus `n_valid`;
`labels` mandatory and always carrying `response_id`, `language`, `angle_set`,
`included`, `exclusion_reason`; one `origin` provenance record with a `kind` instead of
the three-way union; `describe(coords, by="location" | [...])` returning a tidy frame
with `group_by` as a convenience; `eq=False` with an explicit tolerance comparison; a
stated stance on response weights (the PRD never asks for them).

**8. N < 30 as a hard error at `KernelDensity` (×3).** FR-3.2 says "discouraged …
scatter is offered instead"; the plan raises `SmallSampleError`. `plot.by_group` over
twenty sites fails on the first site with 28 responses instead of drawing scatter.
*Change:* keep the check, move the decision to the drawing layer: `plot.density`
raises, `plot.by_group` and the CLI draw scatter for small groups and say so, and the
geometry object warns with `SmallSampleWarning` and records the N. Say what N counts
(projected responses after exclusions) and note the rule is vacuous for sampled
predicted distributions.

**9. The settings model is unspecified and `Profile` carries M3 fields at M1 (×3).**
`settings.py` names a "Settings root model" with no fields; `Profile.metric_set: str` is
required when no metric set exists until M3; bandwidth, contour fraction, grid
resolution, the N < 30 threshold, quality rules, SPI sample size and the override flags
are each "recorded" but belong to no model; recording a profile "by name and by
resolved values" becomes a lie once a user edits one field; `project.toml` and
`settings.toml` both hold settings. *Change:* `Settings = {profile_name, projection,
density, quality, spi}` with `audio` added at M3; `Profile` is a named preset file;
`profile_name` becomes "custom (from …)" on divergence, computed not stored;
`settings.toml` is the only settings source and `project.toml` holds name, package
version and format version; the JSON sidecar is authoritative for a result.

**10. The narwhals boundary is under-specified in every public signature (×2).**
`IntoFrame` includes lazy frames; "returned in kind" is undefined when a path went in
(`read_survey("export.xlsx")`, `Project.open`); `apply_quality_rules` and `project`
both write `exclusion_reason`, so either `project` returns a new responses frame or the
sentence is wrong; the accumulated-reason dtype and separator are unfixed on a persisted,
filtered column; pandas reads blanks in a Likert column as float NaN while polars reads
nullable Int64, and ty's handling of narwhals' overloads is unverified. *Change:*
accept `IntoDataFrame`, convert through one internal helper that rejects lazy frames,
return native; `read_survey` and `Project` take `backend="pandas" | "polars"`;
`project` never touches the responses frame (`incomplete_attributes` lives on
`Coordinates.labels`); reasons are a `;`-joined string; polars in the `test` group;
list every pandas/pyarrow boundary (`survey.io`, `survey.stats`, Excel export,
`datasets`, `audio.batch`, `project`) in the import-linter contract.

### Important

**11. M1 is overloaded and builds what M1 does not use (×4).** M1 holds instrument,
survey with mapping suggestion against known export schemas, quality rules, projection,
distribution including `SkewNormal` and `ks2d`, four plot families, a nine-command CLI
with an interactive step, the project directory, the ISD loader with caching, two
recipes, plugin discovery with nothing to discover, templates in CSV and Excel, the
quick-start page, tutorial A and the 2022 reproduction. For one part-time maintainer
that is most of a year before a first release, and the plan's own "nothing for later
milestones is built early" is broken by `SkewNormal`, `ks2d` and `plugins`. *Change:*
move `SkewNormal`, `ks2d`, `targets fit` and plugin discovery to M2 (the PRD's §9
sentence putting discovery at M1 should change with them: at M1 the only discoverable
type is a dataset loader, which nobody will publish before 1.0); cut the M1 CLI to
`import`, `validate`, `project`, `describe`, `plot`, `export`; CSV templates only;
mapping suggestion against the SSID and REDCap schemas only, Qualtrics later.

**12. Audio backends: the M3 bet is not credible as written (×3).** sottek-hearing-model
is 0.1.x, AGPL on PyPI and GPL-3 on GitHub, and pulls PyQt6 (itself GPL or commercial,
about 60 MB) as a hard dependency into a headless extra; the mitigation is an upstream
issue "opened by whoever leads M3". phonometry is 3.13-only, one maintainer, released
in 2026, and the environment marker makes the metric list depend on the interpreter, so
one project yields different columns on 3.12 and 3.13 (FR-11.2) unless the Osses and
ECMA fluctuation-strength metrics carry different names, which the plan implies but
does not say. MoSQITo has had no release since April 2024. "Every Table D.1 metric has
a backend on 3.11" is overstated: the same section calls Aures tonality a gap, and
R10/R50 via Daniel–Weber and F10/F50 via ECMA-418-2 are substitutions that should be
named as such. A 3 OS × 4 Python × 4 backend matrix with PyQt6 on CI runners is not
maintainable by one volunteer. *Change:* M3 ships built-in levels and IACC plus
MoSQITo as the single `audio` backend with the gaps documented; sottek-hearing-model in
a separate `audio-ecma` extra once PyQt6 is optional and its licence is clarified;
phonometry as an explicit `phonometry` extra, never an environment marker; the default
backend for a metric never depends on the Python version; full audio matrix on Linux
only with one smoke job per other OS; Apache fork of MoSQITo stated as the fallback.
Write the ADR at M3 planning.

**13. Table D.1 and the metric definitions contradict §2 (×2).** §2 says N_rmc, S_50
and IACC were added in 2024; FR-4.4 lists N_rmc in the base set and only S50 and IACC
as additions. "Tonality" names no method (Aures, DIN 45681 and ECMA-418-2 differ
fundamentally), so the plan's ECMA mapping may be a substitution rather than coverage.
Higher-of-channels (FR-4.6) is under-specified: per metric or per statistic, per frame
or on the final value, and it is meaningless for IACC and ill-defined for signed
statistics. *Change:* one table in §2 with a column per edition and a named tonality
method per edition; state the channel rule as Part 3 states it, record the chosen
channel per cell, exclude cross-channel metrics.

**14. Pooled single-response predictions are not a response distribution (lead).**
FR-5.9 lets a set of per-clip point predictions stand in for survey responses "subject
to the same sample-size rules". Point predictions are conditional means and have less
spread than real responses, so a KS distance against a target fitted to response-level
spread is biased, which is the argument of the 2023 framework paper the PRD cites.
*Change:* keep FR-5.9, but require predicted-from-single-response `Coordinates` to
carry a warning class and a label in every table and plot, and require the explanation
docs to state the bias. ADR at M4.

**15. FR-6.1's "no uncertainty at all" is a mistake for journey C (×2).** A consultant
defending "SPI 71 against KPI 70" needs to know the number moves across seeds and with
N; "run several seeds yourself" is what a GUI user cannot do, and FR-7.3 flags KPI
pass or fail on one draw. The published method reporting a bare number does not oblige
the tool to hide what it knows. *Change (owner decision, respectfully re-raised):*
keep the single-seed number as the published method; add an optional `n_seeds`
parameter returning min, mean and max (Should, M2) and record the test-sample N in the
result. SPI is near-instant, so the cost is a loop.

**16. The library has no channel for licence display or download announcements
(×1, verified).** FR-10.1 and NFR-5 require the licence and "what is fetched" to be
shown; principle 9 forbids printing and logs to a `NullHandler`, so nothing reaches
anyone by default. pooch cannot take a `CancelToken` either, so FR-14.3 for downloads
needs a custom downloader or an honest statement. *Change:* `describe()` returns the
licence text as data; the `ProgressReporter` protocol gains a `message()` method that
the default reporter prints and a GUI binds; downloads either implement cancellation
through a custom pooch downloader or are documented as not cancellable.

**17. Warnings and `filterwarnings = error` will fight each other (×3).**
`LegacyFormulationWarning` fires on every call for a profile the PRD calls defensible
and a consultant chooses deliberately; `MediumConfidenceWarning` would fire per
response; `tqdm.auto` warns in a notebook without ipywidgets; statsmodels, matplotlib
and a dormant MoSQITo emit `DeprecationWarning`s on each upstream release, which with
`error` breaks CI for a volunteer, and the house style forbids weakening the rule.
*Change:* drop `LegacyFormulationWarning` (the formulation is recorded); one
`MediumConfidenceWarning` per projection with the language list; the default reporter
never warns; pre-authorise narrowly scoped per-module `ignore::DeprecationWarning`
entries in the house style so the first contributor does not fight the Stop hook; add a
`CancelledError` and a partial-result type.

**18. Import gating contradicts the layering table (×3).** §4 gates optional imports
"at module import" while `project` and `cli` "may import everything" and `plot`
imports `audio` for `plot.timeseries`, so `Project.open()` on a course laptop without
the audio extra raises `MissingExtraError`. The lazy top-level `__getattr__` is
invisible to ty and to great-docs' static discovery. *Change:* raise `MissingExtraError`
at call time; `audio` and `predict` and their plot functions as explicitly imported
subpackages; eager top-level imports for the M1 and M2 modules with `__all__`, or
`lazy_loader` with a `.pyi` stub; delete the clause that keeps matplotlib core "to
satisfy MoSQITo's undeclared import".

**19. The project directory contradicts itself on "untouched" (×3).** §11 says
`data/responses.csv` is the user's table untouched; §16.1's `import` converts Excel to
CSV, losing dtypes, leading zeros, dates and sheet identity; the sidecar is written
"only when needed", which needs detection logic and makes the format non-uniform; one
`results/projection/` slot cannot hold the two runs a consultant makes to compare the
2019 and 2024 profiles. *Change:* copy the original export verbatim to `data/raw/` with
its hash in the manifest; write the coded table and always write a schema sidecar;
`import` requires `--sheet` for multi-sheet workbooks; results keyed by run name; CLI
commands write CSV to `exports/` by default so a shared folder opens in Excel without
a further step (FR-14.2 was rewritten to admit parquet; this keeps the non-programmer
path honest).

**20. `Signal` holds the whole file (×2).** One hour of 48 kHz stereo float64 is
2.8 GB; FR-4.1a brings hours, and `MetricSpec.windows` slices after loading, too
late. The cache key needs the full file hash, which contradicts FR-1.6's "index without
loading" if computed at index time. *Change now, so M3 adds rather than reshapes:*
the loader yields windows (`read(path, window=...)` iterating `Signal`s); windowing is a
batch-runner concern and backends see one window; hashes are computed once in
`analyse`, never in `index`; more than two channels is refused explicitly (FR-4.1);
per-file calibration reaches `AudioResults.settings` (FR-4.2).

**21. Design and prediction interfaces have gaps (×1 each, verified).** `evaluate`
takes `Mapping[str, Coordinates]`, so the Annex A level and `produced_by` on
`Scenario` never reach the `Evaluation` tables FR-7.5 needs. `predictors.available()`
"lists requirements" by reading `inputs` on the class, which imports torch for a deep
model; contracts belong in metadata the entry point exposes without loading the model.
Nothing enforces FR-5.5's "never require perceptual inputs". `PredictorRef.n_samples`
is meaningless for single-response predictors. `MetricBackend` is one class per
(metric, backend) pair in all but name, and whether entry points yield classes or
instances is unsaid. `SkewNormal.fit` returns a bare model with no convergence,
log-likelihood or method (§3.8 says every result carries settings and versions); CP→DP
exists only below marginal skewness ≈ 0.9953 and MLE in direct parameters hits
boundary solutions, which is why `sn` fits in centred parameters. *Change:* `evaluate`
takes `Scenario`s; `PredictorInfo` and contracts on the entry-point object; a contract
validator that rejects perceptual inputs; `n_samples: int | None` with `kind`; rename
to `MetricImplementation`; `FitResult`; fit in centred parameters as `selm` does.

**22. Overlap and shift against a target are the plan's inventions (×4).** FR-3.3's
"overlap" is undefined in the PRD and the 2022 paper defines no overlap statistic;
Jaccard of two mass contours at one fraction is unpublished and depends on the fraction
and the grid. `shift(site, target)` depends on a seed although the skew-normal mean is
closed-form. KDE mass depends on grid extent and clipping, which the plan does not fix,
so "the same estimator seaborn wraps" only matches seaborn's `levels=[0.5]` if grid
and clipping match; `bandwidth` as scipy's covariance factor is not a bandwidth in axis
units. *Change:* `overlap()` as the overlapping coefficient ∫min(f, g) on a 200 × 200
grid over exactly [−1, 1]² with mass normalised over the square, `contour_overlap(fraction)`
as the Jaccard variant, the target's pdf evaluated directly; `shift` uses the
closed-form mean; record grid, extent, factor and the absolute bandwidth matrix. ADR.

**23. The 2021 "guaranteed fallback" predictor is not guaranteed (×1).** Journey D
calls the linear models the guaranteed default; §10.3 admits their features may be
proprietary (relative approach and HMS tonality are out of scope). Known-location mode
uses intercepts estimated from survey data, which sits uneasily with FR-5.5.
*Change:* resolve §10.3's last row before claiming a guarantee; state that
known-location mode is for sites with prior survey data and is not deployable in the
FR-5.5 sense.

**24. 1.0 at M2 freezes an API with no audio or predictor surface (×1).** SemVer 1.0
promises stability; the interfaces most 0.x users depend on are designed after it,
so either 1.x breaks them or they are designed under pressure not to. *Change:* keep
1.0 at M2 for the course; have the versioning policy (FR-13.4) define a provisional
tier for namespaces introduced after 1.0 with a shorter deprecation window.

**25. Contribution and discovery at M1 precede anything to contribute to (×3).** See
finding 11. The PRD's §9 sentence should move recipes and discovery to the milestone
of their first user (dataset loader at M2 with 1.0, metric at M3, predictor at M4).

### Minor

26. **IEC 61672 does not define calibrating a recording (×2).** It specifies sound
    level meters; the calibrator is IEC 60942; neither describes deriving a gain from a
    recorded segment. Describe the procedure (RMS of the stated segment to the stated
    level per channel, calibrator frequency and level recorded) and cite IEC 60942.
27. **"Exactly" reproduces the 2019 formula only for 1..5 coding and equal angles**
    (checked: (s − 3)/2 ÷ (2 + 2√2) = (s − 3)/(4 + √32), and the midpoint terms cancel
    because Σcos θ = 0). Say so, give a tolerance, and keep the hypothesis property
    tests the plan lists.
28. **WHO-5 scoring unspecified (FR-2.6c).** Five 0..5 items, raw sum × 4 → 0..100,
    cut-offs 50 and 28; no prorating for missing items; state the ISD's coding.
29. **Skew-normal fitting as a Must (FR-3.4).** Journeys C and E specify targets; only F
    (Should) fits one. Demote fitting to Should and add the R `sn` fixture run to §10.3.
30. **FR-12.2 contradicts the milestones.** It requires a journey-B tutorial on the
    bundled dataset for K at M2, but B is M3 and redistributable audio is open. Cover
    A, C, E at M2 and B at M3 with shipped or synthetic audio.
31. **One status column for every exclusion.** The plan has it (`exclusion_reason`);
    the PRD does not require it. Add an FR for one per-response inclusion status and
    reason, and one non-interactive override mechanism shared by all refusals.
32. **`soundscapy map` interactive confirm vs FR-14.1 (×3).** Non-interactive by
    default, writing the suggestion with confidences; `--interactive` prompts.
33. **Untestable Musts.** FR-1.3a "easy"; FR-1.5 thresholds unnamed; FR-3.1 no default
    bandwidth rule in the PRD; FR-8.3 "publication-quality"; FR-12.1 and journey L
    usability claims with no protocol; journey K "every exercise runs unchanged" is
    circular until exercises exist. Name a measure or demote to Should.
34. **Paper reproductions need the full ISD from Zenodo**, so `task docs` is not
    offline as the house style says. State that `docs/papers/` pages are frozen and
    re-executed nightly or on demand, and add a golden table of the 2022 quadrant
    percentages under `tests/regression/` so a change is at least noticed.
35. **Toolchain details the spec should pre-empt (×2).** ruff's `TC001–TC003` move
    imports into `TYPE_CHECKING` blocks, which breaks pydantic; set
    `runtime-evaluated-base-classes = ["pydantic.BaseModel"]` on day one. Expect global
    ignores for `EM101/EM102/TRY003`, `FBT001/FBT002`, the `PD` family (fires on
    narwhals code) and `PLR0912/PLR0915` on CLI commands, each with a reason. Never use
    `slots=True` with `cached_property`. Full 12-job matrix on merges and nightly only;
    PRs run Linux (floor and ceiling) and Windows (one version). The 80 % coverage gate
    fails on the empty Phase 0 package. Document a tested Python ceiling: a new CPython
    before statsmodels or pyarrow wheels means a course laptop needs a compiler.
36. **statsmodels in core.** Spearman, Pearson, Mann–Whitney, Kruskal–Wallis and
    t-tests are scipy; statsmodels is needed only for ANOVA and regression and is half
    the install with pyarrow. Consider an `analysis` extra with scikit-learn.
37. **Gaps the PRD does not mention (×2):** labels in plots for mixed-language
    datasets (NFR-10); timestamps and time zones for session logs, windowed SLM files
    and E1 pre/post ordering; a supported audio format list and a resampling policy
    under "no silent coercion"; missingness codes for non-circumplex items; structured,
    machine-readable errors and warnings for a GUI engine; hash pinning and a safe
    weights format (pickled torch weights execute code) for FR-5.4; the bundled ISD
    subset's licence (CC BY 4.0 with attribution) and whether other loaders download or
    need user-held copies; personal data in exports and in manifest paths when a project
    is shared; a CLI for journey D although P3 is a D actor; `CITATION.cff` for NFR-11.
38. **Angle sets keyed by language only.** A non-ISO circumplex with a declared
    language has no rule; key sets by (circumplex, language, version) and let non-ISO
    circumplexes default to their own angles. Language values arrive as "en" or
    "English", not ISO 639-3; normalise at the boundary.
39. **Copyleft: keep the core import graph permissive (×2).** The owner accepted
    copyleft dependencies; both documents should say explicitly that they live in
    extras and plugins only, so the BSD core stays importable by a commercial GUI, and
    that installing an AGPL extra is the user's combination, not the package's.
40. **"No backwards compatibility in any form"** should exempt data conventions: the
    ISD column names and the `ISOPleasant`/`ISOEventful` coordinate names appear in
    published supplementary data that journey I reproduces.
41. **AGENTS.md** said the plan was draft 0.2; fixed in this commit.

## What both documents do well: keep

- `Coordinates` as the one path for journeys A, C, D and E, with `source` and
  provenance keeping predictions distinguishable (FR-5.6).
- Positions not words; labels per language; the circumplex owning its axis names.
- Settings as frozen data, results carrying resolved settings and versions; TOML for
  what people edit, JSON for what the machine writes.
- The mapping as the schema, with no pandera; quality rules that never drop a row.
- Per-response angle resolution with the set and version recorded.
- The skew-normal research and the adoption of the `157` branch's `dp2cp` with its R
  baseline as the first reference fixture.
- Geometry separate from drawing; no `PlotSpec`; matplotlib conventions for styling.
- The project directory and a CLI command per journey as the GUI-readiness test.
- The four test kinds with markers; tolerances by significant figures.
- The honest LISQM evaluation in Appendix C.
- In the PRD: FR-2.0, FR-2.3, FR-2.4/2.4a, FR-3.x, FR-5.5/5.6, NFR-5, the course
  acceptance cases, and §10.3's list of owner inputs.

## Scope versus capacity

The four reads agree that M1 and M2 as the 1.0 candidate is right and that M1 as
written is too large. Demotions that do not harm M1's purpose (Part 3 analysis, no R):
`SkewNormal` and `ks2d` to M2; the CLI cut to six commands; Excel templates and
`export --format xlsx` later; mapping suggestion limited to the SSID and REDCap
schemas; recipes and plugin discovery with 1.0; FR-8.2 radar as Should. At M2: FR-3.4
fitting and FR-6.4 "one site against many targets" as Should. The 1.0 candidate is then
intake, projection, Part 3 statistics and plots, SPI with shipped targets, E1, the
course tutorials and settings provenance.

## Recommended order of work

1. Owner decisions on findings 1, 3, 4, 8, 15 and 29 (profile naming, FR-2.5a, the
   override, N < 30, SPI seeds, fitting priority), then PRD 0.7.
2. Owner inputs from §10.3 plus the 2-D KS variant, truncation policy and the 2022
   paper's KDE settings; these block M1's tutorial and all of M2's acceptance tests.
3. Plan 0.4 applying findings 5 to 10, 16 to 22 and the M1 re-scope, with ADRs:
   ADR-0001 functional core and records (add method versions and baseline library
   versions); ADR-0002 `SurveyMapping`, coding order, scales per construct, id roles;
   ADR-0003 `Coordinates` shape, NaN rows, labels, provenance, grouping, weights, and
   circumplex fingerprints; ADR-0004 settings model, profiles, project directory and
   sidecars; ADR-0005 distribution semantics (KDE grid and clipping, bandwidth
   recording, N < 30 placement, overlap, shift against targets); ADR-0006 override
   semantics for uncovered and Low-tier languages; ADR-0007 2-D KS variant, tie rule,
   truncation and SPI result fields. Audio (finding 12, 20) and prediction (14, 21,
   23) ADRs at M3 and M4 planning, but the `Signal`-as-window and call-time gating
   decisions are taken now.
4. Then the Phase 0 / M1 spec.

## Owner responses (2026-10-30), applied in PRD 0.7 and plan 0.4

| Finding | Response |
| --- | --- |
| 1. Profile naming | Agreed as proposed. |
| 2. Example 6 and the KS variant | Agreed. |
| 3. FR-2.5a renormalisation | Agreed: a constrained Should. |
| 4. Uncovered-language override | Agreed: three-way, "exclude" recommended. |
| 5. `Mapping` | Agreed in principle. |
| 6. Circumplex identity | Agreed. |
| 7. `Coordinates` shape | Agreed. |
| 8. N < 30 | Agreed. |
| 9. Settings model | Agreed. |
| 10. narwhals boundary | These come from not yet following narwhals' own API and conventions; review them carefully while building. |
| 11. M1 scope | Agreed. |
| 12. Audio backends | Drop full Table D.1 coverage as a hard requirement: missing open-source implementations must not block development. Drop the ECMA backend that brings PyQt6. Keep phonometry as an explicit extra. |
| 13. Table D.1 definitions | As 12. |
| 14. Pooled predictions | Agreed. |
| 15. SPI uncertainty | Agreed: an optional multi-seed summary. |
| 16. Output channel | Revisit "no printing or logging": follow scientific-Python practice for printing and logging, with notebook users in mind and a way to suppress output. Applied as PRD NFR-13. |
| 17. Warnings | Agreed. |
| 18. Import gating | Agreed; handle optional imports by current Python best practice. |
| 19. Project directory | Agreed. |
| 20. `Signal` | Agreed. |
| 21. Interfaces | Agreed. |
| 22. Overlap and shift | Keep only what the soundscape literature uses or other features need. No overlap statistic is defined in the cited literature, so none is planned; the shift vector is in the 2022 paper and Part 4 and stays. |
| 23. 2021 fallback predictor | Stop deciding the model now. Fix the predictor interface and plugin mechanism so whichever model the owner chooses at M4 can be implemented. |
| 24. 1.0 at M2 | Every release from M2 is a 1.0 alpha until the owner has evaluated the work; no API is frozen. |
| 25. Recipes and discovery | Agreed: with the first alpha. |
| 29. Skew-normal fitting | A Must, at the appropriate milestone (M2). |
| 26–28, 30–41 | Not yet answered; listed in PRD 0.7 §10.6. |
