# Review of architecture plan draft 0.4: over-engineering and premature commitment

Date: 2026-10-30. Inputs: `design/PRD.md` (0.7), `design/greenfield_architecture_plan.md`
(0.4), the house-style skill. Outcome: plan draft 0.5.

## Method

The owner asked whether the plan was over-engineered and whether it fixed decisions too
far ahead. A multi-agent workflow answered it in five steps, none of which read earlier
reviews.

1. **Six independent critics.** Five read the plan through one lens each: premature
   commitment, YAGNI, under-specification (the counterweight, so the cut does not go
   too far), ecosystem norms, and maintainer capacity. The sixth wrote a minimal
   architecture from the PRD alone, without reading the plan. Together they returned
   202 claims.
2. **Consolidation.** The claims were merged into 121 items in nine plan areas, with
   disagreements between critics kept.
3. **Adversarial verification.** Each area went to two verifiers, one attacking each
   verdict from the requirements side and one from the rework side.
4. **Synthesis.** A verdict, a decision register and a structure for the rewrite, below.
5. **Draft and check.** Plan 0.5 was drafted and then checked by a completeness critic
   and an over-specification critic. The session lead applied their findings and
   reverted two places where the draft re-opened owner decisions taken the same day
   (plugin discovery at M2 and renormalisation as a constrained Should).

The synthesis follows unchanged. Its G-numbers are cited in plan 0.5.

---


## 1. Verdict

**Yes, the plan is over-engineered. The problem is depth, not shape, and it also specifies too far ahead.** The core shape is sound and should stay:

- a functional core;
- one central point type;
- a circumplex defined by its content;
- the mapping as the schema;
- recorded settings;
- geometry kept separate from drawing;
- a project directory of plain files.

The minimal architecture, written from the PRD alone, arrived at almost the same skeleton. So the plan does not need replacing. It needs cutting back, re-tiering, and filling in at a few specific points.

The four biggest problems:

1. **It gives no signal about how firm each decision is.** The header says "everything is a proposal". §15 says "all decisions are now taken". Field-level code blocks for M3 and M4 types (Signal, MetricImplementation, PredictorSpec, SPISettings, Scenario) are written in the same voice and detail as the M1 contracts. Agents implement code blocks as written, so the plan works as a spec for five milestones at once. That is the lock-in the owner wants to avoid (G9-01, G9-05, G9-06).

2. **It builds mechanism ahead of its first consumer.** The following are all designed before anything uses them:
   - a progress and cancel protocol;
   - an executor-pluggable batch runner;
   - a metric registry with a backend selector;
   - three entry-point groups and a `plugins` CLI;
   - about 19 exception leaves;
   - published JSON schemas;
   - per-procedure METHOD_VERSION constants;
   - `load()` on every record type;
   - a multi-seed SPI;
   - two KS variants;
   - a fully designed renormalisation path;
   - the M2 to M4 CLI flags;
   - a version-pinned M3 library roster.

   Each piece is small, but together they are most of the plan's surface. None of it shapes M1 code, and all of it will be re-derived once real code exists.

3. **The one structural error that will spread is the monolithic `Settings` root in the bottom import layer (G2-02).** It either becomes a hub that every milestone edits or it breaks the layer contract. It also makes a projection result record density, quality and SPI settings it never used, and editing a bandwidth then marks that projection as `profile=None`. Settings must be recorded per operation, and a profile is a bundle of those records. This has to be fixed before the first result format is written.

4. **The detail is in the wrong places.** While M5 shapes are fully specified, the decisions that actually determine whether M1 is correct are missing:
   - whether stored responses are raw or coded (otherwise coding is double-applied, G3-03);
   - where `response_id` comes from (G3-15);
   - one per-response status contract across quality and projection (G3-13, G3-14);
   - axis names that collide with attribute column names (`pleasant`, G3-16);
   - the `ssp.project` function colliding with the `soundscapy.project` subpackage (G1-04);
   - a project that can hold only one survey, although E1 at M2 needs two (G7-12);
   - TOML's lack of null (G7-09);
   - an undefined content-hashing scheme (G7-08);
   - a reference syntax that allows machine-specific paths (G2-06, G7-13);
   - no boundary rule for points that lie exactly on a quadrant axis (G4-05);
   - no shared feature-name vocabulary from M2 to M4 (G5-10);
   - no split between what a metric implementation does and what the framework does (G6-04).

**What to do.** Rewrite the plan to about 550 to 650 normative lines in three tiers:

- **Commit now:** the M1 contracts plus the handful of cross-milestone invariants.
- **Design rules:** how to make later calls.
- **Deferred decisions:** each with a named trigger.

Close the twelve gaps listed under problem 4 in the M1 spec. Move research records, review history and field lists out of the plan.

## 2. What the architecture is

These are the load-bearing decisions. Treat them as the architecture. Everything else is guidance or belongs to a milestone spec.

1. **Functional core with immutable results.** Computation is pure functions over numpy arrays and frames, with no IO, printing, figures or global state. Result records are frozen. Arrays inside them are set `writeable=False` and frames are copied on construction. IO, the project directory and the CLI sit at the edges. No stateful `Project` class exists until a GUI or M3 needs one; until then a single path-helper module owns the directory layout.

2. **The circumplex is an instrument defined by its content.** `Scale`, `Attribute` (position id, angle, labels by ISO 639-3) and `Circumplex` are small frozen value objects. Every analysis works on generic positions. Records keep the instrument and the effective scale used as separate fields. Whether the scale counts toward "same circumplex for scoring" is a PRD clarification (FR-6.3) due before targets ship. Labels such as `pleasant` resolve to positions only at entry points (mapping construction, region selection, plot labels).

3. **`Coordinates` is the one type for survey, predicted and sampled points.**
   - `xy` has shape (n, 2), with NaN rows aligned to the input responses.
   - Labels are stored internally as numpy columns or a `nw.DataFrame` and converted to the native library only at user-facing returns.
   - Which label columns are required depends on `origin.kind`.
   - `origin` is a sequence of segments (one segment at M1), so pooling later does not change stored shapes. Any future concat refuses a circumplex mismatch.

4. **`SurveyMapping` is the schema, and stored responses are raw.** Coding (values, then offset, then reverse, in that fixed order) is applied once, inside the import boundary helper. Never apply it to stored data. The project keeps the byte-for-byte export plus its hash and re-applies the deterministic mapping on open.

5. **One per-response status contract.** There is one status per response that explains every absence across quality rules, missing items, language refusals and scale violations. It is expressed as `included`, `reasons` and `flags`, so "kept but flagged" can be represented. Three rules go with it:
   - a stated rule for where `response_id` comes from;
   - a namespace rule for derived columns (axis outputs and status columns cannot collide with canonical attribute or user columns, and a collision raises an error);
   - `n` and `n_valid` defined once and used by describe, the N < 30 rule and SPI.

6. **Settings are recorded per operation as frozen pydantic models.** Entry points take keyword arguments, plus an optional `settings=` or `profile=`. They resolve these into the operation's recorded model. A profile is a named bundle of per-operation records saved as one TOML file. TOML rule: an omitted key means the default, and no field has a `None` distinct from its default.

7. **Published parameters are versioned data, referenced structurally.** Angle sets (with tiers), targets, profiles and circumplexes ship as TOML files and are never edited or removed once released. Persisted references are structured `{name, version}` plus a content hash in the result, and never machine paths. User files are copied into the project and referenced project-relatively. One canonical hashing helper (sorted-key JSON, defined float formatting, explicit excluded fields, a scheme-version prefix) produces every hash.

8. **Every result carries provenance from the first result.** The envelope is the operation's `settings` plus one extensible `provenance` mapping: versions, created time, input-data hash, mapping hash and upstream run id. It is built by one helper and carried by every record, `Prediction` included. Data files carry a separate citation block (citation, DOI or URL, licence, date, notes). Both can only gain keys.

9. **Geometry is computed in `distribution`; drawing is a thin matplotlib layer.** Density grids, contours, region proportions, shifts and Likert proportions are returned as data. Plot functions take `ax=None`, return the `Axes` and forward keyword arguments. Shared helpers apply prediction labels, survey-language labels and the N check, so individual plot authors cannot forget them. A single estimator, `gaussian_kde`, is used with scipy `bw_method` semantics, and the factor and the bandwidth matrix are recorded.

10. **narwhals stays at the boundary.** The package accepts any eager frame and rejects lazy frames with `TypeError`. One helper converts inbound and outbound. Numpy is used inside. Frames the package creates, and frames read from paths, are pandas.

11. **The project directory is the unit of exchange.** The minimal M1 layout is:
    - `project.toml` (with an integer format version);
    - named datasets under `data/<name>/`, each with its own raw export, hash and `mapping.toml`;
    - `settings.toml` as the default input;
    - result tables as parquet plus an authoritative JSON sidecar written by one save routine;
    - a manifest.

    Per-run settings in a sidecar take precedence over the project defaults. Alphas carry no compatibility promise.

12. **Extension contracts are fixed in shape and grown when needed.** Entry points resolve to one light object per group: metadata is readable without heavy imports, loading is lazy (xarray's `BackendEntrypoint` pattern), and each object declares a plugin-API version. A group is created when it has a real member. For metrics, the framework owns statistics, channel rules, naming and units. Implementations return per-channel series, cross-channel values or standard-defined scalars. The outer shape of the predictor interface (owner decision) is: a spec readable without loading, a declared output kind, and a `deployable` flag. Only deployable predictors are barred from perceptual inputs.

13. **One feature-name vocabulary from M2 onward.** The vocabulary is `<metric>_<statistic>`, with a reserved channel slot, naming the semantic quantity. The computing method or backend goes in provenance. The M2 ISD loader maps the ISD's own column names onto it, M3 emits it, and M4 input contracts bind to it.

14. **Layering is enforced minimally, and the error vocabulary is small and typed.** One ordered import-linter layers list, plus a few protections: the core never imports plot, cli, project or matplotlib, and pandas and polars appear only at IO boundaries. `SoundscapyError` and `SoundscapyWarning` are the roots, subclassing the built-ins (ValueError, ImportError, UserWarning). There is one class per condition, and leaves are added when code first raises them. Logging goes through one package logger with a single silencing call, and that call also governs pooch.

## 3. Decision register

Classification key:

- **Now:** decide in the plan or the M1 spec, before the first persisted file.
- **Rule:** a design rule; the details belong to the spec.
- **Defer:** defer until the named trigger.
- **Simplify:** keep a lighter version.
- **Drop:** remove.

| ID | Item | Class | Trigger | Reason |
|---|---|---|---|---|
| G1-01 | Frozen pydantic settings models | Now | Signature style in M1 spec | FR-11.1/11.3/14.2 need it from the first result. Both verifiers agree. Keyword arguments are the primary input and the model is what gets recorded. |
| G1-02 | Functional core, frozen records | Now | – | Add the rule: arrays read-only, frames copied. Retrofitting this later means rewriting every module. |
| G1-03 | Two-tier API (~20 names at top level) | Rule | M1 spec for the list; settle at M2 course; freeze at first non-alpha | PRD §9 freezes nothing during alphas. The proposal's "freeze at M2" is corrected. |
| G1-04 | `ssp.project` versus `soundscapy.project` collision | Now | Before M1 spec | The import-order bug is real. Rename one of them (for example a `workspace` package or `project_coordinates`). |
| G1-05 | Two-minor-release deprecation cycle | Simplify | FR-13.4 policy at M2 | Alphas may break anything, and the cycle starts at 1.0 final. Method-version bumps still apply. |
| G1-06 | `rng` keyword on random functions | Now | – | Copy SPEC 7 exactly. If `None` is accepted, the seed actually used must be recorded. |
| G1-07 | Logging mechanism | Rule | M1 ISD loader | The verifiers split. NFR-13 (owner) is fixed now; the mechanism is settled at the first log line. Fix the pooch logger and drop the duplicate announcement. |
| G1-08 | Exception hierarchy of ~19 classes | Rule | Roots and built-in bases at M1; each leaf when first raised | Rename `CancelledError` so it does not shadow asyncio's. |
| G2-01 | Scale, Attribute, Circumplex | Now | – | Inputs to M1 projection. Optional fields can be added later without breaking anything. |
| G2-02 | Monolithic `Settings` root in the bottom layer | Simplify (now) | M1 spec, before the first result format | Per-operation records, with profiles as bundles. Otherwise a hub or a layer breach, and false `profile=None`. |
| G2-03 | Persisted circumplex fingerprint and `CircumplexRef` | Defer | M2 spec, when Target files ship | Use value equality at M1. M1 results must embed the full circumplex. |
| G2-04 | Is the scale part of circumplex identity? | Now | M1 records instrument and effective scale separately; identity rule by M2 (PRD clarification) | Otherwise 1..7 surveys can never be scored against 1..5 targets, or the record misstates the scale. |
| G2-05 | AngleSet as versioned TOML with tiers | Now | – | FR-2.4 and FR-2.4a Musts. Fix the identity keys and the tier field; the file layout goes in the M1 spec. |
| G2-06 | `name@version`-or-path reference strings | Now | M1 for angle sets; extend to targets at M2 | Use a structured reference with no paths and record the content hash. A universal DataRef is gold-plating. |
| G2-07 | Per-language angle-set choice | Simplify | – | The verifiers split; resolved by keeping the FR-2.4 Must. Keep an optional language-to-version map over registered sets, drop the path values, and keep a per-response lookup and an `angle_set` column. |
| G2-08 | `equal` override and two knobs | Simplify | – | Both verifiers rejected dropping it. Kept only because FR-2.4 and the owner's §10.5 decision require it. Keep both knobs. The mix report is a count of `angle_set` labels, with no extra machinery. |
| G2-09 | `dispersion` in ProjectionSettings | Simplify | M1 spec, with the item-statistics function | It belongs to item statistics (FR-2.2). The profile carries it. |
| G2-10 | Aletta 2024 formulation kernel | Now (conditional) | Owner supplies a worked example before the M1 spec | The verifiers split. FR-2.3 is an M1 Must, so the kernel is kept, but it cannot be tested without a fixture. If none arrives, see the PRD change in §4. |
| G2-11 | Profiles as TOML, results record resolved values | Now | – | FR-2.9 and FR-11.1. Per operation, following G2-02. `derived_from` goes to the M1 spec. |
| G2-12 | Labels accepted "wherever" a position is expected | Simplify | – | Resolve labels at entry points only. |
| G3-01 | SurveyMapping is the schema | Now | – | M1 Musts. The recipe starts from an annotated ISD `mapping.toml`. |
| G3-02 | Coding order: values, offset, reverse | Now | – | The order is part of the file's meaning. Key normalisation is a spec detail. |
| G3-03 | Raw versus coded invariant | Now | – | Missing load-bearing decision. Otherwise coding is silently applied twice. |
| G3-04 | Mapping field set | Simplify | M1 spec | Make the group and position keying unambiguous and drop free-text language names. Keep the FR-1.3 construct fields (source dominance, WHO-5 and so on) at M1, as the requirements verifier says. |
| G3-05 | Participant and stimulus id roles | Defer | FR-2.4b or M5 SATP/ARAUS loaders | Optional fields can be added later. |
| G3-06 | Dataset group, session, location and recordings tables | Defer | Groups at M2 (ISD feature table); recordings and sessions at M3 | The M1 spec states that location context is carried as columns. |
| G3-07 | Key rule for hierarchy tables | Rule (now) | Applied in M2 groups | Hierarchy tables key on the Ids column names. One line now. |
| G3-08 | narwhals boundary helper | Now | – | Owner decision, implemented at the minimum. Frames the package creates follow one stated default. |
| G3-09 | `backend=` argument, polars extra, pandas 2 CI axis | Simplify | First user request or M2 course | Return pandas from paths. Keep polars in tests for frames passed in. Settle the pandas floor once. |
| G3-10 | Mapping suggestion with confidence and evidence | Simplify | M1 spec with real exports; interactive confirm at M2 | Exact and alias matching returning a small result: the draft mapping plus unmatched columns. |
| G3-11 | Export schemas as TOML files | Simplify | Qualtrics aliases in M2 spec | A Python dict per schema. The tutor who maintains them is hypothetical. |
| G3-12 | Quality rules as rule objects | Rule | M1 spec, with the ISD | Persist enabled names and thresholds, never callables. Check thresholds against the ISD's published exclusions. |
| G3-13 | Flagged-but-kept state | Now | – | Two columns cannot express it. Add a `flags` column before the names are published. |
| G3-14 | Contract between quality status and projection status | Now | – | One status. Decide ScaleError versus the flag and define `n`/`n_valid` before projection is written. |
| G3-15 | Origin of `response_id` | Now | – | Choose require, generate-and-record, or derive. Ids based on row order break on re-sort. |
| G3-16 | Derived column namespace | Now | – | `pleasant` is both an attribute and an axis name. A naming rule and a collision error are needed. |
| G3-17 | Renormalise, ipsatise, cluster machinery | Defer | After 1.0 or with spare M2 capacity; FR-2.4b at M5 | Shoulds. Drop `n_items_present` from the always-present columns. |
| G3-18 | ISD loader via pooch | Now | – | FR-10.1 M1 Must. The pinned version waits on the owner. |
| G3-19 | MOSAIQ `describe()` on loaders | Defer | MOSAIQ declared stable or M5 | Plain citation, licence, version and URL metadata at M1, under a name other than `describe`. |
| G3-20 | DatasetLoader Protocol | Drop | – | Keep the module convention only. |
| G3-21 | Loader mapping placed in `_data/schemas/` | Simplify | – | Put it next to the loader module. |
| G4-01 | Coordinates as the single type | Now | Required labels per origin kind in M1 spec | The central decision. Drop the mean/sd/quantiles methods in favour of describe. |
| G4-02 | Labels stored in the native library | Simplify (now) | – | Store them internally and convert at the edge. Otherwise every consumer needs dual-backend code. |
| G4-03 | Composing Coordinates from several origins | Rule (shape now) | Concat built when journey G or a pooled E1 or course case needs it | The verifiers split. Store `origin` as a sequence from M1 at almost no cost, which avoids a change to stored shapes. No concat code yet. |
| G4-04 | Origin payload union | Defer | M2 sample; M4 prediction | Keep the `kind` discriminator from M1. |
| G4-05 | Rule for points on a region boundary | Now | M1 spec, before the first baseline | Integer Likert scores at 45° put many points on the axes. Record the rule and keep it consistent with the KS tie rule. |
| G4-06 | `gaussian_kde` with scipy semantics | Now | – | Needed for journey I. The bw_adjust conversion arithmetic goes in the reproduction spec. |
| G4-07 | KDE grid fixed to [-1,1]² | Defer | M1 spec at the latest | Make grid extent and normalisation fields. Use the 2022 settings if the owner supplies them; otherwise choose and record. |
| G4-08 | KDE sensitivity helper | Drop | – | No requirement asks for it. Show it in a how-to instead. |
| G4-09 | Class named `KernelDensity` | Simplify | M1 spec | Collides with sklearn's class of the same name. Rename it. |
| G4-10 | SkewNormal and fit layout | Defer | M2 spec, after the sn fixtures | Guidance: direct parameters in sn terminology, fitting in centred parameters, a frozen value object. |
| G4-11 | 0.x msn port and branch notes | Simplify (move out) | M2 skew-normal implementation; ADR for the AGENTS.md exception | The verifiers split. Keep a two-line pointer to the validated port and R baseline in the plan; move the history to an ADR. |
| G4-12 | SPISettings variants and fields | Defer | M2, after the owner confirms the paper's method | Implement the paper's combination only and record its name. |
| G4-13 | Multi-seed SPI | Defer | M2 spec | The verifiers split. It is a Should the owner agreed, so it is not dropped. Keep it out of SPIResult's core fields; a separate helper is likely. |
| G4-14 | `spi()` signature | Rule | M2 spec | Required seed and recorded method fields (FR-6.1). |
| G4-15 | Three ranking entry points | Simplify | M2 spec | One cross-product function returning a tidy table. |
| G4-16 | Target model fields | Rule | M2 spec, with fixtures | The FR-6.3 rules bind. The fields come from fixtures. |
| G4-17 | Plot `ax` contract | Now | – | FR-8.3. Breaking it later breaks every tutorial. |
| G4-18 | Geometry kept out of plotting | Now | – | FR-8.4 counts proportions as geometry. Small survey-level functions are acceptable. |
| G4-19 | Per-author label obligations | Simplify | – | Shared helpers plus one test template. |
| G4-20 | `plot.by_group(layers=…)` | Simplify | – | A `kind=` argument covering the three Part 3 views, overlay or panel, and the built-in N < 30 fallback. No layers list. |
| G5-01 | Design types and `evaluate` | Simplify | M2 spec (Example 6); widen at M4 | Scenarios keyed by name; `produced_by: SurveyRef` now; KPI as a tuple of thresholds; one table. ScenarioInputs at M4. |
| G5-02 | pymoo target search | Defer | M5 journey F | Keep only "returns a Target with provenance". |
| G5-03 | Outer shape of the predictor interface | Now | Freeze as code at M4 | Owner decision. The output contract must allow pleasantness-only models. |
| G5-04 | Concrete predictor contract classes | Rule | M4, against two real predictors | A hard perceptual-input validator would contradict FR-5.3a. Use a `deployable` flag instead. |
| G5-05 | OptionalInputWarning | Drop | – | Absent optional inputs are recorded. A warning that fires on intended use teaches users to ignore warnings. |
| G5-06 | `plugins.discover` at M2 and a `plugins` CLI | Defer | First group with a real member (predictors at M4 at the latest) | The verifiers split. Shipping discovery with no members tests nothing and fixes an untested shape. This overrides PRD §9; see §4. |
| G5-07 | Three entry-point groups | Defer | When an external metric or dataset package exists | Reserve the names as guidance; implement only the predictors group. See the PRD change. |
| G5-08 | In-tree metrics registered through entry points | Simplify | – | Use a plain in-tree tuple, merged with discovered entries; a duplicate raises. |
| G5-09 | Uniform entry-object shape and plugin-API version | Rule (now) | Payloads at M4 | The verifiers split. The convention is cheap to state and is the only contract that crosses package boundaries. |
| G5-10 | Shared feature-name vocabulary | Now | M2 ISD loader spec | Otherwise M4 binds to two vocabularies or published course material gets renamed. |
| G6-01 | Signal, read() and Calibration types | Defer | M3 spec, with real recorder files | Keep three rules: calibrate once and record it, at most two channels, windowed reads. Lean toward soundfile only. |
| G6-02 | MetricImplementation attributes | Rule | M3, after two wrappers | Add definition and citation. Version and licence come from package metadata. |
| G6-03 | Registry keyed by (metric, backend) | Defer | M3 spec | The verifiers split. Phonometry and MoSQITo collide on ISO 532-1 and DIN 45692, so a collision rule is needed at M3, but not a selector now. |
| G6-04 | Ownership between metric implementation and framework | Now | Details finalised in M3 | Missing load-bearing decision. It decides whether a metric recipe is 20 lines or 200. |
| G6-05 | Batch runner design | Defer | M3 spec, with measured runtimes | Rules: the per-file cache is the resume mechanism, and work functions are top-level. joblib is a candidate. |
| G6-06 | Time-series cache | Defer | When FR-4.11 is scheduled | Shoulds. Series stay in memory. |
| G6-07 | Two equalisation filter formats | Simplify | M3 if scheduled | Arrays in the API plus one serialisable form. |
| G6-08 | M3 backend roster and the LISQM appendix | Simplify (move out) | M3 planning, re-running PRD §10.4 item 2 | The verifiers split. Keep the owner policy lines in the plan; move the roster to a dated research note. |
| G6-09 | ProgressReporter and tqdm in core | Defer | M3 batch runner | Use pooch's own progressbar at M1. Use a work-unit label, not a path. |
| G6-10 | CancelToken and partial results | Defer | M3 batch runner spec | Resuming from the cache likely replaces the partial-result payload. Rename the error class. |
| G6-11 | Downloads must take progress and cancel | Simplify | Now | The verifiers split. Downloads report progress through pooch; record non-cancellability as an FR-14.3 deviation. |
| G7-01 | Result envelope | Now | – | One `settings` field plus one provenance mapping, built by one helper and carried by every record. |
| G7-02 | Undefined Provenance type | Now | – | Missing load-bearing decision. A minimal citation block for data files, separate from run provenance. |
| G7-03 | METHOD_VERSION on every procedure | Simplify | First beyond-tolerance change in a baselined procedure | Only baselined procedures get one. Profiles are not versioned. |
| G7-04 | Parquet plus JSON sidecar | Simplify | – | The verifiers split. Resolved: the sidecar is authoritative and written with the table by one routine; mirroring into parquet metadata is optional. |
| G7-05 | Parquet and pyarrow in core | Now | – | Kept only because of owner decisions (FR-14.2 and accepting pyarrow's size, 2026-10-30). Both verifiers say keep. |
| G7-06 | `load()` and round-trip on every record | Simplify | Project.open needs results (M3 or GUI) | Readers only where something reads back. One generic serialisation path so readers can be added later. |
| G7-07 | Result lineage | Now (narrow) | Staleness logic at GUI planning | The verifiers split. Input hash, mapping hash and upstream run id as provenance keys cost almost nothing now and avoid a migration. |
| G7-08 | Canonical hashing with a scheme version | Now | – | A scheme change otherwise refuses every shipped target and misses every cache. |
| G7-09 | TOML has no null | Now | – | Omitted means default. Without this rule FR-11.3 round-trips fail. |
| G7-10 | `show_versions()` | Simplify | With the M1 provenance helper | A convenience that comes free with the provenance helper; not load-bearing. |
| G7-11 | Coded copy plus schema sidecar | Simplify | – | Raw export plus hash; re-apply the mapping. The id rule moves to the import spec. |
| G7-12 | Project holds one survey | Now | – | Missing load-bearing decision. E1 at M2 needs two. Use named datasets under `data/<name>/`. |
| G7-13 | Project portability | Rule (now) | Vendoring automation when sharing is built | The verifiers split. Resolved: no absolute paths in persisted references; import copies user files in. Cheap now, a migration later. |
| G7-14 | Format versioning and settings precedence | Now | Precedence in M1; policy by M2 alpha | One paragraph: integer versions, no alpha compatibility, per-run settings win. |
| G7-15 | Stateful Project class | Defer | GUI design or M3 | Plain functions plus one path-helper module. |
| G7-16 | Full layout as a contract | Rule | – | Direction of travel only; fix just the M1 subset. |
| G7-17 | Published JSON schemas | Defer | GUI work, after the fields survive M3 | pydantic generates them on demand. |
| G7-18 | Serving both in-process and out-of-process GUIs | Simplify | If GUI planning chooses out-of-process | The contract is the library plus the project directory. |
| G7-19 | CLI principle and M1 commands | Rule | M1 spec | One library call per command. Fix the claim that commands touch only the project directory. |
| G7-20 | M2 to M4 command names and flags | Defer | Each milestone spec | They wrap signatures that are not yet fixed. |
| G7-21 | Default CSV copies plus an `export` command | Simplify | M1 spec | Choose one; keep an Excel path. |
| G7-22 | C and E1 CLI at M2 | Rule | M2 spec | The verifiers split. Kept only because of PRD FR-14.5. Recommended narrowing in §4. |
| G8-01 | Thirteen subpackages | Rule | Create each with its first code | Reconcile `plugins` between the tree and the layer table. |
| G8-02 | Detailed import-linter contracts | Simplify | Extend per milestone | Minimal Phase 0 protections only. |
| G8-03 | Extras list | Defer | The milestone of each extra's first module | Phonometry stays as an owner decision. |
| G8-04 | Pre-emptive ruff ignores | Simplify | First real hit; `runtime-evaluated-base-classes` at Phase 0 | Lives only in house-style. |
| G8-05 | statsmodels in core | Defer | First package feature that needs it; docs group otherwise | scipy.stats covers FR-2.6 and FR-2.6a. |
| G8-06 | seaborn and an `analysis` extra | Drop | – | No Must needs them. seaborn at most in the test group. |
| G8-07 | Own `tidy()` tables | Rule | M1 survey spec | Use pingouin's semantics and effect sizes with snake_case column names. |
| G8-08 | Test kinds and templates | Rule | Markers at Phase 0; templates with each recipe | Recipes arrive at M2, M3 and M4 under PRD §9. |
| G8-09 | Golden table for the 2022 reproduction | Drop | – | Contradicts the owner's decision to keep no numeric baseline for that paper. Edit the §14 M1 row to match. |
| G9-01 | No split between commitments and guidance | Now | Before the M1 spec | The central fix. §2 of this report is the commit list. |
| G9-02 | Concept budget per recipe | Rule | Each recipe; the M5 trial is the gate | A numeric cap now would be a guess. Record the concept and file counts in each recipe instead. |
| G9-03 | Phase 0 ADRs | Simplify | End of the first M1 slice | Write each ADR from code that has exercised the decision. |
| G9-04 | Decisions recorded four times | Simplify | Now | One register (§15); PRD §10 points to it; rows become ADR links over time. |
| G9-05 | Field-level code blocks | Rule | M1 and M2 specs | Label them illustrative. Keep defaults only where the PRD or the science fixes them, with the citation. |
| G9-06 | A 1,480-line plan used as a guide | Simplify | Same pass as G9-01 and G9-07 | Restructure in place, not as a second document. |
| G9-07 | Stale and inconsistent names | Simplify | Now | One pass: MetricImplementation and SurveyMapping everywhere; DatasetLoader removed; `plugins` placed in the tree. |
| G9-08 | Recipe milestone labels | Defer | M2 plot and dataset; M3 metric; M4 predictor | The conventions stay as M1 guidance in §5. |

## 4. PRD changes

The owner decides each of these. They are listed because the plan cannot be simplified honestly without them.

1. **FR-6.3: clarify what circumplex identity is for scoring.** State whether the response scale is part of the identity, or whether identity is positions, angles and axes with the effective scale recorded separately. Without this, a 1..7 survey can never be scored against a shipped 1..5 target (FR-6.4). This is a clarification, not a cut, and it is required before M2.

2. **§9 and FR-13.3: plugin discovery ships with the first entry-point group that has a real member.** That is the predictors group at M4 at the latest, or earlier if an external metric or dataset package appears. The group names `metrics` and `datasets` are reserved but not implemented until a third party exists. Shipping discovery at M2 with no members tests only monkeypatching and fixes an untested shape into 1.0a1.

3. **FR-2.3: make the Aletta 2024 formulation conditional on a fixture.** Add an Aletta 2024 worked example to the owner inputs (§10.3). If none exists by the M1 spec, move the formulation to Should. The angle set remains an M1 Must in any case.

4. **FR-2.5a: move renormalisation to Later, on user demand.** No standard defines it, it carries six constraints, and the Must ("exclude, with a reason") needs none of that machinery. Use exclusion counts from course and ISD use to decide whether it matters.

5. **FR-14.5: narrow the M2 CLI to E1 `evaluate <brief.toml>` plus scoring against shipped or bespoke target files.** Target fitting and editing from the CLI waits for GUI planning. The Brief file is needed anyway (FR-14.2), so E1 costs little. C's fitting and editing commands have no primary M2 user, because the course runs in notebooks.

6. **FR-14.3: record an explicit exception for downloads.** They report progress through pooch and are not cancellable. Revisit only if a GUI needs cancellation; pooch's `downloader=` hook makes that a later fix.

7. **FR-1.5: replace the thresholds that are untestable as worded** with "defaults that reproduce the ISD's published processing", and add the ISD exclusion counts to the owner inputs.

8. **FR-2.4: keep per-language version choice, but scope it to registered sets.** Chosen by name and version, with no file paths. User sets with provenance are registered first.

Considered and not changed: FR-14.2's columnar format (owner decision), FR-6.1's multi-seed Should (agreed), and the FR-1.3 construct list (M1 Must).

## 5. Comparison with the minimal architecture

**What the minimal draft got right that the plan buries or lacks:**

- **Rule 8: "an abstraction comes with its second implementation."** This single rule would have prevented most of the register's defer and drop rows (metric backend selection, three plugin groups, the DatasetLoader protocol, two KS variants, executor injection, two WAV readers). The rewritten plan should adopt it verbatim, with the predictor interface as the stated owner exception.
- **Rule 10: "each milestone ships alone."** A clean test for any design call.
- **One status and reason pair per response, and "row count never shrinks."** This is the invariant the plan scatters across two places and never merges (G3-13, G3-14). The minimal draft states it as a central concept.
- **Hierarchy as id columns plus joins, not object graphs.** This matches the verdicts on G3-06 and G3-07 and is stated more crisply than in the plan.
- **An `origin` on every row (observed, or `predicted:<model@version>`).** This anticipates G4-03 better than the plan's single origin record.
- **A decisions-left-open table with triggers, and a "where heavier implementations should be resisted" list.** These are exactly the two artefacts the plan lacks (G9-01). Both should be copied into the rewrite almost as they stand.
- **Plot types as a recipe, with no registry until a third party asks.** The same goes for extension groups being "added in the milestone that first needs it" (D7).
- **Length.** About 230 lines covering the same skeleton is evidence of the target size.

**Where the minimal draft is wrong or thin, and the plan is right:**

- **Settings representation.** It proposes stdlib dataclasses with hand validation, deferring pydantic. Both verifiers held that pydantic is needed from M1 for FR-11.1, FR-11.3 and FR-14.2, and that switching later touches every module. The plan is right.
- **Progress and cancellation (D9).** It puts these hooks on all long functions now. The register defers them to M3, because no M1 or M2 operation needs them and the batch-runner design will decide whether a partial-result payload exists. The minimal draft over-reaches here.
- **Coordinates as columns of the response table (D2).** This fails for sampled target points and pooled predictions, and it causes the `pleasant` namespace collision directly. The plan's separate `Coordinates` type with row-aligned NaN is the better decision. Keep it, and keep the minimal draft's row-count invariant inside it.
- **Persisted circumplex hash at M1 (D1).** The register defers the persisted fingerprint to M2, uses value equality at M1, and adds the canonical-hashing rule (G7-08), which the minimal draft does not mention.
- **Details it omits that the plan rightly specifies or the verifiers added:**
  - the narwhals boundary helper and lazy-frame rejection;
  - pooch with record-version pinning;
  - the coding order of operations;
  - scipy KDE semantics with a recorded bandwidth matrix;
  - SPEC 7 `rng`;
  - the pooch logger defect;
  - built-in exception bases.

  The gaps neither document addressed were also missed by the minimal draft: raw versus coded, `response_id`, the TOML null rule, projects holding several datasets, the feature vocabulary, the metric/framework boundary, the region-boundary rule, and the `ssp.project` name collision.

**Net:** use the minimal draft's skeleton, rules and open-decisions table as the frame of the rewrite. Use the plan's verified M1 contracts as the content. Add the gaps from the verified register.

## 6. Proposed structure for the rewritten plan

Target: about 550 to 650 normative lines, down from 1,480.

| § | Section | Target length | Content |
|---|---|---|---|
| 0 | Status and how to read | 15 lines | The three tiers (commit, rule, deferred) and the rule in this section's closing paragraph. |
| 1 | Forces | 35 lines | Capacity, faithfulness, four front ends, generality, the big unknowns, alpha freedom, install simplicity. |
| 2 | Design rules | 70 lines | The current §2 principles merged with the minimal draft's rules 1 to 10, including "second implementation" and "each milestone ships alone". Add the resistance list (§8 of the minimal draft) as a subsection. |
| 3 | Commitments | 40 lines | One table, a row per decision in §2 of this report, each tagged with the milestone it binds from. This is the only list of things that are not open. |
| 4 | Load-bearing decisions | 230 to 280 lines | One subsection per commitment (10 to 25 lines each): what is fixed, why, what it rules out, and what is left to the spec. Prose and invariants only. Code appears only as a clearly labelled illustrative sketch of at most 10 lines. |
| 5 | Module map and layering | 45 lines | The dependency diagram, one ordered layer list, and the minimal Phase 0 import-linter protections. Subpackages are created with their first code. |
| 6 | Conventions | 60 lines | API tiers, keyword arguments resolved into recorded settings, `rng`, frames at the boundary, column naming and namespace, errors and warnings (roots plus the one-class-per-condition rule), logging, plot `ax` contract, deprecation during alphas. |
| 7 | Milestone direction | 70 lines | For each of M1 to M5: what it adds and which deferred decisions it must take. No field lists, flags or library versions. |
| 8 | Deferred decisions | 50 lines | The register's defer rows as one table: decision, trigger, constraints already binding it. |
| 9 | Owner decisions | 20 lines | One line each, linking to PRD §10.5, with the ADR link once written. Replaces the current §15 duplication. |
| A | Appendix A (non-normative) | ≤120 lines | Illustrative type sketches for M1 types only, marked "illustrative; the M1 spec is authoritative". |
| B | Appendix B (non-normative) | ≤30 lines | Extension recipe outlines with corrected milestones (plot and dataset at M2, metric at M3, predictor at M4) and current names. |
| C | Appendix C (non-normative) | ≤30 lines | Glossary: position, instrument, angle set, profile, origin, status, run. |

**Moves out of the plan entirely:**

| Content | New home |
|---|---|
| Current §13 (departures from draft 0.1) and all review provenance | An archive file next to the old draft. Not normative and not linked from AGENTS.md. |
| 0.x branch notes (msn port, scipy-msn breakage, test counts) | ADR recording the AGENTS.md exception, plus the M2 spec. The plan keeps a two-line pointer. |
| Appendix C (LISQM evaluation) and the version-pinned M3 roster | A dated research note under `design/research/`, re-run at M3 planning. The plan keeps the policy lines (MoSQITo default, phonometry as an explicit extra, no PyQt6). |
| §6 tooling, §16.2 dependencies, ruff ignores, filterwarnings exemptions | `house-style` and `pyproject.toml`. |
| §8 test detail beyond "four test kinds, tolerances justified, hypothesis for invariants" | `house-style` and the specs. |
| §16.1 M2 to M4 CLI tables, §16.3 quality rule thresholds, SPISettings, SkewNormal fields, the §9 batch-runner and progress design, the §12 checklist items | The milestone specs. §12's surviving content folds into the §2 rules. |
| §17 open findings | The deferred-decisions table (§8 of the new plan), or the issue tracker. |
| Phase 0 ADR-0001 to ADR-0004 | Written after the first M1 vertical slice, from code that exercised the decisions. |

**The rule for what goes in the plan and what goes in a milestone spec:**

Content belongs in the plan only if at least one of these holds:

- it crosses a module or milestone boundary;
- reversing it after the first persisted file or alpha release would force a migration or rewrite across modules;
- it constrains how future decisions are made (a rule or a trigger).

Everything else goes in the spec of the milestone that first needs it. That includes field lists, defaults, signatures, flags, thresholds, library choices and versions, file layouts beyond the M1 subset, and recipe text.

One exception: a default fixed by the PRD or the cited science may appear in the plan with its source cited. Code in the plan is always labelled illustrative. When the plan and a spec disagree, the spec wins for its milestone, and the plan is updated in the same change.
---

## Owner responses (2026-10-31), applied in PRD 0.8 and plan 0.5 §9.2

The six proposed PRD changes in plan 0.5 §9.2 were all accepted. On the first, the owner's
reading: the scale is part of the projection's identity. Once projected, coordinates on the
same two axes are cross-comparable whatever their original scales or labels. Whether
coordinates are comparable across axis pairs (for example indoor against outdoor
circumplexes) is an open scientific question. The plan therefore refuses cross-axis-space
scoring by default and records an explicit override.
