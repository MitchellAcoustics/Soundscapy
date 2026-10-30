# Product Requirements: an open-source soundscape assessment, analysis and design package

Status: **draft 0.1 for review** (2026-10-30). This document states what the package
must do and for whom. It deliberately says nothing about how it is built; the
architecture plan is a separate document and must be checked against this one.

Decisions already taken with the product owner are marked **[agreed]**. Items the
product owner has not yet ruled on are marked **[assumed]** and listed again in
§10 for confirmation.

---

## 1. Purpose

Provide the reference implementation of the ISO 12913 analysis and design workflow
(Part 3 data analysis, including its 2024 revision, and Part 4 design and
intervention) together with the research methods that extend it: the distributional
representation of soundscape perception, language-adjusted circumplex projection,
Soundscape Perception Indices (SPI), psychoacoustic analysis of in-situ recordings,
and predictive modelling of perception from audio.

The package must be usable in three ways: as a Python library, inside teaching
notebooks, and as the engine of a graphical application for non-programmers.

### 1.1 Problems with the current tool that this release must fix [agreed]

1. **Installation and setup.** The R dependency and basic Python setup defeat
   non-programmers; Colab has been the workaround.
2. **Undelivered science.** Translation-adjusted angles, SPI and predictive modelling
   exist in papers but not as usable tools; the unified framework proposed in 2023
   has not been built.
3. **Audio analysis.** Slow; binaural-only; no handling of calibration signals; not
   integrated with the survey side of the package.
4. **Plotting and survey analysis.** Inconsistent, hard to use, hard to customise,
   hard to develop.
5. **Documentation and maintainability.** Repetitive docs; hard to extend or
   maintain.

### 1.2 Success, one year after release [agreed]

- Forward-thinking consultants use the tool, SPI and predictive modelling in real
  projects.
- Researchers adopt the adjusted-angle projection and SPI at the rate they adopted
  the circumplex density plots, because usable tools now exist.
- Continued and increased uptake in papers and in the ISO 12913 standards process.
- The certification course teaches on the new tool, with fewer participants lost at
  installation.

---

## 2. Scientific and standards baseline

The package implements, and must stay faithful to, the following. Where two sources
disagree the later one wins and the earlier one is kept as a compatibility option.

| Source | What it fixes |
|---|---|
| ISO/TS 12913-2:2018 | Method A questionnaire (8 PAQ items, 4 source-dominance items, overall quality, appropriateness), Method B scales, Method C interviews; binaural recording and equalisation |
| ISO/TS 12913-3:2019 | Scale coding (Table A.1), fixed 45° projection divided by (4+√32), median and range, Spearman with ties, binaural metric set (Table D.1), higher-of-channels rule |
| ISO/AWI TS 12913-3 (2024 revision draft) | General angle projection with λ scaling (A.1 to A.4), per-response projection, mean and SD per group, interquartile range for ordinal items, three named visualisations (scatter; KDE with decile contours; 50th-percentile contour), N < 30 rule, N_rmc, S_50 and IACC added, placeholder section on translated attributes |
| ISO/CD TS 12913-4 (2025) | Design process: communication framework, Soundscape Design Brief with ambition statement and KPIs, pre- and post-intervention assessment, evaluation report; SPI named as a quantitative KPI; Example 6 (baseline SPI 56 against a tranquil target, success at SPI > 70) |
| SSID Protocol (Mitchell et al. 2020) | Location → Session → Group → Response hierarchy; one 30-s binaural clip per group; item coding; environmental and session metadata; id conventions |
| Mitchell, Aletta & Kang 2022 | Distributional representation; projection before aggregation; "percent likely to perceive"; 50% contour as provisional |
| Mitchell & Aletta 2023; Aletta et al. 2024 | Language-specific adjusted angles for the eight attributes |
| Mitchell et al. 2024 (SPI) | Bivariate skew-normal targets (direct parameters), two-dimensional KS distance, SPI = 100(1 − D), bespoke and reference targets, NSGA-II derivation of reference targets from rankings |
| Mitchell et al. 2021 (lockdown model) | Psychoacoustic feature set and conventions; multilevel model with location effects; evaluation by MAE and location-mean R² |
| Mitchell et al. 2023 (framework) | Goals (accuracy, automation, comparison) and constraints (measurable inputs only, automatic calculation, generalisability, robustness to missing optional inputs) for predictive models |
| Hou et al. 2023, 2026 | Deep audio predictors (DCNN-CaF, SoundAQnet): audio in, sound sources, scene, 8 PAQs and ISO coordinates out; single-response prediction |
| Liang et al. 2026 (datasets review); MOSAIQ schema draft | Twenty datasets, their heterogeneity, FAIR status; two-layer catalogue plus clip-level package; item-level responses kept separate from derived coordinates; explicit missingness; provenance |

---

## 3. Personas and priority

| # | Persona | Priority this release | What they need |
|---|---|---|---|
| P1 | Soundscape researcher (Python or GUI) | **Primary** | Reproducible, citable, standards-traceable analysis; publication figures; the recent methods as working tools; a way to extend the tool with new methods |
| P5 | Course participant (researchers ~40%, acoustics consultants ~25%, local-authority environmental health officers ~25%, designers ~10%) | **Primary** | Install and run the complete workflow in a session without prior Python; guided exercises; the same tool after the course |
| P6 | Contributor or maintainer (volunteer, mixed experience) | **Primary** | Add a metric, dataset, predictor or plot from a recipe without understanding the whole package; confidence that changes do not break the science |
| P2 | Acoustic consultant | Features primary, direct use after GUI | Defensible assessment, SPI against a client-agreed target, predicted effect of interventions, report outputs; Windows; limited coding |
| P3 | Architect, urban or landscape designer | Features primary, direct use after GUI | Evaluate design options visually against a design brief; no coding |
| P4 | Modelling or ML researcher | Secondary | Harmonised datasets, feature pipelines, a place to plug in models; served by the above |

[agreed]

---

## 4. User journeys

Priority for this stage of work: **Must**, **Should** (nice to have), **Later**.

### A. Survey to findings — Must
*Actor:* P1, P5, P2.
*Has:* a survey export (REDCap, Excel, CSV) from an ISO 12913-2 Method A study,
possibly in a non-English language, with location and session identifiers.
*Wants:* validated data; circumplex coordinates using the correct angles for the
language; per-location distributions; statistics required by Part 3; figures and
tables for a paper or report.
*Done when:* the user can go from export to Part 3-compliant statistics and the three
Part 3 visualisations for every location in one session, and every number is
reproducible from a saved configuration.
*Hardest step today:* getting an arbitrary export into the expected shape.

### B. Recordings to features — Must
*Actor:* P1, P2.
*Has:* tens to thousands of recordings (mono, binaural or multichannel), calibration
information, and a table linking recordings to survey groups.
*Wants:* the Part 3 Table D.1 metric set and the research feature set, per channel and
as representative single values, in one tidy table joined to the survey data.
*Done when:* 1,000 thirty-second recordings process overnight on a laptop, the run
is resumable, the settings are saved with the results, and selected metrics agree
with reference implementations within stated tolerances.
*Hardest step today:* speed, calibration, trusting the numbers.

### C. Score a site — Must
*Actor:* P1, P2, P3.
*Has:* observed or predicted perception for a site, and a notion of the desired
soundscape (a quadrant, a published reference target, or a client-specified shape).
*Wants:* an SPI value with its uncertainty, the target shown against the data, and an
explanation a client can follow.
*Done when:* Part 4 Example 6 can be reproduced (baseline SPI 56 against the
tranquil target) and a user can define a bespoke target in the GUI's terms (centre,
spread, skew) without knowing skew-normal parameters.
*Hardest step today:* defining the target.

### D. Predict without a survey — Must
*Actor:* P2, P3, P1.
*Has:* recordings (measured or auralised) for a site with no survey, plus whatever
context a chosen model requires.
*Wants:* predicted perception from a named, versioned model; per-clip predictions and,
where the model supports it, a predicted distribution; a clear statement of what the
model was trained on and what it needs.
*Done when:* at least one published single-response predictor and the published
linear models run end to end from audio, and a third-party model can be added as a
plugin without changing the package. Distributional predictors are supported by the
interfaces but not required to ship [agreed].

### E. Evaluate a design against a brief — Must
*Actor:* P2, P3.
*Has:* a Soundscape Design Brief per Part 4 (ambition statement, target, KPI
threshold), a baseline assessment, and one or more scenarios, each supplying audio
(auralised or recorded at a comparable site) plus model-specific design parameters
[agreed].
*Wants:* baseline, scenario and target distributions on one circumplex; SPI and
shift for each scenario; a ranked comparison; the material for a Part 4 evaluation
report.
*Done when:* the full Part 4 Example 6 story (brief → baseline → intervention →
post-assessment → check against brief) can be executed, both with a post-intervention
survey and with predicted scenarios.

### F. Derive a reference target — Should
*Actor:* P1.
*Has:* a ranking of sites from an external measure.
*Wants:* an optimised target with provenance, reproducing the ranking.

### G. Cross-dataset research — Should
*Actor:* P1, P4.
*Has:* ISD, ARAUS, SATP, DeLTA and own data.
*Wants:* them in one harmonised structure that keeps item-level responses, scales,
languages, settings and dataset identity explicit, compatible with the MOSAIQ schema.

### H. Train and benchmark a predictor — Later
### I. Reproduce a paper — Must
*Actor:* P1, P5. *Done when:* the figures and tables of the 2022, 2024 (SPI) and 2021
papers regenerate from the ISD with one notebook each, within documented tolerance.

### J. Long-term monitoring — Later

### K. Complete the course — Must (added)
*Actor:* P5.
*Has:* a laptop (often Windows, often locked down) or a browser; no Python.
*Wants:* to run journeys A, C, E and a reduced B on provided and own data during a
taught session, and to keep using the tool afterwards.
*Done when:* a participant with no Python reaches a first circumplex plot inside the
first hour by the documented path, and every course exercise runs unchanged in the
supported environments.

### L. Extend the package — Must (added)
*Actor:* P6, P1.
*Wants:* to add a psychoacoustic metric, a dataset loader, a predictor or a plot type
by following a written recipe, with tests that tell them whether the science still
holds.
*Done when:* each of the four extension types has a recipe, an example, and a test
template, and a new contributor has done one in under a day.

---

## 5. Functional requirements

Each requirement is tagged with journeys and priority. "Shall" is a hard
requirement; "should" is desirable.

### 5.1 Data intake and project management

- **FR-1.1** (A, K; Must) Shall import survey data from CSV, Excel and REDCap exports.
- **FR-1.2** (A, K; Must) Shall define and document a canonical survey layout (columns,
  identifiers, coding) that the course teaches and that templates are provided for.
- **FR-1.3** (A; Must) Shall accept any column naming through an explicit mapping of
  columns to the standard constructs (the eight attributes, source dominance, overall
  quality, appropriateness, WHO-5, demographics, identifiers), with the response scale
  and language declared, and shall persist that mapping with the project.
- **FR-1.4** (A, B; Must) Shall represent the SSID hierarchy Location → Session → Group
  → Response and link recordings to groups, environmental logs to sessions and static
  context to locations; shall tolerate datasets that have only some levels.
- **FR-1.5** (A; Must) Shall apply documented quality rules (ISO coding checks,
  all-identical attribute responses, excessive missingness, age and consent flags,
  test and paper-input flags) producing an inclusion decision and reason per response,
  never silently dropping rows; rules shall be configurable.
- **FR-1.6** (B; Must) Shall index audio files without loading them, record their
  format (channels, rate, depth, duration) and link each to a group or clip.
- **FR-1.7** (A to E, K; Must) Shall save and reopen a project: data, mapping,
  settings, results and provenance in a documented on-disk format that the library,
  notebooks, CLI and GUI all read.
- **FR-1.8** (G; Should) Shall import from and export to MOSAIQ-conformant packages,
  preserving item-level responses separately from derived coordinates, explicit
  missingness (present, absent, not reported) and provenance.
- **FR-1.9** (A; Should) Should support ISO 12913-2 Method B data (continuous 1–5
  scales, source ranking, free words) in the data model and basic statistics; Method
  C is out of scope beyond storing transcripts as attachments.

### 5.2 Survey analysis per ISO/TS 12913-3

- **FR-2.1** (A; Must) Shall code Method A responses exactly as Table A.1 (direction
  per part) and shall reject or flag data that violates the declared scale.
- **FR-2.2** (A; Must) Shall compute per-item median and interquartile range (2024
  revision) and, as an option, range (2019 edition).
- **FR-2.3** (A; Must) Shall compute the circumplex projection in the general angle
  form (A.1 to A.4 of the revision) with ρ derived from the declared scale, shall
  reproduce the 2019 fixed formula exactly when standard angles are used, and shall
  record which angles and scale produced every coordinate.
- **FR-2.4** (A; Must) Shall ship the published language-specific adjusted angles and
  select them from the declared survey language; shall allow user-supplied angle sets
  with provenance. Deriving new angle sets from translation data is **Later**
  [assumed].
- **FR-2.5** (A; Must) Shall project per response, then aggregate by any grouping
  (location, session, demographic, time, user-defined); shall report mean, SD,
  quantiles and the proportion of responses in any quadrant, half-plane or
  user-defined region ("percent likely to perceive").
- **FR-2.6** (A; Must) Shall provide the Part 3 correlation requirements: Spearman
  with the tied-ranks formula for ordinal items, Pearson for interval data, with
  p-values; regression and ANOVA through the standard scientific Python stack rather
  than re-implemented.
- **FR-2.7** (A; Should) Should provide site clustering on perceptual and
  psychoacoustic data as named in the revision's general clause.
- **FR-2.8** (A; Should) Should compute Structural Summary Method metrics on the
  circumplex (amplitude, angle, elevation, fit) for a response or group.

### 5.3 Distributional representation

- **FR-3.1** (A, C, E; Must) Shall treat a set of projected responses as a bivariate
  distribution and provide: kernel density estimate with explicit bandwidth, decile
  contours, the contour enclosing a stated fraction of responses (default 50%,
  parameter not constant), marginal distributions, and overlays of several groups.
- **FR-3.2** (A; Must) Shall enforce or warn on the N < 30 rule: density plots are
  discouraged below 30 responses and scatter is offered instead.
- **FR-3.3** (C, E; Must) Shall compute shift vectors and overlap between two
  distributions (observed vs observed, observed vs predicted, either vs target).
- **FR-3.4** (C; Must) Shall represent a parametric target distribution (bivariate
  skew-normal in direct parameters), convert to and from centred parameters where
  admissible, validate positive-definiteness, sample it with a seed, and fit it to
  data.

### 5.4 Audio and psychoacoustic analysis

- **FR-4.1** (B; Must) Shall accept mono, two-channel and multichannel files
  [assumed: WAV and FLAC; ambisonic files handled as multichannel without decoding].
- **FR-4.2** (B; Must) Shall apply calibration from (a) a known level per channel,
  (b) a calibration tone recorded in a separate file or at the head of the file at a
  stated level, and shall record the calibration method and values in the results
  [assumed].
- **FR-4.3** (B; Should) Should apply a user-supplied equalisation filter to
  approximate a monaural measurement microphone, as Part 3 clause 7 requires, and
  record that it was applied [assumed: the filter is not shipped].
- **FR-4.4** (B; Must) Shall compute the Part 3 Table D.1 set: LAeq, LCeq, LAF5,
  LAF95; N5, Navg, Nrmc, N95, N5/N95 (ISO 532-1); S5, Savg, S95 (DIN 45692);
  tonality (ECMA-74); R10, R50; F10, F50; and the 2024 additions S50 and IACC.
- **FR-4.5** (B, D; Should) Should compute the research feature set used by the
  published models: ECMA-418-2 roughness, impulsiveness and fluctuation strength,
  psychoacoustic annoyance, LA10 − LA90, LCeq − LAeq, kurtosis, SEL, and the
  scikit-maad ecoacoustic indices. Proprietary metrics (relative approach, HMS
  tonality) are out of scope [assumed].
- **FR-4.6** (B; Must) Shall compute every metric per channel, shall derive the
  representative single value by the higher-of-channels rule with mean as an option,
  and shall support genuinely cross-channel metrics (IACC) as a distinct kind.
- **FR-4.7** (B; Must) Shall let the user choose statistics per metric (percentiles,
  mean, max, min, RMC, kurtosis, skew) and shall name output columns consistently
  with units recorded.
- **FR-4.8** (B; Must) Shall process folders of recordings in parallel, resume after
  interruption, cache per-file results, report progress, and save the complete
  settings alongside the results.
- **FR-4.9** (B; Must) Shall return results as one tidy table per recording (and per
  channel) that joins to survey data by group or clip identifier, and shall export to
  CSV and Excel.
- **FR-4.10** (B; Must) Shall validate selected metrics against reference signals or
  reference-implementation values with stated tolerances, and publish those
  tolerances.
- **FR-4.11** (B; Should) Should expose metric time series (not only statistics) for
  users who need them, with the frame rate recorded.

### 5.5 Predictive modelling

- **FR-5.1** (D, E; Must) Shall define a predictor interface that any model
  implements: declared inputs (audio and/or features, required and optional context,
  design parameters), declared outputs (per-clip PAQ items, ISO coordinates, sound
  sources, scene; optionally a distribution), and provenance (name, version, training
  data, setting, licence).
- **FR-5.2** (D; Must) Shall discover predictors installed as separate packages
  (plugins) and list them with their requirements.
- **FR-5.3** (D; Must) Shall ship the published linear models of Mitchell et al. 2021
  in deployable (unscaled) form, distinguishing known-location from
  fixed-effects-only prediction.
- **FR-5.4** (D; Must) Shall ship or fetch at least one validated single-response
  audio predictor of the SoundAQnet type, with its weights distributed under their
  licence [assumed: fetched on first use, not bundled].
- **FR-5.5** (D; Must) Shall never require perceptual (survey-derived) inputs for a
  deployable predictor, following the 2023 framework; shall degrade gracefully when
  optional inputs are missing.
- **FR-5.6** (D, E; Must) Shall label every prediction with the model that produced
  it and keep predictions separate from measured responses in all tables and exports
  (MOSAIQ prediction layer).
- **FR-5.7** (D; Should) Should provide the multilevel-model fitting workflow of the
  2021 paper (feature screening, train/test split by location, stepwise selection,
  MAE and location-mean R²) as reusable steps for researchers fitting their own
  models.
- **FR-5.8** (H; Later) Standard benchmark splits and evaluation protocols.

### 5.6 Soundscape Perception Indices

- **FR-6.1** (C, E; Must) Shall compute SPI = 100(1 − D) using the two-dimensional
  KS distance between a test distribution (observed or predicted) and a target, with
  explicit seed and sample size, and shall report the variability due to sampling.
- **FR-6.2** (C; Must) Shall support bespoke targets defined either in direct
  skew-normal parameters or in plain terms (centre, spread, skew direction, quadrant),
  and shall show the target as a distribution before scoring.
- **FR-6.3** (C; Must) Shall ship a small, versioned library of named targets with
  provenance (at minimum the three demonstration targets of the 2024 paper and the
  Part 4 Example 6 tranquil target), clearly labelled as demonstration or reference.
- **FR-6.4** (C, E; Must) Shall score many sites or scenarios against one target and
  one site against many targets, producing rankings.
- **FR-6.5** (F; Should) Should derive a reference target from an a-priori ranking by
  multi-objective optimisation as published, recording the ranking, objectives,
  optimiser settings and Pareto selection.

### 5.7 Design and intervention (ISO/CD TS 12913-4)

- **FR-7.1** (E; Must) Shall represent a Soundscape Design Brief: ambition statement,
  target distribution, KPI thresholds (for example SPI > 70), assessment method, and
  the scenarios under consideration.
- **FR-7.2** (E; Must) Shall represent a scenario as a named set of inputs (audio,
  features, design parameters, or a post-intervention survey) and its resulting
  perception distribution, with the model or survey that produced it.
- **FR-7.3** (E; Must) Shall compare baseline, scenarios and target: SPI, shift
  vectors, quadrant proportions, overlaid contours; shall rank scenarios and flag
  which meet the KPI.
- **FR-7.4** (E; Should) Should classify interventions with the Part 4 Annex A levels
  (source, path/infrastructure, integral/design, receiver) as metadata for
  reporting.
- **FR-7.5** (E; Should) Should generate the content of a Part 4 evaluation report:
  brief, baseline, intervention description, post-assessment, check against brief.

### 5.8 Visualisation

- **FR-8.1** (A, C, E; Must) Shall produce the three Part 3 visualisations (scatter;
  KDE with decile contours or continuous colourmap; simplified 50th-percentile
  contour) on a correctly labelled circumplex, plus overlays by group, marginals,
  shift arrows, target overlays and SPI annotation.
- **FR-8.2** (A; Must) Shall produce item-level plots: stacked Likert bars and radar
  plots of the eight attributes.
- **FR-8.3** (A; Must) Shall produce publication-quality output (vector and raster)
  with consistent, colour-blind-safe defaults, and shall let users customise through
  the conventions of the underlying plotting library rather than a bespoke styling
  system.
- **FR-8.4** (K, GUI; Must) Shall separate the computation of plot geometry (density
  grids, contour polygons, proportions) from drawing, so that a GUI or an interactive
  backend can render the same results.
- **FR-8.5** (B; Should) Should plot metric time series and per-channel comparisons
  for audio results.

### 5.9 Reporting and export

- **FR-9.1** (A to E; Must) Shall export every table to CSV and Excel and every figure
  to PNG, SVG and PDF.
- **FR-9.2** (A, C, E; Should) Should assemble a formatted report (HTML and PDF or
  Word) from a project: data summary, Part 3 statistics, figures, SPI results,
  design-brief comparison, with the settings appendix [assumed].
- **FR-9.3** (I; Must) Shall provide a notebook per supported paper that regenerates
  its figures and tables from the public dataset.

### 5.10 Datasets

- **FR-10.1** (A, I, K; Must) Shall load the International Soundscape Database from
  its public record with integrity checks, licence display and local caching, and
  shall bundle a small subset for teaching and tests.
- **FR-10.2** (G; Should) Should load SATP, ARAUS and DeLTA with their mappings to
  the standard constructs, scales, languages and collection settings declared, never
  rescaling or merging across frameworks silently.
- **FR-10.3** (G; Should) Should describe every loaded dataset with the catalogue
  fields of the MOSAIQ dataset layer.

### 5.11 Reproducibility and provenance

- **FR-11.1** (all; Must) Every result shall record the settings that produced it
  (angles, scale, bandwidth, contour level, seed, metric settings, calibration, model
  version) and the package version.
- **FR-11.2** (all; Must) The same project and settings shall produce the same
  numbers on any supported platform, within documented floating-point tolerance.
- **FR-11.3** (all; Must) Settings shall be shareable as human-readable files and
  attachable to a paper or report.
- **FR-11.4** (B; Must) Audio processing runs shall write a manifest with file
  hashes, versions and timings.

### 5.12 Teaching and documentation

- **FR-12.1** (K; Must) Shall provide a documented, tested path from nothing to a
  first circumplex plot in under an hour for a non-programmer, in each supported
  environment (local install, hosted notebook).
- **FR-12.2** (K; Must) Shall provide course-aligned tutorials for journeys A, B, C
  and E on the bundled dataset, runnable without modification.
- **FR-12.3** (all; Must) Documentation shall follow a single structure (tutorials,
  how-to, reference, explanation) with one source of truth per topic; the explanation
  section shall carry the science and standards references so that notebooks do not
  repeat them.
- **FR-12.4** (all; Must) Every public function shall have a reference entry with
  the standard clause or paper it implements.

### 5.13 Extensibility and contribution

- **FR-13.1** (L; Must) Shall provide recipes, examples and test templates for adding
  a metric, a dataset loader, a predictor and a plot type.
- **FR-13.2** (L; Must) Scientific correctness shall be protected by tests that a
  contributor can run locally: literature fixtures, reference values and regression
  baselines.
- **FR-13.3** (L; Must) Shall support third-party extensions as separate installable
  packages (predictors first; metrics and datasets as well) so that research groups
  can publish methods without modifying the core.
- **FR-13.4** (L; Should) Should document a deprecation and versioning policy so
  course material and papers can pin a version safely.

### 5.14 Readiness for a graphical application

These are requirements on the core so that the GUI (a later deliverable) is
possible without a rewrite.

- **FR-14.1** (Must) All functionality shall be available without a display, a
  notebook or user interaction.
- **FR-14.2** (Must) Every input, setting and result shall be serialisable to a
  documented text format.
- **FR-14.3** (Must) Long-running operations shall report progress and be
  cancellable.
- **FR-14.4** (Must) The project format (FR-1.7) shall be the unit of exchange
  between library, CLI and GUI.
- **FR-14.5** (Must) A command-line interface shall cover journeys A, B, C and E so
  that each can be run without writing Python.
- **FR-14.6** (Must) Audio shall be processable locally; nothing shall require
  uploading audio to a service [assumed, see NFR-5].

---

## 6. Non-functional requirements

- **NFR-1 Installation.** One standard Python package install with no compiler, no R
  and no system libraries beyond wheels, on Windows, macOS and Linux [agreed: standard
  procedures, not necessarily one line]. Heavy optional capabilities (deep models,
  optimisation) install as extras or plugins. A hosted-notebook path must work with
  a single install cell.
- **NFR-2 Platforms.** Current Python versions per the scientific-Python support
  window; Windows is a first-class target because of the consultant and student base.
- **NFR-3 Performance.** 1,000 thirty-second binaural recordings through the Table
  D.1 set overnight on a four-core laptop; survey analysis of 10,000 responses
  interactive (seconds); SPI of one site in under a second at default sample size
  [assumed targets].
- **NFR-4 Reproducibility.** See FR-11; additionally the project shall publish
  regression baselines and tolerances and treat any numeric drift as a breaking
  change.
- **NFR-5 Privacy and data handling.** Recordings may contain speech and client data;
  the package shall not transmit data anywhere except explicit dataset downloads and
  explicit model-weight downloads, both of which shall state what is fetched
  [assumed].
- **NFR-6 Standards traceability.** Each implemented procedure shall cite the clause
  or paper; where editions differ, the edition in use shall be selectable and
  recorded.
- **NFR-7 Licensing.** BSD-3-Clause; no copyleft dependencies in the core; model
  weights distributed only under licences that permit it [assumed].
- **NFR-8 Quality.** Automated tests on all supported platforms; scientific fixtures
  from the literature; reference comparisons for audio; documentation built and
  executed in CI.
- **NFR-9 Maintainability.** Maintainable by a small volunteer team of mixed
  experience: strict public-API boundary, small modules, decision records, no
  bespoke frameworks where an ecosystem library exists.
- **NFR-10 Accessibility and internationalisation.** Colour-blind-safe defaults;
  attribute labels available in the supported survey languages; interface language
  is English for this release [assumed].
- **NFR-11 Longevity.** Citable releases with DOIs; a software paper after the first
  stable release.

---

## 7. Out of scope for this release

- Field data capture (survey apps, REDCap configuration), beyond importing exports.
- Ambisonic decoding, spatial rendering, auralisation; VR playback.
- Method C qualitative analysis (grounded theory coding).
- Deriving new translation angle sets (structural equation modelling) [assumed Later].
- Real-time or sensor-network monitoring (journey J).
- Benchmark splits and leaderboards (journey H).
- Soundscape captioning and any LLM dependency.
- The GUI itself (a separate product built on this core).

---

## 8. Compatibility and migration

[assumed, pending decision] This is a clean rebuild released as a new major version
under the existing name. Existing scripts are not expected to run unchanged; a
migration guide maps old functions to new workflows, and the course material is
rewritten for the new version. The old version remains installable and pinned.

---

## 9. Release scoping

| Milestone | Journeys complete | Headline |
|---|---|---|
| M1 | A, I (2022 paper), K (hosted notebook) | Survey analysis per the Part 3 revision with adjusted angles; no R |
| M2 | C, I (SPI paper) | SPI with targets, in Python |
| M3 | B | Audio pipeline with calibration, Table D.1, parallel and resumable |
| M4 | D, E, I (2021 paper) | Predictors, plugins, design-brief evaluation per Part 4 |
| M5 | L, F, G | Extension recipes, reference-target search, harmonised datasets |
| GUI | K (desktop or web) | Built on the M4 core |

---

## 10. Assumptions awaiting confirmation

1. **Course workflow and environment** (K): hosted notebooks remain the primary
   teaching path until the GUI exists; local install is the secondary path.
2. **GUI deployment**: both local desktop and web must remain possible; audio is
   processed locally in either case (NFR-5, FR-14.6).
3. **GUI outputs**: figures, tables and a saved project are Must; a formatted report
   is Should (FR-9.2).
4. **Audio channels**: mono, two-channel and multichannel accepted; ambisonic
   decoding out of scope (FR-4.1).
5. **Calibration**: known per-channel level and calibration-tone file or segment;
   equalisation filter user-supplied (FR-4.2, FR-4.3).
6. **Metric set**: Table D.1 is Must; ECMA-418-2 and the lockdown research set are
   Should; proprietary metrics are out (FR-4.4, FR-4.5). Reference values from
   ArtemiS are wanted if available.
7. **Scale**: hundreds of recordings per consultancy project, thousands for research
   (NFR-3).
8. **Translation**: apply published angles now; derive new angles later (FR-2.4).
9. **Inputs**: CSV, Excel, REDCap; a canonical layout taught in the course (FR-1.1,
   FR-1.2).
10. **Shipped predictors**: the 2021 linear models and one SoundAQnet-type model with
    weights fetched on first use (FR-5.3, FR-5.4).
11. **Clean break**, same name, new major version, migration guide (§8).
12. **Maintainers**: small core team plus volunteers; no assumption of funded
    developer time beyond the core team.
13. **Licence**: BSD-3, no copyleft dependencies in core (NFR-7).
14. **Method B**: supported in the data model and basic statistics only (FR-1.9).
