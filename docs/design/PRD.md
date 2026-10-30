# Product Requirements: an open-source soundscape assessment, analysis and design package

Status: **draft 0.3 for review** (2026-10-30). This document states what the package
must do and for whom. It deliberately says nothing about how it is built; the
architecture plan is a separate document and must be checked against this one.

Decisions taken with the product owner are marked **[agreed]**. The few points still
open are marked **[open]** and listed in §10.

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
- New developers contribute, because the rebuild made contribution feasible; today
  the package is maintained by one unfunded developer as a side project [agreed].

---

## 2. Scientific and standards baseline

The package implements, and must stay faithful to, the following. The Mitchell papers
and chapters are the canonical reference for the methods; where a standard and a
Mitchell source differ, the Mitchell source is the intended reading and the standard's
edition is kept as a compatibility option. Only the circumplex portions of ISO/TS
12913-3 are in scope; Method B and Method C analysis are not [agreed].

| Source | What it fixes |
|---|---|
| ISO/TS 12913-2:2018 | Method A questionnaire (8 PAQ items, 4 source-dominance items, overall quality, appropriateness); binaural recording and equalisation. Methods B and C are out of scope |
| ISO/TS 12913-3:2019 | Scale coding (Table A.1), fixed 45° projection divided by (4+√32), median and range, Spearman with ties, binaural metric set (Table D.1), higher-of-channels rule |
| ISO/AWI TS 12913-3 (2024 revision draft) | General angle projection (A.1 to A.4), per-response projection, mean and SD per group, interquartile range for ordinal items, three named visualisations (scatter; KDE with decile contours; 50th-percentile contour), N < 30 rule, N_rmc, S_50 and IACC added, placeholder section on translated attributes |
| Mitchell 2025, "A robust approach to soundscape circumplex coordinate projections" (blog, adopted into the Part 3 revision) and Mitchell, Oberman, Aletta & Kang 2026, *Soundscape Across Languages* ch. 4 | **Canonical projection**: centre each score at the scale midpoint μ = (min+max)/2, scale by the half-range ρ = (max−min)/2, project with cos θ_i and sin θ_i, and normalise by Σ\|cos θ_i\| and Σ\|sin θ_i\| so that coordinates lie in [−1, 1] for any angle set and neutral responses map to the origin. Reduces to the 2019 formula for 45° angles. Also: the four-step validation (circular order, SEM fit, SSM location, congruence), confidence tiers (High/Medium/Low), the SATP v1.5 adjusted angles for 13 High-confidence languages, grand-mean ipsatisation per participant, and the recommendation not to project Low-tier languages |
| ISO/CD TS 12913-4 (2025) | Design process: communication framework, Soundscape Design Brief with ambition statement and KPIs, pre- and post-intervention assessment, evaluation report; SPI named as a quantitative KPI; Example 6 (baseline SPI 56 against a tranquil target, success at SPI > 70) |
| SSID Protocol (Mitchell et al. 2020) | Location → Session → Group → Response hierarchy; one 30-s binaural clip per group; item coding; environmental and session metadata; id conventions |
| Mitchell, Aletta & Kang 2022 | Distributional representation; projection before aggregation; "percent likely to perceive"; 50% contour as provisional |
| Mitchell & Aletta 2023; Aletta et al. 2024 | The earlier language-specific adjusted-angle set and generalised projection, superseded by the 2026 chapter but still shipped as a selectable version |
| Torresin et al. 2020 (indoor soundscape circumplex) | Evidence that other attribute sets and circumplexes exist; the tooling must not be tied to the eight outdoor English attributes |
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
*Done when:* a default predictor ships and runs end to end from audio, a large deep
model (SoundAQnet) can be side-loaded as an optional plugin, and a third-party model
can be added as a plugin without changing the package. Both single-response and
distributional predictors are supported by the interfaces; which ships first depends
on suitability (see FR-5.3) [agreed].

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
*Actor:* P5 (Institute of Acoustics Certificate Course in Soundscape Assessment).
*Has:* a laptop and the course notebooks; today the course provides notebooks that
install the package and walk through a tutorial and an exercise on ISD data, after
which students collect their own data in a workshop and analyse it [agreed]. Hosting
of notebooks is handled by the course, not by this package [agreed].
*Wants:* to run journeys A, C and E on ISD data and then on their own survey export
during a taught session, with a reduced B where audio was collected; and to produce
the assessed report (at most eight A4 sides) whose marking rewards ISO 12913-2
reporting compliance, Part 3 analysis rigour, APA-style statistical reporting,
visualisation quality and interpretation.
*Acceptance cases, from the current assessment briefs (indicative, the package will go
beyond them):*
1. Two-site comparison (Regent's Park Broadwalk vs Japanese Garden): individual
   responses plotted on the circumplex per site; which site is more likely calm or
   tranquil; correlations between psychoacoustic metrics and each of the eight scales.
2. Cross-country ISD analysis: associations between calmness, vibrancy, pleasantness
   and WHO-5 well-being, compared across countries and cultures, with charts.
3. City-wide profiles: sound-source dominance profile and attribute scores per
   location, and a prioritisation of locations for intervention.
*Done when:* a participant reaches a first circumplex plot inside the first hour by
the documented local path, every course exercise runs unchanged, and each acceptance
case can be completed with the package's own outputs.

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

- **FR-1.1** (A, K; Must) Shall import survey data from CSV and Excel, which is how
  REDCap, Qualtrics and hand-built surveys arrive in practice [agreed].
- **FR-1.2** (A, K; Must) Shall define and document a canonical survey layout (columns,
  identifiers, coding) that the course teaches, with downloadable templates [agreed].
- **FR-1.3** (A; Must) Shall accept any column naming and layout through an explicit
  mapping of columns to the constructs of the declared instrument (the circumplex
  attributes of the chosen circumplex, source dominance, overall quality,
  appropriateness, WHO-5, demographics, identifiers), with the response scale and
  language declared, and shall persist that mapping.
- **FR-1.3a** (A, K; Must) Shall make building that mapping easy for a non-programmer:
  suggest matches from column names and known export schemas (REDCap/SSID codebook,
  Qualtrics, the previous package's layout), let the user confirm or correct them,
  and save the mapping for reuse on the next export from the same instrument
  [agreed].
- **FR-1.4** (A, B; Must) Shall represent the SSID hierarchy Location → Session → Group
  → Response and link recordings to groups, environmental logs to sessions and static
  context to locations; shall tolerate datasets that have only some levels.
- **FR-1.5** (A; Must) Shall apply documented quality rules (ISO coding checks,
  all-identical attribute responses, excessive missingness, age and consent flags,
  test and paper-input flags) producing an inclusion decision and reason per response,
  never silently dropping rows; rules shall be configurable.
- **FR-1.6** (B; Must) Shall index audio files without loading them, record their
  format (channels, rate, depth, duration) and link each to a group or clip.
- **FR-1.7** (A to E, K; Should) Should save and reopen a project: data, mapping,
  settings, results and provenance in a documented on-disk format that the library,
  notebooks, CLI and GUI all read. Settings and mappings alone shall be saveable from
  the start (FR-11.3); whole-project save and sharing is a later GUI concern
  [agreed].
- **FR-1.8** (G; Should) Shall import from and export to MOSAIQ-conformant packages,
  preserving item-level responses separately from derived coordinates, explicit
  missingness (present, absent, not reported) and provenance.
- **FR-1.9** (A; Out of scope) ISO 12913-2 Method B and Method C data are not
  supported [agreed]. Non-circumplex items that accompany Method A surveys (source
  dominance, overall quality, appropriateness, WHO-5, demographics, free text) are
  carried as columns and summarised, but no Method B or C analysis is provided.

### 5.2 Survey analysis: the circumplex

- **FR-2.0** (A, L; Must) Shall define a circumplex as an instrument specification,
  not as a fixed list of eight English words: a set of attributes identified by
  generic position identifiers (PAQ1…PAQn, ordered counter-clockwise from the
  positive x-axis as in the SATP convention), each with an angle, and with labels per
  language or instrument; two named axes; and a response scale. The outdoor ISO
  12913 circumplex is the default instance. Other circumplexes, for example the
  indoor soundscape circumplex of Torresin et al. 2020, shall be usable with every
  analysis, plot and score in the package without special-casing [agreed].
- **FR-2.1** (A; Must) Shall code Method A responses exactly as Table A.1 (direction
  per part) and shall reject or flag data that violates the declared scale.
- **FR-2.2** (A; Must) Shall compute per-item median and interquartile range (2024
  revision) and, as an option, range (2019 edition).
- **FR-2.3** (A; Must) Shall compute the circumplex projection with the canonical
  robust formulation (Mitchell 2025; chapter 4, 2026): scores centred at the scale
  midpoint and divided by the half-range, projected with cos θ_i and sin θ_i, and
  normalised by Σ|cos θ_i| and Σ|sin θ_i|, so that coordinates lie within [−1, 1] for
  any angle set and a neutral response maps to the origin. Shall reproduce the 2019
  fixed formula exactly when the standard 45° angles are used, shall offer the 2019
  and the Aletta et al. 2024 formulations as selectable legacy variants for
  reproduction of older work, and shall record which formulation, angles and scale
  produced every coordinate [agreed].
- **FR-2.4** (A; Must) Shall ship the published language-specific adjusted-angle sets
  as versioned data: the SATP v1.5 set from the 2026 chapter (13 High-confidence
  languages: arb, deu, ell, eng, hrv, ind, ita, nld, pol, spa, swe, tur, yue) as the
  default, and the Aletta et al. 2024 set as a selectable earlier version; shall
  select the set from the declared survey language, allow the version to be chosen
  per language, and accept user-supplied angle sets with provenance [agreed].
- **FR-2.4a** (A; Must) Shall carry each language's confidence tier with its angles:
  High-tier languages project silently; Medium-tier languages project with a
  warning; Low-tier languages (alb, vie, zsm in v1.5) are refused by default with
  an explicit override, following the chapter's recommendation. Deriving new angle
  sets from translation data (circumplex SEM fitting) is **out of scope** [agreed].
- **FR-2.4b** (A, G; Should) Should offer grand-mean ipsatisation per participant
  (subtract each participant's mean over all attributes and stimuli) as an explicit
  pre-processing option for listening-experiment data, recorded in provenance; it is
  not applied by default to field surveys.
- **FR-2.5** (A; Must) Shall project per response, then aggregate by any grouping
  (location, session, demographic, time, user-defined); shall report mean, SD,
  quantiles and the proportion of responses in any quadrant, half-plane or
  user-defined region ("percent likely to perceive").
- **FR-2.6** (A; Must) Shall provide the Part 3 correlation requirements: Spearman
  with the tied-ranks formula for ordinal items, Pearson for interval data, with
  p-values, including correlations between each psychoacoustic metric and each
  attribute (course acceptance case 1); regression and ANOVA through the standard
  scientific Python stack rather than re-implemented.
- **FR-2.6a** (A, K; Must) Shall provide comparisons between sites or groups with
  the appropriate tests for the measurement level (for example Mann–Whitney or
  Kruskal–Wallis for ordinal items, t-test or ANOVA for coordinates) via the
  scientific Python stack, reporting statistic, degrees of freedom or N, p-value and
  effect size.
- **FR-2.6b** (A, K; Must) Shall present statistical results in a report-ready form
  (APA-style sentence and table) so that students and consultants can place them in
  a report without transcription.
- **FR-2.6c** (A, K; Must) Shall score the WHO-5 well-being index and summarise
  source-dominance profiles per location, since both appear in standard assessment
  reports (course acceptance cases 2 and 3).
- **FR-2.7** (A; Should) Should provide site clustering on perceptual and
  psychoacoustic data as named in the revision's general clause.
- **FR-2.8** (A; Later) Structural Summary Method metrics on the circumplex
  (amplitude, angle, elevation, fit) for a response or group [agreed].

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

- **FR-4.0** (B; Must) Shall not re-implement psychoacoustic, ecoacoustic or
  environmental-acoustics algorithms. Metrics are provided by wrapping existing open
  libraries (MoSQITo, pySQAT and others catalogued by the LISQM list; scikit-maad for
  ecoacoustics), and the metric set on offer is whatever those libraries provide,
  validated and exposed consistently [agreed]. The package's own contribution is
  calibration, batch orchestration, channel rules, statistics and consistent
  outputs.
- **FR-4.1** (B; Must) Shall accept mono (sound-level-meter or general) and
  two-channel (binaural or general stereo) recordings. Ambisonic and other
  multichannel processing is out of scope [agreed].
- **FR-4.1a** (B; Must) Shall handle long-duration recordings (long-term sound-level
  meter files of hours) by computing metrics over user-defined time windows as well
  as whole-file, since consultants bring these alongside 30-s binaural clips
  [agreed].
- **FR-4.2** (B; Must) Shall apply calibration from (a) a known level per channel, or
  (b) a recorded calibrator signal (separate calibration file or a segment at a
  stated level, following the IEC 61672 field-calibration procedure as practised in
  open sound-level-meter tools), and shall record the calibration method and values
  in the results [agreed].
- **FR-4.3** (B; Should) Should apply a user-supplied equalisation filter to
  approximate a monaural measurement microphone, as Part 3 clause 7 requires, and
  record that it was applied. The filter itself is not shipped [agreed].
- **FR-4.4** (B; Must) Shall offer the Part 3 Table D.1 set to the extent the wrapped
  libraries provide it: LAeq, LCeq, LAF5, LAF95; N5, Navg, Nrmc, N95, N5/N95
  (ISO 532-1); S5, Savg, S95 (DIN 45692); tonality; R10, R50; F10, F50; and the
  2024 additions S50 and IACC. Level metrics and IACC, which are elementary, may be
  implemented directly. Gaps in library coverage shall be documented, not filled
  with bespoke implementations.
- **FR-4.5** (B, D; Should) Should offer the research feature set used by the
  published models where libraries provide it: ECMA-418-2 loudness, roughness and
  tonality, fluctuation strength, psychoacoustic annoyance, LA10 − LA90,
  LCeq − LAeq, kurtosis, SEL, and the scikit-maad ecoacoustic indices. Proprietary
  metrics (relative approach, HMS tonality) are out of scope.
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
- **FR-4.10** (B; Must) Shall validate selected metrics against reference signals and
  the ArtemiS reference values the group holds, with stated tolerances, and publish
  those tolerances [agreed: reference values exist].
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
- **FR-5.3** (D; Must) Shall ship one default predictor, chosen after a suitability
  check from: the "Probably Pleasant" ARAUS model (Ooi et al. 2022), which predicts
  a pleasantness distribution only, and the group-level probabilistic model (Tong et
  al. 2026), which predicts a distribution in the circumplex. The suitability check
  shall consider: whether weights and licence permit redistribution, dependency
  weight, and whether pleasantness-only output is acceptable for SPI (the 2023
  framework paper argues it is not) [agreed, choice open].
- **FR-5.3a** (D, I; Should) Should ship the published linear models of Mitchell et
  al. 2021 in deployable (unscaled) form, distinguishing known-location from
  fixed-effects-only prediction; they are cheap and needed to reproduce that paper.
- **FR-5.4** (D; Must) Shall support side-loading large deep models as optional
  plugins, with SoundAQnet (weights and code public on the SoundSCaper repository)
  as the first, its weights fetched on first use under their licence rather than
  bundled [agreed].
- **FR-5.5** (D; Must) Shall never require perceptual (survey-derived) inputs for a
  deployable predictor, following the 2023 framework; shall degrade gracefully when
  optional inputs are missing.
- **FR-5.6** (D, E; Must) Shall label every prediction with the model that produced
  it and keep predictions separate from measured responses in all tables and exports
  (MOSAIQ prediction layer).
- **FR-5.7** (D; Later) The multilevel-model fitting workflow of the 2021 paper
  (feature screening, train/test split by location, stepwise selection, MAE and
  location-mean R²) as reusable steps [agreed].
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
- **FR-9.2** (A, C, E; Later) A formatted report is not required; the GUI and
  library produce tables and figures that users place in their own reports
  [agreed]. The *content* of a Part 4 evaluation (FR-7.5) is still assembled as
  tables and figures.
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
  first circumplex plot in under an hour for a non-programmer using a local install
  and local notebooks. Hosted-notebook provision is outside the package [agreed].
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
- **FR-14.6** (Must) Audio and prediction shall be processable locally; nothing
  shall require uploading audio to a service [agreed, see NFR-5].
- **FR-14.7** (Must) The core shall support two GUI tiers without change: a web
  application covering survey analysis, plotting and SPI, and a desktop application
  adding psychoacoustic analysis and prediction. Neither tier's choice shall shape
  the core beyond the headless and serialisation requirements above [agreed]. The
  GUI follows the conventions of technical analysis software: it drives the
  package's analysis functions and exports CSV/Excel tables and figures.

---

## 6. Non-functional requirements

- **NFR-1 Installation.** One standard Python package install with no compiler, no R
  and no system libraries beyond wheels, on Windows, macOS and Linux [agreed: standard
  procedures, not necessarily one line]. Heavy optional capabilities (deep models,
  optimisation) install as extras or plugins. A hosted-notebook path must work with
  a single install cell.
- **NFR-2 Platforms.** Current Python versions per the scientific-Python support
  window; Windows is a first-class target because of the consultant and student base.
- **NFR-3 Performance.** Consultants bring tens to hundreds of binaural clips or
  long-term sound-level-meter recordings per project; researchers bring thousands of
  clips across surveys [agreed]. Quality of design and implementation comes first;
  no optimisation work is undertaken at this stage beyond what the batch design
  gives for free (parallel files, resumable runs, caching) [agreed]. Indicative,
  non-binding expectations: survey analysis of 10,000 responses stays interactive;
  SPI for one site is near-instant; audio throughput is whatever the wrapped
  libraries deliver, run in parallel.
- **NFR-4 Reproducibility.** See FR-11; additionally the project shall publish
  regression baselines and tolerances and treat any numeric drift as a breaking
  change.
- **NFR-5 Privacy and data handling.** Recordings may contain speech and client data;
  the package shall not transmit data anywhere except explicit dataset downloads and
  explicit model-weight downloads, both of which shall state what is fetched
  [agreed].
- **NFR-6 Standards traceability.** Each implemented procedure shall cite the clause
  or paper; where editions differ, the edition in use shall be selectable and
  recorded.
- **NFR-7 Licensing.** The scientific core stays open source under a permissive
  licence (BSD-3 today; MIT or Apache-2.0 acceptable) and must remain compatible
  with every dependency's licence. An open-core model is a possible future, with
  the Python package open and a GUI licensed commercially, so the core shall avoid
  copyleft (GPL) dependencies that would constrain that, and shall record each
  wrapped audio library's licence as part of adopting it. Model weights are
  redistributed only under licences that permit it [agreed].
- **NFR-8 Quality.** Automated tests on all supported platforms; scientific fixtures
  from the literature; reference comparisons for audio; documentation built and
  executed in CI.
- **NFR-9 Maintainability.** Today one unfunded developer maintains the package as a
  side project; the rebuild must lower the barrier for new contributors [agreed].
  Hence: strict public-API boundary, small modules, decision records, no bespoke
  frameworks where an ecosystem library exists, and no re-implementation of
  algorithms that exist in maintained libraries. Scope must respect this: the Must
  list is sequenced so that each milestone is independently releasable and useful
  (§9).
- **NFR-10 Accessibility and internationalisation.** Colour-blind-safe defaults;
  attribute labels available in the supported survey languages; interface language
  is English for this release [agreed].
- **NFR-11 Longevity.** Citable releases with DOIs; a software paper after the first
  stable release.

---

## 7. Out of scope for this release

- Field data capture (survey apps, REDCap configuration), beyond importing exports.
- Ambisonic and any multichannel (more than two channels) processing, spatial
  rendering, auralisation, VR playback [agreed].
- ISO 12913-2 Method B and Method C data and analysis [agreed].
- Deriving new translation angle sets (circumplex SEM fitting) [agreed].
- Re-implementing psychoacoustic or ecoacoustic algorithms [agreed].
- Formatted report generation [agreed].
- Backwards compatibility with the 0.x package in any form [agreed].
- Real-time or sensor-network monitoring (journey J).
- Benchmark splits and leaderboards (journey H).
- Soundscape captioning and any LLM dependency.
- The GUI itself (a separate product built on this core).

---

## 8. Compatibility and migration

[agreed] This is a clean rebuild released as `soundscapy` 1.0 (the current package is
0.x). It is treated as entirely new and separate code: nothing in the 0.x
architecture, API or naming constrains it, no compatibility shims or migration guide
are built, and no time is spent on backwards compatibility. Adapter functions with
deprecation warnings may be added by the maintainer after the rebuild, outside this
work. Until release, the API and any implemented component may change freely to
meet the requirements; nothing implemented is treated as fixed if it is not working.
The 0.x releases remain installable and pinned for existing scripts and course
material.

---

## 9. Release scoping

| Milestone | Journeys complete | Headline |
|---|---|---|
| M1 | A, I (2022 paper), K (hosted notebook) | Survey analysis per the Part 3 revision with adjusted angles; no R |
| M2 | C, I (SPI paper) | SPI with targets, in Python |
| M3 | B | Audio pipeline with calibration, Table D.1, parallel and resumable |
| M4 | D, E, I (2021 paper) | Predictors, plugins, design-brief evaluation per Part 4 |
| M5 | L, F, G | Extension recipes, reference-target search, harmonised datasets |
| GUI | K (web tier: A, C, E; desktop tier adds B, D) | Built on the M4 core |

Given one part-time maintainer, each milestone is scoped to be releasable and
useful on its own, and M1 to M2 together constitute a usable 1.0 candidate for the
course even if M3 to M5 slip.

---

## 10. Decisions recorded and points still open

All points raised in review have been resolved and are tagged **[agreed]** in the
text: canonical robust projection (FR-2.3); generalised circumplex beyond the eight
outdoor English attributes (FR-2.0); two shipped angle-set versions with the 2026
SATP v1.5 set as default and confidence tiers enforced (FR-2.4, FR-2.4a); Methods B
and C out (FR-1.9, §7); FR-2.8 and FR-5.7 moved to Later; local notebooks only
(FR-12.1); English-only interface (NFR-10); equalisation filter user-supplied
(FR-4.3); performance targets relaxed in favour of design quality (NFR-3); default
predictor to be chosen after a suitability check (FR-5.3); MOSAIQ field dictionary
treated as normative once stable (FR-1.8, FR-10.3). Course acceptance cases are
recorded under journey K.

Remaining open items are implementation-stage questions, not requirements:

1. Which of the two candidate probabilistic predictors becomes the default (FR-5.3).
2. Which wrapped libraries cover which Table D.1 metrics, and therefore what the 1.0
   metric list is (FR-4.0, FR-4.4).
