# Review of PRD draft 0.3

Date: 2026-10-30. Outcome: PRD draft 0.4 (`design/PRD.md`). Historical record: the
items it left open (copyleft, governance, SPI uncertainty) were closed in PRD 0.6, and
its NFR numbering was changed by 0.5.

## Method

Three independent reads, compared afterwards so that none anchored the others:

- **Critical review.** A fresh reviewer checked the PRD against the owner's original
  brief for internal consistency, testability, gaps, scope versus capacity and
  scientific fidelity.
- **Builder check.** A fresh reviewer, given only the PRD and the brief, listed the
  questions they would still need answered before they could plan the build, ranked
  by how much a wrong guess would cost. This is the superpowers "builder check".
- **Lead review.** A full read by the session lead, written down before the other two
  reported.

Findings raised by two or three reads are marked **(×2)** or **(×3)**. Two decisions
taken after 0.3 was written were checked for their consequences: the Python 3.11
floor and in-browser (Pyodide) marimo tutorials.

## Verdict

The PRD is a sound basis for planning survey analysis and SPI (journeys A, C and I for
the 2022 and 2024 papers). It is not yet plannable for prediction and design (D, E,
and I for the 2021 paper), nor for the release sequence. Several Must items rest on
Should, Later or open items. The in-browser tutorial decision adds a constraint on the
core that the PRD did not state. None of this needs the PRD restructured. It needs the
owner decisions in PRD §10.1, most of which can be settled in one sitting.

## What changed in 0.4

- **Editorial fixes, applied directly.** Hosted-versus-local contradictions; priority
  definitions stated once; Shall/Should wording; FR-8.2's "eight attributes";
  targets tied to their circumplex; the Python floor; a risks section (§11); a
  glossary; permission under NFR-9 to implement what no maintained library provides.
- **Proposals, written in place and tagged [proposed].** Each changes scope or an
  [agreed] item, so each needs owner sign-off. All are listed in PRD §10.1.
- **New open questions, tagged [open].** Places where only the owner can answer.

## Findings, ranked

### Blocking: settle before planning

1. **The in-browser tutorials constrain the core (×3).** Tutorials now run under
   Pyodide, so every dependency on the survey, distribution, SPI and plotting path
   needs a pure-Python or Pyodide wheel. This decides the package split, the
   dependency list and whether the web GUI tier can run client-side. The audio and
   deep-model stacks will not run in the browser, so B and D tutorials cannot be
   in-browser. *Proposed:* NFR-1a and revised FR-12.2.
2. **Predictions do not yet become distributions (×3).** SPI, shift vectors and KPI
   checks score distributions, but a single-response predictor yields one point per
   clip. The 2023 framework paper argues distributions are required. If the default
   predictor turns out to be pleasantness-only, 2-D SPI on predicted scenarios is
   impossible. *Proposed:* FR-5.9, a pooling rule with refusal otherwise.
3. **Prediction and design Musts depend on open, Should and Later items (×3).**
   Journey D needs a default predictor that may not pass its suitability check.
   Journey E needs D for its predicted path. Reproducing the 2021 paper needs fitting
   (Later) and features computed in ArtemiS with metrics that are out of scope.
   *Proposed:* linear models as the fallback default; split E into E1 (survey path,
   M2) and E2 (predicted path, M4); reproduce 2021 from the ISD feature table without
   re-fitting.
4. **The project format is a Should, but Musts depend on it (×3).** FR-1.7 is a
   Should, yet FR-14.4 makes the project format the unit of exchange, and the CLI in
   FR-14.5 needs persisted intermediate results. The architecture cannot choose a
   persistence design until this is settled. *Proposed:* a minimal project directory
   as a Must at M1.
5. **Milestones and the course contradict each other (×2).** M1 claimed "K (hosted
   notebook)" while FR-12.1 rules out hosting. K needs A, C and E, which do not all
   exist until M4. *Fixed:* the hosted label is removed. *Proposed:* K becomes
   course-ready at M2, using the ISD's published psychoacoustic table for acceptance
   case 1, with the survey path E1.

### Important

6. **ISO compliance needs explicit profiles.** The PRD makes the Mitchell sources
   canonical, but a consultant must be able to defend numbers against the published
   2019 edition. *Proposed:* FR-2.9 analysis profiles. *Open:* which profile is the
   default, and which angle set an English survey gets.
7. **Angle data availability and the missing-language policy.** M1 cannot ship
   without the SATP v1.5 angles, the full tier list (Medium-tier languages are not
   listed) and the labels. These come from a 2026 chapter that may not be
   redistributable. The PRD is also silent on languages in neither shipped set. The
   language is declared per dataset, yet the ISD and course case 2 mix languages.
   *Proposed:* per-response or per-group language. *Open:* licence and fallback
   policy.
8. **No register of draft dependencies and risks (×3).** The PRD depends on two draft
   standards, a draft schema and an unpublished chapter. *Fixed:* §11 added. The
   builder also needs the draft standards' text, or the Example 6 inputs as fixtures.
9. **The numeric-drift rule is unworkable.** "Any numeric drift is a breaking change"
   cannot hold when wrapped libraries change their numbers. *Proposed:* NFR-4 rewrite
   with tolerances and method versions; FR-13.4 promoted to Must.
10. **The throughput criterion contradicts the no-optimisation stance (×2).** Journey
    B's "1,000 clips overnight" binds, while NFR-3 says throughput is whatever the
    libraries deliver. ECMA-418-2 metrics in pure Python can run slower than real
    time. *Open:* binding or indicative; metric scope for hours-long files.
11. **Licensing of audio dependencies and reference data (×2).** NFR-7 bans copyleft,
    but audio file I/O commonly goes through LGPL libsndfile. The best Table D.1
    coverage may be GPL. If the ArtemiS reference values cannot be published,
    contributors cannot run the audio validation locally, which journey L promises.
    *Open:* LGPL, GPL extras and ArtemiS publication.
12. **Product-level API requirements are missing (×2).** The brief stresses an API
    usable by researchers and data analysts, but the PRD states only headless and
    serialisation requirements. *Proposed:* NFR-12, with dataframes in and out, one
    public namespace and typed functions. *Open:* pandas only, or pandas and polars;
    NFR-13 governance.
13. **Extension is sequenced last, yet P6 is a primary persona.** *Proposed:* deliver
    each recipe with the milestone that introduces its extension type, and plugin
    discovery at M1.
14. **Uncertainty is under-specified.** Journey C promises "SPI with its
    uncertainty", but FR-6.1 covers only target sampling. FR-2.5 reports proportions
    without intervals, which APA reporting needs. *Proposed:* added Should items. The
    SPI paper's default sample size should be checked and adopted (unverified here).
15. **Plain-terms targets have no defined mapping.** No published source maps centre,
    spread and skew to skew-normal parameters. Once in the GUI, the mapping becomes a
    de facto standard. *Open:* the owner's definition.

### Minor

16. **Small scientific under-specifications.** FR-4.4's "tonality" names no method,
    and FR-2.9's profiles should fix it per edition. FR-4.2's IEC 61672 reference
    describes sound level meter calibration, not per-channel binaural tone
    calibration. Both are [agreed] items, so they were not edited; consider rewording.
17. **The GUI requirements are fixed tightly for an out-of-scope product.** FR-14.7
    fixes two GUI tiers as a Shall although the GUI is out of scope; consider making
    it informative. *Fixed:* the GUI row in §9 now puts the web tier on the M2 core.
18. **A previous-layout import is not backward compatibility.** FR-1.3a's
    "previous package's layout" is a data-layout mapping, not API compatibility, so
    it does not conflict with §8.
19. **Unverified dependency claim.** It is not confirmed that "pySQAT" is a Python
    package; SQAT itself is MATLAB. Check before planning around it.

## What the PRD does well: keep

- FR-2.3 states the canonical projection precisely. The critical reviewer checked
  that it reduces exactly to the 2019 (4 + √32) formula at 45°.
- FR-2.0's circumplex-as-instrument specification, and the versioned angle sets with
  confidence tiers (FR-2.4, FR-2.4a), are at the right level for a planner.
- Distributions are the centre of the package (FR-3.x), with explicit bandwidth, a
  parameterised contour level and the N < 30 rule.
- SPI has an explicit seed, sample size and sampling variability (FR-6.1).
- Predictions never use perceptual inputs and are always labelled (FR-5.5, FR-5.6).
- The no-reimplementation stance (FR-4.0, NFR-9), never silently dropping rows
  (FR-1.5), the privacy rule (NFR-5) and the clean-rebuild stance (§8) are all clear.
- The course acceptance cases under K are concrete and testable.

## For the architecture plan

The architecture plan is behind the PRD. Its §3.2 still uses the Aletta 2024 formula
and angles rather than the canonical projection and SATP v1.5, and its phase 4
includes report generation, which the PRD puts out of scope. Check it against PRD
0.4 once the §10.1 decisions are made.

## Owner responses (2026-10-30), applied in PRD 0.5

| Finding | Response |
| --- | --- |
| 1. In-browser core | Marimo dropped across the board. Tutorials are Quarto `.qmd` documents, so no in-browser constraint applies. |
| 2. Predictions to distributions | Prediction stands on its own, not only as an input to SPI. A set of single-response predictions over enough inputs forms a distribution that can be scored. FR-5.9 was rewritten; there is no pooling or refusal rule. |
| 3. Prediction and design dependencies | Single-response predictors do not block E. E1 and E2 kept. The fallback predictor and the 2021 approach remain open. |
| 4. Project format | Agreed: a minimal project directory at M1. |
| 5. Milestones and course | Agreed: the course is ready at M2. |
| 6. ISO profiles | Profiles kept. The 2024 revision profile with the SATP v1.5 angles is the default. |
| 7. Angle data | The chapter is open access and redistributable. Angles apply per response. An uncovered language with adjusted angles raises an overridable error; equal angles throughout proceed. |
| 8. Draft standards | Standards are cited by their public editions. The owner supplies texts and fixtures. |
| 9. Numeric drift | Tolerances follow the significant figures of each reference value. Floating-point drift within them never holds up development. FR-13.4 is a Must. |
| 10. Throughput | Performance is a goal, not a criterion. |
| 11. ArtemiS | The values are publishable. Comparisons are indicative, not requirements. The copyleft question remains open. |
| 12. API requirements | Left to the architecture stage; NFR-12 (API) removed. Governance remains open. |
| 13. Extension sequencing | Agreed: recipes are delivered per milestone. |
| 14. Uncertainty | The APA-style reporting requirement is removed; requirements come from ISO and the Mitchell literature. The SPI test-set uncertainty question remains open. |
| 15. Plain-terms targets | Dropped. Targets use Azzalini's parameterisation and terminology, and `sn`, as in the SPI paper. |
| 16. Metric under-specification | Not over-specified here. Each metric surfaces its source library, method and definition. |
| 17. GUI requirements | Removed. No GUI design is fixed. |
| 18. Previous-package layout | Removed: no previous-package designs are referenced. |
| 19. pySQAT | It is a Python package. Libraries on the LISQM list are assessed at implementation. |
