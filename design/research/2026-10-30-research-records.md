# Research records for the soundscapy rewrite

Status: research note, dated 2026-10-30. Not normative. Moved out of the architecture
plan at draft 0.5 (`design/greenfield_architecture_plan.md`). The plan keeps only the
policy lines that came from this research. Re-run section 3 at M3 planning (PRD §10.4,
item 2) and check section 1 again before the M2 skew-normal spec.

Contents:

1. Is there a Python multivariate skew-normal to wrap?
2. The 0.x skew-normal branches.
3. The LISQM psychoacoustic backend evaluation.

---

## 1. Is there a Python multivariate skew-normal to wrap?

The plan's rule is "wrap, never re-implement, unless there is nothing to wrap". A
check on 2026-10-30 tested whether there is something to wrap for the bivariate
skew-normal that FR-3.4 and FR-6.2 need.

| Candidate | Finding |
|---|---|
| `scipy.stats.skewnorm` | Univariate only. The request for a multivariate version (scipy issue 14870) has been open since 2021. |
| PyPI | No package provides a multivariate skew-normal. Sixteen plausible names were probed. |
| `mvskew` (MIT, PyPI) | Samplers only. No density, no fitting. |
| spc90/multivariate_skew_distributions (GitHub) | Density and sampler. No fitting, no centred parameters, no licence. |
| PyMC, TensorFlow Probability | Univariate case only. |
| Azzalini's own site | Lists Python code for the univariate case only. |

**Verdict.** Nothing suitable exists. The package implements the bivariate
skew-normal itself in numpy and scipy, follows Azzalini and the R `sn` package for
parameterisation and terminology (PRD §10.1), and validates every function against
`sn` outputs committed as fixtures. R is not a dependency.

---

## 2. The 0.x skew-normal branches

Two branches of the 0.x codebase explored a Python skew-normal. Both were reviewed,
with their history, on 2026-10-30 and run in that session.

### 2.1 `scipy-msn` (April 2025)

- One work-in-progress commit on top of the rpy2 SPI work. Its author labelled it
  "failing first passes".
- About 1,800 lines: a scipy-style `multivariate_skew_normal_gen` with pdf, cdf, rvs,
  dp2cp, cp2dp and MLE, and a plan to contribute it to scipy.
- On scipy 1.17 it imports private scipy internals (`_PSD.whiten`, `pinvh(rcond=)`)
  that have since been removed. 13 of its 15 tests fail on import or call.
- After patching those imports:
  - `dp2cp` matches R `sn` to 2e-9;
  - `rvs` reproduces the R mean and covariance;
  - `cp2dp` is not the inverse of `dp2cp` (round-trip error 6e-2);
  - `fit` returns its starting values with alpha = 0 on a strongly skewed sample.

**Verdict: not adopted as code.** Its test design is adopted as the test list:
pdf and log-pdf consistency, cdf equal to the multivariate normal at alpha = 0,
sample moments, the direct and centred round trip, and MLE recovery. Its sampler uses
the same conditioning construction the rewrite will use.

### 2.2 `157-remove-r-sn-dependency-for-spi` (May to June 2026)

- Two commits. `_mvskew.py` is a faithful Python translation of `sn`'s internal
  `msn.dp2cp` and `mst.dp2cp`, with `zeta`, `b(nu)`, cumulants and Mardia measures,
  wired into `dp2cp` for the SN, ESN, ST and SC families.
- It carries a committed R-derived baseline.
- On scipy 1.17 and numpy 2, all 13 pure tests pass and the SN result matches R to
  2e-9.
- It stops there. `cp2dp`, sampling and fitting still go through rpy2.

**Verdict: adopted.** The SN and ESN part of `msn_dp2cp` becomes the rewrite's
`dp2cp` after cleaning: drop `np.matrix`, drop the skew-t and skew-Cauchy paths the
PRD does not need, add types. Its R baseline values become the first reference
fixture, and its "R script beside the fixtures" pattern is the one to follow. This
is a carried-over piece of the 0.x codebase, which AGENTS.md otherwise forbids, so
the exception needs an ADR.

### 2.3 What the rewrite implements, and why it beats both

Only the planned implementation is complete:

| Piece | Source |
|---|---|
| Density and log-density (`dmsn`) | Azzalini; validated against `sn` |
| Sampling (`rmsn`) by the conditioning construction, with an optional rejection step that truncates to the circumplex square (the 0.x R wrapper's `sample_mtsn`) | Azzalini; 0.x R wrapper |
| `dp2cp` | The 157 port |
| `cp2dp` by the closed-form inverse for the SN family: marginal skewness to delta, then alpha from the scaled Omega | Azzalini and Capitanio 1999; `sn` `msn.cp2dp` |
| Admissibility check on centred parameters (marginal skewness below about 0.9953) | Azzalini and Capitanio 1999; Arellano-Valle and Azzalini 2008 |
| Maximum-likelihood fitting in centred parameters with `scipy.optimize`, from method-of-moments starting values | As `sn`'s `selm` does; direct-parameter MLE runs to the alpha boundary |

It is a few hundred lines, against the 1,800 of the scipy-style draft, because it
does not carry scipy's generic multivariate framework, and it uses no private scipy
API. Contributing to scipy later stays possible, because the API can mirror scipy's,
but it is not on the path to 1.0. Fitting is a Must at M2 (owner, 2026-10-30).

---

## 3. The LISQM psychoacoustic backend evaluation

### 3.1 Method

Every project in the LISQM list (https://github.com/ZhenxianLi/LISQM) with a Python
interface was checked on 2026-10-30 against the hard requirements for the Python
environment (NFR-1, NFR-2):

- installable from PyPI as wheels on Windows, macOS and Linux;
- no compiler, no MATLAB, no R;
- Python 3.11 and later.

Licence was recorded but, by the owner's decision of the same day, is not a filter
(copyleft is acceptable, NFR-7). Complete Table D.1 coverage is not a requirement and
a missing open-source implementation never blocks development (FR-4.4). LISQM
computes nothing itself, so coverage claims are the projects' own.

MATLAB, Julia, Rust-only, C#, Pure Data and C-only entries fail the first requirement
and are not tabulated. They include SQAT, AMT, AARAE, ITA-Toolbox, PsySound3,
PsychoacousticMetrics.jl, Kirin Hypha, Mosqito.NET, SoundPalette, zwickerloudness and
epnl.

### 3.2 Findings

| Project | Licence | Install, Python, dependencies | Coverage | Status at the check | Verdict |
|---|---|---|---|---|---|
| MoSQITo 1.2.1 | Apache-2.0 | PyPI; `>=3.5`; numpy, scipy, pyuff; matplotlib used by ECMA roughness but undeclared | ISO 532-1 stationary and time-varying loudness, DIN 45692 sharpness, Daniel–Weber roughness, ECMA-418-2 loudness and roughness (pre-2025 edition), ECMA-74 prominence and tone-to-noise ratios, SII | Established. No release since April 2024. Imports and runs on 3.11 and 3.13 with numpy 2; 1 kHz at 74 dB gives 10.9 sone, consistent with ISO 532-1 | **Default backend**, pinned to a tested range |
| phonometry 3.3.0 | MIT | PyPI; **`>=3.13`**; numpy, scipy | ISO 532-1/2/3; ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength; DIN 45692; Osses fluctuation strength; psychoacoustic annoyance; ECMA-418-1; ISO 1996-2; levels, weighting, octave bands | Released 2026. One maintainer. 1,893 self-reported conformance checks in CI. Has a DOI | **Optional backend in its own `phonometry` extra**, never an environment marker on `audio` |
| scikit-maad 1.5.2 | BSD-3 | PyPI; `>=3.9`; numpy, scipy, pandas, matplotlib, scikit-image, pywavelets | 50+ ecoacoustic indices, SPL | Active | **Backend for FR-4.5 ecoacoustic indices**, `audio` extra |
| acoustic-toolbox 0.2.2 | BSD-3 | PyPI; `>=3.11`; numba, pysoundfile, pyoctaveband, matplotlib, pandas | Weighting, Leq, octave bands, ISO 1996-2 tonal audibility | Active fork of the archived python-acoustics | Not used. Level metrics are about thirty lines of scipy. Revisit for ISO 1996-2 |
| sottek-hearing-model 0.1.14 | AGPL-3 on PyPI, GPL-3 on GitHub | PyPI; `>=3.11`; numpy, scipy, bottleneck, plus matplotlib, tqdm and **PyQt6** as hard dependencies | ECMA-418-2 (2025) loudness, roughness, tonality, fluctuation strength, translated from the SQAT authors' verified MATLAB | Active | **Not adopted** (owner, 2026-10-30). A 0.1.x package with a hard dependency on a GPL GUI toolkit of about 60 MB has no place in a headless extra, and its licence is ambiguous. Revisit if PyQt6 becomes optional upstream |
| MetaSona 0.2.2 | GPL-3 combined (Apache, BSD, MIT parts) | Wheels for all three OS; `>=3.9,<3.15`; numpy, scipy | Zwicker loudness, DIN sharpness, Daniel–Weber roughness, Aures tonality, ECMA-418-2 (2025) loudness, roughness, tonality; 15 to 70 times faster than MoSQITo | New. Authors call roughness and tonality experimental | Not yet. A candidate fast backend, and the only Aures tonality, once validated |
| pySQAT | None stated | Not on PyPI; `requirements.txt` with PySide6 | SQAT's set: ISO 532-1, ECMA-418-2, DIN 45692, Daniel–Weber, Osses fluctuation strength, Aures, psychoacoustic annoyance, EPNL | Developing; validation workflow, no published results | Unusable without a licence. The most complete candidate otherwise. Watch |
| ecma418-torch | MIT | Not on PyPI; requires torch | ECMA-418-2 (2025) roughness, fluctuation strength | One commit | Excluded: torch, unproven |
| iso532-1-rs | Apache-2.0 | No wheels; maturin source build | ISO 532-1, with MoSQITo-parity tests | New | Excluded: no wheels. A possible fast backend later |
| refmap-psychoacoustics | GPL-3 | Not on PyPI | Aural detectability (Python part only) | Active | Not needed |
| rota-order 0.7.0 | GPL-3 | PyPI | ECMA-418-1 tone-to-noise and prominence ratios | Active | Not needed; MoSQITo covers them |
| torch_amt 0.2.0 | GPL-3 | PyPI; torch | Moore–Glasberg time-varying loudness | Active | Not needed; torch, and not in Table D.1 |
| LoudnessModel | BSD-2 | No release | Moore et al. 2018 time-varying loudness | Dormant | Not needed |
| PyDSM 0.15.2 | GPL-3 | PyPI | ISO 226 contours | Dormant | Not needed |
| PsychoacousticParametersMeasurer | None stated | Not on PyPI | Zwicker set | Legacy, 2021 | Unusable |
| NumpyLibforPsychoAcoustic | None stated | Not on PyPI | Moore–Glasberg loudness | New | Unusable without a licence |
| MoSQITo-FDP, python-acoustics, SUAVE, deeuu/loudness | Apache, BSD-3, LGPL-2.1, GPL-3 | Archived or dormant | Early ECMA-418-2, ISO 1996-2, EPNL, loudness | Legacy | Not used |
| PsyTools | Unknown | Code not public | Unknown | Unknown | Not evaluable |

### 3.3 Policy that followed (kept in the plan)

- MoSQITo is the single backend of the `audio` extra at M3, with level metrics and
  IACC built in.
- phonometry is an explicit `phonometry` extra for Python 3.13 and later, never an
  environment marker, so `soundscapy[audio]` gives the same metric list on every
  Python version.
- No backend that depends on PyQt6 is adopted.
- MoSQITo risks and mitigations: dormant since April 2024, and an undeclared
  matplotlib import. Pin a tested range (NFR-4). matplotlib is a core dependency
  anyway. Apache-2.0 permits a maintained fork or vendored patch if it stays dormant.

### 3.4 What M3 would offer on these findings

- `soundscapy[audio]` on any supported Python: levels (LAeq, LCeq, LAF5, LAF95,
  LA10 − LA90, LCeq − LAeq, SEL, kurtosis) and IACC built in; loudness (N5, Navg,
  Nrmc, N95, N5/N95) and sharpness (S5, Savg, S50, S95) from MoSQITo; roughness by the
  Daniel–Weber method, offered under that name; ECMA-74 prominence and tone-to-noise
  ratios from MoSQITo; ecoacoustic indices from scikit-maad.
- `soundscapy[phonometry]` on 3.13 and later adds ECMA-418-2:2025 loudness,
  roughness, tonality and fluctuation strength, Osses fluctuation strength and
  psychoacoustic annoyance.
- Gaps with `soundscapy[audio]` on 3.11 and 3.12: fluctuation strength, ECMA-418-2
  roughness and tonality, Aures tonality and psychoacoustic annoyance. Aures tonality
  exists only in an unmerged MoSQITo pull request, in unlicensed pySQAT and in
  experimental MetaSona. The `phonometry` extra closes all but Aures tonality.
- Where Table D.1 names a method a backend does not implement, the metric appears
  under the method the backend implements, and the substitution is documented, not
  presented as coverage (FR-4.4). Which tonality each Table D.1 edition means is an
  owner input.
- A later package, a licence change in pySQAT, a MoSQITo release or
  sottek-hearing-model dropping PyQt6 closes a gap by adding one implementation.

### 3.5 What to re-check at M3 planning

Releases and Python ranges of every row above; whether MoSQITo has released or been
forked; whether pySQAT has a licence; whether MetaSona's roughness and tonality are
validated; whether sottek-hearing-model's PyQt6 dependency is optional; phonometry's
maintenance and Python floor.
