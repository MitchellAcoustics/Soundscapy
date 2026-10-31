# Notes: Soundscape Across Languages, chapter 4 (2026)

Mitchell, A., Oberman, T., Aletta, F. and Kang, J. (2026). Soundscape Translation Data:
Analysis, Validation, and Application. Chapter 4 in *Soundscape Across Languages*. Springer
Nature Singapore. Copyright: the authors, under exclusive licence to Springer Nature.

These notes were transcribed from the author proof (29 pages, dated 2026-08-13) on 2026-10-31.
The proof itself is not in the repository: it carries an exclusive-licence notice and no
open-access statement, so redistributing it in a public repository needs the owner's
confirmation. The tables below are factual data the package ships anyway (FR-2.4, FR-2.4a).
Proof page numbers are given for checking against the published version.

## Canonical robust projection (proof p. 23, §4.5)

P_ISO = Σ cos θ_i (ξ_i − μ) / (ρ Σ |cos θ_i|) and E_ISO = Σ sin θ_i (ξ_i − μ) / (ρ Σ |sin θ_i|),
where ξ_i is the score on scale i, θ_i its adjusted angle for the language, μ = (min + max)/2
the scale midpoint and ρ = (max − min)/2 the half-range.

Use: adjusted angles for High-tier languages; for Medium-tier languages "the adjusted angles
from Step 2 may still improve cross-language comparability, though results should be
interpreted with greater caution"; for Low-tier languages "we advise against applying adjusted
angles or calculating projected coordinates".

## Confidence tiers, SATP v1.5 (proof p. 19, Table 4.7)

CI is the Step 1 correspondence index, Score the Step 2 SEM fit score (maximum 3), Min R² the
minimum SSM R² across scales (Step 3), Cosine the Step 4 congruence.

| Language | CI | Score | Min R² | Cosine | Tier |
| --- | --- | --- | --- | --- | --- |
| alb | 0.354 | | | | Low |
| vie | 0.667 | | | | Low |
| zsm | 0.674 | | | | Low |
| cmn | 0.812 | 3.0 | 0.611 | | Medium |
| tha | 0.847 | 2.0 | | | Medium |
| por | 0.722 | 2.0 | | | Medium |
| bra | 0.778 | 2.0 | | | Medium |
| kor | 0.819 | 2.0 | | | Medium |
| fra | 0.931 | 1.0 | | | Medium |
| jpn | 0.826 | 1.0 | | | Medium |
| hrv | 0.854 | 3.0 | 0.982 | 0.969 | High |
| ind | 0.771 | 3.0 | 0.868 | 0.943 | High |
| ita | 0.903 | 3.0 | 0.883 | 0.987 | High |
| yue | 0.729 | 3.0 | 0.865 | 0.941 | High |
| eng | 0.986 | 3.0 | 0.986 | 0.992 | High |
| nld | 0.799 | 3.0 | 0.892 | 0.973 | High |
| pol | 0.882 | 3.0 | 0.943 | 0.982 | High |
| ell | 0.910 | 3.0 | 0.959 | 0.981 | High |
| spa | 0.951 | 3.0 | 0.976 | 0.991 | High |
| swe | 0.965 | 3.0 | 0.988 | 0.992 | High |
| deu | 0.972 | 3.0 | 0.965 | 0.991 | High |
| tur | 0.819 | 3.0 | 0.851 | 0.957 | High |
| arb | 0.889 | 3.0 | 0.906 | 0.989 | High |

cmn moved from High (Aletta et al. 2024) to Medium in v1.5 because of Step 3; its SEM results
and derived angles are identical between the two dataset versions (proof p. 24, §4.6.1).

## Adjusted angles, SATP v1.5 High tier (proof p. 20, Table 4.8)

Degrees, equal-communality CircE model, PAQ1 (pleasant) fixed at 0°. Position order: PAQ1
pleasant, PAQ2 vibrant, PAQ3 eventful, PAQ4 chaotic, PAQ5 annoying, PAQ6 monotonous, PAQ7
uneventful, PAQ8 calm.

| Language | PAQ1 | PAQ2 | PAQ3 | PAQ4 | PAQ5 | PAQ6 | PAQ7 | PAQ8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arb | 0.0 | 36.0 | 45.0 | 135.0 | 167.0 | 201.0 | 242.0 | 308.0 |
| deu | 0.0 | 64.0 | 97.0 | 132.0 | 182.0 | 254.0 | 282.0 | 336.0 |
| ell | 0.0 | 72.0 | 86.0 | 133.0 | 161.0 | 233.0 | 267.0 | 328.0 |
| eng | 0.0 | 46.0 | 94.0 | 138.0 | 177.0 | 231.0 | 275.0 | 340.0 |
| hrv | 0.0 | 84.0 | 93.0 | 160.0 | 173.0 | 243.0 | 273.0 | 354.0 |
| ind | 0.0 | 53.0 | 104.0 | 123.0 | 139.0 | 202.0 | 284.0 | 308.0 |
| ita | 0.0 | 57.0 | 104.0 | 142.0 | 170.0 | 274.0 | 285.0 | 336.0 |
| nld | 0.0 | 43.0 | 111.0 | 125.0 | 174.0 | 257.0 | 307.0 | 341.0 |
| pol | 0.0 | 69.0 | 91.0 | 128.0 | 176.0 | 266.0 | 274.0 | 339.0 |
| spa | 0.0 | 41.0 | 103.0 | 147.0 | 174.0 | 238.0 | 279.0 | 332.0 |
| swe | 0.0 | 66.0 | 87.0 | 146.0 | 175.0 | 249.0 | 275.0 | 335.0 |
| tur | 0.0 | 55.0 | 97.0 | 106.0 | 157.0 | 254.0 | 289.0 | 313.0 |
| yue | 0.0 | 39.0 | 43.0 | 159.0 | 168.0 | 240.0 | 282.0 | 329.0 |

The table's caption says "each language passing Step 2", but it lists only the 13 High-tier
languages. cmn passed Step 2, has no row, and per §4.6.1 its angles equal the Aletta et al. 2024
ones (PAQ2 about 18°, PAQ3 about 38°).

## Inconsistencies in the proof, reported to the owner

- The abstract and the concluding remarks (p. 26) say "ten" High-confidence languages, then
  list thirteen; Table 4.7 and the body give 13 High, 7 Medium, 3 Low (23 languages).
- Table 4.7's caption says "all 22 languages in SATP v1.5"; the table has 23 rows.
- Table 4.8's caption and contents disagree on which languages are included (see above).
