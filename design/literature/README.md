# Reference literature

The papers and chapters the package implements. Read the source before you implement or change
a procedure, and cite it in the procedure's docstring `References` section (PRD FR-12.4). The
PRD §2 table lists every source; this folder holds the ones available as files.

Not part of the published docs. Each file is redistributed under the licence shown, with the
attribution given here.

| File | Source | Licence | Use it for |
| --- | --- | --- | --- |
| `mitchell-aletta-kang-2022-jasa-el.pdf` | Mitchell, A., Aletta, F. and Kang, J. (2022). How to analyse and represent quantitative soundscape data. *JASA Express Letters* 2, 037201. https://doi.org/10.1121/10.0009794 | CC BY 4.0 | The distributional representation, projection before aggregation, the 2019 projection formula (Eq. 2), the 50th-percentile contour as a simplified view, "percent likely to perceive". The 2022 reproduction (journey I). |
| `aletta-et-al-2024-applied-acoustics.pdf` | Aletta, F., Mitchell, A., Oberman, T., Kang, J. and 52 others (2024). Soundscape descriptors in eighteen languages: Translation and validation through listening experiments. *Applied Acoustics* 224, 110109. https://doi.org/10.1016/j.apacoust.2024.110109 | CC BY 4.0 | The Aletta 2024 adjusted-angle set and formulation (FR-2.3, FR-2.4), the SSM validation, translated attribute labels (NFR-10). |
| `mitchell-et-al-2026-satp-chapter-4-notes.md` | Notes on Mitchell, A., Oberman, T., Aletta, F. and Kang, J. (2026). Soundscape Translation Data: Analysis, Validation, and Application. Chapter 4 in *Soundscape Across Languages* (Springer Nature). | Notes only; see the file | The canonical robust projection, SATP v1.5 confidence tiers and adjusted angles (FR-2.3, FR-2.4, FR-2.4a). |

## Facts the owner has confirmed from these sources (2026-10-31)

- The 2022 paper used ISD v0.2.3 (its §6). Panel (a) of its figure is Russell Square. Its density
  plots used seaborn's KDE defaults with `bw_adjust=1.2`.
- "The contour enclosing 50%" means the 50th-percentile (median) contour of the 2022 paper: the
  simplified view whose shaded area contains 50% of the responses. 50 is the default; users may
  choose another percentile (FR-3.1).
- The Aletta 2024 angles are the published values: English monotonous 231°, Italian chaotic 142°.
- Translated attribute labels come from the ISO/TS 12913-3 revision, Aletta et al. 2024 or the
  2026 chapter.
