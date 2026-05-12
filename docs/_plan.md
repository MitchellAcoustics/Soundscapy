# Soundscapy Docs Restructure Plan

Status: **Phase 1 complete** (infrastructure setup) — content restructure pending.

---

## Architecture: Two-site split

| Site | Tool | URL | Purpose |
|------|------|-----|---------|
| Main docs | Zensical (MkDocs Material) | `drandrewmitchell.com/Soundscapy/` | Reference + Explanation |
| User guide | Quarto | `drandrewmitchell.com/Soundscapy/tutorials/` | Tutorials + How-to + Publications |

**Rationale:** Zensical is the faster entry point and handles static content better (Material theme, clean nav, mkdocstrings API reference). Quarto handles executable notebooks natively (freeze/cache, Colab links, ipynb output). Splitting along Diátaxis lines gives each tool what it's best at.

**Deployment:** Both build into `site/`. Zensical runs first (`zensical build` → `site/`), Quarto runs second (`quarto render --no-execute` → `site/tutorials/`). Single GitHub Pages artifact.

---

## Diátaxis content split

### Zensical site (`docs/`)

- **Explanation** — `background.md`, theory, concepts, "what is a soundscape"
- **Reference** — full API docs via mkdocstrings

### Quarto site (`examples/`)

- **Getting Started** — QuickStart, beginner tutorial
- **Tutorials** — numbered learning sequence (2–6)
- **How-to Gallery** — BinauralAnalysis + gallery pages extracted from tutorial tails
- **Publications** — standalone canonical documents (paper + IoA teaching resource)

---

## Content map and decisions

### Keep as-is (Phase 1)

All `.qmd` files stay in `examples/` without content changes during Phase 1.

### Phase 2 content restructure

| File | Action | Notes |
|------|--------|-------|
| `background.qmd` | Move content → `docs/background.md` on zensical; remove from quarto nav | Already has a counterpart at `docs/background.md`; content should be merged/updated there |
| `QuickStart.qmd` | Keep, strip background prose section | The academic background section duplicates `background.md`; make it a pure quick-reference how-to |
| `0_QuickStart_for_Beginners.qmd` | Keep as-is | Primary beginner tutorial |
| `HowToAnalyseAndRepresentSoundscapes.qmd` | Keep as-is, move to Publications section | Reproducible companion to Mitchell, Aletta & Kang (2022, JASA). Do not modify. |
| `1_Understanding_Soundscape_Analysis.qmd` | **Remove** — duplicates `background.md` with inferior prose | Best sections (if any) can be folded into `docs/background.md` |
| `2_Working_with_Soundscape_Survey_Data.qmd` | Keep; add callout marking gallery tail | Core tutorial |
| `3_Advanced_Visualization_Techniques.qmd` | Keep; gallery tail → dedicated gallery page | Most gallery content is here |
| `4_Understanding_Soundscape_Perception_Index.qmd` | Keep as-is | SPI tutorial, distinct |
| `5_Working_with_Soundscape_Databases.qmd` | Keep; remove loading/validation intro (already in tutorial 2) | Minor dedup |
| `6_Soundscape_Assessment_Tutorial.qmd` | Keep as capstone tutorial | Decide: merge extra depth from this into `0_QuickStart_for_Beginners` or keep separate as "complete workflow" |
| `IoA_Soundscape_Assessment_Tutorial_v2.qmd` | Keep as-is, move to Publications section | Carefully crafted IoA workshop teaching document. Do not modify. |
| `BinauralAnalysis.qmd` | Move to How-to Gallery section | Distinct audience (`soundscapy[audio]`), task-oriented |
| `about.qmd` | Replace placeholder with real content or delete | Currently just "About this site" with no body |

### Open decisions for Phase 2

1. **`0_QuickStart_for_Beginners` vs `6_Soundscape_Assessment_Tutorial`**: Both declare identical learning objectives. Either (a) frame `0_` as intro and `6_` as capstone with explicit progression, or (b) merge `6_`'s extra depth (stats, extended SPI) into `0_` and remove `6_`.
2. **Gallery extraction**: Create standalone `gallery/visualization.qmd` and `gallery/analysis.qmd` pages pulling the "here are all the plot types/variations" sections out of tutorials 3 and 2. Format like seaborn/matplotlib gallery — short self-contained examples, minimal prose.
3. **Publications citation blocks**: Add Quarto `citation:` frontmatter to both `HowToAnalyseAndRepresentSoundscapes.qmd` and `IoA_Soundscape_Assessment_Tutorial_v2.qmd` to display a proper citation block.

---

## Proposed quarto sidebar (Phase 2 target)

```yaml
sidebar:
  - id: user-guide
    contents:
      - section: "Getting Started"
        contents:
          - index.qmd
          - QuickStart.qmd
          - 0_QuickStart_for_Beginners.qmd

      - section: "Tutorials"
        contents:
          - 2_Working_with_Soundscape_Survey_Data.qmd
          - 3_Advanced_Visualization_Techniques.qmd
          - 4_Understanding_Soundscape_Perception_Index.qmd
          - 5_Working_with_Soundscape_Databases.qmd
          - 6_Soundscape_Assessment_Tutorial.qmd

      - section: "How-to Gallery"
        contents:
          - BinauralAnalysis.qmd
          # future: gallery/visualization.qmd, gallery/analysis.qmd

      - section: "Publications"
        contents:
          - HowToAnalyseAndRepresentSoundscapes.qmd
          - IoA_Soundscape_Assessment_Tutorial_v2.qmd
```

### Proposed zensical nav (Phase 2 target)

```toml
nav = [
    { "Home" = "index.md" },
    { "Background" = "background.md" },
    { "User Guide ↗" = "https://drandrewmitchell.com/Soundscapy/tutorials/" },
    { "API Reference" = [...] },
    { "Contributing" = "CONTRIBUTING.md" },
    { "News" = [...] },
]
```

---

## Technical setup (implemented in Phase 1)

### Build order

1. `zensical build` → creates `site/` (clean)
2. `quarto render --no-execute` → writes to `site/tutorials/` (uses freeze cache)
3. GitHub Pages uploads `site/` as a single artifact

### Pixi tasks

| Task | Purpose |
|------|---------|
| `zensical-build` | Internal: just `zensical build` |
| `examples-execute` | Dev-only: re-execute all notebooks, update freeze cache |
| `examples-render` | CI-safe: render using freeze, depends-on `zensical-build` |
| `docs-build` | Full build: depends-on `examples-render` |
| `docs-serve` | Main docs local dev: just `zensical serve`, no dependencies |

For tutorial local dev: run `quarto preview` inside `examples/` directly.

### Freeze files — important

Quarto stores freeze outputs at `examples/.quarto/_freeze/`. These **must be committed** for CI `--no-execute` to work.

After any notebook content change:

1. `pixi run examples-execute` (locally)
2. `git add examples/.quarto/_freeze/`
3. Commit with the notebook changes

### URLs

| | URL |
|--|--|
| Main docs | `https://drandrewmitchell.com/Soundscapy/` |
| User guide (quarto) | `https://drandrewmitchell.com/Soundscapy/tutorials/` |

---

## Phase 2 checklist (content restructure — do later)

- [ ] Merge `background.qmd` → `docs/background.md`; promote Background to top-level nav in zensical
- [ ] Remove `1_Understanding_Soundscape_Analysis.qmd` from quarto nav (and eventually delete)
- [ ] Strip background prose from `QuickStart.qmd`
- [ ] Restructure quarto sidebar to match proposed nav above
- [ ] Resolve `0_` vs `6_` overlap
- [ ] Create `gallery/` pages; extract tail sections from tutorials 2 and 3
- [ ] Add citation frontmatter to Publications notebooks
- [ ] Write real content for `about.qmd` or delete it
- [ ] Delete `docs/tutorials/rendered/` directory (no longer referenced)
- [ ] Update `docs/tutorials/index.md` or delete it (Tutorials nav is now external)
