# Upgrading from Soundscapy 0.7 to 0.8.2

This guide walks through the changes you need to be aware of when upgrading
from the `0.7.x` series to the `0.8.2` stable release.

!!! note

    The last user-facing stable release of _Soundscapy_ on PyPI was `v0.7.8`.
    The `0.8.0rc1` through `0.8.0rc10` and `0.8.2.dev1` tags were
    pre-releases. `v0.8.2` consolidates that entire pre-release line into a
    single shipping version, so most users are upgrading across a much larger
    gap than the version number suggests.

## Why this matters

`v0.8.2` introduces two major new analysis modules (**SATP** and **SPI**), a
completely rewritten plotting API (**ISOPlot**), an embedded CircE R runtime,
and a Scientific Python SPEC 1 lazy-loading architecture for optional
dependencies. The plotting and optional-dependency changes are the most
visible to existing users.

## Plotting: `CircumplexPlot` → `ISOPlot`

The biggest user-visible change is that the `CircumplexPlot` class has been
replaced by the new layered `ISOPlot` API. The function-style helpers
`scatter_plot` and `density_plot` remain available and have been
re-implemented on the new backend, so the simplest plotting paths still work
unchanged.

### If you used `scatter_plot` / `density_plot`

No changes required. Imports and call sites continue to work:

```python
from soundscapy import scatter_plot, density_plot

scatter_plot(df, x="ISOPleasant", y="ISOEventful")
density_plot(df, x="ISOPleasant", y="ISOEventful")
```

### If you used `CircumplexPlot` directly

Migrate to `ISOPlot`. The new API is layered: build a base, then add scatter,
density, SPI, or jointplot layers.

**Before (0.7.x):**

```python
from soundscapy.plotting import CircumplexPlot, CircumplexPlotParams

params = CircumplexPlotParams(incl_outline=True)
plot = CircumplexPlot(df, params=params)
plot.scatter().density().show()
```

**After (0.8.2):**

```python
from soundscapy.plotting import ISOPlot

plot = (
    ISOPlot.create(df, x="ISOPleasant", y="ISOEventful")
    .add_scatter()
    .add_density()
)
plot.show()
```

Subplots, custom styling, and SPI integration are all part of the new
`ISOPlot` API. See the **Advanced Visualization Techniques** tutorial in the
docs for a full walkthrough.

## Plotly backend removed

Only the seaborn backend is supported in `v0.8.2`. The `plotly` dependency has
been dropped to keep installation lean. If you need static graphics, use
matplotlib's export from the seaborn backend; if you need interactive plots,
you can pass the underlying matplotlib `Figure` to your interactive viewer of
choice.

If you previously imported `Backend` to switch between seaborn and plotly,
that enum no longer exists.

## Optional dependencies overhaul

The optional-dependency story has been redesigned around
[Scientific Python SPEC 1](https://scientific-python.org/specs/spec-0001/).

### Install variants

| Install command                          | Includes                                                            |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `pip install soundscapy`                 | Core: surveys, plotting, databases. **Slim — no audio or R deps.** |
| `pip install "soundscapy[audio]"`        | Adds `acoustic-toolbox`, `mosqito`, `numba`, `scikit-maad`, `tqdm`. |
| `pip install "soundscapy[r]"`            | Adds `rpy2` for SPI / SATP. Also requires R + the `sn` R package.  |
| `pip install "soundscapy[all]"`          | Everything above.                                                   |

If you forget an extras install and import a module that needs it, you'll get
a clear error like:

```text
ImportError: 'rpy2' required for soundscapy[r], not installed.
Install with:  pip install 'soundscapy[r]'
```

### Renamed extras

The `[spi]` extras name briefly appeared during the rc cycle but has been
removed in favour of `[r]`. Update your `requirements.txt` / `pyproject.toml`
accordingly.

### `import soundscapy` is now slim

`import soundscapy` no longer pulls in audio, R, or any other optional
dependency at import time. Submodules (`soundscapy.audio`, `soundscapy.spi`,
`soundscapy.satp`) are deferred until first access via SPEC 1 lazy loading.

This should be transparent for most users — `from soundscapy import Binaural`
and `from soundscapy.audio import Binaural` both still work — but if you
relied on import-time side effects in optional submodules, you may need to
add an explicit reference to trigger loading.

## R-backed features (SPI / SATP)

CircE R scripts are now bundled with Soundscapy and sourced through `rpy2`.
You no longer need to install CircE separately from GitHub.

### What you need to install

```bash
pip install "soundscapy[r]"
R -q -e "install.packages('sn')"
```

The only external R package still required is `sn`. RTHORR is no longer
required or supported — it has been removed from the codebase.

### Quick start: SATP

```python
from soundscapy import fit_circe

results = fit_circe(df, language="eng", datasource="SATP")
results.table   # tidy DataFrame of fit statistics
```

`fit_circe` returns a `CircEResults` container holding one `CircE` per fitted
model variant. `results.table` exposes the tidy DataFrame view; the
`_repr_html_` makes it render nicely in Jupyter.

## `fit_circe` validation behaviour change

`fit_circe` now **raises by default** when input data fails schema validation
(PAQ values outside `[0, 100]`, missing required columns, and so on). In
earlier pre-release versions, `drop_invalid_rows=True` combined with the
schema bounds could silently drop every row, leaving you with an empty
result.

To restore the older permissive behaviour, pass `errors="warn"`:

```python
results = fit_circe(df, language="eng", datasource="SATP", errors="warn")
```

This emits a `UserWarning` describing the failing rows and continues with the
valid rows only.

## `ipsatize` centering default change

`ipsatize()` (and therefore `fit_circe`) now defaults to **grand-mean**
centering — one scalar per participant — matching the published SATP
analysis (Aletta et al., 2024) and the original R implementation. The legacy
`person_center` function performed column-wise centering (eight scalars per
participant), which produced SRMR values off by ~0.005.

The full `method` argument:

```python
from soundscapy import ipsatize

ipsatize(df, method="grand_mean")    # default; published SATP behaviour
ipsatize(df, method="column_wise")   # legacy person_center behaviour
ipsatize(df, method="row_wise")      # circumplex.ipsatize() behaviour
```

`person_center()` is retained as a thin wrapper for backward compatibility
but new code should use `ipsatize()`.

## IDE and type-checker support

`v0.8.2` ships PEP 561 type stubs (`__init__.pyi`) for the top-level package
and for `soundscapy.audio`, `soundscapy.spi`, and `soundscapy.satp`. This
means imports like

```python
from soundscapy.audio import Binaural
from soundscapy.satp import fit_circe
from soundscapy.spi import MultiSkewNorm
```

now resolve correctly in mypy, pyright, and IDE autocomplete, even before any
optional dependency is installed.

## Removed APIs

- **`CircumplexPlot`** class (replaced by `ISOPlot`).
- **`Backend`** enum (only seaborn is supported now).
- **Plotly plotting backend** (and the `plotly` dependency).
- **`rthorr` R package** integration.
- **CircE-from-GitHub** install requirement (CircE is now bundled).
- **Pydantic `ParamModels`** (replaced by dataclass-based parameter
  handling).

## Notes for contributors

If you're contributing to Soundscapy itself, two project-level tooling shifts
are worth knowing about:

- **Environment management is now driven by [pixi](https://pixi.sh)**.
  Common tasks: `pixi run lint`, `pixi run tests`, `pixi run docs-serve`.
  uv is retained for build, version, and PyPI publish (e.g.
  `pixi run version --bump release`, `pixi run build`).
- **Documentation has migrated from mkdocs to
  [zensical](https://zensical.org)**. The `mkdocs.yml` is gone; nav and
  config live in `zensical.toml`.

See `pixi.toml` and `CONTRIBUTING.md` for the full workflow.
