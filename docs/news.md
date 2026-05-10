# News

## 2026-05-10 Soundscapy 0.8.2 — the long-overdue stable release

The last user-facing stable release of _Soundscapy_ on PyPI was `v0.7.8`.
Since then the project has been on an extended pre-release cycle —
`v0.8.0rc1` through `v0.8.0rc10` and on into `v0.8.2.dev1` — without ever
publishing a non-prerelease tag. **`v0.8.2` consolidates eighteen months of
that work into a single shipping release.**

If you are upgrading from `0.7.x`, please read the
[migration guide](migration-0.7-to-0.8.md) before installing. Most users will
need to make small code changes.

### Breaking changes you should know about first

- **Plotting: `CircumplexPlot` → `ISOPlot`.** The `CircumplexPlot` class has
  been replaced by the new `ISOPlot` API, which uses a layered architecture
  for combining plot types and integrates SPI visualisation. The function-style
  helpers `scatter_plot()` and `density_plot()` still work and have been
  re-implemented on the new backend, so most casual users won't need to change
  anything. If you were customising `CircumplexPlot` directly, the migration
  guide has before/after snippets.
- **Plotly backend removed.** Only the seaborn backend is supported now.
  Removing `plotly` makes installation lighter and removes a dependency that
  was rarely used. If you need static graphics, matplotlib export from the
  seaborn backend is the recommended path.
- **R-backed features install path changed.** Use
  `pip install "soundscapy[r]"` for SPI / SATP features. Only the `sn` R
  package is required externally — CircE is now bundled with Soundscapy.
- **`fit_circe()` now raises by default** when input data fails schema
  validation. Pass `errors="warn"` if you want the older permissive behaviour.
- **Pydantic `ParamModels` removed** in favour of plain dataclasses for
  plotting parameters.

### The new ISOPlot API

The biggest change for everyday users is the ISOPlot rewrite. ISOPlot is built
around a layered plotting model — start from a base, then add scatter, density,
SPI, or jointplot layers. It also has first-class subplot support for
comparing soundscapes across conditions, and integrates the new SPI score
display directly into the visualisation.

The function-style entry points (`scatter_plot`, `density_plot`) remain the
quickest way to make a single plot, while `ISOPlot` itself is the way to build
multi-layer or multi-panel figures. The migration guide and the "Advanced
Visualization Techniques" tutorial in the docs walk through both.

### New: SATP and SPI modules

`v0.8.2` introduces two new R-backed analysis modules:

- **SATP** (Soundscape Attributes Translation Project): structural equation
  modelling for circumplex validation. The primary entry point is
  `fit_circe(data, language, datasource)`, which validates, ipsatizes, and
  fits all four circumplex model types in one call, returning a typed
  `CircEResults` container.
- **SPI** (Soundscape Perception Indices): tools for calculating Soundscape
  Perception Indices using the Multi-dimensional Skewed Normal distribution,
  with score calculation and `ISOPlot` visualisation integration.

Both modules use an embedded CircE R runtime, so you no longer need to
install the CircE package from GitHub yourself. The only external R
dependency is `sn`.

### Slim install + better IDE support

`import soundscapy` no longer pulls in audio, R, or any other optional
dependencies. The package now follows
[Scientific Python SPEC 1 (lazy loading)](https://scientific-python.org/specs/spec-0001/),
which means submodules are deferred until first access. PEP 561 type stubs
(`__init__.pyi`) ship with the package so `from soundscapy.audio import
Binaural` resolves correctly in mypy, pyright, and IDE autocomplete.

If you forget an extras install, you'll get a clear, actionable error message
pointing you at exactly what to install.

### Note for contributors

Project tooling has shifted: environment management is now driven by
[pixi](https://pixi.sh), with uv retained for build, version, and PyPI
publish. See `pixi.toml` for the new task layout (`pixi run lint`,
`pixi run tests`, `pixi run docs-serve`, etc.). Documentation has migrated
from mkdocs to [zensical](https://zensical.org). If you contribute, the
[CONTRIBUTING guide](CONTRIBUTING.md) has the updated workflow.

### Full migration guide

For copy-pasteable before/after snippets and a complete list of removed APIs,
see [Upgrading from 0.7](migration-0.7-to-0.8.md).

## 2026-05-08 Embedded CircE Runtime for R-Backed Features

Soundscapy's R-backed SPI and SATP functionality now uses an embedded CircE runtime.
The CircE R scripts are bundled directly with Soundscapy and sourced through `rpy2`,
so users no longer need to install CircE separately from GitHub.

### What Changed

- CircE is now shipped with Soundscapy as embedded R scripts.
- The Python R wrapper sources the bundled CircE implementation directly.
- Installation guidance now points R users to `pip install "soundscapy[r]"`.
- The only external R package still required for these features is `sn`.

### What You Need to Install

To use SPI and SATP features, install the Python optional dependency and make sure
R can install the `sn` package:

```bash
pip install "soundscapy[r]"
R -q -e "install.packages('sn')"
```

This change makes local setup and packaged installs more reliable by removing the
previous dependency on a separately installed CircE R package.

## 2024-08-15 Significant Enhancements to Soundscapy's Plotting Module

I am pleased to announce a comprehensive update to Soundscapy's plotting module, introducing enhanced flexibility, improved performance, and more extensive customization options for soundscape visualizations. This update represents a substantial improvement in our toolkit's capabilities.

### Modular and Extensible Architecture

The plotting module has undergone a complete restructuring, resulting in a more modular and maintainable codebase. This new architecture not only facilitates easier maintenance but also establishes a robust foundation for future enhancements and extensions.

### Multi-Backend Support

A key feature of this update is the introduction of multi-backend support. While Seaborn remains our primary plotting engine, we have integrated experimental support for Plotly. This addition enables the creation of interactive plots suitable for web-based applications, alongside our traditional static plots, providing users with increased flexibility in their visualization choices.

### Enhanced Customization

We have introduced a new `CircumplexPlot` class that serves as the central mechanism for creating and customizing plots. Complementing this, we've developed a `StyleOptions` class that offers granular control over visualization aesthetics. These additions allow for precise adjustments to plot elements, such as z-order modification and kernel density estimation bandwidth tuning.

### Streamlined API with Dual Interfaces

While we've significantly expanded the capabilities of our plotting module, we've maintained a focus on user accessibility. We now offer two primary interfaces for plot creation:

- Function-based Interface: The `scatter_plot()` and `density_plot()` functions remain available and have been optimized to leverage the new CircumplexPlot class internally. These functions offer a straightforward method for creating standard plots with minimal code.
- Class-based Interface: For users requiring more advanced customization, the `CircumplexPlot` class provides direct access to a wide array of plotting options and methods.

This dual approach ensures that both newcomers and advanced users can efficiently create the visualizations they need.

### Multiple Plot Creation

We've introduced a new `create_circumplex_subplots()` function, designed to simplify the process of creating multiple related plots. This function is particularly useful for comparing soundscapes across different locations or time periods, allowing for easy creation of grid-based visualizations.

### Future Developments

Our development roadmap includes several exciting features:

- Implementation of joint plots for the Seaborn backend
- Further improvements to the Plotly backend, including support for additional plot types and customization options
- Ongoing performance optimizations

### Upgrading and Breaking Changes

It's important to note that this update introduces breaking changes that will require modifications to existing code. The primary areas affected are:

- Import statements: The module structure has changed, necessitating updates to import statements. For example:

  ```python
  from soundscapy.plotting import scatter_plot, density_plot, Backend
  ```

- Function names and parameters: Some function names and their parameters have been modified for consistency and clarity. Please refer to the updated documentation for specific changes.
- Class-based interface: If you were previously using lower-level plotting functions, you may need to transition to the new CircumplexPlot class for advanced customizations.

We strongly recommend reviewing the updated documentation thoroughly when upgrading to this new version. While these changes may require some code adjustments, we believe the improved functionality and flexibility justify the effort.

We are confident that these improvements to the Soundscapy plotting module will significantly enhance your ability to create insightful and visually appealing soundscape visualizations. We look forward to seeing the innovative ways in which our user community will leverage these new capabilities.
