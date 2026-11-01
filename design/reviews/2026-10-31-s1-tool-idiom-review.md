# Review of S1: how the code uses its tools and libraries

Date: 2026-10-31. Inputs: the code at `a8fe18f` (`soundscapy-rewrite`, end of M1 slice S1):
`pyproject.toml`, `Taskfile.yml`, `.pre-commit-config.yaml`, the three workflows,
`docs/great-docs.yml`, the `.claude/` hooks and settings, `src/` and `tests/`. Each tool's
upstream documentation or source, read on 2026-10-31 (links inline). PRD 0.9, plan 0.5 and the
M1 spec, for the decisions behind each choice. Outcome: proposals for the owner. Nothing here
has been applied.

The remote branch has since gained one commit, `e5c0506`, which adds only the S2 plan
(`design/plans/2026-10-31-m1-s2-status-quality.md`). F2 and F18 say where that plan meets them.

## Status (2026-11-01)

Applied on `claude/dev-tools-idiom-review-qfncq4`, each with `task check` passing:

- `eb43612`: F1 (hook commands).
- `8e53c7b`: the move to `prek.toml` with a local `uv lock` hook, F14 and H10.
- `8504ad1`:
  - go-task installed outside the project (owner decision; see "Put `uv run` inside the
    tasks" below);
  - F10, F11, F12, F13;
  - F7's Taskfile half;
  - the wording of H1 and H2.
- `6ba936a`: F7's test half, F15 to F21, H3, H4, H5, H6, H8 and H9.
- `2d6fd53`: F2, F3, F4 and F9, following the owner's decision to keep pandas as scikit-lego
  does. The suite also passes at the `lowest-direct` floors.

- `85ee3d0`: F5 and F6 (owner decision; spec §6.15 amended). Colours come from the active
  `axes.prop_cycle` and are passed explicitly, not left to each Axes' cycle. The marker
  change needs the cycle's length, which matplotlib does not expose per Axes.
- `8bc0569`: F8, option (b) (spec §6.3.9 amended). Option (a), INFO opt-in, was dropped:
  NFR-13 requires a default handler that shows informational messages, including a
  download's licence line (FR-10.1).
- `efc7399` and `c553681`: the simplify pass and the code review (AGENTS.md, Review), and
  their fixes.

Open:

- **Owner decision, raised by the code review.** With narwhals-native nulls (F3),
  `attach` on a pandas frame with an unmatched row makes `ssp_included` an `object` column
  holding NaN. `df[df["ssp_included"]]` then raises "Cannot mask with non-boolean array
  containing NA / NaN values"; the earlier nullable `boolean` worked there. The options:
  - accept this, and document `df[df["ssp_included"].eq(True)]`;
  - return pandas' nullable `boolean` for that column only;
  - revisit F3.
- **Code-review findings skipped, with reasons:**
  - Root logging at WARNING through an implicit `basicConfig()` hides INFO lines. That *is*
    standard logging configuration, which NFR-13 says the user may silence with.
  - pyplot's figure registry as global state. This is owner decision F5; the figure is
    returned to the caller, not hidden.
  - The narwhals canary test asserts current behaviour. `uv.lock` pins narwhals, so the test
    fails only on a deliberate upgrade, when the masking choice should be revisited.
- F22, keep, with no action.
- F23, once the docs site has an address.

## Method and limits

The owner asked whether the code and tooling use each tool as its authors intend, or work
around it, and asked for the brief's own suggestions to be challenged. Each finding gives what
we do, what upstream documents, the consequence, and a recommendation. Where the design departs
from upstream on purpose, the finding names the decision; reopening it is the owner's call.
Upstream documentation was read in four research passes, one per group of tools.

Limits of this pass:

- **Commands ran only at the end.** The cloud image's uv (0.11.32) is older than
  `required-version`, so while the review was written the `PreToolUse` hook failed on every
  shell call (F1). Its claims come from code, upstream docs and upstream source. Applying F1's
  fix restored the shell, and the checks marked *run* were then executed at `94f594e`, where
  `task check` passes (461 tests, 99% coverage). Items still marked **verify** have not been
  run.
- **The documentation sites were unreachable.** The session's network policy blocks
  matplotlib.org, docs.python.org, docs.astral.sh, docs.pytest.org, taskfile.dev and
  narwhals-dev.github.io. Where a statement is marked *source*, it was read from the file the
  docs page is built from, on GitHub, at the version noted. Statements marked *paraphrase* come
  from search extracts and have not been checked word for word.

Priorities: **P1** fix before S2 builds on it; **P2** fix when the file is next touched;
**P3** consider. **Owner** marks a finding that reopens a recorded decision.

## Summary

The boundaries are well chosen, and most of the tooling is used as documented: narwhals' stable
API in one module, lazy matplotlib, uv dependency groups, prek's builtin hooks, and an
import-linter contract for each architectural rule. The deviations fall into four groups:

1. **Agent hooks that a project setting can switch off** (F1). This is what stopped every
   command in the session that wrote this review.
2. **Types erased at the frame boundary** (F2). narwhals provides the TypeVars for "same frame
   type out as in", but we annotate `Any` and suppress the lint rule twelve times.
3. **Suppressions and restated settings where the tool has an option** (F15 to F21). Each is a
   few lines of config.
4. **Two deliberate departures from upstream guidance**, both owner decisions with costs worth
   restating: pyplot-free figures for `ax=None` (F5), and a library logger with its own handler
   and level (F8).

Several of these trace back to the house rules themselves. "House rules that are misspecified"
(H1 to H10) proposes new wording for each.

| ID | Area | Finding | Priority |
|---|---|---|---|
| F1 | Agent hooks | An old uv refuses every shell call via `required-version`; the Stop hook cannot exit | P1, applied |
| F2 | narwhals | Frames typed `Any` (12 `noqa: ANN401`) instead of narwhals' `IntoDataFrameT` | applied |
| F3 | narwhals | Drop the pandas branch in `with_columns` for narwhals-native nulls | applied |
| F4 | narwhals | `to_eager` relies on an undocumented `TypeError` | applied |
| F5 | matplotlib | Pyplot-free `ax=None` departs from matplotlib's library pattern | applied |
| F6 | matplotlib | Hard-coded petroff10 ignores style sheets and property cycles (FR-8.3) | applied |
| F7 | matplotlib | `MPLBACKEND` set in the Taskfile, not in the test suite | applied |
| F8 | logging | The library logger's level bypasses the application's root level | applied |
| F9 | tomlkit | `_tables_last` probably duplicates tomlkit | applied |
| F10 | Taskfile, prek | `sp-repo-review` runs unpinned | applied |
| F11 | Taskfile | `check:fast` stops at the first failing step | applied |
| F12 | CI | Release job holds `id-token` through `task check`; no `--no-sources` | applied |
| F13 | CI | `--locked` per command rather than `UV_LOCKED` per job | applied |
| F14 | CI | prek cache key leaves out the prek version | P3, applied |
| F15 | ruff | `banned-module-level-imports` replaces three `noqa: PLC0415` and enforces lazy imports | applied |
| F16 | ruff | `allow-star-arg-any` for forwarded `**kwargs` | applied |
| F17 | ty | `[tool.ty.environment]` restates ty's defaults | applied |
| F18 | pytest | `tests._helpers` imports rely on undocumented importlib-mode behaviour | applied |
| F19 | pytest | `strict = true` replaces three settings | applied |
| F20 | import-linter | Ignores for imports that do not exist, with the alert silenced | applied |
| F21 | great-docs | Six keys restate defaults | applied |
| F22 | great-docs | Quarto from `quarto-cli` is off upstream's documented path (keep) | P3 |
| F23 | great-docs | No `Documentation` URL | P3 |

## Two suggestions in the brief, challenged

### "Follow narwhals' guidance and do not declare a dataframe library as a dependency" (decided: keep pandas)

What narwhals says (*source*, narwhals `docs/installation.md` on `main`): it "has no required
dependencies: it only ever uses the dataframe libraries you pass into it". A library that lists
backends as direct dependencies means "every consumer is then forced to install" them. The page
suggests a dependency group for testing, or to "re-expose the backends as your library's own
optional extras, so consumers opt in explicitly".

We already follow this for every library except pandas. polars is in the `test` group only.
pyarrow is not a dependency yet. Nothing outside `_frames` may import a dataframe library (an
import-linter contract), and a test proves `import soundscapy` works without polars or pyarrow.

pandas is the one exception, because soundscapy also creates frames where there is no user
frame to copy: `load_isd(subset=True).data`, `read_survey(path).data`, `Coordinates.to_frame()`,
`Status.to_frame()`, and S2's planned `Survey.to_frame()`. narwhals does not choose a library
for those, whatever is installed. *Run*: with pandas and polars both installed and imported,
`nw.from_dict({...})` raises "Calling `from_dict` without `backend` is only supported if all
input values are already Narwhals Series", and `nw.read_csv(path)` requires `backend`.

**Decision (owner, 2026-10-31): follow scikit-lego, narwhals' own showcase project. pandas
stays a required dependency, and frames soundscapy creates are pandas.** The discussion
that led there, recorded so it is not re-run:

- *Evidence from scikit-lego* (`pyproject.toml`, `sklego/datasets.py`,
  `sklego/pandas_utils.py`, `sklego/preprocessing/pandastransformers.py` on `main`; its
  Quansight Labs write-up, "How Narwhals and scikit-lego came together to achieve
  dataframe-agnosticism", sits behind a browser check this session could not pass):
  - It declares `narwhals`, `numpy`, `pandas` and `scikit-learn` as dependencies, and polars
    and pyarrow only in its `test` extra.
  - Its dataset loaders call `pd.read_csv` and return pandas.
  - Frames users pass in go through explicit `nw.from_native(...)` and `nw.to_native(...)`.
  - No module checked uses `@nw.narwhalify`.

  That is the shape of the S1 code.
- *No dataframe library in `dependencies`* was considered and dropped. A plain
  `pip install soundscapy` could not run the quick-start, and the teaching material would
  need an extra. narwhals' advice to keep backends out of dependencies fits a library that
  only receives frames.
- *polars as the internal library* was considered and dropped, with or without a hidden
  conversion to pandas. *Run*: polars' `to_pandas()` needs pyarrow, and gives `object`
  booleans and `float64` integers. `use_pyarrow_extension_array=True` keeps nulls, but gives
  `bool[pyarrow]`-style dtypes. Either way users would carry polars, pyarrow and pandas,
  while the clean nulls stop at the boundary, and our computation, in numpy with masks,
  gains nothing.
- *`@nw.narwhalify`* was considered and dropped. It is typed `-> Callable[..., Any]`; by
  default it passes non-frames and lazy frames through; and narwhals' typing page prefers
  `from_native`/`to_native` as "better at preserving type hints".

What changes as a result: frame types (F2), narwhals-native nulls in `with_columns` (F3), and
`backend=` only if a user asks (plan §8 unchanged). If true nulls for pandas users are wanted
later, the lever is which pandas dtypes we emit (nullable or pyarrow-backed), not polars.

### "Put `uv run` inside the tasks instead of calling `uv run task`"

The tasks already do this: every command in `Taskfile.yml` starts with `uv run`, and the docs
tasks add `--group docs` (house-style; spec §5.3). The outer `uv run` in `uv run task <name>`
is there for a different reason. The `task` binary itself is installed into the project venv
by `go-task-bin` in the `dev` group (spec §5.4), so outside an activated venv nothing else puts
it on PATH.

Dropping the outer `uv run` therefore means installing go-task some other way: Homebrew,
winget, `uv tool install go-task-bin`, or a CI action. That adds a second tool that every
contributor installs and nothing pins (`uv.lock` pins `go-task-bin` 3.54.0 today). It also
reopens the name clash that `stop_check.py` already works around: Taskwarrior installs a
`task` binary too. go-task's docs give no guidance for Python virtualenvs. The brief's premise
is right that the venv belongs inside the task definition, and that is where it already is.

The two layers are redundant only when the outer one is present. "When uv run is used, the
project is locked and synced before invoking the requested command" (*paraphrase*, uv
"Syncing"), so inside `uv run task` each inner `uv run` repeats that check. That costs one uv
start per command, and makes nested runs revisit the lock, which is why the offline docs job
sets `UV_OFFLINE`. The inner layer earns its keep when the Taskfile runs from an activated venv
or a global `task`. **Recommendation: keep both layers,** and say why in a comment at the top of
`Taskfile.yml`, so a later tidy-up does not remove one as duplication. F13 covers the lock
assertion for the nested calls.

**Decision (owner, 2026-11-01): go-task is installed outside the project, like uv.**
Contributors install uv and go-task, by any method; `uv tool install go-task-bin` is
recommended. They run `task <name>`. `go-task-bin` leaves the `dev` group, and
`Taskfile.yml`'s `version: '3.54'` is the minimum go-task it accepts. *Run*: a `version`
above the installed go-task stops with "Schema version (3.99.0) is greater than the current
version of Task", so no other file pins it. Every command inside still runs through `uv run`.
CI installs go-task the same way.

## Findings

### Agent hooks

**F1. A uv older than `required-version` refuses every shell command (P1).**

- *What we do.* `.claude/settings.json` runs `uv run --script .claude/hooks/guard_push.py`
  before every `Bash` call. In this session the hook also ran before `Monitor` calls. The Stop
  and PostToolUse hooks start the same way.
- *What happened.* uv applied `[tool.uv] required-version = ">=0.13.6"` from the project's
  `pyproject.toml` to the PEP 723 script as well. It stopped with "Required uv version
  `>=0.13.6` does not match the running version `0.11.32`", and Claude Code refused the call.
  The guard's body never ran, so its own cloud short-circuit (`CLAUDE_CODE_REMOTE=true`
  returns 0) could not help. Every shell and Monitor call in this session was refused. This
  review was written with read-only tools and committed through the GitHub API.
- *Why it was not caught.* `.claude/cloud/setup.sh` handles exactly this case, but the
  environment's setup script did not run it, or ran it and failed. The repository cannot
  enforce that; the hook command can.
- *The Stop hook has the same failure, and no way out.* `stop_hook_active` is checked inside the
  script, which never starts.
- *Recommendation.*
  1. Start the hooks with `uv run --no-config --script ...`. `--no-config` "disables the
     discovery of any persistent configuration" (*paraphrase*, uv "Configuration files"), and
     the project's `[tool.uv]` table is such configuration. *Run*: with uv 0.11.32,
     `uv run --script .claude/hooks/guard_push.py` exits 2, which Claude Code treats as a
     block; with `--no-config` it runs and exits 0. The hook scripts need nothing from the
     project's uv settings: they are stdlib-only (`dependencies = []`), and they start their
     own `uv run` for project tools.
  2. Put the guard's cloud short-circuit in the command string:
     `[ "$CLAUDE_CODE_REMOTE" = true ] || uv run ...`. That removes the failure where it is
     most likely, since cloud images lag uv releases, and saves a uv start on every Bash call.

  *Applied:* recommendations 1 and 2 are in `.claude/settings.json`. Editing the
  PreToolUse command restored this session's shell without a restart. With a stale uv, the
  Stop hook now starts, and its own `uv run task check:fast` fails with the version message.
  It blocks once, and `stop_hook_active` lets the next stop through instead of looping.
  3. Let `post_edit` and `stop_check` fail open, with a message, when their launcher fails.
     Only the push guard should fail closed, and only in local sessions.
  4. Keep `setup.sh`, and make sure the environment's setup script calls it.

### narwhals and the frame boundary

The boundary follows narwhals' documented conventions where they matter most (*source*,
narwhals `docs/` on `main`; `uv.lock` has 2.28.0):

- `import narwhals.stable.v2 as nw` with `narwhals>=2.0` is what the backwards-compatibility
  page asks of a library: "If you'd like to use `narwhals.stable.v2`, make sure to require at
  least `narwhals>=2.0`".
- `from_native`/`to_native` rather than `@narwhalify`, which the typing page rates "better at
  preserving type hints".
- Eager frames only, with numpy inside.

**F2. Frame types are erased to `Any` at the boundary (P1).**

- *What we do.* Twelve `# noqa: ANN401` comments in `src/` exist to annotate frames as `Any`:
  - `attach`, `with_columns` and `to_native` return "the user's own frame type" as `Any`.
  - `new_frame`, `Coordinates.to_frame` and `Status.to_frame` return `Any`.
  - Five helper functions in `survey/read.py` and `workspace/_io.py` take "a narwhals frame"
    as `Any`.
  - `_nw_dtype` returns a dtype class as `Any`.

  `Survey.data` is `Any` too.
- *What upstream documents.* The typing page's pattern for a function that takes a frame and
  returns one of the same type (*source*, `docs/api-reference/typing.md`):

  ```python
  def func(df_native: IntoDataFrameT) -> IntoDataFrameT:
      df = nw.from_native(df_native, eager_only=True)
      return df.select(b=nw.col("a")).to_native()
  ```

  `narwhals.stable.v2.typing` exports `IntoDataFrameT`.
- *Consequence.* `attach` is the call that hands results back into the user's own frame. It
  tells type checkers and editors nothing: `sspy.attach(df, coords)` is `Any` whether `df` is
  pandas or polars. Each new frame-returning function copies the pattern: the S2 plan's
  Task 4 adds `Survey.to_frame() -> Any` and `_frames.take_rows(frame: object, ...)`.
- *Recommendation.*
  - `attach(frame: IntoDataFrameT, result) -> IntoDataFrameT`, and the same for
    `with_columns`, whose pandas branch needs one `cast`. `to_native` takes
    `nw.DataFrame[IntoDataFrameT]` and returns `IntoDataFrameT`.
  - `new_frame` and `Coordinates.to_frame` get two `@overload`s: `like: None = None` returns
    `pd.DataFrame`, and `like: IntoDataFrameT` returns `IntoDataFrameT`.
  - `Status.to_frame` returns `pd.DataFrame`, and `Survey.data` is annotated `pd.DataFrame`.
    Import pandas under `TYPE_CHECKING`: the import contract already allows that, because
    `exclude_type_checking_imports = true`.
  - The "a narwhals frame" parameters become `nw.DataFrame[Any]`, through an alias
    that `_frames` exports. narwhals is not a forbidden import; only pandas, polars and pyarrow
    are.
  - `_nw_dtype` returns `type[nw.dtypes.DType]`.

  The comments then go, and ty checks the boundary.

**F3. Use narwhals-native nulls in `with_columns`, not a pandas branch (P2, owner decision).**

- *What we do.* `with_columns` builds pandas columns itself (`_pandas_column`), so that a
  masked row holds `pd.NA` in a nullable `boolean` or `Int64` column. For other backends it
  uses `nw.new_series`.
- *What upstream documents.* narwhals has no null model of its own. It gives one API over
  each backend's, and "the classic pandas dtypes make no distinction" between null and NaN
  (*source*, `docs/concepts/null_handling.md`).
- *Decision.* Build masked columns one way for every backend, with
  `nw.when(valid).then(...)`, and let each library represent nulls as it does. *Run*: on
  classic pandas dtypes a masked boolean becomes `Object` and a masked integer `Float64`;
  polars keeps `Boolean` and `Int64` with nulls. The `_pandas_column` branch and its
  `isinstance(native, pd.DataFrame)` test go. Pandas users get `ssp_included` as an object
  column where a row matched nothing. The spec does not fix that dtype; only the docstring
  and `test_with_columns_valid_mask_gives_nulls` change.
- *A trap to pin with a test.* `nw.new_series([True, None], dtype=nw.Boolean,
  backend="pandas")` returns `[True, False]`, silently. Masked data must never go through
  `new_series` with a dtype.
- *Unchanged.* The datetime normalisation in `_value_text` stays. narwhals warns that formats
  differ across tools, so the normalisation has to be ours. `new_frame` and `read_text_table`
  stay pandas, per the decision above.

**F4. `to_eager` classifies a refusal by catching an undocumented `TypeError` (P3).**

`to_eager` calls `from_native(..., eager_only=True)` and catches `TypeError`. It then calls
`from_native(obj, pass_through=True)` to tell a lazy frame from something that is not a frame.
That narwhals raises `TypeError` in the first call is in its source, not its docs. `pass_through`
is documented ("`True`: pass object through as-is"). One `pass_through=True` call, followed by
`isinstance` checks against `nw.DataFrame` and `nw.LazyFrame`, uses only documented behaviour
and calls narwhals once. The `DuplicateError` catch also rests on source alone; keep it, with
its existing test as the guard.

### matplotlib

**F5. `ax=None` builds a pyplot-free `Figure()`, against matplotlib's documented library
pattern (Owner, P2; decide before the next plot function).**

- *What we do.* With `ax=None`, `plot.scatter` draws on `Figure().add_subplot()` and touches
  no pyplot state (spec §6.15, decision B7). Jupyter shows nothing unless the cell ends in
  `.figure`.
- *What upstream documents.* The "Matplotlib Application Interfaces" page describes
  third-party libraries' data-object `plot` methods. Its example draws on the current Axes
  unless one is supplied, and returns it. The page also notes that "Many libraries also allow
  their ``plot`` methods to accept an optional *ax*" (*source*,
  `galleries/users_explain/figure/api_interfaces.rst` at v3.10.7).
  - matplotlib recommends building a `Figure` without pyplot for web servers, where pyplot's
    references to open figures leak memory (*paraphrase*, gallery example "Embedding in a web
    application server").
  - matplotlib 3.11 adds `plt.figure(fig)` to register a standalone figure with pyplot
    (*paraphrase*, What's New 3.11); our floor is 3.10.7.
- *What it costs, all in the repo today.*
  - Every tutorial cell, the quick-start and every docs example must end in `.figure`. A docs
    test enforces it (spec §6.15), and `test_install_page_says_the_notebook_cell_shows_the_plot`
    checks the install page's wording about it.
  - The quick-start is two programs: the script saves, the notebook cell ends in `.figure`.
  - `plt.show()` in a user's script shows nothing, because the figure is not registered with
    pyplot.
  - `[[tool.ty.overrides]]` switches off `unresolved-attribute` for
    `docs/user_guide/_quickstart.py`. It exists because `Axes.figure` is typed
    `Figure | SubFigure` and `SubFigure` has no `savefig` (*source*,
    `lib/matplotlib/artist.pyi` at v3.10.7).
  - M1 still has five plot functions to write (`density`, `fraction_contour`, `joint`,
    `likert`, `radar`), and each inherits the convention.
- *What it buys.* No pyplot state (AGENTS.md: no hidden global state), and no growing figure
  registry in a long-running process. But the GUI (FR-14.7) and any web service will pass their
  own `ax` from their own canvas, so they never take the `ax=None` path. The callers who do
  are notebook and script users, and they expect pyplot behaviour.
- *Recommendation.* Reopen B7 before the next plot function is written. Follow the
  documented pattern, with one change: when `ax=None`, create a new figure with
  `plt.subplots()` rather than reusing `plt.gca()`, so the circumplex frame is never drawn twice
  on a reused Axes. Then:
  - The `.figure` convention, its docs test, the wording test and the second quick-start
    variant can go.
  - The ty override can go too. The quick-start can save through pyplot, or through
    `ax.get_figure(root=True)`, which is typed `Figure | None` (*source*, same stub).
  - Tests need an autouse fixture that calls `plt.close("all")`. Without it, pyplot's
    open-figure warning (`figure.max_open_warning`, default 20, *paraphrase*) becomes a
    failure under `filterwarnings = ["error"]`.
  - `test_no_pyplot_figure_created` is inverted.

  If the owner keeps B7, nothing changes, but the cost list above should be recorded beside the
  decision.

**F6. Group colours are hard-coded, so matplotlib's own styling routes are ignored (P2).**

- *What we do.* `scatter` passes `color=` from `mpl.color_sequences["petroff10"]` for every
  group, unless the caller passes a colour keyword.
- *What upstream documents.*
  - Without a colour argument, `Axes.scatter` takes "the next color of the Axes' current
    'shape and fill' color cycle", which "defaults to rcParams['axes.prop_cycle']"
    (*paraphrase*, scatter docstring).
  - matplotlib's convention for changing colours is the property cycle and style sheets,
    applied globally or with `style.context`. A package can ship its own style sheet, which
    users load as `plt.style.use("package.style_name")` (*paraphrase*, customizing guide).
  - matplotlib has shipped a `petroff10` style sheet since 3.10.0, our floor (*paraphrase*,
    What's New 3.10).
- *Consequence.* FR-8.3 says users customise "through the conventions of the underlying
  plotting library rather than a bespoke styling system". Today `plt.style.use(...)`,
  `mpl.rc_context(...)` and an Axes the user prepared with `ax.set_prop_cycle(...)` have no
  effect on group colours. Only the per-call `color=` keyword does.
- *Recommendation.*
  - Stop passing `color=`, so each group's `scatter` call takes the Axes' next colour: the
    matplotlib-native route.
  - Give the colour-blind-safe default as a style. When soundscapy creates the figure, it
    creates the Axes inside `mpl.style.context("petroff10")`. An `ax` the user passes keeps
    its own cycle.
  - Keep the marker cycle manual, since `scatter` does not take markers from the property
    cycle.
  - Group colours stay stable across plots, because every group, empty or not, still makes one
    `scatter` call. *Run*: an empty `scatter` call takes a colour from the cycle, and the
    next call gets the following one.

  The owner decides whether a user's global style should beat petroff10 when `ax=None`. FR-8.3
  reads as yes.

**F7. `MPLBACKEND=Agg` is set for every task instead of for the tests (P3).**

- *What we do.* `Taskfile.yml` sets `env: {MPLBACKEND: Agg}` globally (house-style).
- *Consequence.* The library never uses pyplot, so the variable affects only tests that import
  pyplot, and the docs kernel.
  - It is missing whenever the tests run outside `task`: `uv run pytest` directly, or an IDE's
    test runner.
  - It is silently overridden when the caller's shell sets `MPLBACKEND`. In go-task, "a
    variable already present in the process environment takes precedence over Taskfile env
    values" (*paraphrase*, taskfile.dev "Environment").
  - It is present for `task docs`, where the Jupyter kernel inherits it (**verify** whether
    that displaces ipykernel's inline default).
- *Recommendation.* Call `matplotlib.use("Agg")` at the top of `tests/conftest.py`, and drop
  the global variable. This matters more if F5 is adopted, because the tests would then create
  pyplot figures on every CI platform.

### Logging

**F8. The library logger's level bypasses the application's logging configuration
(Owner, P2).**

- *What we do.* `_log.py` sets the `soundscapy` logger to INFO and adds `_DefaultHandler`,
  which writes to stderr unless the root logger has handlers (spec §6.3.9, owner decision 25,
  NFR-13).
- *What upstream documents.*
  - The Logging HOWTO advises libraries to add no handler other than `NullHandler`, because
    "the configuration of handlers is the prerogative of the application developer who uses
    your library" (*source*, CPython `Doc/howto/logging.rst`, "Configuring Logging for a
    Library").
  - When a record propagates, it goes "directly to the ancestor loggers' handlers - neither the
    level nor filters of the ancestor loggers in question are considered" (*paraphrase*,
    `Logger.propagate`). CPython 3.13's `Logger.callHandlers` confirms this: it checks only
    each handler's level.
- *Consequence.* The handler defers to the application, but the level does not. An application
  that calls `logging.basicConfig(level=logging.WARNING)` to quiet its libraries sets the root
  logger's level, not its handler's. soundscapy's INFO summaries still pass the soundscapy
  logger, and the application's own root handler prints them. Only `sspy.set_log_level(...)`
  silences them.
- *Recommendation.* Keep the visible default: NFR-13 requires it, and the HOWTO's
  `NullHandler` route cannot give it. Make the deferral complete instead. When the root logger
  has handlers and `set_log_level` has not been called, drop records below the root logger's
  effective level. That is a `logging.Filter` of about ten lines, tested with `basicConfig` in
  a subprocess. The alternative is to document the behaviour in the `set_log_level` docstring.
  §6.3.9 is normative, so either change goes through the spec.

### pydantic and tomlkit

pydantic is used as its docs intend:

- one frozen, closed base model;
- per-field `Strict*` types where TOML input must not be coerced;
- `BeforeValidator` and `PlainSerializer` for the pair types;
- ruff's `runtime-evaluated-base-classes`, so `TC` fixes never break a model.

`validate()` unwraps soundscapy errors that pydantic wrapped. pydantic converts `ValueError`
into field errors, and our errors subclass `ValueError`, so the unwrap is a consequence of our
error hierarchy rather than a workaround.

**F9. `_tables_last` probably duplicates what tomlkit already does (P3).**

- *What we do.* `survey/mapping.py::_tables_last` moves plain keys ahead of tables before
  `tomlkit.dumps`.
- *What upstream does.* When building a document (not when parsing one), tomlkit's
  `Container.append` inserts a plain key/value before the first table, via
  `_get_last_index_before_table` (*source*, tomlkit `main`, `tomlkit/container.py`).
- *Recommendation.* **Verify** that the 0.13.1 floor behaves the same: delete the helper and
  run `test_toml_round_trip.py` under the `lowest` job's resolution. If it passes, delete the
  helper for good. Separately, `scripts/make_isd_subset.py::_lines` can return
  `tomlkit.items.Array` instead of `Any  # noqa: ANN401`.

### Taskfile and uv

The double `uv run` layer is discussed under the challenged suggestions above. Two further
points:

**F10. `sp-repo-review` is the one unpinned tool in the toolchain (P2).**

- *What we do.* Both the `repo-review` task and the `repo: local` prek hook run
  `uvx 'sp-repo-review[cli]' .`, which takes whatever release is newest that day.
- *Consequence.* House-style requires `task repo-review` to exit 0. A new sp-repo-review
  release that adds a check can turn it red with no change in this repository, and the failure
  cannot be reproduced on an older checkout. House-style's prek section says to pin every
  `rev`, and the local hook escapes that rule.
- *What upstream documents.* sp-repo-review ships a pre-commit hook,
  `repo: https://github.com/scientific-python/cookie`, `id: sp-repo-review`, pinned by `rev`
  (*paraphrase*, sp-repo-review on PyPI; repo-review docs).
- *Recommendation.* Replace the local hook with that one, still at `stages: [manual]`, so that
  `prek auto-update` bumps it like every other hook. Then point the task at the same hook
  (`uv run prek run sp-repo-review --stage manual --all-files`; prek's flag is `--stage`),
  so a single pin serves both.

**F11. `check:fast` stops at the first failing step (P3).**

`cmds: [task: lint, task: typecheck]` runs the two in turn and stops at the first failure.
`deps` "run concurrently by default", and `output: group` buffers each task's output until it
finishes (*paraphrase*, taskfile.dev "Dependencies" and "Output"). With
`deps: [lint, typecheck]` and `output: group`, the Stop hook would get lint and type errors in
one pass, and a little sooner. Keep `check` sequential, so the tests run only once lint and
types pass.

### CI

**F12. The release job holds the publishing token while it runs the whole dev toolchain (P2).**

- *What we do.* `release.yml` has one job with `id-token: write`. It runs `uv sync --locked`,
  then `uv run --locked task check` (every dev and test dependency executes code), then
  `uv build` and `uv publish`.
- *What upstream documents.* uv's packaging guide recommends `uv build --no-sources` before
  publishing, so the package builds as it would without `tool.uv.sources`. Its workflow example
  also splits building from publishing, and gives `id-token: write` to the publish job only
  (*paraphrase*, docs.astral.sh/uv/guides/package/).
- *Recommendation.* Use two jobs. The first runs the check, builds with `--no-sources`, and
  uploads `dist/`. The second runs in the `pypi` environment, is the only job with `id-token:
  write`, and downloads `dist/` and publishes it. `--no-sources` changes nothing today, since
  only great-docs (a docs-group dependency) has a source, but it keeps the published build
  independent of uv-only settings.

**F13. Lock assertions are made per command rather than per job (P3).**

`UV_LOCKED` is "equivalent to the --locked command-line argument" (*paraphrase*, uv environment
variables). Set `UV_LOCKED: "1"` at workflow level, and unset it in the `lowest` job, which must
re-resolve. That covers the `uv run` calls nested inside the Taskfile, which run without
`--locked` today, and makes the scattered `--locked` flags unnecessary. setup-uv already reads
`required-version` from `pyproject.toml` when it is given no `version` input (*paraphrase*,
setup-uv), so CI and local uv agree.

**F14. The prek cache key leaves out the prek version (P3).**

The `hooks` job keys its cache on `.pre-commit-config.yaml` alone. prek's CI docs say cache
correctness depends on "prek version, platform, config, and hook revisions" (*paraphrase*, prek
CI docs). prek's version comes from `uv.lock`, so add `hashFiles('uv.lock')` to the key. Keep
`uv run prek` rather than the official `j178/prek-action`, so `uv.lock` stays the one pin. *Applied*
with the move to `prek.toml`: the key is `hashFiles('prek.toml', 'uv.lock')`.

### ruff, ty, pytest, import-linter

Statements about ruff, ty and pytest in this part were read from the sources of each tool's
docs at the versions in `uv.lock`: ruff 0.17.3, ty 0.0.95 and pytest 9.2.0 (*source*).

**F15. Configure ruff for lazy matplotlib instead of suppressing it (P2).**

- *What we do.* Three `# noqa: PLC0415  # matplotlib is imported lazily` comments in `plot/`.
  The lazy-import rule itself is enforced by import-linter at `task lint`, and by a subprocess
  test.
- *What upstream documents.* PLC0415 "will ignore import statements configured in
  `lint.flake8-tidy-imports.banned-module-level-imports` if the rule
  `banned-module-level-imports` [TID253] is enabled". TID253 "checks for module-level imports
  that should instead be imported lazily (e.g., within a function definition, or an
  `if TYPE_CHECKING:` block)". `select = ["ALL"]` already enables it, and it matches
  submodules.
- *Recommendation.* Add `[tool.ruff.lint.flake8-tidy-imports] banned-module-level-imports =
  ["matplotlib"]` and delete the three comments. *Run*: with that setting, ruff reports all
  three comments as unused (`RUF100`). ruff then enforces the rule at edit time,
  through the PostToolUse hook, as well as at `task lint`. When `audio` and `predict` arrive,
  add their heavy dependencies to the same list; plan §5 says those are always imported
  lazily.

**F16. `allow-star-arg-any` exists for the `**kwargs` pass-through (P3).**

`plot/circumplex.py` carries `**kwargs: Any,  # noqa: ANN401` for keywords forwarded to
`Axes.scatter`. ruff's `lint.flake8-annotations.allow-star-arg-any` controls "whether to
suppress `ANN401` for dynamically typed `*args` and `**kwargs` arguments", and every plot
function will forward keywords the same way. Set it, and delete the comment. *Run*: with the
option set, ruff reports the comment as unused.

- The other ANN401 comments are the frame annotations of F2. Those should be fixed, not
  configured away.
- The four `PLR2004` comments on `2` should stay. A narrow, commented ignore is the house rule,
  and allowing 2 everywhere would weaken it.

**F17. `[tool.ty.environment]` restates what ty derives (P3).**

- When `python-version` is unset, ty reads `project.requires-python`.
- When `root` is unset, "the project root (`.`) is always included", and `./src` is included
  when it exists and is not a package (ty configuration reference).

So `python-version = "3.11"` and `root = ["./src", "."]` both repeat the defaults, including
the `.` that the comment says `tests._helpers` needs. *Run*: `ty check` passes with the
table deleted. Delete the table. Keep
`[tool.ty.src] exclude`, and keep the override until F5 is settled. House-style's "set the
Python version and source roots only" becomes "set only what ty cannot derive". ty already
reports unused suppressions (`unused-ignore-comment`, a warning by default), so the
`ty: ignore` comments in the tests stay honest.

**F18. `from tests._helpers import ...` relies on undocumented pytest behaviour (P2).**

- *What we do.* Five test modules import shared constants from `tests/_helpers.py`. There is
  no `tests/__init__.py` and no `pythonpath` setting, and the import mode is `importlib`.
- *What upstream documents.* One of importlib mode's listed drawbacks: "Testing utility modules
  in the tests directories (for example a tests.helpers module containing test-related
  functions/classes) are not importable" (pytest "pytest import mechanisms and sys.path").
- *Consequence.* It works today through the way pytest names and registers the modules it
  imports by path. That is not a documented contract, and a pytest release could break it.
  The S2 plan adds a second helper module, `tests/_downloads.py`, the same way.
- *Recommendation.* Add `pythonpath = ["."]`, a documented setting, so that `tests` is
  importable as a namespace package from the root. The src layout still keeps the working
  tree's `soundscapy` off the path, because it sits under `src/`. The alternative, turning the
  constants into `conftest.py` fixtures, changes five modules for no gain.

**F19. pytest 9 has one switch for strictness (P3).**

- `strict = true` (pytest 9.0) "enables" `strict_config`, `strict_markers`,
  `strict_parametrization_ids` and `strict_xfail`.
- The docs add: "you should only enable strict mode if you use a pinned/locked version of
  pytest". `uv.lock` pins it.
- It replaces `--strict-markers`, `--strict-config` and `xfail_strict = true`, and adds strict
  parametrisation ids.
- `[tool.pytest]`, with native TOML types, has been supported since 9.0 beside
  `[tool.pytest.ini_options]`. The docs do not prefer either.

*Run*: `pytest -o strict=true --co` collects all 461 tests, so strict parametrisation ids break
nothing today. Adopt `strict = true`; the `minversion = "9.2"` already in place covers it.
Moving to the new
table is optional. House-style's pytest bullet changes with it.

**F20. import-linter pre-registers ignores for imports that do not exist (P3).**

The two forbidden contracts list `soundscapy._frames -> polars` and `-> pyarrow`, though
neither import exists. They also list `soundscapy.plot -> matplotlib`, though only its
submodules import matplotlib. They then set `unmatched_ignore_imports_alerting = "none"` to
silence the result. That setting also hides any ignore that goes stale later. List only the
ignores that match today, and drop the setting. Whoever adds `import pyarrow` to `_frames` then
adds its ignore line in the same reviewed commit.

### prek

*Applied* (owner request, after the first draft of this review): the hooks are configured in
prek's native `prek.toml`, and `.pre-commit-config.yaml` is gone.

- *Why.* `repo: builtin` already made the YAML file unusable by the standard `pre-commit`
  tool: builtin configs "are not compatible with the standard pre-commit tool" (prek docs).
  So YAML bought no compatibility, and prek's docs and its vendored skill prefer `prek.toml`
  for a new setup. The file is plain TOML 1.0 (`[[repos.hooks]]` tables). The converter
  `prek util yaml-to-toml` emits TOML 1.1 multiline inline tables, which Python's `tomllib`
  cannot read.
- *uv version.* `astral-sh/uv-pre-commit` installs `uv==<rev>` in its own hook environment
  ("rev: uv version", uv's pre-commit guide). That made two uv versions,
  `required-version` and the hook's `rev`, with nothing linking them. prek has no setting that
  reads `[tool.uv]`. It finds a compatible uv on PATH for Python hook environments, or
  installs one (`PREK_UV_SOURCE`, `PREK_UV_VERSION`), but that uv only builds hook
  environments. The `uv-lock` hook is now a `repo = "local"`, `language = "system"` hook
  running `uv lock`. It uses the project's own uv, which `required-version` already checks,
  so `pyproject.toml` is the one place the uv version lives. With a uv below the floor, the
  hook fails with uv's own "Required uv version" message, the intended behaviour.
- *Cost.* repo-review looks for `.pre-commit-config.yaml`. Its PY006 now fails, and is ignored
  with a reason. Its `PC` checks are skipped, so their nine ignores went too; the `PC` checks
  that used to pass (ruff-format present, ruff shows fixes) are no longer reported.
- *Run.* `prek validate-config prek.toml` succeeds, `prek run --all-files` passes, and
  `task repo-review` exits 0.

### great-docs and Quarto

Read from great-docs at the pinned commit `af17c76`: `great_docs/assets/great-docs.default.yml`,
`great_docs/_build/quarto.py` and the user guide pages (*source*).

**F21. `docs/great-docs.yml` restates six defaults (P3).**

`parser: numpy`, `dynamic: true`, `jupyter: python3`, `user_guide: user_guide`,
`freeze: {mode: auto}` and `cli: {enabled: false}` are all great-docs' defaults. Only
`sections` and `bibliography` change anything.

- Trim the file to those two, and add `cli: {enabled: true}` when the CLI arrives (S8).
- Restating defaults would guard against upstream changing them. But the pin moves only by a
  reviewed bump, and that is the moment to read upstream's changes.
- house-style `reference.md` describes `cli` as `enabled: true`. That is the S8 intent, not
  the current file, so reword it.

**F22. Quarto comes from a route great-docs does not document (P3, keep).**

- great-docs looks Quarto up with `shutil.which("quarto")`, and calls it "a system-level
  dependency of great-docs ... not bundled with the Python package".
- Its own CI recipe (`great-docs setup-github-pages`) uses pip and
  `quarto-dev/quarto-actions/setup`.
- The repo's `quarto-cli` package from PyPI works because it puts `quarto` on PATH inside
  `uv run`.

That is a deliberate choice (spec §5.4: contributors install only uv and git), and 1.10.20
meets great-docs' minimum of 1.9.18, so keep it. But it is off upstream's documented path, so a
change in either package can break `task docs` first. Keep the spec's system-Quarto fallback in
CONTRIBUTING.md.

**F23. No `Documentation` URL (P3).**

great-docs finds the repository from `[project.urls]` (`Repository`, `Source`, `GitHub` or
`Homepage`), so `Source` is enough for source links. The site URL comes from `Documentation` or
`site_url`. Without either, the build leaves `<site-url>` placeholders. Add `Documentation`
once the site has an address.

## House rules that are misspecified

Several findings above are not agent mistakes: the code does what AGENTS.md, CLAUDE.md or the
`house-style` skill told it to do. An agent that fixes the code without the rule will see the
rule again and undo the fix. Each item below names the rule, what is wrong with it, and the
proposed wording. The owner approved them, and all ten are applied (see Status at the top).

**H1. "uv only. Never run bare `pip` or `python`" made the hooks depend on project settings
(AGENTS.md, CLAUDE.md, house-style hook paragraph).** The rule is right for agents' commands.
Applied to the hook launchers, it put every hook behind `uv run --script`, which honours
`required-version` (F1). Proposed addition to CLAUDE.md: "Hooks start with
`uv run --no-config --script`, so no project setting can stop the agent's tools. The push
guard skips uv entirely in cloud sessions." The `.claude/settings.json` part is applied.

**H2. "Fix the code, or add a narrowly scoped ignore with a comment" prefers a per-line
`noqa` to the tool's own option (AGENTS.md, Don't).** Read literally, the rule offers two
choices, and F15 and F16 show the gap between them. ruff's `banned-module-level-imports`
enforces lazy matplotlib everywhere, while three `noqa` comments only excuse three lines.
Proposed wording: "Weaken lint or type rules to make errors go away. Fix the code. Where the
tool has a documented option for the pattern (a module that must be imported lazily,
forwarded `**kwargs`), use the option. Otherwise add a narrowly scoped ignore with a comment
giving the reason."

**H3. `PLR0913` is ignored globally (house-style, ruff ignores).** The stated reason is that
"scientific functions legitimately take many keyword arguments". *Run*: at ruff's default
`max-args = 5`, one function in `src/` trips the rule, `provenance.envelope`, which already
carries a comment giving its reason. A global ignore for one function contradicts the skill's
own "prefer a per-file ignore to a global one". The M1 plot signatures (spec §6.15) will exceed
five, so the proposal is to keep a ceiling rather than none: drop `PLR0913` from `ignore` and
set `[tool.ruff.lint.pylint] max-args` to the largest signature the spec fixes, with that as
the comment.

**H4. "`env: {MPLBACKEND: Agg}` is set globally" (house-style, Taskfile).** The variable
misses `uv run pytest` and IDE runners, loses to a value already in the shell, and leaks into
the docs kernel (F7). Proposed: "The test suite sets matplotlib's backend in
`tests/conftest.py`; the Taskfile sets no environment."

**H5. "`[tool.ty]`: set the Python version and source roots only" (house-style, other tool
sections).** ty derives both (F17, *run*). Proposed: "Set only what ty cannot derive. Today that
is `[tool.ty.src] exclude` and any per-file override, each with a comment."

**H6. The pytest bullet lists `--strict-markers`, `--strict-config` and `xfail_strict = true`
(house-style).** pytest 9 replaces the three with `strict = true` (F19, *run*). Proposed:
"`strict = true` (pytest ≥ 9.0, pinned by `uv.lock`)", with the rest of the bullet unchanged.

**H7. The skill specifies the one unpinned tool (house-style, Taskfile table and prek
section).** The `repo-review` row says `uvx 'sp-repo-review[cli]' .`, and the prek section
describes it as a local hook. That contradicts the same section's "pin every `rev`" (F10).
Proposed: the upstream `sp-repo-review` hook at `stages: [manual]`, with the task running
`uv run prek run sp-repo-review --stage manual --all-files`.

**H8. "Coverage gate `fail_under = 80`; raise it, never lower it" (house-style).** *Run*:
coverage at `94f594e` is 99.09%. A gate nineteen points below the real figure would let a fifth
of the package lose its tests before CI noticed. The rule already says "raise it". Proposed:
`fail_under = 95` now, reviewed at each milestone, with the reason that the gate sits just
below the current figure.

**H9. great-docs keys and the plot-display convention (house-style `reference.md`).** The
keys table invites restating defaults, and its `cli` row describes `enabled: true`, the S8
intent, as if it were current (F21). Proposed: list only the keys the site sets, and mark
`cli` "from S8". The "Plot display" bullet (`ax.figure` at the end of every cell) stands or
falls with F5.

**H10. prek specifics (house-style, prek section).** *Applied*: house-style now names
`prek.toml`, TOML 1.0 tables, and a local `uv lock` hook in place of `astral-sh/uv-pre-commit`.

Rules that hold up under the same reading, and should stay:

- the Python floor;
- `uv_build`;
- numpy docstrings with a `References` section;
- dependency groups for tooling;
- `uv run task` as the entry point (see the challenged suggestions above);
- prek's builtin hooks;
- `filterwarnings = ["error"]` with scoped exceptions;
- the src layout.

## What is already idiomatic, and should stay

So that a later tidy-up does not undo them:

- **narwhals.** One boundary module, `narwhals.stable.v2`, `eager_only=True` and numpy inside.
  An import-linter contract keeps pandas and polars out of everything else, and the
  frame-agnostic tests run against both pandas and polars.
- **matplotlib.** The `ax` contract returns the Axes, never calls `show()`, and forwards
  `**kwargs` to the artist. matplotlib is imported lazily, so `import soundscapy` stays quick
  and quiet. Tests assert on artist data, never on pixels.
- **uv.**
  - Dependency groups for tooling (not extras), with `default-groups`.
  - `uv_build`, and a committed lock.
  - `setup-uv` with its cache, and a `lowest-direct` job proving the floors.
  - Trusted publishing, with a tag and version check and no cache in the publishing job.
- **ruff.** `select = ["ALL"]` with a commented reason for every ignore, and
  `runtime-evaluated-base-classes` for pydantic models.
- **prek.**
  - The `builtin` hygiene hooks, so there is nothing to download.
  - actionlint from PyPI.
  - ty kept out of the commit hooks.
  - Every `rev` pinned, apart from F10.
- **great-docs.**
  - `user_guide`, plus custom `sections` for tutorials and papers.
  - A project-wide bibliography.
  - Quarto freeze, with `docs/_freeze/` committed.
  - Shared quick-start code pulled in with `{{< include >}}` rather than copied.
- **pytest.**
  - Strict markers and config, and `filterwarnings = ["error"]`.
  - A `-m` default in `addopts` that `test:reference` and `test:network` override, with a test
    that proves it.

## Suggested order

House-style dictates several of the settings above (the Taskfile's `MPLBACKEND`, ty's
environment table, pytest's flags, the great-docs keys). Change house-style in the same PR as
each setting, or the next agent will put the setting back.

1. **One tooling PR, with no behaviour change:** F10, F13 to F21, with H1 to H10 changing the
   rules in the same PR. Each is a few lines. `task check` and `task repo-review` must still
   pass. F1's hook commands are already applied.
2. **Frame typing, before S2's Task 4 adds `Survey.to_frame` and `take_rows`:** F2, with F3
   and F4 if wanted.
3. **Owner decisions, before the next plot function:**
   - F5 and F6 together, since both change what `ax=None` creates. F7 follows F5.
   - F8, which needs a change to spec §6.3.9.
4. **Before the first tag:** F12.
5. **Whenever the file is next touched:** F9, F11, F22, F23.
