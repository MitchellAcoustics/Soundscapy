---
title: pandas-only frame boundary - Plan
type: refactor
date: 2026-11-27
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# pandas-only frame boundary - Plan

## Goal Capsule

- **Objective:** A user who passes pandas frames to soundscapy gets exactly the results, hashes and columns they got before, a user who passes any other frame gets an error that tells them how to convert it, and the package installs and imports without narwhals or polars.
- **Means:** Replace the narwhals layer in `src/soundscapy/_frames.py` with direct pandas code, and move its callers onto it (KTD1, KTD4).
- **Authority:** owner decision #2 as reversed on 2026-11-27 (`design/greenfield_architecture_plan.md` §4.10 and §11 row 47), then spec draft 0.6 §6.3.8 and §12 (`design/specs/2026-10-31-phase0-m1-design.md`), then this plan. Where this plan and the spec disagree, the spec wins and this plan is corrected.
- **Execution profile:** One PR, on branch `m1-pandas-only` cut from the tip of `soundscapy-rewrite` (at or after `997a891`, the merge of PR #171), targeting `soundscapy-rewrite`. The AGENTS.md review sequence closes it: `task check`, `/simplify`, `task check`, `/code-review`.
- **Stop conditions:** Stop and report if the pinned hash `FRAME_VECTOR`, or any hash or null representation pinned in U1, changes. Stop if a pandas-only rule cannot reproduce a narwhals result that an existing test pins, rather than editing the test to match.
- **Who finishes:** An agent builds and commits. In a local session a hook blocks `git push`, so the owner pushes and opens the PR.

---

## Product Contract

### Summary

The frame boundary becomes pandas-only. narwhals leaves the runtime dependencies and polars leaves the test group. Every public function that takes a frame accepts a pandas `DataFrame` and nothing else, and `Coordinates.to_frame` always returns pandas. All of S1's pandas behaviour, including every hash, stays the same.

### Problem Frame

S1 put narwhals behind `_frames` so that a polars user would get polars frames back. PR #171 (merged 2026-11-27) then went further: it typed `attach` and `to_frame` with narwhals' `IntoDataFrameT`, so the caller's frame type comes back, and it made nulls in added columns follow narwhals. On 2026-11-27 the owner reversed the narwhals decision (plan §4.10). No persona needs native polars output. The boundary was S1's largest module, and its cost falls on every contributor and every extension recipe. S2 is planned against the new spec §6.3.8, and S4 adds pyarrow and the workspace, so each slice built on narwhals raises the cost of removing it. The rework has to land before S2 starts.

### Requirements

**Behaviour for pandas callers**

- R1. For pandas input, every S1 public function returns the same values, dtypes, columns, null representations and hashes as on `soundscapy-rewrite` after PR #171.
- R2. `Coordinates.to_frame` always returns a pandas frame and no longer takes `like`.
- R3. `attach` accepts a pandas frame and returns a new pandas frame, typed `pandas.DataFrame`. The user's frame and its index are left unchanged. Added boolean columns stay nullable `"boolean"` with `pd.NA` for an unmatched row, so `df[df["ssp_included"]]` keeps working.

**Refusing other frames**

- R4. Each public function that takes a frame raises `TypeError` for anything that is not a pandas `DataFrame`. For a polars or Arrow frame, the message tells the user to call `.to_pandas()` first.
- R5. A pandas frame with repeated column names still raises `ColumnCollisionError` naming the repeats.

**Dependencies and layering**

- R6. narwhals is not a runtime dependency and polars is not a test dependency. `import soundscapy` succeeds with narwhals, polars and pyarrow all absent.
- R7. Only `soundscapy._frames` imports pandas or pyarrow, and no module imports polars (import contract, spec §5.2.1).
- R8. Provenance stops recording a narwhals version.

**Records**

- R9. Docstrings and ADRs no longer say that polars is supported or that results match across libraries.

### Key Decisions

- **pandas is the one frame type at the public surface** (session-settled: user-directed — chosen over keeping narwhals at the boundary or deciding later before S4: narwhals' cost fell on every contributor and recipe, and no persona needs native polars output). Governs R1 to R8.
- **Other frames are refused, not converted** (session-settled: user-approved — chosen over converting polars or Arrow input to pandas at intake: conversion would keep a second library in the boundary, and adding it later is cheap when a user asks). Governs R4.
- **`to_frame(like=)` is removed outright, with no deprecation step** (session-settled: user-approved — chosen over deprecating it first: nothing has been released, and plan §6 lets alphas break anything). Governs R2.

### Success Criteria

- `FRAME_VECTOR` in `tests/test_frames.py`, and every hash U1 pins, are byte-identical before and after the change.
- `task check` passes with narwhals and polars absent from `uv.lock`.

### Scope Boundaries

- No change to S2 work, the CLI, mapping suggestion, or any science code.
- No pyarrow or parquet work; that is S4.
- Considered and not built: a polars-to-pandas conversion at intake. It would mean another library inside the boundary, and nobody has asked for it. A user request would change that (plan §8, deferred row G3-09).
- Considered and not built: a deprecation shim for `to_frame(like=)`. Nothing has been released that could depend on it.

#### Deferred to Follow-Up Work

- Landing the design documents from the brainstorm branch (PRD 1.0, plan 0.6, spec 0.6, S2 plan amendment) on `soundscapy-rewrite`. This PR's authority sits in them (see Open Questions).

### Open Questions

- Gate on merging: the design documents that authorise this change (spec 0.6 §6.3.8 and §12, plan 0.6 §4.10) are uncommitted on branch `claude/compound-engineering-brainstorm-519d9f`, which is rebased onto `997a891` (PR #171 merged). It should merge into `soundscapy-rewrite` before this PR, or this PR should carry those files. The owner decides which.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **`_frames` works on pandas directly; narwhals is removed.** The entry point is a `require_frame` check (spec §6.3.8), which returns the pandas frame itself. Readers take a pandas frame and use pandas and numpy only. Implements the first Key Decision; cites R1 to R7.
- KTD2. **Callers annotate frames as `pandas.DataFrame`, importing pandas only under `TYPE_CHECKING`.** This is the pattern PR #171 set in `survey.read`, `survey.status` and `distribution.coordinates`. The import contract excludes type-checking imports (`exclude_type_checking_imports = true` in `pyproject.toml`), so pandas still runs only inside `_frames`. The narwhals aliases `EagerFrame` and `IntoDataFrameT` go. Cites R2, R3, R7.
- KTD3. **The column classification matches what narwhals did for pandas input.** These rules decide `columns_float`, `column_text`, `frame_hash` and `attach`'s id text. If they move, hashes move.
  1. Numeric means an integer, unsigned or float dtype, either numpy or pandas nullable (`Int64`, `Float64` and the like). A boolean dtype is never numeric. Decide this with pandas' dtype predicates rather than numpy dtype kinds, so that pyarrow-backed integer and float dtypes, which narwhals counted as numeric, stay numeric once S4 adds pyarrow.
  2. A boolean column, `bool` or nullable `boolean`, hashes as bool only when it holds no nulls. Otherwise it hashes as text.
  3. Every other dtype is text from each value's Python form: object, `string`, pandas 3 `str`, categorical, datetime, timedelta, period. This is the rule `_value_text` already implements, including truncating a `Timestamp` to whole microseconds.
  4. Missing is NaN, `None`, `pd.NA` or `NaT`, found with `isna` before any value is read.
  5. "Float" in `attach`'s id rule means the float dtypes of rule 1 (numpy and nullable).
  6. A column that `with_columns` adds, at a row where `valid` is False, holds a null as PR #171 left it. A float column holds NaN. An integer column becomes `float64` with NaN. A boolean column is always the nullable `"boolean"` dtype, masked or not, with `pd.NA` for a null. U1 pins the text-column case.

  U1 pins the results on `soundscapy-rewrite` after PR #171, so that any drift fails a test. Cites R1, R3.
- KTD4. **Refusal is one check in `require_frame`, with a message chosen by the object's type.** An object whose type comes from the `polars` or `pyarrow` modules gets the `.to_pandas()` message. Anything else gets "expected a pandas DataFrame, got <type>". Nothing imports polars or pyarrow to do this, so the check works when neither is installed. Implements the second Key Decision; cites R4.
- KTD5. **Duplicate column names are checked in `require_frame`.** pandas allows them and narwhals refused them, so the check moves into soundscapy to keep R5's error. Cites R5.
- KTD6. **The `frame_lib` fixture runs over two pandas forms: default dtypes and nullable dtypes.** This replaces the polars parameter, so missing-value handling stays tested on both kinds of column (session-settled: user-approved — chosen over plain pandas only: nullable integers are what a null in an integer id produces under `convert_dtypes`, and narwhals handled them). Cites R1.

### Implementation constraints

- `pandas-stubs` is already in the dev group, so `ty` checks the new pandas calls. Ignores stay narrowly scoped, each with a reason (AGENTS.md).
- The pandas floor stays 2.2.1, and the `lowest` CI job runs it. pandas 3 with copy-on-write and the `str` default runs in the main matrix. Both must pass the same tests (spec §6.3.8).
- `readonly_copy`, `TextTable` and `read_text_table` already work on pandas and numpy only. They carry over unchanged.

### Sequencing

U1 runs first, against the current narwhals code, so that the pandas rules of KTD3 are checked against the old results. U2 changes `_frames` and every caller in one commit, because the internal signatures change shape and a half-migrated tree would not pass `task check`. U3 then removes the dependencies and narrows the contracts. U4 updates the records.

---

## Implementation Units

### U1. Pin current pandas behaviour

- **Goal:** Pin, as tests, the results that pandas-only code has to reproduce, before anything changes.
- **Requirements:** R1, R3, R5.
- **Dependencies:** none.
- **Files:** `tests/test_frames.py`, `tests/workspace/test_attach.py`.
- **Approach:**
  1. Add pandas-only hash tests for each dtype family in KTD3: nullable `Int64` with a null, `Float64`, `boolean` with and without nulls, `string`, pandas 3 `str` where available, datetime at nanosecond precision, categorical, timedelta.
  2. Assert equality between the pandas forms that must agree. For timedelta and any family with no partner, pin a literal hash.
  3. Add `attach` cases for a nullable `Int64` id column holding a null, and a `Float64` id column.
  4. Pin `with_columns`' null representation for each added column kind (KTD3 rule 6), including the text column, which no test pins today.
- **Execution note:** Characterization first. Every new test must pass on the narwhals implementation as PR #171 left it, before U2 starts. A test that fails now records a narwhals behaviour to check with the owner, not a target to change.
- **Patterns to follow:** `test_frame_hash_normalisation` and `test_frame_hash_vector` in `tests/test_frames.py`; `test_attach_integer_ids_made_float_by_a_null` in `tests/workspace/test_attach.py`.
- **Test scenarios:**
  - `{"i": pd.array([1, None, 3], dtype="Int64")}` hashes equal to the same content with `float64` and NaN.
  - `pd.array([0.5, None], dtype="Float64")` hashes equal to `float64` `[0.5, nan]`.
  - Nullable `boolean` `[True, False]` hashes equal to numpy `bool` `[True, False]`. With a null it hashes equal to an object column `[True, None]` and differs from the null-free form.
  - A `string`-dtype column `["a", None]` hashes equal to an object column `["a", None]`.
  - A `datetime64[ns]` value at `...00.000000500` hashes equal to the same instant truncated to microseconds.
  - A timedelta column `[pd.Timedelta("1s"), None]` hashes to a literal pinned from the current code.
  - `attach` on a frame whose `RecordID` is nullable `Int64` with one null aligns the other rows and gives nulls on the null row.
  - `attach` on a `Float64` id column holding whole numbers matches ids read as `"1"`, `"2"`.
  - `with_columns` with one masked row adds a float column with NaN there, an integer column as `float64` with NaN, a boolean column as `"boolean"` with `pd.NA`, and a text column whose null value is pinned as found.
- **Verification:** The new tests pass on the unchanged code, and `FRAME_VECTOR` is untouched.

### U2. pandas-only `_frames` and its callers

- **Goal:** `_frames` and every module that uses it work on pandas alone, with no narwhals import.
- **Requirements:** R1 to R5, R9 (docstrings).
- **Dependencies:** U1.
- **Files:**
  - `src/soundscapy/_frames.py`
  - `src/soundscapy/survey/read.py`
  - `src/soundscapy/survey/status.py` (import list only)
  - `src/soundscapy/workspace/_io.py`
  - `src/soundscapy/distribution/coordinates.py`
  - `tests/conftest.py`
  - `tests/test_frames.py`
  - `tests/workspace/test_attach.py`
  - `tests/distribution/test_coordinates.py`
  - `tests/test_frame_types.py`
- **Approach:**
  1. In `_frames`, replace `to_eager` and `to_native`, with their overloads, by `require_frame` (KTD1, KTD4, KTD5). Remove `EagerFrame`, `DEFAULT_BACKEND` and `_LAZY_MESSAGE`.
  2. Re-implement `columns_float`, `column_text`, `_texts`, `_hash_chunks` and `frame_hash` on pandas under the KTD3 rules.
  3. Rewrite `with_columns` on pandas with the null rules of KTD3 rule 6, and drop `_nw_dtype`. Drop `new_frame`'s `like` and its overloads.
  4. Give `_frames` a small helper that tells whether a column is a float column, so that `workspace._io._id_text` stops reading dtypes itself. Its name is the builder's choice.
  5. In the callers, replace `to_eager(...)` with `require_frame(...)`, and retype frame parameters as `pandas.DataFrame` (KTD2). Drop `like` and the `IntoDataFrameT` overloads from `Coordinates.to_frame` (R2) and `attach` (R3).
  6. Rewrite docstrings that mention polars or narwhals, including `attach`'s and `column_text`'s ("the same for pandas and polars").
  7. In tests, reparametrise `frame_lib` (KTD6). Delete the lazy-frame and polars round-trip tests, and `test_new_series_with_dtype_turns_a_pandas_null_boolean_false`, which pins a narwhals behaviour. In `tests/test_frame_types.py`, keep the pandas `assert_type` checks and delete the polars ones. Turn the polars comparisons in the hash tests into the pandas pairs that U1 established. Replace `test_attach_polars` and `test_to_frame_like_polars` with refusal tests.
- **Patterns to follow:** spec §6.3.8 for the function list and behaviour; the `TYPE_CHECKING` pandas imports in `src/soundscapy/survey/status.py`.
- **Test scenarios:**
  - Covers R4: `require_frame` on an object of a stub class whose `__module__` is `polars.dataframe.frame` raises `TypeError` naming `.to_pandas()`. Do the same for a stub from `pyarrow.lib`.
  - Covers R4: `require_frame` on a dict, a list and a pandas `Series` raises `TypeError` naming the type.
  - Covers R4: `attach`'s frame argument refuses a polars-module stub with the same message. In S1 it is the only public function that takes a user frame; `read_survey` takes a path.
  - Covers R5: a pandas frame built with two `"a"` columns raises `ColumnCollisionError` listing `'a'`.
  - `columns_float` over the default and nullable `frame_lib` forms gives the same read-only matrix, with NaN for each null.
  - `columns_float` refuses a nullable `boolean` column and a `string` column with `CodingError` naming the column.
  - `column_text` gives `""` for a null in object, `string` and nullable `Int64` columns.
  - `with_columns` keeps a non-default pandas index, and leaves the input frame unchanged.
  - Covers R2: `Coordinates.to_frame()` returns a pandas frame, and passing `like=` raises `TypeError`.
  - Covers R3: `attach` to `survey.data` still aligns every row and returns pandas, and `assert_type` gives `pandas.DataFrame`.
  - Covers R3: `attach` with one unmatched row gives `ssp_included` as `"boolean"` with `pd.NA` there, and `df[df["ssp_included"]]` drops that row.
  - Every U1 test and `FRAME_VECTOR` pass unchanged.
- **Verification:** Nothing in `src/` imports `narwhals`. All of U1's tests pass unchanged. `task check` passes, with narwhals still installed but unused.

### U3. Dependencies, contracts and provenance

- **Goal:** narwhals and polars leave the project's dependency set, and the guards prove they are not needed.
- **Requirements:** R6, R7, R8.
- **Dependencies:** U2.
- **Files:**
  - `pyproject.toml`
  - `uv.lock`
  - `tests/licences.toml`
  - `src/soundscapy/provenance.py`
  - `tests/test_provenance.py`
  - `tests/test_package.py`
- **Approach:**
  1. Remove narwhals from `dependencies` and polars from the `test` group with `uv remove`. The commit message gives the reason, as AGENTS.md requires.
  2. Leave the import contract as PR #171 set it: only `soundscapy._frames -> pandas` is ignored, and polars stays in `forbidden_modules`.
  3. Remove narwhals from `tests/licences.toml`.
  4. Remove `"narwhals"` from provenance `_LIBRARIES`, and give the `extra_libraries` docstring an example other than polars, such as `"openpyxl"`.
  5. Extend `test_import_without_polars_or_pyarrow` in `tests/test_package.py`, and its twin in `tests/test_frames.py`, so that they also block narwhals.
- **Patterns to follow:** `test_import_without_polars_or_pyarrow` in `tests/test_package.py`; the dependency-change commit convention in AGENTS.md.
- **Test scenarios:**
  - Covers R6: a subprocess with `sys.modules` entries for `narwhals`, `polars` and `pyarrow` set to `None` imports `soundscapy` and runs `sspy.project(sspy.load_isd(subset=True))` without error.
  - Covers R8: `provenance["libraries"]` holds numpy, pandas and pydantic, and has no `narwhals` key.
  - The licence tests pass, with every runtime dependency still listed.
- **Verification:** `uv.lock` has no `narwhals` or `polars` package entry. `task lint` keeps all import contracts. `task check` passes.

### U4. Records

- **Goal:** The ADRs and the remaining docs describe the pandas-only boundary.
- **Requirements:** R9.
- **Dependencies:** U2.
- **Files:**
  - `design/adr/0001-coordinates-carry-settings.md`
  - `design/adr/0002-response-id.md`
  - `design/adr/0004-h1-hashing.md`
- **Approach:** Add a dated amendment line to each ADR, saying that pandas is the only frame library as of 2026-11-27 (plan §4.10) and naming what changed. In ADR 0001 that is "keeps pandas out of the other modules". In ADR 0002 it is "in pandas or polars". In ADR 0004 it is the cross-library claim and the test it names. Leave each decision's text standing; the amendment line says what no longer holds.
- **Test expectation:** none -- these are documentation records with no executable content.
- **Verification:** `rg -i "polars|narwhals" design/adr src` finds only the amendment lines and the import contract's forbidden list.

---

## Verification Contract

| Gate | Command | Proves |
|---|---|---|
| Full check | `task check` | lint, import contracts, `ty`, the full default test run (spec §5.3) |
| Frame tests | `uv run pytest tests/test_frames.py tests/workspace/test_attach.py tests/distribution/test_coordinates.py` | R1 to R5 at the boundary |
| Isolation | `uv run pytest tests/test_package.py` | R6 |
| Docs | `task docs` | tutorials and the quick-start still run on the pandas-only package |
| Lock | inspect `uv.lock` after `uv remove` | narwhals and polars are gone (R6) |

CI on `soundscapy-rewrite` PRs runs the three-OS matrix and the `lowest` job (pandas 2.2.1). Both must be green before the PR is marked ready.

---

## Definition of Done

- Every unit's Verification holds, and `task check` passes.
- `FRAME_VECTOR` and every hash pinned in U1 are unchanged.
- `src/` has no narwhals import. `pyproject.toml` and `uv.lock` list neither narwhals nor polars.
- `/simplify` and `/code-review` have run on the final diff and their findings are addressed (AGENTS.md Review).
- No dead code from the narwhals path remains: no unused helper, no `pass_through` branch, no lazy-frame message constant.
