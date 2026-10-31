# M1 Slice S1 (Thin Slice, Contracts First) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the load-bearing contracts of M1 (errors, value types, hashing, provenance, logging, the frame boundary, status, the data registry) and one thin path on them, so that `sspy.plot.scatter(sspy.project(sspy.load_isd(subset=True), angles="equal"), by="location")` returns Axes whose offsets equal the projected xy.

**Architecture:** Bottom-layer modules (`errors`, `_model`, `provenance`, `_log`, `_frames`, `_data`) come first, each with its tests. `instrument` adds `Scale` and `Circumplex` read from a shipped, hash-pinned TOML file. `projection.formulas` is a pure numpy kernel. `survey` reads a CSV through a `SurveyMapping` into a `Survey`; `projection.project` turns it into `distribution.Coordinates`; `workspace.attach` and `plot.scatter` consume it. Layers follow the import contracts already in `pyproject.toml` (spec §5.2.1).

**Tech Stack:** Python 3.11+, numpy, pydantic 2, narwhals (`stable.v2`), pandas, polars (tests), matplotlib (lazy), tomlkit and `tomllib`, pytest and hypothesis; uv, ruff, ty, import-linter.

**Spec:** `design/specs/2026-10-31-phase0-m1-design.md` (draft 0.4), §9 row S1, with §6.1 to §6.5, §6.8.3, §6.9, §6.10, §6.15, §6.16, §6.19 to §6.21, §7 and §11.4. §11.4 overrides earlier spec sentences it contradicts. Read each section a task cites before starting the task; this plan does not repeat normative blocks it can cite.

## Global Constraints

- uv only: `uv run`, `uvx`, `uv add`; never bare `python` or `pip`. Never run a command that waits on stdin.
- Done for every task means `uv run task check` passes (lint, import contracts, ty, tests with branch coverage at 80% or more).
- Runtime dependency floors are spec §6.19's: numpy 1.26, pandas 2.2, narwhals 2.0, pydantic 2.7, matplotlib 3.10, tomlkit 0.13. Every dev and test dependency gets a floor at the current release (§11.4).
- Public API names, signatures and module placement follow spec §6.1 and the sections cited per task. Where S1 ships a narrower signature than the final M1 one, the task says which later slice widens it; the narrower form must be a prefix of the final one (same names, same order, keyword-only).
- Every optional keyword defaults to `_UNSET` (the `_Unset` sentinel in `soundscapy._model`), never `None` (§6.3.4). Drawing arguments of plot functions are the exception (§6.15).
- Only `soundscapy._frames` imports pandas, polars or pyarrow; only `soundscapy.plot` imports matplotlib, and only inside functions; only `soundscapy.provenance` imports `hashlib` (§6.3.5, §6.3.8, §6.15).
- Every public object has a numpy docstring; every function implementing a procedure has a `References` section with a `[@key]` that exists in `docs/references.bib` (house-style).
- Floats in FrozenModels are `StrictFloat` and reject ±inf and NaN (§6.2.1), so TOML files write floats with a decimal point (`1.0`, not `1`).
- Every float comparison in a test has an explicit tolerance with a comment giving the reason.
- `filterwarnings = ["error"]`: every expected warning is asserted with `pytest.warns`.
- Commit `uv.lock` with every dependency change and give the reason. Commit messages end with the attribution lines your session supplies.
- Branch `soundscapy-rewrite`. Do not reformat files you do not otherwise change.

## Review Focus

1. **A user's mapping names a column the CSV does not have** (a typo, or a renamed export). Expected: `MappingError` naming each missing column and the mapping key, before any coding runs. Pinned in Task 8 (`test_missing_mapped_column_raises`).
2. **A ragged CSV** (a row with fewer cells than the header, as some exporters trim trailing empties, or more cells). Expected: short rows padded with `""` (Plan call); a long row raises `FileFormatError` naming its row number. Duplicate header names raise `FileFormatError`. Pinned in Task 3 (`test_read_text_table_ragged_rows`, `test_read_text_table_duplicate_header`).
3. **Numeric text in the forms the ISD really uses** (`"4"`, `"4.0"`, `" 4 "`, blank). Expected: 4.0, 4.0, 4.0, NaN; a comma decimal with `decimal="."` raises `CodingError` listing the text. Pinned in Task 7 (`test_apply_coding_isd_forms`).
4. **`attach` to a frame the user re-read with `pd.read_csv`**, where `RecordID` is now an integer column and the rows are re-sorted. Expected: alignment by id still matches every row, in the frame's order; if no frame row matches, `AlignmentError` (Plan call), never a frame of nulls. Pinned in Task 11 (`test_attach_to_reread_frame_with_int_ids`, `test_attach_no_match_raises`).
5. **`angles=` given an angle set before S3 ships them** (`"satp@1.5"`, `"aletta-2024@1"`). Expected: `SettingsError` naming `angles` and saying only `"equal"` is available in this build, not a pydantic traceback. Pinned in Task 10 (`test_project_rejects_unknown_angles`).

---

### Task 1: Dependencies, `errors`, `_model`

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Create: `src/soundscapy/errors.py`, `src/soundscapy/_model.py`, `tests/licences.toml`, `tests/test_licences.py`, `tests/test_errors.py`, `tests/test_model.py`

**Interfaces:**
- Produces:
  - `soundscapy.errors`: every class of spec §6.3.10, both tables. Errors subclass `SoundscapyError` and the built-in in the table; `__init__(self, message: str, *, hint: str = "")`; `str(err) == message`; `err.hint`. Warnings subclass `SoundscapyWarning(UserWarning)`.
  - `soundscapy._model.FrozenModel` (normative, §6.2.1).
  - `soundscapy._model._Unset` (an `enum.Enum` with one member) and `_UNSET: Final = _Unset.UNSET`. Plan call: an enum, so `x is _UNSET` narrows for ty.
  - `soundscapy._model.Citation(FrozenModel)`: `citation`, `doi_or_url`, `licence`, `date`, `notes`, all `StrictStr` defaulting to `""` (§6.3.7). Plan call: it lives here because instrument, datasets and later angle sets all use it.
  - `soundscapy._model.StrPairs`: `Annotated[tuple[tuple[str, str], ...], ...]` with a before-validator turning a table (`dict`) into pairs and a serializer turning pairs back into a table in `mode="json"` (§6.4.4 table form). A second alias for `tuple[tuple[str, tuple[str, ...]], ...]` (scale labels) and one for `tuple[tuple[str, StrictFloat], ...]` (value maps) follow the same pattern. Plan call: one reusable mechanism for every key-to-value field, not a validator per model.
  - `to_toml(model: FrozenModel) -> str` and `from_toml(cls: type[M], text: str) -> M`: `model_dump(mode="json", by_alias=True)` written by tomlkit; `tomllib.loads` then `model_validate`. `from_toml` re-raises `pydantic.ValidationError` as `SettingsError` (message keeps pydantic's field paths).
  - Namespace constants (§6.3.3): `SSP_PREFIX = "ssp_"`, `COL_RESPONSE_ID`, `COL_INCLUDED`, `COL_REASONS`, `COL_FLAGS`, `COL_LANGUAGE`, `COL_ANGLE_SET` (values `ssp_response_id` ... `ssp_angle_set`), `axis_column(axis_id: str) -> str` returning `f"ssp_{axis_id}"`. Plan call: they live in `_model` because `distribution` (which may not import `survey`) and `workspace` both need them and `_model` is already in the bottom layer; a new module would need a layer-contract edit.

- [ ] **Step 1: Add the S1 dependencies**

```bash
uv add 'numpy>=1.26' 'pandas>=2.2' 'narwhals>=2.0' 'pydantic>=2.7' 'matplotlib>=3.10' 'tomlkit>=0.13'
uv add --group dev 'pandas-stubs>=3.1.0.261029'
uv add --group test 'polars>=2.1.0'
```

Reasons (commit message): numpy, arrays in the core; pandas, frames read from paths (2.2 is the Anaconda course floor); narwhals, the frame boundary (`stable.v2`); pydantic, settings and value types (2.7 is the spec floor); matplotlib, `plot.scatter` and `petroff10` (3.10 is the first with it); tomlkit, writing mapping TOML deterministically; pandas-stubs, so ty checks `_frames`; polars, the second frame library in the frame tests. Plan call: scipy, contourpy, pyarrow, pooch, tqdm, click and openpyxl are not added in S1, because nothing in S1 imports them; S5 (scipy, contourpy), S4 (pyarrow, openpyxl export), S7 (openpyxl read), S8 (click) and S9 (pooch, tqdm) add them.

- [ ] **Step 2: Write the failing tests**
  - `tests/licences.toml`: spec §6.19's table as `[licences] numpy = "BSD-3-Clause ..."`, every row (all 13, so later slices only add to `pyproject.toml`). `tests/test_licences.py::test_every_runtime_dependency_has_a_licence_entry` (each name in `project.dependencies` is a key) and `::test_installed_metadata_declares_a_licence` (`License-Expression`, `License` or a `License ::` classifier is non-empty for each).
  - `tests/test_errors.py`: one parametrised test over a table of (class, built-in) pairs copied from §6.3.10, asserting `issubclass(cls, SoundscapyError)` and `issubclass(cls, builtin)`, `str(cls("m", hint="h")) == "m"`, `.hint == "h"`; each warning subclasses `SoundscapyWarning` and `UserWarning`; `test_no_error_name_shadows_a_builtin` (no name in `errors.__all__` is in `dir(builtins)`); `DataIntegrityError` exists and `IntegrityError` does not.
  - `tests/test_model.py`: `FrozenModel` subclass rejects `inf`, `-inf`, `nan` in a `StrictFloat` field, rejects extra keys, is frozen; `StrPairs` field validates from `{"a": "x"}` and from `(("a", "x"),)`, equal values, and `model_dump(mode="json")` gives `{"a": "x"}`; `to_toml`/`from_toml` round-trip a model holding a `StrPairs`, a tuple and a `"none"`-literal field to an equal model; `from_toml` on a bad value raises `SettingsError`; `axis_column("ISOPleasant") == "ssp_ISOPleasant"`; `_UNSET is _Unset.UNSET`.

- [ ] **Step 3: Run them to verify they fail**: `uv run pytest tests/test_errors.py tests/test_model.py tests/test_licences.py -v`, expected FAIL (import errors).

- [ ] **Step 4: Implement `errors.py` and `_model.py`** with the interfaces above. Note for every later task: pydantic wraps a `ValueError` raised inside a validator (which every soundscapy error is) into `ValidationError`. Validators raise plain `ValueError` with a clear message; public entry points (`from_toml`, `read_mapping`, `read_survey`, `project`) catch `ValidationError` and raise the soundscapy error the spec names.

- [ ] **Step 5: Run the tests to verify they pass**, then `uv run task check`. Expected: green.

- [ ] **Step 6: Commit** `feat: errors, FrozenModel and S1 runtime dependencies` (the body lists each dependency with its reason; `uv.lock` included).

---

### Task 2: `provenance` hashing and envelope, `_log`

**Files:**
- Create: `src/soundscapy/provenance.py`, `src/soundscapy/_log.py`, `tests/test_provenance.py`, `tests/test_log.py`

**Interfaces:**
- Consumes: `FrozenModel`, `HashSchemeError` (Task 1).
- Produces (Plan call on the helper names: the spec fixes "one helper in `provenance`" and its byte rules, not the names; these are the one helper's entry points):
  - `HASH_PREFIX: Final = "h1-sha256:"`
  - `hash_bytes(data: bytes | Iterable[bytes]) -> str` (the only `hashlib` user; an iterable is hashed as its concatenation, so `_frames.frame_hash` can stream)
  - `hash_file(path: str | os.PathLike[str]) -> str` (raw bytes)
  - `canonical_json(value: pydantic.BaseModel | JSONValue) -> bytes` (§6.3.5 row 2; models via `model_dump(mode="json", by_alias=True)`; Plan call: `by_alias=True` everywhere, because the `schema` field needs an alias, see Task 7)
  - `hash_json(value: pydantic.BaseModel | JSONValue) -> str`
  - `check_hash(value: str) -> str`: returns the 64-hex digest; any other prefix or a malformed digest raises `HashSchemeError`
  - `envelope(*, operation: str, settings: FrozenModel, input_hash: str, mapping_hash: str = "", data_refs: Sequence[Mapping[str, str]] = (), extra_libraries: Sequence[str] = (), upstream_run: str = "", **extra: object) -> dict[str, object]`: the §6.3.6 normative keys. `libraries` holds the installed versions of numpy, scipy, pandas, pyarrow, narwhals, pydantic and contourpy, plus each name in `extra_libraries`. Plan call: a listed library that is not installed is left out (scipy, contourpy and pyarrow arrive in later slices); once they are runtime dependencies every key is present. `extra` keys must belong to `OPTIONAL_KEYS` (`input_hashes`, `upstream`, `quality_skipped`, `profile`, `profile_inherited`, `overrides`, `response_id_source`, `read`, `id_columns`), else `ValueError`, so a typo cannot add a key. `created_utc` is `datetime.now(UTC)` as ISO 8601 with `Z`, seconds precision. Plan call: `profile` and `overrides` are not written by S1 results, because profiles arrive in S3 ("the mapping can only gain keys", §6.3.6). `id_columns` is a Plan call, see Task 10.
  - `show_versions() -> str`: multi-line `key: value` for soundscapy, python, platform and each library; never prints.
  - `soundscapy._log.logger` and `set_log_level(level: int | str) -> None`, the normative mechanism of §6.3.9 verbatim.

- [ ] **Step 1: Write the failing tests**
  - `tests/test_provenance.py::test_golden_vectors`, parametrised, the §6.3.5 table exactly: `hash_bytes(b"") == "h1-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"`; `hash_json({}) == "h1-sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"`; `hash_json({"b": [1.5, "é"], "a": 1}) == "h1-sha256:2bb202b35963ad9e57f87e5828e072cc53f4720ecd10a7d625f850411a31c4c6"`; `hash_json({"clip": "none"}) == "h1-sha256:7c457b553a03102b7220154eb95650e02d2840d02f397a2d226afba1fa85a1dd"`.
  - `::test_non_finite_json_names_the_key`: `hash_json({"a": float("inf")})` raises `ValueError` matching `"a"`; `{"b": [1.0, float("nan")]}` matches `b[1]`.
  - `::test_hash_file_equals_hash_bytes` (tmp file), `::test_model_and_its_dump_hash_alike`, `::test_check_hash` (`"sha256:..."` and `"h2-sha256:..."` raise `HashSchemeError`; a valid one returns the hex).
  - `::test_hashlib_only_in_provenance`: walk `src/soundscapy/**/*.py`; only `provenance.py` contains `import hashlib` or `from hashlib`.
  - `::test_envelope_keys`: the normative key set of §6.3.6 that S1 writes (`soundscapy_version`, `python_version`, `platform`, `libraries`, `hash_scheme == "h1"`, `data_refs`, `created_utc` ending `Z`, `operation`, `input_hash`, `mapping_hash`, `settings_hash`, `upstream_run`); `libraries` has numpy, pandas, narwhals, pydantic; `settings_hash == hash_json(settings)`; an unknown extra key raises `ValueError`.
  - `::test_show_versions_returns_and_never_prints` (capsys empty).
  - `tests/test_log.py`: `caplog` receives an INFO record from `logger` (propagation); `set_log_level("WARNING")` drops INFO (restore in teardown); `test_default_handler_writes_to_stderr_without_root_handlers` and `test_default_handler_defers_to_root_handlers`, each in a subprocess (`sys.executable -c`), because pytest installs root handlers: the first logs once to stderr; the second calls `logging.basicConfig()` first and the message appears exactly once.

- [ ] **Step 2: Run them to verify they fail**: `uv run pytest tests/test_provenance.py tests/test_log.py -v`.
- [ ] **Step 3: Implement** `provenance.py` and `_log.py` to the interfaces. `canonical_json` walks the value first to name the first non-finite float's path.
- [ ] **Step 4: Run the tests to verify they pass**; `uv run task check` green; `uv run pytest tests/test_package.py` still shows a silent import.
- [ ] **Step 5: Commit** `feat: h1 hashing, provenance envelope and the soundscapy logger`.

---

### Task 3: `_frames` boundary

**Files:**
- Create: `src/soundscapy/_frames.py`, `tests/conftest.py`, `tests/test_frames.py`

**Interfaces:**
- Consumes: `hash_bytes`, `canonical_json` (Task 2); `CodingError`, `ColumnCollisionError`, `EncodingError`, `FileFormatError` (Task 1).
- Produces (§6.3.8; S1 subset):
  - `to_eager(obj: object) -> nw.DataFrame[Any]`
  - `columns_float(frame: nw.DataFrame[Any], names: Sequence[str]) -> NDArray[np.float64]` of shape (n, len(names)), copied, read-only
  - `column_text(frame: nw.DataFrame[Any], name: str) -> NDArray[np.object_]` of Python `str`, nulls as `""`, read-only
  - `to_native(frame: nw.DataFrame[Any]) -> Any`
  - `with_columns(frame: object, columns: Mapping[str, NDArray[Any]]) -> Any` (native in, same library out; `ColumnCollisionError` on an existing name)
  - `new_frame(columns: Mapping[str, NDArray[Any]], like: object | None = None) -> Any` (pandas, or the library of `like`)
  - `TextTable` (frozen dataclass): `frame` (pandas, every cell `str`), `row_numbers: NDArray[np.int64]` (physical record numbers, header = 1), `n_blank: int`, `delimiter: str`, `encoding: str`
  - `read_text_table(path: str | os.PathLike[str], *, encoding: str, delimiter: str) -> TextTable`. Plan call: keywords rather than §6.3.8's `options` object, because `ReadOptions` lives in `survey`, above `_frames`. The `csv` module reads every cell as text; a record whose cells are all empty after stripping is skipped and counted, keeping its number out of use (§6.3.2 rule 3); a short row is padded with `""`, a long row raises `FileFormatError` with its record number (Plan call); duplicate header names raise `FileFormatError`; a decode failure raises `EncodingError` with hint `encoding="cp1252"`.
  - `frame_hash(frame: object) -> str` (§6.3.5 row 3).
  - `Frame` type alias for typing elsewhere (`narwhals.typing.IntoDataFrame`).
- **`frame_hash` byte layout, with one Plan call.** Header and per-column rules exactly as §6.3.5. The spec's "text as UTF-8 values joined by U+001F" lets two frames collide (columns `["a"]`, `["b"]` and `["ab"]`, `[""]` give the same bytes under the same header). Plan call: each text column's bytes are preceded by their length as a little-endian `uint64`. Numeric and bool columns have fixed lengths given by the header and need no prefix. Datetimes become `datetime.isoformat()` text of the Python value (pandas `Timestamp` and polars give the same text for naive values without nanoseconds).

- [ ] **Step 1: Write the failing tests.** `tests/conftest.py` adds the `frame_lib` fixture, parametrised over `"pandas"` and `"polars"`, returning a `make(dict) -> native frame` callable (§7). In `tests/test_frames.py`:
  - `test_lazy_frame_raises_type_error`: `to_eager(pl.LazyFrame(...))` raises `TypeError` with message exactly `"lazy frames are not accepted; call .collect() first"`; a `dict`, a pandas Series and a polars Series raise `TypeError`.
  - `test_columns_float_copies_and_is_read_only` (`frame_lib`): mutating the source frame afterwards does not change the array; `arr.flags.writeable is False`; null and NaN both give NaN.
  - `test_columns_float_refuses_text_and_bool` (`frame_lib`): `CodingError` naming the column.
  - `test_column_text_nulls_are_empty`; `test_with_columns_collision` (`ColumnCollisionError`); `test_new_frame_follows_like` (pandas default, polars when `like` is polars).
  - `test_frame_hash_vector`: content `{"i": [1, None, 3], "f": [0.5, 1.5, -2.0], "t": ["a", None, "é"]}` built as pandas (where `i` becomes `float64`) and as polars (where `i` stays `Int64`) gives one hash, equal to a pinned constant `FRAME_VECTOR`. Pin it once from the implementation (§11.3 leaves the value to the builder) and assert `frame.dtypes` show the two different integer representations, so the test proves normalisation.
  - `test_frame_hash_normalisation`: pandas `int64` [1, 2, 3] equals pandas `float64` [1.0, 2.0, 3.0] equals polars `Int64`; a text column `["1", "2", "3"]` differs; a bool column without nulls differs from the same column with a null; a naive datetime column hashes alike in pandas and polars; `test_frame_hash_text_columns_do_not_collide` (the `["a"],["b"]` against `["ab"],[""]` case).
  - `test_read_text_table_numbers_records`: header, two records, a blank line, one record gives `row_numbers == [2, 3, 5]` and `n_blank == 1`; a UTF-8 BOM is stripped under `utf-8-sig`; `test_read_text_table_ragged_rows`; `test_read_text_table_duplicate_header`; `test_read_text_table_bad_encoding` (`EncodingError`, hint mentions `cp1252`).
- [ ] **Step 2: Run them to verify they fail**: `uv run pytest tests/test_frames.py -v`.
- [ ] **Step 3: Implement `_frames.py`** with `narwhals.stable.v2`. narwhals `InvalidOperationError`, `ColumnNotFoundError` and `DuplicateError` are re-raised as soundscapy errors with backend-neutral messages (§6.3.8).
- [ ] **Step 4: Run the tests to verify they pass** on the locked pandas 3.x, then on the floors (the pandas 2.2 run of the S1 done-when; CI's `lowest` job repeats it). `$SCRATCH` is any directory outside the repository:

```bash
UV_RESOLUTION=lowest-direct UV_PROJECT_ENVIRONMENT="$SCRATCH/lowest-venv" uv run --python 3.11 pytest tests/test_frames.py -p no:cacheprovider
git restore uv.lock
```

Expected: both runs pass; `uv tree --depth 1` in the lowest venv shows pandas 2.2.x. Then `uv run task check` green.
- [ ] **Step 5: Commit** `feat: frame boundary with frame_hash`.

---

### Task 4: `_data` registry, `Scale`, `Attribute`, `Axis`, `Circumplex`, the outdoor instrument

**Files:**
- Create: `src/soundscapy/_data/__init__.py`, `src/soundscapy/_data/circumplexes/iso-12913-outdoor/1.toml`, `src/soundscapy/instrument/__init__.py`, `src/soundscapy/instrument/scale.py`, `src/soundscapy/instrument/circumplex.py`, `tests/test_data.py`, `tests/instrument/test_scale.py`, `tests/instrument/test_circumplex.py`
- Modify: `docs/references.bib` (first entries)

**Interfaces:**
- Consumes: `FrozenModel`, `Citation`, `StrPairs`, `to_toml`/`from_toml` (Task 1); `hash_bytes` (Task 2); `DataIntegrityError`, `DataReferenceError`, `MappingError` (Task 1).
- Produces:
  - `soundscapy._data`: `PINS: Final[Mapping[str, str]]` (posix path relative to `_data/` to `h1` hash); `read_bytes(relpath: str) -> bytes` (verifies the pin, `DataIntegrityError` naming both hashes; an unregistered path raises `DataReferenceError`); `read_toml(relpath: str) -> dict[str, Any]`; `pin(relpath: str) -> str`; `as_path(relpath: str) -> ContextManager[Path]` (verified, via `importlib.resources.as_file`). Plan call: every read verifies its pin, not only the test, so an edited installed file fails loudly at run time too.
  - `soundscapy.instrument.scale.Scale` (§6.2.2): fields as the table; `labels` uses the scale-labels pairs alias; properties `midpoint`, `half_range`, `range`; `contains(values: ArrayLike) -> NDArray[np.bool_]`.
  - `soundscapy.instrument.circumplex`: `Attribute`, `Axis`, `Translation`, `Circumplex` (§6.2.3 with §11.4: `Attribute.labels` holds the reference language only; a translation for the reference language is rejected). Plan call: `Circumplex` has `schema_name: Literal["soundscapy.circumplex"]` aliased `schema` and `schema_version: Literal[1]`, the shape angle sets and mappings use. Properties `positions -> tuple[str, ...]` and `angles_deg -> tuple[float, ...]` (Plan call: properties, since they take no argument); methods `label(position: str, language: str = "") -> str` (blank means the reference language; falls back to it for any language without a translation) and `position_for(key: str) -> str`. `Axis.labels` defaults to `()` and the shipped file leaves it empty: axis titles are the axis ids and the end labels come from the attributes (§6.2.3 last paragraph).
  - `circumplex(name: str = "iso-12913-outdoor", version: str = "1") -> Circumplex`, reading `circumplexes/<name>/<version>.toml`; an unknown pair raises `DataReferenceError` listing the shipped ones.
  - `soundscapy.instrument` re-exports `Scale`, `Circumplex`, `circumplex`.
- **`iso-12913-outdoor/1.toml` content** (§6.2.3): `name`, `version = "1"`, `reference_language = "eng"`; eight `[[attributes]]` PAQ1..PAQ8 at `0.0, 45.0, ..., 315.0` with `labels = { eng = "pleasant" }` ... `calm`; two `[[axes]]` (`ISOPleasant` PAQ1/PAQ5, `ISOEventful` PAQ3/PAQ7); `[scale]` `minimum = 1.0`, `maximum = 5.0`, `kind = "ordinal"`, `[scale.labels] eng = ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"]` (the 0.x wording, `git show origin/main:src/soundscapy/surveys/survey_utils.py`, until the owner checks Table A.1, §11.2); `[quadrants]` `"++" = "PAQ2"`, `"-+" = "PAQ4"`, `"--" = "PAQ6"`, `"+-" = "PAQ8"`; no translations (S3 adds them); `[citation]` citing ISO/TS 12913-2:2018 (the attribute set) and ISO/TS 12913-3:2019 (the 45° angles), `doi_or_url = ""` (paywalled standard), `licence = "BSD-3-Clause (this file)"`, `date = "2019"`, notes saying the scale wording awaits the owner's Table A.1 check. Plan call: the Table A.1 transcription test of §11.4 moves to S3 with the labels, because its only source in hand is the same 0.x wording.
- **`docs/references.bib`** gains `iso12913-2-2018`, `iso12913-3-2019`, `mitchell2022how` (JASA EL 2, 037201, doi 10.1121/10.0009794), `mitchell2026translation` (chapter 4 of *Soundscape Across Languages*, Springer Nature, 2026) and `isd2024` (Zenodo, doi 10.5281/zenodo.10672568), with the authors as `design/literature/README.md` and §6.20.1 give them.

- [ ] **Step 1: Write the failing tests**
  - `tests/instrument/test_scale.py`: `Scale(minimum=1.0, maximum=5.0)` has midpoint 3.0, half_range 2.0, range 4.0 (exact); `minimum >= maximum` and ordinal non-integer bounds are rejected; `contains([1, 5, 3.5, 0, 6, nan, inf])` is `[T, T, F, F, F, F, F]` for ordinal and `[T, T, T, F, F, F, F]` for interval; `int` bounds are rejected (`StrictFloat`); hypothesis `test_scale_toml_round_trip` (finite bounds, both kinds) gives an equal model and equal `hash_json`.
  - `tests/instrument/test_circumplex.py`: `circumplex()` has positions `("PAQ1", ..., "PAQ8")`, angles `(0.0, 45.0, ..., 315.0)`, axes ids `("ISOPleasant", "ISOEventful")`, scale 1..5 ordinal; `label("PAQ6") == "monotonous"`, `label("PAQ1", "deu") == "pleasant"` (fallback); `position_for("Calm") == "PAQ8"`, `position_for("PAQ3") == "PAQ3"`, `position_for("loud")` raises `MappingError`; validators reject non-contiguous positions, non-increasing angles, an angle of 360.0, a missing quadrant key, an axis naming an unknown position, a translation for `eng`; `circumplex(version="9")` raises `DataReferenceError`; `test_circumplex_toml_round_trip` (dump and reload equal, equal hash).
  - `tests/test_data.py`: `test_every_shipped_file_matches_its_pin`; `test_every_shipped_file_is_pinned` (every non-`.py` file under `src/soundscapy/_data/` has a `PINS` entry and every entry exists); `test_tampered_bytes_raise` (monkeypatch the reader to return altered bytes: `DataIntegrityError` whose message holds both hashes); `test_unregistered_path_raises`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement**, write the TOML, compute its pin with `uv run python -c "from soundscapy.provenance import hash_file; print(hash_file('src/soundscapy/_data/circumplexes/iso-12913-outdoor/1.toml'))"` and add it to `PINS`.
- [ ] **Step 4: Run the tests to verify they pass**; `uv run task check` green.
- [ ] **Step 5: Commit** `feat: data registry with hash pins and the ISO outdoor circumplex`.

---

### Task 5: Projection kernel and the S1 science tests

**Files:**
- Create: `src/soundscapy/projection/__init__.py`, `src/soundscapy/projection/formulas.py`, `tests/projection/test_formulas.py`, `tests/science/test_projection.py`

**Interfaces:**
- Consumes: `Scale` (Task 4), `FormulationError` (Task 1).
- Produces: `project_scores(scores: NDArray[np.float64], angles_deg: NDArray[np.float64], scale: Scale, formulation: Literal["canonical", "iso2019"]) -> NDArray[np.float64]` (§6.9.1). `scores` and `angles_deg` are (n, k) in position order; returns (n, 2) float64; rows with any NaN score return NaN. canonical and iso2019 exactly as the §6.9.1 formulas; iso2019 is computed from its own formula (not routed through canonical), so the exhaustive test compares two implementations. `iso2019` raises `FormulationError` unless the scale is 1..5 and every row's angles equal `(0, 45, ..., 315)`. A zero Σ|cos θ_i| or Σ|sin θ_i| in any row raises `FormulationError` (Plan call: fail loudly rather than divide by zero). Shape mismatches raise `ValueError`. No clipping or clamping. Plan call: the `aletta2024` formulation and its `Literal` member are S3's (§9 S3 row), so S1's `Literal` has two members. Docstring `References`: `[@mitchell2026translation]` (canonical, proof p. 23, §4.5), `[@iso12913-3-2019]` and `[@mitchell2022how]` (Eq. 2).

- [ ] **Step 1: Write the failing tests.** `tests/science/test_projection.py` has `pytestmark = pytest.mark.science` and cites its sources in docstrings:
  - `test_iso2019_hand_values`: (p, v, e, ch, a, m, u, ca) = (5,3,3,3,1,3,3,3) gives x = √2 − 1, y = 0; (4,5,2,1,2,1,4,5) gives x = (1 + 2√2)/(2 + 2√2), y = −1/(2 + 2√2) (closed forms, §11.4), both under `iso2019` and `canonical`, `pytest.approx(abs=1e-12)  # float64 sums of 8 terms; closed forms are exact`. Docstring: derived by hand from the ISO/TS 12913-3:2019 formula.
  - `test_canonical_equals_iso2019_exhaustively`: all 5^8 = 390,625 responses (`itertools.product(range(1, 6), repeat=8)` into one array), `np.testing.assert_allclose(canon, iso, rtol=0, atol=1e-12)  # spec §7.2; measured max 2.8e-16`.
  - Hypothesis properties (§7.1): strategy draws a scale (ordinal integer bounds, or interval floats in [−10, 10] with `maximum − minimum >= 0.5`), eight strictly increasing angles in `[0, 360)` (sorted unique floats rounded to 0.01°), and scores inside the scale. `test_canonical_square_bounds` (`|x| <= 1 + 1e-12`, `|y| <= 1 + 1e-12`), `test_neutral_maps_to_origin` (canonical; and iso2019 on the nominal angles), `test_constant_shift` (canonical: adding c to every item, staying in scale, moves x by `c·Σcos θ_i / (h·Σ|cos θ_i|)` and y likewise, `atol=1e-12  # float rounding of sums of 8 terms`).
  - `tests/projection/test_formulas.py`: NaN in one item gives a NaN row only for that row; `iso2019` on `Scale(1.0, 7.0)` raises `FormulationError`; `iso2019` on non-nominal angles raises `FormulationError`; shape mismatch raises `ValueError`; output is float64 with shape (n, 2); no clamping: (5,5,5,3,1,1,1,3) gives x = y = (2 + √2)/(2 + 2√2) under both formulations, radius √2·(2 + √2)/(2 + 2√2) = 1.0824 > 1 (§6.9.1), `abs=1e-12  # closed form`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement `project_scores`** with numpy float64 sums along axis 1.
- [ ] **Step 4: Run the tests to verify they pass**; `uv run task check` green.
- [ ] **Step 5: Commit** `feat: canonical and ISO 2019 projection kernel with science tests`.

---

### Task 6: `Status`, `Reason`, `Flag`

**Files:**
- Create: `src/soundscapy/survey/__init__.py`, `src/soundscapy/survey/status.py`, `tests/survey/test_status.py`

**Interfaces:**
- Consumes: `COL_INCLUDED`, `COL_REASONS`, `COL_FLAGS` (Task 1); `_frames.new_frame` (Task 3).
- Produces (§6.3.1): `Reason` and `Flag` as `enum.StrEnum` with the normative codes in the normative order; `Status` frozen dataclass (`eq=False`, `slots=True`) with `included: NDArray[np.bool_]` (read-only), `reasons: tuple[tuple[Reason, ...], ...]`, `flags: tuple[tuple[Flag, ...], ...]`. `__post_init__` checks equal lengths, `included[i] == (len(reasons[i]) == 0)`, codes unique and in enum order (else `ValueError`). `Status.all_included(n: int) -> Status` (Plan call: the constructor helper readers use). `with_reason(mask: ArrayLike, reason: Reason) -> Status`, `with_flag(mask, flag) -> Status` (new objects, codes kept unique and ordered). `counts() -> dict[str, int]`: Plan call, keys `"n"`, `"included"`, `"excluded"`, then every `Reason` and every `Flag` code (zero when absent), so the keys never depend on the data. `to_frame() -> Any`: the three status columns (`ssp_included`; `ssp_reasons` and `ssp_flags` joined with `;`, `""` when none), pandas.

- [ ] **Step 1: Write the failing tests**: the two code tuples equal §6.3.1's lists in order; `with_reason` on an already-excluded row keeps one copy; adding reasons out of order stores them in enum order; `to_frame` text for a row with two reasons is `"scale_violation;all_identical"`; `counts()` on an all-included status of 3 rows; hypothesis `test_status_invariants` (random sequences of `with_reason`/`with_flag` on random masks keep `included == (len(reasons) == 0)`, unique ordered codes, length n); arrays read-only; a mismatched `included` raises `ValueError`.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**, `uv run task check` green.
- [ ] **Step 5: Commit** `feat: per-response Status with Reason and Flag codes`.

---

### Task 7: `SurveyMapping`, `Coding`, mapping TOML

**Files:**
- Create: `src/soundscapy/survey/mapping.py`, `tests/survey/test_mapping.py`, `tests/survey/fixtures/minimal_mapping.toml`

**Interfaces:**
- Consumes: `FrozenModel`, `StrPairs` and the value-map alias (Task 1); `Scale`, `circumplex` (Task 4); `MappingError`, `CodingError`, `DataReferenceError`.
- Produces (§6.4.1, S1 subset; Plan call on scope: S1 maps attributes, ids and language, as §9 S1 says; S2 adds `source_dominance`, `overall_quality`, `appropriateness`, `who5`, `age`, `flags` and `carry`, and the full ISD mapping):
  - `SurveyMapping`: `schema_name: Literal["soundscapy.mapping"]` aliased `schema` (Plan call and a spec defect: a pydantic field named `schema` emits "shadows an attribute in parent BaseModel", a `UserWarning` that `filterwarnings = ["error"]` turns into a failure; with an alias the TOML key and the hash input stay `schema`, using `populate_by_name=True`, which pydantic 2.7 has, and `by_alias=True` on every dump), `schema_version: Literal[1]`, `instrument: StrictStr = "iso-12913-outdoor"`, `instrument_version: StrictStr = "1"`, `ids: IdSpec`, `language: LanguageSpec = LanguageSpec()`, `groups: Groups`.
  - `IdSpec`: `response`, `response_key`, `response_key_blank_ok`, `location`, `session`, `group`, `record` per §6.4.1 (all of it: the id rules are S1's). Validators: not both `response` and `response_key`; `response_key_blank_ok` a proper subset of `response_key`.
  - `LanguageSpec`: `column`, `value`, `coding` (not both `column` and `value`; every coded value matches `^[a-z]{3}$`).
  - `Groups`: `attributes: ItemGroup` (required). `ItemGroup`: `columns: StrPairs`, `scale: Scale`, `coding: Coding = Coding()`. Plan call: when `scale` is omitted a model validator fills in the circumplex scale, so the stored model always holds a concrete `Scale`.
  - `Coding` (all §6.4.1 fields) and `apply_coding(texts: NDArray[np.object_], coding: Coding, scale: Scale, *, item_keys: Sequence[str], response_ids: NDArray[np.object_]) -> NDArray[np.float64]` applying §6.4.2 steps 1 to 5 to an (n, k) text array. Plan call: the whole coding order ships in S1 (it is a few lines and `Coding` is an S1 deliverable); S2 adds the mapping-level tests for missing codes, value maps and reverse on real exports.
  - Attribute keys accept positions or reference-language labels and are normalised to positions in position order; every circumplex position must be mapped (else `ValueError` in the validator, surfaced as `MappingError` by `read_mapping` and `read_survey`). An unknown instrument raises `DataReferenceError`.
  - `read_mapping(path: str | os.PathLike[str]) -> SurveyMapping` and `write_mapping(mapping: SurveyMapping, path: str | os.PathLike[str]) -> None`, with tomlkit and the four §6.4.4 writer rules; a blank value raises `MappingError` naming the key.

- [ ] **Step 1: Write the failing tests**
  - `test_label_and_position_keys_hash_alike`: `{"pleasant": "x", ...}` and `{"PAQ1": "x", ...}` give equal models and equal `hash_json`.
  - `test_unmapped_position_raises` (`read_mapping` of a file missing `calm`: `MappingError` naming `PAQ8`/`calm`); `test_bad_language_code_raises` (`coding = { English = "ENG" }`); `test_response_and_key_together_raise`; `test_blank_value_raises_naming_key`.
  - `test_write_mapping_byte_round_trip`: `tests/survey/fixtures/minimal_mapping.toml` (identity keys, `[ids]` with the ISD four-part `response_key` and roles, `[language] column = "Language"`, `[groups.attributes.columns]` by English name in circumplex order) reads, writes and compares byte for byte; the writer emits `pleasant`, never `PAQ1`; default-valued fields (empty `coding`, the default `scale`) are omitted. Plan call: S1 writes the default scale by omission; a non-default scale is written as a `[groups.attributes.scale]` table.
  - `test_apply_coding_isd_forms`: `["4", "4.0", " 4 ", "", "NA"]` give `[4, 4, 4, nan, nan]` (exact); `"3,5"` with `decimal="."` raises `CodingError` listing `"3,5"`, its count and up to five response ids; with `decimal=","` gives 3.5.
  - Hypothesis `test_coding_order` (§7.1): for random value maps, offsets and reverse sets, the result equals value map, then offset, then `minimum + maximum − score` for reversed keys; reversing twice is the identity (exact).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**, `uv run task check` green.
- [ ] **Step 5: Commit** `feat: minimal SurveyMapping, Coding and mapping TOML`.

---

### Task 8: `read_survey` from a path, `response_id`, `Survey`

**Files:**
- Create: `src/soundscapy/survey/read.py`, `tests/survey/test_read.py`, `tests/survey/fixtures/*.csv` (small hand-written CSVs)

**Interfaces:**
- Consumes: Tasks 1 to 7 (`read_text_table`, `column_text`, `apply_coding`, `Status`, `envelope`, `hash_file`, `hash_json`, `logger`).
- Produces:
  - `ReadOptions(FrozenModel)`, S1 fields: `format: Literal["auto", "csv", "tsv"] = "auto"` (by suffix; any other suffix raises `FileFormatError`), `encoding: StrictStr = "utf-8-sig"`, `delimiter: StrictStr = "auto"`. Plan call: in S1 `"auto"` resolves to `,` for csv and tab for tsv; sniffing among `,`, `;`, tab and `|`, `xlsx`, `sheet`, `header_row` and `skip_rows` arrive in S7. `decimal` is not a field (§11.4). The resolved options are recorded.
  - `ImportSettings(FrozenModel)`: `on_scale_violation: Literal["error", "exclude"] = "error"`. Plan call: S2 adds `quality` and the `exclude` path; S1 always raises `ScaleViolationError` on a violation (§6.4.3 message and hint), so a 0..4 export never projects (an anti-goal of §3.3).
  - `SourceRef(FrozenModel)`: `name` (file name), `hash` (raw file `h1`), `read_options: ReadOptions`.
  - `Survey` frozen dataclass (`eq=False`, `slots=True`), fields of §6.5: `data`, `mapping`, `scores: dict[str, NDArray]` (S1: `"attributes"` only, (n, 8), position order, read-only), `response_id`, `ids: dict[str, NDArray]` (mapped roles only), `language`, `status`, `settings`, `source`, `provenance`. Method `n() -> int`. Plan call: `to_frame` and `select` arrive in S2 with the status table; S1 has no caller.
  - `read_survey(source: str | os.PathLike[str], mapping: SurveyMapping | str | os.PathLike[str], *, read_options: ReadOptions | Mapping[str, object] | _Unset = _UNSET) -> Survey`. Plan call: the §6.5 signature's later keywords arrive with their features (S2 `quality`, `on_scale_violation`, `settings`; S3 `profile`), and in-memory frame sources arrive in S7; a non-path source raises `TypeError` naming the S1 limitation.
  - Behaviour, in order: the `ssp_` collision check (`ColumnCollisionError`, §6.3.3); mapped columns missing from the header raise `MappingError` listing each with its mapping key; attributes coded (§6.4.2), then the scale check; ids per §6.3.2 rules 1 to 3 (rule 4 is S7's in-memory case) with `MissingResponseIdError` and `DuplicateResponseIdError` listing up to five offenders and their row numbers, and `MappingError` for a key part containing `/`; language per `LanguageSpec` (`""` when unmapped). Status: Plan call, S1 applies the one always-on rule, `incomplete_attributes` (§6.7: "always, not configurable at M1"), so a row with a missing attribute is excluded with a reason rather than included with NaN xy; S2 adds every other rule. One INFO log line: rows read, blank rows skipped, included, excluded by reason.
  - Provenance: `envelope(operation="read_survey", settings=..., input_hash=source.hash, mapping_hash=hash_json(mapping), data_refs=[{kind: "circumplex", name, version, hash}], response_id_source=..., read=resolved options dump, id_columns={role: column})`.
  - Private `_read_survey(..., data_refs: Sequence[Mapping[str, str]] = ())` that `read_survey` wraps, so `datasets.isd` can add its dataset reference (Plan call; `Survey` is frozen).

- [ ] **Step 1: Write the failing tests** (fixtures are 3 to 6 row CSVs with the eight attribute columns in the ISD's CSV order):
  - `test_reads_by_name_not_order`: attribute columns in CSV order pleasant, chaotic, vibrant, ... land in position order in `scores["attributes"]`.
  - `test_response_id_from_key` (`"L1/S1/G1/7"`, `response_id_source == "key"`), `test_response_id_from_column` (stripped), `test_response_id_from_row` (header, two rows, a blank row, one row: ids `row-2`, `row-3`, `row-5`).
  - `test_blank_key_part_raises` and `test_blank_key_part_allowed` (listed in `response_key_blank_ok`: id keeps an empty segment, `"L1/S1/G1/"`); `test_duplicate_ids_raise` (message holds both row numbers); `test_key_part_with_slash_raises`.
  - `test_ssp_column_collision_raises`; `test_missing_mapped_column_raises`; `test_scale_violation_raises` (a 0..4 file: message names the column, the value `0` and its count; hint names `offset` and `on_scale_violation="exclude"`).
  - `test_incomplete_attributes_excluded` (one blank attribute: `included` false, reason `incomplete_attributes`); `test_info_summary_logged` (`caplog`, one record at INFO); `test_survey_arrays_read_only`; `test_provenance_keys` (§6.3.6 keys of Task 2 plus `response_id_source`, `read`, `id_columns`; `input_hash == hash_file(path)`).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**, `uv run task check` green.
- [ ] **Step 5: Commit** `feat: read_survey from a CSV path with response ids`.

---

### Task 9: ISD subset generator, bundled subset, `datasets.isd.load(subset=True)`

**Files:**
- Create: `scripts/make_isd_subset.py`, `src/soundscapy/_data/datasets/isd/subset-v1.0.csv`, `src/soundscapy/_data/datasets/isd/subset-v1.0.toml`, `src/soundscapy/_data/datasets/isd/registry.toml`, `src/soundscapy/_data/datasets/isd/isd_mapping.toml`, `src/soundscapy/datasets/__init__.py`, `src/soundscapy/datasets/isd.py`, `tests/datasets/test_isd.py`, `tests/science/test_isd_subset.py`
- Modify: `src/soundscapy/_data/__init__.py` (four pins)

**Interfaces:**
- Consumes: `_read_survey`, `read_mapping` (Tasks 7, 8); `_data.as_path`, `read_toml`, `pin` (Task 4).
- Produces: `soundscapy.datasets.isd.load(*, subset: bool = False, version: str | _Unset = _UNSET) -> Survey`. `version` unset means the registry `default`. With `subset=True` only `"1.0.1-alpha.1"` is accepted, else `DataReferenceError` listing it. Plan call: `subset=False` raises `NotImplementedError` whose message says the full download arrives with `datasets.isd.fetch` (S9) and to pass `subset=True`; S2 adds `quality`, `on_scale_violation`, `settings`, S3 `profile`, S9 `locations` and the full path. The survey's `data_refs` add `{kind: "dataset", name: "isd-subset", version: "1.0.1-alpha.1", hash: pin("datasets/isd/subset-v1.0.csv")}`. `soundscapy.datasets` imports `isd`.
- **Generator** (§6.20.1): PEP 723 script (`dependencies = ["tomlkit>=0.13"]`), stdlib otherwise. `--source PATH` or a download of `https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv`; refuses unless the source sha256 is `27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735`. Keeps rows whose `LocationID` is one of the 15 §6.20.1 locations and the 40 §6.20.1 columns, cell text verbatim. Plan call: rows in source order and columns in source order (so `LCeq_L(C)` precedes `LAeq_L(A)`, as in the source header); written with `csv` (`lineterminator="\n"`, minimal quoting), UTF-8 without BOM. The sidecar `subset-v1.0.toml`: Plan call on keys beyond the normative `[citation]` block: `schema = "soundscapy.dataset_sidecar"`, `schema_version = 1`, `name = "isd-subset"`, `source_version = "1.0.1-alpha.1"`, `source_record = "10672568"`, `source_file`, `source_sha256`, `subset_hash` (`h1-sha256:` of the CSV, computed in the script with `hashlib`), `n_rows = 1552`, `locations`, `columns`, `[features] columns = [...]`, `units_verified = false`, then `[citation]` verbatim from §6.20.1. Running it twice gives identical bytes.
- **`registry.toml`**: Plan call, S1 ships the normative header (`schema`, `schema_version`, `concept_doi`, `default = "1.0.1-alpha.1"`) and only the `[versions."1.0.1-alpha.1"]` entry, every §6.20.3 key filled; its `feature_columns` are the 96 header columns after `RecordingLength`. S9 adds the `1.0.1-alpha` and `0.2.3` entries with fetch.
- **`isd_mapping.toml`**: Plan call, lives under `_data/datasets/isd/` (§11.4) and in S1 holds the §6.4.4 example's identity keys, `[ids]`, `[language]` and `[groups.attributes.columns]` only; S2 extends it to the full normative example and re-pins it.

- [ ] **Step 1: Generate the data**

```bash
curl -sS -o "$SCRATCH/isd-1.0.1-alpha.1.csv" "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv"
uv run scripts/make_isd_subset.py --source "$SCRATCH/isd-1.0.1-alpha.1.csv"
uv run scripts/make_isd_subset.py --source "$SCRATCH/isd-1.0.1-alpha.1.csv" && git diff --exit-code src/soundscapy/_data/datasets/isd/
```

Expected: the second run changes nothing. (Checked while writing this plan: the source has 3589 rows; the 15 locations give 1552, eng 1213, spa 126, nld 89, cmn 69, ita 55; 1350 rows carry `LAeq_L(A)`; 16 rows have a blank attribute.) Pin the four files in `_data.PINS`.

- [ ] **Step 2: Write the failing tests**
  - `tests/science/test_isd_subset.py` (`science` marker): the CSV matches its pin; it has 1552 rows and the 40 columns; language counts `{"eng": 1213, "spa": 126, "nld": 89, "cmn": 69, "ita": 55}`; the 15 locations exactly; 1350 non-blank `LAeq_L(A)`; the sidecar `[citation]` equals §6.20.1's block and its `subset_hash` equals the CSV's pin.
  - `tests/datasets/test_isd.py`: `load(subset=True)` returns a `Survey` with `n() == 1552`, `response_id_source == "key"`, all ids unique, the dataset `data_refs` entry; 16 rows excluded by `incomplete_attributes` (Plan call: S1 asserts only this always-on count; S2 pins the full 1552/16/0/10/8 set); `load(subset=True, version="1.0.1-alpha")` raises `DataReferenceError`; `load()` raises `NotImplementedError`; `isd_mapping.toml` round-trips byte for byte through `read_mapping`/`write_mapping` (§6.4.4).
- [ ] **Step 3: Run to verify they fail**, **Step 4: implement `datasets/isd.py`**, **Step 5: pass**; `uv run task check` green; `uv run task build` and `unzip -l dist/*.whl | grep -E 'subset-v1.0.csv|1.toml|registry.toml'` lists the data files.
- [ ] **Step 6: Commit** `feat: bundled ISD subset, its generator and load(subset=True)` (body: CC BY 4.0, record 10672568, sha256 checked).

---

### Task 10: `AngleSelection` (equal only), `ProjectionSettings`, `project`, `Coordinates`, `OriginSegment`

**Files:**
- Create: `src/soundscapy/instrument/angles.py`, `src/soundscapy/projection/project.py`, `src/soundscapy/distribution/__init__.py`, `src/soundscapy/distribution/coordinates.py`, `tests/projection/test_project.py`, `tests/distribution/test_coordinates.py`
- Modify: `pyproject.toml` (add the "distribution never imports survey" contract of §5.2.1 verbatim, now that its source module exists)

**Interfaces:**
- Consumes: `Survey`, `Status` (Tasks 6, 8); `project_scores` (Task 5); `Circumplex`, `Scale`; namespace constants; `envelope`, `hash_json`; `_frames.new_frame`.
- Produces:
  - `soundscapy.instrument.angles.AngleSelection(FrozenModel)`: `default: Literal["equal"] = "equal"`. Plan call: S3 widens `default` to `AngleSetRef | Literal["equal"]` (taking its default from the profile, `satp@1.5` under `sal-2026`) and adds `per_language`, `on_uncovered`, `on_low_confidence`; S1's default `"equal"` is provisional and the S3 change is a recorded behaviour change.
  - `soundscapy.projection.project.ProjectionSettings(FrozenModel)`: `formulation: Literal["canonical", "iso2019"] = "canonical"`, `angles: AngleSelection = AngleSelection()`.
  - `project(survey: Survey, *, formulation: Literal["canonical", "iso2019"] | _Unset = _UNSET, angles: Literal["equal"] | AngleSelection | _Unset = _UNSET) -> Coordinates`. A string other than `"equal"` (or a bad formulation) raises `SettingsError` naming the field and saying which values this build accepts. Plan call: `on_uncovered`, `on_low_confidence`, `angle_sets`, `settings`, `profile` arrive in S3 and `carry` in S2. Projects included rows only (excluded rows stay NaN); the Coordinates' status is the survey's (no projection reasons arise with equal angles); the Survey is unchanged. Docstring `References` as Task 5.
  - `soundscapy.distribution.coordinates.OriginSegment(FrozenModel)` (§6.10 fields; `n: StrictInt`; `angle_sets` here `("equal",)`; `dataset` is `SourceRef.name`).
  - `Coordinates` frozen dataclass (`eq=False`, `slots=True`): `xy`, `circumplex`, `labels`, `origin`, `settings: FrozenModel`, `provenance` (§6.10). Labels for kind `survey`: `ssp_response_id`, `ssp_included`, `ssp_reasons`, `ssp_flags`, `ssp_language`, `ssp_angle_set` (`"equal"` for projected rows, `""` for rows not projected: Plan call), then each mapped id role under the user's column name. Constructor copies every array and makes it read-only.
  - Module functions `n(coords: Coordinates) -> int` and `n_valid(coords: Coordinates) -> int` (included and finite x and y), the one definition; methods `n()` and `n_valid()` call them. `to_frame(like: object | None = None) -> Any` (labels then `ssp_ISOPleasant`, `ssp_ISOEventful`); `select(mask: ArrayLike) -> Coordinates` (origin `n` updated); `split(by: str) -> dict[str, Coordinates]`, sorted keys, `by` an id role, `"language"`, or a label column name; an unknown `by` raises `MappingError` listing the choices.
  - Plan call: `provenance["id_columns"]` (`{role: user column}`, copied from the survey) is how `split("location")` finds `LocationID`; §6.10's fields give no other place, and §6.3.6 lets the envelope gain keys. Coordinates provenance: `operation="project"`, `input_hash`, `mapping_hash` and `data_refs` copied from the survey, `settings_hash` of the `ProjectionSettings`, `response_id_source`, `read`, `id_columns`.

- [ ] **Step 1: Write the failing tests**
  - `tests/projection/test_project.py`: on a fixture CSV holding the two §7.2 hand-value responses, `project(read_survey(...), formulation="iso2019", angles="equal").xy` equals the closed forms (`abs=1e-12`, as Task 5); default formulation is canonical and gives the same values; an excluded (incomplete) row has NaN xy and `ssp_angle_set == ""`; the survey's arrays are unchanged; `test_project_rejects_unknown_angles` (`"satp@1.5"` raises `SettingsError` mentioning `angles`); `project(..., angles=AngleSelection())` equals `angles="equal"`; `coords.settings` is a `ProjectionSettings` and `provenance["settings_hash"] == hash_json(coords.settings)`; `test_project_row_order_invariance` (hypothesis permutation of the fixture's rows: xy, status and ids permute identically, exact); on the subset, `n() == 1552`, `n_valid() == 1536`, every finite x and y within `[-1, 1]`.
  - `tests/distribution/test_coordinates.py`: arrays read-only; `split("location")` on the subset has the 15 locations in sorted order and sizes summing to 1552; `split("language")` has five keys; `split("nonsense")` raises `MappingError`; `select` updates `origin[0].n`; `to_frame()` has the label columns plus `ssp_ISOPleasant`/`ssp_ISOEventful`; `to_frame(like=<polars frame>)` is polars; `OriginSegment` round-trips through TOML with an equal hash; `ProjectionSettings` and `AngleSelection` round-trip likewise.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement**, and add the contract to `pyproject.toml`. **Step 4: Pass**; `uv run task check` green (`lint-imports` reports five contracts kept).
- [ ] **Step 5: Commit** `feat: project with equal angles into Coordinates`.

---

### Task 11: `attach`

**Files:**
- Create: `src/soundscapy/workspace/__init__.py`, `src/soundscapy/workspace/_io.py`, `tests/workspace/test_attach.py`

**Interfaces:**
- Consumes: `Coordinates` (Task 10); `_frames.to_eager`, `column_text`, `with_columns`; namespace constants; `AlignmentError`, `ColumnCollisionError`; `logger`.
- Produces: `attach(frame: Frame, result: Coordinates) -> Any` (§6.16), in `workspace/_io.py`, re-exported by `soundscapy.workspace`. Appends `ssp_<axis id>` for both axes, `ssp_included`, `ssp_reasons`, `ssp_flags`, `ssp_language`, `ssp_angle_set` to a new frame in the user's library. Alignment by `provenance["response_id_source"]`: `column`/`key` join by id built from the frame's columns as §6.3.2 builds them (text, stripped, joined with `/`), keeping the frame's row order, frame rows not in the result get nulls; `row` aligns by position and needs the survey's row count. `AlignmentError` when the id columns are missing, the frame's ids are not unique, the row count differs for `row`, or (Plan call) no frame row matches. One INFO line with the alignment (`"id"` or `"position"`) and the matched count. Plan call: the per-response `Table` form of `attach` arrives with `Table` (S5/S6).

- [ ] **Step 1: Write the failing tests**: `test_attach_to_survey_data` (subset: `attach(survey.data, coords)` has 40 + 7 columns, `ssp_ISOPleasant` equals `coords.xy[:, 0]` with NaN where excluded, the user's frame is unchanged); `test_attach_to_reread_frame_with_int_ids` (`pd.read_csv` of the subset, rows reversed: every row matches and values follow the frame order); `test_attach_polars` (`frame_lib`); `test_attach_partial_frame` (half the rows: nulls nowhere, matched count logged); `test_attach_no_match_raises`; `test_attach_row_ids_by_position` and its row-count `AlignmentError`; `test_attach_never_overwrites` (frame already holding `ssp_ISOPleasant`: `ColumnCollisionError`); `test_attach_missing_key_column_raises`.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**; `uv run task check` green.
- [ ] **Step 5: Commit** `feat: attach coordinates to the user's frame`.

---

### Task 12: `plot.scatter`, the top-level API, the quick-start

**Files:**
- Create: `src/soundscapy/plot/__init__.py`, `src/soundscapy/plot/_common.py`, `src/soundscapy/plot/circumplex.py`, `tests/plot/test_scatter.py`
- Modify: `src/soundscapy/__init__.py`, `tests/test_package.py`, `docs/user_guide/_quickstart.py`, `docs/user_guide/01-install.qmd`, `tests/test_docs_structure.py`

**Interfaces:**
- Consumes: `Coordinates`, `split` (Task 10); `Circumplex.label`.
- Produces:
  - `soundscapy.plot.scatter(coords: Coordinates, *, by: str | _Unset = _UNSET, ax: Axes | None = None, label_language: str | None = None, **kwargs: Any) -> Axes` (§6.15). matplotlib is imported inside the function; type hints under `TYPE_CHECKING`. `ax=None` builds `matplotlib.figure.Figure()` and `fig.add_subplot()`, no pyplot. `by` unset or `""`: one series of the valid rows. Otherwise one `ax.scatter` per group from `coords.split(by)`, in sorted key order, labelled with the key, colours from `matplotlib.color_sequences["petroff10"]` cycled (Plan call: the subset's 15 locations reuse colours after ten; the docstring says so), and a legend. Plan call: a group with no valid rows still gets an empty labelled series, so a group keeps its colour across plots. `**kwargs` go to each `ax.scatter` call (a user `color` overrides the cycle). Returns the Axes; never calls `show()`.
  - `plot._common.circumplex_axes(ax: Axes, circumplex: Circumplex, label_language: str | None) -> None`: the ±1 square limits, equal aspect, axis lines through the origin, axis titles (the axis ids) and the quadrant labels (`circumplex.label` of each quadrant position, reference language when `None`). `check_n` is S5's.
  - `soundscapy/__init__.py`: imports `plot`, `datasets`, `workspace`, `errors` eagerly (§6.21) and exposes, in `__all__`, the S1 names of the §6.21 table: `read_survey`, `read_mapping`, `write_mapping`, `load_isd` (= `datasets.isd.load`), `Survey`, `SurveyMapping`, `Coding`, `ReadOptions`, `Circumplex`, `Scale`, `AngleSelection`, `ProjectionSettings`, `Coordinates`, `circumplex`, `project`, `attach`, `show_versions`, `set_log_level`, `__version__`, `plot`, `datasets`, `workspace`, `errors`. Later slices add the rest.
  - `docs/user_guide/_quickstart.py`: exactly the three normative lines of §6.21 (keep any blank line ruff's isort inserts after the import).

- [ ] **Step 1: Write the failing tests**
  - `tests/plot/test_scatter.py` (§7.6): `test_offsets_equal_valid_xy` (ungrouped: the one collection's offsets equal `coords.xy[valid]`, exact); `test_grouped_offsets` (by `"location"`: concatenated offsets in sorted key order equal each group's valid xy, legend texts equal the keys); `test_default_colours_are_petroff10`; `test_no_pyplot_figure_created` (`plt.get_fignums() == []` after `ax=None`); `test_returns_the_given_ax`; `test_kwargs_reach_artist` (`color="k"`); `test_axis_and_quadrant_labels` (English quadrant labels vibrant, chaotic, monotonous, calm present among the Axes texts; `label_language="deu"` falls back to the same); `test_saves_png_svg_pdf` (`tmp_path`, each file non-empty).
  - `tests/test_package.py`: `test_matplotlib_not_imported` (subprocess: `import soundscapy, sys; assert "matplotlib" not in sys.modules`); the existing silent-import test unchanged; `test_all_names_resolve` (every name in `__all__` is an attribute).
  - `tests/test_docs_structure.py`: `test_quickstart_is_the_first_plot` (the file contains `load_isd(subset=True)`, `angles="equal"`, `by="location"` and `savefig("first-plot.png")`) and `test_quickstart_runs` (subprocess `sys.executable docs/user_guide/_quickstart.py` with `cwd=tmp_path` and `MPLBACKEND=Agg`: exit 0, empty stdout, `first-plot.png` exists; stderr is not asserted, because matplotlib's first-run font-cache message may appear there in a fresh environment).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement**, replace `_quickstart.py`, and edit `01-install.qmd`: "It prints the installed soundscapy version" becomes "It saves your first plot, a scatter of the bundled ISD subset by location, as `first-plot.png`"; both "paste the same code" places (JupyterLab and the hosted notebook) show the notebook form of §6.21 in a plain ```` ```python ```` block ending `sspy.plot.scatter(coords, by="location").figure`. The two `{{< include _quickstart.py >}}` lines the docs test requires stay; Plan call: the hosted-notebook include is replaced by the notebook-form block, and the first include (Step 1, "save this code") stays.
- [ ] **Step 4: Pass**; `uv run task check` green.
- [ ] **Step 5: Commit** `feat: plot.scatter, the S1 top-level API and the first-plot quick-start`.

---

### Task 13: Done-when check, ADRs, Taskfile note, docs build

**Files:**
- Create: `design/adr/0001-coordinates-carry-settings.md`, `design/adr/0002-response-id.md`, `design/adr/0003-ssp-column-namespace.md`, `design/adr/0004-h1-hashing.md`, `tests/science/test_s1_done.py`
- Modify: `Taskfile.yml` (comments only), `docs/_freeze/` if `task docs` changes it

**Interfaces:**
- Consumes: everything above. Produces no code API.

- [ ] **Step 1: Write the done-when test** `tests/science/test_s1_done.py::test_thin_slice` (`science` marker): `coords = sspy.project(sspy.load_isd(subset=True), angles="equal")`; `ax = sspy.plot.scatter(coords, by="location")`; `ax` is a matplotlib `Axes`; the vertical stack of every collection's offsets, in sorted location order, equals the stack of each `coords.split("location")[key]` valid xy (exact, `np.testing.assert_array_equal`: the plot draws the array it is given); the total equals `coords.n_valid()`. Run it: PASS (every piece exists). This is the S1 done-when command.

- [ ] **Step 2: Taskfile carry-over (M5).** S1 adds no `reference` and no `network` test (the first `network` test is S2's full-ISD counts; the first `reference` test is S3's aletta regeneration; the subset row checks are S9's). The `|| test $? -eq 5` clauses therefore stay. Edit only the two comments so they name the right slices: `test:reference` "Remove this tolerance in the commit that adds the first reference test (M1 slice S3)"; `test:network` "Remove it in the commit that adds the first network test (M1 slice S2)".

- [ ] **Step 3: Write the four ADRs**, each with Status (Accepted, 2026-10-31), Context, Decision, Consequences and the code that exercised it (plan §6: ADRs are written from code). One page each; cite the spec section, do not restate it.
  - `0001-coordinates-carry-settings.md`: no `ProjectionResult`; `Coordinates` holds its `settings` (typed `FrozenModel` so `distribution` does not import `projection`) and `provenance`; labels are numpy arrays, frames only at the return (§6.10, §11.1). Exercised by `project`, `split`, `attach`.
  - `0002-response-id.md`: column, then key, then physical row number; never memory order; recorded as `response_id_source`; `attach` aligns by the rule recorded (§6.3.2, §6.16). Exercised by `read_survey` and `attach`; the ISD four-part key.
  - `0003-ssp-column-namespace.md`: every derived column in a frame holding user columns starts `ssp_`; `read_survey` refuses `ssp_` input columns up front; `attach` never overwrites (§6.3.3). Names the constants in `_model`.
  - `0004-h1-hashing.md`: one helper, `h1-sha256:` prefix, canonical JSON with `by_alias=True`, `frame_hash` normalisation, golden vectors in `tests/test_provenance.py`, and this plan's length-prefix call for text columns (§6.3.5).

- [ ] **Step 4: Run the full gate and the docs build**

```bash
uv run task check
uv run task docs
git status --short docs/_freeze
```

Expected: `task check` green with coverage at 80% or more; `task docs` renders offline and the API reference lists the S1 names; commit any `docs/_freeze/` change with this task. If great-docs flags a docstring, fix the docstring.

- [ ] **Step 5: Commit** `docs(adr): S1 decisions; test: S1 done-when` (the ADRs, the test, the Taskfile comments, any `_freeze` change).

- [ ] **Step 6: Finish the unit of work** per AGENTS.md: `/simplify` on the S1 change, `uv run task check`, then `/code-review` and its fixes.

---

## Plan calls

Each is marked "Plan call:" where it is made. A reviewer should check them.

1. `_Unset` is an enum in `_model` with `_UNSET` its member (Task 1).
2. `Citation`, the pair-table aliases, `to_toml`/`from_toml` and the `ssp_` namespace constants live in `_model` (Task 1).
3. S1 adds only the runtime dependencies S1 imports; the rest come with their slices (Task 1).
4. The hashing helper's entry points are `hash_bytes`, `hash_file`, `hash_json`, `canonical_json`, `check_hash`; dumps use `by_alias=True` (Task 2).
5. The envelope omits libraries not yet installed, guards extra keys with `OPTIONAL_KEYS`, and S1 results write no `profile`/`overrides` (Task 2).
6. `frame_hash` prefixes each text column with its byte length (Task 3; fixes a spec defect).
7. `read_text_table` takes keywords instead of a `ReadOptions` object; short rows are padded, long rows and duplicate headers raise (Task 3).
8. Every `_data` read verifies its pin (Task 4).
9. `Circumplex` carries `schema`/`schema_version` like the other data files; `positions` and `angles_deg` are properties; `Axis.labels` is empty in the shipped file (Task 4).
10. The scale labels use the 0.x capitalisation, and the Table A.1 fixture test moves to S3 (Task 4).
11. A zero Σ|cos| or Σ|sin| raises `FormulationError`; the S1 kernel's `Literal` has no `aletta2024` (Task 5).
12. `Status.all_included` and the fixed key set of `counts()` (Task 6).
13. `SurveyMapping` maps attributes, ids and language only in S1; the `schema` field is `schema_name` with alias `schema`; the whole coding order ships in S1; the default item scale is filled from the instrument and omitted when written (Task 7).
14. `read_survey` in S1 takes paths only, has `read_options` as its one keyword, resolves `delimiter="auto"` by suffix, always raises on a scale violation, and applies only `incomplete_attributes`; `Survey.to_frame`/`select` wait for S2; `_read_survey` takes extra `data_refs` (Task 8).
15. The subset keeps source row and column order; the sidecar's non-citation keys; S1's registry holds one version; S1's `isd_mapping.toml` is minimal, under `_data/datasets/isd/`; `load(subset=False)` raises `NotImplementedError` (Task 9).
16. `AngleSelection` ships with `default: Literal["equal"] = "equal"` in S1, a provisional default S3 replaces with the profile's (Task 10).
17. `provenance["id_columns"]` maps id roles to user columns, used by `split`; `ssp_angle_set` is `""` for rows not projected (Task 10).
18. `attach` raises `AlignmentError` when no frame row matches; the `Table` form waits for `Table` (Task 11).
19. `plot.scatter` cycles `petroff10` beyond ten groups and keeps an empty series for a group without valid rows (Task 12).
20. The install page's hosted-notebook include becomes the notebook-form block (Task 12).
21. The M5 exit-5 clauses stay through S1; only their comments change (Task 13).

## Spec defects found while planning

- §6.3.5: text columns "joined by U+001F" with no column delimiter let two different frames hash alike (Plan call 6).
- §6.4.1 (and the angle-set and profile files): a pydantic field named `schema` warns "shadows an attribute in parent BaseModel", which `filterwarnings = ["error"]` makes fatal (Plan call 13).
- §6.8.1's example writes `angles_deg = [0, 46, ...]` as TOML integers, while `StrictFloat` fields (§6.2.1, `Attribute.angle_deg`) reject integers. S3 must either write `0.0` or type `angles_deg` as `float` with a non-strict validator; flagged for the S3 plan.
- §6.3.8 gives `read_text_table(path, options)`, but `ReadOptions` lives in `survey`, above `_frames` in the layer contract (Plan call 7).
- §6.10 lists no field from which `split(by="location")` can learn that the role `location` is the user's `LocationID` column (Plan call 17).
