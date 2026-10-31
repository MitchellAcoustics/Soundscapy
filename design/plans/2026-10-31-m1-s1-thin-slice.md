# M1 Slice S1 (Thin Slice, Contracts First) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the load-bearing contracts of M1 (errors, value types, hashing, provenance, logging, the frame boundary, status, the data registry) and one thin path on them, so that `sspy.plot.scatter(sspy.project(sspy.load_isd(subset=True), angles="equal"), by="location")` returns Axes whose offsets equal the projected xy.

**Architecture:** Bottom-layer modules (`errors`, `_model`, `provenance`, `_log`, `_frames`, `_data`) come first, each with its tests. `instrument` adds `Scale` and `Circumplex` read from a shipped, hash-pinned TOML file. `projection.formulas` is a pure numpy kernel. `survey` reads a CSV through a `SurveyMapping` into a `Survey`; `projection.project` turns it into `distribution.Coordinates`; `workspace.attach` and `plot.scatter` consume it. Layers follow the import contracts already in `pyproject.toml` (spec §5.2.1).

**Tech Stack:** Python 3.11+, numpy, pydantic 2, narwhals (`stable.v2`), pandas, polars (tests), matplotlib (lazy), tomlkit and `tomllib`, pytest and hypothesis; uv, ruff, ty, import-linter.

**Spec:** `design/specs/2026-10-31-phase0-m1-design.md` (draft 0.4, with the S1 plan-review amendments marked "Amended at S1 plan review"), §9 row S1, with §6.1 to §6.5, §6.8.3, §6.9, §6.10, §6.15, §6.16, §6.19 to §6.21, §7 and §11.4. §11.4 overrides earlier spec sentences it contradicts. Read each section a task cites before starting the task; this plan does not repeat normative blocks it can cite.

## Global Constraints

- uv only: `uv run`, `uvx`, `uv add`; never bare `python` or `pip`. Never run a command that waits on stdin.
- Done for every task means `uv run task check` passes (lint, import contracts, ty, tests with branch coverage at 80% or more).
- Runtime dependency floors are spec §6.19's as amended: numpy 1.26, pandas 2.2.1, narwhals 2.0, pydantic 2.7, matplotlib 3.10.7, tomlkit 0.13. Every dev and test dependency gets a floor at the current release (§11.4).
- Public API names, signatures and module placement follow spec §6.1 and the sections cited per task. Where S1 ships a narrower signature than the final M1 one, the task says which later slice widens it; the narrower form must be a prefix of the final one (same names, same order, keyword-only).
- Every optional keyword of an operation that maps to a settings or input-description field defaults to `_UNSET` (the `_Unset` sentinel in `soundscapy._model`), never `None` (§6.3.4). Keywords that are not settings keep the defaults written in their task signatures: drawing arguments of plot functions (§6.15); `like=None` on `_frames.new_frame` and `Coordinates.to_frame`; `valid=None` on `_frames.with_columns`; `subset=False` on `datasets.isd.load`; the name and version defaults of `circumplex()`; `language=""` on `Circumplex.label`; the keywords of the private helpers `envelope` and `_read_survey`.
- Only `soundscapy._frames` imports pandas, polars or pyarrow, and in S1 it imports pandas only (polars is a test dependency, pyarrow arrives in S4); only `soundscapy.plot` imports matplotlib, and only inside functions; only `soundscapy.provenance` imports `hashlib` (§6.3.5, §6.3.8, §6.15). No bottom-layer module imports the root package `soundscapy` (Task 2).
- Every public object has a numpy docstring; every function implementing a procedure has a `References` section with a `[@key]` that exists in `docs/references.bib` (house-style).
- Floats in FrozenModels are `StrictFloat`: they reject ±inf, NaN (`allow_inf_nan=False`, §6.2.1), bool and str, and accept integers as floats (checked on pydantic 2.7.0: `{"minimum": 1}` validates to `1.0`). Shipped and fixture TOML files still write floats with a decimal point (`1.0`), so their bytes match what `to_toml` writes and the byte round-trip tests hold.
- Every float comparison in a test has an explicit tolerance with a comment giving the reason.
- `filterwarnings = ["error"]`: every expected warning is asserted with `pytest.warns`.
- `pytest.raises(match=...)` takes a regex: a message fragment holding `[`, `(`, `.` or `*` goes through `re.escape`.
- **Deliberate misuse and ty.** ty checks `tests/` (only `scripts/` is excluded). Input that arrives as untyped data in real use is written that way and needs nothing: a dict through `Model.model_validate`, TOML text through `from_toml`. A typed call misused on purpose (a value outside a `Literal`, an unknown keyword, assignment to a frozen field) carries a one-line ignore naming the rule ty reports, with the reason: `# ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime SettingsError`. Never `cast(Any, ...)`, `setattr` or a file-level or bare ignore to hide the misuse; never relax ty settings. Drop an ignore ty does not need.
- Commit `uv.lock` with every dependency change and give the reason. Commit messages end with the attribution lines your session supplies.
- Branch `soundscapy-rewrite`. Do not reformat files you do not otherwise change.

## Review Focus

1. **A user's mapping names a column the CSV does not have** (a typo, or a renamed export). Expected: `MappingError` naming each missing column and the mapping key, before any id or coding work runs. Pinned in Task 8 (`test_missing_mapped_column_raises`).
2. **A ragged CSV** (a row with fewer cells than the header, as some exporters trim trailing empties, or more cells). Expected: short rows padded with `""` (Plan call); a long row raises `FileFormatError` naming its row number. Duplicate header names raise `FileFormatError`. Pinned in Task 3 (`test_read_text_table_ragged_rows`, `test_read_text_table_duplicate_header`).
3. **Numeric text in the forms the ISD really uses** (`"4"`, `"4.0"`, `" 4 "`, blank). Expected: 4.0, 4.0, 4.0, NaN; a comma decimal with `decimal="."` raises `CodingError` listing the text. Pinned in Task 7 (`test_apply_coding_isd_forms`).
4. **`attach` to a frame the user re-read with `pd.read_csv`**, where `RecordID` is now an integer column and the rows are re-sorted. Expected: alignment by id still matches every row, in the frame's order; frame rows absent from the result get nulls; if no frame row matches, `AlignmentError` (Plan call), never a frame of nulls. Pinned in Task 11 (`test_attach_to_reread_frame_with_int_ids`, `test_attach_frame_with_extra_rows`, `test_attach_no_match_raises`).
5. **`angles=` given an angle set before S3 ships them** (`"satp@1.5"`, `"aletta-2024@1"`). Expected: `SettingsError` naming `angles` and saying only `"equal"` is available in this build, not a pydantic traceback. Pinned in Task 10 (`test_project_rejects_unknown_angles`, both strings).

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
  - `soundscapy._model.StrPairs`: `Annotated[tuple[tuple[str, str], ...], ...]` with a before-validator turning a table (`dict`) into pairs and a serializer turning pairs back into a table in `mode="json"` (§6.4.4 table form). A second alias for `tuple[tuple[str, tuple[str, ...]], ...]` (scale labels) and one for `tuple[tuple[str, StrictFloat], ...]` (value maps) follow the same pattern. All three reject a repeated key, from a table or from pairs, with a `ValueError` naming the key, so the table form of `model_dump(mode="json")` is lossless and equal dumps mean equal models. Plan call: one reusable mechanism for every key-to-value field, not a validator per model.
  - `to_toml(model: FrozenModel) -> str` and `from_toml(cls: type[M], text: str) -> M`: `model_dump(mode="json", by_alias=True)` written by tomlkit; `tomllib.loads` then `model_validate`. `from_toml` re-raises `pydantic.ValidationError` as `SettingsError` (message keeps pydantic's field paths).
  - Namespace constants (§6.3.3): `SSP_PREFIX = "ssp_"`, `COL_RESPONSE_ID`, `COL_INCLUDED`, `COL_REASONS`, `COL_FLAGS`, `COL_LANGUAGE`, `COL_ANGLE_SET` (values `ssp_response_id` ... `ssp_angle_set`), `axis_column(axis_id: str) -> str` returning `f"ssp_{axis_id}"`. Plan call: they live in `_model` because `distribution` (which may not import `survey`) and `workspace` both need them and `_model` is already in the bottom layer; a new module would need a layer-contract edit.

- [ ] **Step 1: Add the S1 dependencies**

```bash
uv add 'numpy>=1.26' 'pandas>=2.2.1' 'narwhals>=2.0' 'pydantic>=2.7' 'matplotlib>=3.10.7' 'tomlkit>=0.13'
uv add --group dev 'pandas-stubs>=3.1.0.261029'
uv add --group test 'polars>=2.1.0'
```

Reasons (commit message): numpy, arrays in the core; pandas, frames read from paths (the 2.2 series is the Anaconda course floor; 2.2.1, because 2.2.0 emits a `DeprecationWarning` on import when pyarrow is absent, which `filterwarnings = ["error"]` and the `-W error` import test make fatal in the `lowest` job); narwhals, the frame boundary (`stable.v2`); pydantic, settings and value types (2.7 is the spec floor); matplotlib, `plot.scatter` and `petroff10` (3.10.7, because 3.10.0 to 3.10.6 call pyparsing's camelCase API and emit `PyparsingDeprecationWarning` on import against pyparsing 3.3, which the `lowest` job installs); tomlkit, writing mapping TOML deterministically; pandas-stubs, so ty checks `_frames`; polars, the second frame library in the frame tests. Both patch floors were checked on Python 3.11 with numpy 1.26.0 under `-W error`, and both stay inside spec §6.19's series (amended to name them). Plan call: scipy, contourpy, pyarrow, pooch, tqdm, click and openpyxl are not added in S1, because nothing in S1 imports them; S5 (scipy, contourpy), S4 (pyarrow, openpyxl export), S7 (openpyxl read), S8 (click) and S9 (pooch, tqdm) add them. Adding pyarrow 15 now would also silence pandas 2.2.0, but it would bring a large dependency forward only to quiet a warning, and it changes pandas 3's default string backend under every S1 frame test.

- [ ] **Step 2: Write the failing tests**
  - `tests/licences.toml`: spec §6.19's table as `[licences] numpy = "BSD-3-Clause ..."`, every row (all 13, so later slices only add to `pyproject.toml`). `tests/test_licences.py::test_every_runtime_dependency_has_a_licence_entry` (each name in `project.dependencies` is a key) and `::test_installed_metadata_declares_a_licence` (`License-Expression`, `License` or a `License ::` classifier is non-empty for each).
  - `tests/test_errors.py`: one parametrised test over a table of (class, built-in) pairs copied from §6.3.10, asserting `issubclass(cls, SoundscapyError)` and `issubclass(cls, builtin)`, `str(cls("m", hint="h")) == "m"`, `.hint == "h"`; each warning subclasses `SoundscapyWarning` and `UserWarning`; `test_no_error_name_shadows_a_builtin` (no name in `errors.__all__` is in `dir(builtins)`); `DataIntegrityError` exists and `IntegrityError` does not.
  - `tests/test_model.py`: a `FrozenModel` subclass rejects `inf`, `-inf`, `nan` in a `StrictFloat` field, rejects extra keys (`model_validate` of a dict), is frozen (assignment raises `ValidationError`, with the ty ignore of Global Constraints); `StrPairs` field validates from `{"a": "x"}` and from `(("a", "x"),)`, equal values, and `model_dump(mode="json")` gives `{"a": "x"}`; `StrPairs` from `(("a", "x"), ("a", "y"))` raises `ValidationError` naming `a`; `to_toml`/`from_toml` round-trip a model holding a `StrPairs`, a tuple and a `"none"`-literal field to an equal model; `from_toml` on a bad value raises `SettingsError`; `axis_column("ISOPleasant") == "ssp_ISOPleasant"`; `_UNSET is _Unset.UNSET`.

- [ ] **Step 3: Run them to verify they fail**: `uv run pytest tests/test_errors.py tests/test_model.py tests/test_licences.py -v`, expected FAIL (import errors).

- [ ] **Step 4: Implement `errors.py` and `_model.py`** with the interfaces above. Note for every later task: pydantic wraps a `ValueError` raised inside a validator (which every soundscapy `ValueError` subclass is) into `ValidationError`. Validators raise plain `ValueError` with a clear message; public entry points (`from_toml`, `read_mapping`, `read_survey`, `project`) catch `ValidationError` and raise the soundscapy error the spec names. A `LookupError` (such as `DataReferenceError`) is not wrapped by pydantic and propagates as itself.

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
  - `JSONValue: TypeAlias = None | bool | int | float | str | Sequence["JSONValue"] | Mapping[str, "JSONValue"]`, defined here (bottom layer) for later modules to import.
  - `hash_bytes(data: bytes | Iterable[bytes]) -> str` (the only `hashlib` user; an iterable is hashed as its concatenation, so `_frames.frame_hash` can stream)
  - `hash_file(path: str | os.PathLike[str]) -> str` (raw bytes)
  - `canonical_json(value: pydantic.BaseModel | JSONValue) -> bytes` (§6.3.5 row 2 as amended; models via `model_dump(mode="json", by_alias=True)`; Plan call: `by_alias=True` everywhere, because the `schema` field needs an alias, see Tasks 4 and 7). A non-`str` mapping key or a value outside `JSONValue` raises `TypeError` naming its path.
  - `hash_json(value: pydantic.BaseModel | JSONValue) -> str`
  - `check_hash(value: str) -> str`: returns the 64-hex digest; any other prefix or a malformed digest raises `HashSchemeError`
  - `envelope(*, operation: str, settings: FrozenModel, input_hash: str, mapping_hash: str = "", data_refs: Sequence[Mapping[str, str]] = (), extra_libraries: Sequence[str] = (), upstream_run: str = "", **extra: object) -> dict[str, object]`: the §6.3.6 normative keys. `libraries` holds the installed versions of numpy, scipy, pandas, pyarrow, narwhals, pydantic and contourpy, plus each name in `extra_libraries`. Plan call: a listed library that is not installed is left out (scipy, contourpy and pyarrow arrive in later slices); once they are runtime dependencies every key is present. `extra` keys must belong to `OPTIONAL_KEYS` (`input_hashes`, `upstream`, `quality_skipped`, `profile`, `profile_inherited`, `overrides`, `response_id_source`, `read`, `id_columns`, `response_id_columns`, `response_key_blank_ok`), else `ValueError`, so a typo cannot add a key. `created_utc` is `datetime.now(UTC)` as ISO 8601 with `Z`, seconds precision. Plan call: `profile` and `overrides` are not written by S1 results, because profiles arrive in S3 ("the mapping can only gain keys", §6.3.6). The three id keys are §6.3.6's amended row (Tasks 8, 10, 11).
  - `soundscapy_version` (in `envelope` and `show_versions`) is `importlib.metadata.version("soundscapy")`, read when called. No bottom-layer module (`errors`, `provenance`, `_frames`, `_data`, `_model`, `_log`, `tables`) imports the root package `soundscapy`: from Task 12 the root imports `plot`, `datasets` and `workspace` eagerly, so such an import would be a cycle and break the layers contract.
  - `show_versions() -> str`: multi-line `key: value` for soundscapy, python, platform and each library; never prints.
  - `soundscapy._log.logger` and `set_log_level(level: int | str) -> None`, the normative mechanism of §6.3.9 verbatim.

- [ ] **Step 1: Write the failing tests**
  - `tests/test_provenance.py::test_golden_vectors`, parametrised, the §6.3.5 table exactly: `hash_bytes(b"") == "h1-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"`; `hash_json({}) == "h1-sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"`; `hash_json({"b": [1.5, "é"], "a": 1}) == "h1-sha256:2bb202b35963ad9e57f87e5828e072cc53f4720ecd10a7d625f850411a31c4c6"`; `hash_json({"clip": "none"}) == "h1-sha256:7c457b553a03102b7220154eb95650e02d2840d02f397a2d226afba1fa85a1dd"`.
  - `::test_non_finite_json_names_the_key`: `hash_json({"a": float("inf")})` raises `ValueError` with `match="a"`; `{"b": [1.0, float("nan")]}` raises with `match=re.escape("b[1]")`.
  - `::test_non_json_value_raises_type_error` (a `set` value; an `int` mapping key).
  - `::test_hash_file_equals_hash_bytes` (tmp file), `::test_model_and_its_dump_hash_alike`, `::test_check_hash` (`"sha256:..."` and `"h2-sha256:..."` raise `HashSchemeError`; a valid one returns the hex).
  - `::test_hashlib_only_in_provenance`: walk `src/soundscapy/**/*.py`; only `provenance.py` contains `import hashlib` or `from hashlib`.
  - `::test_envelope_keys`: the normative key set of §6.3.6 that S1 writes (`soundscapy_version`, `python_version`, `platform`, `libraries`, `hash_scheme == "h1"`, `data_refs`, `created_utc` ending `Z`, `operation`, `input_hash`, `mapping_hash`, `settings_hash`, `upstream_run`); `soundscapy_version == importlib.metadata.version("soundscapy")`; `libraries` has numpy, pandas, narwhals, pydantic; `settings_hash == hash_json(settings)`; an unknown extra key raises `ValueError`.
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
- Modify: `design/specs/2026-10-31-phase0-m1-design.md` (§6.3.5 golden-vector table gains the frame vector, Step 4)

**Interfaces:**
- Consumes: `hash_bytes`, `canonical_json` (Task 2); `CodingError`, `ColumnCollisionError`, `EncodingError`, `FileFormatError` (Task 1).
- Imports: pandas and `narwhals.stable.v2` at module scope; never polars or pyarrow (neither is a runtime dependency in S1). Library dispatch goes through narwhals: `to_eager` is `nw.from_native(obj, eager_only=True)` with its lazy-frame `TypeError` re-raised with the §6.3.8 message; `new_frame` builds with `nw.from_dict(columns, backend=nw.get_native_namespace(like))`, or pandas when `like` is `None`. So `import soundscapy` works with runtime dependencies only (CI's wheel install job).
- Produces (§6.3.8; S1 subset):
  - `to_eager(obj: object) -> nw.DataFrame[Any]`
  - `columns_float(frame: nw.DataFrame[Any], names: Sequence[str]) -> NDArray[np.float64]` of shape (n, len(names)), copied, read-only
  - `column_text(frame: nw.DataFrame[Any], name: str) -> NDArray[np.object_]` of Python `str`, nulls as `""`, read-only
  - `to_native(frame: nw.DataFrame[Any]) -> Any`
  - `with_columns(frame: object, columns: Mapping[str, NDArray[Any]], *, valid: NDArray[np.bool_] | None = None) -> Any` (native in, same library out; `ColumnCollisionError` on an existing name). Plan call: rows where `valid` is False are null in every added column whatever its dtype (polars null; pandas NaN in a float column, `pd.NA` in a `boolean` column, `None` in an object text column), so `attach` can express unmatched rows.
  - `new_frame(columns: Mapping[str, NDArray[Any]], like: object | None = None) -> Any` (pandas, or the library of `like`)
  - `TextTable` (frozen dataclass): `frame` (pandas, every cell `str`), `row_numbers: NDArray[np.int64]` (physical record numbers, header = 1), `n_blank: int`, `delimiter: str`, `encoding: str`
  - `read_text_table(path: str | os.PathLike[str], *, encoding: str, delimiter: str) -> TextTable`. Plan call (§6.3.8 amended): keywords rather than an `options` object, because `ReadOptions` lives in `survey`, above `_frames`. The `csv` module reads every cell as text; a record whose cells are all empty after stripping is skipped and counted, keeping its number out of use (§6.3.2 rule 3); a short row is padded with `""`, a long row raises `FileFormatError` with its record number (Plan call); duplicate header names raise `FileFormatError`; a decode failure raises `EncodingError` with hint `encoding="cp1252"`.
  - `frame_hash(frame: object) -> str` (§6.3.5 row 3 as amended).
  - `Frame` type alias for typing elsewhere (`narwhals.typing.IntoDataFrame`).
- **`frame_hash` layout (Plan call; spec §6.3.5 row 3 amended to match).** Header and the numeric and bool rules exactly as §6.3.5. Text columns change, because the spec's "UTF-8 values joined by U+001F, null as U+0000" collides three ways: across columns (`["a"], ["b"]` against `["ab"], [""]`), inside a column on values holding U+001F (`["a\x1fb", "c"]` against `["a", "b\x1fc"]`), and between null and a literal `"\x00"`. New rule: a text column is written value by value; a non-null value is its UTF-8 byte length as a little-endian `uint64` followed by those bytes; a null is the single `uint64` `0xFFFFFFFFFFFFFFFF` with no bytes after it. No separator and no sentinel character remain, and with `n_rows` in the header the stream decodes uniquely. Numeric and bool columns have fixed widths and are unchanged.
- **Value to text, backend-independent.** Nulls are found first with narwhals `is_null()` (None, NaN, NaT, `pd.NA`). Text comes from each value's Python form (`Series.to_list()`), never from `cast(nw.String)`, whose output differs by backend (pandas `'True'`, polars `'true'`; pandas `'2020-01-01'`, polars `'2020-01-01 00:00:00.000000'`). `str` values as they are; `datetime`, `date` and `time` values (pandas `Timestamp` included) by `.isoformat()`; categorical values as their category text; every other value by `str()`, so booleans give `"True"`/`"False"` on both backends. Checked with narwhals, pandas 3.1 and polars 2.1: a bool column with a null (pandas `Object`, polars `Boolean`) and a naive datetime column with a null give the same texts and nulls in both. Nanosecond datetimes are out of scope in S1 (polars' `to_list` truncates them).

- [ ] **Step 1: Write the failing tests.** `tests/conftest.py` adds the `frame_lib` fixture, parametrised over `"pandas"` and `"polars"`, returning a `make(dict) -> native frame` callable (§7). In `tests/test_frames.py`:
  - `test_lazy_frame_raises_type_error`: `to_eager(pl.LazyFrame(...))` raises `TypeError` with message exactly `"lazy frames are not accepted; call .collect() first"`; a `dict`, a pandas Series and a polars Series raise `TypeError`.
  - `test_frames_works_without_polars`: subprocess `sys.executable -c "import sys; sys.modules['polars'] = None; sys.modules['pyarrow'] = None; import soundscapy._frames"` exits 0 (simulates the wheel-only install in the dev venv).
  - `test_columns_float_copies_and_is_read_only` (`frame_lib`): mutating the source frame afterwards does not change the array; `arr.flags.writeable is False`; null and NaN both give NaN.
  - `test_columns_float_refuses_text_and_bool` (`frame_lib`): `CodingError` naming the column.
  - `test_column_text_nulls_are_empty`; `test_with_columns_collision` (`ColumnCollisionError`); `test_with_columns_valid_mask_gives_nulls` (`frame_lib`: a bool, a float and a text column; the masked-out rows are null under narwhals `is_null()`, the others keep their values); `test_new_frame_follows_like` (pandas default, polars when `like` is polars).
  - `test_frame_hash_vector`: content `{"i": [1, None, 3], "f": [0.5, 1.5, -2.0], "t": ["a", None, "é"]}` built as pandas (where `i` becomes `float64`) and as polars (where `i` stays `Int64`) gives one hash, equal to a pinned constant `FRAME_VECTOR`. Pin it once from the implementation (§11.3 leaves the value to the builder) and assert `frame.dtypes` show the two different integer representations, so the test proves normalisation.
  - `test_frame_hash_normalisation`: pandas `int64` [1, 2, 3] equals pandas `float64` [1.0, 2.0, 3.0] equals polars `Int64`; a text column `["1", "2", "3"]` differs; a bool column without nulls differs from the same column with a null; a bool column with a null hashes alike in pandas (`Object`) and polars (`Boolean`); a naive datetime column with a null hashes alike in pandas and polars.
  - `test_frame_hash_text_does_not_collide`, parametrised: `["a"], ["b"]` against `["ab"], [""]` (two columns); `["a\x1fb", "c"]` against `["a", "b\x1fc"]`; `[None]` against `["\x00"]`. Each pair hashes differently.
  - `test_read_text_table_numbers_records`: header, two records, a blank line, one record gives `row_numbers == [2, 3, 5]` and `n_blank == 1`; a UTF-8 BOM is stripped under `utf-8-sig`; `test_read_text_table_ragged_rows`; `test_read_text_table_duplicate_header`; `test_read_text_table_bad_encoding` (`EncodingError`, hint mentions `cp1252`).
- [ ] **Step 2: Run them to verify they fail**: `uv run pytest tests/test_frames.py -v`.
- [ ] **Step 3: Implement `_frames.py`** with `narwhals.stable.v2`. narwhals `InvalidOperationError`, `ColumnNotFoundError` and `DuplicateError` are re-raised as soundscapy errors with backend-neutral messages (§6.3.8).
- [ ] **Step 4: Record the frame vector in the spec.** Add one row to §6.3.5's golden-vector table: "frame `{"i": [1, None, 3], "f": [0.5, 1.5, -2.0], "t": ["a", None, "é"]}` (pandas or polars) | `<FRAME_VECTOR>`", and in the paragraph after the table change "The builder adds the frame vectors" to say the frame vector was pinned in S1 Task 3 under the amended text rule (the `Density` and `Densities` vectors stay S5's). Commit the spec edit with this task.
- [ ] **Step 5: Run the tests to verify they pass** on the locked pandas 3.x, then on the floors (the pandas 2.2 run of the S1 done-when; CI's `lowest` job repeats it). `$SCRATCH` is any directory outside the repository:

```bash
UV_RESOLUTION=lowest-direct UV_PROJECT_ENVIRONMENT="$SCRATCH/lowest-venv" uv run --python 3.11 pytest tests/test_frames.py tests/test_package.py -p no:cacheprovider
UV_RESOLUTION=lowest-direct UV_PROJECT_ENVIRONMENT="$SCRATCH/lowest-venv" uv tree --depth 1
git restore uv.lock
```

Expected: both runs pass (`tests/test_package.py` proves the silent import on the floors); `uv tree` shows pandas 2.2.1 and numpy 1.26.0. Then `uv run task check` green.
- [ ] **Step 6: Commit** `feat: frame boundary with frame_hash` (with the spec's frame-vector row).

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
  - `soundscapy.instrument.circumplex`: `Attribute`, `Axis`, `Translation`, `Circumplex` (§6.2.3 with §11.4: `Attribute.labels` holds the reference language only; a translation for the reference language is rejected). Plan call: `Circumplex` has `schema_name: Literal["soundscapy.circumplex"]` as `Field(alias="schema")` and `schema_version: Literal[1]`, both required with no default and always written (the identity keys of §6.4.4), with `model_config = FrozenModel.model_config | {"populate_by_name": True}`, the same shape as Task 7's `SurveyMapping` (§6.4.1 as amended), so `Circumplex.model_validate({**c.model_dump(), **changes})` (§6.2.1's derived-record idiom) works. Properties `positions -> tuple[str, ...]` and `angles_deg -> tuple[float, ...]` (Plan call: properties, since they take no argument); methods `label(position: str, language: str = "") -> str` (blank means the reference language; falls back to it for any language without a translation) and `position_for(key: str) -> str`. `Axis.labels` defaults to `()` and the shipped file leaves it empty: axis titles are the axis ids and the end labels come from the attributes (§6.2.3 last paragraph, drawn in Task 12).
  - `circumplex(name: str = "iso-12913-outdoor", version: str = "1") -> Circumplex`, reading `circumplexes/<name>/<version>.toml`; an unknown pair raises `DataReferenceError` listing the shipped ones.
  - `soundscapy.instrument` re-exports `Scale`, `Circumplex`, `circumplex`.
- **`iso-12913-outdoor/1.toml` content** (§6.2.3), top-level keys first: `schema = "soundscapy.circumplex"`, `schema_version = 1`, `name`, `version = "1"`, `reference_language = "eng"`; eight `[[attributes]]` PAQ1..PAQ8 at `0.0, 45.0, ..., 315.0` with `labels = { eng = "pleasant" }` ... `calm`; two `[[axes]]` (`ISOPleasant` PAQ1/PAQ5, `ISOEventful` PAQ3/PAQ7); `[scale]` `minimum = 1.0`, `maximum = 5.0`, `kind = "ordinal"`, `[scale.labels] eng = ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"]` (the 0.x wording, `git show origin/main:src/soundscapy/surveys/survey_utils.py`, until the owner checks Table A.1, §11.2); `[quadrants]` `"++" = "PAQ2"`, `"-+" = "PAQ4"`, `"--" = "PAQ6"`, `"+-" = "PAQ8"`; no translations (S3 adds them); `[citation]` citing ISO/TS 12913-2:2018 (the attribute set) and ISO/TS 12913-3:2019 (the 45° angles), `doi_or_url = ""` (paywalled standard), `licence = "BSD-3-Clause (this file)"`, `date = "2019"`, notes saying the scale wording awaits the owner's Table A.1 check. Plan call: the Table A.1 transcription test of §11.4 moves to S3 with the labels, because its only source in hand is the same 0.x wording; spec §9's S3 row is amended to carry it.
- **`docs/references.bib`** gains `iso12913-2-2018`, `iso12913-3-2019`, `mitchell2022how` (JASA EL 2, 037201, doi 10.1121/10.0009794), `mitchell2026translation` (chapter 4 of *Soundscape Across Languages*, Springer Nature, 2026) and `isd2024` (Zenodo, doi 10.5281/zenodo.10672568), with the authors as `design/literature/README.md` and §6.20.1 give them. Plan call: the Mitchell 2025 blog post ("A robust approach to soundscape circumplex coordinate projections", PRD §2) gets no entry in S1, because its URL is not in hand; canonical cites chapter 4 alone until the owner supplies the URL. Never invent one.

- [ ] **Step 1: Write the failing tests**
  - `tests/instrument/test_scale.py`: `Scale(minimum=1.0, maximum=5.0)` has midpoint 3.0, half_range 2.0, range 4.0 (exact); `minimum >= maximum` and ordinal non-integer bounds are rejected; `contains([1, 5, 3.5, 0, 6, nan, inf])` is `[T, T, F, F, F, F, F]` for ordinal and `[T, T, T, F, F, F, F]` for interval; `Scale.model_validate({"minimum": 1, "maximum": 5})` validates to `1.0`/`5.0`, equals the float form and has the same `hash_json`; bool (`True`) and str (`"1"`) bounds raise `ValidationError` (`StrictFloat`); hypothesis `test_scale_toml_round_trip` (finite bounds, both kinds) gives an equal model and equal `hash_json`.
  - `tests/instrument/test_circumplex.py`: `circumplex()` has positions `("PAQ1", ..., "PAQ8")`, angles `(0.0, 45.0, ..., 315.0)`, axes ids `("ISOPleasant", "ISOEventful")`, scale 1..5 ordinal; `label("PAQ6") == "monotonous"`, `label("PAQ1", "deu") == "pleasant"` (fallback); `position_for("Calm") == "PAQ8"`, `position_for("PAQ3") == "PAQ3"`, `position_for("loud")` raises `MappingError`; validators reject non-contiguous positions, non-increasing angles, an angle of 360.0, a missing quadrant key, an axis naming an unknown position, a translation for `eng`, and a dict without `schema`; `Circumplex.model_validate(c.model_dump()) == c` and `Circumplex.model_validate(c.model_dump(by_alias=True)) == c`; `circumplex(version="9")` raises `DataReferenceError`; `test_circumplex_toml_round_trip` (dump and reload equal, equal hash, and the dump's first key is `schema`).
  - `tests/test_data.py`: `test_every_shipped_file_matches_its_pin`; `test_every_shipped_file_is_pinned`: the shipped list is `[p for p in root.rglob("*") if p.is_file() and p.suffix not in {".py", ".pyc"} and "__pycache__" not in p.parts]` (Python writes `__pycache__` under the editable src tree), each has a `PINS` entry, and every entry exists as a file; `test_tampered_bytes_raise` (monkeypatch the reader to return altered bytes: `DataIntegrityError` whose message holds both hashes); `test_unregistered_path_raises`.
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
- Produces: `project_scores(scores: NDArray[np.float64], angles_deg: NDArray[np.float64], scale: Scale, formulation: Literal["canonical", "iso2019"]) -> NDArray[np.float64]` (§6.9.1). `scores` and `angles_deg` are (n, k) in position order; returns (n, 2) float64; rows with any NaN score return NaN. canonical and iso2019 exactly as the §6.9.1 formulas; iso2019 is computed from its own formula (not routed through canonical), so the exhaustive test compares two implementations. `iso2019` raises `FormulationError` unless the scale is 1..5 and every row's angles equal `(0, 45, ..., 315)`. Plan call: `FormulationError` naming the axis when Σ|cos θ_i| < 1e-9 (x) or Σ|sin θ_i| < 1e-9 (y) in any row (§6.3.10 row amended), never an exact `== 0` test: after `deg2rad`, cos and sin of 90°, 180° and 270° leave float64 residues of 6e-17 to 1.2e-16 per term (angles (0, 180) give Σ|sin| = 1.2e-16; (90, 270) give Σ|cos| = 2.4e-16), and 1e-9 sits far below any real angle set's sums, which are O(1). Only a bare kernel call can reach it: a validated circumplex has k distinct increasing angles. Shape mismatches raise `ValueError`. No clipping or clamping. Plan call: the `aletta2024` formulation and its `Literal` member are S3's (§9 S3 row), so S1's `Literal` has two members. Docstring `References`: `[@mitchell2026translation]` (canonical, proof p. 23, §4.5), `[@iso12913-3-2019]` and `[@mitchell2022how]` (Eqs. 1 and 2).

- [ ] **Step 1: Write the failing tests.** `tests/science/test_projection.py` has `pytestmark = pytest.mark.science` and cites its sources in docstrings:
  - `test_iso2019_hand_values`: (p, v, e, ch, a, m, u, ca) = (5,3,3,3,1,3,3,3) gives x = √2 − 1, y = 0; (4,5,2,1,2,1,4,5) gives x = (1 + 2√2)/(2 + 2√2), y = −1/(2 + 2√2) (closed forms, §11.4), both under `iso2019` and `canonical`, `pytest.approx(abs=1e-12)  # float64 sums of 8 terms; closed forms are exact`. Docstring: derived by hand from the ISO/TS 12913-3:2019 formula.
  - `test_canonical_equals_iso2019_exhaustively`: all 5^8 = 390,625 responses (`itertools.product(range(1, 6), repeat=8)` into one array), `np.testing.assert_allclose(canon, iso, rtol=0, atol=1e-12)  # spec §7.2; measured max 2.8e-16`.
  - Hypothesis properties (§7.1). The canonical strategy draws a scale (ordinal integer bounds, or interval floats in [−10, 10] with `maximum − minimum >= 0.5`), eight strictly increasing angles in `[0, 360)` (sorted unique floats rounded to 0.01°), and scores inside the scale. `test_canonical_square_bounds` (`|x| <= 1 + 1e-12`, `|y| <= 1 + 1e-12  # spec §7.1: float rounding of sums of 8 terms`). `test_neutral_maps_to_origin` and `test_constant_shift` each run under canonical (drawn scale and angles) and under iso2019, where the scale is fixed at `Scale(minimum=1.0, maximum=5.0)`, the angles at (0, 45, ..., 315), scores are integers in 1..5 and the shift c is an integer keeping every item in 1..5. Neutral: all items at the midpoint give (0, 0). Shift: adding c moves x by `c·Σcos θ_i / (h·Σ|cos θ_i|)` and y likewise, which is 0 on both axes under iso2019 because Σcos θ_i = Σsin θ_i = 0 at 45° steps. All `atol=1e-12  # float rounding of sums of 8 terms`.
  - `tests/projection/test_formulas.py`: NaN in one item gives a NaN row only for that row; `iso2019` on `Scale(minimum=1.0, maximum=7.0)` raises `FormulationError`; `iso2019` on non-nominal angles raises `FormulationError`; shape mismatch raises `ValueError`; output is float64 with shape (n, 2); `test_degenerate_angle_sums_raise`: k = 2 rows with angles (90, 270) raise `FormulationError` naming x, and (0, 180) naming y, under canonical; `test_no_clamping`: (5,5,5,1,1,1,1,5) gives x = 1, y = √2 − 1 under both formulations, radius √(4 − 2√2) = 1.0823922 (§6.9.1's maximum on the 1..5 grid), each `pytest.approx(abs=1e-12)  # closed form; float64 sums of 8 terms`, and `np.hypot(x, y) > 1.08`, so a clip to the unit circle fails. Do not pin x's float64 residual: it depends on the summation order.
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
- Produces (§6.3.1): `Reason` and `Flag` as `enum.StrEnum` with the normative codes in the normative order; `Status` frozen dataclass (`eq=False`, `slots=True`) with `included: NDArray[np.bool_]` (read-only), `reasons: tuple[tuple[Reason, ...], ...]`, `flags: tuple[tuple[Flag, ...], ...]`. `__post_init__` checks equal lengths, `included[i] == (len(reasons[i]) == 0)`, codes unique and in enum order (else `ValueError`). `Status.all_included(n: int) -> Status` (Plan call: the constructor helper readers use). `with_reason(mask: ArrayLike, reason: Reason) -> Status`, `with_flag(mask, flag) -> Status` (new objects, codes kept unique and ordered). `counts() -> dict[str, int]`: Plan call, keys `"n"`, `"included"`, `"excluded"`, then every `Reason` and every `Flag` code (zero when absent), so the keys never depend on the data. `label_arrays() -> dict[str, NDArray[Any]]` (Plan call): `{COL_INCLUDED: read-only bool array, COL_REASONS: read-only object array of codes joined with ";" ("" when none), COL_FLAGS: likewise}`, the one place the text form is built. `to_frame() -> Any` is `_frames.new_frame(self.label_arrays())` (pandas).

- [ ] **Step 1: Write the failing tests**: the two code tuples equal §6.3.1's lists in order; `with_reason` on an already-excluded row keeps one copy; adding reasons out of order stores them in enum order; `label_arrays()[COL_REASONS]` for a row with two reasons is `"scale_violation;all_identical"` and `to_frame()` holds the same texts; `counts()` on an all-included status of 3 rows; hypothesis `test_status_invariants` (random sequences of `with_reason`/`with_flag` on random masks keep `included == (len(reasons) == 0)`, unique ordered codes, length n); arrays read-only; a mismatched `included` raises `ValueError`.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**, `uv run task check` green.
- [ ] **Step 5: Commit** `feat: per-response Status with Reason and Flag codes`.

---

### Task 7: `SurveyMapping`, `Coding`, mapping TOML

**Files:**
- Create: `src/soundscapy/survey/mapping.py`, `tests/survey/test_mapping.py`, `tests/survey/fixtures/minimal_mapping.toml`, `tests/survey/fixtures/nondefault_mapping.toml`

**Interfaces:**
- Consumes: `FrozenModel`, `StrPairs` and the value-map alias (Task 1); `Scale`, `circumplex` (Task 4); `MappingError`, `CodingError`, `DataReferenceError`.
- Produces (§6.4.1, S1 subset; Plan call on scope: S1 maps attributes, ids and language, as §9 S1 says; S2 adds `source_dominance`, `overall_quality`, `appropriateness`, `who5`, `age`, `flags` and `carry`, and the full ISD mapping):
  - `SurveyMapping`: `schema_name: Literal["soundscapy.mapping"]` as `Field(alias="schema")` (Plan call and a spec defect, §6.4.1 amended: a pydantic field named `schema` emits "shadows an attribute in parent BaseModel", a `UserWarning` that `filterwarnings = ["error"]` turns into a failure; with an alias the TOML key and the hash input stay `schema`, using `populate_by_name=True`, which pydantic 2.7 has, and `by_alias=True` on every dump), `schema_version: Literal[1]`, `instrument: StrictStr = "iso-12913-outdoor"`, `instrument_version: StrictStr = "1"`, `ids: IdSpec`, `language: LanguageSpec = LanguageSpec()`, `groups: Groups`.
  - `IdSpec`: `response`, `response_key`, `response_key_blank_ok`, `location`, `session`, `group`, `record` per §6.4.1 (all of it: the id rules are S1's). Validators: not both `response` and `response_key`; `response_key_blank_ok` a proper subset of `response_key`.
  - `LanguageSpec`: `column`, `value`, `coding` (not both `column` and `value`; `value` and every coded value match `^[a-z]{3}$`).
  - `Groups`: `attributes: ItemGroup` (required). `ItemGroup`: `columns: StrPairs`, `scale: Scale` (no default, and no instrument-dependent validator: `ItemGroup` does not know the instrument), `coding: Coding = Coding()`.
  - Plan call, the instrument-aware step: a `mode="before"` model validator on `SurveyMapping` resolves `circumplex(instrument, instrument_version)` from the raw input (field defaults when the keys are absent; an unknown instrument raises `DataReferenceError`, a `LookupError`, which propagates unwrapped). It writes `circumplex.scale` into `groups.attributes.scale` when that key is absent, so the stored model always holds a concrete `Scale`; it normalises the attribute keys (positions or reference-language labels, via `position_for`) to positions in position order; a label and its position both present (`pleasant` and `PAQ1`) raise `ValueError` naming both keys; an unmapped position raises `ValueError` naming the position and its label. `read_mapping` and `read_survey` surface these `ValueError`s as `MappingError`.
  - `Coding` (all §6.4.1 fields) and `apply_coding(texts: NDArray[np.object_], coding: Coding, scale: Scale, *, item_keys: Sequence[str], response_ids: NDArray[np.object_]) -> NDArray[np.float64]` applying §6.4.2 steps 1 to 5 to an (n, k) text array; `response_ids` feed the `CodingError` message. Plan call: the whole coding order ships in S1 (it is a few lines and `Coding` is an S1 deliverable); S2 adds the mapping-level tests for missing codes, value maps and reverse on real exports.
  - `read_mapping(path: str | os.PathLike[str]) -> SurveyMapping` and `write_mapping(mapping: SurveyMapping, path: str | os.PathLike[str]) -> None`, with tomlkit and the four §6.4.4 writer rules; `write_mapping` omits `scale` when it equals `circumplex(instrument, instrument_version).scale` and writes attribute keys as that circumplex's labels; a blank value raises `MappingError` naming the key.

- [ ] **Step 1: Write the failing tests**
  - `test_label_and_position_keys_hash_alike`: `{"pleasant": "x", ...}` and `{"PAQ1": "x", ...}` give equal models and equal `hash_json`.
  - `test_label_and_position_same_item_raises` (`pleasant` and `PAQ1` both mapped: `MappingError` via `read_mapping` naming both).
  - `test_unmapped_position_raises` (`read_mapping` of a file missing `calm`: `MappingError` naming `PAQ8`/`calm`); `test_unknown_instrument_raises` (`DataReferenceError`); `test_bad_language_code_raises` (`coding = { English = "ENG" }`); `test_response_and_key_together_raise`; `test_blank_value_raises_naming_key`.
  - `test_write_mapping_byte_round_trip`: `tests/survey/fixtures/minimal_mapping.toml` (identity keys, `[ids]` with the ISD four-part `response_key` and roles, `[language] column = "Language"`, `[groups.attributes.columns]` by English name in circumplex order) reads, writes and compares byte for byte; the writer emits `pleasant`, never `PAQ1`; default-valued fields (empty `coding`, the default `scale`) are omitted. Plan call: S1 writes the default scale by omission; a non-default scale is written as a `[groups.attributes.scale]` table.
  - `test_mapping_model_round_trip`, parametrised over both fixtures (`nondefault_mapping.toml` has a 1..7 interval scale, a `value_map`, an `offset`, a `reverse` key, `decimal = ","` and a language `coding`): `read_mapping` of the written file equals the original model with an equal `hash_json`, and the non-default file also round-trips byte for byte.
  - `test_apply_coding_isd_forms`: `["4", "4.0", " 4 ", "", "NA"]` give `[4, 4, 4, nan, nan]` (exact); `"3,5"` with `decimal="."` raises `CodingError` listing `"3,5"`, its count and up to five response ids; with `decimal=","` gives 3.5.
  - Hypothesis `test_coding_order` (§7.1): value-map targets and offsets come from `st.integers(-20, 20).map(lambda k: k / 2)` (half-integers in [−10, 10]) and scale bounds are integer-valued floats, so every intermediate value is a small dyadic rational and float64 sums and differences are exact. Assert exactly that `apply_coding` equals `minimum + maximum - (value_map[text] + offset)` for reversed keys and `value_map[text] + offset` for the others, and that applying `minimum + maximum - s` twice returns `s`. Comment: exactness holds because the strategy draws dyadic values (`6 - (6 - 0.3)` is `0.2999999999999998` in float64).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**, `uv run task check` green.
- [ ] **Step 5: Commit** `feat: minimal SurveyMapping, Coding and mapping TOML`.

---

### Task 8: `read_survey` from a path, `response_id`, `Survey`

**Files:**
- Create: `src/soundscapy/survey/read.py`, `tests/survey/test_read.py`, `tests/survey/fixtures/*.csv` (small hand-written CSVs)

**Interfaces:**
- Consumes: Tasks 1 to 7 (`read_text_table`, `column_text`, `apply_coding`, `Status`, `envelope`, `hash_file`, `hash_json`, `logger`, `circumplex`); `_data.pin` (Task 4).
- Produces:
  - `ReadOptions(FrozenModel)`, S1 fields: `format: Literal["auto", "csv", "tsv"] = "auto"` (by suffix; any other suffix raises `FileFormatError`), `encoding: StrictStr = "utf-8-sig"`, `delimiter: StrictStr = "auto"`. Plan call: in S1 `"auto"` resolves to `,` for csv and tab for tsv; sniffing among `,`, `;`, tab and `|`, `xlsx`, `sheet`, `header_row` and `skip_rows` arrive in S7. `decimal` is not a field (§11.4). The resolved options are recorded.
  - `ImportSettings(FrozenModel)`: `on_scale_violation: Literal["error", "exclude"] = "error"`. Plan call: S2 adds `quality` and the `exclude` path; S1 always raises `ScaleViolationError` on a violation (§6.4.3 message and hint), so a 0..4 export never projects (an anti-goal of §3.3).
  - `SourceRef(FrozenModel)`: `name` (file name), `hash` (raw file `h1`), `read_options: ReadOptions`.
  - `Survey` frozen dataclass (`eq=False`, `slots=True`), fields of §6.5: `data`, `mapping`, `scores: dict[str, NDArray]` (S1: `"attributes"` only, (n, 8), position order, read-only), `response_id`, `ids: dict[str, NDArray]` (mapped roles only), `language`, `status`, `settings`, `source`, `provenance`. Method `n() -> int`. Plan call: `to_frame` and `select` arrive in S2 with the status table; S1 has no caller.
  - `read_survey(source: str | os.PathLike[str], mapping: SurveyMapping | str | os.PathLike[str], *, read_options: ReadOptions | Mapping[str, object] | _Unset = _UNSET) -> Survey`. Plan call: the §6.5 signature's later keywords arrive with their features (S2 `quality`, `on_scale_violation`, `settings`; S3 `profile`), and in-memory frame sources arrive in S7; a non-path source raises `TypeError` naming the S1 path-only limitation. `read_survey` resolves `mapping` (a path through `read_mapping`) and `read_options` (unset or a mapping into `ReadOptions`), then calls `_read_survey`.
  - Private `_read_survey(source: str | os.PathLike[str], mapping: SurveyMapping, *, read_options: ReadOptions, data_refs: Sequence[Mapping[str, str]] = ()) -> Survey`, so `datasets.isd` can add its dataset reference (Plan call; `Survey` is frozen). Its `data_refs` are appended after the circumplex entry.
  - **Behaviour, in order** (ids come before coding because `CodingError` and `ScaleViolationError` list response ids, so id errors take precedence over coding and scale errors):
    1. The `ssp_` collision check (`ColumnCollisionError`, §6.3.3).
    2. Mapped columns missing from the header raise `MappingError` listing each with its mapping key.
    3. Response ids per §6.3.2 rules 1 to 3 (rule 4 is S7's in-memory case): a blank id (a blank or whitespace `response` cell, or a blank key part not in `response_key_blank_ok`) raises `MissingResponseIdError`, a duplicate `DuplicateResponseIdError`, each listing up to five offenders with their row numbers; a key part containing `/` raises `MappingError`.
    4. Language: with `column`, strip each cell; a blank cell gives `""`; a cell whose text is a `coding` key is replaced by its code; any other result not matching `^[a-z]{3}$` raises `MappingError` listing the distinct texts, their counts and up to five response ids. With `value`, every row gets that code. With neither, every row gets `""`.
    5. Attributes coded (§6.4.2) by `apply_coding(..., item_keys=..., response_ids=<the ids of step 3>)`.
    6. The scale check (§6.4.3) on non-missing coded cells only: a cell violates when `~np.isnan(v) & ~scale.contains(v)`; NaN cells are missing, not violations, and are left to `incomplete_attributes`. `ScaleViolationError` gives per column the offending values with counts and up to five response ids.
    7. Status: Plan call, S1 applies the one always-on rule, `incomplete_attributes` (§6.7: "always, not configurable at M1"), so a row with a missing attribute is excluded with a reason rather than included with NaN xy; S2 adds every other rule.
    8. One INFO log line: rows read, blank rows skipped, included, excluded by reason.
  - Provenance: `envelope(operation="read_survey", settings=..., input_hash=source.hash, mapping_hash=hash_json(mapping), data_refs=[{"kind": "circumplex", "name": mapping.instrument, "version": mapping.instrument_version, "hash": _data.pin(f"circumplexes/{mapping.instrument}/{mapping.instrument_version}.toml")}, *data_refs], response_id_source=..., read=<resolved options dump>, id_columns={role: user column for each mapped role}, response_id_columns=<[response] for column, the response_key columns in key order for key, [] for row>, response_key_blank_ok=<list, in key order>)`. Plan call: shipped data is referenced by its registry pin (the file-bytes `h1`, §6.3.7), never by `hash_json` of the parsed model, matching Task 9's dataset entry. The three id keys follow §6.3.6's amended row: `id_columns` is how `split` finds a role's column, the other two are how `attach` rebuilds ids (Tasks 10, 11).

- [ ] **Step 1: Write the failing tests** (fixtures are 3 to 6 row CSVs with the eight attribute columns in the ISD's CSV order):
  - `test_reads_by_name_not_order`: attribute columns in CSV order pleasant, chaotic, vibrant, ... land in position order in `scores["attributes"]`.
  - `test_response_id_from_key` (`"L1/S1/G1/7"`, `response_id_source == "key"`), `test_response_id_from_column` (stripped), `test_response_id_from_row` (header, two rows, a blank row, one row: ids `row-2`, `row-3`, `row-5`).
  - `test_blank_key_part_raises` and `test_blank_key_part_allowed` (listed in `response_key_blank_ok`: id keeps an empty segment, `"L1/S1/G1/"`); `test_blank_response_column_raises` (`ids.response = "ID"`, one whitespace ID cell: `MissingResponseIdError` listing its row number); `test_duplicate_ids_raise` (message holds both row numbers); `test_key_part_with_slash_raises`.
  - `test_id_error_precedes_coding_error` (a file with a duplicate id and an uncodable cell raises `DuplicateResponseIdError`).
  - `test_language_from_column_with_coding` (`"English"` coded to `"eng"`; a blank cell gives `""`); `test_language_uncoded_text_raises` (`"English"` with no coding: `MappingError` listing it); `test_language_from_value`.
  - `test_ssp_column_collision_raises`; `test_missing_mapped_column_raises`; `test_unknown_suffix_raises` (`.txt` with `format="auto"`: `FileFormatError` naming the suffix); `test_frame_source_raises_type_error` (a pandas frame as `source`: `TypeError` naming the path-only limitation, with the ty ignore of Global Constraints); `test_scale_violation_raises` (a 0..4 file: message names the column, the value `0`, its count and the response id of an offending row; hint names `offset` and `on_scale_violation="exclude"`; a blank cell in the same file is not reported).
  - `test_incomplete_attributes_excluded` (one blank attribute: `included` false, reason `incomplete_attributes`); `test_info_summary_logged` (`caplog`, one record at INFO); `test_survey_arrays_read_only`; `test_provenance_keys` (§6.3.6 keys of Task 2 plus `response_id_source`, `read`, `id_columns`, `response_id_columns`, `response_key_blank_ok` for a key-source and a column-source file; `input_hash == hash_file(path)`; `data_refs[0] == {"kind": "circumplex", "name": "iso-12913-outdoor", "version": "1", "hash": _data.PINS["circumplexes/iso-12913-outdoor/1.toml"]}`).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**, `uv run task check` green.
- [ ] **Step 5: Commit** `feat: read_survey from a CSV path with response ids`.

---

### Task 9: ISD subset generator, bundled subset, `datasets.isd.load(subset=True)`

**Files:**
- Create: `scripts/make_isd_subset.py`, `src/soundscapy/_data/datasets/isd/subset-v1.0.csv`, `src/soundscapy/_data/datasets/isd/subset-v1.0.toml`, `src/soundscapy/_data/datasets/isd/registry.toml`, `src/soundscapy/_data/datasets/isd/isd_mapping.toml`, `src/soundscapy/datasets/__init__.py`, `src/soundscapy/datasets/isd.py`, `tests/datasets/test_isd.py`, `tests/science/test_isd_subset.py`
- Modify: `src/soundscapy/_data/__init__.py` (four pins)

**Interfaces:**
- Consumes: `_read_survey`, `ReadOptions` (Task 8); `read_mapping` (Task 7); `_data.as_path`, `read_toml`, `pin` (Task 4).
- Produces: `soundscapy.datasets.isd.load(*, subset: bool = False, version: str | _Unset = _UNSET) -> Survey`. `version` unset means the registry `default`. With `subset=True` only `"1.0.1-alpha.1"` is accepted, else `DataReferenceError` listing it. Plan call: `subset=False` raises `NotImplementedError` whose message says the full download arrives with `datasets.isd.fetch` (S9) and to pass `subset=True`; S2 adds `quality`, `on_scale_violation`, `settings`, S3 `profile`, S9 `locations` and the full path. `load` reads `registry.toml` through `read_toml`, takes `versions[version].mapping` (a file name relative to `datasets/isd/`) and `versions[version].read_options`, reads the mapping with `read_mapping` on `_data.as_path("datasets/isd/" + mapping)`, and calls `_read_survey` with `ReadOptions(**read_options)` (an empty table in S1) (§6.20.2: "reads with the mapping file and read options its registry entry names"; never hardcode the mapping file name). The survey's `data_refs` add `{kind: "dataset", name: "isd-subset", version: "1.0.1-alpha.1", hash: pin("datasets/isd/subset-v1.0.csv")}`. `soundscapy.datasets` imports `isd`.
- **Generator** (§6.20.1): PEP 723 script (`dependencies = ["tomlkit>=0.13"]`), stdlib otherwise. `--source PATH` or a download of `https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv`; refuses unless the source sha256 is `27e54cde99fba274c0be32d1d81e944ad35d7402cf4a40ed763bbf4dae90f735`. Keeps rows whose `LocationID` is one of the 15 §6.20.1 locations and the 40 §6.20.1 columns, cell text verbatim. Plan call: rows in source order and columns in source order (so `LCeq_L(C)` precedes `LAeq_L(A)`, as in the source header); written with `csv` (`lineterminator="\n"`, minimal quoting), UTF-8 without BOM. The sidecar `subset-v1.0.toml`: Plan call on keys beyond the normative `[citation]` block: `schema = "soundscapy.dataset_sidecar"`, `schema_version = 1`, `name = "isd-subset"`, `source_version = "1.0.1-alpha.1"`, `source_record = "10672568"`, `source_file`, `source_sha256`, `subset_hash` (`h1-sha256:` of the CSV, computed in the script with `hashlib`), `n_rows = 1552`, `locations`, `columns`, `[features] columns = [...]`, `units_verified = false`, then `[citation]` verbatim from §6.20.1. Running it twice gives identical bytes.
- **`registry.toml`**: Plan call, S1 ships the normative header (`schema`, `schema_version`, `concept_doi`, `default = "1.0.1-alpha.1"`) and only the `[versions."1.0.1-alpha.1"]` entry, every §6.20.3 key filled (`mapping = "isd_mapping.toml"`, `read_options = {}` among them); its `feature_columns` are the 96 header columns after `RecordingLength`. S9 adds the `1.0.1-alpha` and `0.2.3` entries with fetch.
- **`isd_mapping.toml`**: Plan call, lives under `_data/datasets/isd/` (§11.4) and in S1 holds the §6.4.4 example's identity keys, `[ids]`, `[language]` and `[groups.attributes.columns]` only; S2 extends it to the full normative example and re-pins it.

- [ ] **Step 1: Generate the data and check determinism**

```bash
curl -sS -o "$SCRATCH/isd-1.0.1-alpha.1.csv" "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv"
uv run scripts/make_isd_subset.py --source "$SCRATCH/isd-1.0.1-alpha.1.csv"
sha256sum src/soundscapy/_data/datasets/isd/subset-v1.0.csv src/soundscapy/_data/datasets/isd/subset-v1.0.toml > "$SCRATCH/first.sha"
uv run scripts/make_isd_subset.py --source "$SCRATCH/isd-1.0.1-alpha.1.csv"
sha256sum -c "$SCRATCH/first.sha"
```

Expected: both files report `OK` (`git diff` cannot show this: the files are untracked on the first run). (Checked while writing this plan: the source has 3589 rows; the 15 locations give 1552, eng 1213, spa 126, nld 89, cmn 69, ita 55; 1350 rows carry `LAeq_L(A)`; 16 rows have a blank attribute.) Pin the four files in `_data.PINS`.

- [ ] **Step 2: Write the failing tests**
  - `tests/science/test_isd_subset.py` (`science` marker): the CSV matches its pin; it has 1552 rows and the 40 columns; language counts `{"eng": 1213, "spa": 126, "nld": 89, "cmn": 69, "ita": 55}`; the 15 locations exactly; 1350 non-blank `LAeq_L(A)`; the sidecar `[citation]` equals §6.20.1's block and its `subset_hash` equals the CSV's pin.
  - `tests/datasets/test_isd.py`: `load(subset=True)` returns a `Survey` with `n() == 1552`, `response_id_source == "key"`, all ids unique, the dataset `data_refs` entry; 16 rows excluded by `incomplete_attributes` (Plan call: S1 asserts only this always-on count; S2 pins the full 1552/16/0/10/8 set); `load(subset=True, version="1.0.1-alpha")` raises `DataReferenceError`; `load()` raises `NotImplementedError`; `test_load_uses_the_registry_mapping` (monkeypatch the parsed registry entry's `mapping` to `"missing.toml"`: `load(subset=True)` raises `DataReferenceError` for the unregistered path); `isd_mapping.toml` round-trips byte for byte through `read_mapping`/`write_mapping` (§6.4.4).
- [ ] **Step 3: Run to verify they fail**, **Step 4: implement `datasets/isd.py`**, **Step 5: pass**; `uv run task check` green; `uv run task build` and `unzip -l dist/*.whl | grep -E 'subset-v1.0.csv|1.toml|registry.toml'` lists the data files.
- [ ] **Step 6: Commit** `feat: bundled ISD subset, its generator and load(subset=True)` (body: CC BY 4.0, record 10672568, sha256 checked).

---

### Task 10: `AngleSelection` (equal only), `ProjectionSettings`, `project`, `Coordinates`, `OriginSegment`

**Files:**
- Create: `src/soundscapy/instrument/angles.py`, `src/soundscapy/projection/project.py`, `src/soundscapy/distribution/__init__.py`, `src/soundscapy/distribution/coordinates.py`, `tests/projection/test_project.py`, `tests/distribution/test_coordinates.py`, `tests/test_toml_round_trip.py`, `tests/projection/fixtures/` (CSVs and a 1..7 mapping)
- Modify: `pyproject.toml` (add the "distribution never imports survey" contract of §5.2.1 verbatim, now that its source module exists)

**Interfaces:**
- Consumes: `Survey`, `Status.label_arrays()` (Tasks 6, 8); `project_scores` (Task 5); `circumplex`, `Circumplex`, `Scale` (Task 4); namespace constants; `envelope`, `hash_json`; `_frames.new_frame`.
- Produces:
  - `soundscapy.instrument.angles.AngleSelection(FrozenModel)`: `default: Literal["equal"] = "equal"`. Plan call: S3 widens `default` to `AngleSetRef | Literal["equal"]` (taking its default from the profile, `satp@1.5` under `sal-2026`) and adds `per_language`, `on_uncovered`, `on_low_confidence`; S1's default `"equal"` is provisional and the S3 change is a recorded behaviour change.
  - `soundscapy.projection.project.ProjectionSettings(FrozenModel)`: `formulation: Literal["canonical", "iso2019"] = "canonical"`, `angles: AngleSelection = AngleSelection()`.
  - `project(survey: Survey, *, formulation: Literal["canonical", "iso2019"] | _Unset = _UNSET, angles: Literal["equal"] | AngleSelection | _Unset = _UNSET) -> Coordinates`. A string other than `"equal"` (or a bad formulation) raises `SettingsError` naming the field and saying which values this build accepts, in soundscapy's own words (not pydantic's message). Plan call: `on_uncovered`, `on_low_confidence`, `angle_sets`, `settings`, `profile` arrive in S3 and `carry` in S2. The instrument is `circumplex(survey.mapping.instrument, survey.mapping.instrument_version)`, stored as `Coordinates.circumplex`. The scale passed to `project_scores` and stored as `OriginSegment.effective_scale` is the declared item scale `survey.mapping.groups.attributes.scale` (Task 7 always fills it in), never `circumplex.scale` (§6.9.1: "taken from the declared scale"). Angles are `np.broadcast_to(np.asarray(circ.angles_deg, dtype=np.float64), (n_included, k))`. Projects included rows only (excluded rows stay NaN); the Coordinates' status is the survey's (no projection reasons arise with equal angles); the Survey is unchanged. Docstring `References` as Task 5.
  - `soundscapy.distribution.coordinates.OriginSegment(FrozenModel)` (§6.10 fields; `n: StrictInt`; `angle_sets` here `("equal",)`; `dataset` is `SourceRef.name`).
  - `Coordinates` frozen dataclass (`eq=False`, `slots=True`): `xy`, `circumplex`, `labels`, `origin`, `settings: FrozenModel`, `provenance` (§6.10). Labels for kind `survey`: `ssp_response_id`, `ssp_included`, `ssp_reasons`, `ssp_flags` (these three from `survey.status.label_arrays()`, never re-joined), `ssp_language`, `ssp_angle_set` (`"equal"` for projected rows, `""` for rows not projected: Plan call), then each mapped id role under the user's column name. Constructor copies every array and makes it read-only.
  - Module functions `valid_mask(coords: Coordinates) -> NDArray[np.bool_]` (read-only; `labels[COL_INCLUDED]` and both xy columns finite), the one definition of a valid row (§6.3.1); `n(coords: Coordinates) -> int` and `n_valid(coords: Coordinates) -> int` (`int(valid_mask(coords).sum())`); methods `valid_mask()`, `n()` and `n_valid()` call them. `to_frame(like: object | None = None) -> Any` (labels then `ssp_ISOPleasant`, `ssp_ISOEventful`); `select(mask: ArrayLike) -> Coordinates` (origin `n` updated); `split(by: str) -> dict[str, Coordinates]`, sorted keys, `by` an id role, `"language"`, or a label column name; an unknown `by` raises `MappingError` listing the choices.
  - Plan call: `split("location")` finds `LocationID` through `provenance["id_columns"]["location"]` (§6.3.6 amended; §6.10's fields give no other place). Coordinates provenance: `operation="project"`; `input_hash`, `mapping_hash`, `data_refs`, `response_id_source`, `read`, `id_columns`, `response_id_columns` and `response_key_blank_ok` copied unchanged from the survey; `settings_hash` of the `ProjectionSettings`.

- [ ] **Step 1: Write the failing tests**
  - `tests/projection/test_project.py`: on a fixture CSV holding the two §7.2 hand-value responses, `project(read_survey(...), formulation="iso2019", angles="equal").xy` equals the closed forms (`abs=1e-12`, as Task 5); default formulation is canonical and gives the same values; an excluded (incomplete) row has NaN xy and `ssp_angle_set == ""`; the survey's arrays are unchanged.
  - `test_project_uses_the_declared_scale`: a fixture mapping with `[groups.attributes.scale] minimum = 1.0, maximum = 7.0, kind = "ordinal"` and a row (7,4,4,4,1,4,4,4) projects under canonical to x = √2 − 1, y = 0 (`abs=1e-12  # closed form`; the circumplex's 1..5 scale would give x = 3/(2 + 2√2) = 0.6213); the same survey with `formulation="iso2019"` raises `FormulationError`; `coords.origin[0].effective_scale == survey.mapping.groups.attributes.scale`.
  - `test_project_rejects_unknown_angles`, parametrised over `"satp@1.5"` and `"aletta-2024@1"` (each call with the ty ignore of Global Constraints): `type(exc) is SettingsError`; `"angles"` and `'"equal"'` both in `str(exc)`; `"validation error"` (pydantic's banner) not in `str(exc)`. A bad `formulation` likewise.
  - `project(..., angles=AngleSelection())` equals `angles="equal"`; `coords.settings` is a `ProjectionSettings` and `provenance["settings_hash"] == hash_json(coords.settings)`.
  - `test_project_row_order_invariance`: a fixture CSV whose mapping sets `[ids] response = "ID"` (a `column` source, so ids travel with their rows; `row-<n>` ids follow physical position by §6.3.2 rule 3 and are outside this property, §7.1 as amended), with at least one incomplete row. `@given(st.permutations(range(n)))` with `@settings(deadline=None)  # per-example file I/O`; each permuted CSV is written inside the test body under `tempfile.TemporaryDirectory()` (a function-scoped fixture such as `tmp_path` fails hypothesis's health check). `labels["ssp_response_id"]`, `status.included`, the reasons and `xy` equal the unpermuted result indexed by the permutation, exactly (`np.testing.assert_array_equal`, which treats NaN as equal).
  - On the subset: `n() == 1552`, `n_valid() == 1536`, `valid_mask` sums to 1536 and is False on every excluded row, and every finite x and y satisfies `np.abs(v) <= 1 + 1e-12  # spec §7.1 square-bound tolerance: float rounding of sums of 8 terms can put |x| one ulp above 1, depending on the summation order`.
  - `tests/distribution/test_coordinates.py`: arrays read-only; `split("location")` on the subset has the 15 locations in sorted order and sizes summing to 1552; `split("language")` has five keys; `split("nonsense")` raises `MappingError`; `select` updates `origin[0].n`; `to_frame()` has the label columns plus `ssp_ISOPleasant`/`ssp_ISOEventful`; `to_frame(like=<polars frame>)` is polars.
  - `tests/test_toml_round_trip.py` (§7.1 "every FrozenModel", S1 done-when "TOML round-trips"): `test_every_frozen_model_round_trips`, parametrised over one non-default instance of each S1 model: `Citation`, `Scale`, `Attribute`, `Axis`, `Translation`, `Circumplex`, `Coding`, `IdSpec`, `LanguageSpec`, `ItemGroup`, `Groups`, `SurveyMapping`, `ReadOptions`, `ImportSettings`, `SourceRef` (built from the subset CSV), `AngleSelection`, `ProjectionSettings`, `OriginSegment`; asserts `from_toml(type(m), to_toml(m)) == m` and equal `hash_json` (exact). Hypothesis variants for `ReadOptions` (format in auto/csv/tsv, any encoding and delimiter text) and `ImportSettings` (both literals). `test_round_trip_covers_every_model`: after importing every `soundscapy` module, walk `FrozenModel.__subclasses__()` recursively, keep classes whose `__module__` starts with `soundscapy.`, and fail naming any class with no instance in the parametrisation, so later slices must extend it.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement**, and add the contract to `pyproject.toml`. **Step 4: Pass**; `uv run task check` green (`lint-imports` reports five contracts kept).
- [ ] **Step 5: Commit** `feat: project with equal angles into Coordinates`.

---

### Task 11: `attach`

**Files:**
- Create: `src/soundscapy/workspace/__init__.py`, `src/soundscapy/workspace/_io.py`, `tests/workspace/test_attach.py`

**Interfaces:**
- Consumes: `Coordinates` (Task 10); `_frames.to_eager`, `column_text`, `with_columns` (with `valid`); namespace constants; `AlignmentError`, `ColumnCollisionError`; `logger`.
- Produces: `attach(frame: Frame, result: Coordinates) -> Any` (§6.16), in `workspace/_io.py`, re-exported by `soundscapy.workspace`. Appends `ssp_<axis id>` for both axes, `ssp_included`, `ssp_reasons`, `ssp_flags`, `ssp_language`, `ssp_angle_set` to a new frame in the user's library. Alignment by `provenance["response_id_source"]`:
  - `column`/`key`: each frame id is built from exactly the columns in `provenance["response_id_columns"]`, in that order: each cell through `column_text`, stripped, parts joined with `/`; a blank part is allowed only for a column in `provenance["response_key_blank_ok"]`. A listed column missing from the frame raises `AlignmentError` naming it. The join keeps the frame's row order; frame rows whose id is not in the result get nulls in every added column (`with_columns(..., valid=matched)`).
  - `row`: aligns by position and needs the survey's row count.
  - `AlignmentError` when an id column is missing, the frame's ids are not unique, the row count differs for `row`, or (Plan call) no frame row matches. One INFO line with the alignment (`"id"` or `"position"`) and the matched count. Plan call: the per-response `Table` form of `attach` arrives with `Table` (S5/S6).

- [ ] **Step 1: Write the failing tests**: `test_attach_to_survey_data` (subset: `attach(survey.data, coords)` has 40 + 7 columns, `ssp_ISOPleasant` equals `coords.xy[:, 0]` with NaN where excluded, the user's frame is unchanged); `test_attach_to_reread_frame_with_int_ids` (`pd.read_csv` of the subset, rows reversed: every row matches and values follow the frame order); `test_attach_polars` (`frame_lib`); `test_attach_partial_frame` (half the rows: every row matched, matched count logged); `test_attach_frame_with_extra_rows` (`frame_lib`: one extra row whose id is not in the result is null in every `ssp_` column, `ssp_included` included, and the logged matched count excludes it); `test_attach_no_match_raises`; `test_attach_row_ids_by_position` and its row-count `AlignmentError`; `test_attach_column_source` (a `response = "ID"` survey, where `ID` is no id role); `test_attach_key_not_equal_to_roles` (`response_key = ["site", "resp_no"]` with only `location = "site"` mapped); `test_attach_never_overwrites` (frame already holding `ssp_ISOPleasant`: `ColumnCollisionError`); `test_attach_missing_key_column_raises` (the `AlignmentError` names the dropped column from `provenance["response_id_columns"]`).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Pass**; `uv run task check` green.
- [ ] **Step 5: Commit** `feat: attach coordinates to the user's frame`.

---

### Task 12: `plot.scatter`, the top-level API, the quick-start

**Files:**
- Create: `src/soundscapy/plot/__init__.py`, `src/soundscapy/plot/_common.py`, `src/soundscapy/plot/circumplex.py`, `tests/plot/test_scatter.py`
- Modify: `src/soundscapy/__init__.py`, `tests/test_package.py`, `docs/user_guide/_quickstart.py`, `docs/user_guide/01-install.qmd`, `tests/test_docs_structure.py`

**Interfaces:**
- Consumes: `Coordinates`, `split`, `valid_mask` (Task 10); `Circumplex.label`.
- Produces:
  - `soundscapy.plot.scatter(coords: Coordinates, *, by: str | _Unset = _UNSET, ax: Axes | None = None, label_language: str | None = None, **kwargs: Any) -> Axes` (§6.15). matplotlib is imported inside the function; type hints under `TYPE_CHECKING`. `ax=None` builds `matplotlib.figure.Figure()` and `fig.add_subplot()`, no pyplot. `by` unset or `""`: one series, `coords.xy[valid_mask(coords)]`. Otherwise one `ax.scatter` per group from `coords.split(by)`, in sorted key order, each `g.xy[valid_mask(g)]`, labelled with the key, and a legend. Plan call: colours follow `matplotlib.color_sequences["petroff10"]` in order; on each full pass through the ten colours the marker changes (`"o"`, `"s"`, `"^"`, cycled), so every (colour, marker) pair is unique for up to 30 groups and the subset's 15 locations stay distinguishable in the quick-start plot; a user `color` or `marker` kwarg overrides the cycle; the docstring says so. Plan call: a group with no valid rows still gets an empty labelled series, so a group keeps its colour and marker across plots. `**kwargs` go to each `ax.scatter` call. Returns the Axes; never calls `show()`.
  - `plot._common.circumplex_axes(ax: Axes, circumplex: Circumplex, label_language: str | None) -> None`: the ±1 square limits, equal aspect, axis lines through the origin; at each end of each axis the label of the attribute there, `circumplex.label(axis.positive, lang)` at +1 and `circumplex.label(axis.negative, lang)` at −1, x axis then y axis (§6.2.3 last paragraph; §6.15 "axis and quadrant labels in the chosen language"); the axis ids as axis titles (`set_xlabel`/`set_ylabel`); the quadrant labels (`circumplex.label` of each quadrant position). `lang` is `label_language`, or `""` (the reference language) when it is `None`. `check_n` is S5's.
  - `soundscapy/__init__.py`: imports `plot`, `datasets`, `workspace`, `errors` eagerly (§6.21) and exposes, in `__all__`, the S1 names of the §6.21 table: `read_survey`, `read_mapping`, `write_mapping`, `load_isd` (= `datasets.isd.load`), `Survey`, `SurveyMapping`, `Coding`, `ReadOptions`, `Circumplex`, `Scale`, `AngleSelection`, `ProjectionSettings`, `Coordinates`, `circumplex`, `project`, `attach`, `show_versions`, `set_log_level`, `__version__`, `plot`, `datasets`, `workspace`, `errors`. Later slices add the rest.
  - `docs/user_guide/_quickstart.py`: exactly the three normative lines of §6.21 (keep any blank line ruff's isort inserts after the import).

- [ ] **Step 1: Write the failing tests**
  - `tests/plot/test_scatter.py` (§7.6): `test_offsets_equal_valid_xy` (ungrouped: the one collection's offsets equal `coords.xy[valid_mask(coords)]`, exact); `test_grouped_offsets` (by `"location"`: concatenated offsets in sorted key order equal each group's `g.xy[valid_mask(g)]`, legend texts equal the keys); `test_default_colours_are_petroff10`; `test_more_than_ten_groups_are_distinct` (the subset by `"location"`: the 15 (facecolor, marker path) pairs are pairwise distinct, and the legend handles carry the same pairs); `test_no_pyplot_figure_created` (`plt.get_fignums() == []` after `ax=None`); `test_returns_the_given_ax`; `test_kwargs_reach_artist` (`color="k"`); `test_axis_and_quadrant_labels` (pleasant, annoying, eventful, uneventful and vibrant, chaotic, monotonous, calm all among the Axes texts; `label_language="deu"` gives the same English texts, since the S1 file ships no translations); `test_saves_png_svg_pdf` (`tmp_path`, each file non-empty).
  - `tests/test_package.py`: `test_matplotlib_not_imported` (subprocess: `import soundscapy, sys; assert "matplotlib" not in sys.modules`); `test_import_without_polars_or_pyarrow` (subprocess: `sys.modules['polars'] = None; sys.modules['pyarrow'] = None; import soundscapy` exits 0); the existing silent-import test unchanged; `test_all_names_resolve` (every name in `__all__` is an attribute).
  - `tests/test_docs_structure.py`: `test_quickstart_is_the_first_plot` (the file contains `load_isd(subset=True)`, `angles="equal"`, `by="location"` and `savefig("first-plot.png")`) and `test_quickstart_runs` (subprocess `[sys.executable, str(GUIDE / "_quickstart.py")]`, using the module's absolute `GUIDE`, with `cwd=tmp_path`, `env={**os.environ, "MPLBACKEND": "Agg"}`, `capture_output=True, text=True`: exit 0, empty stdout, `(tmp_path / "first-plot.png").exists()`; stderr is not asserted, because matplotlib's first-run font-cache message may appear there in a fresh environment).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement**, replace `_quickstart.py`, and edit `01-install.qmd`: "It prints the installed soundscapy version" becomes "It saves your first plot, a scatter of the bundled ISD subset by location, as `first-plot.png`". The docs test (`test_install_page_pins_the_release_and_includes_the_quickstart_files`) needs one `{{< include _quickstart.py >}}` line, so the Step 1 include ("save the first-plot code") stays unchanged. Plan call: the Hosted-notebook include (the page's last line) is replaced by the notebook form of §6.21 in a plain ```` ```python ```` block ending `sspy.plot.scatter(coords, by="location").figure`, and the same block goes after the `uv run jupyter lab` block under "paste the same code into a notebook cell".
- [ ] **Step 4: Pass**; `uv run task check` green. Then rerun the Task 3 Step 5 floor commands on the whole suite (`uv run --python 3.11 pytest -p no:cacheprovider` under the same two variables, then `git restore uv.lock`): this is the first task that imports matplotlib, so it proves the 3.10.7 floor; `uv tree --depth 1` shows matplotlib 3.10.7.
- [ ] **Step 5: Commit** `feat: plot.scatter, the S1 top-level API and the first-plot quick-start`.

---

### Task 13: Done-when check, ADRs, Taskfile note, docs build

**Files:**
- Create: `design/adr/0001-coordinates-carry-settings.md`, `design/adr/0002-response-id.md`, `design/adr/0003-ssp-column-namespace.md`, `design/adr/0004-h1-hashing.md`, `tests/science/test_s1_done.py`
- Modify: `Taskfile.yml` (comments only), `docs/_freeze/` if `task docs` changes it

**Interfaces:**
- Consumes: everything above. Produces no code API.

- [ ] **Step 1: Write the done-when test** `tests/science/test_s1_done.py::test_thin_slice` (`science` marker): `coords = sspy.project(sspy.load_isd(subset=True), angles="equal")`; `ax = sspy.plot.scatter(coords, by="location")`; `ax` is a matplotlib `Axes`; the vertical stack of every collection's offsets, in sorted location order, equals the stack of each `g.xy[valid_mask(g)]` for `g` in `coords.split("location").values()` (exact, `np.testing.assert_array_equal`: the plot draws the array it is given); the total equals `coords.n_valid()`. Run it: PASS (every piece exists). This is the S1 done-when command.

- [ ] **Step 2: Taskfile carry-over (M5).** S1 adds no `reference` and no `network` test (the first `network` test is S2's full-ISD counts; the first `reference` test is S3's aletta regeneration; the subset row checks are S9's). The `|| test $? -eq 5` clauses therefore stay. Edit only the two comments so they name the right slices: `test:reference` "Remove this tolerance in the commit that adds the first reference test (M1 slice S3)"; `test:network` "Remove it in the commit that adds the first network test (M1 slice S2)".

- [ ] **Step 3: Write the four ADRs**, each with Status (Accepted, 2026-10-31), Context, Decision, Consequences and the code that exercised it (plan §6: ADRs are written from code). One page each; cite the spec section, do not restate it.
  - `0001-coordinates-carry-settings.md`: no `ProjectionResult`; `Coordinates` holds its `settings` (typed `FrozenModel` so `distribution` does not import `projection`) and `provenance`; labels are numpy arrays, frames only at the return (§6.10, §11.1). Exercised by `project`, `split`, `attach`.
  - `0002-response-id.md`: column, then key, then physical row number; never memory order; recorded as `response_id_source` with the columns it was built from (`response_id_columns`, `response_key_blank_ok`, §6.3.6 as amended); `attach` aligns by the rule and columns recorded (§6.3.2, §6.16). Exercised by `read_survey` and `attach`; the ISD four-part key.
  - `0003-ssp-column-namespace.md`: every derived column in a frame holding user columns starts `ssp_`; `read_survey` refuses `ssp_` input columns up front; `attach` never overwrites (§6.3.3). Names the constants in `_model`.
  - `0004-h1-hashing.md`: one helper, `h1-sha256:` prefix, canonical JSON with `by_alias=True`, `frame_hash` normalisation, the per-value length-prefixed text encoding with its null marker and the backend-independent value-to-text rule (§6.3.5 as amended), golden vectors in `tests/test_provenance.py` and the frame vector in `tests/test_frames.py`.

- [ ] **Step 4: Run the full gate, the floors and the docs build**

```bash
uv run task check
UV_RESOLUTION=lowest-direct UV_PROJECT_ENVIRONMENT="$SCRATCH/lowest-venv" uv run --python 3.11 pytest -p no:cacheprovider
git restore uv.lock
uv run task docs
git status --short docs/_freeze
```

Expected: `task check` green with coverage at 80% or more; the floor run passes the whole suite (what CI's `lowest` job runs); `task docs` renders offline and the API reference lists the S1 names; commit any `docs/_freeze/` change with this task. If great-docs flags a docstring, fix the docstring.

- [ ] **Step 5: Commit** `docs(adr): S1 decisions; test: S1 done-when` (the ADRs, the test, the Taskfile comments, any `_freeze` change).

- [ ] **Step 6: Finish the unit of work** per AGENTS.md: `/simplify` on the S1 change, `uv run task check`, then `/code-review` and its fixes.

---

## Plan calls

Each is marked "Plan call:" where it is made. A reviewer should check them.

1. `_Unset` is an enum in `_model` with `_UNSET` its member (Task 1).
2. `Citation`, the pair-table aliases (which reject repeated keys), `to_toml`/`from_toml` and the `ssp_` namespace constants live in `_model` (Task 1).
3. S1 adds only the runtime dependencies S1 imports, at the patch floors pandas 2.2.1 and matplotlib 3.10.7 that keep the `lowest` job warning-free; pyarrow stays S4's (Task 1; §6.19 amended).
4. The hashing helper's entry points are `hash_bytes`, `hash_file`, `hash_json`, `canonical_json`, `check_hash`; dumps use `by_alias=True`; `JSONValue` lives in `provenance` (Task 2).
5. The envelope omits libraries not yet installed, guards extra keys with `OPTIONAL_KEYS`, and S1 results write no `profile`/`overrides`; the version comes from `importlib.metadata` (Task 2).
6. `frame_hash` writes each text value as a `uint64` byte length plus its bytes, null as `0xFFFFFFFFFFFFFFFF`, and turns non-text values into text from their Python form, never `cast(String)` (Task 3; §6.3.5 amended).
7. `read_text_table` takes keywords instead of a `ReadOptions` object; short rows are padded, long rows and duplicate headers raise (Task 3; §6.3.8 amended).
8. `_frames.with_columns` takes a `valid` mask that writes nulls; `_frames` never imports polars or pyarrow (Task 3).
9. Every `_data` read verifies its pin; the pin-coverage test skips `.py`, `.pyc` and `__pycache__` (Task 4).
10. `Circumplex` carries required `schema`/`schema_version` (alias, `populate_by_name`) like the other data files; `positions` and `angles_deg` are properties; `Axis.labels` is empty in the shipped file (Task 4).
11. The scale labels use the 0.x capitalisation, and the Table A.1 fixture test moves to S3 (Task 4; §9 S3 amended); the Mitchell 2025 blog gets no bib entry until its URL is supplied (Task 4).
12. Σ|cos| or Σ|sin| below 1e-9 raises `FormulationError`; the S1 kernel's `Literal` has no `aletta2024` (Task 5; §6.3.10 amended).
13. `Status.all_included`, the fixed key set of `counts()`, and `label_arrays()` as the one text form (Task 6).
14. `SurveyMapping` maps attributes, ids and language only in S1; the `schema` field is `schema_name` with alias `schema`; the whole coding order ships in S1; a before-validator on `SurveyMapping` fills the default item scale from the instrument, normalises attribute keys and rejects label/position duplicates; the default scale is omitted when written (Task 7).
15. `read_survey` in S1 takes paths only, has `read_options` as its one keyword, resolves `delimiter="auto"` by suffix, builds ids before coding, always raises on a scale violation (missing cells excluded from the check), and applies only `incomplete_attributes`; the circumplex `data_refs` entry uses the file pin; `Survey.to_frame`/`select` wait for S2; `_read_survey` takes extra `data_refs` (Task 8).
16. The subset keeps source row and column order; the sidecar's non-citation keys; S1's registry holds one version; S1's `isd_mapping.toml` is minimal, under `_data/datasets/isd/`; `load(subset=False)` raises `NotImplementedError`; `load` takes its mapping and read options from the registry entry (Task 9).
17. `AngleSelection` ships with `default: Literal["equal"] = "equal"` in S1, a provisional default S3 replaces with the profile's (Task 10).
18. Provenance records `id_columns` (roles), `response_id_columns` and `response_key_blank_ok`, used by `split` and `attach`; `ssp_angle_set` is `""` for rows not projected; `valid_mask` is the one valid-row definition (Task 10; §6.3.6 amended).
19. `attach` raises `AlignmentError` when no frame row matches; the `Table` form waits for `Table` (Task 11).
20. `plot.scatter` cycles `petroff10` and changes marker on each pass of ten colours; an empty series is kept for a group without valid rows (Task 12).
21. The install page's hosted-notebook include becomes the notebook-form block, and the same block follows `uv run jupyter lab` (Task 12).
22. The M5 exit-5 clauses stay through S1; only their comments change (Task 13).

## Spec defects found while planning

Each standing defect is amended in the spec, marked "Amended at S1 plan review (2026-10-31)".

- §6.3.5 row 3: text "joined by U+001F" with null as U+0000 collides across columns, inside a column on values holding U+001F, and between null and `"\x00"`; and "any other value `str()`" does not say where the value comes from, so `cast(String)` gives backend-dependent text. Amended to the per-value length-prefixed encoding and the Python-value text rule (Plan call 6). The frame golden vector is recorded in §6.3.5 by Task 3 Step 4.
- §6.4.1 (and every model with a `schema` key): a pydantic field named `schema` warns "shadows an attribute in parent BaseModel", which `filterwarnings = ["error"]` makes fatal. §6.4.1 amended (field `schema_name`, `Field(alias="schema")`, `populate_by_name=True`); §6.2.1 and §6.3.5 row 2 amended to `model_dump(mode="json", by_alias=True)` (Plan calls 4, 10, 14).
- §6.3.8 gives `read_text_table(path, options)`, but `ReadOptions` lives in `survey`, above `_frames` in the layer contract. Amended to keywords (Plan call 7).
- §6.10 and §6.16: no field tells `split(by="location")` the user's column for a role, and nothing tells `attach` the `response` column or the ordered `response_key` columns and their blank-allowed parts (which need not be roles). §6.3.6 amended with a row for `id_columns`, `response_id_columns` and `response_key_blank_ok` (Plan call 18).
- §6.19: the floors pandas 2.2 and matplotlib 3.10 resolve to 2.2.0 and 3.10.0 in the `lowest` job, whose imports warn (pyarrow absent; pyparsing 3.3 deprecations) and fail under `filterwarnings = ["error"]`. Amended to 2.2.1 and 3.10.7, checked by install on Python 3.11 with numpy 1.26.0 (Plan call 3).
- §6.3.10 `FormulationError` row says canonical accepts any angles, but a degenerate set divides by (nearly) zero. Amended (Plan call 12).
- §11.4's last row (Table A.1 fixture and test) is in no §9 slice row. §9 S3 "Delivers" amended to carry `tests/science/fixtures/iso12913_3_table_a1.csv` and its test (Plan call 11).
- §7.1 row-order invariance says ids permute identically, which is false for `row-<n>` ids (they follow physical position, §6.3.2 rule 3). Amended to column and key sources.
- §6.23 calls the 2019 formula "the paper's Eq. 2"; in Mitchell, Aletta and Kang 2022 it is Eqs. 1 (ISOPleasant) and 2 (ISOEventful). Amended, and `design/literature/README.md` likewise.

Withdrawn: the draft's §6.8.1 entry (integer `angles_deg` against `StrictFloat`) was not a defect. pydantic 2.7.0 `StrictFloat` accepts integers and returns floats (`[0, 46]` validates to `(0.0, 46.0)`); it rejects bool and str. §6.8.1's example validates as written.

## Notes for later slices

- S2: the `exclude` scale-violation path writes `scale_violation` through Task 8 step 6's mask; extend `isd_mapping.toml` and re-pin it; add every new FrozenModel to `tests/test_toml_round_trip.py`.
- S3: integer `angles_deg` in angle-set TOML validate to floats under `StrictFloat`; ship the Table A.1 fixture and test (§9 S3 as amended); add the `aletta2024` `Literal` member; widen `AngleSelection.default`.
- S4: adding pyarrow 15 changes pandas 3's default string backend; rerun the frame and hash tests on both.
- S9: the registry entry's `mapping` and `read_options` drive `load` already; the new versions only add entries.
