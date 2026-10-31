# 0004. One hashing helper, scheme `h1`

- Status: Accepted, 2026-10-31
- Spec: §6.3.5 (as amended at the S1 plan review), §6.3.6, §6.2.1

## Context

Provenance records hashes of inputs, mappings, settings and shipped data, and a hash is only
useful if two runs on different machines, libraries and backends agree on equal content.
The draft's frame encoding (values joined by U+001F, null as U+0000) let different frames
collide, and "`str()` of any other value" left text to the backend.

## Decision

`soundscapy.provenance` is the only module importing `hashlib`; a test greps the package
for others. Every hash is `h1-sha256:<64 lowercase hex>` (`HASH_PREFIX`); `check_hash`
raises `HashSchemeError` for any other prefix. Entry points: `hash_bytes`, `hash_file`
(raw bytes, chunked), `hash_json` and `canonical_json`, and `check_hash`.

- Models and JSON values: the bytes of `json.dumps(value, sort_keys=True,
  separators=(",", ":"), ensure_ascii=False, allow_nan=False)`, with models first dumped by
  `model_dump(mode="json", by_alias=True)` so a field named by alias (`schema`) hashes under
  its file name. A non-finite float or a non-JSON value raises, naming its path.
- Frames (`_frames.frame_hash`): normalise, then hash. Numeric columns become float64 with
  null as NaN, every NaN written as `0x7FF8000000000000`; boolean columns without nulls
  stay `uint8`; every other column is text. The stream is the canonical JSON of
  `{"columns", "dtypes", "n_rows"}`, a 0x00 byte, then each column in order. Text is encoded
  per value as a little-endian `uint64` UTF-8 byte length then the bytes, a null as
  `0xFFFFFFFFFFFFFFFF` alone, so no two frames share a stream by value content.
- Text comes from each value's Python form (`_value_text`), never a backend `cast(String)`:
  `str` as is, datetimes, dates and times truncated to microseconds and written by
  `isoformat()`, anything else `str()`. pandas and polars therefore agree.

## Consequences

- An integer column and the same values as float hash alike, deliberately: pandas turns an
  integer column with a gap into float. A text column of digits stays distinct.
- Changing any rule above changes every stored hash, so it needs a new scheme prefix
  (`h2`), not an edit of `h1`.
- Pandas 3's default string backend (S4) must not change a hash; the frame tests run on
  both backends then.

## Exercised by

Golden vectors in `tests/test_provenance.py::test_golden_vectors` (empty bytes, `{}`, a
key-sorting and non-ASCII value, a settings model), the frame vector `FRAME_VECTOR` in
`tests/test_frames.py::test_frame_hash_vector` (one content, as pandas and polars), the
`hashlib` import test, and `envelope` in `read_survey` and `project`.
