# 0002. How a response id is built and found again

- Status: Accepted, 2026-10-31
- Spec: §6.3.2, §6.3.6 (as amended at the S1 plan review), §6.16

## Context

Everything downstream, from reasons in `Status` to `attach`, identifies a response by its
`ssp_response_id`. The id must survive reordering of the user's file, must be unique, and
must let `attach` find the same response in the user's own frame later, which holds
different columns and may be in a different order. Memory order is no identity.

## Decision

`survey.read._response_ids` builds the id by the first rule that applies:

1. the mapped `response` column, stripped (`response_id_source = "column"`);
2. the mapped `response_key` columns, each stripped, joined with `/`
   (`"key"`); a part holding `/` is a `MappingError`, and a blank part raises unless its
   column is in `response_key_blank_ok`;
3. otherwise `row-<n>`, with n the physical record number in the raw file, the header
   being row 1 (`"row"`). Skipped blank rows leave their numbers unused, so ids do not
   shift.

A blank or repeated id raises `MissingResponseIdError` or `DuplicateResponseIdError`, naming
up to five offenders with row numbers. The ISD's key is the four parts
`LocationID/SessionID/GroupID/RecordID`; no single ISD column is unique.

The rule and its inputs are recorded in the provenance envelope: `response_id_source`,
`response_id_columns` (the columns the id was built from, in key order; `[]` for `row`) and
`response_key_blank_ok` (the key columns allowed to be blank). `workspace.attach` reads
those and nothing else of the mapping:

- `column` or `key`: rebuild each frame row's id from the recorded columns with the same
  text rule (`_frames.column_text`, stripped, joined with `/`) and match on the id. The
  frame's order is kept; a row with no match, or a blank part not in the blank-allowed
  list, gets nulls. Duplicate frame ids, a missing id column, or no match at all raise
  `AlignmentError`.
- `row`: align by position, and raise if the row counts differ.

## Consequences

- Reordering or filtering the user's frame cannot misalign results; only the `row` rule
  depends on position, and it checks the count.
- Text, not dtype, is the id's basis, so an integer id column in the user's frame matches
  the text the file reader saw, in pandas or polars.
- A change to the id rule changes ids, hence the `response_id_source` record: a result
  states how its ids were made, and `attach` refuses a result whose source it does not know.

## Exercised by

`read_survey` (all three rules, on small files and the ISD subset), `workspace.attach`
(id and position alignment, unmatched and blank-part rows), and the ISD four-part key in
`tests/survey` and `tests/workspace`.
