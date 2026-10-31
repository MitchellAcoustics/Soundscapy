# 0003. The `ssp_` column namespace

- Status: Accepted, 2026-10-31
- Spec: §6.3.3

## Context

`attach` adds soundscapy's results to a frame that already holds the user's columns. A
derived column named like a user column would either overwrite the user's data or be
silently renamed, and the clash could surface long after the survey was read.

## Decision

Every derived column placed in a frame that also holds user columns starts with `ssp_`.
`soundscapy._model` owns the names, and no other module spells them:

- `SSP_PREFIX = "ssp_"`;
- `COL_RESPONSE_ID`, `COL_INCLUDED`, `COL_REASONS`, `COL_FLAGS`, `COL_LANGUAGE`,
  `COL_ANGLE_SET`;
- `axis_column(axis_id)`, giving `ssp_<axis id>` (`ssp_ISOPleasant`, `ssp_ISOEventful`),
  so the axis columns follow the instrument, not a fixed pair of names.

Two rules enforce it:

- `read_survey` raises `ColumnCollisionError` on any input column starting with
  `ssp_`, before anything else runs (`_check_columns`), so the clash cannot appear mid-analysis.
- `attach` never overwrites: `_frames.with_columns` raises `ColumnCollisionError` when an
  added name is already present, leaving the user's frame untouched.

Tables the package builds with no user columns (describe output, statistics) use plain
names.

## Consequences

- A user frame is safe to pass to `attach` twice only if the first result is dropped; the
  second call raises instead of overwriting.
- Adding a derived column means adding a constant to `_model`; the prefix check covers it
  without a further change.
- A user's own `ssp_` column is refused at read time and cannot be mapped around; renaming it
  in the source is the remedy, which the error message names.

## Exercised by

`read_survey` (`test_ssp_column_collision_raises`), `workspace.attach` and
`_frames.with_columns` (collision tests), and `Coordinates.labels`, whose keys are these
constants.
