# 0001. Coordinates carry their settings and provenance

- Status: Accepted, 2026-10-31
- Spec: phase 0 and M1 design §6.10, §11.1

## Context

The architecture plan sketched a `ProjectionResult` wrapping the coordinates, the settings
that made them and the provenance envelope. Every consumer (`split`, `select`, `plot`,
`attach`, the sidecar writer) would then take two objects or unwrap one. The layer
contract also puts `distribution` below `projection`, so `Coordinates` cannot name
`ProjectionSettings` without importing upward.

## Decision

There is no `ProjectionResult`. `soundscapy.distribution.coordinates.Coordinates` is a
frozen dataclass that holds `xy`, `circumplex`, `labels`, `origin`, `settings` and
`provenance` itself. Three choices follow.

- `settings` is typed as the `FrozenModel` base class from `_model`, so `distribution`
  imports nothing from `projection`. `project` puts a `ProjectionSettings` there; code
  that needs the concrete type validates it back (the sidecar reader will, S4).
- `labels` is a dict of read-only numpy arrays (bool, float64 or object arrays of `str`),
  one per per-response column, in column order. No frame is held. `to_frame` builds a
  pandas frame, or one of the caller's library via `like`, only at the point of return,
  through `_frames.new_frame`, which keeps pandas and polars out of the other modules.
- `provenance` is a plain dict from `provenance.envelope`. It carries what later calls need
  to act on the result without the survey: `id_columns` (role to the user's column, which
  `split(by="location")` uses), `response_id_source`, `response_id_columns` and
  `response_key_blank_ok` (see ADR 0002).

The arrays are copied and set read-only in `__post_init__`; the origin segments' row counts
must sum to the row count, so a result cannot disagree with itself.

## Consequences

- One object flows from `project` to `split`, `select`, `plot.scatter` and `attach`; each
  reads only what it needs from it.
- `Coordinates` is not a pydantic model (it holds arrays), so its equality is identity
  (`eq=False`); tests compare the arrays.
- A result built by hand needs a settings model and an envelope; `Coordinates` checks
  shapes and origin counts, not the provenance keys, so the readers of those keys
  (`split`, `attach`) raise their own specific errors when a key is absent or malformed.
- A later slice that widens `OriginSegment.kind` (pooling, M2) changes no stored shape.

## Exercised by

`projection.project.project` (builds it), `Coordinates.split` and `Coordinates.select`
(rebuild it per group, sharing settings and provenance), `workspace.attach` (reads
`labels` and `provenance`), and `tests/science/test_s1_done.py`.
