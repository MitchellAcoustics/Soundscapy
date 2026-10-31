# Governance

soundscapy has one maintainer, Andrew Mitchell ([@MitchellAcoustics](https://github.com/MitchellAcoustics)).

## Releases

Andrew Mitchell cuts every release. Releases follow the milestones in section 9 of the
product requirements (`design/PRD.md`), with patch releases as needed. From the M2 release
on, releases are labelled as 1.0 alphas (`1.0a1`, `1.0a2`, ...) until the owner has
evaluated what was built.

## Changes that affect scientific results

Andrew Mitchell reviews and approves every change that affects scientific results. A
change affects scientific results if it does any of the following:

- moves a regression baseline;
- edits a file under `src/soundscapy/_data/` (shipped data, mappings, profiles, angle sets);
- changes a science fixture under `tests/science/`;
- bumps a method version.

A pull request that does any of these carries the label `science` and is not merged
without the owner's approval. `.github/CODEOWNERS` names `@MitchellAcoustics` on
`src/soundscapy/_data/`, `tests/science/`, `tests/regression/` and `design/`, so GitHub
requests the owner's review whenever those paths change.

Other changes follow the usual review described in `CONTRIBUTING.md`.
