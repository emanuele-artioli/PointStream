# BP53 measurement provenance

This records how the native three-point batch crossed an implementation
identity change. Native outputs were not rewritten except for additive
sidecar files (`measurement-provenance.json`, reconstructed `budget.json`).

## What was encoded, and when

| Event | Commit | What changed |
|---|---|---|
| Encoder/config | `2149a00` | `transport_scale`, 28-byte geometry header, stream path |
| Log-dir guard | `2872a5f` | Refuse an unverified output directory; does not change pixels |
| Control encode | working tree at `2872a5f` | `bg-scale1-crf51` PointStream encode |
| Control predicate | `2ed5c40` | Treat residual `0` as present; driver only |
| Resume | after `2ed5c40` | Reused CRF51 checkpoint; ran half-scale points |

`git diff 2872a5f 2ed5c40 -- src` is empty. The digest change is
`experiments/tier/bp53_background_scale.py`.

## Digests

- Encode-time blob digest at `2872a5f` (`git ls-tree` of `src`,
  `experiments`, `config`, `pyproject.toml`; no working-tree extras):
  `33469a63c42d0fb0c23b71079467c11f85b89fff89d925e15ce48f77285bb00e`
- Resume / aggregate label (current `points/identity.json` and
  `background-scale.json` `input.implementation`):
  `9a7b2dbc8cfa2384e116296aca34ba7f2debf7de416545c9212c36df77124d39`

The live `implementation_digest()` also hashes untracked files, so the
blob digest is not claimed to equal the overwritten first identity.json
value. The two labelled/computed digests differ; `src/` did not.

`points/identity.json` already carries a note that it was refreshed after
the predicate fix. That overwrite is the labelling error; this file is the
preservation of the original encode commit.

## Why pixels are not automatically invalid

The predicate fix only changes whether a measured residual of 0 fails the
BP52 control check. Stream coding, headers, and reconstruction were already
frozen in `src/` at `2149a00` / `2872a5f`. Reusing the checkpoint avoided a
second 4K encode. That is a provenance split, not a demonstrated pixel
change.

## What this does not claim

- Independent-client restore from a standalone 4K package (added to code
  after the native run).
- Hourly clearance for longer runs (3599.72 s gap).
- A cumulative 8 h budget that included controls and survived restart
  during the native batch.
