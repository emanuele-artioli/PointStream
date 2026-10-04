# Demo source identity — 4 October 2026 UTC

The retained reports do not bind the published clip labels to one source set.
The comparison and RTM encoder reports name
`clip_01_factory001_worker001_00001`,
`clip_02_factory001_worker001_00002`, and
`clip_03_factory001_worker001_00000`.
`demo_streams.json` instead names `clip_01_factory035_close_hands.mp4`,
`clip_02_factory001_starve_win.mp4`, and `clip_03_factory001_high_det.mp4`.
Neither naming family proves which source was intended. No source-file hashes
join these reports to the served media.

The three retained reference MP4s are each300 frames,1920×1080,30fps,10seconds.
Their file hashes and native probe receipts are in the
[read-only audit](demo-source-identity-audit.json). Matching shape/duration does
not identify RGB content. The existing explicit media_organizer FFprobe7.1.1
was used; Homebrew's binary cannot load its x265 library. No tool/environment
installation or encoded-media replacement was performed.

## Future export contract

`demo.experiments.export_demo_clips` now requires a reviewed JSON list with
exactly one row for each `clip_01`, `clip_02`, `clip_03`. Each row has an explicit
`clip_id`, `path`, and the actual lowercase64-digit source-file `sha256`.
Relative paths resolve against the manifest directory. Distinct IDs require
distinct files; list order does not assign clip identities. Legacy positional
manifests are rejected before model loading. Select the intended files first,
then hash them with `demo.experiments.clip_identity.file_sha256`; do not copy
hashes from unrelated old reports.

Pass `--source-manifest /absolute/Datasets/.../manifest.json` and the existing
`--curated-dir` option. Submit GPU execution through the gated fleet with the
same entry point and reviewed scale arguments. The specification must also pin
checkpoint, mask/background inputs, native tools, budget, deadline and a
substantive smoke validator. This manifest only establishes source-file
identity; it does not establish exposure, frame/PTS alignment, hand labels,
packet-conditioned rendering, score validity or matched baseline quality.

The exporter checks source hashes before and after each clip and persists the
source and manifest identities in its report. Hash subprocesses have a60-second
read allowance. Output defaults are in `PS_STAGE_DIR` for fleet stages, otherwise
a unique external data run. Explicit work, pitch and report paths must be new
and outside the source tree. Work/pitch directories are created exclusively;
the report is reserved before model work. A preparing report after a failed job
is incomplete evidence. Historical files and models stay read-only; default
checkpoint lookup uses the model resolver's canonical/legacy policy.

## Decision still required

Choose and review the authoritative clip set, exposure and exact frame intervals
before rebuilding media or rescoring. Keep both historical naming families
recoverable. The local `cursor/rtm-sam-ladder-at-risk` branch remains unpublished:
its warning was integrated, but its metric/cache replacements and silent missing
SAM-mask fallback are not a validated correction. The separate diagnostic/media
snapshot remains archived and on its existing branch pending selective intake.
