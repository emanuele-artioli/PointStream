# Experiments: tennis confirmation source freeze

### 2026-11-15 — three admitted source windows frozen before outcomes

Registered 2026-11-15 before execution, within the user's 12-hour authority ending 11:07:34 UTC. The primary agent explicitly authorized this **900 CPU-second, zero-GPU** publication stage, separately from the earlier eligibility audit (104.424 measured CPU seconds). The [stage allocation](submission-ladder-tennis-confirmation-source-publication-stage-20261115.json) records the allocation before execution. No encoding, resize, model or quality run is authorized here.

The primary agent independently reviewed the source/label/event sheets and admitted three physical source groups relative to recorded PointStream development exposure, with permitted RVD encoder-training overlap disclosed. AI source joins are not human identity certification. The [eligibility record](submission-ladder-tennis-confirmation-options-20261115.md) preserves source/judge lineage and limits.

The [immutable window manifest](submission-ladder-tennis-confirmation-source-freeze-20261115.json), SHA256 `df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1`, froze these windows **before outcomes**:

| Physical group | Native window | Duration | Visible / not-visible / unknown |
|---|---|---:|---:|
| Dimitrov–Shevchenko, Rotterdam 2024 | `[0,64)` at 25 fps | 2.56 s | 9 / 3 / 52 |
| Evans–Tiafoe, Washington 2023 | `[0,64)` at 25 fps | 2.56 s | 12 / 0 / 52 |
| Shelton–Mensik, Indian Wells 2024 | `[0,154)` at 60 fps | 154/60 s | 11 / 2 / 141 |

All 282 native frames remain required: 32 official visible times, five officially not-visible times and 245 unknown times. CSV `Frame` is a verified zero-based native index. Normalize the Rotterdam stream's initial PTS 0.04 s only; do not shift labels or resample time. Decode native RGB, then area-resize RGB to 1280×720; scale native points by 2/3 and use the fixed 64×64 crop in that view at official visible times only. Full-frame metrics retain every frame. Unknown is not absence. The method is frozen here; a future implementation/backend and exact code/transform hashes must be frozen before experiments. This publication performs neither decode nor resize.

The archive will contain the three complete original MP4 and CSV files, not only selected frames, with exact source-archive members and hashes authenticated to the original installed member sidecar. It also preserves source-only native PNGs, diagnostic crops, contact sheets, official source/loader text, source/judge metadata responses, options, allocation and publisher code. The original dataset bytes and prior scratch are preserved. Publication uses a new DataRoot directory, complete archive SHA256 readback, independent streaming checks of every member and an independent recomputation of all frozen window rows/counts from the archived CSVs. Final publication files will be 0444 and their parent 0555.

Planned directory: `/home/itec/emanuele/Datasets/pointstream-data/tennis/confirmation-source-20261115T062148Z-297a05e2`. New scratch: `/local/users/emanuele/pointstream/checks/tennis-source-freeze-20261115T062148Z-297a05e2`. The final receipt is written separately after successful verification; failures retain scratch and cannot silently replace the frozen manifest or source selection.

### 2026-11-15 — completed immutable publication and independent readback

The source-only publication completed successfully. The final archive is `/home/itec/emanuele/Datasets/pointstream-data/tennis/confirmation-source-20261115T062148Z-297a05e2/tennis-confirmation-sources.tar`, **160,204,800 bytes**, SHA256 `e5edc12c857658a119bf6d6cbe029ee64c09d6a860f1b975fe0708952895293d`. The source-window manifest remains byte-identical to the pre-execution freeze SHA above, both inside the TAR as `manifest.json` and beside it. The inventory SHA is `f36d8219a5a2a440ce7a7a47ca5f8dc2279b0dc97357b9f4671ff406f4c707f3`.

A separate fresh read-only process verified the complete archive SHA, every one of its 165 member hashes/sizes, readonly modes and the exact CSV window rows/counts. Archive members `sources/{match209_000,match269_000,match235_000}/original.mp4` and `official_ball.csv` preserve the full original inputs; consumers select only the irrevocably frozen windows. All publication files are 0444, parent 0555; all prior audit and publication scratch remains intact. The [completion index](submission-ladder-tennis-confirmation-source-publication-20261115.json) preserves the publisher and independent-validator receipts, input/code hashes, resource identities and exact paths.

Measured stage CPU, including preflight, publisher and independent validation, was **4.363 CPU seconds / 900 allocated**, GPU zero. Publisher wall time was 3.664 s and independent-validator wall time 1.443 s; transfers and local metadata work are excluded from those measured command times. No decode, resize, model, codec, quality scoring or new outcome was performed. The archived provenance Markdown keeps its original bytes; the local Markdown heading was subsequently corrected to the repository's required Experiments/dated-entry format without changing evidence. The source-only freeze is 06:21:48 UTC and the primary agent's later method card is 06:24:52 UTC, both before outcomes.
