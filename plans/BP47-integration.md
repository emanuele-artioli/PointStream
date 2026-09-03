# BP47 — integrate BP44–BP46 before the low-rate search

Owner: Codex (cross-workstream integration). Routine E1 execution goes to Cursor
only after the integration checks pass. No broad encode batch has been launched.

## Preserved inputs

- BP44: `02418fe`, including canonical canvases and context resets.
- BP45: `7b263f3`, including the previously uncommitted repairs.
- BP46: `21c0681`, including the previously uncommitted repairs and reports.
- Reference protocol and roadmap: `a7e5a16`.

These reached `main` in PR #52 (`68a03dc`). Original worktrees remain;
ask the user before removing them. Remote branch deletion is human-only.

## Integration changes

- Checkpoint identity includes full decoded source hashes, context IDs, preset,
  configuration, experiment plan, bounds and probe files, and source-code digest.
  Changed or legacy identity refuses resume; use a new output directory.
  Reference comparisons also check input hashes, implementation and preset.
- Fallback loading still requires the exact requested duration. Ineligible
  intervals route to fallback even when a shorter interval of that scene qualifies.
- `src/runner/fallback.py` now delivers explicitly routed fallback scenes through
  the codec. The existing SOURCE_PASSTHROUGH corner remains raw, not a codec.
  Transport accounting includes a serialized one-byte route tag. The equivalence
  check compares codec payloads and reports routing overhead separately.
  This is not an automatic scene classifier or a complete mixed-scene scheduler.
- PointStream rejects wrong output shapes and reports submitted/succeeded/failed
  counts. Failed points make the sweep exit nonzero.
- Dataset-dependent loader tests are opt-in with `POINTSTREAM_DATA_TESTS=1`, not
  silently assumed available in CI. Synthetic regression tests run everywhere.
- The predictive-background regression compares to fresh independent encodes
  with the same codec, not PNG. A universal AV1-versus-PNG ranking was not a
  valid invariant and failed on this host.

## Validation

- Targeted metric, data-validation, checkpoint and background suite: 128 passed,
  one external-data test skipped (run separately with explicit opt-in).
- End-to-end synthetic smoke: 48-frame static and 48-frame pan scenes, both
  shared and changed contexts, complete delivered dimensions and byte ledger.
- Real-codec small-frame fallback control: passed. No codec ranking claimed.
- Default full suite: 1,047 passed, one external-data test skipped, 84 deselected,
  three expected failures. All test assertions passed. Coverage is about 80%:
  above the existing 77% CI requirement, below the separate 81% local buffer.
  The local coverage command therefore exited nonzero. Neither threshold was
  changed. A same-host pre-integration coverage baseline was not captured, so
  no coverage delta is claimed.
- Full mypy: no issues in 329 files. Lint and import direction: passed.
- External-data loader and reduced-resolution fallback delivery: passed with
  explicit opt-in. Uses 48 frames of the recorded ineligible crowd scene,
  downsampled to 320x180 only for this smoke. PR CI must pass before merge.

Coverage changes are reported by the full gate; no threshold is being raised or
lowered to accommodate these changes.

## Remaining gates / handoff

Native-resolution two-scene one-point PointStream preflight: **passed** on
`68a03dc`, report `plans/BP47-e1-preflight.md`. Source identity, delivered
shape, ledger, VMAF/PSNR/SSIM, and checkpoint resume are recorded. One 48-frame
4K point took 73 min of PointStream wall, so it does not fit the hourly
subprocess budget; per-point checkpoints still cannot resume a killed encoder.

Recovery repairs and their approved interruption tests are tracked in
`BP48-recovery-validation.md`. Native recovery-budget verification and the
slowest-preset AV1/VVC pilot come before curves or a broad staged sweep. Use new
output directories when implementation or source frames change. Gate A is open.

Antigravity data follow-up remains separate: confirmation corpus incomplete.
Keep historically used videos in diagnostics; audit independent match identities
and obtain enough eligible held-out footage for the eventually selected duration.
Do not require all four durations for a claim frozen to just one duration.
