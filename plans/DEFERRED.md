# Deferred — real work, deliberately not now

Things that are genuinely worth doing, are not blocking the submission, and
would otherwise be lost. Each says what it is, why it waits, and what it costs.

Nothing here is a nice-to-have wish: if an item stops being true, delete it.

---

## D1 — 61 mypy errors in the component tests (closed 2026-08-22)

**What.** `mypy --config-file pyproject.toml` reported 61 errors, all in
`tests/components/`, none in `src/`. Mostly `Registry.build` returning
`object`, plus `type: ignore` comments carrying the wrong error code
(`union-attr` on an `attr-defined` error). Five more came from
`tests/test_select_probe_set.py` importing v1 helpers that BP7 deleted.

**Status.** Closed on `phase-d/cleanups`. Tests now narrow `build()` to the
backend they constructed. The v1 probe-set file is a module-level skip;
v2 coverage is `tests/components/test_probe_set.py`. No new ignores.

---

## D2 — SAM3 cannot load

**What.** `ModuleNotFoundError: No module named 'torch.nn.attention'`. Torch
2.2.2 is too old; SAM3 needs a newer torch. Both `detection/sam3` and
`segmentation/sam3` fail construction, honestly and with a stated reason.

**Why deferred.** It blocks `PLAN.md` §7 **P1 item 10** (detector comparison
including SAM3) and nothing in P0. YOLO26 loads and runs, so the pipeline has a
working detector and segmenter.

**Why it still matters.** SAM3 is the open-vocabulary arm, and P1 item 16
(open-vocabulary versus hand-written selection) depends on it too. Losing both
costs the paper a comparison a reviewer may well ask for.

**Cost, and the trap.** *Do not upgrade torch in the `pointstream` env.* Several
forked models here are version-sensitive and a stray upgrade breaks them
silently — this is a standing host rule. The fix is a **second conda env** with
newer torch, and SAM3 invoked across the process boundary. That is a day of
work, not an afternoon, which is precisely why it waits.

---

## D3 — AVC `addroi` is a no-op under QP (verified 2026-08-22)

**What.** ffmpeg's `addroi` filter is on the native AVC command and is listed
in this build's filter table. AV1 and HEVC have native delta-QP maps that
have been driven.

**Finding.** `/opt/local/bin/ffmpeg` (`n7.1.1-56-gc2184b65d2`,
`--enable-libx264`) inserts `addroi=192:128:256:128:-0.588…` under
`--roi-arm native`. At matched QP 45 / preset veryfast on 20 frames of
`assets/real_tennis.mp4` (640×384, centred region, inside offset −30),
baseline and ROI bitstreams were byte-identical (7627 bytes) and luma PSNR
was unchanged in the labelled region (30.09 dB in, 28.18 dB out). Bound
written first: a no-op is both |Δ| < 0.25 dB in the region; measured Δ =
0.00 dB, inside that bound. File size alone would not have been evidence;
here even the bytes matched. A CRF 45 diagnostic on the same clip — not a
paper comparison; the contract forbids CRF for an ROI arm — moved the
labelled region +17.00 dB and left the outside at −0.00 dB, so libx264 can
honour the side data when AQ is active. Native AVC ROI is therefore
unusable under the QP discipline the other rungs use; keep the pixel arm.
The CRF diagnostic bound (+0.25 to +4 dB) was too tight: qoffset −0.59 at
CRF 45 is most of the addroi scale toward lossless, not a mild AQ nudge.

**Why it still matters.** `NOTE(sec:evaluation)` item (c) commits the paper
to giving every baseline region control *wherever its encoder supports it*.
Say that AVC's encoder supports `addroi` only under CRF, which this
comparison is not allowed to use, rather than leaving the arm silent or
pretending the QP flag works.

**Status.** Closed as a recorded finding. Do not put native AVC `addroi` in
the evaluation bin.

---

## D4 — SVD-licensed generators (MOFA-Video, StableAnimator)

**What.** Two backends refuse a shippable load because inference needs
Stability-AI-licensed SVD weights that are not bundled.

- **MOFA-Video** — registered; construction refuses by design.
- **StableAnimator** — wrapped on `phase-bp/bp4`. HF card Apache-2.0
  (`FrancisRing/StableAnimator`, checked 2026-08-22); GitHub code MIT;
  inference needs SVD-XT plus InsightFace `antelopev2`. Construction succeeds;
  loading the real stack raises until a runtime is injected.

**Why deferred.** For MOFA, `PLAN.md` §6.2 routes around it by rendering sparse
trajectories into the ControlNet backbone, which is a *better* experiment — it
keeps `eval-object`'s backbone genuinely fixed. For StableAnimator, it cannot
be the shipped quality flagship until SVD is cleared; Animate-Anyone stays the
evaluable incumbent. Do not `pip install` the missing stack into the
`pointstream` env.

**Revisit only if** a licence-cleared SVD-free flagship appears, or the
rendered-trajectory arm turns out to be a strawman.
