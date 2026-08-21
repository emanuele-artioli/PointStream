# B1 — Codec and region control

**Owns exclusively:** `src/components/codec/**`, `tests/components/test_codec*.py`.
**Implements:** `src/contracts/codecs.py` — read it first; capabilities and
request validation are already there, and this stream builds the execution.

## What to build

A codec component that turns a validated `EncodeRequest` into an actual encode,
across four rungs:

| Rung | Driver | Invocation | Notes |
|---|---|---|---|
| AVC | ffmpeg | `libx264` | the speed rung; what makes a real-time target reachable |
| HEVC | **binary** | `kvazaar` | no ffmpeg encoder on this host |
| AV1 | **binary** | `SvtAv1EncApp` | region control only via the binary |
| VVC | ffmpeg | `libvvenc` | expected slow — that is useful, report encode time |

**Driving standalone binaries is a requirement, not an implementation detail.**
Both rungs with working region control are reachable only that way.

Also build:
- **Region-of-interest arms.** Native where the encoder has it (AV1 and HEVC
  expose per-block delta-QP maps); an **in-house pixel-domain arm** — degrade
  non-salient blocks before encoding — for the rest. VVC has no region map at
  all, so it needs the in-house path to be comparable. Prior art worth reading:
  `/home/itec/emanuele/presley/src/presley/components/roi.py` and
  `encode_utils.py`.
- **Encoder resolution that records path and version**, not just a name.

## Traps specific to this stream

**Resolve tools by path and version.** This host has carried two builds of
SVT-AV1 where only one had `--roi-map-file`. Testing the wrong one reads as
"region control does not work" for reasons unrelated to region control.

**Matched QP is not matched rate.** A region arm at the same QP as its baseline
simply spends more bytes, and more bits buying more quality is not a result. The
claim that matters is redistribution at equal bitrate, which needs the base QP
binary-searched to a matched byte count. `codecs.assert_matched_rate_control`
encodes the rule; call it before reporting any cross-arm number.

**AV1 region offsets are q_index units** (0–255) against `--qp` (0–63), roughly
four per QP step. Predicting an effect in QP units overestimates it fourfold.
Very large offsets are not simply stronger — past about −120/+60 the regions
*converge* and both lose quality.

**Verify `addroi` before believing it.** The existing `RoiVideoPanoramaEncoder`
drives ffmpeg's `addroi` filter with nothing checking the output differs from the
same encode without it. A prior project abandoned that path unfinished.

## Existing work to build on

`experiments/verify_codec_roi.py` measures whether a region surface actually
moves quality, and already establishes that AV1's is precisely localized at
matched QP. Extend it with `--match-bitrate` rather than writing a new harness.

## Done when

- Every rung encodes and decodes through one command builder, replacing the
  scattered ffmpeg construction sites.
- Every rung has a region arm, or a documented reason it cannot.
- `libsvtav1` + `yuv444p` raises rather than silently emitting yuv420p.
- Encoder path and version appear in the run record.
- Matched-bitrate region comparison runs and reports redistribution.
- `ruff`, `mypy`, and the stream's tests pass; import direction clean.
