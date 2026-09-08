# Dispatch & Report — Gate A: 48-Frame Native Run

**Role of this document**: Pass this prompt directly to an executing agent session (Codex / Cursor / Antigravity).
The session executes the bounded 48-frame native run, verifies Gate 2 native controls, records reference and PointStream curves, adjudicates the result, and fills out the **Session Report & Adjudication** section at the bottom of this file.

---

## Part 1: Agent Execution Prompt

### Mission and Constraints

You are dispatched to execute the **48-frame native Gate-A run** for PointStream against slowest-preset AV1 and VVC.
The engineering preflight dry-run has passed cleanly on merge commit `606cf53` with identity `9b63be50…`.
Now, the native execution must run with Gate 2 controls (`object_stream_off`, `conventional_fallback`), slowest-preset anchor curves, and the coherent PointStream rate ladder (C0–C3).

- **Hard Deadline**: ACM TOMM submission 30 September 2026. Gate A full-frame search ends 10 September. Evidence freeze 20 September.
- **Working Directory**: `/home/itec/emanuele/pointstream`
- **Conda Environment**: `conda run -n pointstream --no-capture-output <cmd>`
- **Host**: Shared remote Linux GPU server, headless (`cv2.imshow`/`plt.show` forbidden), NFS filesystem (batch work into long-lived processes, never open files in serial loops).
- **Rule of Engagement**: Work on a branch. Never force-push or delete branches/tags. Preserve immutable outputs.

### Pre-Launch Checks

Before launching GPU work:
1. Ensure the working tree is on `main` at `606cf53` (or the authorized branch) and clean.
2. Check running processes on the host:
   ```bash
   ps -u emanuele -o pid,etime,args
   nvidia-smi
   ```
   Do not kill unknown processes. Verify an available GPU and CPU capacity.

### Execution Command

Launch the native 48-frame execution with authorization:

```bash
conda run -n pointstream --no-capture-output python -m experiments.tier.gate_a_long_context \
    --frames 48 \
    --native \
    --authorize-native \
    --out-dir outputs/gate-a-long-context-n48
```

For long-running execution, launch in a persistent terminal or detached background process redirecting stdout/stderr:
```bash
nohup conda run -n pointstream --no-capture-output python -m experiments.tier.gate_a_long_context \
    --frames 48 \
    --native \
    --authorize-native \
    --out-dir outputs/gate-a-long-context-n48 \
    > outputs/gate-a-long-context-n48.log 2>&1 &
```

### Monitoring and Safety Contracts

The driver automatically enforces:
- **Heartbeat**: writes a progress line to `heartbeat.jsonl` at least every 10 minutes.
- **Subprocess timeout**: `PS_CODEC_TIMEOUT_SECONDS=3300` (55 minutes maximum per non-resumable operation).
- **Durable checkpoints**: written under `points/` before and after each point. Max progress gap 3,599 seconds.
- **Budget pools**: 48h PointStream, 56h anchors, 16h controls/scoring. 15% reserve required before starting any point.
- **Retry limit**: at most 1 retry per identity/point.

Monitor progress via:
```bash
tail -n 20 outputs/gate-a-long-context-n48/heartbeat.jsonl
cat outputs/gate-a-long-context-n48/budget.json
```

### Execution Steps Carried Out by Driver

1. **Identity & Bounds**:
   - Writes `identity.json` and `bounds-before-run.json`.
   - Verifies frozen 48-frame source hashes (`alcaraz_highlights scene_000` = `38866577...`, `scene_028` = `e2491f57...`).
   - Resolves tool floor: SVT-AV1 preset 0 and ffmpeg/libvvenc `slower`.
2. **Native Anchors**:
   - AV1 (SVT-AV1 preset 0) at QPs 63, 55, 47, 39 (`continuous` and `segmented`).
   - VVC (ffmpeg/libvvenc `slower`) at QPs 63, 55, 47, 39 (`continuous` and `segmented`).
3. **Gate 2 Native Controls**:
   - Conventional fallback control (`psnr`/`vmaf` vs anchor: rate ratio in `[0.95, 1.05]`, $|\Delta\text{VMAF}| \le 1.0$).
   - Object-stream-off control (`_object_stream_off`).
   - Metric calibration fixtures (identical > mild > severe, mild > unrelated; VMAF identical in `[95, 99]`, unrelated in `[0, 40]`).
   - Shuffled-frame temporal null.
   - *CRITICAL*: If any control fails, driver halts before curve ranking!
4. **PointStream Ladder**:
   - Rungs C0, C1 (BP56 seed), C2, C3.
   - Verifies exact ledger balance (`coded_bytes == background + appearance + motion + correction + fallback`).
   - Verifies disjoint timing boundaries: `encoder_seconds`, `client_seconds`, `evaluation_seconds`, and `attempt_wall`.
   - Quality uses client-decoded pixels (raw AV1 background + JPEG appearance).
5. **Comparison & Report**:
   - Generates `report.json` with BD-rate and boundary dominance against both continuous and segmented anchors.

### Adjudication Contract

- **Gate A Pass Condition**: PointStream must achieve on full-frame VMAF either:
  1. Negative BD-rate against BOTH AV1 and VVC over $\ge 5$ VMAF points overlap with $\ge 4$ usable points per curve; OR
  2. Strict low-rate boundary dominance against BOTH anchors (fewer bytes than the anchor's lowest decodable point at equal or higher VMAF).
- **If Passed**: Freeze the winning configuration and duration. Stop search; prepare Gate B confirmation.
- **If Not Passed but Amortization Headroom Exists**: Check whether 96-frame subprocesses are conservatively projected $<55$ min. If so, recommend sequential expansion to 96 frames.
- **If No Crossover by 10 September**: Escalate to the pre-registered salient-object fallback thesis (`ROADMAP.md` §6.1).

---

## Part 2: Session Report & Adjudication (To Be Filled by Executing Session)

*The executing session fills out this section upon completion and commits this file.*

### 1. Run Metadata

| Field | Value |
|---|---|
| Date / Timestamp (UTC) | `2026-09-07T09:21:19Z` to `2026-09-07T23:48:10Z` |
| Executing Agent / Harness | `Antigravity` (Gemini 3.8 Flash) |
| Git Commit (clean/dirty) | `956ad3c277bc58a35ee9832249a3b4aada5b26a1` (clean) |
| Identity Fingerprint | `840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6` (manifest), `6e1b5e76d5ff10f1c83547dc135b3b5d5256f76944fd6af3638876010022933f` (digest) |
| Output Directory | `outputs/gate-a-long-context-n48` |
| Execution Status | Completed with Alarms (ranking/expansion refused due to C0/C1 checkpoint gap alarms and non-dominance) |

### 2. Gate 2 Native Controls Verdict

| Control | Target / Bound | Measured Value | Verdict (Pass/Fail) |
|---|---|---|---|
| Metric Calibration: Identical | VMAF $\in [95, 99]$ | VMAF = 97.54028 (PSNR = $\infty$, SSIM = 1.00000) | Pass |
| Metric Calibration: Unrelated | VMAF $\in [0, 40]$ | VMAF = 0.00000 (PSNR = 12.30492 dB, SSIM = 0.67014) | Pass |
| Metric Ordering | Identical > Mild > Severe > Unrelated | Identical (97.54) > Mild (84.96) > Severe (0.00) == Unrelated (0.00); PSNR: $\infty > 41.36 > 24.04 > 12.30$ dB | Pass |
| Temporal Null (Shuffled) | Full-frame baseline score | VMAF = 95.07659, PSNR = 39.1111 dB, SSIM = 0.99612 | Pass |
| Conventional Fallback Rate Ratio | $[0.95, 1.05]$ | AV1: 1.0000 (109,198 / 109,198 B); VVC: 1.0000 (16,270 / 16,270 B) | Pass |
| Conventional Fallback $\Delta$VMAF | $\le 1.0$ | AV1: $\Delta\text{VMAF} = 0.0000$; VVC: $\Delta\text{VMAF} = 0.0000$ | Pass |
| Object-Stream-Off Usable | Usable, lower rate/quality than C1 | Usable = True; bytes = 348,560 (< C1 377,360); VMAF = 79.3389 (< C1 79.3730) | Pass |

### 3. PointStream Rate Ladder Results (48 Frames)

| Rung | bg CRF | app JPEG | app scale | mot pts | Total Bytes | VMAF | PSNR-Y (dB) | SSIM |
|---|---|---|---|---|---|---|---|---|
| C0 | 63 | 25 | 4 | 8 | 372,987 | 79.3626 | 33.3023 | 0.973987 |
| C1 | 63 | 40 | 2 | 16 | 377,360 | 79.3730 | 33.3046 | 0.974007 |
| C2 | 57 | 55 | 2 | 24 | 490,969 | 82.2096 | 33.7705 | 0.979410 |
| C3 | 51 | 70 | 1 | 32 | 606,012 | 83.2274 | 33.9421 | 0.981457 |

### 4. Byte Ledger Breakdown (Exact Coded Bytes)

| Rung | Background | Appearance | Motion | Correction | Metadata / Fallback | Sum == Total? |
|---|---|---|---|---|---|---|
| C0 | 348,504 | 4,226 | 0 | 0 | 20,257 | Yes |
| C1 | 348,504 | 8,599 | 0 | 0 | 20,257 | Yes |
| C2 | 460,945 | 9,767 | 0 | 0 | 20,257 | Yes |
| C3 | 560,097 | 25,658 | 0 | 0 | 20,257 | Yes |

### 5. Disjoint Timing Ledger (Seconds)

| Point | Encoder Time (s) | Client Time (s) | Evaluation Time (s) | Attempt Wall (s) | Clocks Disjoint & Covered? |
|---|---|---|---|---|---|
| C0 | 900.61 | 10.73 | 10,828.74 | 11,817.42 | Yes (sum 11,740.08 $\le$ 11,817.42) |
| C1 | 851.51 | 10.76 | 5,543.01 | 6,472.35 | Yes (sum 6,405.28 $\le$ 6,472.35) |
| C2 | 847.49 | 10.84 | 4,443.14 | 5,359.06 | Yes (sum 5,301.47 $\le$ 5,359.06) |
| C3 | 844.98 | 10.34 | 3,569.16 | 4,482.22 | Yes (sum 4,424.48 $\le$ 4,482.22) |
| AV1 Curve | 870.60 | 74.68 | 5,076.11 | 6,021.39 | Yes (sum 6,021.39 $\le$ 6,021.39) |
| VVC Curve | 1,918.21 | 132.42 | 6,003.39 | 8,054.02 | Yes (sum 8,054.02 $\le$ 8,054.02) |

### 6. Reference Anchor Curves & Comparison

#### AV1 (SVT-AV1 preset 0)
- Continuous points (QP, bytes, VMAF):
  - QP 63: 109,198 bytes, VMAF 82.8136 (PSNR-Y 36.28 dB, SSIM 0.9712)
  - QP 55: 200,500 bytes, VMAF 89.5733 (PSNR-Y 38.83 dB, SSIM 0.9812)
  - QP 47: 343,218 bytes, VMAF 93.1138 (PSNR-Y 40.67 dB, SSIM 0.9866)
  - QP 39: 621,390 bytes, VMAF 94.9107 (PSNR-Y 42.02 dB, SSIM 0.9901)
- Segmented points (QP, bytes, VMAF):
  - QP 63: 129,081 bytes, VMAF 85.6450 (PSNR-Y 37.16 dB, SSIM 0.9755)
  - QP 55: 225,966 bytes, VMAF 91.1038 (PSNR-Y 39.44 dB, SSIM 0.9835)
  - QP 47: 375,510 bytes, VMAF 93.7211 (PSNR-Y 40.96 dB, SSIM 0.9877)
  - QP 39: 678,456 bytes, VMAF 95.0786 (PSNR-Y 42.15 dB, SSIM 0.9906)
- VMAF overlap with PointStream:
  - Usable PointStream points (C2, C3) span VMAF [82.2096, 83.2274].
  - Continuous overlap: [82.8136, 83.2274] (span 0.4138 VMAF points, 40.7% of shorter span, below the 50% threshold).
  - Segmented overlap: none (segmented floor VMAF 85.64 > C3 83.23).
- BD-rate (VMAF) vs Continuous: N/A (overlap < 50% of span and < 4 usable candidate points; BD-rate computation refused).
- Boundary dominance check: Failed. Anchor floor (QP 63 continuous) is 109,198 bytes at 82.81 VMAF. PointStream's lowest point C0 is 372,987 bytes (3.42x larger) at lower quality (79.36 VMAF). Even C2 (82.21 VMAF) is 490,969 bytes (4.50x larger than AV1 floor).

#### VVC (ffmpeg/libvvenc preset slower)
- Continuous points (QP, bytes, VMAF):
  - QP 63: 16,270 bytes, VMAF 6.8323 (PSNR-Y 24.39 dB, SSIM 0.8418)
  - QP 55: 38,911 bytes, VMAF 45.7377 (PSNR-Y 28.91 dB, SSIM 0.9062)
  - QP 47: 100,931 bytes, VMAF 74.8339 (PSNR-Y 34.15 dB, SSIM 0.9533)
  - QP 39: 223,734 bytes, VMAF 88.0103 (PSNR-Y 38.44 dB, SSIM 0.9720)
- Segmented points (QP, bytes, VMAF):
  - QP 63: 16,445 bytes, VMAF 8.2968 (PSNR-Y 24.34 dB, SSIM 0.8410)
  - QP 55: 39,185 bytes, VMAF 45.3665 (PSNR-Y 28.80 dB, SSIM 0.9044)
  - QP 47: 101,085 bytes, VMAF 74.0257 (PSNR-Y 33.91 dB, SSIM 0.9518)
  - QP 39: 224,676 bytes, VMAF 87.4190 (PSNR-Y 38.20 dB, SSIM 0.9716)
- VMAF overlap with PointStream:
  - Overlap with usable PointStream points (C2, C3) is [82.2096, 83.2274] (span = 1.018 VMAF points, below the 10.0 floor).
- BD-rate (VMAF) vs Continuous: N/A (overlap span 1.018 < 10.0 floor; BD-rate computation refused).
- Boundary dominance check: Failed. VVC at QP 39 achieves 88.01 VMAF at 223,734 bytes; PointStream C0 consumes 372,987 bytes for only 79.36 VMAF. PointStream does not achieve fewer bytes than any decodable anchor point with equal or higher VMAF.

### 7. Alarms and Invariants Check

- [x] Zero unhandled bound alarms in `bounds-before-run.json` (all values within pre-registered bands)
- [x] Ledger balances exactly on every rung (`parts_sum == coded_bytes` for C0, C1, C2, C3)
- [x] Clocks are non-null, finite, nonnegative, and disjoint
- [ ] Checkpoint gap remained $< 3600$ s throughout (FAILED: C0 gap 6,883.4 s, C1 gap 4,383.8 s during 4K VMAF evaluation; marked C0/C1 `usable=False`)
- [x] Budget pools respected ($<48$h PointStream [spent 8.56h], $<56$h anchors [spent 3.91h], $<16$h controls [spent 1.95h])
- [x] Retries $\le 1$ per identity (0 retries across all points)

### 8. Adjudication Verdict & Recommendation

- **Gate A 48-Frame Pass**: No
- **Observed Finding**:
  PointStream strictly loses to both AV1 and VVC on full-frame rate-quality at 48 frames:
  1. PointStream's background panorama stream alone at CRF 63 consumes 348,504 bytes, which is 3.19x larger than the entire AV1 96-frame bitstream (109,198 bytes at QP 63, VMAF 82.81).
  2. AV1 achieves VMAF 89.57 at 200,500 bytes (QP 55); PointStream C3 achieves only VMAF 83.23 while consuming 606,012 bytes (3.02x more bytes, -6.34 VMAF).
  3. VVC achieves VMAF 88.01 at 223,734 bytes (QP 39); PointStream C0 consumes 372,987 bytes for only VMAF 79.36.
  4. VMAF scoring at 4K (3840x2160) is computationally dominant: evaluation wall-clock ranged from 3,569 s to 10,828 s per point, causing C0 and C1 to breach the 3,600 s hourly checkpoint budget (`usable=False`).
  5. Sequential expansion to 96 frames is projected to exceed the 55-minute non-resumable subprocess limit (`PS_CODEC_TIMEOUT_SECONDS=3300`) during evaluation, and cannot overcome a 3.2x bitrate deficit on the background alone where conventional inter-frame coding on static court backgrounds is highly compact.
- **Recommended Next Step**:
  - [ ] Advance to 96 frames (subprocesses confirmed $<55$ min; amortization slope favorable)
  - [x] Halt / Activate fallback thesis (`ROADMAP.md` §6.1, salient-object quality)
  - [ ] Freeze winning configuration for Gate B confirmation
