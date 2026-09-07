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
| Date / Timestamp (UTC) | *(e.g. 2026-09-07T...)* |
| Executing Agent / Harness | *(e.g. Codex / Cursor / Antigravity)* |
| Git Commit (clean/dirty) | *(hash, dirty status)* |
| Identity Fingerprint | *(SHA-256 fingerprint from identity.json)* |
| Output Directory | `outputs/gate-a-long-context-n48` |
| Execution Status | *(Completed / Halted on Alarm / Failed)* |

### 2. Gate 2 Native Controls Verdict

| Control | Target / Bound | Measured Value | Verdict (Pass/Fail) |
|---|---|---|---|
| Metric Calibration: Identical | VMAF $\in [95, 99]$ | | |
| Metric Calibration: Unrelated | VMAF $\in [0, 40]$ | | |
| Metric Ordering | Identical > Mild > Severe > Unrelated | | |
| Temporal Null (Shuffled) | Full-frame baseline score | | |
| Conventional Fallback Rate Ratio | $[0.95, 1.05]$ | | |
| Conventional Fallback $\Delta$VMAF | $\le 1.0$ | | |
| Object-Stream-Off Usable | Usable, lower rate/quality than C1 | | |

### 3. PointStream Rate Ladder Results (48 Frames)

| Rung | bg CRF | app JPEG | app scale | mot pts | Total Bytes | VMAF | PSNR-Y (dB) | SSIM |
|---|---|---|---|---|---|---|---|---|
| C0 | 63 | 25 | 4 | 8 | | | | |
| C1 | 63 | 40 | 2 | 16 | | | | |
| C2 | 57 | 55 | 2 | 24 | | | | |
| C3 | 51 | 70 | 1 | 32 | | | | |

### 4. Byte Ledger Breakdown (Exact Coded Bytes)

| Rung | Background | Appearance | Motion | Correction | Metadata / Fallback | Sum == Total? |
|---|---|---|---|---|---|---|
| C0 | | | | 0 | | *(Yes/No)* |
| C1 | | | | 0 | | *(Yes/No)* |
| C2 | | | | 0 | | *(Yes/No)* |
| C3 | | | | 0 | | *(Yes/No)* |

### 5. Disjoint Timing Ledger (Seconds)

| Point | Encoder Time (s) | Client Time (s) | Evaluation Time (s) | Attempt Wall (s) | Clocks Disjoint & Covered? |
|---|---|---|---|---|---|
| C0 | | | | | |
| C1 | | | | | |
| C2 | | | | | |
| C3 | | | | | |
| AV1 Curve | N/A | N/A | | | |
| VVC Curve | N/A | N/A | | | |

### 6. Reference Anchor Curves & Comparison

#### AV1 (SVT-AV1 preset 0)
- Continuous points (QP, bytes, VMAF):
- Segmented points (QP, bytes, VMAF):
- VMAF overlap with PointStream:
- BD-rate (VMAF) vs Continuous: *(% or N/A)*
- Boundary dominance check:

#### VVC (ffmpeg/libvvenc preset slower)
- Continuous points (QP, bytes, VMAF):
- Segmented points (QP, bytes, VMAF):
- VMAF overlap with PointStream:
- BD-rate (VMAF) vs Continuous: *(% or N/A)*
- Boundary dominance check:

### 7. Alarms and Invariants Check

- [ ] Zero unhandled bound alarms in `bounds-before-run.json`
- [ ] Ledger balances exactly on every rung
- [ ] Clocks are non-null, finite, nonnegative, and disjoint
- [ ] Checkpoint gap remained $< 3600$ s throughout
- [ ] Budget pools respected ($<48$h PointStream, $<56$h anchors, $<16$h controls)
- [ ] Retries $\le 1$ per identity

### 8. Adjudication Verdict & Recommendation

- **Gate A 48-Frame Pass**: *(Yes / No)*
- **Observed Finding**: *(Brief plain-language summary of size, quality, and runtime findings)*
- **Recommended Next Step**:
  - [ ] Advance to 96 frames (subprocesses confirmed $<55$ min; amortization slope favorable)
  - [ ] Halt / Activate fallback thesis (`ROADMAP.md` §6.1, salient-object quality)
  - [ ] Freeze winning configuration for Gate B confirmation
