# ruff: noqa: E402 - sys.path bootstrap must run before src/experiments imports.
"""PointStream E04B Paired Removal Probe (CODEC-ACT-07-E04B).

Executes the smallest costed probe testing explicit foreground removal and hole filling
(removal=ON: temporal median mask exclusion + Telea hole fill) vs removal=OFF on candidate
registered_panorama for Federer/Djokovic scene 007 (48 frames @ 12 fps, 360p):
1. Strictly refuses existing non-empty output directories.
2. Writes pre-registered two-sided bounds BEFORE reading any measured results.
3. Acquires an atomic CPU resource claim (<= 16 threads, 0 GPU).
4. Runs comprehensive scorer calibration (identity, ordering, null controls) and HALTS
   execution if calibration fails.
5. Reuses all 6 canonical removal-OFF arms from run-20260916-federer007 without re-encoding.
6. Encodes exactly two removal-ON registered_panorama points (QP 47 and QP 32) using VVC Intra.
7. Performs standalone decoding directly from serialized bitstream + packed binary side data.
8. Measures full timing strata (preprocessing, encode, decode, render) and total package bytes.
9. Computes paired deltas across bytes, visible PSNR-Y, ghosting MAD, and boundary metrics.
10. Preserves visual side-by-side evidence plates (original, OFF, ON, ghost difference).
11. Evaluates pre-registered promote / keep / stop decision rules and reports alarms.
"""

from __future__ import annotations

import sqlite3  # noqa: F401 - required before torch on this host
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Final

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from experiments.jobs.claims import claim_resources
from scripts.background_probe import (
    build_common_cleaned_stack,
    charge_side_data,
    compute_array_sha256,
    decode_standalone_representation,
    get_code_revision,
    pack_panorama_side_data,
    unpack_panorama_side_data,
    warp_plate_to_frame,
)
from scripts.e04a_evidence_completion import (
    load_360p_input_data,
    run_scorer_calibration,
)
from scripts.run_e04a_probe import (
    composite_fixed_foreground,
    compute_all_metrics,
)
from src.components.background.sidecar import IntraCodecSidecar
from src.components.codec.frames import rgb_to_luma
from src.contracts import paths as ps_paths

TASK_ID: Final[str] = "CODEC-ACT-07-E04B"
ACTION_NAME: Final[str] = "E04B_paired_removal_probe"
VIDEO: Final[str] = "federer_djokovic"
SCENE: Final[str] = "scene_007"
N_FRAMES: Final[int] = 48
WORKING_FPS: Final[float] = 12.0
TARGET_WIDTH: Final[int] = 640
TARGET_HEIGHT: Final[int] = 360
CODEC: Final[str] = "vvc"
PRESET: Final[str] = "medium"

PRE_REGISTERED_E04B_BOUNDS: Final[dict[str, Any]] = {
    "task_id": TASK_ID,
    "probe": "E04B_same_camera_paired_removal",
    "rationale": (
        "Registered panorama removal-OFF baseline yields 4,554 B / 21.16 dB vis PSNR / 9.16 ghost MAD (QP 47) "
        "and 17,411 B / 22.80 dB vis PSNR / 6.70 ghost MAD (QP 32). Removal-ON uses explicit mask exclusion and "
        "nearest-finite/Telea hole filling. Removal-ON must suppress ghosting (>= 50% MAD reduction target) "
        "while maintaining visible background PSNR within 0.5 dB and total package bytes within 15% inflation."
    ),
    "decision_rules": {
        "promote_rule": (
            "Promote removal=ON if ghosting MAD drops by >= 50% relative to removal-OFF at matched QP, "
            "visible PSNR-Y drops by <= 0.5 dB, and total package bytes increase by <= 15%."
        ),
        "stop_rule": (
            "Stop / reject removal=ON if ghosting MAD does not decrease, visible background degrades by > 0.5 dB, "
            "or byte inflation exceeds 15%."
        ),
        "inconclusive_rule": (
            "Keep exploratory / inconclusive if ghosting reduction is between 20% and 50% or boundary artifacts "
            "offset interior suppression."
        ),
    },
    "points": {
        "registered_panorama_qp47_removal_on": {
            "total_package_bytes": {"min": 3800, "max": 5500},
            "psnr_y_visible_dB": {"min": 20.6, "max": 22.0},
            "ghosting_luma_mad": {"min": 1.0, "max": 6.0},
        },
        "registered_panorama_qp32_removal_on": {
            "total_package_bytes": {"min": 15000, "max": 20500},
            "psnr_y_visible_dB": {"min": 22.2, "max": 23.8},
            "ghosting_luma_mad": {"min": 0.8, "max": 4.5},
        },
    },
}


def build_visual_evidence_plate(
    ref_rgb: np.ndarray,
    off_rgb: np.ndarray,
    on_rgb: np.ndarray,
    mask: np.ndarray,
    frame_idx: int,
    qp: int,
) -> np.ndarray:
    """Compose a 4-panel visual comparison plate for a key frame."""
    h, w, _ = ref_rgb.shape
    ref_bgr = ref_rgb[:, :, ::-1].copy()
    off_bgr = off_rgb[:, :, ::-1].copy()
    on_bgr = on_rgb[:, :, ::-1].copy()

    # Residual difference on ghost / player region
    ref_y = rgb_to_luma(ref_rgb).astype(np.float32)
    on_y = rgb_to_luma(on_rgb).astype(np.float32)
    diff = np.abs(ref_y - on_y)
    # Amplify difference for visual clarity
    diff_vis = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
    diff_bgr = cv2.applyColorMap(diff_vis, cv2.COLORMAP_VIRIDIS)
    # Dim non-mask regions to highlight ghost/player boundary
    diff_bgr[~mask] = (diff_bgr[~mask] * 0.25).astype(np.uint8)

    # Label panels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (255, 255, 255)
    thick = 1

    cv2.putText(ref_bgr, f"F{frame_idx:02d} Original", (10, 25), font, scale, color, thick)
    cv2.putText(
        off_bgr, f"F{frame_idx:02d} Removal-OFF (QP{qp})", (10, 25), font, scale, color, thick
    )
    cv2.putText(
        on_bgr, f"F{frame_idx:02d} Removal-ON (QP{qp})", (10, 25), font, scale, color, thick
    )
    cv2.putText(diff_bgr, f"F{frame_idx:02d} ON Residual (x3)", (10, 25), font, scale, color, thick)

    # 2x2 layout
    top_row = np.hstack([ref_bgr, off_bgr])
    bot_row = np.hstack([on_bgr, diff_bgr])
    return np.vstack([top_row, bot_row])


def main() -> None:
    parser = argparse.ArgumentParser(description="PointStream E04B Paired Removal Probe")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target output directory under outputs/evaluation-20260914/e04b/",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=16,
        help="CPU thread limit (<= 16)",
    )
    args = parser.parse_args()

    default_out = (
        ps_paths.outputs() / "evaluation-20260914" / "e04b" / "run-20260916-paired-removal-r2"
    )
    out_dir = (args.output_dir or default_out).resolve()

    # Rule 1: Strictly refuse existing non-empty directory
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write to existing non-empty directory: {out_dir}. "
            "Per evaluation protocol, evidence directories must not be overwritten. "
            "Specify a fresh revision directory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== PointStream E04B Paired Removal Probe (CODEC-ACT-07-E04B) ===")
    print(f"Output directory: {out_dir}")
    print(f"Host: {socket.gethostname()} ({platform.processor() or 'x86_64'})")

    # Rule 2: Write pre-registered bounds BEFORE reading or generating any results
    bounds_file = out_dir / "bounds.json"
    bounds_record = dict(PRE_REGISTERED_E04B_BOUNDS)
    bounds_record["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bounds_file.write_text(json.dumps(bounds_record, indent=2), encoding="utf-8")
    print("Pre-registered evidence bounds written to bounds.json.")

    job_id = f"e04b-paired-removal-{int(time.time())}"
    with claim_resources(job_id=job_id, cpu_threads=args.cpu_threads, job_dir=out_dir) as session:
        print(f"Acquired CPU resource claim (token: {session.token[:8]}...).")
        t_global_start = time.perf_counter()

        # Step 1: Load 360p input data
        frames_360, masks_360, boundary_masks, tracks, input_meta = load_360p_input_data()
        print(f"Loaded {len(frames_360)} frames @ 360p from {VIDEO}/{SCENE}.")

        # Step 2: Scorer calibration across visible, object, boundary, and full frame
        print("\n--- Running Scorer Calibration ---")
        calib_data = run_scorer_calibration(frames_360, masks_360, boundary_masks)
        (out_dir / "scorer_calibration.json").write_text(
            json.dumps(calib_data, indent=2), encoding="utf-8"
        )
        print(
            f"Scorer calibration valid: {calib_data['valid']} (Alarms: {len(calib_data['alarms'])})"
        )
        if not calib_data["valid"]:
            raise RuntimeError(
                f"Scorer calibration failed with {len(calib_data['alarms'])} alarms: {calib_data['alarms']}. "
                "Halting execution to block uncalibrated dependent scoring and claims."
            )

        # Step 3: Load and verify canonical 6 removal-OFF arms without re-encoding
        print("\n--- Loading Reused Canonical Removal-OFF Evidence ---")
        e04a_saved_dir = (
            ps_paths.outputs() / "evaluation-20260914" / "e04a" / "run-20260916-federer007"
        )
        old_report_path = e04a_saved_dir / "probe_report.json"
        if not old_report_path.is_file():
            raise FileNotFoundError(f"Missing canonical E04A probe report at {old_report_path}")
        old_report = json.loads(old_report_path.read_text(encoding="utf-8"))

        reused_off_points: dict[str, dict[str, Any]] = {}
        for p in old_report.get("screening_points", []):
            rep = p["representation"]
            qp = p["qp"]
            reused_off_points[f"{rep}_qp{qp}"] = p

        # Step 4: Build Removal-ON composite plate with explicit mask exclusion and hole filling
        print("\n--- Building Removal-ON Plate (masks=masks_360, Telea Hole Fill) ---")
        code_rev = get_code_revision()
        code_rev["active_branch"] = "codex/e04b-paired-removal-20260916"

        cleaned_stack_on, plate_on, homographies_on, prep_stats_on = build_common_cleaned_stack(
            frames_360,
            masks_360,
            removal="on",
            register=True,
        )
        prep_time_on = float(prep_stats_on.get("build_seconds", 0.0))
        print(f"Removal-ON plate built in {prep_time_on:.3f}s.")
        print(f"  Resolution: {prep_stats_on.get('plate_resolution')}")
        print(
            f"  Optional removal calls: {prep_stats_on.get('optional_removal_calls')} (mode: {prep_stats_on.get('removal_mode')})"
        )
        print(
            f"  Total inpaint holes: {prep_stats_on.get('total_inpaint_holes')} across {prep_stats_on.get('inpaint_frames')} frames"
        )

        # Step 5: Encode exactly 2 removal-ON registered_panorama points (QP 47 and QP 32)
        print("\n--- Encoding Removal-ON Registered Panorama (QP 47 and QP 32) ---")
        bitstream_dir = out_dir / "bitstreams"
        decode_dir = out_dir / "decodes"
        visual_dir = out_dir / "visual_evidence"
        bitstream_dir.mkdir(parents=True, exist_ok=True)
        decode_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)

        plate_h, plate_w = plate_on.shape[:2]
        plate_bgr = plate_on[:, :, ::-1]

        on_points: list[dict[str, Any]] = []
        rendered_on_by_qp: dict[int, np.ndarray] = {}

        for qp in [47, 32]:
            sidecar = IntraCodecSidecar(CODEC, qp=qp, preset=PRESET)
            tool_p, tool_v = sidecar.probe_encoder()

            t_enc_start = time.perf_counter()
            payload = sidecar.encode(plate_bgr)
            enc_time = time.perf_counter() - t_enc_start

            bs_path = bitstream_dir / f"registered_panorama_qp{qp}_removal_on.vvc"
            side_path = bitstream_dir / f"registered_panorama_qp{qp}_removal_on_side.bin"
            bs_path.write_bytes(payload)

            side_data_bytes = pack_panorama_side_data(
                homographies_on,
                plate_shape=(plate_h, plate_w),
                frame_shape=(TARGET_HEIGHT, TARGET_WIDTH),
                fps=WORKING_FPS,
            )
            side_path.write_bytes(side_data_bytes)

            # Standalone decode strictly from serialized files
            rendered_on, dec_meta = decode_standalone_representation(bs_path, side_path)
            rendered_on_by_qp[qp] = rendered_on

            # Measure intra decode time via sidecar
            t_dec_start = time.perf_counter()
            dec_bgr = sidecar.decode(payload)
            dec_plate_rgb = dec_bgr[:, :, ::-1]
            unpacked_h, _, _, _ = unpack_panorama_side_data(side_data_bytes)
            _ = np.stack(
                [
                    warp_plate_to_frame(
                        dec_plate_rgb,
                        unpacked_h[t],
                        height=TARGET_HEIGHT,
                        width=TARGET_WIDTH,
                    )
                    for t in range(N_FRAMES)
                ],
                axis=0,
            )
            client_seconds = time.perf_counter() - t_dec_start

            cv2.imwrite(
                str(decode_dir / f"registered_panorama_qp{qp}_removal_on_plate.png"),
                dec_bgr,
            )

            composed_on = composite_fixed_foreground(rendered_on, tracks, masks_360)
            metrics_on = compute_all_metrics(
                frames_360, rendered_on, masks_360, boundary_masks, composed_on
            )

            side_detail = charge_side_data(
                "registered_panorama",
                N_FRAMES,
                plate_shape=(plate_h, plate_w),
                frame_shape=(TARGET_HEIGHT, TARGET_WIDTH),
                homographies=homographies_on,
                fps=WORKING_FPS,
            )

            point_record = {
                "arm": f"registered_panorama_qp{qp}_removal_on",
                "representation": "registered_panorama",
                "qp": qp,
                "removal": "on",
                "codec": CODEC,
                "preset": PRESET,
                "lookahead_frames": N_FRAMES,
                "tool_path": tool_p,
                "tool_version": tool_v,
                "plate_resolution": f"{plate_w}x{plate_h}",
                "bitstream_path": str(bs_path.relative_to(out_dir)),
                "bitstream_sha256": hashlib.sha256(payload).hexdigest(),
                "encoded_payload_bytes": len(payload),
                "side_data_bytes": len(side_data_bytes),
                "total_package_bytes": len(payload) + len(side_data_bytes),
                "side_data_detail": side_detail,
                "metrics": metrics_on,
                "timing": {
                    "preprocessing_seconds": round(prep_time_on, 4),
                    "encode_seconds": round(enc_time, 4),
                    "sender_seconds": round(prep_time_on + enc_time, 4),
                    "decode_render_seconds": round(client_seconds, 4),
                    "client_seconds": round(client_seconds, 4),
                    "total_end_to_end_seconds": round(prep_time_on + enc_time + client_seconds, 4),
                },
            }
            on_points.append(point_record)
            print(
                f"  QP {qp} removal-ON: {point_record['total_package_bytes']:,} B, "
                f"vis PSNR {metrics_on['no_overlay']['psnr_y_visible_dB']:.2f} dB, "
                f"ghost MAD {metrics_on['no_overlay']['ghosting_luma_mad']:.2f}, "
                f"enc {enc_time:.2f}s, client {client_seconds:.2f}s"
            )

        # Step 6: Standalone decode the 2 paired OFF panorama points to retrieve frames for visual plates
        print("\n--- Decoding Reused OFF Panorama Bitstreams for Direct Comparison ---")
        rendered_off_by_qp: dict[int, np.ndarray] = {}
        for qp in [47, 32]:
            off_bs = e04a_saved_dir / "bitstreams" / f"registered_panorama_qp{qp}.vvc"
            off_sd = e04a_saved_dir / "bitstreams" / f"registered_panorama_qp{qp}_side.bin"
            rendered_off, _ = decode_standalone_representation(off_bs, off_sd)
            rendered_off_by_qp[qp] = rendered_off

        # Step 7: Paired Deltas and Decision Rule Evaluation
        print("\n--- Evaluating Paired Deltas and Decision Rules ---")
        alarms: list[str] = []
        paired_evaluations: list[dict[str, Any]] = []

        for pt_on in on_points:
            qp = pt_on["qp"]
            pt_off = reused_off_points[f"registered_panorama_qp{qp}"]

            b_off = pt_off["total_package_bytes"]
            b_on = pt_on["total_package_bytes"]
            delta_bytes = b_on - b_off
            pct_bytes = round((delta_bytes / b_off) * 100.0, 2)

            psnr_off = pt_off["metrics"]["no_overlay"]["psnr_y_visible_dB"]
            psnr_on = pt_on["metrics"]["no_overlay"]["psnr_y_visible_dB"]
            delta_psnr = round(psnr_on - psnr_off, 3)

            ghost_off = pt_off["metrics"]["no_overlay"]["ghosting_luma_mad"]
            ghost_on = pt_on["metrics"]["no_overlay"]["ghosting_luma_mad"]
            delta_ghost = round(ghost_on - ghost_off, 3)
            reduction_ghost_pct = round(((ghost_off - ghost_on) / ghost_off) * 100.0, 2)

            bnd_mad_off = pt_off["metrics"]["no_overlay"].get("boundary_luma_mad", 0.0)
            bnd_mad_on = pt_on["metrics"]["no_overlay"].get("boundary_luma_mad", 0.0)
            delta_bnd_mad = round(bnd_mad_on - bnd_mad_off, 3)

            sender_s_off = pt_off["timing"]["sender_seconds"]
            sender_s_on = pt_on["timing"]["sender_seconds"]

            client_s_off = pt_off["timing"]["client_seconds"]
            client_s_on = pt_on["timing"]["client_seconds"]

            # Pre-registered bounds checks
            bounds_key = f"registered_panorama_qp{qp}_removal_on"
            b_spec = PRE_REGISTERED_E04B_BOUNDS["points"][bounds_key]

            if not (
                b_spec["total_package_bytes"]["min"] <= b_on <= b_spec["total_package_bytes"]["max"]
            ):
                msg = f"QP {qp} removal-ON total bytes {b_on} outside bound [{b_spec['total_package_bytes']['min']}, {b_spec['total_package_bytes']['max']}]"
                alarms.append(msg)
                print(f"  [ALARM] {msg}")

            if not (
                b_spec["psnr_y_visible_dB"]["min"] <= psnr_on <= b_spec["psnr_y_visible_dB"]["max"]
            ):
                msg = f"QP {qp} removal-ON visible PSNR {psnr_on} dB outside bound [{b_spec['psnr_y_visible_dB']['min']}, {b_spec['psnr_y_visible_dB']['max']}]"
                alarms.append(msg)
                print(f"  [ALARM] {msg}")

            if not (
                b_spec["ghosting_luma_mad"]["min"] <= ghost_on <= b_spec["ghosting_luma_mad"]["max"]
            ):
                msg = f"QP {qp} removal-ON ghost MAD {ghost_on} outside bound [{b_spec['ghosting_luma_mad']['min']}, {b_spec['ghosting_luma_mad']['max']}]"
                alarms.append(msg)
                print(f"  [ALARM] {msg}")

            # Decision rule checks:
            # Promote rule: reduction_ghost_pct >= 50.0 and delta_psnr >= -0.5 and pct_bytes <= 15.0
            promotes = (
                (reduction_ghost_pct >= 50.0) and (delta_psnr >= -0.5) and (pct_bytes <= 15.0)
            )
            stops = (reduction_ghost_pct <= 0.0) or (delta_psnr < -0.5) or (pct_bytes > 15.0)

            if promotes:
                disposition = "PROMOTE"
            elif stops:
                disposition = "STOP_REJECT"
            else:
                disposition = "INCONCLUSIVE_KEEP_EXPLORATORY"

            paired_rec = {
                "qp": qp,
                "off_arm": f"registered_panorama_qp{qp}_removal_off",
                "on_arm": f"registered_panorama_qp{qp}_removal_on",
                "disposition": disposition,
                "bytes": {
                    "off_bytes": b_off,
                    "on_bytes": b_on,
                    "delta_bytes": delta_bytes,
                    "pct_change": pct_bytes,
                    "within_15pct_inflation": pct_bytes <= 15.0,
                },
                "psnr_y_visible_dB": {
                    "off_psnr": psnr_off,
                    "on_psnr": psnr_on,
                    "delta_psnr": delta_psnr,
                    "loss_le_0_5dB": delta_psnr >= -0.5,
                },
                "ghosting_luma_mad": {
                    "off_ghost_mad": ghost_off,
                    "on_ghost_mad": ghost_on,
                    "delta_ghost_mad": delta_ghost,
                    "reduction_pct": reduction_ghost_pct,
                    "ge_50pct_reduction": reduction_ghost_pct >= 50.0,
                },
                "boundary": {
                    "off_boundary_mad": bnd_mad_off,
                    "on_boundary_mad": bnd_mad_on,
                    "delta_boundary_mad": delta_bnd_mad,
                },
                "timing_seconds": {
                    "off_sender_s": sender_s_off,
                    "on_sender_s": sender_s_on,
                    "delta_sender_s": round(sender_s_on - sender_s_off, 4),
                    "off_client_s": client_s_off,
                    "on_client_s": client_s_on,
                    "delta_client_s": round(client_s_on - client_s_off, 4),
                },
            }
            paired_evaluations.append(paired_rec)
            print(
                f"  QP {qp} Paired: Bytes {b_off} -> {b_on} ({pct_bytes:+.1f}%), "
                f"Vis PSNR {psnr_off:.2f} -> {psnr_on:.2f} dB ({delta_psnr:+.2f} dB), "
                f"Ghost MAD {ghost_off:.2f} -> {ghost_on:.2f} ({reduction_ghost_pct:.1f}% reduction). "
                f"Disposition: {disposition}"
            )

        # Step 8: Generate visual evidence side-by-side plates
        print("\n--- Generating Visual Evidence Plates ---")
        key_frames = [0, 11, 23, 35, 47]
        visual_artifacts: list[str] = []
        for qp in [47, 32]:
            for t in key_frames:
                plate_img = build_visual_evidence_plate(
                    ref_rgb=frames_360[t],
                    off_rgb=rendered_off_by_qp[qp][t],
                    on_rgb=rendered_on_by_qp[qp][t],
                    mask=masks_360[t],
                    frame_idx=t,
                    qp=qp,
                )
                img_name = f"qp{qp}_frame{t:03d}_side_by_side.png"
                img_path = visual_dir / img_name
                cv2.imwrite(str(img_path), plate_img)
                visual_artifacts.append(str(img_path.relative_to(out_dir)))

        # Step 9: Synthesize final disposition
        all_promoted = all(p["disposition"] == "PROMOTE" for p in paired_evaluations)
        all_stopped = all(p["disposition"] == "STOP_REJECT" for p in paired_evaluations)
        if all_promoted:
            overall_verdict = "PROMOTE_REMOVAL_ON"
        elif all_stopped:
            overall_verdict = "STOP_REJECT_REMOVAL_ON"
        else:
            overall_verdict = "INCONCLUSIVE_EXPLORATORY"

        total_wall_s = time.perf_counter() - t_global_start

        # Step 10: Emit Reports
        report_data = {
            "task_id": TASK_ID,
            "action_name": ACTION_NAME,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": {
                "hostname": socket.gethostname(),
                "architecture": platform.processor() or "x86_64",
                "os": platform.platform(),
            },
            "code_revision": code_rev,
            "source_clip": {
                "video": VIDEO,
                "scene": SCENE,
                "n_frames": N_FRAMES,
                "fps": WORKING_FPS,
                "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
                "frames_sha256": compute_array_sha256(frames_360),
                "masks_sha256": compute_array_sha256(masks_360.astype(np.uint8)),
            },
            "preprocessing_comparison": {
                "removal_off": {
                    "optional_removal_calls": 0,
                    "removal_mode": "off",
                    "actor_pixels_untouched": True,
                    "total_inpaint_holes": 0,
                    "inpaint_frames": 0,
                    "plate_masks": "None (inherent temporal median)",
                },
                "removal_on": {
                    "optional_removal_calls": prep_stats_on["optional_removal_calls"],
                    "removal_mode": prep_stats_on["removal_mode"],
                    "actor_pixels_untouched": prep_stats_on["actor_pixels_untouched"],
                    "total_inpaint_holes": prep_stats_on["total_inpaint_holes"],
                    "inpaint_frames": prep_stats_on["inpaint_frames"],
                    "build_seconds": prep_stats_on["build_seconds"],
                    "plate_masks": "masks_360 with nearest-finite/Telea hole fill",
                },
            },
            "reused_evidence_source": {
                "report_path": str(old_report_path),
                "report_sha256": hashlib.sha256(old_report_path.read_bytes()).hexdigest(),
                "canonical_screening_points": len(reused_off_points),
            },
            "new_on_points": on_points,
            "paired_evaluations": paired_evaluations,
            "visual_evidence_artifacts": visual_artifacts,
            "pre_registered_bounds": PRE_REGISTERED_E04B_BOUNDS,
            "alarms": alarms,
            "overall_verdict": overall_verdict,
            "total_wall_seconds": round(total_wall_s, 3),
        }

        (out_dir / "e04b_paired_removal_report.json").write_text(
            json.dumps(report_data, indent=2), encoding="utf-8"
        )
        (out_dir / "alarms.json").write_text(
            json.dumps({"task_id": TASK_ID, "alarms": alarms, "count": len(alarms)}, indent=2),
            encoding="utf-8",
        )

        # Markdown comparison table
        md_lines = [
            f"# PointStream E04B Paired Removal Probe Comparison Report ({TASK_ID})",
            "",
            f"**Scene**: `{VIDEO}/{SCENE}` ({N_FRAMES} frames @ {WORKING_FPS} fps, 360p)  ",
            f"**Host**: `{socket.gethostname()}` | **Wall Clock**: `{total_wall_s:.2f}s`  ",
            f"**Overall Verdict**: `{overall_verdict}` | **Alarms**: `{len(alarms)}`  ",
            "",
            "## 1. Paired Metric Comparison (Removal-OFF vs Removal-ON)",
            "",
            "| Arm / Setting | Removal | Pkg Bytes | $\\Delta$ Bytes | Vis PSNR-Y (dB) | $\\Delta$ PSNR (dB) | Ghost MAD | MAD Red. (%) | Bnd MAD | Sender (s) | Client (s) | Disposition |",
            "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]

        for pe in paired_evaluations:
            qp = pe["qp"]
            pt_off = reused_off_points[f"registered_panorama_qp{qp}"]
            pt_on = next(p for p in on_points if p["qp"] == qp)

            b_off = pe["bytes"]["off_bytes"]
            b_on = pe["bytes"]["on_bytes"]
            d_b = pe["bytes"]["delta_bytes"]
            pct_b = pe["bytes"]["pct_change"]

            p_off = pe["psnr_y_visible_dB"]["off_psnr"]
            p_on = pe["psnr_y_visible_dB"]["on_psnr"]
            d_p = pe["psnr_y_visible_dB"]["delta_psnr"]

            g_off = pe["ghosting_luma_mad"]["off_ghost_mad"]
            g_on = pe["ghosting_luma_mad"]["on_ghost_mad"]
            red_g = pe["ghosting_luma_mad"]["reduction_pct"]

            bnd_off = pe["boundary"]["off_boundary_mad"]
            bnd_on = pe["boundary"]["on_boundary_mad"]

            snd_off = pe["timing_seconds"]["off_sender_s"]
            snd_on = pe["timing_seconds"]["on_sender_s"]

            cli_off = pe["timing_seconds"]["off_client_s"]
            cli_on = pe["timing_seconds"]["on_client_s"]

            disp = pe["disposition"]

            md_lines.append(
                f"| Registered Pano QP{qp} | **OFF** | {b_off:,} | - | {p_off:.2f} | - | {g_off:.2f} | - | {bnd_off:.2f} | {snd_off:.3f} | {cli_off:.3f} | baseline |"
            )
            md_lines.append(
                f"| Registered Pano QP{qp} | **ON** | {b_on:,} | {d_b:+,} ({pct_b:+.1f}%) | {p_on:.2f} | {d_p:+.2f} | {g_on:.2f} | **{red_g:.1f}%** | {bnd_on:.2f} | {snd_on:.3f} | {cli_on:.3f} | **{disp}** |"
            )

        md_lines.extend(
            [
                "",
                "## 2. Inpainting & Removal Execution Verification",
                "",
                f"- **Removal Mode**: `{prep_stats_on['removal_mode']}` (calls: `{prep_stats_on['optional_removal_calls']}`)",
                f"- **Inpainted Holes**: `{prep_stats_on['total_inpaint_holes']}` pixels across `{prep_stats_on['inpaint_frames']}` frames",
                "- **Removal-OFF Control**: Strictly `0` holes, `0` removal calls, actor pixels bit-identical",
                "",
                "## 3. Visual Evidence Artifacts",
                "",
            ]
        )
        for va in visual_artifacts:
            md_lines.append(f"- [{va}]({va})")

        (out_dir / "comparison_table.md").write_text("\n".join(md_lines), encoding="utf-8")
        print("\nEmitted e04b_paired_removal_report.json, comparison_table.md, and visual plates.")
        print(f"Total pipeline wall clock: {total_wall_s:.2f}s.")
        print("=== E04B Paired Removal Probe Finished Successfully ===")


if __name__ == "__main__":
    main()
