"""Benchmark independent YOLO26x post-decode pose recovery vs. PointStream direct paid joints on Shanghai."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
import os
from pathlib import Path
from statistics import mean
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from experiments.tennis import common_player_packet as packet
from experiments.tennis.player_requalification import sha256
from experiments.tennis.test_extension_native_score import select_arm
from experiments.tennis.test_joint_native_motion import inputs as load_shanghai_arms
from src.pointstream.tracked_patch import yuv_to_rgb

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
NATIVE_CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-joint-extension-20261122/native"
sys.path.insert(0, str(CARD))
from score_native import FPS, NT, StreamingYUV, get_support, prep  # noqa: E402

YOLO26X_SHA256 = "08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d"
YOLO26L_SHA256 = "ad33da8a29ea5772318c4c980844e47b56792d2b63815ad4e8e09c078c7d1abf"
CARRIER_SHA256 = "3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af"
SYDNEY_CARRIER_SHA256 = "a70a444b65f905285830a5fc2561e752708ac24ea0a5e02d538d85ac742dd99c"

COCO_SIGMAS = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
    dtype=np.float64,
)


def box_iou(a: tuple[int, int, int, int] | np.ndarray, b: tuple[int, int, int, int] | np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def predict_full(model: YOLO, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = model.predict(bgr, imgsz=1280, conf=0.15, verbose=False, device="cpu")[0]
    boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4), dtype=np.float32)
    kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None and len(res.boxes) else np.zeros((0, 17, 3), dtype=np.float32)
    return boxes, kps


def predict_crop(model: YOLO, bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Top-down padded crop pose prediction, mapped back to 1080p native pixels."""
    h_img, w_img = bgr.shape[:2]
    bw = max(16.0, float(box[2] - box[0]))
    bh = max(16.0, float(box[3] - box[1]))
    pad_x = max(32.0, 0.45 * bw)
    pad_y = max(32.0, 0.35 * bh)
    x0 = max(0, int(round(box[0] - pad_x)))
    y0 = max(0, int(round(box[1] - pad_y)))
    x1 = min(w_img, int(round(box[2] + pad_x)))
    y1 = min(h_img, int(round(box[3] + pad_y)))
    crop = bgr[y0:y1, x0:x1]
    res = model.predict(crop, imgsz=384, conf=0.15, verbose=False, device="cpu")[0]
    if not len(res.boxes) or res.keypoints is None:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    kps = res.keypoints.data.cpu().numpy()
    local_target = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
    best = int(np.argmax([box_iou(b, local_target) for b in boxes]))
    if box_iou(boxes[best], local_target) < 0.15:
        return None
    kp = kps[best].copy()
    kp[:, 0] += x0
    kp[:, 1] += y0
    return kp


def match_player(
    pred_boxes: np.ndarray,
    pred_kps: np.ndarray,
    target_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    if len(pred_boxes) == 0:
        return None
    ious = [box_iou(b, target_box) for b in pred_boxes]
    best = int(np.argmax(ious))
    if ious[best] >= 0.15:
        return pred_kps[best]
    return None


def score_pose(
    pred: np.ndarray | None,
    ref: np.ndarray,
    ref_box: tuple[int, int, int, int],
    conf_thr: float = 0.3,
) -> dict[str, Any]:
    bw = max(1.0, float(ref_box[2] - ref_box[0]))
    bh = max(1.0, float(ref_box[3] - ref_box[1]))
    area = bw * bh
    valid = ref[:, 2] >= conf_thr
    n_valid = int(valid.sum())
    if pred is None or n_valid == 0:
        return {
            "detected": False,
            "pck_01": 0.0,
            "pck_02": 0.0,
            "oks": 0.0,
            "ankle_err_px": None,
            "hip_err_px": None,
        }
    d = np.linalg.norm(pred[:, :2] - ref[:, :2], axis=-1)
    pred_ok = pred[:, 2] > 0.0
    hit_01 = valid & pred_ok & (d <= 0.1 * bh)
    hit_02 = valid & pred_ok & (d <= 0.2 * bh)
    e = np.exp(-(d[valid] ** 2) / (2.0 * area * (2.0 * COCO_SIGMAS[valid]) ** 2))
    oks = float(np.where(pred_ok[valid], e, 0.0).mean())

    ankle_err = None
    if valid[15] and valid[16] and pred_ok[15] and pred_ok[16]:
        ref_ank = 0.5 * (ref[15, :2] + ref[16, :2])
        pred_ank = 0.5 * (pred[15, :2] + pred[16, :2])
        ankle_err = float(np.linalg.norm(pred_ank - ref_ank))

    hip_err = None
    if valid[11] and valid[12] and pred_ok[11] and pred_ok[12]:
        ref_hip = 0.5 * (ref[11, :2] + ref[12, :2])
        pred_hip = 0.5 * (pred[11, :2] + pred[12, :2])
        hip_err = float(np.linalg.norm(pred_hip - ref_hip))

    return {
        "detected": True,
        "pck_01": float(hit_01.sum() / n_valid),
        "pck_02": float(hit_02.sum() / n_valid),
        "oks": oks,
        "ankle_err_px": ankle_err,
        "hip_err_px": hip_err,
    }


def tight_box(mask: np.ndarray, offset: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    x0, y0 = offset
    return (int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0 + 1, int(ys.max()) + y0 + 1)


def load_event_arms(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]], Fraction]:
    """Load matched arms, player support masks, and FPS for Shanghai or an extension event."""
    arms: list[dict[str, Any]] = []
    if args.event == "match102":
        arms_path = load_shanghai_arms(args)
        arms = json.loads(arms_path.read_text())
        supports, _ = get_support(args.features)
        return arms, supports, FPS

    b = json.loads((NATIVE_CARD / f"{args.event}-binding.json").read_text())
    receipts = NATIVE_CARD / f"{args.event}-crop-receipt.json"
    bank = NATIVE_CARD.parents[1] / "cohort-expansion-20261122/test-physical"
    fps = Fraction(b["fps"])
    supports, _ = get_support(
        args.features,
        receipt=receipts,
        tracks=[b["roles"]["near"], b["roles"]["far"]],
    )
    specs = [
        (args.ordinary_arm, args.ordinary, args.ordinary_report_sha256, None),
        (args.neural_arm, args.neural, args.neural_report_sha256, None),
        (args.joint_arm, args.witness, None, args.sender),
    ]
    for arm_name, data_path, rep_sha, sender_path in specs:
        ns = argparse.Namespace(
            event=args.event,
            arm=arm_name,
            data=data_path,
            report_sha256=rep_sha,
            sender=sender_path,
            receipt=None,
        )
        arms.append(select_arm(ns, b, bank))

    if args.roi is not None:
        if args.roi_arm == "roi-320-floor":
            sec = json.loads((NATIVE_CARD / "full-confirmation-20261123/secondary-physical-roi.json").read_text())
            row = next(r for r in sec["rows"] if r["label"] == "roi-320-floor")
            arms.append(
                dict(
                    name="roi-320-floor",
                    kind="ivf_tar",
                    width=320,
                    height=192,
                    bytes=row["roi_bytes"],
                    archive=str(args.roi),
                    member="publish/q280.ivf",
                    sha256=row["roi_sha256"],
                    decoded_sha256=row["roi_decoded_sha256"],
                )
            )
        else:
            ns_roi = argparse.Namespace(
                event=args.event,
                arm=args.roi_arm,
                data=args.roi,
                report_sha256=args.roi_report_sha256,
                sender=None,
                receipt=None,
            )
            arms.append(select_arm(ns_roi, b, bank))

    return arms, supports, fps


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected = {"smoke": 4, "full": 48}[os.environ["PS_STAGE"]]
    if args.frames != expected:
        raise ValueError(f"frames={args.frames} mismatches stage {os.environ['PS_STAGE']} ({expected})")

    expected_carrier_sha = args.carrier_sha256
    if args.event == "match127" and expected_carrier_sha == CARRIER_SHA256:
        expected_carrier_sha = SYDNEY_CARRIER_SHA256
    for path, h in (
        (args.yolo26x, YOLO26X_SHA256),
        (args.yolo26l, YOLO26L_SHA256),
        (args.carrier, expected_carrier_sha),
    ):
        if sha256(path) != h:
            raise ValueError(f"hash mismatch for {path}")

    arms, supports, fps = load_event_arms(args)

    carrier_bytes = args.carrier.read_bytes()
    t_parse0 = time.perf_counter()
    for _ in range(30):
        _, _, decoded_packet = packet.unpack(carrier_bytes)
    parse_ms_total = (time.perf_counter() - t_parse0) * 1000.0 / 30.0
    paid_kps = {
        "near": np.asarray(decoded_packet[f"{args.event}_near"][0]["kp"], dtype=np.float32),
        "far": np.asarray(decoded_packet[f"{args.event}_far"][0]["kp"], dtype=np.float32),
    }

    yolo26x = YOLO(str(args.yolo26x))
    yolo26l = YOLO(str(args.yolo26l))

    ticks = sorted({min(NT - 1, int(round(i * (NT - 1) / (expected - 1)))) for i in range(expected)})
    if len(ticks) != expected:
        ticks = list(range(0, NT, max(1, NT // expected)))[:expected]

    pixels_path = args.pixels if args.pixels is not None else args.source
    cap = cv2.VideoCapture(str(pixels_path), cv2.CAP_FFMPEG)
    arm_names = [r["name"] for r in arms] + ["direct_joints"]
    records_full: dict[str, list[dict[str, Any]]] = {n: [] for n in arm_names}
    records_crop: dict[str, list[dict[str, Any]]] = {n: [] for n in arm_names}
    source_agreement: list[dict[str, Any]] = []
    full_times_s: list[float] = []
    crop_times_s: list[float] = []

    with ExitStack() as scope:
        images: dict[str, Any] = {}
        for r in arms:
            work = Path(scope.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
            images[r["name"]] = prep(
                argparse.Namespace(source=args.source, source_pixels=pixels_path, frames=NT, fps=fps),
                r,
                work,
            )[0]
            if isinstance(images[r["name"]], StreamingYUV):
                scope.callback(images[r["name"]].close)

        for tick in range(NT):
            ok, bgr_src = cap.read()
            if not ok:
                raise RuntimeError(f"failed source frame {tick}")
            arm_frames: dict[str, np.ndarray] = {}
            for r in arms:
                if isinstance(images[r["name"]], StreamingYUV) or tick in ticks:
                    arm_frames[r["name"]] = images[r["name"]][tick]

            if tick not in ticks:
                continue

            near_box = tight_box(supports[tick][0][1], supports[tick][0][0][:2])
            far_box = tight_box(supports[tick][1][1], supports[tick][1][0][:2])
            role_boxes = {"near": near_box, "far": far_box}

            # High-res top-down crop reference from independent YOLO26x-Pose, cross-checked against YOLO26l-Pose
            ref_by_role: dict[str, np.ndarray] = {}
            for role, box in role_boxes.items():
                ref_kp = predict_crop(yolo26x, bgr_src, box)
                alt_kp = predict_crop(yolo26l, bgr_src, box)
                if ref_kp is not None:
                    ref_by_role[role] = ref_kp
                    if alt_kp is not None:
                        source_agreement.append({"tick": tick, "role": role, **score_pose(alt_kp, ref_kp, box)})

            for role, ref_kp in ref_by_role.items():
                sc = score_pose(paid_kps[role][tick], ref_kp, role_boxes[role])
                records_full["direct_joints"].append({"tick": tick, "role": role, **sc})
                records_crop["direct_joints"].append({"tick": tick, "role": role, **sc})

            for r in arms:
                name = r["name"]
                fr = arm_frames[name]
                if r["kind"] not in ("rgb_npy", "rgb_zseq"):
                    rgb = cv2.resize(
                        yuv_to_rgb(fr, r["width"], r["height"], False),
                        (1920, 1080),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    rgb = fr
                bgr_arm = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                t0 = time.perf_counter()
                p_boxes, p_kps = predict_full(yolo26x, bgr_arm)
                full_times_s.append(time.perf_counter() - t0)

                tc0 = time.perf_counter()
                for role, ref_kp in ref_by_role.items():
                    box = role_boxes[role]
                    pred_f = match_player(p_boxes, p_kps, box)
                    records_full[name].append({"tick": tick, "role": role, **score_pose(pred_f, ref_kp, box)})
                    pred_c = predict_crop(yolo26x, bgr_arm, box)
                    records_crop[name].append({"tick": tick, "role": role, **score_pose(pred_c, ref_kp, box)})
                crop_times_s.append(time.perf_counter() - tc0)

    cap.release()

    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ank = [r["ankle_err_px"] for r in rows if r["ankle_err_px"] is not None]
        hip = [r["hip_err_px"] for r in rows if r["hip_err_px"] is not None]
        return {
            "n": len(rows),
            "detect_recall": mean(1.0 if r["detected"] else 0.0 for r in rows),
            "pck_01": mean(r["pck_01"] for r in rows),
            "pck_02": mean(r["pck_02"] for r in rows),
            "oks": mean(r["oks"] for r in rows),
            "ankle_err_px": mean(ank) if ank else None,
            "ankle_valid": len(ank),
            "hip_err_px": mean(hip) if hip else None,
            "hip_valid": len(hip),
        }

    summary_full = {
        name: {
            "overall": agg(rows),
            "near": agg([r for r in rows if r["role"] == "near"]),
            "far": agg([r for r in rows if r["role"] == "far"]),
        }
        for name, rows in records_full.items()
    }
    summary_crop = {
        name: {
            "overall": agg(rows),
            "near": agg([r for r in rows if r["role"] == "near"]),
            "far": agg([r for r in rows if r["role"] == "far"]),
        }
        for name, rows in records_crop.items()
    }
    out = {
        "event": args.event,
        "frames_evaluated": len(ticks),
        "ticks": ticks,
        "source_yolo26x_vs_yolo26l": agg(source_agreement),
        "timing": {
            "direct_packet_parse_240frames_ms": parse_ms_total,
            "direct_packet_parse_per_frame_ms": parse_ms_total / NT,
            "yolo26x_full_frame_cpu_s": mean(full_times_s),
            "yolo26x_two_crops_cpu_s": mean(crop_times_s),
        },
        "summary_full_frame": summary_full,
        "summary_topdown_crop": summary_crop,
        "records_full_frame": records_full,
        "records_topdown_crop": records_crop,
    }
    (stage / "pose_machine.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def validate(args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    expected = {"smoke": 4, "full": 48}[os.environ["PS_STAGE"]]
    data = json.loads((stage / "pose_machine.json").read_text())
    assert data["frames_evaluated"] == expected
    agree = data["source_yolo26x_vs_yolo26l"]
    assert agree["pck_02"] >= 0.70, f"source YOLO26x vs YOLO26l PCK@0.2 too low: {agree}"
    n_obs = data["summary_full_frame"]["direct_joints"]["overall"]["n"]
    assert n_obs >= int(1.5 * expected)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "frames_evaluated": expected,
                    "source_agreement_pck02": agree["pck_02"],
                    "observations": n_obs,
                },
                "event": data.get("event", "match102"),
                "timing": data["timing"],
                "summary_full_frame": data["summary_full_frame"],
                "summary_topdown_crop": data["summary_topdown_crop"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "source",
        "features",
        "ordinary",
        "neural",
        "witness",
        "carrier",
        "yolo26x",
        "yolo26l",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--pixels", type=Path)
    p.add_argument("--roi", type=Path)
    p.add_argument("--sender", type=Path)
    p.add_argument("--event", choices=("match102", "match127", "match105"), default="match102")
    p.add_argument("--carrier-sha256", default=CARRIER_SHA256)
    p.add_argument("--ordinary-arm", default="ordinary-320-warm")
    p.add_argument("--roi-arm", default="roi-320-floor")
    p.add_argument("--neural-arm", default="dcvc-480-q26")
    p.add_argument("--joint-arm", default="joint-crf70")
    p.add_argument("--ordinary-report-sha256")
    p.add_argument("--roi-report-sha256")
    p.add_argument("--neural-report-sha256")
    p.add_argument("--frames", type=int, choices=(4, 48), required=True)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        validate(p.parse_args(["--validate", *argv[3:]]))
    else:
        args = p.parse_args()
        if args.validate:
            validate(args)
        else:
            run_eval(args)


if __name__ == "__main__":
    main()
