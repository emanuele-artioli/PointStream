"""Benchmark independent DWPose post-decode pose recovery vs. PointStream direct paid joints on Shanghai."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
import onnxruntime as ort
from ultralytics import YOLO

from experiments.tennis import common_player_packet as packet
from experiments.tennis.player_requalification import sha256
from experiments.tennis.test_joint_native_motion import inputs as load_shanghai_arms
from src.pointstream.tracked_patch import yuv_to_rgb

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
from score_native import FPS, NT, StreamingYUV, get_support, prep  # noqa: E402

YOLOX_SHA256 = "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"
DWPOSE_SHA256 = "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"
YOLO26_SHA256 = "08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d"
CARRIER_SHA256 = "3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af"

COCO_SIGMAS = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
    dtype=np.float64,
)
MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.65) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: list[int] = []
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        order = rest[inter / union <= iou_thr]
    return keep


def box_iou(a: tuple[int, int, int, int] | np.ndarray, b: tuple[int, int, int, int] | np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


class DWPoseONNX:
    """Pure ONNXRuntime DWPose detector + SimCC pose estimator (CPU)."""

    def __init__(self, det_path: Path, pose_path: Path, threads: int) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.det = ort.InferenceSession(str(det_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(str(pose_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.det_in = self.det.get_inputs()[0].name
        self.pose_in = self.pose.get_inputs()[0].name
        grids, strides = [], []
        for s in (8, 16, 32):
            gs = 640 // s
            yv, xv = np.meshgrid(np.arange(gs), np.arange(gs), indexing="ij")
            grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
            strides.append(np.full((gs * gs, 1), s, dtype=np.float32))
        self.grid = np.concatenate(grids, 0).astype(np.float32)
        self.stride = np.concatenate(strides, 0)

    def detect_people(self, bgr: np.ndarray, score_thr: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
        h, w = bgr.shape[:2]
        scale = min(640.0 / h, 640.0 / w)
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        nh, nw = int(h * scale), int(w * scale)
        padded[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.det.run(None, {self.det_in: inp})[0][0]
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(np.clip(out[:, 2:4], -20.0, 20.0)) * self.stride
        scores = out[:, 4] * out[:, 5]
        mask = scores >= score_thr
        if not np.any(mask):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        boxes = np.concatenate([xy[mask] - wh[mask] / 2.0, xy[mask] + wh[mask] / 2.0], axis=1) / scale
        sc = scores[mask]
        keep = nms_boxes(boxes, sc, 0.65)
        return boxes[keep], sc[keep]

    def estimate_pose(self, bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = max(10.0, (x2 - x1) * 1.25), max(10.0, (y2 - y1) * 1.25)
        aspect = 288.0 / 384.0
        if bw > bh * aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        src_pts = np.array([[cx, cy], [cx, cy - bh * 0.5], [cx - bh * 0.5, cy]], dtype=np.float32)
        dst_pts = np.array([[144.0, 192.0], [144.0, 0.0], [0.0, 192.0]], dtype=np.float32)
        mat = cv2.getAffineTransform(src_pts, dst_pts)
        crop = cv2.warpAffine(bgr, mat, (288, 384), flags=cv2.INTER_LINEAR)
        inp = ((crop.astype(np.float32) - MEAN_BGR) / STD_BGR).transpose(2, 0, 1)[None].astype(np.float32)
        sx, sy = self.pose.run(None, {self.pose_in: inp})
        ix = np.argmax(sx[0, :17], axis=-1)
        iy = np.argmax(sy[0, :17], axis=-1)
        conf = 0.5 * (sx[0, np.arange(17), ix] + sy[0, np.arange(17), iy])
        px = ix.astype(np.float32) / 2.0
        py = iy.astype(np.float32) / 2.0
        inv_mat = cv2.invertAffineTransform(mat)
        xy = np.stack([px, py, np.ones_like(px)], axis=-1) @ inv_mat.T
        return np.concatenate([xy, conf[:, None]], axis=-1)

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, _ = self.detect_people(bgr)
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 17, 3), dtype=np.float32)
        kps = np.stack([self.estimate_pose(bgr, b) for b in boxes], axis=0)
        return boxes, kps


def match_player(
    pred_boxes: np.ndarray,
    pred_kps: np.ndarray,
    target_box: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, float]:
    if len(pred_boxes) == 0:
        return None, 0.0
    ious = [box_iou(b, target_box) for b in pred_boxes]
    best = int(np.argmax(ious))
    if ious[best] >= 0.15:
        return pred_kps[best], float(ious[best])
    # Fallback: check if any skeleton's confident joints fall inside padded target_box
    tx0, ty0, tx1, ty1 = target_box
    for i, kp in enumerate(pred_kps):
        pts = kp[kp[:, 2] > 0.2, :2]
        if len(pts) >= 3:
            inside = ((pts[:, 0] >= tx0) & (pts[:, 0] <= tx1) & (pts[:, 1] >= ty0) & (pts[:, 1] <= ty1)).mean()
            if inside >= 0.5:
                return kp, float(ious[i])
    return None, 0.0


def score_pose(pred: np.ndarray | None, ref: np.ndarray, ref_box: tuple[int, int, int, int], conf_thr: float = 0.3) -> dict[str, Any]:
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


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected = {"smoke": 4, "full": 48}[os.environ["PS_STAGE"]]
    if args.frames != expected:
        raise ValueError(f"frames={args.frames} mismatches stage {os.environ['PS_STAGE']} ({expected})")

    for path, h in (
        (args.dwpose_det, YOLOX_SHA256),
        (args.dwpose_pose, DWPOSE_SHA256),
        (args.yolo26, YOLO26_SHA256),
        (args.carrier, CARRIER_SHA256),
    ):
        if sha256(path) != h:
            raise ValueError(f"hash mismatch for {path}")

    arms_path = load_shanghai_arms(args)
    arms = json.loads(arms_path.read_text())
    supports, _ = get_support(args.features)

    # Parse paid direct 17-joint packet and time CPU parsing across 30 repetitions.
    carrier_bytes = args.carrier.read_bytes()
    t_parse0 = time.perf_counter()
    for _ in range(30):
        _, _, decoded_packet = packet.unpack(carrier_bytes)
    parse_ms_total = (time.perf_counter() - t_parse0) * 1000.0 / 30.0
    paid_kps = {
        "near": np.asarray(decoded_packet["match102_near"][0]["kp"], dtype=np.float32),
        "far": np.asarray(decoded_packet["match102_far"][0]["kp"], dtype=np.float32),
    }

    dwpose = DWPoseONNX(args.dwpose_det, args.dwpose_pose, threads=threads)
    yolo26 = YOLO(str(args.yolo26))

    ticks = sorted({min(NT - 1, int(round(i * (NT - 1) / (expected - 1)))) for i in range(expected)})
    if len(ticks) != expected:
        ticks = list(range(0, NT, max(1, NT // expected)))[:expected]

    cap = cv2.VideoCapture(str(args.pixels), cv2.CAP_FFMPEG)
    arm_names = [r["name"] for r in arms] + ["direct_joints"]
    records: dict[str, list[dict[str, Any]]] = {n: [] for n in arm_names}
    source_agreement: list[dict[str, Any]] = []
    dwpose_times_s: list[float] = []

    with ExitStack() as scope:
        images: dict[str, Any] = {}
        for r in arms:
            work = Path(scope.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
            images[r["name"]] = prep(
                argparse.Namespace(source=args.source, source_pixels=args.pixels, frames=NT, fps=FPS),
                r,
                work,
            )[0]
            if isinstance(images[r["name"]], StreamingYUV):
                scope.callback(images[r["name"]].close)

        for tick in range(NT):
            ok, bgr_src = cap.read()
            if not ok:
                raise RuntimeError(f"failed source frame {tick}")
            # Advance StreamingYUV sequentially if any arm uses it; otherwise memmap random access is fine.
            arm_frames: dict[str, np.ndarray] = {}
            for r in arms:
                if isinstance(images[r["name"]], StreamingYUV) or tick in ticks:
                    arm_frames[r["name"]] = images[r["name"]][tick]

            if tick not in ticks:
                continue

            # Unpadded tight boxes from SAM source supports
            near_box = tight_box(supports[tick][0][1], supports[tick][0][0][:2])
            far_box = tight_box(supports[tick][1][1], supports[tick][1][0][:2])
            role_boxes = {"near": near_box, "far": far_box}

            dw_src_boxes, dw_src_kps = dwpose.predict(bgr_src)
            y26_res = yolo26.predict(bgr_src, imgsz=1280, conf=0.25, verbose=False, device="cpu")[0]
            y26_boxes = y26_res.boxes.xyxy.cpu().numpy() if len(y26_res.boxes) else np.zeros((0, 4))
            y26_kps = y26_res.keypoints.data.cpu().numpy() if y26_res.keypoints is not None else np.zeros((0, 17, 3))

            ref_by_role: dict[str, np.ndarray] = {}
            for role, box in role_boxes.items():
                dw_kp, _ = match_player(dw_src_boxes, dw_src_kps, box)
                y26_kp, _ = match_player(y26_boxes, y26_kps, box)
                if dw_kp is not None:
                    ref_by_role[role] = dw_kp
                    if y26_kp is not None:
                        sc = score_pose(y26_kp, dw_kp, box)
                        source_agreement.append({"tick": tick, "role": role, **sc})

            # Score direct transmitted joints against DWPose reference
            for role, ref_kp in ref_by_role.items():
                kp = paid_kps[role][tick]
                sc = score_pose(kp, ref_kp, role_boxes[role])
                records["direct_joints"].append({"tick": tick, "role": role, **sc})

            # Score post-decode DWPose on each coded arm
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
                td0 = time.perf_counter()
                p_boxes, p_kps = dwpose.predict(bgr_arm)
                dwpose_times_s.append(time.perf_counter() - td0)
                for role, ref_kp in ref_by_role.items():
                    pred_kp, _ = match_player(p_boxes, p_kps, role_boxes[role])
                    sc = score_pose(pred_kp, ref_kp, role_boxes[role])
                    records[name].append({"tick": tick, "role": role, **sc})

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

    summary = {
        name: {
            "overall": agg(rows),
            "near": agg([r for r in rows if r["role"] == "near"]),
            "far": agg([r for r in rows if r["role"] == "far"]),
        }
        for name, rows in records.items()
    }
    out = {
        "frames_evaluated": len(ticks),
        "ticks": ticks,
        "source_dwpose_vs_yolo26x": agg(source_agreement),
        "timing": {
            "direct_packet_parse_240frames_ms": parse_ms_total,
            "direct_packet_parse_per_frame_ms": parse_ms_total / NT,
            "dwpose_cpu_per_frame_s": mean(dwpose_times_s),
        },
        "summary": summary,
        "records": records,
    }
    (stage / "pose_machine.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def validate(args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    expected = {"smoke": 4, "full": 48}[os.environ["PS_STAGE"]]
    data = json.loads((stage / "pose_machine.json").read_text())
    assert data["frames_evaluated"] == expected
    agree = data["source_dwpose_vs_yolo26x"]
    assert agree["pck_02"] >= 0.70, f"source DWPose vs YOLO26x PCK@0.2 too low: {agree}"
    assert data["summary"]["direct_joints"]["overall"]["n"] >= int(1.8 * expected)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "frames_evaluated": expected,
                "source_agreement": agree,
                "timing": data["timing"],
                "summary": data["summary"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "source",
        "pixels",
        "features",
        "ordinary",
        "roi",
        "neural",
        "witness",
        "carrier",
        "dwpose-det",
        "dwpose-pose",
        "yolo26",
    ):
        p.add_argument("--" + k, type=Path, required=True)
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
