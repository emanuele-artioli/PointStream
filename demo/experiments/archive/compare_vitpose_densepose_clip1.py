"""Archived clip-1 comparison of RTMPose, ViTPose-L, and DensePose.

RTMPose stayed the encoder. This script remains for ablations.

Compare RTMPose, ViTPose-L whole-body, and DensePose on eight frames of clip 1.

ViTPose is the COCO-WholeBody ONNX (133 joints). Hand joints are slices 91:112
and 112:133, the same order RTMPose uses with to_openpose=False. Person boxes
come from the DW-Pose YOLOX detector. When that detector finds nobody, ViTPose
runs on the full frame.

DensePose is a body IUV model. It does not emit 21 hand joints. Its panel is
the detected hand-part region, and it is left out of the joint packet and the
pairwise joint matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.hand_keypoints import FINGER_COLORS, HAND_CONNECTIONS, FrameHandPose, SingleHand

logger = logging.getLogger("vitpose_densepose_compare")

CLIP = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated/clip_01_factory035_close_hands.mp4")
VITPOSE_ONNX = Path(
    "/home/itec/emanuele/pointstream-data/weights/vitpose/onnx/wholebody/vitpose-l-wholebody.onnx"
)
DENSEPOSE_WEIGHTS = Path(
    "/home/itec/emanuele/pointstream-data/weights/densepose/model_final_c6ab63.pkl"
)
DENSEPOSE_CONFIG = Path(
    "/home/itec/emanuele/pointstream-data/third_party/detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
)
FRAME_W = 1920
FRAME_H = 1080
FPS = 30.0
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
# DensePose chart parts. 3 is the right hand and 4 is the left hand.
HAND_PARTS = {3, 4}


def payload_kbps(packets: list[bytes], fps: float = FPS) -> float:
    if not packets:
        return 0.0
    return (sum(len(packet) for packet in packets) * 8) / (len(packets) / fps) / 1000.0


def _tool(name: str) -> str:
    candidate = Path("/opt/local/bin") / name
    return str(candidate) if candidate.is_file() else name


def sample_clip(source: Path, dest: Path, count: int) -> None:
    probe = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json", str(source),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    rate = stream.get("r_frame_rate", "30/1")
    num, den = rate.split("/")
    fps = float(num) / float(den)
    frames = int(stream.get("nb_frames") or 0)
    if frames <= 0:
        frames = max(1, int(float(stream.get("duration") or 1.0) * fps))
    if frames <= count:
        indices = list(range(frames))
    else:
        indices = [int(round(i * (frames - 1) / (count - 1))) for i in range(count)]
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _tool("ffmpeg"), "-y", "-i", str(source), "-vf", f"select='{select}'",
            "-vsync", "vfr", str(dest),
        ],
        check=True, capture_output=True,
    )
    logger.info("sampled frames %s from %s into %s", indices, frames, dest)


def read_frames(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (FRAME_W, FRAME_H):
            frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def pose_from_hands(frame_idx: int, hands: list[tuple[str, np.ndarray]]) -> FrameHandPose:
    built: list[SingleHand] = []
    for side, pixels in hands:
        pts = np.asarray(pixels, dtype=np.float64).reshape(21, 2)
        x1 = int(max(0, np.min(pts[:, 0])))
        y1 = int(max(0, np.min(pts[:, 1])))
        x2 = int(min(FRAME_W, np.max(pts[:, 0])))
        y2 = int(min(FRAME_H, np.max(pts[:, 1])))
        norms = [[float(p[0]) / FRAME_W, float(p[1]) / FRAME_H, 0.0] for p in pts]
        built.append(
            SingleHand(side, 0.9, [x1, y1, x2, y2], norms, [[float(p[0]), float(p[1])] for p in pts])
        )
    return FrameHandPose(frame_idx=frame_idx, hands=built)


def encoder_report(poses: list[FrameHandPose]) -> dict[str, float]:
    from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
    from demo.pipeline.keypoint_compressor import KeypointCompressor

    packets = [KeypointCompressor.compress_frame(pose, FRAME_W, FRAME_H) for pose in poses]
    decoded = [
        FrameHandPose(index, KeypointCompressor.decompress_frame(packet, FRAME_W, FRAME_H))
        for index, packet in enumerate(packets)
    ]
    scores = score_pose_tracks(poses, decoded)
    return {
        "keypoint_kbps": payload_kbps(packets),
        "roundtrip_mpjpe_px": scores["mpjpe_pixels"],
        "roundtrip_pck50_all_gt": scores["pck50_all_gt"],
        "mean_packet_bytes": float(np.mean([len(packet) for packet in packets])) if packets else 0.0,
    }


def person_boxes(detector, frame: np.ndarray) -> list[list[float]]:
    boxes = detector(frame)
    if boxes is None or len(np.asarray(boxes)) == 0:
        return []
    return [box.tolist() for box in np.asarray(boxes).reshape(-1, 4)]


def vitpose_poses(model, detector, frames: list[np.ndarray]) -> list[FrameHandPose]:
    poses: list[FrameHandPose] = []
    full_frame_count = 0
    for index, frame in enumerate(frames):
        boxes = person_boxes(detector, frame)
        if not boxes:
            boxes = [[0, 0, FRAME_W, FRAME_H]]
            full_frame_count += 1
        keypoints, scores = model(frame, boxes)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
            scores = scores[None, ...]
        hands: list[tuple[str, np.ndarray]] = []
        for person, person_scores in zip(keypoints, scores):
            for side, sl in (("Left", LEFT_HAND), ("Right", RIGHT_HAND)):
                pts = person[sl]
                conf = person_scores[sl]
                if pts.shape[0] < 21 or float(np.mean(conf)) < 0.25:
                    continue
                hands.append((side, pts[:, :2]))
        poses.append(pose_from_hands(index, hands))
    logger.info("ViTPose full-frame fallback on %s/%s frames", full_frame_count, len(frames))
    return poses


def draw_pose(frame: np.ndarray, pose: FrameHandPose, label: str) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    for hand in pose.hands:
        points = hand.landmarks_pixel
        for bone_index, (start, end) in enumerate(HAND_CONNECTIONS):
            p1 = (int(points[start][0]), int(points[start][1]))
            p2 = (int(points[end][0]), int(points[end][1]))
            cv2.line(canvas, p1, p2, FINGER_COLORS[min(bone_index // 4, 4)], 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
    cv2.rectangle(canvas, (0, 0), (460, 36), (0, 0, 0), -1)
    cv2.putText(canvas, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (460, 36), (0, 0, 0), -1)
    cv2.putText(canvas, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def load_densepose():
    if not DENSEPOSE_WEIGHTS.is_file() or not DENSEPOSE_CONFIG.is_file():
        raise FileNotFoundError(f"missing DensePose files: {DENSEPOSE_WEIGHTS} {DENSEPOSE_CONFIG}")
    densepose_root = DENSEPOSE_CONFIG.parents[1]
    if str(densepose_root) not in sys.path:
        sys.path.insert(0, str(densepose_root))
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from densepose import add_densepose_config

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(str(DENSEPOSE_CONFIG))
    cfg.MODEL.WEIGHTS = str(DENSEPOSE_WEIGHTS)
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return DefaultPredictor(cfg)


def densepose_panels(predictor, frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    import cv2
    from densepose.vis.extractor import DensePoseResultExtractor

    extractor = DensePoseResultExtractor()
    panels = []
    part_counts: dict[str, int] = {}
    hand_pixels = 0
    for frame in frames:
        canvas = frame.copy()
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        results, boxes_xywh = extractor(instances)
        if results and boxes_xywh is not None:
            for result, box in zip(results, np.asarray(boxes_xywh)):
                if result is None:
                    continue
                labels = np.asarray(result.labels)
                unique, counts = np.unique(labels, return_counts=True)
                for part, count in zip(unique.tolist(), counts.tolist()):
                    part_counts[str(int(part))] = part_counts.get(str(int(part)), 0) + int(count)
                hand = np.isin(labels, list(HAND_PARTS))
                hand_pixels += int(hand.sum())
                x, y, width, height = [float(v) for v in np.asarray(box).reshape(-1)[:4]]
                x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
                crop = canvas[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue
                if hand.shape[:2] != crop.shape[:2]:
                    mask = cv2.resize(
                        hand.astype(np.uint8),
                        (crop.shape[1], crop.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    mask = hand.astype(np.uint8)
                tint = crop.copy()
                tint[mask > 0] = (0, 255, 255)
                crop[mask > 0] = cv2.addWeighted(crop, 0.45, tint, 0.55, 0)[mask > 0]
        panels.append(draw_label(canvas, "densepose hands"))
    summary = {"part_pixel_counts": part_counts, "hand_part_pixels": hand_pixels, "hand_part_ids": sorted(HAND_PARTS)}
    return panels, summary


def montage(columns: list[tuple[str, list[np.ndarray]]], dest: Path) -> None:
    import cv2

    rows = []
    count = len(columns[0][1])
    for index in range(count):
        row = [cv2.resize(panels[index], (480, 270), interpolation=cv2.INTER_AREA) for _, panels in columns]
        rows.append(np.concatenate(row, axis=1))
    image = np.concatenate(rows, axis=0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), image)


def judge_matrix(named: dict[str, list[FrameHandPose]]) -> dict[str, dict[str, float]]:
    from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks

    matrix: dict[str, dict[str, float]] = {}
    for reference_name, reference in named.items():
        matrix[reference_name] = {}
        for other_name, other in named.items():
            scores = score_pose_tracks(reference, other)
            matrix[reference_name][other_name] = {
                "mpjpe_px": scores["mpjpe_pixels"],
                "detection_rate": scores["detection_rate"],
                "pck50_all_gt": scores["pck50_all_gt"],
                "handedness_agreement": scores["handedness_agreement"],
            }
    return matrix


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=CLIP)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    from demo.evaluation.pose_backends import extract_rtm_hand
    from demo.pipeline.maps.model_paths import require
    from rtmlib import ViTPose, YOLOX

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sampled = out / "clip1_sample.mp4"
    sample_clip(args.video, sampled, args.frames)
    frames = read_frames(sampled)

    logger.info("running RTMPose")
    rtm = extract_rtm_hand(sampled)
    if not VITPOSE_ONNX.is_file():
        raise FileNotFoundError(VITPOSE_ONNX)
    logger.info("running ViTPose-L wholebody")
    detector = YOLOX(str(require("dwpose_det")), det_mode="human", score_thr=0.3, device="cuda")
    vitpose = ViTPose(str(VITPOSE_ONNX), to_openpose=False, device="cuda")
    vit = vitpose_poses(vitpose, detector, frames)

    named = {"rtmpose": rtm, "vitpose": vit}
    columns: list[tuple[str, list[np.ndarray]]] = [
        ("rtmpose", [draw_pose(frame, pose, "rtmpose") for frame, pose in zip(frames, rtm)]),
        ("vitpose", [draw_pose(frame, pose, "vitpose-l") for frame, pose in zip(frames, vit)]),
    ]
    densepose_note: dict = {}
    try:
        logger.info("running DensePose")
        predictor = load_densepose()
        panels, densepose_note = densepose_panels(predictor, frames)
        columns.append(("densepose", panels))
    except Exception as exc:
        logger.exception("DensePose did not run")
        densepose_note = {"error": f"{type(exc).__name__}: {exc}"}

    import cv2

    sheets = out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for label, panels in columns:
        strip = np.concatenate(
            [cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA) for panel in panels],
            axis=0,
        )
        cv2.imwrite(str(sheets / f"{label}.png"), strip)
    montage_path = out / "montage.png"
    montage(columns, montage_path)
    report = {
        "clip": str(args.video),
        "frames": len(frames),
        "vitpose_onnx": str(VITPOSE_ONNX),
        "densepose_weights": str(DENSEPOSE_WEIGHTS),
        "encoders": {name: encoder_report(poses) for name, poses in named.items()},
        "judges": judge_matrix(named),
        "densepose": densepose_note,
        "notes": {
            "vitpose": "ViTPose-L COCO-WholeBody ONNX. Hands are joints 91:112 and 112:133. Person boxes are YOLOX class 0; a missed detection uses the full frame.",
            "densepose": "R_101 FPN s1x chart. Hand parts are labels 3 and 4. No 21-joint packet.",
            "judge": "The pairwise matrix is joint agreement. DensePose is visual only.",
        },
        "montage": str(montage_path),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s", out / "report.json")


if __name__ == "__main__":
    main()
