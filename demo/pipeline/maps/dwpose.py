"""Pack DWPose hands (PK), face (PF), and body (PB) into one DWB2 payload.

The combined file is the counted pose payload: a presence count per part, then
box-relative u8 landmarks delta-coded with exponential-Golomb. Empty parts are
not written out as packets. The per-part PK/PF/PB files stay the quantizer's
intermediate form and are what the preview is drawn from.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from demo.evaluation.pose_backends import (
    FrameWholeBody,
    dwpose_ort_device,
    extract_dwpose_wholebody,
    wholebody_to_frame_hands,
)
from demo.evaluation.profile_map import gpu_name, profile_map
from demo.pipeline.background_codec import read_video_frames_robust
from demo.pipeline.hand_keypoints import HAND_CONNECTIONS, FrameHandPose, SingleHand
from demo.pipeline.keypoint_compressor import KeypointCompressor
from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.model_paths import require
from demo.pipeline.maps.pose_delta import MAGIC_DWB2, encode_pose_stream
from demo.pipeline.maps.rgba_preview import write_rgba_png_sequence

MAGIC_FACE = b"PF"
MAGIC_BODY = b"PB"
MAGIC_COMBINED = MAGIC_DWB2

COCO17_LIMBS = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass
class LandmarkInstance:
    confidence: float
    bbox: list[int]
    landmarks_pixel: list[list[float]]
    scores: list[float] | None = None


def _bbox_from_xy(
    xy: np.ndarray,
    scores: np.ndarray,
    frame_w: int,
    frame_h: int,
    conf_thr: float = 0.3,
    pad: float = 0.25,
    min_visible: int = 4,
    min_span: float = 8.0,
    max_span_frac: float = 0.85,
) -> list[int] | None:
    vis = np.asarray(scores) >= conf_thr
    if int(np.count_nonzero(vis)) < int(min_visible):
        return None
    pts = xy[vis]
    if pts.size == 0:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    span_x = float(xs.max() - xs.min())
    span_y = float(ys.max() - ys.min())
    if span_x < min_span and span_y < min_span:
        return None
    if span_x > max_span_frac * frame_w and span_y > max_span_frac * frame_h:
        return None
    pad_x = max(1.0, span_x * pad)
    pad_y = max(1.0, span_y * pad)
    x1 = int(max(0, xs.min() - pad_x))
    y1 = int(max(0, ys.min() - pad_y))
    x2 = int(min(frame_w, xs.max() + pad_x))
    y2 = int(min(frame_h, ys.max() + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def instance_from_part(
    part: np.ndarray,
    frame_w: int,
    frame_h: int,
    conf_thr: float = 0.3,
    min_visible: int = 4,
    min_span: float = 8.0,
    max_span_frac: float = 0.85,
) -> LandmarkInstance | None:
    xy = np.asarray(part[:, :2], dtype=np.float32)
    scores = np.asarray(part[:, 2], dtype=np.float32) if part.shape[1] >= 3 else np.ones(len(part), dtype=np.float32)
    bbox = _bbox_from_xy(
        xy,
        scores,
        frame_w,
        frame_h,
        conf_thr=conf_thr,
        min_visible=min_visible,
        min_span=min_span,
        max_span_frac=max_span_frac,
    )
    if bbox is None:
        return None
    vis = scores >= conf_thr
    return LandmarkInstance(
        confidence=float(np.mean(scores[vis])) if np.any(vis) else 0.0,
        bbox=bbox,
        landmarks_pixel=[[float(p[0]), float(p[1])] for p in xy],
        scores=[float(s) for s in scores],
    )


def compress_instances(
    magic: bytes,
    instances: list[LandmarkInstance],
    frame_w: int,
    frame_h: int,
    n_kpts: int,
) -> bytes:
    """u8 landmarks relative to bbox. Magics: PF (face 68), PB (body 17)."""
    packet = bytearray()
    packet.extend(magic)
    n = min(len(instances), 255)
    packet.append(n)
    for inst in instances[:n]:
        conf_quant = min(127, int(inst.confidence * 127))
        packet.append(conf_quant & 0x7F)
        x1, y1, x2, y2 = inst.bbox
        bx1 = int(np.clip(round((x1 / frame_w) * 255), 0, 255))
        by1 = int(np.clip(round((y1 / frame_h) * 255), 0, 255))
        bx2 = int(np.clip(round((x2 / frame_w) * 255), 0, 255))
        by2 = int(np.clip(round((y2 / frame_h) * 255), 0, 255))
        packet.extend(struct.pack("4B", bx1, by1, bx2, by2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        lms = inst.landmarks_pixel[:n_kpts]
        while len(lms) < n_kpts:
            lms.append([float(x1), float(y1)])
        for lm in lms:
            lx = int(np.clip(round(((lm[0] - x1) / bw) * 255), 0, 255))
            ly = int(np.clip(round(((lm[1] - y1) / bh) * 255), 0, 255))
            packet.extend(struct.pack("2B", lx, ly))
    return bytes(packet)


def decompress_instances(
    data: bytes,
    magic: bytes,
    n_kpts: int,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> list[LandmarkInstance]:
    if len(data) < 3 or data[:2] != magic:
        return []
    num = data[2]
    offset = 3
    out: list[LandmarkInstance] = []
    per = 1 + 4 + n_kpts * 2
    for _ in range(num):
        if offset + per > len(data):
            break
        conf = float(data[offset] & 0x7F) / 127.0
        offset += 1
        bx1, by1, bx2, by2 = struct.unpack_from("4B", data, offset)
        offset += 4
        x1 = int((bx1 / 255.0) * frame_w)
        y1 = int((by1 / 255.0) * frame_h)
        x2 = int((bx2 / 255.0) * frame_w)
        y2 = int((by2 / 255.0) * frame_h)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        lms: list[list[float]] = []
        for _k in range(n_kpts):
            lx, ly = struct.unpack_from("2B", data, offset)
            offset += 2
            lms.append([x1 + (lx / 255.0) * bw, y1 + (ly / 255.0) * bh])
        out.append(LandmarkInstance(confidence=conf, bbox=[x1, y1, x2, y2], landmarks_pixel=lms))
    return out


def compress_face(instances: list[LandmarkInstance], frame_w: int, frame_h: int) -> bytes:
    return compress_instances(MAGIC_FACE, instances, frame_w, frame_h, 68)


def decompress_face(data: bytes, frame_w: int = 1920, frame_h: int = 1080) -> list[LandmarkInstance]:
    return decompress_instances(data, MAGIC_FACE, 68, frame_w, frame_h)


def compress_body(instances: list[LandmarkInstance], frame_w: int, frame_h: int) -> bytes:
    return compress_instances(MAGIC_BODY, instances, frame_w, frame_h, 17)


def decompress_body(data: bytes, frame_w: int = 1920, frame_h: int = 1080) -> list[LandmarkInstance]:
    return decompress_instances(data, MAGIC_BODY, 17, frame_w, frame_h)


def write_packet_stream(packets: list[bytes], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for pkt in packets:
            fh.write(struct.pack("<I", len(pkt)))
            fh.write(pkt)
    return path


def read_packet_stream(path: Path) -> list[bytes]:
    data = path.read_bytes()
    packets: list[bytes] = []
    offset = 0
    while offset + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, offset)
        offset += 4
        packets.append(data[offset : offset + n])
        offset += n
    return packets


def _clip_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 1e-3 else 30.0


def _draw_points(
    canvas: np.ndarray,
    pts: list[list[float]],
    color: tuple[int, int, int],
    radius: int = 2,
    scores: list[float] | None = None,
    conf_thr: float = 0.3,
) -> None:
    for i, pt in enumerate(pts):
        if scores is not None and i < len(scores) and scores[i] < conf_thr:
            continue
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_limbs(
    canvas: np.ndarray,
    pts: list[list[float]],
    limbs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    color: tuple[int, int, int],
    scores: list[float] | None = None,
    conf_thr: float = 0.3,
) -> None:
    for a, b in limbs:
        if a >= len(pts) or b >= len(pts):
            continue
        if scores is not None:
            if a < len(scores) and scores[a] < conf_thr:
                continue
            if b < len(scores) and scores[b] < conf_thr:
                continue
        p1 = (int(pts[a][0]), int(pts[a][1]))
        p2 = (int(pts[b][0]), int(pts[b][1]))
        cv2.line(canvas, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def render_preview_hands(pose: FrameHandPose, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for hand in pose.hands:
        _draw_limbs(canvas, hand.landmarks_pixel, HAND_CONNECTIONS, (0, 255, 0))
        _draw_points(canvas, hand.landmarks_pixel, (0, 0, 255), radius=3)
    return canvas


def render_preview_face(instances: list[LandmarkInstance], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for inst in instances:
        _draw_points(canvas, inst.landmarks_pixel, (0, 255, 255), radius=1, scores=inst.scores, conf_thr=0.4)
    return canvas


def render_preview_body(instances: list[LandmarkInstance], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for inst in instances:
        _draw_limbs(canvas, inst.landmarks_pixel, COCO17_LIMBS, (255, 180, 0), scores=inst.scores, conf_thr=0.5)
        _draw_points(canvas, inst.landmarks_pixel, (0, 0, 255), radius=3, scores=inst.scores, conf_thr=0.5)
    return canvas


def render_preview_combined(
    pose: FrameHandPose,
    face_inst: list[LandmarkInstance],
    body_inst: list[LandmarkInstance],
    width: int,
    height: int,
) -> np.ndarray:
    """BGRA canvas: transparent unless a part is present this frame."""
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr = cv2.add(bgr, render_preview_body(body_inst, width, height))
    bgr = cv2.add(bgr, render_preview_face(face_inst, width, height))
    bgr = cv2.add(bgr, render_preview_hands(pose, width, height))
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgra[:, :, 3] = np.where(luma > 0, 230, 0).astype(np.uint8)
    return bgra


def pack_combined_payload(
    hand_packets: list[bytes],
    face_packets: list[bytes],
    body_packets: list[bytes],
) -> bytes:
    if not (len(hand_packets) == len(face_packets) == len(body_packets)):
        raise ValueError("combined DW-Pose streams must have the same frame count")
    return encode_pose_stream(list(zip(hand_packets, face_packets, body_packets)))


def _write_preview_bgr(frames: list[np.ndarray], stem: Path, fps: float) -> tuple[Path, int]:
    png = stem.with_suffix(".png")
    cv2.imwrite(str(png), frames[0])
    for i, frame in enumerate(frames[1:8], start=1):
        cv2.imwrite(str(stem.with_name(f"{stem.name}.{i:02d}.png")), frame)
    mp4 = stem.with_suffix(".mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if writer.isOpened():
        for frame in frames:
            writer.write(frame)
        writer.release()
    else:
        writer.release()
    # OpenCV mp4v is not playable in Chrome; the inspector HUD uses the PNG poster.
    return png, png.stat().st_size if png.is_file() else 0


def _write_map_sidecar(
    *,
    map_name: str,
    backend: str,
    payload: Path,
    preview: Path,
    preview_bytes: int,
    duration_s: float,
    n_frames: int,
    fps: float,
    stats: dict,
    extra: dict,
    out_dir: Path,
) -> MapStream:
    stream = MapStream(
        map=map_name,
        backend=backend,
        payload_path=str(payload),
        payload_bytes=payload.stat().st_size,
        preview_path=str(preview),
        preview_bytes=preview_bytes,
        duration_s=duration_s,
        n_frames=n_frames,
        fps=fps,
        extract_ms_p50=float(stats["extract_ms_p50"]),
        extract_ms_p95=float(stats["extract_ms_p95"]),
        pack_ms_p50=float(stats["pack_ms_p50"]),
        codec_ms_p50=float(stats["codec_ms_p50"]),
        decode_ms_p50=float(stats["decode_ms_p50"]),
        gpu=str(stats.get("gpu") or gpu_name()),
        kind="native",
        extra=extra,
    )
    write_sidecar(stream, out_dir / f"{map_name}.json")
    return stream


def pack_frame_parts(
    frame: FrameWholeBody,
) -> tuple[bytes, bytes, bytes, FrameHandPose, list[LandmarkInstance], list[LandmarkInstance]]:
    hands_pose = wholebody_to_frame_hands(frame)
    face_inst: list[LandmarkInstance] = []
    body_inst: list[LandmarkInstance] = []
    for person in frame.people:
        face = instance_from_part(
            person.face,
            frame.width,
            frame.height,
            conf_thr=0.4,
            min_visible=12,
            min_span=12.0,
            max_span_frac=0.55,
        )
        if face is not None:
            face_inst.append(face)
        body = instance_from_part(
            person.body,
            frame.width,
            frame.height,
            conf_thr=0.5,
            min_visible=5,
            min_span=40.0,
            max_span_frac=0.95,
        )
        if body is not None:
            body_inst.append(body)
    hand_pkt = KeypointCompressor.compress_frame(hands_pose, frame.width, frame.height)
    face_pkt = compress_face(face_inst, frame.width, frame.height)
    body_pkt = compress_body(body_inst, frame.width, frame.height)
    return hand_pkt, face_pkt, body_pkt, hands_pose, face_inst, body_inst


def run_dwpose(clip: Path, out_dir: Path, max_frames: int | None) -> list[MapStream]:
    pose_path = require("dwpose_pose")
    require("dwpose_det")
    frames_bgr = read_video_frames_robust(clip, max_frames=max_frames)
    if not frames_bgr:
        raise RuntimeError(f"no frames decoded from {clip}")
    fps = _clip_fps(clip)
    wb_frames = extract_dwpose_wholebody(clip, max_frames=max_frames)
    n = min(len(wb_frames), len(frames_bgr))
    wb_frames = wb_frames[:n]
    duration_s = n / fps
    backend = pose_path.name

    hand_packets: list[bytes] = []
    face_packets: list[bytes] = []
    body_packets: list[bytes] = []
    prev_combined: list[np.ndarray] = []
    n_hands_detected = 0
    n_face_detected = 0
    n_body_detected = 0
    n_blank = 0

    for wb in wb_frames:
        hand_pkt, face_pkt, body_pkt, hands_pose, face_inst, body_inst = pack_frame_parts(wb)
        n_hands_detected += len(hands_pose.hands)
        n_face_detected += len(face_inst)
        n_body_detected += len(body_inst)
        if not hands_pose.hands and not face_inst and not body_inst:
            n_blank += 1
        hand_packets.append(hand_pkt)
        face_packets.append(face_pkt)
        body_packets.append(body_pkt)
        prev_combined.append(render_preview_combined(hands_pose, face_inst, body_inst, wb.width, wb.height))

    write_packet_stream(hand_packets, out_dir / "dwpose_hands.pk.bin")
    write_packet_stream(face_packets, out_dir / "dwpose_face.pf.bin")
    write_packet_stream(body_packets, out_dir / "dwpose_body.pb.bin")
    combined_payload = out_dir / "dwpose.bin"
    combined_payload.write_bytes(pack_combined_payload(hand_packets, face_packets, body_packets))

    preview_dir = out_dir / "preview"
    preview_bytes = write_rgba_png_sequence(prev_combined, preview_dir)

    sample = frames_bgr[0]
    if (sample.shape[1], sample.shape[0]) != (1920, 1080):
        sample = cv2.resize(sample, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

    from demo.evaluation.pose_backends import _get_dwpose_wholebody

    model = _get_dwpose_wholebody()
    sample_wb = wb_frames[0]

    def extract_fn(frame: np.ndarray) -> FrameWholeBody:
        kpts, scores = model(frame)
        from demo.evaluation.pose_backends import people_from_rtm

        people = people_from_rtm(kpts, scores)
        return FrameWholeBody(frame_idx=0, people=people, width=frame.shape[1], height=frame.shape[0])

    def pack_fn(wb: FrameWholeBody) -> bytes:
        return pack_frame_parts(wb)[0]

    def decode_fn(blob: bytes) -> list[SingleHand]:
        return KeypointCompressor.decompress_frame(blob, sample_wb.width, sample_wb.height)

    stats = profile_map(extract_fn, sample, pack_fn=pack_fn, decode_fn=decode_fn, n_warmup=1, n_runs=3)

    streams = [
        _write_map_sidecar(
            map_name="dwpose",
            backend=backend,
            payload=combined_payload,
            preview=preview_dir,
            preview_bytes=preview_bytes,
            duration_s=duration_s,
            n_frames=n,
            fps=fps,
            stats=stats,
            extra={
                "topology": "coco_wb_133",
                "magic": "DWB2",
                "coding": "box-u8 exp-golomb delta; absent parts omitted",
                "smooth": "one_euro",
                "ort_device": dwpose_ort_device(),
                "parts": ["hands", "face", "body"],
                "n_kpts_hands": 21,
                "n_kpts_face": 68,
                "n_kpts_body": 17,
                "n_hands_detected": n_hands_detected,
                "n_face_detected": n_face_detected,
                "n_body_detected": n_body_detected,
                "n_blank_frames": n_blank,
                "overlay": "rgba",
            },
            out_dir=out_dir,
        ),
    ]
    write_sidecar(streams[0], out_dir / "sidecar.json")
    return streams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack DWPose whole-body into three native map streams.")
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        require("dwpose_pose")
        require("dwpose_det")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not args.clip.is_file():
        print(f"clip not found: {args.clip}", file=sys.stderr)
        return 2

    try:
        run_dwpose(args.clip, args.out, args.max_frames)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
