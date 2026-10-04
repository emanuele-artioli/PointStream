"""Rate of a hand-pose packet when the packet covers a whole segment.

AV1 and the keypoint packet have to cover the same span. A ten-second hold-out
encoded as one video is one metadata packet, not one packet per hand. This
module only measures that packet: how many bytes each packing choice uses, and
how far the decoded joints move, in pixels. It does not score a generator.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand
from demo.pipeline.keypoint_compressor import KeypointCompressor

FPS = 30.0
FRAME_W = 1920
FRAME_H = 1080


def _clip_u(value: float, bits: int) -> int:
    limit = (1 << bits) - 1
    return int(np.clip(round(value), 0, limit))


def _quantize_hand(hand: dict, bits: int, frame_w: int, frame_h: int) -> dict:
    x1, y1, x2, y2 = [float(v) for v in hand["box"]]
    box = (
        _clip_u((x1 / frame_w) * 255.0, 8),
        _clip_u((y1 / frame_h) * 255.0, 8),
        _clip_u((x2 / frame_w) * 255.0, 8),
        _clip_u((y2 / frame_h) * 255.0, 8),
    )
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    joints = []
    for px, py in hand["landmarks_pixel"]:
        joints.append(
            (
                _clip_u(((float(px) - x1) / bw) * ((1 << bits) - 1), bits),
                _clip_u(((float(py) - y1) / bh) * ((1 << bits) - 1), bits),
            )
        )
    side = 1 if str(hand.get("side", "")).lower().startswith("r") else 0
    conf = _clip_u(float(hand.get("confidence", 1.0)) * 127.0, 7)
    return {"side": side, "conf": conf, "box": box, "joints": joints}


def _dequant_pixels(packed: dict, bits: int, frame_w: int, frame_h: int) -> np.ndarray:
    bx1, by1, bx2, by2 = packed["box"]
    x1 = (bx1 / 255.0) * frame_w
    y1 = (by1 / 255.0) * frame_h
    x2 = (bx2 / 255.0) * frame_w
    y2 = (by2 / 255.0) * frame_h
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    span = float((1 << bits) - 1)
    points = []
    for lx, ly in packed["joints"]:
        points.append([x1 + (lx / span) * bw, y1 + (ly / span) * bh])
    return np.asarray(points, dtype=np.float64)


def _pack_joints(joints: list[tuple[int, int]], bits: int) -> bytes:
    """Pack (x, y) pairs at ``bits`` each, low bits first."""
    width = bits * 2
    acc = 0
    filled = 0
    out = bytearray()
    for x, y in joints:
        acc |= (int(x) & ((1 << bits) - 1)) << filled
        filled += bits
        acc |= (int(y) & ((1 << bits) - 1)) << filled
        filled += bits
        while filled >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            filled -= 8
        _ = width
    if filled:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_joints(blob: bytes, bits: int, count: int = 21) -> list[tuple[int, int]]:
    acc = 0
    filled = 0
    index = 0
    joints = []
    mask = (1 << bits) - 1
    for _ in range(count):
        while filled < bits:
            acc |= blob[index] << filled
            index += 1
            filled += 8
        x = acc & mask
        acc >>= bits
        filled -= bits
        while filled < bits:
            acc |= blob[index] << filled
            index += 1
            filled += 8
        y = acc & mask
        acc >>= bits
        filled -= bits
        joints.append((x, y))
    return joints


def _hand_bytes(packed: dict, bits: int) -> bytes:
    body = bytearray()
    body.append((packed["side"] << 7) | (packed["conf"] & 0x7F))
    body.extend(packed["box"])
    body.extend(_pack_joints(packed["joints"], bits))
    return bytes(body)


def _per_frame_packet(frame_hands: list[dict], bits: int, frame_w: int, frame_h: int) -> bytes:
    chosen = sorted(frame_hands, key=lambda hand: (hand.get("side", ""), hand["box"][0]))[:2]
    packet = bytearray(b"PF")
    packet.append(bits)
    packet.append(len(chosen))
    for hand in chosen:
        packet.extend(_hand_bytes(_quantize_hand(hand, bits, frame_w, frame_h), bits))
    return bytes(packet)


def per_frame_bytes(frames: list[list[dict]], bits: int, frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> bytes:
    return b"".join(_per_frame_packet(hands, bits, frame_w, frame_h) for hands in frames)


def _fit_signed(delta: int, bits: int) -> int | None:
    limit = 1 << (bits - 1)
    if delta < -limit or delta >= limit:
        return None
    return delta + limit


def segment_delta_payload(
    frames: list[list[dict]],
    bits: int,
    *,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> bytes:
    """Uncompressed segment. A keyframe, then residuals of the quantized codes.

    The box is an 8-bit code, so its residual is signed 8-bit. Joint residuals
    use ``bits``. A hand whose residual does not fit is a new keyframe. Decoding
    this payload restores the quantized codes exactly.
    """
    raw = bytearray()
    raw.append(bits & 0xFF)
    raw.extend(struct.pack("<H", len(frames)))
    previous: dict[int, dict] = {}
    for hands in frames:
        chosen = sorted(hands, key=lambda hand: str(hand.get("side", "")))[:2]
        raw.append(len(chosen))
        seen = set()
        for hand in chosen:
            packed = _quantize_hand(hand, bits, frame_w, frame_h)
            side = packed["side"]
            seen.add(side)
            prior = previous.get(side)
            box_delta = None
            joint_delta = None
            if prior is not None:
                box_delta = [_fit_signed(b - p, 8) for b, p in zip(packed["box"], prior["box"])]
                joint_delta = [
                    (_fit_signed(x - px, bits), _fit_signed(y - py, bits))
                    for (x, y), (px, py) in zip(packed["joints"], prior["joints"])
                ]
                if any(v is None for v in box_delta) or any(v[0] is None or v[1] is None for v in joint_delta):
                    box_delta = None
                    joint_delta = None
            if box_delta is None:
                raw.append(0x80 | side)
                raw.extend(packed["box"])
                raw.extend(_pack_joints(packed["joints"], bits))
            else:
                raw.append(side)
                raw.extend(int(v) for v in box_delta)
                raw.extend(_pack_joints([(int(a), int(b)) for a, b in joint_delta], bits))
            previous[side] = packed
        for side in list(previous):
            if side not in seen:
                previous.pop(side)
    return bytes(raw)


def segment_delta_bytes(
    frames: list[list[dict]],
    bits: int,
    *,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> bytes:
    """One zlib packet for every frame in the segment."""
    return zlib.compress(segment_delta_payload(frames, bits, frame_w=frame_w, frame_h=frame_h), level=9)


def decode_segment_delta(packet: bytes, frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> list[list[np.ndarray]]:
    raw = zlib.decompress(packet)
    bits = raw[0]
    n_frames = struct.unpack_from("<H", raw, 1)[0]
    offset = 3
    previous: dict[int, dict] = {}
    decoded: list[list[np.ndarray]] = []
    for _ in range(n_frames):
        n_hands = raw[offset]
        offset += 1
        frame = []
        seen = set()
        for _hand_index in range(n_hands):
            flags = raw[offset]
            offset += 1
            side = flags & 0x01
            seen.add(side)
            if flags & 0x80:
                box = tuple(raw[offset : offset + 4])
                offset += 4
                joint_bytes = (21 * bits * 2 + 7) // 8
                joints = _unpack_joints(raw[offset : offset + joint_bytes], bits)
                offset += joint_bytes
                packed = {"side": side, "conf": 0, "box": box, "joints": joints}
            else:
                prior = previous[side]
                box_delta = raw[offset : offset + 4]
                offset += 4
                box = tuple((prior["box"][i] + int(box_delta[i]) - 128) & 0xFF for i in range(4))
                joint_bytes = (21 * bits * 2 + 7) // 8
                residual = _unpack_joints(raw[offset : offset + joint_bytes], bits)
                offset += joint_bytes
                limit = 1 << (bits - 1)
                span = (1 << bits) - 1
                joints = [
                    ((prior["joints"][i][0] + residual[i][0] - limit) & span, (prior["joints"][i][1] + residual[i][1] - limit) & span)
                    for i in range(21)
                ]
                packed = {"side": side, "conf": 0, "box": box, "joints": joints}
            previous[side] = packed
            frame.append(_dequant_pixels(packed, bits, frame_w, frame_h))
        for side in list(previous):
            if side not in seen:
                previous.pop(side)
        decoded.append(frame)
    return decoded


def _decode_per_frame(packet: bytes, frame_w: int, frame_h: int) -> list[np.ndarray]:
    if packet[:2] != b"PF":
        raise ValueError("not a per-frame packet")
    bits = packet[2]
    n_hands = packet[3]
    offset = 4
    out = []
    for _ in range(n_hands):
        side_conf = packet[offset]
        offset += 1
        box = tuple(packet[offset : offset + 4])
        offset += 4
        joint_bytes = (21 * bits * 2 + 7) // 8
        joints = _unpack_joints(packet[offset : offset + joint_bytes], bits)
        offset += joint_bytes
        out.append(
            _dequant_pixels(
                {"side": side_conf >> 7, "conf": side_conf & 0x7F, "box": box, "joints": joints},
                bits,
                frame_w,
                frame_h,
            )
        )
    return out


def landmark_error(
    frames: list[list[dict]],
    bits: int,
    *,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> float:
    """Mean joint error, in pixels, after quantizing each hand on its own."""
    errors = []
    for hands in frames:
        chosen = sorted(hands, key=lambda hand: (hand.get("side", ""), hand["box"][0]))[:2]
        decoded = _decode_per_frame(_per_frame_packet(chosen, bits, frame_w, frame_h), frame_w, frame_h)
        for hand, pred in zip(chosen, decoded):
            gt = np.asarray(hand["landmarks_pixel"], dtype=np.float64)[:, :2]
            errors.append(float(np.linalg.norm(pred - gt, axis=1).mean()))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def shipped_packet_bytes(frames: list[list[dict]], frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> int:
    """Bytes of the 8-bit compressor already used on the website, one packet per frame."""
    total = 0
    for index, hands in enumerate(frames):
        pose_hands = []
        for hand in sorted(hands, key=lambda item: (item.get("side", ""), item["box"][0]))[:2]:
            pose_hands.append(
                SingleHand(
                    handedness="Right" if str(hand.get("side", "")).lower().startswith("r") else "Left",
                    confidence=min(1.0, float(hand.get("confidence", 1.0))),
                    bbox=[int(v) for v in hand["box"]],
                    landmarks_norm=hand.get("landmarks_norm") or [[0, 0, 0]] * 21,
                    landmarks_pixel=hand["landmarks_pixel"],
                )
            )
        packet = KeypointCompressor.compress_frame(
            FrameHandPose(frame_idx=index, hands=pose_hands),
            frame_w=frame_w,
            frame_h=frame_h,
        )
        total += len(packet)
    return total


def _windows(frames: list[list[dict]], length: int) -> list[list[list[dict]]]:
    return [frames[start : start + length] for start in range(0, len(frames), length)]


def score_tracks(
    frames: list[list[dict]],
    *,
    segment_lengths: tuple[int, ...] = (1, 30, 300),
    fps: float = FPS,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> list[dict]:
    """Bytes and kbps for each technique at each segment length.

    ``segment_lengths`` are in frames. 300 frames is ten seconds at 30 fps.
    The per-frame techniques do not get smaller when the segment grows; the
    delta packet does, because one zlib stream covers the whole segment.
    """
    rows = []
    duration = len(frames) / fps if frames else 0.0
    for length in segment_lengths:
        windows = _windows(frames, length)
        shipped = sum(shipped_packet_bytes(window, frame_w, frame_h) for window in windows)
        rows.append(_row("shipped_per_frame_u8", length, shipped, frames, 8, duration, frame_w, frame_h))
        for bits in (8, 6, 4):
            raw = sum(len(per_frame_bytes(window, bits, frame_w, frame_h)) for window in windows)
            rows.append(_row(f"per_frame_u{bits}", length, raw, frames, bits, duration, frame_w, frame_h))
            packed = sum(len(segment_delta_bytes(window, bits, frame_w=frame_w, frame_h=frame_h)) for window in windows)
            rows.append(_row(f"segment_delta_u{bits}_zlib", length, packed, frames, bits, duration, frame_w, frame_h))
    return rows


def _row(name: str, length: int, nbytes: int, frames: list[list[dict]], bits: int, duration: float, frame_w: int, frame_h: int) -> dict:
    kbps = (nbytes * 8) / (duration * 1000.0) if duration else 0.0
    return {
        "technique": name,
        "segment_frames": length,
        "bytes": nbytes,
        "kbps": kbps,
        "mean_joint_px": landmark_error(frames, bits, frame_w=frame_w, frame_h=frame_h) if "u4" in name or "u6" in name or "u8" in name else None,
    }


def load_selected(path: Path) -> list[list[dict]]:
    """One entry per frame that the gallery kept, in frame order."""
    rows = json.loads(path.read_text())
    rows = sorted(rows, key=lambda row: int(row["frame_idx"]))
    frames = []
    for row in rows:
        if row.get("aisle") or row.get("look"):
            continue
        chosen = [hand for hand in row.get("hands") or [] if hand.get("selected")]
        frames.append(chosen)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=Path, action="append", required=True)
    parser.add_argument("--segments", default="1,30,300")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lengths = tuple(int(part) for part in args.segments.split(",") if part)
    report = {"schema": "pointstream.hand_packet_rate.v1", "clips": []}
    for path in args.pose:
        frames = load_selected(path)
        report["clips"].append(
            {
                "pose": str(path),
                "frames": len(frames),
                "hands": sum(len(hands) for hands in frames),
                "rows": score_tracks(frames, segment_lengths=lengths),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
