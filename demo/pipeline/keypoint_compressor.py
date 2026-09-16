"""Keypoint compressor: quantizes and bitpacks MediaPipe hand keypoints into ultra-low-bitrate wire format."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand


class KeypointCompressor:
    """Serializes 21-joint hand pose into ~47 bytes per hand."""

    MAGIC = b"PK"  # PointStream Keypoints

    @staticmethod
    def compress_frame(pose: FrameHandPose, frame_w: int = 1920, frame_h: int = 1080) -> bytes:
        num_hands = min(len(pose.hands), 2)
        packet = bytearray()
        packet.extend(KeypointCompressor.MAGIC)
        packet.append(num_hands)

        for hand in pose.hands[:num_hands]:
            # Handedness and confidence
            is_right = 1 if hand.handedness.lower() == "right" else 0
            conf_quant = min(127, int(hand.confidence * 127))
            flags = (is_right << 7) | (conf_quant & 0x7F)
            packet.append(flags)

            # Bounding box normalized to [0, 255]
            x1, y1, x2, y2 = hand.bbox
            bx1 = int(np.clip(round((x1 / frame_w) * 255), 0, 255))
            by1 = int(np.clip(round((y1 / frame_h) * 255), 0, 255))
            bx2 = int(np.clip(round((x2 / frame_w) * 255), 0, 255))
            by2 = int(np.clip(round((y2 / frame_h) * 255), 0, 255))
            packet.extend(struct.pack("4B", bx1, by1, bx2, by2))

            # 21 landmarks normalized relative to bbox [0, 255]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            for lm in hand.landmarks_pixel:
                lx = int(np.clip(round(((lm[0] - x1) / bw) * 255), 0, 255))
                ly = int(np.clip(round(((lm[1] - y1) / bh) * 255), 0, 255))
                packet.extend(struct.pack("2B", lx, ly))

        return bytes(packet)

    @staticmethod
    def decompress_frame(data: bytes, frame_w: int = 1920, frame_h: int = 1080) -> list[SingleHand]:
        if len(data) < 3 or data[:2] != KeypointCompressor.MAGIC:
            return []

        num_hands = data[2]
        offset = 3
        hands: list[SingleHand] = []

        for _ in range(num_hands):
            if offset + 47 > len(data):
                break
            flags = data[offset]
            is_right = (flags >> 7) & 1
            handedness = "Right" if is_right else "Left"
            conf = float(flags & 0x7F) / 127.0
            offset += 1

            bx1, by1, bx2, by2 = struct.unpack_from("4B", data, offset)
            offset += 4
            x1 = int((bx1 / 255.0) * frame_w)
            y1 = int((by1 / 255.0) * frame_h)
            x2 = int((bx2 / 255.0) * frame_w)
            y2 = int((by2 / 255.0) * frame_h)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            landmarks_pixel: list[list[float]] = []
            landmarks_norm: list[list[float]] = []
            for _ in range(21):
                lx, ly = struct.unpack_from("2B", data, offset)
                offset += 2
                px = x1 + (lx / 255.0) * bw
                py = y1 + (ly / 255.0) * bh
                landmarks_pixel.append([px, py])
                landmarks_norm.append([px / frame_w, py / frame_h, 0.0])

            hands.append(
                SingleHand(
                    handedness=handedness,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    landmarks_norm=landmarks_norm,
                    landmarks_pixel=landmarks_pixel,
                )
            )

        return hands

    @staticmethod
    def calculate_stream_bitrate(packets: list[bytes], fps: float = 30.0) -> float:
        """Returns bitrate in kbps."""
        if not packets:
            return 0.0
        total_bytes = sum(len(p) for p in packets)
        duration_sec = len(packets) / fps
        return (total_bytes * 8) / (duration_sec * 1000.0)
