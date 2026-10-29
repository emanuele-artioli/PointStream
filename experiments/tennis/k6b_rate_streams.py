"""Small real CPU transports for saved inputs, with explicit framing and decoders."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import numpy as np

from src.segmentation.masks import compress, decompress, decode_rle, encode_rle

HEADER = struct.Struct(">4sII32s")


def seal(magic: bytes, metadata: dict[str, Any], body: bytes) -> bytes:
    meta = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    raw = struct.pack(">I", len(meta)) + meta + body
    coded = compress(raw)
    return HEADER.pack(magic, len(raw), len(coded), hashlib.sha256(raw).digest()) + coded


def unseal(blob: bytes, magic: bytes) -> tuple[dict[str, Any], bytes]:
    if len(blob) < HEADER.size:
        raise ValueError("Truncated transport header")
    tag, raw_size, coded_size, digest = HEADER.unpack_from(blob)
    if tag != magic or not 4 <= raw_size <= 512 * 1024**2 or len(blob) != HEADER.size + coded_size:
        raise ValueError("Transport identity/length mismatch")
    raw = decompress(blob[HEADER.size :])
    if len(raw) != raw_size or hashlib.sha256(raw).digest() != digest:
        raise ValueError("Transport integrity mismatch")
    n = struct.unpack_from(">I", raw)[0]
    if n > len(raw) - 4:
        raise ValueError("Truncated metadata")
    return json.loads(raw[4 : 4 + n]), raw[4 + n :]


def encode_keypoints(metadata: dict[str, Any], units: list[dict[str, Any]]) -> bytes:
    """q4 velocity/every4 samples plus EVERY-frame visibility, run timing and headers.

    No pooled symbol distribution or untransmitted code table is used. Confidence
    magnitudes are discarded only after geometry/map equivalence is checked by
    the caller: the native renderer uses the binary confident-joint predicate.
    """
    body = bytearray()
    entries = []
    for u in units:
        t, kp = np.asarray(u["t"], np.int64), np.asarray(u["kp"], np.float64)
        if not len(t) or kp.shape != (len(t), 17, 3) or np.any(np.diff(t) <= 0):
            raise ValueError("Invalid keypoint track")
        cuts = [0, *(np.flatnonzero(np.diff(t) != 1) + 1).tolist(), len(t)]
        runs = []
        for a, b in zip(cuts, cuts[1:]):
            samples = kp[a:b:4, :, :2] / 4
            integer = np.rint(samples).astype(np.int64)
            if not np.array_equal(integer, samples):
                raise ValueError("The frozen sent coordinates are not q4-grid values")
            prediction = integer.copy()
            if len(integer) > 1:
                prediction[1] = integer[1] - integer[0]
            if len(integer) > 2:
                prediction[2:] = integer[2:] - 2 * integer[1:-1] + integer[:-2]
            if np.any((prediction < -32768) | (prediction > 32767)):
                raise ValueError("Coordinate residual exceeds declared int16 format")
            values = prediction.astype("<i2").tobytes()
            visibility = np.packbits(kp[a:b, :, 2].reshape(-1) >= 0.5).tobytes()
            # KP_CONF is asserted by the native worker before this codec is used.
            body.extend(values)
            body.extend(visibility)
            runs.append([int(t[a]), b - a])
        entries.append({"name": u["name"], "runs": runs})
    return seal(
        b"KPS1",
        {
            **metadata,
            "units": entries,
            "q": 4,
            "every": 4,
            "predictor": "closed_loop_velocity",
            "visibility": "every_frame_threshold_0.5",
        },
        bytes(body),
    )


#: KPS2 codes exactly the geometry KPS1 sends (q4 samples every 4 frames, linear
#: in between, every-frame visibility) with W2's adaptive range coder instead of
#: raw int16 symbols (docs/experiments/w2-pose-rate.md). Decoded keypoints are
#: identical; only the bytes change.
KPS2_PROFILE: dict[str, Any] = {"q": 4, "key": 4, "predictor": "previous", "interp": "linear", "vis": "all"}


def encode_keypoints_kps2(metadata: dict[str, Any], units: list[dict[str, Any]]) -> bytes:
    """Range-coded keypoints; raises unless the decoder reproduces every input exactly.

    The inputs must already lie on KPS1's sent geometry (what ``decode_keypoints``
    returns for a KPS1 packet); anything else is rejected, never approximated.
    """
    from experiments.tennis.w2_pose_rate import Setting, encode_window

    coded = encode_window(
        [{"t": np.asarray(u["t"], np.int64), "kp": np.asarray(u["kp"], np.float64)} for u in units],
        Setting(q=4, key=4, predictor="previous", interp="linear", vis="all"),
    )
    for u, e in zip(units, coded.expected):
        kp = np.asarray(u["kp"], np.float64)
        if not (
            np.array_equal(np.asarray(u["t"], np.int64), e["t"])
            and np.array_equal(kp[:, :, :2], e["kp"][:, :, :2])
            and np.array_equal(kp[:, :, 2] >= 0.5, e["kp"][:, :, 2] >= 0.5)
        ):
            raise ValueError("Keypoints are not on the KPS1 sent geometry; KPS2 would change them")
    return seal(b"KPS2", {**metadata, "names": [u["name"] for u in units], "profile": KPS2_PROFILE}, coded.blob)


def transcode_kps1_to_kps2(blob: bytes) -> bytes:
    """Re-code a KPS1 packet as KPS2 and verify both decode to the same keypoints."""
    meta, units = decode_keypoints(blob)
    for key in ("units", "q", "every", "predictor", "visibility"):
        meta.pop(key, None)
    out = encode_keypoints_kps2(meta, units)
    _, again = decode_keypoints(out)
    for a, b in zip(units, again):
        if not (np.array_equal(a["t"], b["t"]) and np.array_equal(a["kp"], b["kp"])):
            raise ValueError("KPS2 transcoding changed the decoded keypoints")
    return out


def _decode_kps2(blob: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from experiments.tennis.w2_pose_rate import decode_window

    meta, raw = unseal(blob, b"KPS2")
    if meta.get("profile") != KPS2_PROFILE:
        raise ValueError("Unknown keypoint coding profile")
    decoded = decode_window(raw)
    if len(decoded) != len(meta["names"]):
        raise ValueError("Unit count does not match the stream")
    units = []
    for name, u in zip(meta["names"], decoded):
        if u["t"].size and (u["t"][0] < 0 or u["t"][-1] >= meta["native_frames"]):
            raise ValueError("Invalid frame run")
        units.append({"name": name, "t": u["t"], "kp": u["kp"]})
    return meta, units


def decode_keypoints(blob: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if blob[:4] == b"KPS2":
        return _decode_kps2(blob)
    meta, raw = unseal(blob, b"KPS1")
    offset = 0
    units = []
    if (meta["q"], meta["every"], meta["predictor"]) != (4, 4, "closed_loop_velocity"):
        raise ValueError("Unknown keypoint coding profile")
    for entry in meta["units"]:
        times: list[int] = []
        frames: list[np.ndarray] = []
        for start, n in entry["runs"]:
            if not 0 < n <= 240 or start < 0 or start + n > meta["native_frames"]:
                raise ValueError("Invalid frame run")
            ns = (n + 3) // 4
            count = ns * 17 * 2
            size = count * 2
            if offset + size > len(raw):
                raise ValueError("Truncated coordinate symbols")
            residual = (
                np.frombuffer(raw[offset : offset + size], "<i2")
                .astype(np.int64)
                .reshape(ns, 17, 2)
            )
            offset += size
            sent = residual.copy()
            if ns > 1:
                sent[1] += sent[0]
            for i in range(2, ns):
                sent[i] += 2 * sent[i - 1] - sent[i - 2]
            sent_xy = sent.astype(np.float64) * 4
            xy = np.empty((n, 17, 2), np.float64)
            for i in range(ns - 1):
                for j in range(4):
                    weight = j / 4
                    xy[4 * i + j] = (1 - weight) * sent_xy[i] + weight * sent_xy[i + 1]
            xy[4 * (ns - 1) :] = sent_xy[-1]
            size = (n * 17 + 7) // 8
            if offset + size > len(raw):
                raise ValueError("Truncated visibility symbols")
            vis = np.unpackbits(np.frombuffer(raw[offset : offset + size], np.uint8))[
                : n * 17
            ].reshape(n, 17)
            offset += size
            frames.extend(np.concatenate([xy, vis[:, :, None]], 2))
            times.extend(range(start, start + n))
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Overlapping/reordered frame runs")
        units.append(
            {
                "name": entry["name"],
                "t": np.asarray(times, np.int64),
                "kp": np.asarray(frames, np.float64),
            }
        )
    if offset != len(raw):
        raise ValueError("Trailing coordinate payload")
    return meta, units


def encode_masks(masks: np.ndarray, mode: str, factor: int = 1) -> bytes:
    masks = np.asarray(masks, bool)
    if masks.ndim != 4 or not all(masks.shape) or mode not in ("packed", "xor", "rle"):
        raise ValueError("Invalid matte stream")
    if mode == "rle":
        raw = json.dumps(
            [encode_rle(m) for frame in masks for m in frame], separators=(",", ":")
        ).encode()
    else:
        transformed = masks.copy()
        if mode == "xor":
            transformed[1:] ^= masks[:-1]
        raw = np.packbits(transformed.reshape(-1)).tobytes()
    return seal(
        b"MSK1",
        {
            "shape": list(masks.shape),
            "mode": mode,
            "native_grid_factor": factor,
            "output_shape": [1080, 1920],
            "timing": "shared_KPS1_window_header",
        },
        raw,
    )


def decode_masks(blob: bytes) -> tuple[dict[str, Any], np.ndarray]:
    meta, raw = unseal(blob, b"MSK1")
    shape = tuple(int(v) for v in meta["shape"])
    if len(shape) != 4 or not all(0 < v <= 1920 for v in shape) or np.prod(shape) > 300_000_000:
        raise ValueError("Invalid matte dimensions")
    if meta["mode"] == "rle":
        rows = json.loads(raw)
        if len(rows) != shape[0] * shape[1]:
            raise ValueError("Wrong matte count")
        masks = np.stack([decode_rle(row) for row in rows]).reshape(shape)
    elif meta["mode"] in ("packed", "xor"):
        n = int(np.prod(shape))
        if len(raw) != (n + 7) // 8:
            raise ValueError("Wrong packed matte length")
        masks = np.unpackbits(np.frombuffer(raw, np.uint8))[:n].reshape(shape).astype(bool)
        if meta["mode"] == "xor":
            np.bitwise_xor.accumulate(masks, axis=0, out=masks)
    else:
        raise ValueError("Unknown matte coding mode")
    return meta, masks


def encode_reference(
    image: np.ndarray, metadata: dict[str, Any], quality: int | None = None
) -> bytes:
    import cv2

    extension = ".png" if quality is None else ".jpg"
    parameters = (
        [cv2.IMWRITE_PNG_COMPRESSION, 6] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    ok, coded = cv2.imencode(extension, image[:, :, ::-1], parameters)
    if not ok:
        raise ValueError("Reference encoder failed")
    return seal(
        b"REF1",
        {**metadata, "codec": extension, "quality": quality, "color": "RGB8"},
        coded.tobytes(),
    )


def decode_reference(blob: bytes) -> tuple[dict[str, Any], np.ndarray]:
    import cv2

    meta, raw = unseal(blob, b"REF1")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Reference decoder failed")
    return meta, image[:, :, ::-1]
