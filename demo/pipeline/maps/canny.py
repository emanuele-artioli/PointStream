"""OpenCV Canny edge map for the maps gallery.

Canny runs on luma (OpenCV Rec.601 gray). The gallery writes one recipe at
the AV1 ladder resolutions (180 / 240 / 360 / 540 / 720 / 1080): thresholds
100/200, pre-blur 0.8, Canny after area-downscale, then keep pixels within
2px of the YOLOE contour and the DW-Pose strokes. The default ``canny`` rung
is 240p.

Native payload (``payload.bin``)
--------------------------------
v3: encode each plane several ways and keep the smallest after one zstd/zlib:
packbits stacked, packbits XOR-delta, sparse set-bit varints, and a lossless
Freeman chain of the set pixels. Factory Canny flickers, so XOR often loses;
downscale + fewer edges is the real RD lever. The chain is only kept when it
round-trips the mask and is smaller. v2 is packbits+XOR volume. v1
independently zstd'd each frame.

This is **not** a color overlay of edges on RGB, and **not** H.264/AV1.

::

    Offset 0:  magic     4 bytes   b"CNNY"
           4:  version   uint32    3
           8:  n_frames  uint32
          12:  height    uint32
          16:  width     uint32
          20:  flags     uint32    bit0 = XOR delta, bit1 = sparse varint, bit2 = chain
          24:  zstd/zlib of packed planes

Preview is an RGBA PNG sequence (white edges, transparent ground) under
``preview/``. It is never counted as payload.

CLI::

    PYTHONPATH=. python -m demo.pipeline.maps.canny --clip PATH \\
        --out demo/outputs/maps/<clip_stem>/canny/ [--lo 50 --hi 150] [--max-frames N]
"""

from __future__ import annotations

import argparse
import struct
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from demo.evaluation.profile_map import profile_map
from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.encode import pack_binary_mask, unpack_binary_mask
from demo.pipeline.maps.rgba_preview import write_rgba_png_sequence

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None
    import zlib
else:
    import zlib

HYSTERESIS_PAIRS: tuple[tuple[int, int], ...] = ((50, 150), (100, 200), (150, 250))
DEFAULT_LO, DEFAULT_HI = HYSTERESIS_PAIRS[0]

# One recipe, three resolutions. Thresholds 100/200, pre-blur 0.8, Canny on
# the downscaled luma, then keep pixels within GATE_RADIUS of the YOLOE
# contour and DW-Pose strokes. That is the map that stays under AV1 CRF 63.
TUNED_LO, TUNED_HI = 100, 200
TUNED_BLUR = 0.8
GATE_RADIUS = 2
CANNY_RUNGS: tuple[tuple[str, int, int, int, float, bool], ...] = (
    ("canny_180", 180, TUNED_LO, TUNED_HI, TUNED_BLUR, True),
    ("canny", 240, TUNED_LO, TUNED_HI, TUNED_BLUR, True),
    ("canny_360", 360, TUNED_LO, TUNED_HI, TUNED_BLUR, True),
    ("canny_540", 540, TUNED_LO, TUNED_HI, TUNED_BLUR, True),
    ("canny_720", 720, TUNED_LO, TUNED_HI, TUNED_BLUR, True),
    ("canny_1080", 1080, TUNED_LO, TUNED_HI, TUNED_BLUR, True),
)

PAYLOAD_NAME = "payload.bin"
PREVIEW_DIRNAME = "preview"
SIDECAR_NAME = "sidecar.json"

PAYLOAD_MAGIC = b"CNNY"
PAYLOAD_VERSION = 3
_HEADER_V1 = struct.Struct("<4sIIII")  # magic, version, n_frames, height, width
_HEADER_V2 = struct.Struct("<4sIIIII")  # + flags (v2 and v3)
_LEN = struct.Struct("<I")
FLAG_XOR_DELTA = 1
FLAG_SPARSE = 2
FLAG_CHAIN = 4

PAYLOAD_FORMAT = "cnny-v3-min(packbits,xor,sparse,chain)+zstd"

# Freeman 8-connected steps: E, SE, S, SW, W, NW, N, NE.
_FREEMAN_DX = (1, 1, 0, -1, -1, -1, 0, 1)
_FREEMAN_DY = (0, 1, 1, 1, 0, -1, -1, -1)
_CHAIN_HEAD = struct.Struct("<HHI")  # x, y, n_steps


def frame_to_luma(frame: np.ndarray) -> np.ndarray:
    """HxW uint8 luma. 3-channel frames are treated as OpenCV BGR."""
    arr = np.ascontiguousarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"expected HxW or HxWxC, got shape {arr.shape}")
    channels = arr.shape[2]
    if channels == 1:
        return np.ascontiguousarray(arr[:, :, 0])
    if channels == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported channel count {channels}")


def extract_canny_frame(frame: np.ndarray, lo: int = DEFAULT_LO, hi: int = DEFAULT_HI) -> np.ndarray:
    """Canny on one frame. Returns HxW uint8 {0,1} (not a color overlay)."""
    luma = frame_to_luma(frame)
    edges = cv2.Canny(luma, int(lo), int(hi))
    return (edges > 0).astype(np.uint8)


def extract_canny_frames(
    frames: Sequence[np.ndarray],
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
) -> list[np.ndarray]:
    """Canny on a list of ndarray frames. No ffmpeg required."""
    return [extract_canny_frame(frame, lo=lo, hi=hi) for frame in frames]


def _as_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"canny payload must be HxW binary, got shape {arr.shape}")
    if arr.shape != (height, width):
        raise ValueError(f"mask shape {arr.shape} != {(height, width)}")
    return (arr > 0).astype(np.uint8)


def _packed_frame_nbytes(height: int, width: int) -> int:
    return (int(height) * int(width) + 7) // 8


def _compress_volume(raw: bytes, level: int = 19) -> bytes:
    if zstd is not None:
        return zstd.ZstdCompressor(level=int(level)).compress(raw)
    zlib_level = 9 if int(level) >= 10 else max(1, min(9, int(level)))
    return zlib.compress(raw, level=zlib_level)


def _decompress_volume(blob: bytes) -> bytes:
    if blob.startswith(b"\x28\xb5\x2f\xfd") and zstd is not None:
        return zstd.ZstdDecompressor().decompress(blob)
    return zlib.decompress(blob)


def _uleb128(n: int) -> bytes:
    if n < 0:
        raise ValueError("uleb128 requires a non-negative integer")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _read_uleb128(buf: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while offset < len(buf):
        byte = buf[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("uleb128 overflow")
    raise ValueError("truncated uleb128")


def _encode_sparse_plane(mask: np.ndarray) -> bytes:
    idx = np.flatnonzero(mask.ravel())
    blob = bytearray(_LEN.pack(int(idx.size)))
    prev = 0
    for pos in idx.tolist():
        blob.extend(_uleb128(int(pos) - prev))
        prev = int(pos)
    return bytes(blob)


def _decode_sparse_plane(buf: bytes, offset: int, height: int, width: int) -> tuple[np.ndarray, int]:
    if offset + _LEN.size > len(buf):
        raise ValueError("truncated sparse canny count")
    (n_set,) = _LEN.unpack_from(buf, offset)
    offset += _LEN.size
    plane = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    for _ in range(int(n_set)):
        delta, offset = _read_uleb128(buf, offset)
        pos += int(delta)
        if pos >= plane.size:
            raise ValueError("sparse canny index out of range")
        plane[pos] = 1
    return plane.reshape(height, width), offset


def scale_hw(src_h: int, src_w: int, target_h: int) -> tuple[int, int]:
    """Match ``target_h`` without upscaling past the source."""
    height = max(1, min(int(src_h), int(target_h)))
    width = max(1, int(round(int(src_w) * height / float(src_h))))
    return height, width


def resize_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = (np.asarray(mask) > 0).astype(np.uint8)
    if arr.shape == (height, width):
        return arr
    interp = cv2.INTER_AREA if (height < arr.shape[0] or width < arr.shape[1]) else cv2.INTER_NEAREST
    scaled = cv2.resize(arr * 255, (width, height), interpolation=interp)
    return (scaled > 0).astype(np.uint8)


def codec_name(flags: int) -> str:
    """Human name for a v3 payload flags word."""
    if flags & FLAG_CHAIN:
        return "chain"
    if flags & FLAG_SPARSE:
        return "sparse"
    if flags & FLAG_XOR_DELTA:
        return "xor"
    return "packbits"


def _trace_chains(mask: np.ndarray) -> list[tuple[int, int, bytes]]:
    """Cover every set pixel with 8-connected Freeman chains.

    Greedy continuation prefers the previous heading so a long curve stays one
    chain. The walk is lossless: every foreground pixel is a start or a step.
    """
    height, width = int(mask.shape[0]), int(mask.shape[1])
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    remaining = {int(y) * width + int(x) for y, x in zip(ys.tolist(), xs.tolist())}
    order = sorted(remaining)
    chains: list[tuple[int, int, bytes]] = []
    heading_offsets = (0, 1, 7, 2, 6, 3, 5, 4)
    for start in order:
        if start not in remaining:
            continue
        remaining.remove(start)
        y, x = divmod(start, width)
        steps = bytearray()
        heading = 0
        while True:
            nxt = None
            chosen = 0
            for offset in heading_offsets:
                direction = (heading + offset) & 7
                ny = y + _FREEMAN_DY[direction]
                nx = x + _FREEMAN_DX[direction]
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                flat = ny * width + nx
                if flat in remaining:
                    nxt = flat
                    chosen = direction
                    break
            if nxt is None:
                break
            remaining.remove(nxt)
            steps.append(chosen)
            y, x = divmod(nxt, width)
            heading = chosen
        chains.append((start % width, start // width, bytes(steps)))
    return chains


def _raster_chains(chains: Sequence[tuple[int, int, bytes]], height: int, width: int) -> np.ndarray:
    plane = np.zeros((height, width), dtype=np.uint8)
    for x, y, steps in chains:
        if y < 0 or x < 0 or y >= height or x >= width:
            raise ValueError("chain start out of range")
        plane[y, x] = 1
        cy, cx = int(y), int(x)
        for raw in steps:
            direction = int(raw)
            if direction > 7:
                raise ValueError(f"bad freeman code {direction}")
            cy += _FREEMAN_DY[direction]
            cx += _FREEMAN_DX[direction]
            if cy < 0 or cx < 0 or cy >= height or cx >= width:
                raise ValueError("chain step left the frame")
            plane[cy, cx] = 1
    return plane


def _encode_chain_plane(mask: np.ndarray) -> bytes:
    chains = _trace_chains(mask)
    blob = bytearray(_LEN.pack(len(chains)))
    for x, y, steps in chains:
        blob.extend(_CHAIN_HEAD.pack(int(x), int(y), len(steps)))
        blob.extend(steps)
    return bytes(blob)


def _decode_chain_plane(buf: bytes, offset: int, height: int, width: int) -> tuple[np.ndarray, int]:
    if offset + _LEN.size > len(buf):
        raise ValueError("truncated chain count")
    (n_chains,) = _LEN.unpack_from(buf, offset)
    offset += _LEN.size
    chains: list[tuple[int, int, bytes]] = []
    for _ in range(int(n_chains)):
        if offset + _CHAIN_HEAD.size > len(buf):
            raise ValueError("truncated chain header")
        x, y, n_steps = _CHAIN_HEAD.unpack_from(buf, offset)
        offset += _CHAIN_HEAD.size
        end = offset + int(n_steps)
        if end > len(buf):
            raise ValueError("truncated chain steps")
        chains.append((int(x), int(y), buf[offset:end]))
        offset = end
    return _raster_chains(chains, height, width), offset


def pack_canny_payload(
    masks: Sequence[np.ndarray],
    *,
    level: int = 19,
    include_chain: bool = True,
) -> bytes:
    """Pack 1-bit frames; keep the smallest lossless coding after one zstd."""
    if not masks:
        raise ValueError("no canny masks to pack")
    first = np.asarray(masks[0])
    if first.ndim != 2:
        raise ValueError(f"canny payload must be HxW binary, got shape {first.shape}")
    height, width = int(first.shape[0]), int(first.shape[1])
    planes: list[np.ndarray] = []
    stacked: list[bytes] = []
    xored: list[bytes] = []
    sparse: list[bytes] = []
    prev: np.ndarray | None = None
    for mask in masks:
        cur = _as_binary_mask(mask, height, width)
        planes.append(cur)
        packed = np.packbits(cur.ravel()).tobytes()
        stacked.append(packed)
        delta = cur if prev is None else np.bitwise_xor(cur, prev)
        xored.append(np.packbits(delta.ravel()).tobytes())
        sparse.append(_encode_sparse_plane(cur))
        prev = cur
    candidates: list[tuple[bytes, int]] = [
        (_compress_volume(b"".join(stacked), level=level), 0),
        (_compress_volume(b"".join(xored), level=level), FLAG_XOR_DELTA),
        (_compress_volume(b"".join(sparse), level=level), FLAG_SPARSE),
    ]
    if include_chain:
        chained = b"".join(_encode_chain_plane(plane) for plane in planes)
        candidates.append((_compress_volume(chained, level=level), FLAG_CHAIN))
    volume, flags = min(candidates, key=lambda item: len(item[0]))
    return _HEADER_V2.pack(
        PAYLOAD_MAGIC, PAYLOAD_VERSION, len(masks), height, width, flags
    ) + volume


def unpack_canny_payload(payload: bytes) -> list[np.ndarray]:
    """Inverse of ``pack_canny_payload`` (v1 per-frame zstd, v2 packbits, v3 sparse)."""
    if len(payload) < _HEADER_V1.size:
        raise ValueError("canny payload shorter than header")
    magic, version, n_frames, height, width = _HEADER_V1.unpack_from(payload, 0)
    if magic != PAYLOAD_MAGIC:
        raise ValueError(f"bad canny magic {magic!r}")
    if version == 1:
        offset = _HEADER_V1.size
        frames: list[np.ndarray] = []
        for _ in range(int(n_frames)):
            if offset + _LEN.size > len(payload):
                raise ValueError("truncated canny blob length")
            (blob_len,) = _LEN.unpack_from(payload, offset)
            offset += _LEN.size
            blob = payload[offset : offset + blob_len]
            if len(blob) != blob_len:
                raise ValueError("truncated canny blob")
            offset += blob_len
            frames.append(unpack_binary_mask(blob, int(height), int(width)))
        return frames
    if version not in {2, 3}:
        raise ValueError(f"unsupported canny payload version {version}")
    _magic, _version, n_frames, height, width, flags = _HEADER_V2.unpack_from(payload, 0)
    raw = _decompress_volume(payload[_HEADER_V2.size :])
    if flags & FLAG_CHAIN:
        frames = []
        offset = 0
        for _ in range(int(n_frames)):
            plane, offset = _decode_chain_plane(raw, offset, int(height), int(width))
            frames.append(plane)
        return frames
    if flags & FLAG_SPARSE:
        frames = []
        offset = 0
        for _ in range(int(n_frames)):
            plane, offset = _decode_sparse_plane(raw, offset, int(height), int(width))
            frames.append(plane)
        return frames
    stride = _packed_frame_nbytes(height, width)
    expected = stride * int(n_frames)
    if len(raw) < expected:
        raise ValueError(f"truncated canny volume ({len(raw)} < {expected})")
    xor_delta = bool(flags & FLAG_XOR_DELTA)
    frames = []
    prev = None
    for i in range(int(n_frames)):
        bits = np.frombuffer(raw[i * stride : (i + 1) * stride], dtype=np.uint8)
        plane = np.unpackbits(bits)[: height * width].reshape(height, width).astype(np.uint8)
        if xor_delta and prev is not None:
            plane = np.bitwise_xor(plane, prev)
        frames.append(plane)
        prev = plane
    return frames


def write_preview_pngs(masks: Sequence[np.ndarray], preview_dir: Path) -> int:
    """RGBA white-on-transparent PNG sequence. Returns total bytes written."""
    vis = []
    for mask in masks:
        plane = (np.asarray(mask) > 0).astype(np.uint8) * 255
        bgra = np.zeros((*plane.shape, 4), dtype=np.uint8)
        bgra[:, :, :3] = plane[:, :, None]
        bgra[:, :, 3] = np.where(plane > 0, 220, 0).astype(np.uint8)
        vis.append(bgra)
    return write_rgba_png_sequence(vis, preview_dir)


def write_canny_map(
    frames: Sequence[np.ndarray],
    out_dir: Path | str,
    *,
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
    fps: float = 30.0,
    n_warmup: int = 2,
    n_runs: int = 8,
    target_height: int | None = None,
    map_name: str = "canny",
    preview_height: int | None = None,
    blur_sigma: float = 0.0,
    resize_first: bool = False,
    filled_masks: Sequence[np.ndarray] | None = None,
    stroke_masks: Sequence[np.ndarray] | None = None,
    gate_radius: int = GATE_RADIUS,
) -> MapStream:
    """Extract, pack, preview, and write sidecar for an in-memory clip."""
    if not frames:
        raise ValueError("no frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src_h, src_w = int(frames[0].shape[0]), int(frames[0].shape[1])
    if target_height is None:
        pack_h, pack_w = src_h, src_w
    else:
        pack_h, pack_w = scale_hw(src_h, src_w, int(target_height))
    if resize_first or float(blur_sigma) > 0:
        from demo.pipeline.maps.canny_quality import extract_tuned_frame

        masks = [
            extract_tuned_frame(
                frame,
                lo=lo,
                hi=hi,
                target_height=pack_h,
                blur_sigma=blur_sigma,
                fullres_then_down=not resize_first,
            )
            for frame in frames
        ]
        masks_full = masks
    else:
        masks_full = extract_canny_frames(frames, lo=lo, hi=hi)
        masks = [resize_binary_mask(mask, pack_h, pack_w) for mask in masks_full]
    if filled_masks is not None or stroke_masks is not None:
        if filled_masks is None or stroke_masks is None:
            raise ValueError("task gate needs both filled masks and pose strokes")
        if len(filled_masks) != len(masks) or len(stroke_masks) != len(masks):
            raise ValueError("task-gate masks must match the frame count")
        from demo.pipeline.maps.canny_quality import gate_to_task_boundary

        masks = [
            gate_to_task_boundary(edge, filled, strokes, radius=int(gate_radius))
            for edge, filled, strokes in zip(masks, filled_masks, stroke_masks)
        ]
        masks_full = masks
    payload = pack_canny_payload(masks)
    payload_path = out / PAYLOAD_NAME
    payload_path.write_bytes(payload)

    vis_h = int(preview_height) if preview_height is not None else src_h
    vis_w = max(1, int(round(src_w * vis_h / float(src_h)))) if vis_h != src_h else src_w
    preview_masks = [resize_binary_mask(mask, vis_h, vis_w) for mask in masks_full]
    preview_dir = out / PREVIEW_DIRNAME
    preview_bytes = write_preview_pngs(preview_masks, preview_dir)

    def extract_fn(frame: np.ndarray) -> np.ndarray:
        if resize_first or float(blur_sigma) > 0:
            from demo.pipeline.maps.canny_quality import extract_tuned_frame

            return extract_tuned_frame(
                frame,
                lo=lo,
                hi=hi,
                target_height=pack_h,
                blur_sigma=blur_sigma,
                fullres_then_down=not resize_first,
            )
        full = extract_canny_frame(frame, lo=lo, hi=hi)
        return resize_binary_mask(full, pack_h, pack_w)

    stats = profile_map(
        extract_fn,
        np.asarray(frames[0]),
        pack_fn=pack_binary_mask,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )

    n_frames = len(masks)
    duration_s = n_frames / float(fps)
    stream = MapStream(
        map=map_name,
        backend=(
            f"opencv-canny-{int(lo)}-{int(hi)}"
            + (f"-blur{float(blur_sigma):.1f}" if float(blur_sigma) > 0 else "")
            + ("-resize" if resize_first else "")
            + f"@{pack_w}x{pack_h}"
        ),
        payload_path=str(payload_path),
        payload_bytes=payload_path.stat().st_size,
        preview_path=str(preview_dir),
        preview_bytes=int(preview_bytes),
        duration_s=duration_s,
        n_frames=n_frames,
        fps=float(fps),
        extract_ms_p50=float(stats["extract_ms_p50"]),
        extract_ms_p95=float(stats["extract_ms_p95"]),
        pack_ms_p50=float(stats["pack_ms_p50"]),
        codec_ms_p50=float(stats["codec_ms_p50"]),
        decode_ms_p50=float(stats["decode_ms_p50"]),
        gpu=str(stats["gpu"]),
        kind="native",
        extra={
            "lo": int(lo),
            "hi": int(hi),
            "blur_sigma": float(blur_sigma),
            "resize_first": bool(resize_first),
            "gate_radius": int(gate_radius) if filled_masks is not None else 0,
            "payload_format": PAYLOAD_FORMAT,
            "pack_height": pack_h,
            "pack_width": pack_w,
            "src_height": src_h,
            "src_width": src_w,
            "overlay": "rgba",
        },
    )
    write_sidecar(stream, out / SIDECAR_NAME)
    return stream


def _probe_fps(clip: Path) -> float:
    cap = cv2.VideoCapture(str(clip))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 1e-6 else 30.0


def write_canny_ladder(
    frames: Sequence[np.ndarray],
    canny_dir: Path | str,
    *,
    fps: float = 30.0,
    n_warmup: int = 2,
    n_runs: int = 8,
    filled_masks: Sequence[np.ndarray] | None = None,
    stroke_masks: Sequence[np.ndarray] | None = None,
    gate_radius: int = GATE_RADIUS,
    preview_height: int | None = None,
) -> list[MapStream]:
    """Write the Canny RD ladder under ``canny_dir`` (clip/canny)."""
    if not frames:
        raise ValueError("no frames")
    root = Path(canny_dir)
    streams: list[MapStream] = []
    for map_name, target_h, lo, hi, blur_sigma, resize_first in CANNY_RUNGS:
        dest = root if map_name == "canny" else root / "rungs" / map_name
        streams.append(
            write_canny_map(
                frames,
                dest,
                lo=lo,
                hi=hi,
                fps=fps,
                n_warmup=n_warmup,
                n_runs=n_runs,
                target_height=target_h,
                map_name=map_name,
                blur_sigma=blur_sigma,
                resize_first=resize_first,
                filled_masks=filled_masks,
                stroke_masks=stroke_masks,
                gate_radius=gate_radius,
                preview_height=preview_height,
            )
        )
    return streams


def run_canny_clip(
    clip: Path | str,
    out_dir: Path | str,
    *,
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
    max_frames: int | None = None,
    ladder: bool = True,
) -> MapStream | list[MapStream]:
    from demo.pipeline.background_codec import read_video_frames_robust

    clip_path = Path(clip)
    frames = read_video_frames_robust(clip_path, max_frames=max_frames)
    if not frames:
        raise ValueError(f"no frames decoded from {clip_path}")
    fps = _probe_fps(clip_path)
    if ladder:
        parent = Path(out_dir)
        # ``run_maps`` passes clip/canny — rungs live under that tree so collect() sees them.
        return write_canny_ladder(frames, parent, fps=fps)
    return write_canny_map(frames, out_dir, lo=lo, hi=hi, fps=fps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract OpenCV Canny as a native 1-bit packed maps-gallery stream."
    )
    parser.add_argument("--clip", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (payload.bin, preview/, sidecar.json)",
    )
    parser.add_argument("--lo", type=int, default=DEFAULT_LO, help="Canny low threshold (default 50)")
    parser.add_argument("--hi", type=int, default=DEFAULT_HI, help="Canny high threshold (default 150)")
    parser.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    args = parser.parse_args(argv)
    run_canny_clip(args.clip, args.out, lo=args.lo, hi=args.hi, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
