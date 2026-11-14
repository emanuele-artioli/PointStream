"""Complete native fallback endpoint. New PSC1 packets do not reinterpret PSTM/PSBL.

This first profile delivers AV1 frames, including ROI signalling when subsequently
qualified. Semantic enhancement profiles require their own measured design gate.
Decode returns planar 8-bit YUV420; no source or annotations are decoder inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import struct
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from src.codecs import svtav1

MAGIC = b"PSC1"
MAX_PACKET = 512 * 1024 * 1024
MAX_DECODED = 512 * 1024 * 1024
HEADER = struct.Struct(">4sI")
FIELDS = {"version", "profile", "width", "height", "fps", "frames", "full_range", "decoder_sha256"}


def validate(meta: dict[str, Any]) -> None:
    if set(meta) != FIELDS or (type(meta["version"]) is not int or meta["version"] != 1) or meta["profile"] != "native-av1":
        raise ValueError("unsupported packet profile or fields")
    for key in ("width", "height", "frames"):
        v = meta[key]
        if type(v) is not int or not 1 <= v <= (8192 if key != "frames" else 65535):
            raise ValueError("invalid geometry/frame count")
    if meta["width"] * meta["height"] * 3 // 2 * meta["frames"] > MAX_DECODED:
        raise ValueError("decoded window exceeds 512 MiB profile limit")
    if meta["width"] % 2 or meta["height"] % 2:
        raise ValueError("YUV420 requires even geometry")
    if type(meta["full_range"]) is not bool:
        raise ValueError("invalid colour range")
    fps = meta["fps"]
    if not isinstance(fps, list) or len(fps) != 2 or any(type(v) is not int or not 1 <= v <= 0xffffffff for v in fps):
        raise ValueError("invalid frame rate")
    rate = Fraction(*fps)
    if not Fraction(1, 1000) <= rate <= 1000 or [rate.numerator, rate.denominator] != fps:
        raise ValueError("frame rate must be reduced and supported")
    digest = meta["decoder_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid decoder identity")


def check_native(meta: dict[str, Any], native: bytes) -> None:
    if len(native) < 32:
        raise ValueError("truncated IVF")
    magic, version, header, codec, width, height, num, den, frames, unused = struct.unpack_from("<4sHH4sHHIIII", native)
    if (magic, version, header, codec, width, height, num, den, frames, unused) != (
            b"DKIF", 0, 32, b"AV01", meta["width"], meta["height"], *meta["fps"], meta["frames"], 0):
        raise ValueError("native IVF metadata differs from packet")
    offset = 32
    for index in range(frames):
        if offset + 12 > len(native):
            raise ValueError("truncated IVF frame")
        size, timestamp = struct.unpack_from("<IQ", native, offset)
        if size == 0 or timestamp != index or offset + 12 + size > len(native):
            raise ValueError("invalid IVF frame size/timestamp")
        offset += 12 + size
    if offset != len(native):
        raise ValueError("native IVF frame count differs from packet")


def pack(meta: dict[str, Any], native: bytes) -> bytes:
    validate(meta)
    check_native(meta, native)
    encoded = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    body = HEADER.pack(MAGIC, len(encoded)) + encoded + native
    if len(body) + 32 > MAX_PACKET:
        raise ValueError("packet too large")
    return body + hashlib.sha256(body).digest()


def unpack(data: bytes) -> tuple[dict[str, Any], bytes]:
    if not HEADER.size + 32 < len(data) <= MAX_PACKET:
        raise ValueError("invalid packet length")
    magic, length = HEADER.unpack_from(data)
    if magic != MAGIC or not 1 <= length <= 4096 or HEADER.size + length >= len(data) - 32:
        raise ValueError("invalid packet header")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("packet checksum mismatch")
    try:
        meta = json.loads(data[HEADER.size:HEADER.size + length])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid packet metadata") from exc
    if not isinstance(meta, dict):
        raise ValueError("invalid packet metadata")
    native = data[HEADER.size + length:-32]
    # Enforce canonical JSON as well as structural validation.
    if pack(meta, native) != data:
        raise ValueError("noncanonical packet")
    return meta, native


def read_packet(path: Path) -> bytes:
    if path.stat().st_size > MAX_PACKET:
        raise ValueError("packet too large")
    return path.read_bytes()


def write_new(path: Path, data: bytes) -> None:
    """Never overwrite a source, packet or saved result."""
    with path.open("xb") as handle:
        handle.write(data)


def inspect(data: bytes) -> dict[str, Any]:
    meta, native = unpack(data)
    duration = meta["frames"] / float(Fraction(*meta["fps"]))
    return {"metadata": meta, "packet_bytes": len(data), "native_bytes": len(native),
            "framing_bytes": len(data) - len(native), "duration_seconds": duration,
            "wire_kbps": len(data) * 8 / duration / 1000,
            "packet_sha256": hashlib.sha256(data).hexdigest(),
            "resources": {"decoder_sha256": meta["decoder_sha256"]},
            "timing": "encode/decode telemetry is external; no runtime inferred from packet"}


def probe_video(source: Path, meta: dict[str, Any], *, selected: bool, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Check real native geometry/colour and selected source PTS, never nominal FPS alone."""
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if selected:
        command += ["-read_intervals", f"%+#{meta['frames']}", "-show_frames"]
    command += ["-show_streams", "-show_entries",
                "stream=width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries:frame=best_effort_timestamp_time",
                "-of", "json", str(source)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"native probe failed: {exc.stderr[-1000:]}") from exc
    record = json.loads(result.stdout)
    stream = record["streams"][0]
    if (stream["width"], stream["height"]) != (meta["width"], meta["height"]):
        raise ValueError("coded/source geometry differs from packet")
    if selected:
        frames = record.get("frames", [])
        if len(frames) != meta["frames"]:
            raise ValueError("source frame count unavailable")
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
        step = 1 / float(Fraction(*meta["fps"]))
        if any(not math.isfinite(t) for t in timestamps) or any(abs((t - timestamps[0]) - i * step) > 2e-5 for i, t in enumerate(timestamps)):
            raise ValueError("variable or mismatched source cadence")
    else:
        if stream.get("pix_fmt") != "yuv420p" or stream.get("color_range") != ("pc" if meta["full_range"] else "tv"):
            raise ValueError("coded colour range/pixel format differs from packet")
    # This profile uses explicit BT.709 colour semantics. Unknown or HDR inputs
    # need a separately specified conversion rather than ffmpeg defaults.
    if any(stream.get(k) != "bt709" for k in ("color_space", "color_transfer", "color_primaries")):
        raise ValueError("profile requires explicit BT.709 colour metadata")
    return {"tool": svtav1.tool(ffprobe), "command": command, "result": record}


def encode(source: Path, output: Path, config: dict[str, Any], *, scratch: Path,
           encoder: str = "SvtAv1EncApp", decoder: str = "dav1d", ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Input is raw YUV420 or video, explicitly selected in configuration.

    Video must already have the declared geometry/cadence. ffmpeg does not
    resize or retime; conversion to the declared colour range is explicit.
    Native metadata and byte count are checked.
    """
    required = {"width", "height", "fps", "frames", "full_range", "crf", "preset", "input_format"}
    if set(config) != required or config["input_format"] not in ("video", "yuv420p"):
        raise ValueError("invalid encode configuration")
    if (type(config["crf"]) not in (int, float) or not math.isfinite(config["crf"]) or not 0 <= config["crf"] <= 70
            or type(config["preset"]) is not int or not 0 <= config["preset"] <= 13):
        raise ValueError("invalid encoder settings")
    if output.exists():
        raise FileExistsError(output)
    decoder_info = svtav1.tool(decoder)
    encoder_info = svtav1.tool(encoder)
    meta = {k: config[k] for k in ("width", "height", "fps", "frames", "full_range")}
    meta.update(version=1, profile="native-av1", decoder_sha256=decoder_info["sha256"])
    validate(meta)
    with tempfile.TemporaryDirectory(prefix="psc-encode-", dir=scratch) as temporary:
        work = Path(temporary)
        native_source = source
        conversion: dict[str, Any] = {}
        if config["input_format"] == "video":
            conversion["source_probe"] = probe_video(source, meta, selected=True, ffprobe=ffprobe)
            # Use y4m's self-describing header to verify actual geometry/cadence.
            y4m = work / "input.y4m"
            command = [ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-map", "0:v:0", "-an",
                       "-frames:v", str(meta["frames"]), "-vf",
                       "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=auto:out_range=" + ("full" if meta["full_range"] else "limited"),
                       "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(y4m)]
            conversion = {**conversion, "tool": svtav1.tool(ffmpeg), "command": command, **svtav1.run(command, 600)}
            with y4m.open("rb") as handle:
                tokens = handle.readline(4096).decode("ascii").split()
                if not all(v in tokens for v in (f'W{meta["width"]}', f'H{meta["height"]}', f'F{meta["fps"][0]}:{meta["fps"][1]}', "Ip")):
                    raise ValueError("video geometry/cadence/interlace differs from configuration")
                size = meta["width"] * meta["height"] * 3 // 2
                native_source = work / "input.yuv"
                with native_source.open("xb") as target:
                    for _ in range(meta["frames"]):
                        if handle.readline(4096).strip() != b"FRAME":
                            raise ValueError("missing/unsupported Y4M frame")
                        frame = handle.read(size)
                        if len(frame) != size:
                            raise ValueError("truncated Y4M")
                        target.write(frame)
                    if handle.read(1):
                        raise ValueError("unexpected Y4M frames")
        expected = meta["width"] * meta["height"] * 3 // 2 * meta["frames"]
        if native_source.stat().st_size != expected:
            raise ValueError("source size differs from configuration")
        stream = work / "stream.ivf"
        command = svtav1.encode_command(encoder, native_source, stream, width=meta["width"], height=meta["height"],
                                      fps=Fraction(*meta["fps"]), frames=meta["frames"], crf=config["crf"],
                                      preset=config["preset"], full_range=meta["full_range"],
                                      extra=["--color-primaries", "1", "--transfer-characteristics", "1", "--matrix-coefficients", "1"])
        timing = svtav1.run(command, 600)
        packet = pack(meta, stream.read_bytes())
        with source.open("rb") as source_handle:
            source_digest = hashlib.file_digest(source_handle, "sha256").hexdigest()
        write_new(output, packet)
        return {**inspect(packet), "operation": "encode", "encoder": encoder_info,
                "decoder": decoder_info, "command": command, "conversion": conversion, "seconds": timing,
                "config": config, "source_sha256": source_digest}


def decode(packet: Path, output: Path, *, scratch: Path, decoder: str = "dav1d", ffprobe: str = "ffprobe") -> dict[str, Any]:
    data = read_packet(packet)
    meta, native = unpack(data)
    if output.exists():
        raise FileExistsError(output)
    identity = svtav1.tool(decoder)
    if identity["sha256"] != meta["decoder_sha256"]:
        raise ValueError("installed decoder identity mismatch")
    with tempfile.TemporaryDirectory(prefix="psc-decode-", dir=scratch) as temporary:
        work = Path(temporary)
        stream, decoded = work / "stream.ivf", work / "decoded.yuv"
        stream.write_bytes(native)
        native_probe = probe_video(stream, meta, selected=False, ffprobe=ffprobe)
        command = svtav1.decode_command(decoder, stream, decoded, threads=1)
        command += ["--limit", str(meta["frames"]), "--sizelimit", str(meta["width"] * meta["height"])]
        timing = svtav1.run(command, 600)
        if decoded.stat().st_size != meta["width"] * meta["height"] * 3 // 2 * meta["frames"]:
            raise ValueError("native decoder output geometry/frame count mismatch")
        pixels = decoded.read_bytes()
        write_new(output, pixels)
        return {**inspect(data), "operation": "decode", "decoder": identity, "command": command,
                "seconds": timing, "native_probe": native_probe, "decoded_sha256": hashlib.sha256(pixels).hexdigest(),
                "isolation": "separate native process; OS source-denial not established by this entrypoint"}
