"""Exercise existing native and ball packet boundaries on the shortest and longest authenticated points."""

import argparse
from fractions import Fraction
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from src.codecs import svtav1
from src.pointstream import ball, ball_stream, codec, tracked_patch


HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def raw_hashes(path, count, frame_bytes):
    with path.open("rb") as file:
        rows = []
        for _ in range(count):
            pixels = file.read(frame_bytes)
            assert len(pixels) == frame_bytes
            rows.append(sha256(pixels).hexdigest())
        assert not file.read(1)
    return rows


def smoke(rally_id, source_root, output):
    source_record = json.loads((HERE / f"{rally_id.split('_')[0]}-native.json").read_text())
    row = next(r for r in source_record["rallies"] if r["rally_id"] == rally_id)
    source = source_root / row["filename"]
    assert sha(source) == row["clip_sha256"]
    first, stop = row["native_range"]
    n = stop - first
    w, h = (source_record["media"][k] for k in ("width", "height"))
    fps = Fraction(source_record["media"]["avg_frame_rate"])
    frame_bytes = w * h * 3 // 2
    raw = output / "source.yuv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            str(raw),
        ],
        capture_output=True,
        check=True,
        timeout=90,
    )
    source_hashes = raw_hashes(raw, n, frame_bytes)
    assert (
        sha256("\n".join(source_hashes).encode()).hexdigest()
        == row["proof"]["yuv420p"]["ordered_hash_sha256"]
    )
    rational = [fps.numerator, fps.denominator]
    packet, display = output / "native.psc", output / "native.yuv"
    encoded = codec.encode(
        raw,
        packet,
        dict(
            width=w,
            height=h,
            frames=n,
            fps=rational,
            full_range=False,
            crf=42,
            preset=12,
            input_format="yuv420p",
        ),
        scratch=output,
    )
    decoded = codec.decode(packet, display, scratch=output)
    meta, ivf = codec.unpack(packet.read_bytes())
    assert meta["frames"] == n and meta["fps"] == rational
    assert len(svtav1.ivf_frames(ivf)) == n
    markers = (0, n - 1)
    # Deliberately unrelated to any ball location. This only checks wire and display geometry.
    track = [(30, 30) if t in markers else None for t in range(n)]
    looks = [ball.Appearance((255, 228, 46), 3, 0, 0) if t in markers else None for t in range(n)]
    payload = ball.encode_frames(track, looks, w, h)
    stream = ball_stream.Stream(w, h, fps, n, "svt-av1-ivf", "svt-av1-local-smoke", ivf, payload)
    sidecar_packet = output / "marker.psbl"
    sidecar_packet.write_bytes(ball_stream.pack(stream))
    received = ball_stream.unpack(sidecar_packet.read_bytes())
    assert received == stream and received.duration == Fraction(n) / fps
    marker_display = output / "marker.yuv"
    wrapped = tracked_patch.decode_native(received.native, marker_display, meta, "dav1d")
    assert wrapped["decoded_yuv_sha256"] == decoded["decoded_sha256"]
    frames = np.memmap(marker_display, np.uint8, "r", shape=(n, h * 3 // 2, w))
    centers, decoded_looks = ball.decode_frames(received.ball_payload, n, w, h)
    assert centers == track and decoded_looks == looks
    velocity = ball.velocities(centers)
    shown = []
    for t in (0, 1, n // 2, n - 2, n - 1):
        canvas = tracked_patch.yuv_to_rgb(frames[t], w, h, False)
        rendered = ball.render(canvas, t, centers, velocity, decoded_looks)
        changed = int(np.count_nonzero(np.any(canvas != rendered, axis=-1)))
        assert bool(changed) == (t in markers)
        shown.append(
            {
                "tick": t,
                "changed_pixels": changed,
                "display_sha256": sha256(rendered.tobytes()).hexdigest(),
            }
        )
    native_frames = raw_hashes(display, n, frame_bytes)
    return {
        "rally_id": rally_id,
        "native_range": [first, stop],
        "cadence": rational,
        "dimensions": [w, h],
        "source_count": n,
        "original_video_sha256": source_record["source_sha256"],
        "clip_sha256": sha(source),
        "source_yuv_sha256": sha(raw),
        "source_frame_sha256": source_hashes,
        "ordinary_packet_sha256": sha(packet),
        "ivf_sha256": sha256(ivf).hexdigest(),
        "ordinary_bytes": packet.stat().st_size,
        "ordinary_ivf_bytes": len(ivf),
        "marker_packet_sha256": sha(sidecar_packet),
        "marker_packet_bytes": sidecar_packet.stat().st_size,
        "duration_seconds": str(Fraction(n) / fps),
        "decoded_bytes": display.stat().st_size,
        "decoded_yuv_sha256": decoded["decoded_sha256"],
        "decoded_frame_sha256": native_frames,
        "marker_yuv_sha256": wrapped["decoded_yuv_sha256"],
        "markers_checked": shown,
        "tools": {k: encoded[k] for k in ("encoder", "decoder")},
        "ffmpeg": svtav1.tool("ffmpeg"),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir()
    results = []
    for tag in ("v006_p0014", "v007_p0108"):
        target = a.out / tag
        target.mkdir()
        results.append(smoke(tag, a.sources, target))
    (a.out / "receipt.json").write_text(
        json.dumps(
            {
                "scope": "Mechanical smoke with arbitrary markers, no tracker or rate-quality evidence",
                "rallies": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        [
            (
                r["rally_id"],
                r["source_count"],
                r["duration_seconds"],
                r["ordinary_ivf_bytes"],
                r["decoded_yuv_sha256"],
            )
            for r in results
        ]
    )
