"""Re-score the preserved London deliveries on the common complete display."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np

from experiments.tennis import k5c_independent_ball, windows
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 960, 544


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def luma_ssim(original, decoded):
    def blur(pixels):
        return cv2.GaussianBlur(pixels, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)

    x, y = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) for image in (original, decoded)
    )
    mx, my = blur(x), blur(y)
    vx = np.maximum(blur(x * x) - mx * mx, 0)
    vy = np.maximum(blur(y * y) - my * my, 0)
    covariance = blur(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    luminance = (2 * mx * my + c1) / (mx * mx + my * my + c1)
    contrast = (2 * covariance + c2) / (vx + vy + c2)
    return float(np.mean(luminance * contrast, dtype=np.float64))


def measure(originals, displays, ticks):
    if not ticks:
        return None
    psnr = []
    ssim = []
    for tick in ticks:
        original, display = originals[tick], displays[tick]
        mse = float(np.mean((original.astype(np.float32) - display) ** 2, dtype=np.float64))
        assert mse > 0
        psnr.append(10 * math.log10(255**2 / mse))
        ssim.append(luma_ssim(original, display))
    return {"rgb_psnr": mean(psnr), "luma_ssim": mean(ssim), "frames": len(ticks)}


def source_frames(path, n):
    video = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        assert abs(video.get(cv2.CAP_PROP_FPS) - 25) < 0.001
        frames = []
        for _ in range(n):
            ok, frame = video.read()
            assert ok and frame.shape == (720, 1280, 3)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        assert not video.read()[0]
        return np.stack(frames)
    finally:
        video.release()


def decode(stream, coded_size, n, scratch):
    width, height = coded_size
    assert len(svtav1.ivf_frames(stream)) == n
    native, yuv, rgb = (
        scratch / filename for filename in ("stream.ivf", "pixels.yuv", "pixels.npy")
    )
    native.write_bytes(stream)
    yuv.unlink(missing_ok=True)
    svtav1.run(svtav1.decode_command("dav1d", native, yuv, threads=1), timeout=120)
    assert yuv.stat().st_size == n * width * height * 3 // 2
    raw_rgb = windows.rgb_frames(yuv, width, height, n, rgb)
    return np.stack(
        [cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) for frame in raw_rgb]
    )


def compare(root, row, ordinary_arms, scratch):
    rally_id = row["rally_id"]
    directory = root / rally_id
    baseline = json.loads((HERE / "local-av1" / f"{rally_id}.json").read_text())
    assert sha(directory / f"{rally_id}.mkv") == baseline["source_sha256"]
    n = row["frames"]
    originals = source_frames(directory / f"{rally_id}.mkv", n)
    sender_path = HERE / "tenniset-ball" / f"{rally_id}.json"
    assert sha(sender_path) == row["receipt_sha256"]
    sender = json.loads(sender_path.read_text())
    assert digest(originals) == sender["source_context_sha256"]
    groups = {"all": list(range(n)), "court": [], "fallback": []}
    route_file = HERE / "routing" / f"{rally_id}.json"
    route = json.loads(route_file.read_text())
    assert route["source_sha256"] == baseline["source_sha256"]
    assert route["native_fps"] == sender["fps"]
    for shot in route["shots"]:
        phase = "court" if shot["predicted"] == "court" else "fallback"
        groups[phase].extend(range(shot["start"], shot["stop"]))
    assert sorted(groups["court"] + groups["fallback"]) == groups["all"]
    assert len(groups["fallback"]) == row["fallback_ticks"]

    results = {}
    decoded_cache = {}
    for price, candidate in row["selected"].items():
        coded_size = candidate["coded_size"]
        width = coded_size[0]
        packet = directory / f"{width}-{candidate['ceiling']}.psbl"
        assert sha(packet) == candidate["sha256"]
        stream = ball_stream.unpack(packet.read_bytes())
        assert stream.frames == n and stream.fps == Fraction(25)
        control = next(arm["selected"][price] for arm in ordinary_arms if arm["size"] == coded_size)
        item = directory / control["chosen"]
        expected = next(
            a["sha256"]
            for arm in baseline["arms"]
            if arm["width"] == width
            for a in arm["available"]
            if a["file"] == item.name
        )
        assert sha(item) == expected and item.stat().st_size == control["bytes"]
        control_bytes = item.read_bytes()
        for native in (stream.native, control_bytes):
            key = hashlib.sha256(native).hexdigest()
            if key not in decoded_cache:
                decoded_cache[key] = decode(native, coded_size, n, scratch)
        base = decoded_cache[candidate["base_sha256"]]
        av1 = decoded_cache[expected]
        positions, appearance = ball.decode_frames(stream.ball_payload, n, WIDTH, HEIGHT)
        rendered = k5c_independent_ball.render_frames(base, positions, appearance)
        assert digest(rendered) == candidate["displayed_sha256"]
        assert all(np.array_equal(rendered[t], base[t]) for t in groups["fallback"])
        scores = {}
        previous = {}
        for arm, display in (("av1", av1), ("exact_base", base), ("ball", rendered)):
            key = digest(display)
            if key not in previous:
                previous[key] = {
                    group: measure(originals, display, ticks) for group, ticks in groups.items()
                }
            scores[arm] = previous[key]
        results[price] = {
            "coded_size": coded_size,
            "actual_bytes": {
                "av1": len(control_bytes),
                "exact_base": len(stream.native),
                "ball": packet.stat().st_size,
            },
            "ball_packet_sha256": sha(packet),
            "native_av1_sha256": expected,
            "ball_display_sha256": digest(rendered),
            "changed_frames": sum(
                not np.array_equal(a, b) for a, b in zip(rendered, base, strict=True)
            ),
            "scores": scores,
        }
    return {
        "rally": rally_id,
        "native_frames": n,
        "route_sha256": sha(route_file),
        "metrics": results,
    }


def run(args):
    began = time.process_time()
    cv2.setNumThreads(1)
    summary_path = HERE / "tenniset-ball-summary.json"
    rows = json.loads(summary_path.read_text())["rows"]
    controls = json.loads((HERE / "local-av1-summary.json").read_text())
    outputs = []
    with TemporaryDirectory() as scratch:
        for row in rows[3:4] if args.smoke else rows:
            ordinary = controls[row["rally_id"]]["arms"]
            result = compare(args.artifacts, row, ordinary, Path(scratch))
            outputs.append(result)
            print(result["rally"], result["native_frames"], flush=True)
    result = {
        "scope": "Full common 960x544 display, no foreground annotations",
        "delivery_receipt_sha256": sha(summary_path),
        "scorer_sha256": sha(Path(__file__)),
        "dav1d": svtav1.tool("dav1d"),
        "opencv": cv2.__version__,
        "psnr": "arithmetic mean of per-frame RGB PSNR",
        "ssim": "arithmetic mean of spatial and frame means of 11x11 Gaussian luma SSIM",
        "gpu_seconds": 0,
        "rows": outputs,
        "cpu_seconds": time.process_time() - began,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
