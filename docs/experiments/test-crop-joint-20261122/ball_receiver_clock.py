"""Time the exact paid background-and-ball canvas at native cadence on one CPU thread."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import cv2
import numpy as np

from experiments.tennis import med_mixed_channel as mixed
from experiments.tennis.player_requalification import array_sha
from src.codecs import svtav1

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(a):
    start = time.perf_counter()
    # GCD reports 10 after setNumThreads(1) on the M1 Pro, while zero disables parallel dispatch.
    cv2.setNumThreads(0)
    assert cv2.getNumThreads() == 1
    sender = json.loads((HERE / "match102-sender.json").read_text())
    row = next(r for r in sender["rates"] if r["crf"] == 50)
    joint, cache, carrier = [a.paid / n for n in ("rally.psj", "initial.psr", "players.pscp")]
    assert [sha(f) for f in (joint, cache, carrier)] == [
        row[k] for k in ("joint_sha256", "init_sha256", "carrier_sha256")
    ]
    decoder = svtav1.tool("dav1d")
    read = time.perf_counter()
    channel = mixed.decode(joint, cache, carrier, row["carrier_sha256"])
    parse_s = time.perf_counter() - read
    assert channel["wire"]["ball_positions"] == 217
    muted = dict(channel, pos=[None] * 240)
    timings = []
    pixels = {}
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        base = Path(td) / "paid.ivf"
        base.write_bytes(channel["base"])
        assert sha(base) == row["base_sha256"]
        for i, order in enumerate((("paid", "muted"), ("muted", "paid"), ("paid", "muted"))):
            yuv = Path(td) / f"pass{i}.yuv"
            t = time.perf_counter()
            subprocess.run(
                svtav1.decode_command(decoder["path"], base, yuv, threads=1),
                check=True,
                capture_output=True,
                timeout=30,
            )
            native = time.perf_counter() - t
            assert sha(yuv) == row["decoded_sha256"] and yuv.stat().st_size == 240 * 144 * 160
            raw = np.memmap(yuv, np.uint8, "r", shape=(240, 144, 160))
            if i == 0:
                t = time.perf_counter()
                for idx in (0, 39, 119, 239):
                    f = mixed.canvas(raw[idx], idx, channel)
                    assert f.shape == (1080, 1920, 3) and f.dtype == np.uint8
                assert 1.5 * (time.perf_counter() - t) * 240 / 4 < 240
            pass_times = {"native_s": native, "order": order}
            for mode in order:
                c = channel if mode == "paid" else muted
                t = time.perf_counter()
                c0 = time.process_time()
                last = 0
                for idx in range(240):
                    frame = mixed.canvas(raw[idx], idx, c)
                    last += int(frame[0, 0, 0])
                elapsed = time.perf_counter() - t
                assert last > 0
                pass_times[mode + "_canvas_s"] = elapsed
                pass_times[mode + "_cpu_s"] = time.process_time() - c0
            timings.append(pass_times)
        for mode, c in (("paid", channel), ("muted", muted)):
            pixels[mode] = [array_sha(mixed.canvas(raw[idx], idx, c)) for idx in range(240)]
    changed = [i for i in range(240) if pixels["paid"][i] != pixels["muted"][i]]
    assert changed and all(channel["pos"][i] is not None for i in changed)
    total = [r["native_s"] + r["paid_canvas_s"] for r in timings]
    duration = 240 * 1001 / 30000
    result = dict(
        joint_sha256=sha(joint),
        cache_sha256=sha(cache),
        source=sender["source_sha256"],
        frames=240,
        native_fps=sender["native_fps"],
        cpu=platform.processor(),
        machine=platform.machine(),
        platform=platform.platform(),
        opencv=cv2.__version__,
        opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        native_decoder=decoder,
        opencv_threads=cv2.getNumThreads(),
        native_decoder_threads=1,
        cache_parse_s=parse_s,
        passes=timings,
        displayed_at=[1920, 1080],
        ball_processing_at=[960, 544],
        clip_seconds=duration,
        median_payload_decode_and_canvas_s=statistics.median(total),
        worst_payload_decode_and_canvas_s=max(total),
        median_fps=240 / statistics.median(total),
        meets_native_throughput=statistics.median(total) <= duration,
        sender_present=217,
        pixels_changed_ticks=len(changed),
        paid_canvas_sha256=pixels["paid"],
        price=dict(running_bytes=row["joint_bytes"], cold_cache_bytes=row["init_bytes"]),
        wall_s=time.perf_counter() - start,
    )
    assert result["wall_s"] < 300 and result["opencv_threads"] == 1
    result["cpu_model"] = a.cpu_model
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["median_fps"],
        result["median_payload_decode_and_canvas_s"],
        result["worst_payload_decode_and_canvas_s"],
        parse_s,
        len(changed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("paid", "scratch", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--cpu-model", required=True)
    main(p.parse_args())
