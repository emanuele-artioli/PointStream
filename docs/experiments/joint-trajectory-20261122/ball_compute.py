"""Time two real ball-study packets through a common-sized display buffer."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from src.pointstream import ball, ball_stream
from src.pointstream.tracked_patch import decode_native, yuv_to_rgb


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(args):
    args.out.mkdir()
    cv2.setNumThreads(1)
    receipt = json.loads((args.run / "result.json").read_text())
    encoder = receipt["tools"]["dav1d"]
    assert sha(Path(encoder["path"])) == encoder["sha256"]
    a = receipt["anchors"]["plain160Q61"]
    inputs = {
        "ordinary": Path(a["stream"]).read_bytes(),
        "trajectory": (args.run / "stream.psbl").read_bytes(),
    }
    assert hashlib.sha256(inputs["ordinary"]).hexdigest() == a["stream_sha256"]
    assert hashlib.sha256(inputs["trajectory"]).hexdigest() == receipt["sha256"]
    rows = []
    # All packet checks and decoder output verification use the production decoding path.
    for pair, order in enumerate(
        (("ordinary", "trajectory"), ("trajectory", "ordinary"), ("ordinary", "trajectory"))
    ):
        for arm in order:
            start = time.perf_counter()
            wire = inputs[arm]
            if arm == "trajectory":
                payload = ball_stream.unpack(wire)
                native = payload.native
                pts, appearance = ball.decode_frames(payload.ball_payload, 240, 960, 544)
                velocities = ball.velocities(pts)
            else:
                native = wire
            path = args.out / f"{arm}-{pair}.yuv"
            decode_native(
                native, path, dict(width=160, height=96, frames=240, fps=[50, 1]), encoder["path"]
            )
            native_end = time.perf_counter()
            raw = np.memmap(path, np.uint8, "r", shape=(240, 144, 160))
            digest = hashlib.sha256()
            original_hashes = {}
            for t, frame in enumerate(raw):
                image = yuv_to_rgb(frame, 160, 96, False)
                if arm == "trajectory":
                    image = cv2.resize(image, (960, 544), interpolation=cv2.INTER_CUBIC)
                    image = ball.render(image, t, pts, velocities, appearance)
                    # Authentication is outside the reported runs; keep identical timed digest work for both.
                    if pair == 0:
                        original_hashes[str(t)] = hashlib.sha256(image).hexdigest()
                digest.update(cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC))
            elapsed = time.perf_counter() - start
            if pair == 0 and arm == "trajectory":
                assert original_hashes == receipt["decoded_frame_hashes"]
            assert sha(path) == (
                sha(args.run / "independent.yuv") if arm == "trajectory" else a["decoded_sha256"]
            )
            rows.append(
                dict(
                    arm=arm,
                    pair=pair,
                    seconds=elapsed,
                    parse_and_native_seconds=native_end - start,
                    full_rgb_sha256=digest.hexdigest(),
                    input_bytes=len(wire),
                )
            )
            assert sum(r["seconds"] for r in rows) < 120
    summary = {}
    for arm in inputs:
        chosen = [r for r in rows if r["arm"] == arm]
        assert len({r["full_rgb_sha256"] for r in chosen}) == 1
        # The first pair additionally authenticates every intermediate trajectory RGB frame.
        summary[arm] = statistics.mean(r["seconds"] for r in chosen if r["pair"] > 0)
    report = dict(
        rows=rows,
        mean_last_two_seconds=summary,
        opencv=cv2.__version__,
        system=platform.platform(),
        device=subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        decoder=encoder,
        input_receipt_sha256=sha(args.run / "result.json"),
        script_sha256=sha(Path(__file__)),
        duration=4.8,
        frames=240,
    )
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("out", type=Path)
    main(cli.parse_args())
