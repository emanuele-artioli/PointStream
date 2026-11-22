"""Tighten the two largest London rate gaps using only conventional AV1 encodes."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2

from heldout_display_metrics import HERE, compare, decode, measure, sha, source_frames
from native_av1 import command
from src.codecs import svtav1

TARGETS = {"v007_p0108": "20400", "v007_p0022": "33000"}


def run(artifacts, output):
    cv2.setNumThreads(1)
    began = time.monotonic()
    manifest = json.loads((HERE / "tenniset-ball-summary.json").read_text())
    controls = json.loads((HERE / "local-av1-summary.json").read_text())
    outputs = []
    for row in manifest["rows"]:
        name = row["rally_id"]
        if name not in TARGETS:
            continue
        price, n = TARGETS[name], row["frames"]
        candidate = row["selected"][price]
        target = candidate["bytes"]
        assert candidate["coded_size"] == [640, 360]
        source = artifacts / name / f"{name}.mkv"
        assert sha(source) == controls[name]["source_sha256"]
        with TemporaryDirectory() as work:
            scratch = Path(work)
            raw = scratch / "codesource.yuv"
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(source),
                "-an",
                "-vf",
                "scale=640:360:flags=area:out_range=tv,format=yuv420p",
                "-f",
                "rawvideo",
                str(raw),
            ]
            command(cmd)
            assert raw.stat().st_size == n * 640 * 360 * 3 // 2
            history = []
            lo, hi = 67 * 4, 70 * 4
            for _ in range(7):
                if hi - lo < 2:
                    break
                step = (lo + hi) // 2
                probe = scratch / f"ordinary-q{step / 4:g}.ivf"
                args = svtav1.encode_command(
                    "SvtAv1EncApp",
                    raw,
                    probe,
                    width=640,
                    height=360,
                    fps=Fraction(25),
                    frames=n,
                    crf=step / 4,
                    preset=4,
                    full_range=False,
                )
                svtav1.run(args, timeout=120)
                size = probe.stat().st_size
                history.append(
                    {"qp": step / 4, "path": probe.name, "bytes": size, "sha256": sha(probe)}
                )
                if abs(size / target - 1) <= 0.005:
                    break
                if size > target:
                    lo = step
                else:
                    hi = step
            admissible = [item for item in history if item["bytes"] <= candidate["ceiling"]]
            assert admissible, (name, history)
            best = min(admissible, key=lambda item: abs(item["bytes"] - target))
            best_native = (scratch / best["path"]).read_bytes()
            assert len(best_native) == best["bytes"]
            physical = compare(artifacts, row, controls[name]["arms"], scratch)["metrics"][price]
            original = source_frames(source, n)
            matched = decode(best_native, [640, 360], n, scratch)
            output.parent.mkdir(exist_ok=True)
            ivf = output.parent / f"london-{name}-{price}-closest-av1.ivf"
            ivf.write_bytes(best_native)
            assert sha(ivf) == best["sha256"]
            results = {
                "rally": name,
                "ceiling_bps": int(price),
                "frames": n,
                "ball_delivery_sha256": candidate["sha256"],
                "ball_bytes": target,
                "tested": history,
                "selected": best,
                "selected_fractional_rate_gap": best["bytes"] / target - 1,
                "ball": physical["scores"]["ball"]["all"],
                "original_coarse_av1": physical["scores"]["av1"]["all"],
                "tighter_av1": measure(original, matched, list(range(n))),
                "source_sha256": sha(source),
                "av1_input_sha256": sha(raw),
            }
            outputs.append(results)
            print(name, target, best["bytes"], round(time.monotonic() - began, 1), flush=True)
    tools = {name: svtav1.tool(name) for name in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    output.write_text(
        json.dumps(
            {
                "scope": "Whole picture on two large preflight rate mismatches, no foreground labels",
                "tools": tools,
                "scorer_sha256": sha(HERE / "heldout_display_metrics.py"),
                "worker_sha256": sha(Path(__file__)),
                "gpu_seconds": 0,
                "wall_seconds": time.monotonic() - began,
                "results": outputs,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifacts, args.output)
