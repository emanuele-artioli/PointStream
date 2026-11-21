"""Apply the existing court gate to the two routed rallies and marked crowd shots."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import court, k8, k8b, passage_native

INTERVALS = {
    "first": (76626, 76866),
    "crowd_point": (81848, 82292),
    "crowd_interlude": (84351, 84485),
    "return": (87691, 87931),
}


def run(a):
    if (
        len(os.sched_getaffinity(0)) != 1
        or passage_native.gap.sha(a.source) != passage_native.gap.SOURCE_SHA
    ):
        raise ValueError("Wrong original or CPU allocation")
    start = time.monotonic()
    ff = passage_native.svtav1.tool("ffmpeg")
    if ff["sha256"] != passage_native.gap.TOOLS["ffmpeg"]:
        raise ValueError("Wrong FFmpeg")
    cv2.setNumThreads(1)
    work = Path(os.environ["PS_SCRATCH_DIR"])
    records = {}
    spans = (
        list(INTERVALS.items())
        if a.frames == 1058
        else [(k, INTERVALS[k]) for k in ("first", "crowd_point")]
    )
    for key, (lo, hi) in spans:
        clock = time.monotonic()
        n = hi - lo if a.frames == 1058 else 24
        seek = [
            ff["path"],
            "-v",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-ss",
            f"{(lo + 1) / 50:.6f}",
            "-i",
            str(a.source),
        ]
        scorefile = work / f"{key}.log"
        cut = seek + [
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            f"trim=end_frame={n},setpts=PTS-STARTPTS,select='gte(scene,0.10)',metadata=print:file={scorefile}",
            "-t",
            str(n / 50),
            "-f",
            "null",
            "-",
        ]
        subprocess.run(cut, check=True, timeout=120)
        cuts = [
            r["frame"]
            for r in k8.parse_scenes(scorefile.read_text(), 50)
            if r["score"] >= k8b.DECISION["scene_threshold"] and 0 < r["frame"] < n
        ]
        cmd = seek + [
            "-frames:v",
            str(n),
            "-filter_threads",
            "1",
            "-vf",
            "scale=1920:1080:flags=area:out_range=tv,format=yuv420p",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ]
        rows = []
        last = None
        for t, rgb in enumerate(k8b.frames(cmd, n)):
            if t % k8b.CHUNK == 0:
                last = None
            r = court.register_v2(rgb, last)
            last = np.array(r["homography"]) if r["homography"] is not None else None
            if r["fits"] != (r["error_px"] is not None and r["error_px"] <= k8b.DECISION["fit_px"]):
                raise ValueError("Fit flag differs from gate")
            rows.append(dict(fits=bool(r["fits"]), error=r["error_px"]))
        parts = [
            dict(start=lo + x, end=lo + y, **k8b.prediction(rows[x:y]))
            for x, y in k8b.clip_shots(rows, cuts, 50)
        ]
        records[key] = dict(
            frames=n,
            seconds=time.monotonic() - clock,
            cuts=cuts,
            parts=parts,
            rows=rows,
            cmd=cmd,
            cut_cmd=cut,
        )
    r = dict(
        source=passage_native.gap.SOURCE_SHA,
        frames=a.frames,
        decisions=k8b.DECISION,
        ff=ff,
        records=records,
        seconds=time.monotonic() - start,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(json.dumps(r, indent=2) + "\n")


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    smoke = os.environ["PS_STAGE"] == "smoke"
    expected = 48 if smoke else 1058
    tests = dict(
        source=r["source"] == passage_native.gap.SOURCE_SHA,
        frames=sum(x["frames"] for x in r["records"].values()) == expected,
        settings=r["decisions"] == k8b.DECISION,
        cases=set(r["records"]) == ({"first", "crowd_point"} if smoke else set(INTERVALS)),
    )
    for k, x in r["records"].items():
        tests[k] = (
            len(x["rows"]) == x["frames"]
            and sum(s["end"] - s["start"] for s in x["parts"]) == x["frames"]
            and all(s["predicted"] in ("court", "non_court") for s in x["parts"])
        )
    estimate = None
    if smoke:
        estimate = (
            1.5
            * (
                480 * r["records"]["first"]["seconds"] / 24
                + 578 * r["records"]["crowd_point"]["seconds"] / 24
            )
            + 90
        )
        tests["fits"] = estimate < 1550
    result = dict(passed=all(tests.values()), checks=tests, forecast=estimate)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--frames", type=int, choices=(48, 1058))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
