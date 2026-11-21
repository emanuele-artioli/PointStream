"""Generate native neural controls with independently repeated receiver decodes."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import zipfile

import cv2
import numpy as np

from experiments.tennis import starved_dcvc_regions as score
from experiments.tennis import w3_neural_baseline as producer
from experiments.tennis.k4b_features import convert_command
from src.codecs import quality, svtav1

SOURCE = "b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d"
YUV = "5a26ceb4c08fdfac2726805bb37c5cefe29ff18b9a3c733690fd4e263d3d972d"


def run(a):
    import torch

    start = time.monotonic()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and "6000 Ada" in torch.cuda.get_device_name(0)
    assert (score.sha(a.source), score.sha(a.regions), score.sha(a.frames)) == (
        SOURCE,
        score.REGIONS,
        score.FRAMES,
    )
    assert score.sha(a.vgg) == score.WEIGHTS["vgg"]
    net = quality.load_dists(a.vgg, "cuda")
    work = Path(os.environ["PS_SCRATCH_DIR"])
    publish = work / "publish"
    publish.mkdir()
    src = work / "source.yuv"
    tool = svtav1.tool("ffmpeg")
    command = convert_command(tool["path"], a.source, src, 0, 240)
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    assert score.sha(src) == YUV
    a.image_ckpt = a.image
    a.video_ckpt = a.video
    a.stream_policy = "legacy"
    qps = [30] if a.count == 1 else [30, 42]
    points = []
    rows = []
    attempts = []
    with zipfile.ZipFile(a.regions) as regions, zipfile.ZipFile(a.frames) as frames:
        records = json.loads(regions.read("regions.json"))
        ticks = [10] if a.count == 1 else list(range(10, 240, 20))
        for qp in qps:
            for attempt in range(2):
                dest = work / f"q{qp}-try{attempt}"
                dest.mkdir()
                r = producer.code_point(
                    a, src, dest, 50, 240, dict(scale=2, qp=qp, target_kbps=0, est_kbps=0)
                )
                attempts.append(
                    dict(
                        qp=qp,
                        attempt=attempt,
                        accepted=r["accepted"],
                        failure=r.get("failure"),
                        deterministic=r.get("within_process_deterministic"),
                        maximum_difference=r.get("max_decode_diff"),
                    )
                )
                if r["accepted"]:
                    break
            else:
                continue
            raw = Path(r["decoded"])
            assert raw.stat().st_size == 240 * 544 * 960 * 3 // 2 and r["identical_frame_hashes"]
            bitstream = raw.parent / "stream.psdc"
            assert (
                score.sha(bitstream) == r["container_sha256"]
                and r["bytes"] == bitstream.stat().st_size
            )
            shutil.copy(bitstream, publish / f"q{qp}.psdc")
            reports = json.loads((raw.parent / "reports.json").read_text())
            expected = reports["decode"]["decode-a"]["passes"][0]["frame_sha256"]
            assert len(expected) == 240 and all(
                p["frame_sha256"] == expected
                for v in reports["decode"].values()
                for p in v["passes"]
            )
            (publish / f"q{qp}-frames.json").write_text(json.dumps(expected))
            meta = {
                k: r[k]
                for k in (
                    "qp",
                    "kbps",
                    "bytes",
                    "size",
                    "frames",
                    "container_sha256",
                    "check_a_i_frame",
                    "check_b_two_processes",
                    "identical_frame_hashes",
                    "within_process_deterministic",
                )
            }
            pixels = np.memmap(raw, "uint8", "r", shape=(240, 816, 960))
            rows.extend(score.measure(net, pixels, meta, qp, ticks, records, regions, frames))
            points.append(meta)
    result = dict(
        points=points,
        rows=rows,
        attempts=attempts,
        elapsed_seconds=time.monotonic() - start,
        source_sha256=SOURCE,
        stream_policy=a.stream_policy,
        cuda_launch_blocking=False,
        source_yuv_sha256=YUV,
        conversion=command,
        ffmpeg=tool,
        gpu=torch.cuda.get_device_name(0),
        metric=quality.dists_record(a.vgg),
    )
    (publish / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as t:
        r = json.load(t.extractfile("publish/result.json"))
    smoke = os.environ["PS_STAGE"] == "smoke"
    forecast = r["elapsed_seconds"] * (3.0 if smoke else 1) + 100
    checks = dict(
        source=r["source_yuv_sha256"] == YUV,
        exact_decode=all(v["identical_frame_hashes"] for v in r["points"]),
        numerical_control=all(abs(row["anchor_delta"]) <= 1e-4 for row in r["rows"]),
        complete=len(r["rows"]) == (3 if smoke else 72),
        cost=not smoke or forecast < 600,
    )
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=all(checks.values()), checks=checks, forecast_seconds=forecast))
    )
    if not all(checks.values()):
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "regions", "frames", "image", "video", "vgg"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--count", type=int, choices=(1, 2))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
