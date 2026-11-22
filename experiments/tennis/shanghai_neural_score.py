"""Score a sealed DCVC display on the same Shanghai labels and detector as PSBL."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import al028_ball_sender as source
from experiments.tennis import shanghai_ball_display_freeze as freeze
from experiments.tennis import shanghai_ball_frozen_score as primary
from experiments.tennis.k5b_monotrack_cpu import CpuZoo
from experiments.tennis.k5d_encoder import crop64
from experiments.visor.h3b import dists_crops
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb


def run(a):
    start = time.monotonic()
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    assert freeze.sha(a.neural) == a.neural_sha and freeze.sha(a.display) == a.display_sha
    display = json.loads(freeze.member(a.display, "report.json"))
    assert (
        display["matched"]
        and display["frames"] == 240
        and display["source_sha256"] == freeze.SOURCE
    )
    source_rgb = source.source_frames(a.source, n, "shanghai")
    expected = display["rgb_sha256"] if n == 240 else display["prefix_rgb_sha256"]
    assert hashlib.sha256(source_rgb).hexdigest() == expected["source"]
    neural = json.loads(freeze.member(a.neural, "result.json"))
    point = neural["native"]
    assert (
        neural["frames"] == 240
        and neural["source_sha256"] == freeze.SOURCE
        and point["accepted"]
        and point["identical_frame_hashes"]
    )
    data = freeze.member(a.neural, "decoded.yuv")
    w, h = point["size"]
    assert (w, h) in ((960, 544), (480, 272))
    assert len(data) == 240 * w * h * 3 // 2 and freeze.digest(data) == neural["raw_sha256"]
    raw = np.frombuffer(data, np.uint8).reshape(240, h * 3 // 2, w)
    work = Path(os.environ["PS_SCRATCH_DIR"])
    rgb = np.lib.format.open_memmap(
        work / "display.npy", mode="w+", dtype=np.uint8, shape=(n, 544, 960, 3)
    )
    inputs = np.lib.format.open_memmap(
        work / "network.npy", mode="w+", dtype=np.uint8, shape=(n, 720, 1280, 3)
    )
    sse = 0
    for t in range(n):
        assert hashlib.sha256(raw[t]).hexdigest() == neural["yuv_frames"][t]
        rgb[t] = cv2.resize(
            yuv_to_rgb(raw[t], w, h, False), (960, 544), interpolation=cv2.INTER_CUBIC
        )
        inputs[t] = cv2.resize(rgb[t], (1280, 720), interpolation=cv2.INTER_AREA)
        delta = source_rgb[t].astype(np.int32) - rgb[t].astype(np.int32)
        sse += int(np.sum(delta * delta, dtype=np.int64))
    with tarfile.open(a.annotations) as archive:
        labels = archive.extractfile(primary.LABEL_MEMBER).read()
    assert freeze.digest(labels) == primary.LABEL_SHA
    primary.oracle.bind_config(a.wasb_code, primary.CONFIG)
    assert (
        freeze.sha(a.wasb_weights / "wasb_tennis_best.pth.tar") == primary.WEIGHT
        and freeze.sha(a.backbone) == primary.BACKBONE
    )
    judge = CpuZoo(a.wasb_code, a.wasb_weights, names=("wasb-step1",))
    cfg = judge.settings["wasb-step1"]["cfg"]
    assert next(judge.settings["wasb-step1"]["net"].parameters()).device.type == "cpu"
    assert (
        cfg["model"]["frames_in"],
        cfg["model"]["frames_out"],
        cfg["detector"]["step"],
        cfg["detector"]["postprocessor"]["score_threshold"],
    ) == (3, 3, 1, 0.5)
    pred = judge.run("wasb-step1", inputs)
    assert len(pred) == n
    dists = quality.load_dists(a.backbone, "cpu")
    mask = np.ones((64, 64), bool)
    rows = []
    for r in csv.DictReader(io.StringIO(labels.decode())):
        t = int(r["Frame"])
        if not 0 <= t < n:
            continue
        visible = int(r["Visibility"])
        assert visible in (0, 1)
        det, x, y = pred[t]
        error = score = None
        if visible:
            xx, yy = int(r["X"]), int(r["Y"])
            if det:
                error = math.hypot(x * 1.5 - xx, y * 1.5 - yy)
            crops = [crop64(img[t], xx / 2, yy * 544 / 1080) for img in (rgb, source_rgb)]
            score = float(dists_crops(dists, [crops[0]], crops[1], mask, "cpu")[0])
            assert math.isfinite(score)
        rows.append(
            dict(
                tick=t,
                visible=visible,
                detected=bool(det),
                error_native_px=error,
                ball_crop_dists=score,
            )
        )
    result = dict(
        frames=n,
        stage=os.environ["PS_STAGE"],
        source_sha256=freeze.SOURCE,
        neural_archive_sha256=a.neural_sha,
        neural_raw_sha256=neural["raw_sha256"],
        packet_sha256=point["container_sha256"],
        bytes=point["bytes"],
        kbps=point["kbps"],
        qp=point["qp"],
        size=[w, h],
        display_archive_sha256=a.display_sha,
        judge_checkpoint_sha256=primary.WEIGHT,
        label_sha256=primary.LABEL_SHA,
        rgb_sha256=hashlib.sha256(rgb).hexdigest(),
        whole_rgb_psnr=10 * math.log10(255**2 * n * 960 * 544 * 3 / sse),
        source_rgb_sha256=expected["source"],
        rows=rows,
        elapsed_seconds=time.monotonic() - start,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    n = r["frames"]
    rows = r["rows"]
    checks = {
        "source": r["source_sha256"] == freeze.SOURCE and r["label_sha256"] == primary.LABEL_SHA,
        "cohort": len(set(x["tick"] for x in rows)) == len(rows)
        and all(0 <= x["tick"] < n for x in rows),
        "metric": all((x["ball_crop_dists"] is not None) == bool(x["visible"]) for x in rows),
        "rate": abs(r["kbps"] - r["bytes"] / 1200) < 1e-9 and r["whole_rgb_psnr"] > 0,
    }
    if r["stage"] == "full":
        checks["count"] = n == 240 and len(rows) == 49 and sum(x["visible"] for x in rows) == 26
    else:
        checks["forecast"] = n == 24 and r["elapsed_seconds"] * 1.5 * 10 + 90 < 1700
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks)) + "\n"
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "display",
        "neural",
        "annotations",
        "wasb-code",
        "wasb-weights",
        "backbone",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--neural-sha")
    p.add_argument("--display-sha")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
