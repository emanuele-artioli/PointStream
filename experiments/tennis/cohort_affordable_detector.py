"""Measure the qualified independent detector on the strongest affordable ordinary control."""

from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import zlib

import cv2
import numpy as np
import torch
from experiments.tennis import cohort_ball_score as score
from experiments.tennis import cohort_ball_judge as judge
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis import shanghai_ball_frozen_score as prior
from experiments.tennis.k5b_trackers import Zoo
from src.codecs import svtav1


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def transport(path):
    with tarfile.open(path) as tar:
        record = json.load(tar.extractfile("display.json"))
        pixels = zlib.decompress(tar.extractfile("display.rgb.z").read())
    assert len(pixels) == record["frames"] * 960 * 544 * 3
    assert hashlib.sha256(pixels).hexdigest() == record["rgb_sha256"]
    return record, np.frombuffer(pixels, np.uint8).reshape(record["frames"], 544, 960, 3)


def freeze(a, n, stage, work, receipt, qualified):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert score.sha(a.rate) == receipt["rate_archive_sha256"]
    meta = json.loads(a.metadata.read_text())
    rates, streams, _ = score.load_bank(a.rate, meta)
    best = receipt["best_affordable"]
    assert qualified["rgb_sha256"]["ordinary"] == receipt["ordinary"]["rgb_sha256"]
    assert best["sha256"] != receipt["ordinary_sha256"]
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == score.TOOLS["dav1d"]
    selected = next(r for r in rates["rows"] if r["native_sha256"] == best["sha256"])
    assert (
        selected["bytes"]
        == best["bytes"]
        <= max(receipt["candidate_bytes"], receipt["ordinary_bytes"])
    )
    ordinary = rates["ordinary"]
    assert ordinary["native_sha256"] == receipt["ordinary_sha256"]
    witness = score.native(streams[ordinary["native_sha256"]], ordinary, work, decoder["path"])
    check = hashlib.sha256()
    for frame in witness[:n]:
        check.update(score.display(frame, ordinary["width"], ordinary["height"]))
    expected = receipt if n == 240 else json.loads(a.smoke_score.read_text())
    assert check.hexdigest() == expected["ordinary"]["rgb_sha256"]
    raw = score.native(streams[best["sha256"]], selected, work, decoder["path"])
    digest = hashlib.sha256()
    prefix = hashlib.sha256()
    compressor = zlib.compressobj(1)
    packed = work / "display.rgb.z"
    with packed.open("wb") as f:
        for t, frame in enumerate(raw[:n]):
            img = score.display(frame, selected["width"], selected["height"])
            digest.update(img)
            if t < 24:
                prefix.update(img)
            f.write(compressor.compress(img))
        f.write(compressor.flush())
    r = dict(
        match=receipt["match"],
        frames=n,
        source_sha256=receipt["source_sha256"],
        labels_sha256=receipt["labels_sha256"],
        packet_sha256=best["sha256"],
        rate_archive_sha256=receipt["rate_archive_sha256"],
        bytes=best["bytes"],
        kbps=best["kbps"],
        width=selected["width"],
        height=selected["height"],
        crf=selected["crf"],
        witness_rgb_sha256=check.hexdigest(),
        rgb_sha256=digest.hexdigest(),
        prefix_sha256=prefix.hexdigest(),
    )
    write_json(work / "display.json", r)
    with tarfile.open(stage / "published.tar", "w") as tar:
        for name in ("display.json", "display.rgb.z"):
            tar.add(work / name, arcname=name)
    return r


def detect(a, n, stage, work, receipt, qualified):
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 9)
    assert "6000 Ada" in torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = False
    assert score.sha(a.display) == a.display_sha
    r, arr = transport(a.display)
    assert r["frames"] == 240 and r["source_sha256"] == receipt["source_sha256"]
    assert (
        r["packet_sha256"] == receipt["best_affordable"]["sha256"]
        and r["match"] == receipt["match"]
    )
    assert r["witness_rgb_sha256"] == receipt["ordinary"]["rgb_sha256"]
    frames = arr[:n]
    h = hashlib.sha256(frames).hexdigest()
    assert h == (r["prefix_sha256"] if n == 24 else r["rgb_sha256"])
    oracle.bind_config(a.wasb_code, prior.CONFIG)
    assert score.sha(a.wasb_weights / "wasb_tennis_best.pth.tar") == prior.WEIGHT
    net = Zoo(a.wasb_code, a.wasb_weights, names=("wasb-step1",))
    cfg = net.settings["wasb-step1"]["cfg"]
    assert next(net.settings["wasb-step1"]["net"].parameters()).device.type == "cuda"
    assert (
        cfg["model"]["frames_in"],
        cfg["model"]["frames_out"],
        cfg["detector"]["step"],
        cfg["detector"]["postprocessor"]["score_threshold"],
    ) == (3, 3, 1, 0.5)
    labels = score.marks(a.labels, receipt["labels_sha256"], n)
    output = judge.detect(net, frames, labels, work, "strongest")
    original = [v for v in qualified["detections"]["ordinary"]["rows"] if v["tick"] < n]
    assert [(v["tick"], v["visibility"]) for v in original] == [
        (v["tick"], v["visibility"]) for v in output["rows"]
    ]
    return dict(
        r,
        frames=n,
        judged_rgb_sha256=h,
        source_hits10=qualified["source_hits10"],
        source_visible=qualified["visible"],
        detector=output,
        model_sha256=prior.WEIGHT,
        display_archive_sha256=a.display_sha,
    )


def run(a):
    started = time.monotonic()
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    stage = Path(os.environ["PS_STAGE_DIR"])
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "strongest"
    work.mkdir()
    receipt = json.loads(a.score.read_text())
    qualified = json.loads(a.qualification.read_text())
    assert (
        qualified["source_qualified"]
        and qualified["source_hits10"] >= 0.8 * qualified["visible"] > 0
    )
    assert (
        qualified["match"] == receipt["match"] and receipt["frames"] == qualified["frames"] == 240
    )
    assert qualified["rgb_sha256"]["source"] == receipt["source_rgb_sha256"]
    assert qualified["source_sha256"] == receipt["source_sha256"]
    assert qualified["labels_sha256"] == receipt["labels_sha256"]
    assert qualified["rate_archive_sha256"] == receipt["rate_archive_sha256"]
    assert qualified["judge_config_sha256"] == prior.CONFIG
    assert qualified["judge_checkpoint_sha256"] == prior.WEIGHT
    result = (freeze if a.mode == "freeze" else detect)(a, n, stage, work, receipt, qualified)
    result.update(
        mode=a.mode,
        qualification_sha256=score.sha(a.qualification),
        elapsed=time.monotonic() - started,
    )
    write_json(stage / "strongest.json", result)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "strongest.json").read_text())
    checks = dict(packet=len(r["packet_sha256"]) == 64, priced=r["kbps"] > 0 and r["bytes"] > 0)
    if r["mode"] == "freeze":
        proof, arr = transport(stage / "published.tar")
        checks["portable"] = proof["rgb_sha256"] == r["rgb_sha256"] and len(arr) == r["frames"]
    else:
        checks["qualified"] = r["source_hits10"] >= 0.8 * r["source_visible"] > 0
        checks["marked"] = len(r["detector"]["rows"]) > 0 or r["frames"] == 24
        checks["readout"] = r["model_sha256"] == prior.WEIGHT and len(r["judged_rgb_sha256"]) == 64
    if r["frames"] == 24:
        checks["forecast"] = 1.5 * r["elapsed"] * 10 + 45 < (1200 if r["mode"] == "freeze" else 540)
    safe = all(checks.values())
    write_json(Path(os.environ["PS_VALIDATION_PATH"]), dict(passed=safe, checks=checks))
    if not safe:
        raise ValueError(checks)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("freeze", "judge"))
    for name in (
        "rate",
        "score",
        "smoke-score",
        "qualification",
        "metadata",
        "display",
        "labels",
        "wasb-code",
        "wasb-weights",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--display-sha")
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
