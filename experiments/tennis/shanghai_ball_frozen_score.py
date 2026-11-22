"""Score only a hash-frozen prospective Shanghai ball display and official marks."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.tennis.k5b_monotrack_cpu import CpuZoo
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis.k5d_encoder import crop64
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import ball, ball_stream, tracked_patch
from experiments.tennis import shanghai_ball_display_freeze as freeze

WEIGHT = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"
LABEL_MEMBER = "tennis/all/match284/csv/000_ball.csv"
LABEL_SHA = "d1180a3eabfc30db3e034bcf7d2d3017d0bc2686155a892d0880d1fced63df98"
BACKBONE = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"


def run(a):
    start = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    assert (
        freeze.sha(a.source) == freeze.SOURCE
        and freeze.sha(a.ordinary) == a.ordinary_sha256
        and freeze.sha(a.candidate) == a.candidate_sha256
        and freeze.sha(a.display) == a.display_sha256
    )
    report = json.loads(freeze.member(a.display, "report.json"))
    assert (
        report["matched"] and report["frames"] == 240 and report["source_sha256"] == freeze.SOURCE
    )
    assert (
        report["ordinary_archive_sha256"] == a.ordinary_sha256
        and report["candidate_archive_sha256"] == a.candidate_sha256
    )
    w, h = report["width"], report["height"]
    o, b = report["ordinary"], report["candidate"]
    native = freeze.member(a.ordinary, o["selected_file"])
    packed = freeze.member(a.candidate, b["selected_file"])
    assert freeze.digest(native) == o["sha256"] and freeze.digest(packed) == b["sha256"]
    packet = ball_stream.unpack(packed)
    assert freeze.digest(packet.ball_payload) == report["payload_sha256"]
    positions, looks = ball.decode_frames(packet.ball_payload, 240, 960, 544)
    velocity = ball.velocities(positions)
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == report["decoder_sha256"]
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "shanghai-score"
    work.mkdir()
    arrays = []
    for name, data, row in [("ordinary", native, o), ("candidate", packet.native, b)]:
        ivf = work / (name + ".ivf")
        ivf.write_bytes(data)
        yuv = work / (name + ".yuv")
        subprocess.run(
            svtav1.decode_command(decoder["path"], ivf, yuv, threads=1),
            check=True,
            capture_output=True,
            timeout=120,
        )
        assert freeze.sha(yuv) == row["decoded_yuv_sha256"]
        arrays.append(np.memmap(yuv, np.uint8, "r", shape=(240, h * 3 // 2, w)))
    displays = {
        k: np.lib.format.open_memmap(
            work / (k + ".npy"), mode="w+", dtype=np.uint8, shape=(a.frames, 544, 960, 3)
        )
        for k in ("source", "ordinary", "candidate")
    }
    hs = {k: hashlib.sha256() for k in displays}
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    for t in range(a.frames):
        ok, bgr = cap.read()
        assert ok and bgr.shape == (1080, 1920, 3)
        source = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (960, 544), interpolation=cv2.INTER_AREA
        )
        out = [
            cv2.resize(
                tracked_patch.yuv_to_rgb(v[t], w, h, False),
                (960, 544),
                interpolation=cv2.INTER_CUBIC,
            )
            for v in arrays
        ]
        out[1] = ball.render(out[1], t, positions, velocity, looks)
        for name, img in zip(displays, [source, *out], strict=True):
            displays[name][t] = img
            hs[name].update(img.tobytes())
    cap.release()
    hashes = {k: v.hexdigest() for k, v in hs.items()}
    expected = report["prefix_rgb_sha256"] if a.frames == 24 else report["rgb_sha256"]
    assert hashes == expected
    # No official coordinate is opened until all three reconstructed RGB hashes agree.
    with tarfile.open(a.annotations) as tar:
        member = tar.extractfile(LABEL_MEMBER)
        assert member is not None
        csv_bytes = member.read()
    assert freeze.digest(csv_bytes) == LABEL_SHA
    labels = []
    seen = set()
    for row in csv.DictReader(io.StringIO(csv_bytes.decode())):
        t = int(row["Frame"])
        kind = int(row["Visibility"])
        if 0 <= t < a.frames:
            assert kind in (0, 1) and t not in seen
            seen.add(t)
            labels.append(
                {"tick": t, "visibility": kind, "x": int(row["X"] or 0), "y": int(row["Y"] or 0)}
            )
    if a.frames == 240:
        assert (
            sum(x["visibility"] == 1 for x in labels) == 26
            and sum(x["visibility"] == 0 for x in labels) == 23
        )
    oracle.bind_config(a.wasb_code, CONFIG)
    assert (
        freeze.sha(a.wasb_weights / "wasb_tennis_best.pth.tar") == WEIGHT
        and freeze.sha(a.backbone) == BACKBONE
    )
    net = CpuZoo(a.wasb_code, a.wasb_weights, names=("wasb-step1",))
    cfg = net.settings["wasb-step1"]["cfg"]
    assert next(net.settings["wasb-step1"]["net"].parameters()).device.type == "cpu"
    assert (
        cfg["model"]["frames_in"],
        cfg["model"]["frames_out"],
        cfg["detector"]["step"],
        cfg["detector"]["postprocessor"]["score_threshold"],
    ) == (3, 3, 1, 0.5)
    dists = quality.load_dists(a.backbone, "cpu")
    mask = np.ones((64, 64), bool)
    self_crop = crop64(displays["source"][0], 480, 272)
    self_started = time.monotonic()
    self_score = float(dists_crops(dists, [self_crop], self_crop, mask, "cpu")[0])
    self_seconds = time.monotonic() - self_started
    assert math.isfinite(self_score) and abs(self_score) < 1e-5
    out = []
    for name in displays:
        prepared = np.lib.format.open_memmap(
            work / (name + "-wasb.npy"), mode="w+", dtype=np.uint8, shape=(a.frames, 720, 1280, 3)
        )
        for t in range(a.frames):
            prepared[t] = cv2.resize(displays[name][t], (1280, 720), interpolation=cv2.INTER_AREA)
        then = time.monotonic()
        pred = net.run("wasb-step1", prepared)
        infer = time.monotonic() - then
        assert len(pred) == a.frames
        rows = []
        for label in labels:
            t = label["tick"]
            det, x, y = pred[t]
            kind = label["visibility"]
            error = (
                math.hypot(x * 1.5 - label["x"], y * 1.5 - label["y"])
                if det and kind == 1
                else None
            )
            point = positions[t] if name == "candidate" else None
            direct_error = (
                math.hypot(point[0] * 2 - label["x"], point[1] * 1080 / 544 - label["y"])
                if point is not None and kind == 1
                else None
            )
            dval = None
            if kind == 1 and name != "source":
                xx = label["x"] / 2
                yy = label["y"] * 544 / 1080
                ref = crop64(displays["source"][t], xx, yy)
                crop = crop64(displays[name][t], xx, yy)
                dval = float(dists_crops(dists, [crop], ref, mask, "cpu")[0])
                assert math.isfinite(dval)
            rows.append(
                {
                    "tick": t,
                    "visibility": kind,
                    "detected": bool(det),
                    "x": float(x),
                    "y": float(y),
                    "error_native_px": error,
                    "direct_present": point is not None,
                    "direct_error_native_px": direct_error,
                    "ball_crop_dists": dval,
                }
            )
        sse = 0
        if name != "source":
            for t in range(a.frames):
                delta = displays["source"][t].astype(np.int32) - displays[name][t].astype(np.int32)
                sse += int(np.sum(delta * delta, dtype=np.int64))
        out.append(
            {
                "mode": name,
                "rows": rows,
                "infer_seconds": infer,
                "rgb_sse": sse,
                "rgb_psnr": (
                    10 * math.log10(255**2 * a.frames * 544 * 960 * 3 / sse) if sse else None
                ),
            }
        )
    result = {
        "stage": os.environ["PS_STAGE"],
        "frames": a.frames,
        "decision": report["decision"],
        "labels_sha256": LABEL_SHA,
        "display_archive_sha256": a.display_sha256,
        "rgb_sha256": hashes,
        "wasb_checkpoint_sha256": WEIGHT,
        "config_sha256": CONFIG,
        "dists_backbone_sha256": BACKBONE,
        "device": "cpu",
        "dists_self_score": self_score,
        "dists_self_seconds": self_seconds,
        "label_rows": len(labels),
        "results": out,
        "elapsed_seconds": time.monotonic() - start,
    }
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    (pub / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    checks = {
        "frames": r["frames"] == n,
        "rgb": len(r["rgb_sha256"]) == 3,
        "arms": len(r["results"]) == 3,
        "same_labels": all(len(x["rows"]) == r["label_rows"] for x in r["results"]),
    }
    if n == 24:
        checks["forecast"] = (
            1.5 * (r["elapsed_seconds"] * 10 + r["dists_self_seconds"] * 52) + 180 < 2200
        )
    if n == 240:
        checks["labels"] = r["label_rows"] == 49
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n"
    )
    assert all(checks.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in (
        "source",
        "ordinary",
        "candidate",
        "display",
        "annotations",
        "wasb-code",
        "wasb-weights",
        "backbone",
    ):
        p.add_argument("--" + n, type=Path)
    for n in ("ordinary-sha256", "candidate-sha256", "display-sha256"):
        p.add_argument("--" + n)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
