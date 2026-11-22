"""Source-pixel racket paste oracle; no crop pixels leave the host."""

import argparse
import hashlib
import itertools
import json
import math
import os
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.codecs import svtav1
from src.pointstream import tracked_patch

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
MODEL = Path("/home/itec/emanuele/Models/YOLO/yolo11x.pt")
SOURCE = ROOT / "jobs/fleet/inbox/20261115T081356Z-4b32fe68/full/published.tar"
PACKETS = ROOT / "jobs/fleet/inbox/20261121T223136Z-e9219942/full/published.tar"
LABELS = ROOT / "tennis/equipment-window-20261122/source-6f403798.json"
OUT = Path("/local/users/emanuele/pointstream/audits/racket-crop-gate-20261122")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def match(gt, pred):
    assert len(gt) <= 2 and len(pred) <= 100
    best = None
    for assign in itertools.product([-1] + list(range(len(pred))), repeat=len(gt)):
        used = [x for x in assign if x >= 0]
        if len(used) != len(set(used)):
            continue
        vals = [iou(a, pred[j]) if j >= 0 else 0.0 for a, j in zip(gt, assign, strict=True)]
        score = (sum(v >= 0.5 for v in vals), sum(vals))
        if best is None or score > best[0]:
            best = (score, assign, vals)
    return {
        "assignment": best[1],
        "iou": best[2],
        "unassociated_predictions": len(pred) - sum(v >= 0.5 for v in best[2]),
    }


def predictions(net, img):
    result = net.predict(
        source=img[:, :, ::-1],
        device="cpu",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        classes=[38],
        verbose=False,
    )[0]
    return [
        {"box": [float(x) for x in b], "confidence": float(c)}
        for b, c in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    a = parser.parse_args()
    n = 16 if a.mode == "smoke" else 64
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / (a.mode + ".json")
    assert not output.exists()
    start = time.monotonic()
    assert sha(MODEL) == "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
    assert sha(SOURCE) == "e9875c1aa522d86d2f0a96df11cec89429800d437964ab5d58d4f0a9de3173a9"
    assert sha(PACKETS) == "3b36b814be46163af665e1f57ce8942c2fb9774b8f8d6320523e58340f8b241d"
    assert sha(LABELS) == "6f4037988ccfb6799943385dfcd9a6d056690f5ef6ed0494a6720462dc60bfac"
    with tarfile.open(SOURCE) as tar:
        source_receipt = json.loads(tar.extractfile("publish/receipt.json").read())
        raw = tar.extractfile("publish/source.rgb").read()
    assert (
        source_receipt["source_id"] == "match209_000"
        and hashlib.sha256(raw).hexdigest() == source_receipt["source"]["rgb_sha256"]
    )
    frames = np.frombuffer(raw, np.uint8).reshape(64, 720, 1280, 3)
    with tarfile.open(PACKETS) as tar:
        report = json.loads(tar.extractfile("publish/result.json").read())
        entry = next(x for x in report["results"] if x["mode"] == "plain")
        stream = tar.extractfile("publish/" + entry["file"]).read()
    assert (
        hashlib.sha256(stream).hexdigest() == entry["sha256"]
        and abs(entry["kbps"] - 61.353125) < 1e-8
    )
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == report["decoder"]["sha256"]
    yuv_path = OUT / (a.mode + "-base.yuv")
    assert not yuv_path.exists()
    d = tracked_patch.decode_native(
        stream,
        yuv_path,
        {"width": 320, "height": 180, "fps": [25, 1], "frames": 64},
        decoder["path"],
    )
    assert d["decoded_yuv_sha256"] == entry["decoded_sha256"]
    base = np.memmap(yuv_path, np.uint8, "r", shape=(64, 270, 320))
    rows = json.loads(LABELS.read_text())["sources"]["match209"]["rows"]
    truth = {}
    for row in rows:
        if row["objects"]:
            truth[row["tick"]] = [
                [
                    max(0, math.floor(x * 2 / 3)),
                    max(0, math.floor(y * 2 / 3)),
                    min(1280, math.ceil((x + w) * 2 / 3)),
                    min(720, math.ceil((y + h) * 2 / 3)),
                ]
                for x, y, w, h in [o["bbox_xywh"] for o in row["objects"]]
            ]
    net = YOLO(str(MODEL))
    assert net.names[38] == "tennis racket"
    h = hashlib.sha256()
    all_ticks = []
    scored = []
    infer_seconds = 0.0
    for t in range(n):
        source = frames[t]
        then = time.monotonic()
        pred = predictions(net, source)
        infer_seconds += time.monotonic() - then
        canvas = cv2.resize(
            tracked_patch.yuv_to_rgb(base[t], 320, 180, False),
            (1280, 720),
            interpolation=cv2.INTER_CUBIC,
        )
        positions = []
        for obj in pred:
            x0, y0, x1, y1 = obj["box"]
            box = [
                max(0, math.floor(x0)),
                max(0, math.floor(y0)),
                min(1280, math.ceil(x1)),
                min(720, math.ceil(y1)),
            ]
            x0, y0, x1, y1 = box
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1] = source[y0:y1, x0:x1]
                positions.append(box)
        h.update(canvas.tobytes())
        all_ticks.append({"tick": t, "predictions": pred, "crop_positions": positions})
        if t in truth:
            then = time.monotonic()
            out = predictions(net, canvas)
            infer_seconds += time.monotonic() - then
            scored.append(
                {
                    "tick": t,
                    "truth_boxes": truth[t],
                    "source_match": match(truth[t], [v["box"] for v in pred]),
                    "oracle_match": match(truth[t], [v["box"] for v in out]),
                    "oracle_predictions": out,
                }
            )

    def hits(key):
        return sum(v >= 0.5 for row in scored for v in row[key]["iou"])

    result = {
        "mode": a.mode,
        "frames": n,
        "model_sha256": sha(MODEL),
        "source_sha256": sha(SOURCE),
        "packet_sha256": entry["sha256"],
        "base_kbps": entry["kbps"],
        "decoded_yuv_sha256": d["decoded_yuv_sha256"],
        "labels_sha256": sha(LABELS),
        "oracle_rgb_sha256": h.hexdigest(),
        "all_ticks": all_ticks,
        "labeled_results": scored,
        "source_hits_iou50": hits("source_match"),
        "oracle_hits_iou50": hits("oracle_match"),
        "placements": sum(len(row["crop_positions"]) for row in all_ticks),
        "unlabeled_ticks": sum(t not in truth for t in range(n)),
        "infer_seconds": infer_seconds,
        "elapsed_seconds": time.monotonic() - start,
    }
    result["gate"] = (
        result["elapsed_seconds"] + 1.5 * infer_seconds * (64 / n) + 120 < 800
        if a.mode == "smoke"
        else len(scored) == 5 and hits("source_match") >= 4 and hits("oracle_match") >= 4
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "mode",
                    "frames",
                    "source_hits_iou50",
                    "oracle_hits_iou50",
                    "placements",
                    "infer_seconds",
                    "elapsed_seconds",
                    "gate",
                )
            }
        )
    )
    assert result["gate"]


if __name__ == "__main__":
    main()
