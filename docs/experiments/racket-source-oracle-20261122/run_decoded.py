"""Bounded CPU-only racket source oracle on three authenticated source archives."""

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
from src.codecs import svtav1
from src.pointstream import tracked_patch
import torch
from ultralytics import YOLO

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
MODEL = Path("/home/itec/emanuele/Models/YOLO/yolo11x.pt")
LABELS = ROOT / "tennis/equipment-window-20261122/source-6f403798.json"
ARCHIVES = {
    "match209": (
        "20261115T081356Z-4b32fe68",
        "e9875c1aa522d86d2f0a96df11cec89429800d437964ab5d58d4f0a9de3173a9",
    ),
    "match235": (
        "20261115T082635Z-4c193c5e",
        "506485f5c85ef7ec4ecacecf920a0c0039174db5d1717bab41b0aff2a4de5311",
    ),
    "match269": (
        "20261115T082631Z-8331f54a",
        "a7e1ca9f15e9cba300e2cbeed976f30499211018b441d11aa5b8c6c2cadb5aab",
    ),
}
PACKETS = {
    "match209": (
        "20261121T223136Z-e9219942",
        "3b36b814be46163af665e1f57ce8942c2fb9774b8f8d6320523e58340f8b241d",
    ),
    "match235": (
        "20261121T222531Z-e7aae2b8",
        "59d374d4296dc7c467df42ac330f53aa8e22773cdf255a4a6a23733db2264e97",
    ),
    "match269": (
        "20261121T223136Z-d6a41fc2",
        "ae4275d1ac54d423d1fa162fe3b9e15677bb0d95f45565a95ac6fc637bdf4944",
    ),
}
OUT = Path("/local/users/emanuele/pointstream/audits/racket-source-oracle-decoded-20261122")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iou(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    a = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (a.mode + ".json")
    assert not out.exists()
    assert sha(MODEL) == "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
    assert sha(LABELS) == "6f4037988ccfb6799943385dfcd9a6d056690f5ef6ed0494a6720462dc60bfac"
    labels = json.loads(LABELS.read_text())["sources"]
    start = time.monotonic()
    net = YOLO(str(MODEL))
    rows = []
    elapsed = 0.0
    assert net.names[38] == "tennis racket"
    for name, (job, expected) in ARCHIVES.items():
        archive = ROOT / "jobs/fleet/inbox" / job / "full/published.tar"
        assert sha(archive) == expected
        with tarfile.open(archive) as tar:
            rec = json.loads(tar.extractfile("publish/receipt.json").read())
            raw = tar.extractfile("publish/source.rgb").read()
        assert (
            hashlib.sha256(raw).hexdigest() == rec["source"]["rgb_sha256"]
            and rec["source_id"] == name + "_000"
        )
        frames = np.frombuffer(raw, np.uint8).reshape(labels[name]["frames"], 720, 1280, 3)
        packet_job, packet_sha = PACKETS[name]
        packet_archive = ROOT / "jobs/fleet/inbox" / packet_job / "full/published.tar"
        assert sha(packet_archive) == packet_sha
        with tarfile.open(packet_archive) as pt:
            packet_report = json.loads(pt.extractfile("publish/result.json").read())
            entry = next(v for v in packet_report["results"] if v["mode"] == "plain")
            encoded = pt.extractfile("publish/" + entry["file"]).read()
        assert hashlib.sha256(encoded).hexdigest() == entry["sha256"]
        decoder = svtav1.tool("dav1d")
        assert decoder["sha256"] == packet_report["decoder"]["sha256"]
        path = OUT / (name + "-" + a.mode + ".yuv")
        meta = {
            "width": 320,
            "height": 180,
            "fps": [packet_report["fps"], 1],
            "frames": packet_report["frames"],
        }
        decoded_info = tracked_patch.decode_native(encoded, path, meta, decoder["path"])
        assert decoded_info["decoded_yuv_sha256"] == entry["decoded_sha256"]
        yuv = np.memmap(path, np.uint8, "r", shape=(packet_report["frames"], 270, 320))
        rs = [r for r in labels[name]["rows"] if r["objects"]]
        rs = rs[:1] if a.mode == "smoke" else rs
        for row in rs:
            tick = row["tick"]
            start_infer = time.monotonic()
            display = cv2.resize(
                tracked_patch.yuv_to_rgb(yuv[tick], 320, 180, False),
                (1280, 720),
                interpolation=cv2.INTER_CUBIC,
            )
            pred = net.predict(
                source=display[:, :, ::-1],
                device="cpu",
                imgsz=640,
                conf=0.25,
                iou=0.7,
                classes=[38],
                verbose=False,
            )[0]
            elapsed += time.monotonic() - start_infer
            boxes = pred.boxes.xyxy.cpu().numpy().tolist()
            conf = pred.boxes.conf.cpu().numpy().tolist()
            assert all(math.isfinite(float(x)) for b in boxes for x in b)
            gt = []
            for obj in row["objects"]:
                x, y, w, h = obj["bbox_xywh"]
                gt.append(
                    [
                        max(0, math.floor(x * 2 / 3)),
                        max(0, math.floor(y * 2 / 3)),
                        min(1280, math.ceil((x + w) * 2 / 3)),
                        min(720, math.ceil((y + h) * 2 / 3)),
                    ]
                )
            # Exhaustive assignment: tiny number of official objects; cap candidate count conservatively.
            assert len(gt) <= 2 and len(boxes) <= 100
            options = [-1] + list(range(len(boxes)))
            best = None
            for ass in itertools.product(options, repeat=len(gt)):
                chosen = [j for j in ass if j >= 0]
                if len(chosen) != len(set(chosen)):
                    continue
                vals = [iou(gt[i], boxes[j]) if j >= 0 else 0.0 for i, j in enumerate(ass)]
                score = (sum(v >= 0.5 for v in vals), sum(vals))
                if best is None or score > best[0]:
                    best = (score, ass, vals)
            rows.append(
                {
                    "source": name,
                    "tick": tick,
                    "truth_boxes": gt,
                    "predicted_boxes": boxes,
                    "confidence": conf,
                    "assigned_prediction_indices": best[1],
                    "assigned_iou": best[2],
                }
            )
        del raw, frames
    result = {
        "mode": a.mode,
        "device": "cpu",
        "model_sha256": sha(MODEL),
        "labels_sha256": sha(LABELS),
        "elapsed_seconds": time.monotonic() - start,
        "inference_seconds": elapsed,
        "rows": rows,
    }
    if a.mode == "smoke":
        result["forecast_seconds"] = result["elapsed_seconds"] + 1.5 * elapsed * 5 + 120
        result["gate"] = len(rows) == 3 and result["forecast_seconds"] < 600
    else:
        result["gate"] = len(rows) == 15 and sum(len(r["truth_boxes"]) for r in rows) == 16
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": a.mode,
                "seconds": result["elapsed_seconds"],
                "inference_seconds": elapsed,
                "gate": result["gate"],
                "forecast_seconds": result.get("forecast_seconds"),
            }
        )
    )
    assert result["gate"]


if __name__ == "__main__":
    main()
