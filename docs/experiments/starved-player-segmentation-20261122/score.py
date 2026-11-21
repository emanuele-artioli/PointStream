"""Match independent decoded-video segmentations to blinded source body labels."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
LABEL_HASH = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"


def decode(png):
    buf = np.frombuffer(base64.b64decode(png, validate=True), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) > 0


def main(predictions, key, labels, output):
    assert hashlib.sha256(labels.read_bytes()).hexdigest() == LABEL_HASH
    source = {row["tick"]: row for row in json.loads(labels.read_text())["frames"]}
    direct = json.loads((HERE.parents[1] / "paper/figures/player-tasks/results.json").read_text())[
        "mask"
    ]["rows"]
    strata = {r["tick"]: "assisted" if r["proposed"] else "unprompted" for r in direct}
    with tarfile.open(predictions) as archive:
        pred = json.load(archive.extractfile("publish/predictions.json"))
    keys = json.loads(key.read_text())
    assert set(keys) == set(pred["rows"]) and len(keys) == 48
    rows = []
    extra = {}
    for digest, (arm, tick) in keys.items():
        persons = source[tick]["players"]
        actual = [decode(p["mask"]["png_base64"]) for p in persons]
        assert len(actual) == 2 and all(a.shape == (1080, 1920) for a in actual)
        targets = [int(np.count_nonzero(a)) for a in actual]
        assert targets == [p["mask"]["foreground_pixels"] for p in persons]
        candidates = pred["rows"][digest]
        costs = np.zeros((2, len(candidates)))
        for col, p in enumerate(candidates):
            x, y, w, h = p["rect"]
            mask = decode(p["png"])
            assert mask.shape == (h, w) and np.count_nonzero(mask) == p["pixels"]
            for i, ref in enumerate(actual):
                overlap = int(np.count_nonzero(ref[y : y + h, x : x + w] & mask))
                costs[i, col] = overlap / (targets[i] + p["pixels"] - overlap)
        indexes = linear_sum_assignment(-costs)
        found = {int(a): float(costs[a, b]) for a, b in zip(*indexes) if costs[a, b] > 0}
        extra.setdefault(arm, {})[tick] = len(candidates) - len(found)
        for i in (0, 1):
            rows.append(
                dict(arm=arm, tick=tick, player=i + 1, stratum=strata[tick], iou=found.get(i, 0.0))
            )
    for r in direct:
        rows.append(
            dict(
                arm="receiver_mask",
                tick=r["tick"],
                player=r["player"],
                stratum=strata[r["tick"]],
                iou=r["iou"],
            )
        )
    means = {}
    for arm in sorted({r["arm"] for r in rows}):
        means[arm] = {}
        for group in ("all", "unprompted", "assisted"):
            cells = [
                r["iou"]
                for r in rows
                if r["arm"] == arm and (group == "all" or r["stratum"] == group)
            ]
            means[arm][group] = dict(
                n=len(cells),
                mean=float(np.mean(cells)),
                recall_iou50=float(np.mean(np.array(cells) >= 0.5)),
            )
    assert len(rows) == 120 and all(
        s["all"]["n"] == 24 and s["unprompted"]["n"] == 6 for s in means.values()
    )
    receipt = dict(
        predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        key_sha256=hashlib.sha256(key.read_bytes()).hexdigest(),
        label_sha256=LABEL_HASH,
        inference={k: v for k, v in pred.items() if k != "rows"},
        means=means,
        extras=extra,
        rows=rows,
    )
    with output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("predictions", "key", "labels", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    main(a.predictions, a.key, a.labels, a.output)
