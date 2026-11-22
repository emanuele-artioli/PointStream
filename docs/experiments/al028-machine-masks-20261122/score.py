"""Compare paired masks in native source coordinates with fixed person matching."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from statistics import mean
import tarfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent


def decode_png(raw):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid mask")
    return image > 0


def matched(truth, predictions):
    scores = np.zeros((2, len(predictions)), np.float64)
    for i, ref in enumerate(truth):
        for j, mask in enumerate(predictions):
            scores[i, j] = np.count_nonzero(ref & mask) / np.count_nonzero(ref | mask)
    rows, cols = linear_sum_assignment(-scores)
    found = {i: float(scores[i, j]) for i, j in zip(rows, cols, strict=True) if scores[i, j] > 0}
    return [found.get(i, 0.0) for i in range(2)], len(predictions) - len(found)


def score(predictions, labels, display, binding_path):
    binding = json.loads(binding_path.read_text())
    assert hashlib.sha256(labels.read_bytes()).hexdigest() == binding["normalized_labels_sha256"]
    with tarfile.open(predictions) as archive:
        report = json.load(archive.extractfile("publish/predictions.json"))
    assert report["frames"] == binding["archive_sha256"] and set(report["rows"]) == set(
        binding["keys"]
    )
    numbered = {(arm, tick): key for key, (arm, tick) in binding["keys"].items()}
    rows = []
    for frame in json.loads(labels.read_text())["frames"]:
        tick = frame["tick"]
        truth = []
        for p in frame["players"]:
            record = p["mask"]
            raw = base64.b64decode(record["png_base64"], validate=True)
            mask = decode_png(raw)
            assert hashlib.sha256(raw).hexdigest() == record["sha256"]
            assert (
                p["status"] == "accept"
                and mask.sum() == record["foreground_pixels"]
                and mask.shape == (1080, 1920)
            )
            truth.append(mask)
        for arm in sorted({mode for mode, _ in numbered} | {"receiver_mask"}):
            predicted = []
            if arm == "receiver_mask":
                for i, expected in enumerate(binding["receiver_masks"][str(tick)], 1):
                    raw = (
                        display / "private-review" / f"{tick:03d}/receiver-mask-{i}.png"
                    ).read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == expected
                    predicted.append(decode_png(raw))
            else:
                for p in report["rows"][numbered[arm, tick]]:
                    x, y, w, h = p["rect"]
                    canvas = np.zeros((1080, 1920), bool)
                    canvas[y : y + h, x : x + w] = decode_png(
                        base64.b64decode(p["png"], validate=True)
                    )
                    assert canvas.sum() == p["pixels"]
                    predicted.append(canvas)
            ious, extras = matched(truth, predicted)
            rows.append(dict(arm=arm, tick=tick, iou=ious, extra_predictions=extras))
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        cells = [v for r in rows if r["arm"] == arm for v in r["iou"]]
        assert len(cells) == 24
        summary[arm] = dict(
            mean_iou=mean(cells),
            at_least_half=sum(s >= 0.5 for s in cells),
            persons=len(cells),
            extra_predictions=sum(r["extra_predictions"] for r in rows if r["arm"] == arm),
        )
    return dict(
        source_labels=binding["normalized_labels_sha256"],
        blind_archive=binding["archive_sha256"],
        predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        worker={k: v for k, v in report.items() if k != "rows"},
        summary=summary,
        rows=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("predictions", "labels", "display", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--binding", type=Path, default=HERE / "binding.json")
    a = parser.parse_args()
    result = score(a.predictions, a.labels, a.display, a.binding)
    with a.output.open("x") as writer:
        writer.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
