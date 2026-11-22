"""Annotate a fixed source-only view with the selected and other original mask tracks."""

import argparse
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from experiments.tennis.k4b_pose import player_masks
from experiments.tennis.player_requalification import sha256
from src.segmentation.masks import ClipMasks


def run(a):
    cv2.setNumThreads(1)
    assert sha256(a.source) == a.source_sha and sha256(a.features) == a.feature_sha
    a.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(a.features) as tar:
        prefix = f"publish/test__{a.event}@120/"
        root = json.load(tar.extractfile(prefix + "features.json"))
        admission = json.load(tar.extractfile("publish/admission.json"))
        data = tar.extractfile(prefix + "sam-masks.rle").read()
    assert admission["source_sha256"] == a.source_sha
    path = a.output / "native-masks.rle"
    path.write_bytes(data)
    clip = ClipMasks.load(path)
    owners = sorted(root["features"])
    chosen = admission["ids"]
    rows = []
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    sheet = []
    for t in range(240):
        ok, im = cap.read()
        assert ok and im.shape[:2] == (1080, 1920)
        if t not in (0, 47, 95, 143, 191, 239):
            continue
        masks = player_masks(clip.frames[t], root["aliases"], owners)
        v = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        boxes = {}
        for p, m in masks.items():
            name = str(int(p.rsplit(":", 1)[-1]))
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            x, y, w, h = cv2.boundingRect(np.column_stack([xs, ys]).astype(np.int32))
            boxes[p] = [x, y, w, h]
            col = (20, 245, 245) if p in chosen else (235, 204, 76)
            cv2.rectangle(v, (x, y), (x + w, y + h), col, 2 if p in chosen else 1)
            cv2.putText(v, name, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)
        cv2.putText(
            v,
            f"Source tick {t}",
            (16, 515),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        sheet.append(v)
        rows.append(dict(tick=t, boxes=boxes))
    cap.release()
    assert len(sheet) == 6
    thumb = np.vstack([np.hstack(sheet[i : i + 2]) for i in range(0, 6, 2)])
    assert cv2.imwrite(str(a.output / "role-source.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (a.output / "role-source.json").write_text(
        json.dumps(
            dict(
                event=a.event,
                selected=chosen,
                source_sha256=a.source_sha,
                features_sha256=a.feature_sha,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(chosen, rows[0]["boxes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("source", "features", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("event", "source-sha", "feature-sha"):
        p.add_argument("--" + k, required=True)
    run(p.parse_args())
