"""Recover the prespecified source player supports on every native target tick."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from experiments.tennis.k4b_pose import player_masks
from experiments.tennis.player_requalification import array_sha
from src.segmentation.masks import ClipMasks


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(a):
    receipt = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "six-match-source-panel-20261122/alcaraz031-native-crop.json"
        ).read_text()
    )
    assert sha(a.features) == receipt["prior_feature_archive_sha256"]
    prefix = "publish/test__alcaraz031@120/"
    with tarfile.open(a.features) as f:
        cfg = json.load(f.extractfile(prefix + "features.json"))
        rle = f.extractfile(prefix + "sam-masks.rle").read()
    a.out.mkdir(exist_ok=False)
    (a.out / "source.rle").write_bytes(rle)
    clip = ClipMasks.load(a.out / "source.rle")
    ids = ["person:000000000006", "person:000000000002"]
    assert set(ids) == set(receipt["selected_ids"])
    rows = []
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert sha(a.source) == receipt["source_sha256"] and cap.get(cv2.CAP_PROP_FPS) == 25
    checks = []
    for tick in range(240):
        ok, image = cap.read()
        assert ok and image.shape == (1080, 1920, 3)
        checks.append(array_sha(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        people = player_masks(clip.frames[tick], cfg["aliases"], ids)
        for i, n in enumerate(ids):
            m = np.repeat(np.repeat(people[n], 2, axis=0), 2, axis=1)
            assert m.shape == (1080, 1920) and m.any()
            x, y, w, h = cv2.boundingRect(m.astype("uint8"))
            x0, y0 = max(0, x - 32), max(0, y - 32)
            x1, y1 = min(1920, x + w + 32), min(1080, y + h + 32)
            p = a.out / f"{tick:03d}-p{i}.png"
            assert cv2.imwrite(str(p), m[y0:y1, x0:x1].astype("uint8") * 255)
            rows.append(
                dict(
                    tick=tick,
                    object_index=i,
                    role=["far", "near"][i],
                    unit=n,
                    mask=p.name,
                    sha256=sha(p),
                    xyxy=[x0, y0, x1, y1],
                    area=int(m.sum()),
                )
            )
    cap.release()
    r = dict(
        ticks=list(range(240)),
        rows=rows,
        source_start=0,
        nonempty=len(rows),
        source_sha256=receipt["source_sha256"],
        source_rgb_hashes=checks,
        source_mask_sha256=hashlib.sha256(rle).hexdigest(),
        feature_archive_sha256=sha(a.features),
        method="Both source-SAM players at every native tick, tight mask plus 32 display pixels; model-derived rather than manually drawn.",
    )
    (a.out / "supports.json").write_text(json.dumps(r, indent=2) + "\n")
    print("native SHA-bound player masks", len(rows), "ticks", len(checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("features", "source", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    run(p.parse_args())
