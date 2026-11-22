"""Measure coverage of official racket landmarks by the two source player masks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from experiments.tennis.k4b_pose import player_masks
from src.segmentation.masks import ClipMasks, decompress


def score(labels, archive, event, limit):
    base = Path(__file__).resolve().parent.parent
    feature = json.loads((base / "test-joint-admission-20261122" / f"{event}-outcome.json").read_text())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == feature["archive_sha256"]
    records = json.loads(labels.read_text())
    source = next(r for r in json.loads((base / "cohort-expansion-20261122/staged.json").read_text())["selection"] if r["match"] == event)
    with tarfile.open(archive) as ar:
        prefix = f"publish/test__{event}@120/"
        def get(name):
            return ar.extractfile(prefix + name).read()
        window = json.loads(get("window.json"))
        assert window["sha256"] == source["source"]["sha256"] and (window["start"], window["frames"]) == (0, 240)
        selected = json.loads(ar.extractfile("publish/admission.json").read())["ids"]
        assert selected == feature["full"]["ids"] and len(set(selected)) == 2
        aliases = json.loads(get("features.json"))["aliases"]
        clip = ClipMasks.from_doc(json.loads(decompress(get("sam-masks.rle"))))
    assert len(clip.frames) == 240
    members = sorted(k for k in records if k.startswith(f"tennis/all/{event}/racket/000/"))
    if limit:
        members = members[:limit]
    yy, xx = np.ogrid[-5:6, -5:6]
    kernel = ((xx * xx + yy * yy) <= 25).astype(np.uint8)
    rows = []
    for name in members:
        tick = int(Path(name).stem)
        assert 0 <= tick < 240
        masks = player_masks(clip.frames[tick], aliases, selected)
        union = np.zeros((540,960), np.uint8)
        for m in masks.values():
            assert m.shape == union.shape
            union |= m.astype(np.uint8)
        native = cv2.resize(union, (1920,1080), interpolation=cv2.INTER_NEAREST)
        nearby = cv2.dilate(native, kernel)
        for obj, truth in enumerate(records[name]["objects"]):
            assert len(truth["keypoints"]) == 5
            for part, (x,y,v) in zip(("top","bottom","handle","left","right"),truth["keypoints"],strict=True):
                row = dict(tick=tick, object_index=obj, part=part, visible=v > 0)
                if v > 0:
                    xi,yi = int(np.floor(x + .5)), int(np.floor(y + .5))
                    assert 0 <= xi < 1920 and 0 <= yi < 1080
                    row.update(exact=bool(native[yi,xi]), within_five=bool(nearby[yi,xi]))
                rows.append(row)
    summary = {}
    for group in ("head","handle"):
        subset = [r for r in rows if (r["part"]=="handle") == (group=="handle")]
        c = Counter(k for r in subset for k in ("visible","exact","within_five") if r.get(k))
        summary[group] = dict(c,all_landmarks=len(subset))
    return dict(event=event, source_sha256=window["sha256"],feature_sha256=digest,
                labels={name:records[name]["sha256"] for name in members},
                selected_players=selected,summary=summary,rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event")
    parser.add_argument("labels",type=Path)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--first-tick",action="store_true")
    args = parser.parse_args()
    result = score(args.labels,args.archive,args.event,args.first_tick)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(args.event,result["summary"])
