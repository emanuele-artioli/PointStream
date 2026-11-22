"""Check physical frame alignment across two RGB decoders of the authenticated source."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(a):
    cv2.setNumThreads(1)
    meta = json.loads((a.review / "review.json").read_text())
    rendered = json.loads((a.receiver / "result.json").read_text())
    assert sha(a.receiver / "result.json") == a.receiver_sha
    assert (
        sha(a.source)
        == meta["source_video_sha256"]
        == "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
    )
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 25
    records = []
    for r in meta["frames"]:
        t = r["frame_index"]
        key = f"{t:03d}/source"
        ref = a.review / r["source"]
        png = a.receiver / "private-review" / (key + ".png")
        assert (
            sha(ref) == r["source_sha256"]
            and sha(png) == rendered["private_review_png_sha256"][key]
        )
        mac = cv2.imread(str(ref))
        linux = cv2.imread(str(png))
        assert mac.shape == linux.shape == (1080, 1920, 3)
        delta = linux.astype(np.int16) - mac.astype(np.int16)
        # Adjacent-frame alternatives are evaluated after removing the common
        # channel offset. The mobile player regions give this check sensitivity.
        region = np.zeros(mac.shape[:2], bool)
        for p in r["players"]:
            path = a.review / p["proposal"]
            assert sha(path) == p["sha256"]
            region |= cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, t - 1)
        errors = []
        for tt in (t - 1, t, t + 1):
            ok, frame = cap.read()
            assert ok
            if tt == t:
                assert np.array_equal(frame, mac)
            dif = linux[region].astype(np.float32) - frame[region]
            errors.append(float(np.abs(dif - np.median(dif, axis=0)).mean()))
        maximum = int(np.abs(delta).max())
        records.append(
            dict(
                tick=t,
                review_png_sha256=sha(ref),
                linux_png_sha256=sha(png),
                max_abs_channel_difference=maximum,
                mean_absolute_channel_difference=float(np.abs(delta).mean()),
                ordered_neighbour_errors=errors,
            )
        )
        if maximum > 4 or errors.index(min(errors)) != 1:
            raise ValueError(f"Unresolved native frame alignment at {t}: {records[-1]}")
    assert len(records) == 12
    out = dict(
        source_sha256=meta["source_video_sha256"],
        source_review_sha256=meta["review_sha256"],
        receiver_sha256=a.receiver_sha,
        source_decoder=cv2.__version__,
        rows=records,
    )
    with a.output.open("x") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "review", "receiver", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("receiver_sha")
    run(p.parse_args())
