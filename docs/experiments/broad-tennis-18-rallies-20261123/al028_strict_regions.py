"""Score a physical CRF54 display on the existing accepted Al028 source labels."""

import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "al028-neural-display-20261122"))
from score_bodies import EXPECTED, REVIEW, WEIGHTS, load_rgb, quality, sha, tensor  # noqa: E402


def anchors():
    family = HERE.parent / "reviewed-foreground-20261122"
    plain = []
    neural = []
    for f in sorted(family.glob("ordinary-*.json")):
        plain += json.loads(f.read_text())["points"]
    for p in json.loads((family / "joint-neural-native.json").read_text())["points"]:
        if p["mode"] == "DCVC-UF" and p["bytes"] <= 24480:
            neural.append(p)
    roi = json.loads((HERE.parent / "two-rally-roi-20261122/foreground-scores.json").read_text())[
        "al028"
    ]["arms"]
    banks = {"ordinary": plain, "ROI": roi, "neural_producer": neural}
    selected = {}
    for name, rows in banks.items():
        available = [r for r in rows if r["bytes"] <= 24480]
        r = min(available, key=lambda row: (row["foreground"], row["bytes"]))
        selected[name] = dict(
            bytes=r["bytes"],
            kbps=r["bytes"] / 1200,
            wire_sha256=r["stream_sha256"],
            candidates=len(available),
            players=r.get("paired_player", r.get("players")),
            ball=r.get("paired_ball", r.get("ball")),
            foreground=r["foreground"],
            archived_only=r.get("archived_only", False),
        )
    return selected


def main(results, samples, out):
    labels = Path(
        "/Users/manu/Datasets/pointstream-submission-20261118/al028-author-marks-7bb50c395742/annotations.json"
    )
    assert sha(labels) == "098fc0b387cbc895fb64c543670023f87a26da155a68383a0b80d540ce6ab3f9"
    accepted = json.loads(labels.read_text())
    source_meta = json.loads((REVIEW / "review.json").read_text())
    assert (
        sha(REVIEW / "review.json")
        == "508617ef01170813202eec89a154b0b61890756427a143ac2f2b0e429b00f419"
    )
    assert accepted["review_sha256"] == source_meta["review_sha256"]
    r = json.loads(results.read_text())
    assert r["frames"] == 240 and r["wire_bytes"] == 24257 and sha(samples) == r["samples_sha256"]
    assert r["wire_sha256"] == "1093859b4bcf14d0ebe1f1abe08fdf518acd21b763ed2b23327b2826c9f47020"
    assert sha(WEIGHTS) == EXPECTED
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    t0 = time.process_time()
    net = quality.load_dists(WEIGHTS, "cpu")
    rows = []
    with tarfile.open(samples) as archive:
        for frame, mark in zip(source_meta["frames"], accepted["frames"], strict=True):
            tick = mark["tick"]
            assert tick == frame["frame_index"]
            ref = REVIEW / frame["source"]
            assert sha(ref) == frame["source_sha256"] == mark["source_png_sha256"]
            original = load_rgb(ref)
            regular = archive.extractfile(f"{tick:03d}.png")
            assert regular is not None
            b = regular.read()
            assert (
                hashlib.sha256(b).hexdigest()
                == r["private_review_png_sha256"][f"{tick:03d}/joint-54"]
            )
            display = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )

            def measure(y, x, m):
                target = tensor(original[y, x])
                return quality.masked_dists(
                    net, tensor(display[y, x]), target, torch.from_numpy(m.copy())[None, None]
                ).item()

            people = {}
            for p, v in zip(frame["players"], mark["players"], strict=True):
                assert p["id"] == v["id"] and v["status"] == "accept"
                path = REVIEW / p["proposal"]
                assert sha(path) == p["sha256"] == v["proposal_sha256"]
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
                y, x = np.nonzero(mask)
                yy = slice(max(0, y.min() - 32), min(1080, y.max() + 33))
                xx = slice(max(0, x.min() - 32), min(1920, x.max() + 33))
                people[p["id"]] = measure(yy, xx, mask[yy, xx])
            ball = None
            if mark["ball"]["status"] == "visible":
                x, y = mark["ball"]["center_xy"]
                ball = measure(
                    slice(y - 32, y + 32), slice(x - 32, x + 32), np.ones((64, 64), bool)
                )
            rows.append(dict(tick=tick, people=people, player=mean(people.values()), ball=ball))
    common = [v for v in rows if v["ball"] is not None]
    assert len(rows) == 12 and len(common) == 11
    player = mean(q["player"] for q in common)
    ball_mean = mean(q["ball"] for q in common)
    record = dict(
        kbps=24257 / 1200,
        bytes=24257,
        wire_sha256=r["wire_sha256"],
        full_display_sha256=r["rgb_stream_sha256"]["54"],
        appeared_cached_bytes=157517,
        reviewed_body_all_24=mean(q["player"] for q in rows),
        paired_player=player,
        paired_ball=ball_mean,
        foreground=(player + ball_mean) / 2,
        rows=rows,
        anchors=anchors(),
        source_labels_sha256=sha(labels),
        native_receipt_sha256=sha(results),
        weights_sha256=EXPECTED,
        script_sha256=sha(Path(__file__)),
        cpu_seconds=time.process_time() - t0,
        scope="11 common times for the equal-class mean, 24 accepted silhouettes; single development rally, warm cache, actual complete packet.",
    )
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in record.items()
            if k in ("kbps", "paired_player", "paired_ball", "foreground", "anchors")
        }
    )


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
