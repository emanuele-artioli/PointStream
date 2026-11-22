"""Score independent human marks on the three complete streams at 20.4 kbps."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

import cv2
import numpy as np
import torch

from score_bodies import EXPECTED, REVIEW, WEIGHTS, load_rgb, quality, sha, tensor

ARMS = {"ordinary": "av1_320_q52_75", "neural": "dcvc_q24", "joint": "joint-20"}


def regions(label, source_row):
    for person, proposal in zip(label["players"], source_row["players"], strict=True):
        assert person["id"] == proposal["id"] and person["proposal_sha256"] == proposal["sha256"]
        if person["status"] in ("missing", "unsure"):
            assert person["mask"] is None
            continue
        assert person["status"] in ("accept", "edit")
        record = person["mask"]
        blob = base64.b64decode(record["png_base64"], validate=True)
        assert hashlib.sha256(blob).hexdigest() == record["sha256"]
        mask = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        yy, xx = np.nonzero(mask)
        assert mask.shape == (1080, 1920) and len(xx) == record["foreground_pixels"] > 0
        y0, y1 = max(0, int(yy.min()) - 32), min(1080, int(yy.max()) + 33)
        x0, x1 = max(0, int(xx.min()) - 32), min(1920, int(xx.max()) + 33)
        yield person["id"], (x0, y0, x1, y1), mask[y0:y1, x0:x1]
    ball = label["ball"]
    if ball["status"] == "visible":
        x, y = ball["center_xy"]
        assert 0 <= x < 1920 and 0 <= y < 1080
        yield "ball", (x - 32, y - 32, x + 32, y + 32), np.ones((64, 64), bool)
    else:
        assert ball["status"] in ("missing", "unsure") and ball["center_xy"] is None


def crop(rgb, rect):
    x0, y0, x1, y1 = rect
    region = rgb[max(0, y0) : min(1080, y1), max(0, x0) : min(1920, x1)]
    return cv2.copyMakeBorder(
        region, max(0, -y0), max(0, y1 - 1080), max(0, -x0), max(0, x1 - 1920), cv2.BORDER_REPLICATE
    )


def summarize(rows, decisions):
    persons = ("archived-track-0", "archived-track-1")
    common = [
        r["tick"]
        for r in decisions
        if r["ball"] == "visible" and all(r[p] in ("accept", "edit") for p in persons)
    ]
    result = {}
    for arm in ARMS:
        scores = {(r["tick"], r["region"]): r["dists"] for r in rows if r["arm"] == arm}

        def avg(values):
            return mean(values) if values else None

        bodies = [v for (_, k), v in scores.items() if k in persons]
        balls = [v for (_, k), v in scores.items() if k == "ball"]
        body = avg([mean([scores[t, p] for p in persons]) for t in common])
        ball = avg([scores[t, "ball"] for t in common])
        result[arm] = dict(
            body_observations=len(bodies),
            body_dists=avg(bodies),
            visible_balls=len(balls),
            ball_dists=avg(balls),
            paired_body_dists=body,
            paired_ball_dists=ball,
            paired_foreground_dists=(body + ball) / 2 if common else None,
        )
    return {"jointly_scorable_ticks": common, "decisions": decisions, "arms": result}


def run(a):
    assert sha(a.labels) == a.labels_sha and sha(a.receivers / "result.json") == a.receipt_sha
    labels = json.loads(a.labels.read_text())
    source = json.loads((REVIEW / "review.json").read_text())
    assert (
        labels["schema"] == "pointstream-al028-human-review-v1"
        and labels["review_sha256"]
        == source["review_sha256"]
        == "fb41c943575b35e6ecec36916b8cfab41b8023744614a98681b7cce884681eee"
    )
    assert (labels["width"], labels["height"], labels["fps"]) == (1920, 1080, [25, 1]) and labels[
        "source_sha256"
    ] == source["source_video_sha256"]
    receiver = json.loads((a.receivers / "result.json").read_text())
    hashes = receiver["private_review_png_sha256"]
    assert (
        receiver["private_review_ticks"]
        == [r["tick"] for r in labels["frames"]]
        == list(range(10, 240, 20))
    )
    assert sha(WEIGHTS) == EXPECTED
    start = time.process_time()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    net = quality.load_dists(WEIGHTS, "cpu")
    scores = []
    decisions = []
    forecast = None
    for frame, src in zip(labels["frames"], source["frames"], strict=True):
        t = frame["tick"]
        assert t == src["frame_index"] and frame["source_png_sha256"] == src[
            "source_sha256"
        ] == sha(REVIEW / src["source"])
        ref = load_rgb(REVIEW / src["source"])
        rgb = {}
        for arm, name in {**ARMS, "self": "source"}.items():
            key = f"{t:03d}/{name}"
            path = a.receivers / "private-review" / (key + ".png")
            assert hashes[key] == sha(path)
            rgb[arm] = load_rgb(path)
        assert np.array_equal(ref, rgb["self"])
        decisions.append(
            dict(
                tick=t,
                ball=frame["ball"]["status"],
                **{p["id"]: p["status"] for p in frame["players"]},
            )
        )
        supports = list(regions(frame, src))
        if not supports:
            continue
        pilot = time.process_time()
        for name, rect, mask in supports:
            target = tensor(crop(ref, rect))
            support = torch.from_numpy(np.ascontiguousarray(mask))[None, None]
            for arm, img in rgb.items():
                if arm == "self" and forecast is not None:
                    continue
                d = quality.masked_dists(net, tensor(crop(img, rect)), target, support).item()
                assert np.isfinite(d)
                if arm == "self":
                    assert abs(d) < 1e-5
                else:
                    scores.append(dict(tick=t, region=name, arm=arm, dists=d))
        if forecast is None:
            forecast = (time.process_time() - pilot) * 36 / len(supports) * 1.5 + 60
            assert forecast < 600
        assert time.process_time() - start < 600
    report = summarize(scores, decisions)
    rates = json.loads((Path(__file__).parent / "common-budget.json").read_text())
    report.update(
        rows=scores,
        review_sha256=labels["review_sha256"],
        labels_sha256=a.labels_sha,
        reconstruction_sha256=a.receipt_sha,
        model=quality.dists_record(WEIGHTS),
        elapsed_cpu=time.process_time() - start,
        forecast_cpu=forecast,
        actual_kbps={
            "ordinary": rates["selected"]["ordinary"]["kbps"],
            "joint": rates["selected"]["joint"]["kbps"],
            "neural": rates["reference_rate_kbps"],
        },
        joint_startup_bytes=rates["cache_bytes"],
        edge_rule="Replicate source edge pixels for a marked ball within 32 pixels of the border, identically for all arms.",
    )
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "receivers", "output"):
        p.add_argument(name, type=Path)
    for name in ("labels-sha", "receipt-sha"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())
