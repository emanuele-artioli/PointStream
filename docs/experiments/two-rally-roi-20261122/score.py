"""Score the newly encoded ROI displays on the original author-reviewed regions."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import zipfile

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "al028-neural-display-20261122"))
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
from crops_support import supports  # noqa: E402
from score_bodies import EXPECTED, REVIEW, WEIGHTS, load_rgb, quality, sha, tensor  # noqa: E402
from score_review import crop, regions  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

TMP = Path("/private/tmp")


def source_regions(event):
    rows = {}
    if event == "nadal":
        root = Path("/Users/manu/Datasets/pointstream-submission-20261118")
        label = (
            root
            / "foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json"
        )
        assert sha(label) == label.stem
        d = json.loads(label.read_text())
        archive = root / "ps-development-source-refs-20261118-native/source-only-review.zip"
        assert sha(archive) == d["bundle_sha256"]
        with zipfile.ZipFile(archive) as z:
            for row in d["frames"]:
                t = row["tick"]
                blob = z.read(f"frame-{t:03d}/source.png")
                assert hashlib.sha256(blob).hexdigest() == row["source_png_sha256"]
                rgb = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                region = [
                    (key, (x.start, y.start, x.stop, y.stop), mask > 0)
                    for key, y, x, mask in supports(row)
                ]
                rows[t] = (rgb, region)
    else:
        label = TMP / "ps-al028-backward-check-1041/annotations.json"
        assert sha(label) == "098fc0b387cbc895fb64c543670023f87a26da155a68383a0b80d540ce6ab3f9"
        d = json.loads(label.read_text())
        review = json.loads((REVIEW / "review.json").read_text())
        assert d["review_sha256"] == review["review_sha256"]
        for label_row, s in zip(d["frames"], review["frames"], strict=True):
            p = REVIEW / s["source"]
            assert s["source_sha256"] == label_row["source_png_sha256"] == sha(p)
            rows[label_row["tick"]] = (load_rgb(p), list(regions(label_row, s)))
    return rows


def evaluate(source, candidate, model):
    w = candidate["width"]
    h = candidate["height"]
    p = Path(candidate["decoded"])
    assert sha(p) == candidate["decoded_sha256"] and p.stat().st_size == 240 * h * w * 3 // 2
    bank = np.memmap(p, np.uint8, "r", shape=(240, h * 3 // 2, w))
    scores = []
    for t, (original, supports_on_tick) in source.items():
        img = cv2.resize(
            yuv_to_rgb(bank[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
        )
        per = {}
        for identifier, rect, mask in supports_on_tick:
            a = tensor(crop(img, rect))
            b = tensor(crop(original, rect))
            m = torch.from_numpy(np.ascontiguousarray(mask))[None, None]
            with torch.inference_mode():
                per[identifier] = quality.masked_dists(model, a, b, m).item()
        players = [v for k, v in per.items() if k != "ball"]
        scores.append(dict(tick=t, regions=per, players=mean(players), ball=per.get("ball")))
    paired = [r for r in scores if r["ball"] is not None]
    player = mean(r["players"] for r in paired)
    ball = mean(r["ball"] for r in paired)
    return dict(
        scores=scores,
        support=len(paired),
        players=player,
        ball=ball,
        foreground=(player + ball) / 2,
    )


def run(a):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert sha(WEIGHTS) == EXPECTED
    model = quality.load_dists(WEIGHTS, "cpu")
    orig = json.loads((HERE.parent / "reviewed-foreground-20261122/figure-data.json").read_text())
    result = {}
    for ev, name, w, h, control in [
        (
            "nadal",
            "Nadal–Kyrgios",
            240,
            136,
            TMP / "ps-joint-trajectory-controls-20261122/20-240/plain-63-240",
        ),
        ("al028", "Alcaraz–Hurkacz", 640, 384, TMP / "ps-nov22-al028-640-bill-24494/q62.25"),
    ]:
        source = source_regions(ev)
        baseline = orig["per_shot"][name]["AV1"]
        assert (
            abs(
                (control / "stream.ivf").stat().st_size
                * 8
                * (50 if ev == "nadal" else 25)
                / 240
                / 1000
                - baseline["kbps"]
            )
            < 1e-7
        )
        row = dict(
            width=w,
            height=h,
            decoded=str(control / "decoded.yuv"),
            decoded_sha256=sha(control / "decoded.yuv"),
        )
        check = evaluate(source, row, model)
        assert abs(check["foreground"] - baseline["foreground"]) < 1e-6, (
            name,
            check["foreground"],
            baseline["foreground"],
        )
        summary = dict(anchor_check=check["foreground"], arms=[])
        for p in sorted(a.bank.glob(f"native-{ev}-*/result.json")):
            d = json.loads(p.read_text())
            s = d["selected"]
            r = dict(
                width=d["width"],
                height=d["height"],
                offset=d["offset"],
                trial_sha256=sha(p),
                tested_bytes=[x["bytes"] for x in d["trials"].values()],
            )
            if s and s["effective"]:
                r.update(
                    bytes=s["bytes"],
                    stream_sha256=s["sha256"],
                    decoded_sha256=s["decoded_sha256"],
                    changed_luma=s["changed_luma"],
                    **evaluate(source, dict(r, decoded=s["decoded"], decoded_sha256=s["decoded_sha256"]), model),
                )
            summary["arms"].append(r)
        result[ev] = summary
    with a.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bank", type=Path)
    p.add_argument("out", type=Path)
    run(p.parse_args())
