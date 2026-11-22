"""Score newly marked ball crops on saved Al028 displays and join accepted body masks."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "al028-neural-display-20261122"
sys.path.insert(0, str(PRIOR))
from score_bodies import EXPECTED, REVIEW, WEIGHTS, load_rgb, quality, sha, tensor  # noqa: E402
from score_review import crop  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

LABELS = "098fc0b387cbc895fb64c543670023f87a26da155a68383a0b80d540ce6ab3f9"
PRIMARY = "2eb230df276f52780fd34a11331743c2895c3fe94bde05c6645dc0165ee33559"


def read(path):
    return json.loads(path.read_text())


def displayed(raw, width, height):
    return cv2.resize(
        yuv_to_rgb(raw, width, height, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
    )


def score(config, human, model, anchor, originals):
    by_person = {(r["tick"], r["person"]): r["dists"] for r in config["body_rows"]}
    assert len(by_person) == 24
    values = []
    parity = []
    for i, label in enumerate(human["frames"]):
        tick = label["tick"]
        if label["ball"]["status"] != "visible":
            continue
        x, y = label["ball"]["center_xy"]
        rect = (x - 32, y - 32, x + 32, y + 32)
        pixel = crop(config["load"](i, tick), rect)
        test = tensor(pixel)
        good = tensor(crop(originals[tick], rect))
        value = quality.masked_dists(
            model, test, good, torch.ones_like(good[:, :1], dtype=torch.bool)
        ).item()
        assert np.isfinite(value)
        if config.get("linux_control"):
            checked = crop(config["linux_control"](i, tick), rect)
            expected = quality.masked_dists(
                model, tensor(checked), good, torch.ones_like(good[:, :1], dtype=torch.bool)
            ).item()
            assert abs(expected - anchor["ordinary", tick]) < 1e-6
            parity.append(
                dict(
                    tick=tick,
                    max_channel_difference=int(
                        np.max(np.abs(checked.astype(np.int16) - pixel.astype(np.int16)))
                    ),
                    native_dists=value,
                    linux_dists=expected,
                )
            )
        if config.get("primary_arm"):
            assert abs(value - anchor[config["primary_arm"], tick]) < 1e-6
        values.append(
            {
                "tick": tick,
                "ball": value,
                "player": mean(by_person[tick, p["id"]] for p in label["players"]),
            }
        )
    player = mean(r["player"] for r in values)
    ball = mean(r["ball"] for r in values)
    return {k: v for k, v in config.items() if k not in {"load", "body_rows", "linux_control"}} | {
        "rows": values,
        "cross_host_control": parity,
        "body_all_24": mean(by_person.values()),
        "paired_player": player,
        "paired_ball": ball,
        "foreground": (player + ball) / 2,
    }


def ordinary(args):
    dataset = read(PRIOR / "ordinary-three-size-curves.json")["points"]
    registry = {}
    for path in args.ordinary_receipts:
        for row in read(path)["rows"]:
            registry[row["sha256"]] = Path(row["path"])
    assert len(dataset) == 52 and all(r["stream_sha256"] in registry for r in dataset)
    primary_sha = read(PRIOR / "common-budget.json")["selected"]["ordinary"]["stream_sha256"]
    for row in dataset[args.start : args.end]:
        path = registry[row["stream_sha256"]]
        raw = path / "decoded.yuv"
        ivf = path / "stream.ivf"
        assert (
            sha(ivf) == row["stream_sha256"]
            and sha(raw) == row["raw_sha256"]
            and ivf.stat().st_size == row["bytes"]
        )
        w, h = row["width"], row["height"]
        video = np.memmap(raw, np.uint8, "r", shape=(240, h * 3 // 2, w))

        def linux_control(i, t):
            base = Path("/private/tmp/ps-al028-review-validated")
            assert sha(base / "result.json") == PRIMARY
            key = f"{t:03d}/av1_320_q52_75"
            file = base / "private-review" / (key + ".png")
            assert sha(file) == read(base / "result.json")["private_review_png_sha256"][key]
            return load_rgb(file)

        yield dict(
            mode="AV1",
            bytes=row["bytes"],
            kbps=row["kbps"],
            width=w,
            height=h,
            crf=row["crf"],
            stream_sha256=row["stream_sha256"],
            raw_sha256=row["raw_sha256"],
            body_rows=row["rows"],
            linux_control=linux_control if row["stream_sha256"] == primary_sha else None,
            load=lambda i, t: displayed(video[t], w, h),
        )


def others(args):
    for folder, points in [
        (args.base, "three-joint-rungs.json"),
        (args.high, "upper-joint-rung.json"),
    ]:
        record = read(PRIOR / points)
        assert sha(folder / "result.json") == record["reconstruction_sha256"]
        result = read(folder / "result.json")
        for arm in sorted({r["arm"] for r in record["rows"]}):
            if arm in ("joint-11", "joint-21"):
                price = read(
                    HERE.parent / f"joint-replication-feasibility-20261122/al028-{arm}kbps.json"
                )
                size = price["ideal_prewarmed_bytes"]
                digest = price["joint_sha256"]
            else:
                bill = "common-budget.json" if arm == "joint-20" else "common-budget-q36.json"
                price = read(PRIOR / bill)["selected"]["joint"]
                size = price["bytes"]
                digest = price["stream_sha256"]

            def load(i, t):
                key = f"{t:03d}/{arm}"
                path = folder / "private-review" / (key + ".png")
                assert sha(path) == result["private_review_png_sha256"][key]
                return load_rgb(path)

            bodies = [
                {"tick": r["tick"], "person": b["person"], "dists": b["dists"]}
                for r in record["rows"]
                if r["arm"] == arm
                for b in r["body"]
            ]
            yield dict(
                mode="PointStream",
                setting=arm,
                bytes=size,
                kbps=size / 1200,
                stream_sha256=digest,
                body_rows=bodies,
                load=load,
                primary_arm="joint" if arm == "joint-20" else None,
            )

    scores = read(HERE.parent / "al028-q36-export-20261122/body-scores.json")
    for qp, folder, origin, name in [
        (36, args.q36, scores, "q36"),
        (
            18,
            args.q18,
            read(HERE.parent / "al028-lower-neural-20261122/q18-body-scores.json"),
            "candidate",
        ),
    ]:
        meta = read(folder / "review12.json")
        yuv = folder / "review12.yuv"
        assert sha(yuv) == meta["review12_sha256"] == origin[f"{name}_review12_sha256"]
        arr = np.memmap(yuv, np.uint8, "r", shape=(12, 816, 960))

        def load(i, t):
            assert (
                meta["ticks"][i] == t
                and hashlib.sha256(arr[i]).hexdigest() == meta["frame_sha256"][i]
            )
            return displayed(arr[i], 960, 544)

        yield dict(
            mode="DCVC-UF",
            setting=f"Q{qp}",
            archived_only=qp == 18,
            bytes=round(meta["kbps"] * 1200),
            kbps=meta["kbps"],
            stream_sha256=meta["packet_sha256"],
            body_rows=[r for r in origin["rows"] if r["arm"] == name],
            load=load,
        )

    rate = read(PRIOR / "common-budget.json")["reference_rate_kbps"]
    assert sha(args.base / "result.json") == PRIMARY
    q24 = read(args.base / "result.json")["private_review_png_sha256"]

    def load24(i, t):
        key = f"{t:03d}/dcvc_q24"
        path = args.base / "private-review" / (key + ".png")
        assert sha(path) == q24[key]
        return load_rgb(path)

    yield dict(
        mode="DCVC-UF",
        setting="Q24",
        archived_only=False,
        bytes=round(rate * 1200),
        kbps=rate,
        stream_sha256="08ca4898595d5e78c5f634559acd1fc13af369e2d07c3d4170c1cc764633d214",
        body_rows=[
            {"person": r["region"], "tick": r["tick"], "dists": r["dists"]}
            for r in read(HERE / "matched-three.json")["rows"]
            if r["arm"] == "neural" and r["region"] != "ball"
        ],
        load=load24,
        primary_arm="neural",
    )


def main(args):
    assert sha(args.labels) == LABELS and sha(WEIGHTS) == EXPECTED
    assert 0 < args.end - args.start <= 12
    source = read(REVIEW / "review.json")
    labels = read(args.labels)
    originals = {}
    assert labels["review_sha256"] == source["review_sha256"]
    for annotated, frame in zip(labels["frames"], source["frames"], strict=True):
        tick = annotated["tick"]
        assert (
            tick == frame["frame_index"]
            and sha(REVIEW / frame["source"])
            == frame["source_sha256"]
            == annotated["source_png_sha256"]
        )
        for human, proposal in zip(annotated["players"], frame["players"], strict=True):
            assert (
                human["status"] == "accept"
                and human["id"] == proposal["id"]
                and human["proposal_sha256"]
                == proposal["sha256"]
                == sha(REVIEW / proposal["proposal"])
            )
        originals[tick] = load_rgb(REVIEW / frame["source"])
    prior = read(HERE / "matched-three.json")
    assert prior["labels_sha256"] == LABELS and prior["reconstruction_sha256"] == PRIMARY
    anchor = {(r["arm"], r["tick"]): r["dists"] for r in prior["rows"] if r["region"] == "ball"}
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = quality.load_dists(WEIGHTS, "cpu")
    configs = ordinary(args) if args.kind == "ordinary" else others(args)
    count = args.end - args.start if args.kind == "ordinary" else 7
    start = time.process_time()
    results = []
    for cfg in configs:
        results.append(score(cfg, labels, model, anchor, originals))
        if len(results) == 1:
            forecast = (time.process_time() - start) * count * 1.5 + 30
            assert forecast < 600
        assert time.process_time() - start < 600
    assert len(results) == count
    with args.out.open("x") as f:
        json.dump(
            dict(
                input_labels_sha256=LABELS,
                backbone=EXPECTED,
                accepted_model_masks=24,
                cost_forecast=forecast,
                cpu_seconds=time.process_time() - start,
                points=results,
            ),
            f,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["ordinary", "others"])
    p.add_argument("labels", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--ordinary-receipts", nargs="+", type=Path)
    for name in ("base", "high", "q36", "q18"):
        p.add_argument("--" + name, type=Path)
    main(p.parse_args())
