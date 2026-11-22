"""Build a source-only player review on the fixed official-label quantiles."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image

from experiments.tennis.k4b_pose import player_masks
from src.segmentation.masks import ClipMasks, decompress

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


EVENTS = {
    "match102": (
        "sh102",
        "Sinner–Shelton, Shanghai 2023",
        "994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee",
    ),
    "match127": (
        "sy127",
        "Shapovalov–Safiullin, Sydney 2022",
        "7c7b678cf671e22092efe5284a19cd113bca49ee125af558d844c886fc31c19d",
    ),
}


def build(args):
    short, title, feature_sha = EVENTS[args.event]
    data = json.loads((HERE.parent / "cohort-expansion-20261122/staged.json").read_text())
    record = next(r for r in data["selection"] if r["match"] == args.event)
    assert digest(args.source.read_bytes()) == record["source"]["sha256"]
    assert digest(args.ball.read_bytes()) == record["labels"]["sha256"]
    with args.ball.open() as file:
        records = {
            int(r["Frame"]): int(r["Visibility"])
            for r in csv.DictReader(file)
            if 0 <= int(r["Frame"]) < 240 and int(r["Visibility"]) in (0, 1)
        }
    population = sorted(records)
    ticks = [population[i] for i in np.rint(np.linspace(0, len(population) - 1, 12)).astype(int)]
    assert len(set(ticks)) == 12
    assert digest(args.features.read_bytes()) == feature_sha
    with tarfile.open(args.features) as a:
        prefix = f"publish/test__{args.event}@120/"
        ids = json.load(a.extractfile("publish/admission.json"))["ids"]
        aliases = json.load(a.extractfile(prefix + "features.json"))["aliases"]
        raw = a.extractfile(prefix + "sam-masks.rle").read()
        clip = ClipMasks.from_doc(json.loads(decompress(raw)))
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.output / "assets"
    assets.mkdir()
    capture = cv2.VideoCapture(str(args.source))
    frames = []
    for tick in range(max(ticks) + 1):
        ok, bgr = capture.read()
        assert ok and bgr.shape == (1080, 1920, 3)
        if tick not in ticks:
            continue
        source = assets / f"source-{tick:03}.png"
        assert cv2.imwrite(str(source), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        active = player_masks(clip.frames[tick], aliases, ids)
        players = []
        for number, identifier in enumerate(ids):
            mask = cv2.resize(
                active[identifier].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST
            )
            prop = assets / f"proposal-{tick:03}-{number}.png"
            Image.fromarray(mask * 255).save(prop)
            overlay = assets / f"overlay-{tick:03}-{number}.png"
            rgba = np.zeros((1080, 1920, 4), np.uint8)
            rgba[:, :, :3] = [(0, 132, 199), (192, 38, 211)][number]
            rgba[:, :, 3] = mask * 85
            Image.fromarray(rgba).save(overlay)
            players.append(
                dict(
                    id=identifier,
                    proposal="assets/" + prop.name,
                    sha256=digest(prop.read_bytes()),
                    overlay="assets/" + overlay.name,
                    overlay_sha256=digest(overlay.read_bytes()),
                    pixels=int(mask.sum()),
                )
            )
        frames.append(
            dict(
                id=f"{short}-tick-{tick:03}",
                frame_index=tick,
                source="assets/" + source.name,
                source_sha256=digest(source.read_bytes()),
                players=players,
            )
        )
    capture.release()
    review = dict(
        version=1,
        title=title + " · source player masks",
        storage_prefix=f"pointstream-{short}-source-review-",
        download_name=f"{short}-source-annotation-decisions.json",
        review_id=f"{short}-source-only-20261122",
        source_group=args.event + "/000",
        source_video_sha256=record["source"]["sha256"],
        source_video_fps=list(map(int, record["video"]["r_frame_rate"].split("/"))),
        window_start_in_clip=0,
        matte_transport_sha256=digest(args.features.read_bytes()),
        matte_member_sha256=digest(raw),
        official_ball_sha256=record["labels"]["sha256"],
        frames=frames,
    )
    canonical = json.dumps(review, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    review["review_sha256"] = digest(canonical)
    (args.output / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    template = (
        HERE.parent / "joint-replication-feasibility-20261122/review_template.html"
    ).read_text()
    (args.output / "index.html").write_text(
        template.replace("__REVIEW_JSON__", json.dumps(review, separators=(",", ":")))
    )
    print(
        json.dumps(
            dict(
                sha=review["review_sha256"],
                ticks=ticks,
                official_visible=sum(records[t] for t in ticks),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "ball", "features", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--event", choices=EVENTS, default="match102")
    build(parser.parse_args())
