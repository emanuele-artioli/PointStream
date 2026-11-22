"""Private source and native receiver crops at three fixed ranks in the registered frame set."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FIGURE_RANKS = (1, 5, 9)


def verified(path, sha):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    return Image.open(path).convert("RGB")


def restrict_box(x, y, w, h, width, height):
    w, h = min(w, width), min(h, height)
    x, y = min(max(0, round(x)), width - w), min(max(0, round(y)), height - h)
    return x, y, x + w, y + h


def panel(frames, source, mark, review, roles, title, labels, tick):
    installed = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = str(installed) if installed.is_file() else "DejaVuSans.ttf"
    head = ImageFont.truetype(font, 32)
    sub = ImageFont.truetype(font, 21)
    small = ImageFont.truetype(font, 18)
    w, h = frames[0].size
    boxes = []
    people = {p["id"]: p for p in source["players"]}
    assert set(people) == set(roles) and len(roles) == 2
    for role in roles:
        p = people[role]
        path = review / p["proposal"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == p["sha256"]
        mask = np.array(Image.open(path).convert("L"))
        assert mask.shape == (h, w)
        yy, xx = np.nonzero(mask)
        boxes.append(
            restrict_box(
                xx.min() - 32, yy.min() - 32, int(np.ptp(xx) + 65), int(np.ptp(yy) + 65), w, h
            )
        )
    boxes.append(restrict_box(int(mark["X"]) - 32, int(mark["Y"]) - 32, 64, 64, w, h))
    result = Image.new("RGB", (1360, 865), "#ffffff")
    d = ImageDraw.Draw(result)
    d.text((38, 23), title, fill="#162735", font=head)
    d.text(
        (40, 73),
        f"Source tick {tick} of 0–239   ·   Same source region across each row   ·   Cached player appearance",
        fill="#41525b",
        font=sub,
    )
    xs = [193, 490, 787, 1084]
    for x, title in zip(
        xs,
        labels,
        strict=True,
    ):
        d.text((x + 104, 128), title, anchor="mm", fill="#142734", font=sub)
    for j, (name, box) in enumerate(zip(("Near player", "Far player", "Ball"), boxes, strict=True)):
        y = 165 + 211 * j
        d.text((38, y + 84), name, fill="#203846", font=sub)
        bw, bh = box[2] - box[0], box[3] - box[1]
        scale = min(209 / bw, 186 / bh)
        size = (round(bw * scale), round(bh * scale))
        for x, im in zip(xs, frames, strict=True):
            tile = im.crop(box).resize(size, Image.Resampling.NEAREST)
            result.paste(tile, (x + (209 - size[0]) // 2, y + (186 - size[1]) // 2))
    d.text(
        (40, 819),
        "Private research comparison. Player crops follow source-model masks; ball crops use independent official locations.",
        font=small,
        fill="#41525b",
    )
    return result


def run(a):
    review = json.loads((a.review / "review.json").read_text())
    external = json.loads(a.comparison.read_text())
    assert external["source"] == review["source_video_sha256"]
    assert hashlib.sha256(a.ball.read_bytes()).hexdigest() == review["official_ball_sha256"]
    measured = external["measured_rungs"][a.crf]
    controls = measured["cases"]["cached"]["controls"]
    selected = [controls[k]["best"] for k in ("ordinary", "neural")] + [measured["joint"]]
    labels = ["Source"] + [
        f"{name} · {row['kbps']:.2f} kbps"
        for name, row in zip(("AV1", "DCVC-UF", "PointStream"), selected, strict=True)
    ]
    keys = [row["sha256"] for row in selected]
    if a.role_binding:
        binding = json.loads(a.role_binding.read_text())
        assert binding["source_sha256"] == external["source"]
        roles = [binding["roles"][role] for role in ("near", "far")]
    else:
        registration = json.loads(
            (HERE.parent / "test-crop-joint-20261122/match102-result.json").read_text()
        )
        assert registration["source_sha256"] == external["source"]
        roles = sorted(
            registration["selected_ids"],
            key=registration["median_height_px"].__getitem__,
            reverse=True,
        )
    title = review["title"].removesuffix(" · source player masks")
    arms = []
    for path, sha in zip((a.ordinary, a.neural, a.joint), keys, strict=True):
        rows = json.loads(path.read_text())
        assert rows["bitstream_sha256"] == sha
        arms.append((path.parent, {r["tick"]: r for r in rows["frames"]}))
    with a.ball.open() as f:
        marks = {int(r["Frame"]): r for r in csv.DictReader(f) if r["Visibility"] == "1"}
    chosen = {r["frame_index"]: r for r in review["frames"]}
    pages = []
    assert len(external["ticks"]) == 12 and sorted(chosen) == external["ticks"]
    for rank in FIGURE_RANKS:
        t = external["ticks"][rank]
        row = chosen[t]
        frames = [verified(a.review / row["source"], row["source_sha256"])]
        for parent, rows in arms:
            sample = rows[t]
            frames.append(verified(parent / sample["path"], sample["png_sha256"]))
        pages.append(panel(frames, row, marks[t], a.review, roles, title, labels, t))
    pages[0].save(a.out, save_all=True, append_images=pages[1:], resolution=145, quality=96)
    pages[1].save(a.out.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "ordinary", "neural", "joint", "ball", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=HERE / "matched-rung-50.json")
    parser.add_argument("--crf", default="50")
    parser.add_argument("--role-binding", type=Path)
    run(parser.parse_args())
