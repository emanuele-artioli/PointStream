"""Private source and native receiver crops at three fixed ranks in the registered frame set."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TICKS = (31, 112, 198)


def verified(path, sha):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    return Image.open(path).convert("RGB")


def restrict_box(x, y, w, h, width, height):
    w, h = min(w, width), min(h, height)
    x, y = min(max(0, round(x)), width - w), min(max(0, round(y)), height - h)
    return x, y, x + w, y + h


def panel(frames, source, mark, inputs, tick):
    installed = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = str(installed) if installed.is_file() else "DejaVuSans.ttf"
    head = ImageFont.truetype(font, 32)
    sub = ImageFont.truetype(font, 21)
    small = ImageFont.truetype(font, 18)
    w, h = frames[0].size
    boxes = []
    for p in source["players"]:
        path = inputs.review / p["proposal"]
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
    d.text((38, 23), "Sinner–Shelton, Shanghai 2023", fill="#162735", font=head)
    d.text(
        (40, 73),
        f"Native frame {tick} of 240   ·   Same source region across each row   ·   Cached player appearance",
        fill="#41525b",
        font=sub,
    )
    xs = [193, 490, 787, 1084]
    for x, title in zip(
        xs,
        ("Source", "AV1 · 17.71 kbps", "DCVC-UF · 17.41 kbps", "PointStream · 17.78 kbps"),
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
    external = json.loads((Path(__file__).parent / "matched-rung-50.json").read_text())
    measured = external["measured_rungs"]["50"]
    keys = [
        measured["cases"]["cached"]["controls"][k]["best"]["sha256"] for k in ("ordinary", "neural")
    ] + [measured["joint"]["sha256"]]
    arms = []
    for path, sha in zip((a.ordinary, a.neural, a.joint), keys, strict=True):
        rows = json.loads(path.read_text())
        assert rows["bitstream_sha256"] == sha
        arms.append((path.parent, {r["tick"]: r for r in rows["frames"]}))
    with a.ball.open() as f:
        marks = {int(r["Frame"]): r for r in csv.DictReader(f) if r["Visibility"] == "1"}
    chosen = {r["frame_index"]: r for r in review["frames"]}
    pages = []
    for t in TICKS:
        row = chosen[t]
        frames = [verified(a.review / row["source"], row["source_sha256"])]
        for parent, rows in arms:
            sample = rows[t]
            frames.append(verified(parent / sample["path"], sample["png_sha256"]))
        pages.append(panel(frames, row, marks[t], a, t))
    pages[0].save(a.out, save_all=True, append_images=pages[1:], resolution=145, quality=96)
    pages[1].save(a.out.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "ordinary", "neural", "joint", "ball", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
