"""Show the previously selected stroke on actual 21.5 kbps receiver displays."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
SOURCE = Path(
    "/Users/manu/Datasets/pointstream-submission-20261118/ps-native-ctl-20261118-full/source.yuv"
)
TICKS = (78, 86, 94)
CROP = (854, 390, 1324, 1032)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def crop_yuv(frame, w, h):
    img = cv2.resize(yuv_to_rgb(frame, w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC)
    x0, y0, x1, y1 = CROP
    return img[y0:y1, x0:x1]


def main(args):
    cv2.setNumThreads(1)
    args.output.mkdir(exist_ok=False)
    joint = json.loads((HERE / "result.json").read_text())["streams"]["20"]
    check = json.loads((args.replay / "result.json").read_text())["streams"]["20"]
    assert (
        all(
            joint[key] == check[key]
            for key in ("bytes", "cold_bytes", "wire_sha256", "all_240_rgb_sha256")
        )
        and joint["kbps"] == 21.545
    )
    control = json.loads((HERE / "control-scores.json").read_text())
    candidates = [
        p
        for p in control["points"]
        if p["cap"] == 20 and abs(p["kbps"] / joint["kbps"] - 1) <= 0.005
    ]
    best = min(candidates, key=lambda p: (p["players"], p["kbps"]))
    w = best["width"]
    h = {160: 96, 240: 136, 480: 272}[w]
    report = json.loads((args.controls / f"20-{w}" / "result.json").read_text())
    native = next(
        r
        for r in report["selected"].values()
        if r["mode"] == best["mode"] and r["q"] == best["crf"]
    )
    yp = Path(native["decoded"])
    assert sha(yp) == best["decoded_sha256"]
    assert sha(SOURCE) == "6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c"
    source = np.memmap(SOURCE, np.uint8, "r", shape=(240, 816, 960))
    plain = np.memmap(yp, np.uint8, "r", shape=(240, h * 3 // 2, w))
    arms = [
        ("Source", "960 × 544 working video", [crop_yuv(source[t], 960, 544) for t in TICKS]),
        (
            "AV1" if best["mode"] == "plain" else "ROI AV1",
            f"{best['kbps']:.3f} kbps",
            [crop_yuv(plain[t], w, h) for t in TICKS],
        ),
    ]
    if args.neural:
        n = json.loads((args.neural / "stroke.json").read_text())
        assert n["ticks"] == list(TICKS) and n["frames_verified"] == 240
        nn = np.memmap(args.neural / "stroke.yuv", np.uint8, "r", shape=(3, 816, 960))
        assert all(
            hashlib.sha256(frame).hexdigest() == digest
            for frame, digest in zip(nn, n["hashes"], strict=True)
        )
        arms.append(("DCVC-UF", f"{n['kbps']:.3f} kbps", [crop_yuv(f, 960, 544) for f in nn]))
    arms.append(
        (
            "PointStream",
            f"{joint['kbps']:.3f} kbps with valid cache",
            [cv2.imread(str(args.replay / f"serve-20-{t}.png"))[:, :, ::-1] for t in TICKS],
        )
    )
    fig, axes = plt.subplots(3, len(arms), figsize=(2.0 * len(arms) + 1.1, 6.75))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.839, bottom=0.10, wspace=0.02, hspace=0.055)
    for j, (title, rate, imgs) in enumerate(arms):
        axes[0, j].set_title(title + "\n" + rate, fontsize=9, pad=5)
        for i, image in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(image, interpolation="antialiased")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(
                    ("Preparation", "Near contact", "Follow-through")[i]
                    + f"\n{TICKS[i] / 50:.2f} s",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
    fig.suptitle(
        "A player’s serve at 21.5 kbps", fontsize=15, weight="bold", x=0.04, y=0.98, ha="left"
    )
    fig.text(
        0.04,
        0.925,
        "The same source-selected crop and timestamps appear in every column.",
        fontsize=9,
    )
    fig.text(
        0.04,
        0.050,
        f"PointStream requires {joint['cold_kbps']:.3f} kbps when both player references are sent for this 4.8 s rally.",
        fontsize=8,
    )
    fig.text(
        0.04,
        0.022,
        "The generated players retain body shape better than the low-rate AV1 image. Racket grip remains unreliable.",
        fontsize=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.output / f"joint-serve.{ext}", dpi=170)
    plt.close(fig)
    receipt = dict(
        ticks=TICKS,
        crop=CROP,
        anchor=best,
        joint=joint,
        display_sha256=sha(args.output / "joint-serve.png"),
        replay_sha256=sha(args.replay / "result.json"),
        script_sha256=sha(Path(__file__)),
        crop_sha256={
            name: [hashlib.sha256(im.tobytes()).hexdigest() for im in imgs]
            for name, _, imgs in arms
        },
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("replay", type=Path)
    cli.add_argument("controls", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--neural", type=Path)
    main(cli.parse_args())
