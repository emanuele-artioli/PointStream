"""Render a private source-aligned ball comparison from authenticated real decodes."""

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from src.pointstream.tracked_patch import yuv_to_rgb

ROOT = Path(__file__).resolve().parents[4]
ITEM = Path(__file__).resolve().parent
DATA = Path("/Users/manu/Datasets/pointstream-submission-20261118")
SOURCE = DATA / "ps-development-source-refs-20261118-native/source-only-review.zip"
BUNDLE_SHA = "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
LABEL_SHA = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
LABELS = DATA / "foreground-human-reviewed-20261118" / (LABEL_SHA + ".json")
PLAIN = DATA / "ps-native-ctl-20261118-full/plain-48/decoded.yuv"
PLAIN_SHA = "4f549befc1445ae4baca3a8da55c822261a78b08f98a8b212fc702f81998499b"
CANDIDATE = Path("/private/tmp/ps-matched-ball-20261122-frames")
TICKS = (10, 90, 170)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixels(raw):
    return np.array(Image.open(BytesIO(raw)).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    assert sha(SOURCE) == BUNDLE_SHA and sha(LABELS) == LABEL_SHA and sha(PLAIN) == PLAIN_SHA
    labels = {row["tick"]: row for row in json.loads(LABELS.read_text())["frames"]}
    receipt_path = ROOT / "docs/experiments/matched-ball-rate-20261122/render-result.json"
    cost = json.loads(receipt_path.read_text())
    input_path = ROOT / "docs/experiments/matched-ball-rate-20261122/crop-inputs.json"
    inputs = json.loads(input_path.read_text())
    assert inputs["source"] == "b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d"
    assert inputs["labels"] == LABEL_SHA and cost["source_bundle_sha256"] == BUNDLE_SHA
    assert cost["candidate_kbps"] == 303.71 and cost["baseline_kbps"] == 303.425
    delivered = {row["tick"]: row for row in cost["saved"]}
    base = np.memmap(PLAIN, np.uint8, "r", shape=(240, 816, 960))
    images = {}
    checks = []
    with zipfile.ZipFile(SOURCE) as bundle:
        for tick in TICKS:
            source_png = bundle.read(f"frame-{tick:03d}/source.png")
            assert hashlib.sha256(source_png).hexdigest() == labels[tick]["source_png_sha256"]
            original = pixels(source_png)
            candidate_file = CANDIDATE / f"{tick:03d}.png"
            assert sha(candidate_file) == delivered[tick]["sha256"]
            candidate = pixels(candidate_file.read_bytes())
            assert hashlib.sha256(candidate).hexdigest() == inputs["arms"]["matched"][str(tick)]
            plain = cv2.resize(
                yuv_to_rgb(base[tick], 960, 544, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
            x, y = labels[tick]["ball"]["center_xy"]
            assert labels[tick]["ball"]["status"] == "visible" and 32 <= x < 1888 and 32 <= y < 1048
            crop = np.s_[y - 32 : y + 32, x - 32 : x + 32]
            images[tick] = (original, plain, candidate)
            origin = delivered[tick]["ball_origin"]
            covered = (
                origin is not None
                and origin[0] <= x < origin[0] + 32
                and origin[1] <= y < origin[1] + 32
            )
            checks.append(
                dict(
                    tick=tick,
                    source_frame_png_sha256=labels[tick]["source_png_sha256"],
                    plain_rgb_sha256=hashlib.sha256(plain).hexdigest(),
                    candidate_png_sha256=sha(candidate_file),
                    candidate_rgb_sha256=hashlib.sha256(candidate).hexdigest(),
                    crop_bounds=[x - 32, y - 32, x + 32, y + 32],
                    ball_center=[x, y],
                    tile_origin=origin,
                    reviewed_ball_center_covered=bool(covered),
                    crop_rgb_sha256=[
                        hashlib.sha256(image[crop].tobytes()).hexdigest() for image in images[tick]
                    ],
                )
            )
    assert checks[0]["tile_origin"] is None and all(
        row["reviewed_ball_center_covered"] for row in checks[1:]
    )
    fig = plt.figure(figsize=(12.5, 6.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.16)
    frame_ax = fig.add_subplot(outer[0, 0])
    frame_ax.imshow(images[90][0], interpolation="nearest")
    frame_ax.axis("off")
    frame_ax.set_title("Source rally frame at 1.80 s", fontsize=12, fontweight="bold", pad=10)
    frame_ax.text(
        0.5,
        -0.055,
        "Sender information includes players, ball location and court appearance.",
        transform=frame_ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#42535d",
    )
    for index, person in enumerate(labels[90]["players"], start=1):
        mask = (
            np.array(
                Image.open(BytesIO(base64.b64decode(person["mask"]["png_base64"]))).convert("L")
            )
            > 0
        )
        ys, xs = np.nonzero(mask)
        x, y, w, h = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        frame_ax.add_patch(
            Rectangle(
                (x - 10, y - 10), w + 20, h + 20, fill=False, edgecolor="#00e0e0", linewidth=1.7
            )
        )
        frame_ax.text(
            x - 10,
            y - 22,
            f"Player {index}",
            fontsize=9,
            color="black",
            bbox=dict(facecolor="#00e0e0", edgecolor="none", pad=2),
        )
    ball_x, ball_y = labels[90]["ball"]["center_xy"]
    frame_ax.add_patch(
        Rectangle(
            (ball_x - 32, ball_y - 32), 64, 64, fill=False, edgecolor="#ffde4a", linewidth=1.7
        )
    )
    frame_ax.annotate(
        "Ball crop",
        xy=(ball_x, ball_y),
        xytext=(ball_x + 130, ball_y - 90),
        color="#ffde4a",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#ffde4a", lw=1.5),
        bbox=dict(facecolor="black", alpha=0.65, pad=3, edgecolor="none"),
    )
    right = outer[0, 1].subgridspec(3, 3, wspace=0.06, hspace=0.48)
    headings = ["Source", "AV1", "AV1 + ball tiles"]
    for row, tick in enumerate(TICKS):
        center = labels[tick]["ball"]["center_xy"]
        x, y = center
        for col in range(3):
            ax = fig.add_subplot(right[row, col])
            ax.imshow(images[tick][col][y - 32 : y + 32, x - 32 : x + 32], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#d2d8db")
            if row == 0:
                ax.set_title(headings[col], fontsize=10, pad=13, fontweight="bold")
            if col == 0:
                ax.text(
                    -0.12,
                    0.5,
                    f"{tick / 50:.2f} s",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
            if col == 2:
                status = "No tile" if checks[row]["tile_origin"] is None else "Tile covers ball"
                ax.text(
                    0.5,
                    -0.13,
                    status,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#657887" if row == 0 else "#007d80",
                )
    fig.suptitle(
        "Preserving the ball at the same delivery rate",
        x=0.035,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.887,
        "Plain AV1 303.425 kbps · AV1 with tracked tiles 303.710 kbps · all transmitted bytes included",
        fontsize=10,
        color="#42535d",
    )
    fig.text(
        0.585,
        0.13,
        "64 × 64 native-pixel crops, enlarged with nearest-neighbor display",
        fontsize=9,
        color="#42535d",
    )
    fig.text(
        0.035,
        0.065,
        "Fixed reviewed ticks 10, 90 and 170 · one development rally · tiles transmitted at 195 of 240 frames",
        fontsize=9,
        color="#42535d",
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.79, bottom=0.20)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.output / f"rally-ball-inset-private.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    result = dict(
        role="Private source-aligned qualitative illustration from fixed reviewed ticks",
        source_bundle_sha256=BUNDLE_SHA,
        labels_sha256=LABEL_SHA,
        plain_decoded_yuv_sha256=PLAIN_SHA,
        candidate_receipt_sha256=sha(receipt_path),
        crop_inventory_sha256=sha(input_path),
        script_sha256=sha(Path(__file__)),
        plain_kbps=cost["baseline_kbps"],
        candidate_kbps=cost["candidate_kbps"],
        source_ticks=list(TICKS),
        images=checks,
        opencv=cv2.__version__,
        numpy=np.__version__,
        player_boxes="bounding boxes of reviewed body masks expanded by 10 native pixels",
        ball_support="64px centered crop from human-reviewed source coordinates",
        figure_states="source views and measured decodes; player regions outlined; no synthetic keypoints",
    )
    (args.output / "rally-ball-inset-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output / "rally-ball-inset-private.pdf")


if __name__ == "__main__":
    main()
