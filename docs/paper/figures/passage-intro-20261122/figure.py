"""Render the measured-passage overview from private checked source derivatives."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--private-dir", type=Path, required=True)
a = p.parse_args()
receipt = json.loads((HERE / "source-frame-receipt.json").read_text())
assert (
    receipt["packet_sha256"] == "c53685733be3ccfce2673bc49e4ad603c998b9eccd24a5e12d6acb70782ada00"
)
rows = receipt["rows"]
for row in rows:
    path = a.private_dir / (row["name"] + ".png")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["derivative_sha256"]
assert (
    hashlib.sha256((a.private_dir / "ball-inset.png").read_bytes()).hexdigest()
    == rows[0]["ball_inset_sha256"]
)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig = plt.figure(figsize=(12, 5.5), facecolor="white")
fig.text(
    0.06,
    0.94,
    "One measured broadcast, two coding paths",
    fontsize=20,
    weight="bold",
    color="#172b3a",
)
fig.text(
    0.06,
    0.885,
    "226.1 s · 11,305 frames · 50 fps · retrospective routing of the indexed passage",
    fontsize=11,
    color="#40535f",
)
xs = [0.06, 0.37, 0.68]
titles = [
    "First rally · source frame",
    "Intervening footage · source frame",
    "Return rally · source frame",
]
for i, (x, row, title) in enumerate(zip(xs, rows, titles, strict=True)):
    fig.text(x, 0.818, title, fontsize=10, weight="bold", color="#172b3a")
    ax = fig.add_axes([x, 0.45, 0.275, 0.34])
    ax.imshow(Image.open(a.private_dir / (row["name"] + ".png")))
    ax.axis("off")
    if i == 0:
        ax.add_patch(Circle((945 / 3, 477 / 3), 11, fill=False, edgecolor="#ffcf35", linewidth=2.5))
    fig.text(
        x,
        0.422,
        f"Tick {row['native_tick']:,} · passage {row['passage_seconds']:.2f} s",
        fontsize=9,
        color="#40535f",
    )
texts = [
    (
        "Object-coded · 4.8 s",
        "Initialize player references; send\nplayer signals, base and ball data.",
        "273,927 B including references",
    ),
    (
        "Conventional fallback · 216.5 s",
        "Encode every intervening frame\nas conventional AV1 video.",
        "540,921 B",
    ),
    (
        "Object-coded · 4.8 s",
        "Reuse player references; send new\nconditioning, base and ball data.",
        "24,723 B",
    ),
]
for x, (h, b, c) in zip(xs, texts, strict=True):
    fig.text(x, 0.36, h, fontsize=10, weight="bold", color="#08718a")
    fig.text(x, 0.285, b, fontsize=9.5, linespacing=1.35)
    fig.text(x, 0.235, c, fontsize=9.5, weight="bold")
fig.text(
    0.06,
    0.14,
    "Rally starts are 221.3 s apart. Section index: 1,484 B. Total delivery: 841,055 B (29.759 kbps).",
    fontsize=10,
)
fig.text(
    0.06,
    0.09,
    "All pictures are source frames, not reconstructed output. The middle frame is a close-up within the paid fallback.",
    fontsize=9,
    color="#40535f",
)
# The marked source location is an independent human-reviewed ball label, not a detection outcome.
inset = fig.add_axes([0.271, 0.615, 0.06, 0.14])
inset.imshow(Image.open(a.private_dir / "ball-inset.png"))
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_edgecolor("#ffcf35")
    spine.set_linewidth(2)
fig.savefig(a.private_dir / "measured-passage-intro.pdf", bbox_inches="tight")
fig.savefig(a.private_dir / "measured-passage-intro.png", dpi=180, bbox_inches="tight")
