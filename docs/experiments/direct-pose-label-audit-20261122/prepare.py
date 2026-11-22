"""Build a private source-only body-joint annotation page from authenticated frames."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/manu/Datasets/pointstream-submission-20261118")
BUNDLE = ROOT / "ps-development-source-refs-20261118-native/source-only-review.zip"
LABELS = (
    ROOT
    / "foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json"
)
BUNDLE_SHA = "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
LABEL_SHA = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
TICKS = list(range(10, 240, 20))
JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
IDS = ["person:000000000000", "person:000000000001"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(
    output,
    bundle=BUNDLE,
    labels=LABELS,
    expected_bundle=BUNDLE_SHA,
    expected_labels=LABEL_SHA,
    synthetic=False,
):
    if output.exists():
        raise FileExistsError("choose a fresh private output directory")
    if sha(bundle.read_bytes()) != expected_bundle or sha(labels.read_bytes()) != expected_labels:
        raise ValueError("source bundle or review manifest checksum differs")
    meta = json.loads(labels.read_text())
    if (
        meta["width"] != 1920
        or meta["height"] != 1080
        or [r["tick"] for r in meta["frames"]] != TICKS
    ):
        raise ValueError("unexpected source geometry or timeline")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    frames = []
    try:
        with zipfile.ZipFile(bundle) as archive:
            for row in meta["frames"]:
                t = row["tick"]
                if [p["id"] for p in row["players"]] != IDS:
                    raise ValueError("source actor identity/order changed")
                raw = archive.read(f"frame-{t:03d}/source.png")
                if sha(raw) != row["source_png_sha256"]:
                    raise ValueError("source tick hash differs")
                target = output / "frames" / f"{t:03d}.png"
                target.write_bytes(raw)
                with Image.open(target) as image:
                    if image.size != (1920, 1080):
                        raise ValueError("source frame dimensions differ")
                frames.append(
                    {
                        "tick": t,
                        "timestamp_seconds": t / 50,
                        "sha256": sha(raw),
                        "file": f"frames/{t:03d}.png",
                    }
                )
        spec = {
            "schema": "pointstream-body-joints-v1",
            "source_sha256": meta["source_sha256"],
            "source_bundle_sha256": expected_bundle,
            "source_label_manifest_sha256": expected_labels,
            "width": 1920,
            "height": 1080,
            "fps": 50,
            "actors": IDS,
            "joints": JOINTS,
            "frames": frames,
            "synthetic_fixture": synthetic,
        }
        manifest = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        (output / "manifest.json").write_bytes(manifest)
        page = (
            (HERE / "review-template.html")
            .read_text()
            .replace("__MANIFEST_JSON__", manifest.decode())
            .replace("__MANIFEST_SHA__", sha(manifest))
        )
        (output / "review.html").write_text(page)
        return spec
    except Exception:
        shutil.rmtree(output)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
