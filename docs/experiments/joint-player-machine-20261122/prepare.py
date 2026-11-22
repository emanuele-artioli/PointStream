"""Bind a 48-image blind segmentation comparison to the scored native displays."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "joint-trajectory-20261122"))
import motion  # noqa: E402


def prepare(args):
    cv2.setNumThreads(1)
    args.output.mkdir()
    crops = motion.crops
    joint = json.loads((motion.HERE / "result.json").read_text())["streams"]["20"]
    binding = json.loads((HERE.parent / "joint-machine-20261122/binding.json").read_text())
    control = json.loads((args.controls / "20-240/result.json").read_text())["selected"]["plain"]
    ordinary_yuv = Path(control["decoded"])
    assert crops.file_sha(ordinary_yuv) == binding["ordinary"]["yuv"]
    ordinary = np.memmap(ordinary_yuv, np.uint8, "r", shape=(240, 204, 240))
    audit = json.loads((motion.HERE / "dcvc-q6-human-regions.json").read_text())
    npath = args.neural / "review12.yuv"
    assert crops.file_sha(npath) == audit["frame_slice_sha256"] and audit["four_passes_match"]
    neural = np.memmap(npath, np.uint8, "r", shape=(12, 816, 960))
    labels = {f["tick"]: f for f in json.loads(crops.LABELS.read_text())["frames"]}
    assert (
        crops.file_sha(crops.LABELS) == binding["labels_sha256"]
        and sorted(labels) == audit["ticks"]
    )
    assert (
        crops.file_sha(crops.BUNDLE)
        == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    )
    digest = hashlib.sha256()
    keys = {}
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(args.joint),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    size = 1080 * 1920 * 3
    with (
        subprocess.Popen(cmd, stdout=subprocess.PIPE) as decoder,
        zipfile.ZipFile(crops.BUNDLE) as sources,
        zipfile.ZipFile(args.output / "frames.zip", "x") as archive,
    ):
        for tick in range(240):
            raw = decoder.stdout.read(size)
            assert len(raw) == size
            digest.update(raw)
            if tick not in labels:
                continue
            src = sources.read(f"frame-{tick:03d}/source.png")
            assert hashlib.sha256(src).hexdigest() == labels[tick]["source_png_sha256"]
            idx = audit["ticks"].index(tick)
            assert (
                hashlib.sha256(neural[idx]).hexdigest()
                == audit["producer_twelve"]["ticks"][idx]["sha256"]
            )
            images = {"source": src}
            displays = {
                "ordinary": motion.view(ordinary[tick], 240, 136),
                "neural": motion.view(neural[idx], 960, 544),
                "joint": np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3),
            }
            for arm, rgb in displays.items():
                ok, data = cv2.imencode(".png", rgb[:, :, ::-1])
                assert ok
                images[arm] = data.tobytes()
            for arm, data in images.items():
                identity = hashlib.sha256(data).hexdigest()
                assert identity not in keys
                info = zipfile.ZipInfo(identity + ".png", date_time=(2026, 1, 1, 0, 0, 0))
                archive.writestr(info, data)
                keys[identity] = (arm, tick)
        assert decoder.stdout.read(1) == b"" and decoder.wait(timeout=20) == 0
    assert digest.hexdigest() == joint["all_240_rgb_sha256"] == binding["joint_display"]["full"]
    assert len(keys) == 48
    (args.output / "key.json").write_text(json.dumps(keys, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frames": crops.file_sha(args.output / "frames.zip"),
                "smoke": {arm: k for k, (arm, t) in keys.items() if t == audit["ticks"][0]},
                "full_joint_rgb": digest.hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("joint", "controls", "neural", "output"):
        cli.add_argument(name, type=Path)
    prepare(cli.parse_args())
