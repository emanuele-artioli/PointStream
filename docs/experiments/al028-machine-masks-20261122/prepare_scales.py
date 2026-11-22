"""Blind and bind the highest affordable actual AV1 stream at each coded width."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import cv2
import numpy as np
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent


def main(inputs, bank, output):
    output.mkdir()
    original = json.loads((HERE / "binding.json").read_text())
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == original["archive_sha256"]
    points = json.loads(
        (HERE.parent / "al028-neural-display-20261122/ordinary-three-size-curves.json").read_text()
    )["points"]
    files = {}
    for w in (160, 320, 640):
        row = max(
            (r for r in points if r["width"] == w and r["bytes"] <= 24494), key=lambda r: r["bytes"]
        )
        origin = [row for path in bank for row in json.loads(path.read_text())["rows"]]
        record = next(i for i in origin if i["sha256"] == row["stream_sha256"])
        raw = Path(record["path"]) / "decoded.yuv"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row["raw_sha256"]
        h = row["height"]
        files[w] = (row, np.memmap(raw, np.uint8, "r", shape=(240, h * 3 // 2, w)))
    assert sum(1 for _ in files) == 3
    key = {}
    with ZipFile(inputs) as original_zip, ZipFile(output / "frames.zip", "x") as out:
        for h, (arm, tick) in original["keys"].items():
            if arm not in ("source", "av1_320_q52_75"):
                continue
            src = original_zip.read(h + ".png")
            assert hashlib.sha256(src).hexdigest() == h
            out.writestr(ZipInfo(h + ".png"), src)
            key[h] = [arm, tick]
            if arm == "source":
                for w in (160, 640):
                    row, raw = files[w]
                    rgb = cv2.resize(
                        yuv_to_rgb(raw[tick], w, row["height"], False),
                        (1920, 1080),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    ok, b = cv2.imencode(".png", rgb[:, :, ::-1])
                    assert ok
                    hashed = hashlib.sha256(b).hexdigest()
                    out.writestr(ZipInfo(hashed + ".png"), b.tobytes())
                    key[hashed] = [f"av1_{w}", tick]
    assert len(key) == 48
    sm = {a: k for k, (a, t) in key.items() if t == 10}
    receipt = dict(
        archive_sha256=hashlib.sha256((output / "frames.zip").read_bytes()).hexdigest(),
        keys=key,
        smoke=sm,
        normalized_labels_sha256=original["normalized_labels_sha256"],
        receiver_masks=original["receiver_masks"],
        source_curves={
            w: {
                k: r[0][k]
                for k in ("width", "height", "kbps", "bytes", "stream_sha256", "raw_sha256")
            }
            for w, r in files.items()
        },
    )
    (output / "binding.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("inputs", "output"):
        p.add_argument(k, type=Path)
    p.add_argument("--bank", nargs="+", type=Path, required=True)
    a = p.parse_args()
    main(a.inputs, a.bank, a.output)
