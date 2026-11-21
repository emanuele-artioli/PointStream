"""Package the existing human-reviewed supports and hash-to-arm mapping."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

LABEL = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
KEY = "c5c79d2b01ea4e2bfc70d53bef1b050f46fe1838cd9af27cad0bcda0cc4f8040"


def check(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Identity changed: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    check(args.labels, LABEL)
    check(args.key, KEY)
    labeled = json.loads(args.labels.read_text())
    keys = json.loads(args.key.read_text())
    ticks = sorted({tick for _, tick in keys.values()})
    if len(ticks) != 12 or len(keys) != 48:
        raise ValueError("Unexpected sample or frame count")
    source = {row["tick"]: row for row in labeled["frames"]}
    masks, centers, counts = [], [], []
    for tick in ticks:
        row = source[tick]
        if row["ball"]["status"] != "visible" or len(row["players"]) != 2:
            raise ValueError("Missing reviewed visible person or ball")
        people = []
        for person in row["players"]:
            image = cv2.imdecode(np.frombuffer(base64.b64decode(person["mask"]["png_base64"], validate=True), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image.shape != (1080, 1920) or np.count_nonzero(image) != person["mask"]["foreground_pixels"]:
                raise ValueError("Body mask identity differs")
            people.append(image > 0)
        source_hashes = [digest for digest, (arm, t) in keys.items() if arm == "source" and t == tick]
        if source_hashes != [row["source_png_sha256"]]:
            raise ValueError("Source PNG and reviewed frame differ")
        masks.append(people)
        counts.append([int(np.count_nonzero(person)) for person in people])
        centers.append(row["ball"]["center_xy"])
    args.output.mkdir()
    np.savez_compressed(args.output / "supports.npz", ticks=np.array(ticks, dtype=np.int32),
                        bodies=np.array(masks, dtype=bool), centers=np.array(centers, dtype=np.int32))
    (args.output / "key.json").write_text(args.key.read_text())
    receipt = {"label_sha256": LABEL, "key_sha256": KEY, "ticks": ticks, "body_pixel_counts": counts,
               "supports_sha256": hashlib.sha256((args.output / "supports.npz").read_bytes()).hexdigest()}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with tarfile.open(args.output / "supports.tar", "x") as tar:
        for name in ("supports.npz", "key.json", "receipt.json"):
            tar.add(args.output / name, arcname=name)
    print(json.dumps({**receipt, "tar_sha256": hashlib.sha256((args.output / "supports.tar").read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
