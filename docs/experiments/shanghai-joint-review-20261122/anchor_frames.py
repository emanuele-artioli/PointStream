"""Render authenticated native AV1 samples with the same display conversion as DCVC."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zlib

import cv2
import numpy as np

from src.pointstream.tracked_patch import yuv_to_rgb


def render(archive_path, review, output, source_witness=None):
    with tarfile.open(archive_path) as archive:
        sampling = json.load(archive.extractfile("sampling.json"))
        assert sampling["ticks"] == [entry["frame_index"] for entry in review["frames"]]
        for name in archive.getnames():
            if not name.endswith("/meta.json"):
                continue
            meta = json.load(archive.extractfile(name))
            prefix = name.split("/")[0]
            width, height = meta["width"], meta["height"]
            directory = output / prefix
            directory.mkdir(exist_ok=False, parents=True)
            frames = []
            for tick in sampling["ticks"]:
                raw = zlib.decompress(archive.extractfile(f"{prefix}/{tick}.z").read())
                assert hashlib.sha256(raw).hexdigest() == meta["frames"][str(tick)]
                yuv = np.frombuffer(raw, np.uint8).reshape(height * 3 // 2, width)
                rgb = cv2.resize(
                    yuv_to_rgb(yuv, width, height, False),
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC,
                )
                path = directory / f"tick-{tick:03}.png"
                assert cv2.imwrite(str(path), rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                assert np.array_equal(cv2.imread(str(path))[:, :, ::-1], rgb)
                frames.append(
                    dict(
                        tick=tick,
                        path=path.name,
                        png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        rgb_sha256=hashlib.sha256(rgb).hexdigest(),
                        **(
                            {
                                "source_rgb_sha256": source_witness["ticks"][str(tick)][
                                    "raw_rgb_sha256"
                                ]
                            }
                            if source_witness
                            else {}
                        ),
                    )
                )
            arm = dict(
                mode=(
                    f"DCVC-UF Q{meta['neural_qp']} {width}x{height}"
                    if "neural_qp" in meta
                    else f"SVT-AV1 {meta['mode']} {width}x{height} {meta['band']}"
                ),
                bytes=meta["bytes"],
                bitstream_sha256=meta["bitstream_sha256"],
                decoded_sha256=meta["decoded_sha256"],
                native_archive_sha256=meta["archive_sha256"],
                job=meta["job"],
                frames=frames,
                source_ticks=240,
                source_sha256=review["source_video_sha256"],
                fps=review["source_video_fps"],
                cv2=cv2.__version__,
            )
            if source_witness:
                assert source_witness["source_sha256"] == arm["source_sha256"]
                arm["source_review_sha256"] = source_witness["review_sha256"]
            (directory / "arm.json").write_text(json.dumps(arm, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "review", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--source-witness", type=Path)
    args = p.parse_args()
    witness = json.loads(args.source_witness.read_text()) if args.source_witness else None
    render(
        args.archive, json.loads((args.review / "review.json").read_text()), args.output, witness
    )
