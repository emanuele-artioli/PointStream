"""Verify original JPEGs and author labels; export a pixel-exact RGB video at their 30 fps."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import cv2
import numpy as np


def ordered(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    dataset = json.loads(Path(__file__).with_name("manifest.json").read_text())
    match = next(e for e in dataset["events"] if e["event_id"] == "tracknet_game9")
    all_rows = []
    with args.archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with tarfile.open(args.archive) as src:
        for r in match["rallies"]:
            rid, prefix = r["rally_id"], r["source_id"]
            label_member = src.extractfile(prefix + "/Label.csv")
            if label_member is None:
                raise ValueError("Missing original CSV")
            label_bytes = label_member.read()
            if hashlib.sha256(label_bytes).hexdigest() != r["label_sha256"]:
                raise ValueError("Original author label hash changed")
            (args.out_dir / (rid + "-labels.csv")).write_bytes(label_bytes)
            labels = list(csv.DictReader(io.StringIO(label_bytes.decode())))
            n = r["native_range"][1]
            names = [x["file name"] for x in labels]
            if names != [f"{i:04d}.jpg" for i in range(n)]:
                raise ValueError("Native image numbering or annotation length changed")
            target = args.out_dir / (rid + ".mkv")
            frame_bytes = 1280 * 720 * 3
            image_hashes, rgb_hashes, visibility = [], [], {}
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                "1280x720",
                "-framerate",
                "30",
                "-i",
                "-",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-threads",
                "2",
                "-pix_fmt",
                "bgr0",
                str(target),
            ]
            with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
                for label, name in zip(labels, names):
                    member = src.extractfile(prefix + "/" + name)
                    if member is None:
                        raise ValueError("Missing original image")
                    raw = member.read()
                    image_hashes.append(hashlib.sha256(raw).hexdigest())
                    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if frame.shape != (720, 1280, 3):
                        raise ValueError("JPEG native geometry differs")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                    ff.stdin.write(frame.tobytes())
                    vis = label["visibility"]
                    visibility[vis] = visibility.get(vis, 0) + 1
                ff.stdin.close()
                if ff.wait():
                    raise RuntimeError("Lossless RGB writer failed")
            validation = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "2",
                "-i",
                str(target),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            with subprocess.Popen(validation, stdout=subprocess.PIPE) as verify:
                for tick, expected in enumerate(rgb_hashes):
                    decoded = verify.stdout.read(frame_bytes)
                    if (
                        len(decoded) != frame_bytes
                        or hashlib.sha256(decoded).hexdigest() != expected
                    ):
                        raise ValueError(f"RGB validation failed at {rid} tick {tick}")
                if verify.stdout.read(1) or verify.wait():
                    raise ValueError("Lossless RGB reader failed or produced extra frames")
            with target.open("rb") as f:
                clip_sha = hashlib.file_digest(f, "sha256").hexdigest()
            all_rows.append(
                dict(
                    rally_id=rid,
                    label_sha256=r["label_sha256"],
                    native_range=[0, n],
                    source_path=prefix,
                    clip_filename=target.name,
                    clip_sha256=clip_sha,
                    original_jpeg_ordered_sha256=ordered(image_hashes),
                    original_rgb_ordered_sha256=ordered(rgb_hashes),
                    original_first_rgb_sha256=rgb_hashes[0],
                    original_last_rgb_sha256=rgb_hashes[-1],
                    native_rgb_replay_equal_frames=n,
                    visibility=visibility,
                )
            )
    with args.receipt.open("x") as f:
        json.dump(
            dict(
                input_subset_tar_sha256=digest,
                parent_zip_sha256=match["rallies"][0]["source_archive_sha256"],
                parent_identity_basis="Archived original Dataset.zip SHA from curator's manifest. Re-read selected members with ZIP CRC validation; the full ZIP was not rehashed in this transfer.",
                cadence=[30, 1],
                clock_basis="Author dataset sequence and reported native display rate. Original broadcast presentation timestamps are unavailable.",
                jpeg_decoder=f"OpenCV {cv2.__version__} imdecode(IMREAD_COLOR)",
                ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                rallies=all_rows,
            ),
            f,
            indent=2,
        )
        f.write("\n")
    print([r["native_rgb_replay_equal_frames"] for r in all_rows])


if __name__ == "__main__":
    main()
