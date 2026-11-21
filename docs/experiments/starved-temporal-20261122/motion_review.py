"""Publish a private 50-fps visual companion to the frozen motion comparison."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import run

HERE = Path(__file__).resolve().parent
RATE = 50


def main(destination):
    destination.mkdir(exist_ok=False)
    evidence = json.loads((HERE / "result.json").read_text())
    output = destination / "serve-comparison.mp4"
    # The 960 x 544 canvas uses the same spatial transform for all columns.
    x0, x1 = 400, 685
    y0, y1 = 166, 538
    width, height = 3 * (x1 - x0) + 5, y1 - y0 + 44
    command = [
        "/opt/homebrew/bin/ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(RATE),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    full_receipt = destination / "motion-replay.json"
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:

        def write_frame(tick, images):
            labels = (
                ("source", "Source"),
                ("plain", "AV1   66.1 kbps"),
                ("pointstream", "PointStream"),
            )
            canvas = np.full((height, width, 3), 251, np.uint8)
            for i, (key, label) in enumerate(labels):
                start = i * (x1 - x0 + 2)
                canvas[44:, start : start + x1 - x0] = images[key][y0:y1, x0:x1]
                cv2.putText(
                    canvas,
                    label,
                    (start + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (24, 36, 41),
                    1,
                    cv2.LINE_AA,
                )
                note = (
                    f"{tick / RATE:.2f} s"
                    if i == 0
                    else ("65.5 cached / 469.3 cold kbps" if i == 2 else "")
                )
                cv2.putText(
                    canvas,
                    note,
                    (start + 4, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.41,
                    (33, 46, 53),
                    1,
                    cv2.LINE_AA,
                )
            encoder.stdin.write(canvas.tobytes())

        try:
            run.main(full_receipt, on_frame=write_frame)
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Failed to encode the private review file")
    metrics = json.loads(full_receipt.read_text())
    assert metrics["means"] == evidence["means"]
    probe = json.loads(
        subprocess.check_output(
            [
                "/opt/homebrew/bin/ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,r_frame_rate,width,height",
                "-of",
                "json",
                str(output),
            ]
        )
    )["streams"][0]
    assert probe["nb_read_frames"] == "240" and probe["r_frame_rate"] == "50/1"
    (destination / "video-receipt.json").write_text(
        json.dumps(
            {
                "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "probe": probe,
                "crop_xyxy": [x0, y0, x1, y1],
                "matches_temporal_scores": True,
                "scope": "Private review derivative of the measured streams; its file size is not a codec rate.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("destination", type=Path)
    main(cli.parse_args().destination)
