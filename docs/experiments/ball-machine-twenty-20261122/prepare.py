"""Bind the exact decoded example sent to the detector to its original wire bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import cv2
import numpy as np

from src.pointstream import ball, ball_stream
from src.pointstream.tracked_patch import yuv_to_rgb

EXP = Path(__file__).resolve().parents[1] / "starved-ball-twenty-20261122"
LABEL = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(root, labels, dest):
    dest.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    receipt = json.loads((EXP / "result.json").read_text())
    assert sha(root / "stream.psbl") == receipt["sha256"] and sha(labels) == LABEL
    paid = ball_stream.unpack((root / "stream.psbl").read_bytes())
    positions, look = ball.decode_frames(paid.ball_payload, 240, 960, 544)
    vel = ball.velocities(positions)
    assert paid.native == (root / "independent.ivf").read_bytes()
    native = np.memmap(root / "independent.yuv", np.uint8, "r", shape=(240, 144, 160))
    output = dest / "shown.mkv"
    ff = shutil.which("ffmpeg")
    enc = [
        ff,
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "960x544",
        "-r",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgr0",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        str(output),
    ]
    proc = subprocess.Popen(enc, stdin=subprocess.PIPE)
    for tick in range(240):
        base = cv2.resize(
            yuv_to_rgb(native[tick], 160, 96, False), (960, 544), interpolation=cv2.INTER_CUBIC
        )
        image = ball.render(base, tick, positions, vel, look)
        assert hashlib.sha256(image).hexdigest() == receipt["decoded_frame_hashes"][str(tick)]
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    assert proc.wait(timeout=180) == 0
    cap = cv2.VideoCapture(str(output), cv2.CAP_FFMPEG)
    try:
        for tick in range(240):
            ok, bgr = cap.read()
            assert (
                ok
                and hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).hexdigest()
                == receipt["decoded_frame_hashes"][str(tick)]
            )
        assert not cap.read()[0]
    finally:
        cap.release()
    manifest = dict(
        source_sha256="b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d",
        label_sha256=LABEL,
        labels=[
            dict(tick=x["tick"], ball=x["ball"]) for x in json.loads(labels.read_text())["frames"]
        ],
        original_wire_sha256=receipt["sha256"],
        display_sha256=sha(output),
        hashes=receipt["decoded_frame_hashes"],
        source_positions=receipt["positions"],
        references={},
    )
    shutil.copyfile(root / "stream.psbl", dest / "pointstream.psbl")
    for name, info in receipt["anchors"].items():
        src = Path(info["stream"])
        assert sha(src) == info["stream_sha256"]
        shutil.copyfile(src, dest / (name + ".ivf"))
        manifest["references"][name] = dict(
            sha256=sha(src),
            bytes=src.stat().st_size,
            width=info["width"],
            height=info["height"],
            yuv_sha256=sha(Path(info["decoded"])),
        )
    (dest / "binding.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tarpath = dest / "inputs.tar"
    with tarfile.open(tarpath, "x") as archive:
        for p in sorted(dest.iterdir()):
            if p != tarpath:
                archive.add(p, arcname=p.name)
    print(
        json.dumps(
            dict(
                file=str(tarpath),
                sha256=sha(tarpath),
                binding_sha256=sha(dest / "binding.json"),
                bytes=tarpath.stat().st_size,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ["root", "labels", "dest"]:
        parser.add_argument(k, type=Path)
    a = parser.parse_args()
    run(a.root, a.labels, a.dest)
