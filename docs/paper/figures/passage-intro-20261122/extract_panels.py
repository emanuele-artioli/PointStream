"""Export three private source-frame derivatives for the authenticated passage."""

import hashlib
import io
import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
SOURCE = ROOT / "tennis/cumulative-joint-20261122/nadal-kyrgios-broadcast.mp4"
PACKET = ROOT / "tennis/cumulative-joint-20261122/three-segment-trajectory-v1/delivery.psb"
FF = "/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffmpeg"
OUT = Path("/local/users/emanuele/pointstream/audits/passage-intro-panels-20261122")
TICKS = {"first": 76716, "intervening_closeup": 77850, "return": 87801}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    assert sha(SOURCE) == "676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66"
    assert sha(PACKET) == "c53685733be3ccfce2673bc49e4ad603c998b9eccd24a5e12d6acb70782ada00"
    assert sha(Path(FF)) == "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2"
    rows = []
    for name, t in TICKS.items():
        png = subprocess.run(
            [
                FF,
                "-v",
                "error",
                "-threads",
                "1",
                "-ss",
                f"{(t + 1) / 50:.2f}",
                "-i",
                str(SOURCE),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-c:v",
                "png",
                "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout
        im = Image.open(io.BytesIO(png)).convert("RGB")
        assert im.size == (1920, 1080)
        small = im.resize((640, 360), Image.Resampling.LANCZOS)
        small.save(OUT / f"{name}.png")
        row = {
            "name": name,
            "native_tick": t,
            "passage_seconds": (t - 76626) / 50,
            "source_png_sha256": hashlib.sha256(png).hexdigest(),
            "derivative_sha256": sha(OUT / f"{name}.png"),
        }
        if name == "first":
            # Human-reviewed mark, 1920x1080 source coordinate, tick 90 of /000.
            x, y = 945, 477
            im.crop((x - 32, y - 32, x + 32, y + 32)).resize(
                (256, 256), Image.Resampling.NEAREST
            ).save(OUT / "ball-inset.png")
            row["ball_center_xy"] = [x, y]
            row["ball_inset_sha256"] = sha(OUT / "ball-inset.png")
        rows.append(row)
    result = {
        "source_sha256": sha(SOURCE),
        "packet_sha256": sha(PACKET),
        "ffmpeg_sha256": sha(Path(FF)),
        "seek_rule": "(native tick + 1) / 50; established passage identity",
        "rows": rows,
    }
    (OUT / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
