"""Create a private, source-authenticated overview of fixed fallback ticks."""

import hashlib
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
SOURCE = ROOT / "tennis/cumulative-joint-20261122/nadal-kyrgios-broadcast.mp4"
FF = "/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffmpeg"
OUT = Path("/local/users/emanuele/pointstream/audits/passage-intro-samples-20261122")
TICKS = [76866 + int(i * 10824 / 11) for i in range(12)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    assert sha(SOURCE) == "676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66"
    assert sha(Path(FF)) == "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2"
    sheet = Image.new("RGB", (1600, 930), "white")
    d = ImageDraw.Draw(sheet)
    rows = []
    for i, t in enumerate(TICKS):
        b = subprocess.run(
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
                "-vf",
                "scale=400:225:flags=area",
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
        im = Image.open(io.BytesIO(b)).convert("RGB")
        x = (i % 4) * 400
        y = (i // 4) * 310
        sheet.paste(im, (x, y))
        d.text((x + 5, y + 230), f"tick {t} / passage {(t - 76626) / 50:.2f} s", fill="black")
        rows.append(
            {
                "tick": t,
                "passage_seconds": (t - 76626) / 50,
                "source_png_sha256": hashlib.sha256(b).hexdigest(),
            }
        )
    sheet.save(OUT / "contact.png")
    (OUT / "receipt.json").write_text(
        json.dumps(
            {"source_sha256": sha(SOURCE), "ffmpeg_sha256": sha(Path(FF)), "rows": rows}, indent=2
        )
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
