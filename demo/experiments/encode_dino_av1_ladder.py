"""Encode existing 1080p DINOv3 PCA previews with the shared AV1 CRF ladder.

The PCA sequence is the mask already computed at 1920x1080. Each rung uses
``av1_output_args`` (SVT-AV1 CRF 63, preset 7), the same recipe as the
camera baseline. Payload kbps is the encoded file, not the int8 feature blob.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_LADDER, av1_output_args

CLIPS = ("clip_01", "clip_02", "clip_03")
FPS = 30.0
N_FRAMES = 300


def encode_sequence(seq_dir: Path, dest: Path, scale: str | None, ffmpeg: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{FPS:.6f}",
        "-start_number",
        "0",
        "-i",
        str(seq_dir / "%06d.png"),
        "-frames:v",
        str(N_FRAMES),
        *av1_output_args(scale),
        str(dest),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        tail = res.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"AV1 encode failed for {dest}: {tail}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    args = parser.parse_args()
    duration_s = N_FRAMES / FPS
    report: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for clip in CLIPS:
        seq = args.maps_root / clip / "dino_feat" / "preview_pca"
        if not (seq / "000000.png").is_file():
            raise FileNotFoundError(seq)
        report[clip] = {}
        for name, scale in AV1_LADDER:
            dest = args.out / clip / f"dino_{name}_crf{AV1_CRF}.mp4"
            encode_sequence(seq, dest, scale, args.ffmpeg)
            nbytes = dest.stat().st_size
            report[clip][name] = {
                "bytes": nbytes,
                "kbps": round(nbytes * 8 / duration_s / 1000.0, 1),
                "path": str(dest),
                "scale": scale or "1920:1080",
            }
            print(f"{clip} {name} {report[clip][name]['kbps']} kbps", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
