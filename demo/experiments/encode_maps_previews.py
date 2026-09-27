"""Prepare maps-gallery previews without wrapping native mask payloads in H.264.

Canny stays a PNG mask sequence (native payload is bit-packed zstd).
YOLOE / pose overlay visualizations are SVT-AV1, matching the demo ladder.
Preview files are never counted as payload.

    PYTHONPATH=. python demo/experiments/encode_maps_previews.py --maps-root PATH
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.experiments.run_maps import rebuild_index_from_disk
from demo.pipeline.maps.canny import PAYLOAD_FORMAT, pack_canny_payload, resize_binary_mask
from demo.pipeline.maps.contract import payload_kbps

# Same encoder family as demo/evaluation/encode_av1_ladder.py (preset 8 ≈ live SVT-AV1).
SVT_AV1 = (
    "-c:v",
    "libsvtav1",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "8",
    "-crf",
    "32",
    "-movflags",
    "+faststart",
    "-an",
    "-vf",
    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
)


def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(cmd, check=True)


def _ffprobe_codec(path: Path) -> str:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def encode_png_dir_webm(png_dir: Path, dest: Path, fps: float) -> Path:
    pngs = sorted(png_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no pngs in {png_dir}")
    if not (png_dir / "000000.png").is_file():
        raise FileNotFoundError(f"expected 000000.png under {png_dir}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.webm")
    if dest.is_file() and dest.stat().st_size > 1024 and not tmp.is_file():
        return dest
    _ffmpeg(
        "-framerate",
        f"{fps:g}",
        "-i",
        str(png_dir / "%06d.png"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
        "-crf",
        "40",
        "-b:v",
        "0",
        "-an",
        "-vf",
        r"scale=w=min(960\,iw):h=-2",
        str(tmp),
    )
    tmp.replace(dest)
    return dest


def _png_seq_dirs(map_dir: Path) -> list[Path]:
    return [p for p in sorted(map_dir.iterdir()) if p.is_dir() and p.name != "rungs" and list(p.glob("*.png"))]


def encode_rgba_webm(map_dir: Path, fps: float) -> list[Path]:
    written: list[Path] = []
    png_dirs = _png_seq_dirs(map_dir)
    if png_dirs:
        dest = map_dir / "preview.webm"
        encode_png_dir_webm(png_dirs[0], dest, fps)
        written.append(dest)
    rungs = map_dir / "rungs"
    if rungs.is_dir():
        for rung in sorted(p for p in rungs.iterdir() if p.is_dir()):
            rung_pngs = _png_seq_dirs(rung)
            if not rung_pngs:
                continue
            dest = rung / "preview.webm"
            encode_png_dir_webm(rung_pngs[0], dest, fps)
            written.append(dest)
    return written


def encode_png_dir_av1(png_dir: Path, dest: Path, fps: float) -> Path:
    pngs = sorted(png_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no pngs in {png_dir}")
    if not (png_dir / "000000.png").is_file():
        raise FileNotFoundError(f"expected 000000.png under {png_dir}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    _ffmpeg(
        "-framerate",
        f"{fps:g}",
        "-i",
        str(png_dir / "%06d.png"),
        *SVT_AV1,
        str(tmp),
    )
    tmp.replace(dest)
    return dest


def encode_svtav1_mp4(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    codec = _ffprobe_codec(src) if src.is_file() else ""
    if src.resolve() == dest.resolve() and codec == "av1":
        return dest
    tmp = dest.with_suffix(".tmp.mp4")
    _ffmpeg("-i", str(src), *SVT_AV1, str(tmp))
    tmp.replace(dest)
    return dest


def _load_binary_pngs(preview_dir: Path) -> list[np.ndarray]:
    masks = []
    for png in sorted(preview_dir.glob("*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"failed to read {png}")
        if img.ndim == 3 and img.shape[2] == 4:
            masks.append((img[:, :, 3] > 0).astype(np.uint8))
        elif img.ndim == 3:
            masks.append((img.max(axis=2) > 0).astype(np.uint8))
        else:
            masks.append((img > 0).astype(np.uint8))
    if not masks:
        raise FileNotFoundError(f"no canny preview pngs in {preview_dir}")
    return masks


def recompress_canny(canny_dir: Path) -> None:
    sidecar_path = canny_dir / "sidecar.json"
    sidecar = json.loads(sidecar_path.read_text())
    preview_dir = canny_dir / "preview"
    masks = _load_binary_pngs(preview_dir)
    pack_h = sidecar.get("pack_height")
    pack_w = sidecar.get("pack_width")
    if pack_h and pack_w:
        masks = [resize_binary_mask(mask, int(pack_h), int(pack_w)) for mask in masks]
    payload = pack_canny_payload(masks)
    payload_path = canny_dir / "payload.bin"
    payload_path.write_bytes(payload)
    duration_s = float(sidecar["duration_s"])
    sidecar["payload_bytes"] = payload_path.stat().st_size
    sidecar["payload_kbps"] = payload_kbps(sidecar["payload_bytes"], duration_s)
    sidecar["payload_format"] = PAYLOAD_FORMAT
    pngs = sorted(preview_dir.glob("*.png"))
    sidecar["preview_path"] = str(preview_dir)
    sidecar["preview_bytes"] = sum(p.stat().st_size for p in pngs)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    stale = canny_dir / "preview.mp4"
    if stale.is_file():
        stale.unlink()


def _update_sidecar_preview(sidecar_path: Path, preview_mp4: Path) -> None:
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["preview_path"] = str(preview_mp4)
    sidecar["preview_bytes"] = preview_mp4.stat().st_size
    sidecar["preview_codec"] = "libsvtav1"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")


def encode_map_dir(map_dir: Path) -> list[Path]:
    written: list[Path] = []
    sidecar_path = map_dir / "sidecar.json"
    fps = 30.0
    map_name = ""
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text())
        fps = float(sidecar.get("fps") or 30.0) or 30.0
        map_name = str(sidecar.get("map") or "")
        if map_name.startswith("canny"):
            if not (map_dir / "payload.bin").is_file():
                recompress_canny(map_dir)
            rungs = map_dir / "rungs"
            if rungs.is_dir():
                for rung in sorted(p for p in rungs.iterdir() if p.is_dir()):
                    if (rung / "sidecar.json").is_file() and not (rung / "payload.bin").is_file():
                        recompress_canny(rung)
            written.extend(encode_rgba_webm(map_dir, fps))
            return written
        if map_name in {
            "yoloe_masks",
            "sam31_masks",
            "dino_feat",
            "dwpose",
        }:
            written.extend(encode_rgba_webm(map_dir, fps))
            return written

    png_dirs = [p for p in map_dir.iterdir() if p.is_dir() and list(p.glob("*.png"))]
    for png_dir in png_dirs:
        dest = map_dir / "preview.mp4"
        encode_png_dir_av1(png_dir, dest, fps)
        if sidecar_path.is_file():
            _update_sidecar_preview(sidecar_path, dest)
        written.append(dest)

    for mp4 in sorted(map_dir.glob("*.mp4")):
        if mp4.name.endswith(".tmp.mp4"):
            continue
        codec = _ffprobe_codec(mp4)
        dest = mp4
        if codec != "av1":
            encode_svtav1_mp4(mp4, dest)
        written.append(dest)
        name = mp4.stem.replace("preview_", "")
        pose_sidecar = map_dir / f"{name}.json"
        if pose_sidecar.is_file():
            _update_sidecar_preview(pose_sidecar, dest)
        canonical = map_dir.parent / name / "sidecar.json"
        if canonical.is_file():
            _update_sidecar_preview(canonical, dest)
        elif sidecar_path.is_file() and map_name not in {"canny", "yoloe_masks"}:
            _update_sidecar_preview(sidecar_path, dest)
    return written


def encode_maps_root(maps_root: Path) -> dict:
    maps_root = Path(maps_root)
    encoded: list[str] = []
    for clip_dir in sorted(p for p in maps_root.iterdir() if p.is_dir()):
        for map_dir in sorted(p for p in clip_dir.iterdir() if p.is_dir()):
            try:
                for dest in encode_map_dir(map_dir):
                    encoded.append(str(dest))
            except Exception as exc:
                print(f"WARN {map_dir}: {exc}", file=sys.stderr)
    index = rebuild_index_from_disk(maps_root)
    (maps_root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    index["_encoded"] = encoded
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-root", type=Path, required=True)
    args = parser.parse_args(argv)
    index = encode_maps_root(args.maps_root)
    print(f"encoded {len(index.get('_encoded') or [])} overlay previews")
    print(f"entries {len(index.get('entries') or [])}")
    for entry in index.get("entries") or []:
        print(
            f"  {entry['clip']}/{entry['map']}: {entry['payload_kbps']:.1f} kbps "
            f"{entry.get('preview_kind')} {entry.get('preview_url')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
