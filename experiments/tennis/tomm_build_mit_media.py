"""Build MIT-licensed external visual crop plate and 2x2 supplementary comparison video."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import zlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.tennis.k4b_pose import player_masks
from src.pointstream.tracked_patch import yuv_to_rgb
from src.segmentation.masks import ClipMasks

from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
CARD_102 = ROOT / "docs/experiments/test-crop-joint-20261122"
CARD_NATIVE = ROOT / "docs/experiments/test-joint-extension-20261122/native"

INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")
TENNIS_DATA = Path("/home/itec/emanuele/Datasets/pointstream-data/tennis")


class EventMedia(TypedDict):
    label: str
    short: str
    fps: Fraction
    tick: int
    ball_tick: int
    ball_xy: tuple[int, int]
    supports: list[list[tuple[tuple[int, int, int, int], np.ndarray]]]
    arms: list[tuple[str, str, np.ndarray]]


def _read_member(tar_path: Path, member: str) -> bytes:
    with tarfile.open(tar_path) as z:
        f = z.extractfile(member)
        assert f is not None, f"Missing {member} in {tar_path}"
        return f.read()


def load_csv_marks(csv_path: Path) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = int(row["Frame"])
            vis = int(row["Visibility"])
            if vis == 1 and 0 <= t < 240:
                out[t] = (int(round(float(row["X"]))), int(round(float(row["Y"]))))
    return out


def decode_mp4_all(mp4_path: Path, n_frames: int = 240) -> np.ndarray:
    """Decode first n_frames of 1080p source MP4 to uint8 RGB array (T, H, W, 3)."""
    cmd = [
        "/opt/local/bin/ffmpeg",
        "-v",
        "error",
        "-i",
        str(mp4_path),
        "-frames:v",
        str(n_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = n_frames * 1080 * 1920 * 3
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} bytes from {mp4_path}, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, 1080, 1920, 3)


def decode_av1_tar_all(tar_path: Path, member_path: str, width: int, height: int, n_frames: int = 240) -> np.ndarray:
    """Decode IVF inside published.tar via ffmpeg into native-upscaled 1080p uint8 RGB."""
    with tempfile.TemporaryDirectory(dir="/local/users/emanuele/pointstream/scratch") as tmp:
        ivf = Path(tmp) / "stream.ivf"
        ivf.write_bytes(_read_member(tar_path, member_path))
        raw = subprocess.check_output(
            [
                "/opt/local/bin/ffmpeg",
                "-v",
                "error",
                "-i",
                str(ivf),
                "-frames:v",
                str(n_frames),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ]
        )
        frame_bytes = width * height * 3 // 2
        if len(raw) != n_frames * frame_bytes:
            raise ValueError(f"Expected {n_frames * frame_bytes} YUV bytes, got {len(raw)}")
        out = np.empty((n_frames, 1080, 1920, 3), dtype=np.uint8)
        for t in range(n_frames):
            yuv = np.frombuffer(raw[t * frame_bytes : (t + 1) * frame_bytes], dtype=np.uint8).reshape(height * 3 // 2, width)
            rgb_low = yuv_to_rgb(yuv, width, height, False)
            out[t] = cv2.resize(rgb_low, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return out


def decode_shanghai_av1_warm(tar_path: Path, n_frames: int = 240) -> np.ndarray:
    """Decode Shanghai 320x192 ordinary warm zlib-compressed YUV inside published.tar."""
    anchor = json.loads((CARD_102.parent / "test102-joint-anchors-20261122/manifest.json").read_text())
    row = next(r for r in anchor["rows"] if r["mode"] == "ordinary" and r["width"] == 320)
    sel = row["selected"]["warm"]
    w, h = 320, int(row["height"])
    raw = zlib.decompress(_read_member(tar_path, sel["decoded_file"]))
    frame_bytes = w * h * 3 // 2
    out = np.empty((n_frames, 1080, 1920, 3), dtype=np.uint8)
    for t in range(n_frames):
        yuv = np.frombuffer(raw[t * frame_bytes : (t + 1) * frame_bytes], dtype=np.uint8).reshape(h * 3 // 2, w)
        rgb_low = yuv_to_rgb(yuv, w, h, False)
        out[t] = cv2.resize(rgb_low, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return out


def decode_dcvc_tar_all(tar_path: Path, width: int = 480, height: int = 272, n_frames: int = 240) -> np.ndarray:
    """Decode DCVC publish/decoded.yuv to 1080p uint8 RGB."""
    raw = _read_member(tar_path, "publish/decoded.yuv")
    frame_bytes = width * height * 3 // 2
    if len(raw) != n_frames * frame_bytes:
        raise ValueError(f"Expected {n_frames * frame_bytes}, got {len(raw)}")
    out = np.empty((n_frames, 1080, 1920, 3), dtype=np.uint8)
    for t in range(n_frames):
        yuv = np.frombuffer(raw[t * frame_bytes : (t + 1) * frame_bytes], dtype=np.uint8).reshape(height * 3 // 2, width)
        rgb_low = yuv_to_rgb(yuv, width, height, False)
        out[t] = cv2.resize(rgb_low, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return out


def load_pointstream_npy_all(tar_path: Path, scratch: Path, tag: str) -> np.ndarray:
    """Extract publish/frames.npy from witness tar to scratch and mmap it."""
    dest = scratch / f"{tag}_ps_frames.npy"
    if not dest.exists():
        with tarfile.open(tar_path) as z:
            src = z.extractfile("publish/frames.npy")
            assert src is not None
            with src, dest.open("wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
    arr = np.load(dest, mmap_mode="r", allow_pickle=False)
    assert arr.shape == (240, 1080, 1920, 3) and arr.dtype == np.uint8
    return arr


def padded_crop(rgb: np.ndarray, box: tuple[int, int, int, int], pad: int = 16, min_aspect: float = 0.72) -> np.ndarray:
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(float(x1 - x0) + 2 * pad, 80.0)
    bh = max(float(y1 - y0) + 2 * pad, 80.0)
    bw = max(bw, bh * min_aspect)
    bh = max(bh, bw / min_aspect * 0.75)
    nx0 = int(max(0, round(cx - 0.5 * bw)))
    ny0 = int(max(0, round(cy - 0.5 * bh)))
    nx1 = int(min(rgb.shape[1], round(cx + 0.5 * bw)))
    ny1 = int(min(rgb.shape[0], round(cy + 0.5 * bh)))
    return np.ascontiguousarray(rgb[ny0:ny1, nx0:nx1])


def ball_crop(rgb: np.ndarray, xy: tuple[int, int], half: int = 40) -> np.ndarray:
    x, y = xy
    cx = int(min(max(x, half), rgb.shape[1] - half))
    cy = int(min(max(y, half), rgb.shape[0] - half))
    return np.ascontiguousarray(rgb[cy - half : cy + half, cx - half : cx + half])


def load_player_boxes(
    features_tar: Path, receipt_path: Path, tracks: list[str] | None = None
) -> list[list[tuple[tuple[int, int, int, int], np.ndarray]]]:
    crop = json.loads(receipt_path.read_text())
    prefix = f"publish/test__{crop['event']}@120/"
    info = json.loads(_read_member(features_tar, prefix + "features.json").decode("utf-8"))
    rle = _read_member(features_tar, prefix + "sam-masks.rle")
    with tempfile.TemporaryDirectory(dir="/local/users/emanuele/pointstream/scratch") as tmp:
        path = Path(tmp) / "sam.rle"
        path.write_bytes(rle)
        clip = ClipMasks.load(path)
    sel_tracks = list(crop["selected_ids"]) if tracks is None else tracks
    timeline: list[list[tuple[tuple[int, int, int, int], np.ndarray]]] = []
    for tick in range(240):
        people = player_masks(clip.frames[tick], info["aliases"], sel_tracks)
        entries: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for tr in sel_tracks:
            mask = np.repeat(np.repeat(people[tr], 2, axis=0), 2, axis=1)
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            x0, y0 = max(0, x - 32), max(0, y - 32)
            x1, y1 = min(1920, x + w + 32), min(1080, y + h + 32)
            entries.append(((x0, y0, x1, y1), np.ascontiguousarray(mask[y0:y1, x0:x1] > 0)))
        timeline.append(entries)
    return timeline


def build_event_frames(event: str, scratch: Path) -> EventMedia:
    """Load 240-frame RGB buffers and object coordinates for match102 (Shanghai) or match127 (Sydney)."""
    if event == "match102":
        src = decode_mp4_all(TENNIS_DATA / "cohort-expansion-20261122/match102-000-fc5d3ee431f8.mp4")
        av1 = decode_shanghai_av1_warm(INBOX / "20261122T105317Z-f1736a83/full/published.tar")
        dcvc = decode_dcvc_tar_all(INBOX / "20261122T104722Z-2bcfb2b7/full/published.tar")
        ps = load_pointstream_npy_all(INBOX / "20261122T120219Z-f2c299f2/full/published.tar", scratch, "match102")
        supports = load_player_boxes(
            INBOX / "20261122T095722Z-14da4d11/full/published.tar", CARD_102 / "match102-result.json"
        )
        # Shanghai receipt order: near is index 0 or far? Let's sort by bbox y_max (larger = near).
        marks = load_csv_marks(TENNIS_DATA / "cohort-expansion-20261122/match102-000-0858ae153e5b.csv")
        return {
            "label": "Shanghai (30 fps, ~17.8 kbps)",
            "short": "Shanghai",
            "fps": Fraction(30000, 1001),
            "tick": 95,
            "ball_tick": 95,
            "ball_xy": marks[95],
            "supports": supports,
            "arms": [
                ("Original", "1080p", src),
                ("SVT-AV1", "17.7 kbps", av1),
                ("DCVC-UF", "17.4 kbps", dcvc),
                ("PointStream", "17.8 kbps", ps),
            ],
        }
    if event == "match127":
        src = decode_mp4_all(TENNIS_DATA / "cohort-expansion-20261122/match127-000-8ebca8b69320.mp4")
        syd_rate = json.loads((CARD_102.parent / "cohort-expansion-20261122/test-physical/match127-rate.json").read_text())
        anchors = json.loads(_read_member(INBOX / "20261122T134858Z-f6a90d00/full/published.tar", "publish/anchors.json").decode("utf-8"))
        chosen = anchors["selected"]["warm"]
        av1 = decode_av1_tar_all(
            INBOX / "20261122T134858Z-f6a90d00/full/published.tar",
            "publish/" + chosen["file"],
            int(anchors["width"]),
            int(anchors["height"]),
        )
        _ = syd_rate
        dcvc = decode_dcvc_tar_all(INBOX / "20261122T141034Z-490fceb9/full/published.tar")
        ps = load_pointstream_npy_all(INBOX / "20261122T224251Z-83f3aed8/full/published.tar", scratch, "match127")
        b = json.loads((CARD_NATIVE / "match127-binding.json").read_text())
        supports = load_player_boxes(
            INBOX / "20261122T113333Z-9f0e3e92/full/published.tar",
            CARD_NATIVE / "match127-crop-receipt.json",
            tracks=[b["roles"]["near"], b["roles"]["far"]],
        )
        marks = load_csv_marks(TENNIS_DATA / "cohort-expansion-20261122/match127-000-dd004f20ec88.csv")
        return {
            "label": "Sydney (60 fps, ~25.8 kbps)",
            "short": "Sydney",
            "fps": Fraction(60, 1),
            "tick": 115,
            "ball_tick": 68,
            "ball_xy": marks[68],
            "supports": supports,
            "arms": [
                ("Original", "1080p", src),
                ("SVT-AV1", "25.4 kbps", av1),
                ("DCVC-UF", "23.1 kbps", dcvc),
                ("PointStream", "25.8 kbps", ps),
            ],
        }
    raise ValueError(f"Unsupported event {event}")


def draw_crop_plate(events_data: list[EventMedia], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 8.0,
        }
    )
    # Build 6 rows: Sydney Near, Sydney Far, Sydney Ball; Shanghai Near, Shanghai Far, Shanghai Ball
    n_rows = 3 * len(events_data)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.52 * n_rows))
    for e_idx, ed in enumerate(events_data):
        t = ed["tick"]
        t_ball = ed["ball_tick"]
        supp_t = ed["supports"][t]
        # Order near (larger y1) first, far second
        boxes_sorted = sorted([b for b, _ in supp_t], key=lambda b: b[3], reverse=True)
        near_box, far_box = boxes_sorted[0], boxes_sorted[1]
        ball_xy = ed["ball_xy"]
        arms = ed["arms"]
        spec_rows: list[tuple[str, int, str, tuple[int, int, int, int] | tuple[int, int]]] = [
            (f"{ed['short']} t={t}\nNear player", t, "player", near_box),
            (f"{ed['short']} t={t}\nFar player", t, "player", far_box),
            (f"{ed['short']} t={t_ball}\nBall (80x80)", t_ball, "ball", ball_xy),
        ]
        for r_off, (row_label, row_t, kind, geom) in enumerate(spec_rows):
            r_idx = 3 * e_idx + r_off
            for c_idx, (col_name, rate_lbl, arr) in enumerate(arms):
                ax = axes[r_idx, c_idx]
                frame = arr[row_t]
                if kind == "player":
                    assert len(geom) == 4
                    patch = padded_crop(frame, geom)
                else:
                    assert len(geom) == 2
                    patch = ball_crop(frame, geom, half=40)
                ax.imshow(patch)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(1.1 if col_name == "PointStream" else 0.5)
                    sp.set_edgecolor("#12695d" if col_name == "PointStream" else "#b0b6bc")
                if r_off == 0 and e_idx == 0:
                    ax.set_title(f"{col_name}\n({rate_lbl})", fontsize=8.0, fontweight="bold" if col_name == "PointStream" else "normal", pad=4)
                elif r_off == 0:
                    ax.set_title(f"{col_name} ({rate_lbl})", fontsize=7.6, fontweight="bold" if col_name == "PointStream" else "normal", pad=3)
                if c_idx == 0:
                    ax.set_ylabel(row_label, fontsize=7.6, rotation=0, ha="right", va="center", labelpad=6)
                if kind == "ball":
                    h, w = patch.shape[:2]
                    ax.plot(w / 2, h / 2, marker="o", markersize=14, markerfacecolor="none", markeredgecolor="#e63946", markeredgewidth=0.75, alpha=0.45)
    fig.subplots_adjust(left=0.145, right=0.99, top=0.93, bottom=0.02, wspace=0.04, hspace=0.16)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=240)
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def stamp_label(bgr: np.ndarray, text: str, highlight: bool = False) -> np.ndarray:
    bg = (24, 105, 93) if highlight else (24, 28, 34)
    cv2.rectangle(bgr, (14, 12), (430, 50), bg, thickness=-1)
    cv2.rectangle(bgr, (14, 12), (430, 50), (220, 226, 232), thickness=1)
    cv2.putText(bgr, text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def write_comparison_video(events_data: list[EventMedia], out_mp4: Path, scratch: Path) -> None:
    """Write a 1920x1080 H.264 2x2 grid video (~12s total) comparing all 4 arms on Sydney and Shanghai."""
    raw_mp4 = scratch / "raw_grid.yuv"
    fps_out = 30
    with raw_mp4.open("wb") as f:
        for ed in events_data:
            arms = ed["arms"]
            short = ed["short"]
            fps = ed["fps"]
            step = 2 if fps == Fraction(60, 1) else 1  # 60fps 240f -> 120f @ 30fps (4s); 30fps 240f -> 240f @ 30fps (8s)
            for t in range(0, 240, step):
                quads = []
                for col_name, rate_lbl, arr in arms:
                    rgb_q = cv2.resize(np.ascontiguousarray(arr[t]), (960, 540), interpolation=cv2.INTER_AREA)
                    bgr_q = cv2.cvtColor(rgb_q, cv2.COLOR_RGB2BGR)
                    stamp_label(bgr_q, f"{short} | {col_name} ({rate_lbl})", highlight=(col_name == "PointStream"))
                    quads.append(bgr_q)
                top = np.hstack([quads[0], quads[1]])
                bot = np.hstack([quads[2], quads[3]])
                canvas = np.vstack([top, bot])
                # 1px clean divider lines
                canvas[539:541, :, :] = 230
                canvas[:, 959:961, :] = 230
                f.write(canvas.tobytes())
    cmd = [
        "/opt/local/bin/ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1920x1080",
        "-r",
        str(fps_out),
        "-i",
        str(raw_mp4),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    raw_mp4.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/local/users/emanuele/pointstream/scratch/mit-media"))
    p.add_argument("--out-pdf", type=Path, required=True)
    p.add_argument("--out-mp4", type=Path, required=True)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    a.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    a.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    events_data = [build_event_frames(ev, a.scratch) for ev in ("match127", "match102")]
    draw_crop_plate(events_data, a.out_pdf)
    write_comparison_video(events_data, a.out_mp4, a.scratch)
    print(json.dumps({"pdf": str(a.out_pdf), "mp4": str(a.out_mp4), "mp4_bytes": a.out_mp4.stat().st_size}))


if __name__ == "__main__":
    main()
