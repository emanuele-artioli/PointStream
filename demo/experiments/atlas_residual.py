"""Cross-clip background atlas on factory 035.

Clip 1 is held out. Neighbor recordings from the same worker supply one or
more plates. The billed rate is a plate id plus a homography per frame plus
the AV1 residual of (frame - warped plate). The plate bytes are not billed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.maps.av1_crf import encode_av1_crf
from src.components.background.plate import build_plate

FFMPEG = os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")
W, H = 426, 240
QUERY = Path(
    "/home/itec/emanuele/Datasets/Egocentric-10K/curated/clip_01_factory035_close_hands.mp4"
)
LIBRARY = sorted(
    Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw/extracted_new").glob(
        "factory035_worker001_*.mp4"
    )
)


def _read_scaled(path: Path, *, step: int = 1, limit: int | None = None) -> list[np.ndarray]:
    """Decode to 426x240 BGR. ``step`` keeps every Nth frame."""
    cmd = [
        FFMPEG,
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={W}:{H},select='not(mod(n\\,{step}))'",
        "-vsync",
        "vfr",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    if limit is not None:
        cmd[4:4] = ["-frames:v", str(limit * step)]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    frame_bytes = W * H * 3
    n = len(raw) // frame_bytes
    frames = []
    for i in range(n):
        chunk = raw[i * frame_bytes : (i + 1) * frame_bytes]
        frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(H, W, 3).copy())
    return frames


def _sample_long(path: Path, every_s: float = 8.0, fps: float = 30.0) -> list[np.ndarray]:
    step = max(int(every_s * fps), 1)
    return _read_scaled(path, step=step)


def _orb_inliers(src: np.ndarray, dst: np.ndarray) -> tuple[int, np.ndarray | None]:
    """Homography mapping ``src`` pixels into ``dst``. Shapes may differ."""
    orb = cv2.ORB_create(nfeatures=1500)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 8:
        return 0, None
    matches = sorted(matches, key=lambda m: m.distance)[:80]
    pts_src = np.float32([k1[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([k2[m.trainIdx].pt for m in matches])
    Hmat, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    if Hmat is None or mask is None:
        return 0, None
    return int(mask.sum()), Hmat


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-8:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _kbps(nbytes: int, n_frames: int, fps: float = 30.0) -> float:
    return (nbytes * 8) / (n_frames / fps) / 1000.0


def _write_mp4(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    out = Path("/home/itec/emanuele/tmp/atlas-residual")
    out.mkdir(parents=True, exist_ok=True)
    query = _read_scaled(QUERY)
    print(f"query frames {len(query)}", flush=True)
    probes = [query[0], query[len(query) // 2], query[-1]]

    ranked = []
    for path in LIBRARY:
        samples = _sample_long(path)
        if not samples:
            continue
        best_inliers = 0
        best_pair = None
        for sample in samples:
            for probe in probes:
                n_in, _ = _orb_inliers(sample, probe)
                if n_in > best_inliers:
                    best_inliers = n_in
                    best_pair = (sample, probe)
        mad = 255.0
        if best_pair is not None and best_pair[0].shape == best_pair[1].shape:
            mad = float(np.mean(np.abs(best_pair[0].astype(np.int16) - best_pair[1].astype(np.int16))))
        overlap = best_inliers >= 40 and mad < 8.0
        ranked.append({"path": str(path), "inliers": best_inliers, "mad": round(mad, 2), "overlap": overlap, "samples": samples})
        print(f"{path.name} inliers={best_inliers} mad={mad:.1f} overlap={overlap} samples={len(samples)}", flush=True)

    donors = [row for row in ranked if not row["overlap"] and row["inliers"] >= 12]
    donors.sort(key=lambda row: row["inliers"], reverse=True)
    donors = donors[:3]
    if not donors:
        donors = sorted([row for row in ranked if not row["overlap"]], key=lambda row: row["inliers"], reverse=True)[:3]
    print("donors", [Path(row["path"]).name for row in donors], flush=True)

    plates = []
    for index, row in enumerate(donors):
        # Spread a few frames. Hands are not masked: this worker's other
        # takes still contain hands, which can burn into the plate.
        frames = row["samples"]
        take = frames[:: max(len(frames) // 8, 1)][:8]
        plate, _packed = build_plate(np.stack(take, axis=0), register=True)
        cv2.imwrite(str(out / f"plate_{index}.png"), plate)
        plates.append(plate)
        print(f"plate {index} {plate.shape} from {Path(row['path']).name} n={len(take)}", flush=True)

    # Per frame, pick the plate warp with the lowest pixel error.
    chosen_residual = []
    chosen_psnr = []
    chosen_id = []
    homography_bytes = 0
    for i, frame in enumerate(query):
        best = None
        for pid, plate in enumerate(plates):
            n_in, Hmat = _orb_inliers(plate, frame)
            if Hmat is None:
                warped = np.full_like(frame, 128)
                n_in = 0
            else:
                warped = cv2.warpPerspective(plate, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            err = float(np.mean(np.abs(frame.astype(np.int16) - warped.astype(np.int16))))
            if best is None or err < best[0]:
                best = (err, pid, warped, n_in, Hmat)
        _err, pid, warped, n_in, Hmat = best
        delta = frame.astype(np.int16) - warped.astype(np.int16)
        residual = np.clip(delta + 128, 0, 255).astype(np.uint8)
        chosen_residual.append(residual)
        chosen_psnr.append(_psnr(frame, warped))
        chosen_id.append(pid)
        homography_bytes += 2 + (0 if Hmat is None else 18)  # id u16 + 9*f16
        if i % 50 == 0:
            print(f"frame {i} plate={pid} inliers={n_in} psnr={chosen_psnr[-1]:.2f}", flush=True)

    # Single best plate held for the whole clip, so the residual stays coherent.
    from collections import Counter
    hold_id = Counter(chosen_id).most_common(1)[0][0]
    hold_plate = plates[hold_id]
    hold_residual = []
    hold_psnr = []
    for frame in query:
        _n, Hmat = _orb_inliers(hold_plate, frame)
        if Hmat is None:
            warped = np.full_like(frame, 128)
        else:
            warped = cv2.warpPerspective(hold_plate, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        delta = frame.astype(np.int16) - warped.astype(np.int16)
        hold_residual.append(np.clip(delta + 128, 0, 255).astype(np.uint8))
        hold_psnr.append(_psnr(frame, warped))

    untouched_mp4 = out / "untouched.mp4"
    pick_mp4 = out / "residual_pick.mp4"
    hold_mp4 = out / "residual_hold.mp4"
    _write_mp4(query, untouched_mp4)
    _write_mp4(chosen_residual, pick_mp4)
    _write_mp4(hold_residual, hold_mp4)

    report = {"donors": [{k: v for k, v in row.items() if k != "samples"} for row in donors], "plates": []}
    for label, src in (("untouched", untouched_mp4), ("residual_pick", pick_mp4), ("residual_hold", hold_mp4)):
        dest = out / f"{label}_crf63.mp4"
        encode_av1_crf(src, dest, scale=None, ffmpeg=FFMPEG)
        report[label] = {
            "bytes": dest.stat().st_size,
            "kbps": round(_kbps(dest.stat().st_size, len(query)), 2),
        }
    meta_kbps = _kbps(homography_bytes, len(query))
    report["warp_metadata_kbps"] = round(meta_kbps, 2)
    report["pick_psnr_mean"] = round(float(np.mean(chosen_psnr)), 2)
    report["hold_psnr_mean"] = round(float(np.mean(hold_psnr)), 2)
    report["hold_plate"] = int(hold_id)
    report["pick_plus_meta_kbps"] = round(report["residual_pick"]["kbps"] + meta_kbps, 2)
    report["hold_plus_meta_kbps"] = round(report["residual_hold"]["kbps"] + meta_kbps, 2)
    report["untouched_kbps"] = report["untouched"]["kbps"]
    report["win"] = report["hold_plus_meta_kbps"] < report["untouched_kbps"]
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
